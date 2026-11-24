#pragma once

// ============================================================================
// kernels_step7j_fdot2.h — Step 7J: RDNA3 `v_dot2_f32_f16` + `__half2` LDS
//   bandwidth push on every decode weight-streaming kernel.
//   gfx1100 (RX 7900 XTX, 96 CU, Wave32).
//
// WHY: Step 7I-1 measured the decode weight streams at 585-849 GB/s against a
// 875 GB/s roof (91% of 960 GB/s).  Every one of them is DRAM-bound but was
// paying 1 MAC per instruction (v_fma_mix_f32) and caching x as FP32 in LDS
// (9.2-12.3 KB per launch, 108 VGPR in the fused MoE).  This module keeps the
// exact same weight bytes, the exact same K-block order and the exact same
// reduction trees, and changes only:
//
//   * x is staged in LDS as `__half2`  -> half the LDS, half the LDS traffic
//     (ds_load_b64-equivalent b128 pairs instead of b128 quads) and half the
//     VGPRs for the register-cached slices;
//   * the inner product is `__builtin_amdgcn_fdot2` (`v_dot2_f32_f16`,
//     2 MACs/instruction, FP32 accumulate) instead of 4 x v_fma_mix_f32 per
//     4 weights;
//   * the NVFP4 MoE puts the gate_up x-buffer and the down act staging in a
//     `union` (they are separated by the 4-shard self-resetting grid barrier),
//     which drops its LDS from 20.6 KB to ~6.3 KB and its VGPRs from 108 to
//     ~58-68 -> 3-4 blocks/CU (6-8 waves/SIMD) instead of 2.
//
// Precision: only *activations* are rounded to FP16 (11-bit mantissa, ~5e-4
// relative).  The 4-bit NVFP4 and 8-bit Q8_0 weights are exact in FP16, every
// dot product accumulates in FP32, and all post-conv / residual / SITU /
// RMSNorm / epilogue math stays FP32.  This is the same activation precision
// the shipped FP16 WMMA prefill path has always used (PPL 3.5889).
//
// Every launch is capture-safe (no host sync, no allocation, self-resetting
// device counters only) so it drops into the existing static decode HIP graph.
// ============================================================================

#include "inkling_engine.h"

namespace inkling {

// ---------------------------------------------------------------------------
// 1. Fused NVFP4 MoE (gate_up + SITU, then down + weighted sum + sconv
//    epilogue) in ONE launch behind the Step 7F 4-shard grid barrier.
//    256 threads, 8 warps.  `GRID` blocks must all be co-resident.
//      192 -> 2 blocks/CU, no VGPR pin (R = 2 down ring)  [best]
//      288 -> 3 blocks/CU, pinned 72 VGPR (6 waves/SIMD, R = 1)
// ---------------------------------------------------------------------------
void launch_moe_nvfp4_fdot2_fused(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * scratch);

void launch_moe_nvfp4_fdot2_fused288(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * scratch);

// Split A/B: the same fdot2 gate_up and down phases as two launches (no grid
// barrier, exact same arithmetic and the same 288/192 shapes).
void launch_moe_nvfp4_fdot2_split(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add);

// ---------------------------------------------------------------------------
// 2. Fused Q8_0 MoE (4 layers) with the same fdot2 + half2 union LDS.
//    192 blocks x 256 threads (2 blocks/CU).
// ---------------------------------------------------------------------------
void launch_moe_q8_fdot2_fused(
    const SoaQ8Experts & gate_up, const SoaQ8Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * scratch);

// ---------------------------------------------------------------------------
// 3. QKV GEMV + qkvr_prep fused, fdot2 variant.
//    `variant` 0 = 288 x 128 (4 warps, 3 blocks/CU)
//              1 = 288 x 256 (8 warps, 3 blocks/CU = 6 waves/SIMD)
//              2 = 384 x 256 (8 warps, 4 blocks/CU = 8 waves/SIMD)
//    Drop-in for launch_qkv_qkvr_prep_fused_decode (same contract, same
//    scratch slots 392..395).
// ---------------------------------------------------------------------------
void launch_qkv_qkvr_prep_fdot2(
    const SoaQ8Matrix & W0, const float * x0, float * y0,
    const SoaQ8Matrix & W1, const float * x1, float * y1,
    const SoaQ8Matrix & W2, const float * x2, float * y2,
    const float * q_in, const float * q_clip_w,
    const float * r_in, const float * r_clip_w, const float * rel_bias_w,
    const float * k_raw, const float * k_conv_w, const float * k_dyn,
    const float * v_raw, const float * v_conv_w, const float * v_dyn,
    float * sconv_hist_kv,
    KVCacheSlot kv_slot,
    float * q_out,
    float * rel_table_out,
    bool apply_tau,
    const StepMeta * d_meta,
    float * scratch,
    hipStream_t stream,
    int variant);

// ---------------------------------------------------------------------------
// 4. wo (2304 x 3072 SoA Q8_0) + fused sconv_post_res, fdot2 variant.
//    `variant` 0 = 6 warps LPR 32 (384 blocks, shipped shape)
//              1 = 8 warps LPR 32 (288 blocks)
//              2 = 4 warps LPR 16 (288 blocks, 8 rows/block)
//              3 = 4 warps LPR 32 (576 blocks, 4 rows/block)
// ---------------------------------------------------------------------------
void launch_soa_q8_gemv_wo_fdot2(
    const SoaQ8Matrix & W, const float * x,
    const float * conv_w, float * sconv_hist, const float * res, float * out,
    hipStream_t stream, int variant);

// ---------------------------------------------------------------------------
// 5. Plain SoA Q8_0 GEMV (lm_head) with half2 LDS + fdot2.
//    `variant` 0 = 8 warps LPR 8 (RPB 32, 16 blocks/CU, shipped lm_head shape)
//              1 = 8 warps LPR 16 (RPB 16, 16 blocks/CU)
// ---------------------------------------------------------------------------
void launch_soa_q8_gemv_fdot2(
    const SoaQ8Matrix & W, const float * x, float * y,
    hipStream_t stream, int variant);

// Step 8A: m-row (1..8) lm_head SoA Q8_0 GEMV.  Every output row r accumulates
// over the same per-lane weight blocks in the same order as
// launch_soa_q8_gemv_fdot2 (FP32 accumulate, fdot2), so y[r*ystride + row] is
// BIT-IDENTICAL to a single-row launch on x + r*N_EMBD — while the weights are
// streamed once for all m rows.  x is [m, N_EMBD] f32, y [m, ystride] f32.
void launch_soa_q8_gemv_fdot2_mrows(
    const SoaQ8Matrix & W, const float * x_rows, float * y_rows,
    int m, hipStream_t stream, int variant);

// Host-side differential check hook (selftest): returns false when the batched
// row loop would not match the single-row shape (never on gfx1100).
bool soa_q8_gemv_fdot2_mrows_supported(const SoaQ8Matrix & W);

// Prime host-side occupancy queries before the static HIP graph is captured.
void step7j_prime_occupancy();

} // namespace inkling
