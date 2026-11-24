#pragma once

// ============================================================================
// kernels_step7h_fused.h — Step 7H: shipped-shape fused MoE, 96-CU parallel
//   A/B/C_v3 (last-block Top-8), wo re-tile, and Pass-0 mix+RMSNorm folded into
//   the QKV prep launch.  gfx1100 (RX 7900 XTX, 96 CU, Wave32).
//
// Implemented in src/kernels_step7h_fused.hip.  Self-contained: links against
// nothing but the HIP runtime and include/inkling_engine.h.  Every counter in
// buf_router_scratch_ is self-resetting so the kernels are HIP-Graph replay safe.
//
// Scratch slot ownership (buf_router_scratch_, 512 floats, zeroed once at init):
//   [0..255]      B/C router p/sel     (same as Step 7D B/C v2)
//   [256]         B/C Top-8 arrival    (self-resetting)
//   [257]         B_v3 arrival         (self-resetting)
//   [264..267]    A apply counters     (untouched here)
//   [280..375]    C_v3 ssq partials (96) / A ssq (96)  (overwritten per call)
//   [376..391]    A/E flags            (untouched here)
//   [392..395]    E barrier            (untouched here; P0 fusion reuses them)
//   [396..399]    Pass-0 mix barrier   (self-resetting, added by Step 7H)
//   [400..404]    MoE fused grid barrier (same contract as Step 7F v3)
// ============================================================================

#include "inkling_engine.h"

namespace inkling {

// ---------------------------------------------------------------------------
// 1. Shipped-shape fused NVFP4 MoE (44 layers, -44 graph nodes)
//    Phase 1: exact shipped 4-lane / 64-weight LDS-broadcast gate_up + SITU
//             (blocks 0..143, 8 warps x 8 pairs, per-lane mapping and FMA order
//             identical to moe_nvfp4_gate_up_situ_kernel<4> => bit-exact)
//    Phase 2: exact shipped down_wsum<1,288> + sconv epilogue, all 288 blocks
//    Separated by a 4-shard self-resetting grid barrier (all 288 blocks x 256
//    threads are co-resident: 3 blocks/CU, 6 waves/SIMD, <= 80 VGPR, 12.4 KB LDS).
// ---------------------------------------------------------------------------
void launch_moe_nvfp4_ship_fused(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w = nullptr, float * post_sconv_h = nullptr,
    const float * post_res_add = nullptr, float * scratch = nullptr,
    float * raw_out = nullptr);

// A/B: same fusion with a 192-block grid (2 blocks/CU, no VGPR pin).
void launch_moe_nvfp4_ship_fused192(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w = nullptr, float * post_sconv_h = nullptr,
    const float * post_res_add = nullptr, float * scratch = nullptr,
    float * raw_out = nullptr);

// ---------------------------------------------------------------------------
// 2. Shipped-shape fused Q8 MoE (4 layers, -4 graph nodes)
//    192 blocks x 256 threads: 8 warps x 6 pair-groups for gate_up (per-pair math
//    identical to moe_q8_gate_up_situ_kernel<8,1> => bit-exact) then the exact
//    shipped down_wsum<2,192> + sconv epilogue for the same 192 blocks.
// ---------------------------------------------------------------------------
void launch_moe_q8_ship_fused(
    const SoaQ8Experts & gate_up, const SoaQ8Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w = nullptr, float * post_sconv_h = nullptr,
    const float * post_res_add = nullptr, float * scratch = nullptr);

// ---------------------------------------------------------------------------
// 3. B_v3: Pass-0 pre-MoE mix + FFN RMSNorm + router + Top-8 in ONE launch.
//    128 blocks x 192 threads (1 expert/block, all 96 CUs busy).  Each block
//    re-computes the exact 192-thread mix_rms_norm into its own LDS (no grid
//    barrier), dots one router row with the exact 1-warp/9-uint4 pattern, and the
//    single last-arriving block runs the branchless parallel-rank Top-8.
//    Drop-in for launch_mix_ffn_norm_router_topk_decode_v2.
// ---------------------------------------------------------------------------
void launch_mix_ffn_norm_router_topk_decode_v3(
    const float * res_query,
    const float * const * states, int n_states,
    const float * ffn_norm_w,
    const __half * router_w,
    const float * exp_probs_b,
    float * x_norm_out,
    int32_t * topk_ids_out,
    float * topk_w_out,
    float * router_scratch,
    hipStream_t stream);

// ---------------------------------------------------------------------------
// 4. C_v3: Pass-1/2 pre-MoE mix + ctrl_down + ctrl_apply + FFN RMSNorm + router
//    + Top-8 in ONE 96-block x 288-thread launch (3x the C_v2 block count).
//    The per-rank and per-channel dot math is identical to C_v2; only the ssq
//    partition changes (96 x 24 channels instead of 32 x 72, deterministic).
// ---------------------------------------------------------------------------
void launch_mix_ctrl_ffn_norm_router_topk_decode_v3(
    const float * res_query,
    const float * const * states, int n_states,
    const float * msg,
    const __half * w_down,
    const __half * w_gain,
    const __half * w_forget,
    const float * ffn_norm_w,
    const __half * router_w,
    const float * exp_probs_b,
    float * out_mixed,
    float * out_gate,
    float * x_norm_out,
    int32_t * topk_ids_out,
    float * topk_w_out,
    float * router_scratch,
    hipStream_t stream);

// ---------------------------------------------------------------------------
// 5. wo (2304 x 3072, SoA Q8_0) + fused sconv_post_res.
//    variant 0 = shipped v2 shape (6 warps, LPR 32, PD 3, 384 blocks)
//    variant 1 = 4-warp / 576-block split of the same K order (bit-identical)
//    variant 2 = 8-warp / 288-block persistent (grid-limited, 32 warps/CU)
// ---------------------------------------------------------------------------
void launch_soa_q8_gemv_wo_v4(
    const SoaQ8Matrix & W, const float * x,
    const float * conv_w, float * sconv_hist, const float * res, float * out,
    hipStream_t stream, int variant = 0);

// ---------------------------------------------------------------------------
// 6. Pass-0 mix_rms_norm folded into qkv_qkvr_prep_fused_decode (-24 nodes).
//    Same contract as launch_qkv_qkvr_prep_fused_decode plus the depth-mix
//    (query/n_states/w_norms/n_q) prologue.  The fused prologue reproduces
//    inkling_mix_rms_norm_2304_kernel<5,{1,3}> bit-exactly per output element.
// ---------------------------------------------------------------------------
void launch_mix_rms_qkv_qkvr_prep_fused_decode(
    const float * mix_query, int n_q,
    const float * const * mix_states, int n_states,
    const float * const * mix_w_norms,
    float * mix_out,
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
    hipStream_t stream);

// Bit-exact n_states-specialised replacement for launch_inkling_mix_rms_norm
// (single block, all state loads unguarded -> they overlap).
void launch_mix_rms_norm_2304_fast(
    const float * query, int n_q,
    const float * const * states, int n_states,
    const float * const * w_norms,
    float * out, float eps, hipStream_t stream);

// Prime the host-side occupancy queries before the static HIP graph is
// captured (hipOccupancyMaxActiveBlocksPerMultiprocessor is not capture-safe).
void step7h_prime_occupancy();

} // namespace inkling
