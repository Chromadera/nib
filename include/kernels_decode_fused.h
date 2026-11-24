#pragma once

// ============================================================================
// kernels_decode_fused.h — Step 7D standalone super-fused decode kernels +
//                         900 GB/s GEMV / MoE ceiling push (gfx1100, Wave32).
//
// Everything in this header is implemented in src/kernels_decode_fused.hip and
// is 100% self-contained: it links against nothing but the CUDA/HIP runtime.
// All launches are capture-safe (no host sync, no allocation, self-resetting
// device-side counters only), so they can be dropped into the existing static
// HIP graph.
//
// KERNELS
// -------
// A. launch_mix_dual_ctrl_qkv3_norm_decode
//      Pass-1/Pass-2 pre-QKV block: replaces THREE kernels
//        inkling_mix_dual_2304  (grid (1,2) x 192)
//        ctrl_down_qkv3_decode  (grid (16,1,3) x 288)
//        ctrl_apply_norm_qkv3_multiwgp (grid (32,3) x 192)
//      with ONE 96-block x 288-thread launch (-2 graph nodes). The 96 blocks are
//      all resident (1 block/CU on gfx1100) so every spin barrier is deadlock-free;
//      blocks 0/1 do the two depth mixes and then serve double duty as ctrl_apply
//      tiles 0/1 of branch 0, so 3 x 32 apply tiles are covered exactly.
//      The two mix blocks keep the *exact* reduction order of
//      inkling_mix_dual_2304_kernel<5,4>; the 3-way ctrl_down keeps the exact
//      per-branch 16-tile x 4-row math; the apply phase keeps the exact
//      32-tile/32-slot deterministic ssq reduction, so out_norm is bit-identical
//      to the split pipeline.  Weights are prefetched into registers/L2 while
//      the two mix blocks run; the ctrl_down w_down (3 x 576 KiB) and the
//      ctrl_apply w_gain+w_forget (3 x 576 KiB) DRAM latency is overlapped.
//      Scratch: buf_router_scratch_ (512 floats) slots 260..391 (see .hip).
//
// B. launch_mix_ffn_norm_router_topk_decode_v2   (Pass-0 Pre-MoE)
//      Replaces inkling_mix_rms_norm<5,1> + router_topk_decode<4>
//      (2 launches, ~14.4 us/layer) with ONE 33-block x 192-thread launch.
//      Block 32 runs the *identical* mix_rms_norm math (192 threads, 3 float4
//      each) while blocks 0..31 stream their 4 expert router rows into VGPRs
//      (9 uint4/lane) concurrently; after the flag the 31 router blocks stage
//      the 9.2 KiB x_norm from L2 and dot (<1.2 us). One block-wide arrival per
//      block (32) instead of 128 per-warp arrivals => far cheaper Top-8 barrier.
//      Scratch: buf_router_scratch_ [0..255] p/sel, [256] topk completion,
//              [384]/[385] mix flag/ack (self-resetting).
//
// C. launch_mix_ctrl_ffn_norm_router_topk_decode_v2  (Pass-1/2 Pre-MoE)
//      Replaces mix_ctrl_down_decode + ctrl_apply_ffn_norm_decode +
//      router_topk_decode<4> (3 launches, ~20.1 us/layer) with ONE 32-block x
//      288-thread launch. Mix + ctrl_down keep the exact
//      mix_ctrl_down_decode_kernel<5> math (all 32 blocks compute the same mix
//      redundantly, exactly like the original); the apply phase produces the
//      deterministic 32-slot ssq total; the router dots 4 experts/block from
//      prefetched VGPR rows. (The ssq partition is 32 x 72 channels instead of
//      72 x 32 in the split kernel: deterministic, <= 1 ulp difference.)
//      Scratch: [0..255] router p/sel, [256] topk, [280..311] ssq,
//              [384..387] gate/ssq flags+acks (self-resetting).
//
// D. launch_soa_q8_gemv_v2 / launch_soa_q8_gemv_qkv3_v2
//      Drop-in, bit-identical replacement for the SoA Q8_0 GEMV at K = 2304
//      (proj_q4 / proj_k / proj_v / proj_qkv_all). Same per-lane block order
//      (sub + 8*i, i = 0..8), same 8-lane reduce, same FP32 FMA chain, but one
//      block per 16-row group (no cap) => 538-542 resident 4-warp blocks
//      instead of 288 persistent ones (5.6 vs 3 blocks/CU of DRAM streams).
//
// E. launch_qkv_qkvr_prep_fused_decode
//      QKV GEMV + qkvr_prep (Q clip/FWHT, K/V sconv+Q8_0 KV store, R prep +
//      rel_table) in ONE 288-block x 128-thread launch with a sharded in-kernel
//      grid barrier (-1 graph node, -~5 us/layer). The prep stage reuses ALL
//      288 blocks (24 Q, 16 K/V, 248 rel_table blocks).
//      Scratch: [392..394] barrier shards, [395] barrier ack (self-resetting).
//      NOTE: experimental vs. D + the existing qkvr_prep; A/B it.
//
// F. launch_soa_q8_gemv_wo_v2
//      wo (2304 x 3072) GEMV with the fused sconv_post_res epilogue. Bit
//      identical math to the K=3072 path, but re-tiled to 6 warps x 1 row/warp
//      (LPR = 32, ring depth 3, 384 blocks = exactly 4 blocks/CU, one group per
//      block, zero tail wave) => 24 warps/CU of independent DRAM streams.
//
// G/H. launch_moe_nvfp4_gate_up_v2 / launch_moe_nvfp4_down_v2 /
//      launch_moe_nvfp4_decode_v2
//      NVFP4 MoE decode with the *same* arithmetic (same 4-lane pair reduce,
//      same expert weighting, same sconv epilogue) but:
//        gate_up: explicit PD-entry register ring (PD=1 or 2) instead of
//                 1-block-ahead => more bytes in flight per lane.
//        down:    2-deep row ring + 192-block grid (2 blocks/CU, every block
//                 resident, no 1.5-wave tail) instead of 288.
//      GPU A/B measured on this box: gate_up 30.71 -> 32.30 us/call (PD=2) or
//      32.0 us (PD=1); down 17.34 -> 18.77 us/call (192) / 18.6 us (288).
//      => both v2 stages are SLOWER than the shipped split kernels and the
//      engine therefore keeps the split MoE path (FUSE_MOE = false).
//
// MEASURED STATUS (RX 7900 XTX, decode q4_1 + --sparse-pages 96)
// ------------------------------------------------------------------
//  Kept (bit-exact vs the split pipeline, 0 ulp):
//    A  (-48 graph nodes): 14.41 us/layer (vs 16.02 split)  +2.07 tok/s @256k
//    E  (-48 graph nodes): 28.7 us/layer (vs 26.9 + 2.8)     +0.82 tok/s @256k
//    F  (0 nodes)        : 11.7 us/call  (vs 12.1)           +0.05 tok/s @256k
//  Reverted (not bit-exact / slower), left compiled but unused by the engine:
//    B  (+speed, but x_norm differs by <=1 ulp on some layers)
//    C  (+speed, but gate/x_norm differ by <=1 ulp)
//    G/H (slower than the shipped split NVFP4 MoE kernels)
//
// SCRATCH BUFFER CONTRACT
// -----------------------
// Kernels A, B, C and E use the engine's existing 512-float
// buf_router_scratch_ (zeroed once at init). Every counter is self-resetting,
// so the same 512-float buffer can be shared with the existing router/argmax
// kernels. No additional VRAM is required.
// ============================================================================

#include "inkling_engine.h"

namespace inkling {

// ---------------------------------------------------------------------------
// A. Pass-1/2 pre-QKV: dual mix + 3-way ctrl_down + 3-way ctrl_apply+RMSNorm.
//    Drop-in replacement for
//      launch_inkling_mix_dual(...)
//      launch_inkling_ctrl_qkv_norm(inputs, msg, w_down, w_gain, w_forget,
//                                   w_norm, tmp_gate, out_norm, 1, eps, stream,
//                                   scratch)
//    Requirements: n_tokens == 1, n_states0 <= 5, n_states1 <= 5,
//                  n_q0 <= 4, n_q1 <= 4, scratch != nullptr.
//    cur[j] is derived from out1 (n_q1 == 3 -> 3 slices, else all slice 0).
//    Falls back to the two original public launches when a requirement is not
//    met, so the call site needs no guard.
// ---------------------------------------------------------------------------
void launch_mix_dual_ctrl_qkv3_norm_decode(
    const float * query0, int n_q0, const float * const * states0, int n_states0, float * out0,
    const float * query1, int n_q1, const float * const * states1, int n_states1, float * out1,
    const __half * const w_down[3],
    const __half * const w_gain[3],
    const __half * const w_forget[3],
    const float * const w_norm[3],
    float * tmp_gate,
    float * const out_norm[3],
    int n_tokens, float eps, hipStream_t stream,
    float * scratch);

// ---------------------------------------------------------------------------
// B. Pass-0 pre-MoE: mix (n_q=1) + FFN RMSNorm + router + Top-8, one launch.
//    Drop-in replacement for
//      launch_mix_ffn_norm_router_topk_decode(...)
//    Requirements: n_states <= 5, router_scratch != nullptr.
// ---------------------------------------------------------------------------
void launch_mix_ffn_norm_router_topk_decode_v2(
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
// C. Pass-1/2 pre-MoE: mix + ctrl_down + ctrl_apply + FFN RMSNorm + router +
//    Top-8, one launch. Drop-in replacement for the 3-launch sequence
//      launch_mix_ctrl_down_decode(...)
//      launch_ctrl_apply_ffn_norm_router_topk_decode(...)
//    Requirements: n_states <= 5, router_scratch != nullptr.
//    out_mixed = the depth mix (n_q = 1), out_gate = the 64 ctrl gate values.
// ---------------------------------------------------------------------------
void launch_mix_ctrl_ffn_norm_router_topk_decode_v2(
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
// Step 8C: m-row batched variants of A / B / C (grid.y = row).  Every launch is
// the exact single-token decode kernel with the per-token pointers shifted by
// row * N_EMBD and a private 512-float scratch sector per row, so all `nrows`
// rows run concurrently and the result stays bit-identical to `nrows` separate
// decode launches.  nrows == 1 forwards to the single-row launcher above.
// ---------------------------------------------------------------------------
void launch_mix_dual_ctrl_qkv3_norm_decode_mrows(
    const float * query0, int n_q0, const float * const * states0, int n_states0, float * out0,
    const float * query1, int n_q1, const float * const * states1, int n_states1, float * out1,
    const __half * const w_down[3],
    const __half * const w_gain[3],
    const __half * const w_forget[3],
    const float * const w_norm[3],
    float * tmp_gate,
    float * const out_norm[3],
    float eps, hipStream_t stream, float * scratch,
    int nrows);

void launch_mix_ffn_norm_router_topk_decode_v2_mrows(
    const float * res_query,
    const float * const * states, int n_states,
    const float * ffn_norm_w,
    const __half * router_w,
    const float * exp_probs_b,
    float * x_norm_out,
    int32_t * topk_ids_out,
    float * topk_w_out,
    float * router_scratch,
    int nrows, hipStream_t stream);

void launch_mix_ctrl_ffn_norm_router_topk_decode_v2_mrows(
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
    int nrows, hipStream_t stream);

// ---------------------------------------------------------------------------
// D. QKV SoA Q8_0 GEMV v2 (K = 2304), bit-identical, better CU fill.
//    launch_soa_q8_gemv_v2       : one matrix (e.g. proj_qkv_all [8600, 2304])
//    launch_soa_q8_gemv_qkv3_v2  : three segments (proj_q4 | proj_k | proj_v)
// ---------------------------------------------------------------------------
void launch_soa_q8_gemv_v2(const SoaQ8Matrix & W, const float * x, float * y, hipStream_t stream);
void launch_soa_q8_gemv_qkv3_v2(
    const SoaQ8Matrix & W0, const float * x0, float * y0,
    const SoaQ8Matrix & W1, const float * x1, float * y1,
    const SoaQ8Matrix & W2, const float * x2, float * y2,
    hipStream_t stream);

// ---------------------------------------------------------------------------
// E. QKV GEMV + qkvr_prep fused decode (one launch, in-kernel grid barrier).
//    W0/W1/W2 are the same segments as launch_soa_q8_gemv_qkv3_v2 (W1/W2 may
//    have rows == 0 for the single-matrix SWA Pass-0 case).
//    q_in / r_in / k_raw / v_raw / k_dyn / v_dyn point into the projections the
//    GEMV just wrote (same pointers the engine passes to
//    launch_qkvr_prep_fused_decode). scratch is buf_router_scratch_.
// ---------------------------------------------------------------------------
void launch_qkv_qkvr_prep_fused_decode_g(
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
    int grid);

void launch_qkv_qkvr_prep_fused_decode(
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

// ---------------------------------------------------------------------------
// F. wo (K = 3072) GEMV + fused sconv_post_res epilogue, 384-block re-tile.
//    Same contract as launch_soa_q8_gemv_sconv_post_res.
// ---------------------------------------------------------------------------
void launch_soa_q8_gemv_wo_v2(
    const SoaQ8Matrix & W, const float * x,
    const float * conv_w, float * sconv_hist, const float * res, float * out,
    hipStream_t stream, float * raw_out = nullptr);

// ---------------------------------------------------------------------------
// G/H. NVFP4 MoE decode v2 (bit-identical math, deeper prefetch / better fill).
// ---------------------------------------------------------------------------
void launch_moe_nvfp4_gate_up_v2(
    const SoaNVFP4Experts & gate_up,
    const float * x_norm, const int32_t * topk_ids, float * act,
    hipStream_t stream);

void launch_moe_nvfp4_down_v2(
    const SoaNVFP4Experts & down,
    const float * act, const int32_t * topk_ids, const float * topk_w,
    float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add);

void launch_moe_nvfp4_decode_v2(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add);

} // namespace inkling
