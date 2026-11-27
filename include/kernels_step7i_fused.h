#pragma once

// ============================================================================
// kernels_step7i_fused.h — Step 7I-1: graph-crushing super-fusions.
//   gfx1100 (RX 7900 XTX, 96 CU, Wave32).
//
// Five fusions, each one replaces two adjacent graph nodes with one launch and
// (where the reference path serialises a single-CTA depth-mix) parallelises the
// mix over the whole grid:
//
//   1. wo_B_fused  : Pass-0  wo(SoA Q8_0) + [mix_rms_norm + ffn_norm + router +
//                    Top-8]  -> 192 x 192    (-24 nodes)
//   2. wo_C_fused  : Pass-1/2 wo + [mix + ctrl_down + ctrl_apply + FFN norm +
//                    router + Top-8] -> 192 x 288 (-24 nodes)
//   3. mix_qkv_fused : Pass-0 attn [mix_rms_norm / rms_norm] + qkv_qkvr_prep
//                    (+ optional page_minmax in the K-store epilogue)
//                    -> 192 x 192 (-24 .. -28 nodes)
//   4. a_qkv_fused : Pass-1/2 [dual depth mix + ctrl_qkv3 + norm] + qkv_qkvr_prep
//                    (+ optional page_minmax).  `a_v3` is the standalone
//                    distributed-A variant (same math, 2 nodes).
//                    -> 192 x 288 (-24 .. -28 nodes with mode 2)
//   5. page_minmax folded into 3/4's K-store epilogue (4 nodes in mode 3 only,
//      8 when 4 is active too).
//
// Every fused kernel reproduces the shipped kernel's arithmetic bit-exactly
// (same per-channel FMA order, same warp butterfly trees, same LDS gathers) so
// the whole 48-slot forward pass stays bit-identical to the Step 7H default
// path.  The only exception is `a_qkv_fused` mode 1/2's ctrl gate/apply
// partition, which is identical to the shipped A kernel by construction.
//
// Scratch: InklingEngine::buf_mix_scratch_ (16384 floats, zeroed once).  All
// counters and partial slots are self-resetting, so the kernels are static
// HIP-graph replay safe.  Slot map (floats):
//   [0..6143]     DOT partials  [q(4)][m(8)][192]
//   [6144..7679]  SSQ partials  [m(8)][192]
//   [7680..8447]  SSQ-out partials [q(4)][192]
//   [8448..8767]  AUX (effw / inv / gate)
//   [8768..9215]  barrier shard counters (5 groups x 8)
//   [9216..9479]  router p/sel/counter
//   [9480..9599]  per-block ssq slots (C v2 tail)
//   [9600..9727]  A mix slot partials (aux)
// ============================================================================

#include "inkling_engine.h"
#include "sparse_global.h"

namespace inkling {

// ---------------------------------------------------------------------------
// 1. wo + B (Pass-0 pre-MoE) in ONE launch: 192 blocks x 192 threads.
//    Each block owns the 12 residual channels {4b..4b+3, 4b+768..4b+771,
//    4b+1536..4b+1539} = exactly the 12 channels of virtual mix thread b, so
//    the wo rows it computes are the last mix state's own channels.
//    Grid barrier -> replicated exact mix_rms_norm<5,1> (every block computes
//    the full 2304-wide state) -> ffn_norm -> 32 router blocks (4 experts
//    each) -> self-resetting Top-8.
// ---------------------------------------------------------------------------
void launch_wo_b_fused(
    const SoaQ8Matrix & W, const float * x,
    const float * conv_w, float * sconv_hist, const float * res, float * out,
    const float * res_query, const float * const * states, int n_states,
    const float * ffn_norm_w,
    const __half * router_w, const float * exp_probs_b,
    float * x_norm_out, int32_t * topk_ids_out, float * topk_w_out,
    float * mix_scratch, hipStream_t stream, float * router_scratch = nullptr);

// ---------------------------------------------------------------------------
// 2. wo + C (Pass-1/2 pre-MoE) in ONE launch: 192 blocks x 288 threads.
//    wo phase uses all 9 warps; grid barrier; blocks 0..31 then run the exact
//    C-v2 body (mix_ctrl_down + gate + ctrl_apply + FFN RMSNorm + router +
//    Top-8), blocks 32..191 exit.  Bit-identical to C v2.
// ---------------------------------------------------------------------------
void launch_wo_c_fused(
    const SoaQ8Matrix & W, const float * x,
    const float * conv_w, float * sconv_hist, const float * res, float * out,
    const float * res_query, const float * const * states, int n_states,
    const float * msg,
    const __half * w_down, const __half * w_gain, const __half * w_forget,
    const float * ffn_norm_w,
    const __half * router_w, const float * exp_probs_b,
    float * out_mixed, float * out_gate,
    float * x_norm_out, int32_t * topk_ids_out, float * topk_w_out,
    float * mix_scratch, hipStream_t stream, float * raw_out = nullptr,
    float * router_scratch = nullptr);

// Step 7J: identical kernel with the wo phase on `v_dot2_f32_f16` + `__half2`
// LDS (s_x 13.8 KB -> 7.7 KB, whole launch 24.9 KB -> 18.3 KB).  Same weight
// bytes, same per-lane block order, same reduction tree, FP32 epilogue.
void launch_wo_c_fused_fdot2(
    const SoaQ8Matrix & W, const float * x,
    const float * conv_w, float * sconv_hist, const float * res, float * out,
    const float * res_query, const float * const * states, int n_states,
    const float * msg,
    const __half * w_down, const __half * w_gain, const __half * w_forget,
    const float * ffn_norm_w,
    const __half * router_w, const float * exp_probs_b,
    float * out_mixed, float * out_gate,
    float * x_norm_out, int32_t * topk_ids_out, float * topk_w_out,
    float * mix_scratch, hipStream_t stream, float * router_scratch = nullptr);

// ---------------------------------------------------------------------------
// 3. Pass-0 attention input norm + qkv_qkvr_prep (+ page_minmax) in ONE launch:
//    192 blocks x 192 threads.  Blocks derive their projection segment and
//    compute the full normed input vector for that segment's query in
//    registers/LDS (no grid barrier before the GEMV), then run the exact
//    shipped GEMV + qkvr_prep body.
//    n_states == 0 selects the plain RMSNorm path (physical layer 0).
// ---------------------------------------------------------------------------
void launch_mix_rms_qkv_qkvr_prep_fused(
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
    bool fuse_minmax,
    SparseGlobalPageMeta sparse_meta,
    float * mix_scratch,
    hipStream_t stream,
    float * router_scratch = nullptr);

// ---------------------------------------------------------------------------
// 4a. Standalone distributed A (`--fuse-a-qkv 1`): 96 x 288, bit-identical to
//     the shipped mix_dual + ctrl_qkv3 + norm launch.
// ---------------------------------------------------------------------------
void launch_a_v3_decode(
    const float * query0, int n_q0, const float * const * states0, int n_states0, float * out0,
    const float * query1, int n_q1, const float * const * states1, int n_states1, float * out1,
    const __half * const w_down[3],
    const __half * const w_gain[3],
    const __half * const w_forget[3],
    const float * const w_norm[3],
    float * tmp_gate,
    float * const out_norm[3],
    float eps, hipStream_t stream, float * mix_scratch,
    float * router_scratch = nullptr);

// ---------------------------------------------------------------------------
// 4b. Super-fused A + qkv_qkvr_prep (`--fuse-a-qkv 2`): 192 x 288.
//     Blocks 0..95 run the exact A_v3 body, then a 192-block barrier and all
//     blocks run the shipped GEMV + qkvr_prep body (x re-read from buf_norm_).
// ---------------------------------------------------------------------------
void launch_a_qkv_qkvr_prep_fused(
    const float * query0, int n_q0, const float * const * states0, int n_states0, float * out0,
    const float * query1, int n_q1, const float * const * states1, int n_states1, float * out1,
    const __half * const w_down[3],
    const __half * const w_gain[3],
    const __half * const w_forget[3],
    const float * const w_norm[3],
    float * tmp_gate,
    float * const out_norm[3],
    float eps,
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
    bool fuse_minmax,
    SparseGlobalPageMeta sparse_meta,
    float * mix_scratch,
    hipStream_t stream,
    float * router_scratch = nullptr);

// No runtime occupancy query is needed (statically co-resident grids); kept so
// the call site mirrors the Step 7F/7H hooks.
void step7i_prime_occupancy();

} // namespace inkling
