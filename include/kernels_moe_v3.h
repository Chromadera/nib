#pragma once

// ============================================================================
// kernels_moe_v3.h — Step 7F: 850+ GB/s weight-streaming decode kernels for
//                    RDNA3 gfx1100 (RX 7900 XTX, 96 CU, Wave32, 960 GB/s).
//
// Everything here is implemented in src/kernels_moe_v3.hip and is 100%
// self-contained: it links against nothing but the CUDA/HIP runtime plus the
// constants/structs of include/inkling_engine.h.  No existing file is touched;
// this module is a pure drop-in / A-B alternative for:
//
//   kernels_gemv_soa.hip : launch_moe_nvfp4_decode, launch_moe_q8_decode,
//                          launch_soa_q8_gemv_sconv_post_res   (wo)
//   kernels_decode_fused.hip : launch_soa_q8_gemv_wo_v2,
//                          launch_moe_nvfp4_decode_v2
//
// ----------------------------------------------------------------------------
// WHY (measured with rocprofv3 @ d=262144, /tmp/prof_step7d_256k):
//
//   kernel                 calls  us/call  GB/s   hardware floor @ 884 GB/s
//   lm_head  (Q8, 492 MB)      1   552.3   891   (the gold standard)
//   moe_nvfp4_gate_up_situ    44    32.4   737   -> ~300 us/tok wasted
//   moe_nvfp4_down_wsum       44    17.6   677   -> ~230 us/tok wasted
//   wo        (Q8, 7.5 MB)    48    12.7   592   -> ~140 us/tok wasted
//   moe_q8_gate_up             4    56.5   798   ->  ~15 us/tok wasted
//   moe_q8_down                4    28.7   787   ->  ~10 us/tok wasted
//   gate_up -> down dispatch  44     ~2.0     -   ->  ~88 us/tok  (fusable)
//
// The three slow kernels all have the *same* two root causes (see the .hip for
// the full ISA analysis), both of which are occupancy / memory-level-parallelism
// problems rather than math problems:
//
//   1. Not enough waves per SIMD to hide ~300-400 cycle GDDR6 latency:
//        moe_nvfp4_gate_up  : 131 VGPRs, 288 blocks x 4 warps -> 3 blocks/CU
//                             = 3 waves/SIMD  (1152 warps of work total)
//        moe_nvfp4_down     : 105 VGPRs, 288 blocks x 8 warps -> 2 blocks/CU
//                             = 4 waves/SIMD
//        lm_head            :  74 VGPRs, 768 blocks x 8 warps -> 3 blocks/CU
//                             = 6 waves/SIMD  (884 GB/s)
//   2. A prefetch ring that is too shallow (wo has PD = 1: every 16 B weight
//      load is consumed in the very next instruction, so each warp alternates
//      "wait ~400 cycles" / "burn 72 VALU ops" instead of overlapping them).
//
// v3 fixes both without touching the arithmetic:
//   * nvfp4 gate_up is re-tiled from 4 lanes/pair (1152 warps, 3 waves/SIMD) to
//     8 lanes/pair (2304 warps) and run as 288 blocks x 256 threads => 6
//     waves/SIMD, with a 2-deep register ring that also prefetches across rows.
//   * nvfp4/q8 down drop BOTH the 12.4 KB LDS staging and the 48-float VGPR
//     activation cache: the 72 activation floats a lane needs are re-read from
//     L1 every row (the weight stream uses `slc dlc` and therefore does not
//     evict them), which frees ~60 VGPRs and allows 384 blocks x 256 threads
//     = 4 blocks/CU = 8 waves/SIMD with zero per-block LDS.
//   * wo gets a 3-deep ring (PD=3, the whole 96-block row is in flight) at
//     unchanged occupancy: 384/288 blocks x 6/8 warps = 6 waves/SIMD.
//   * the two MoE stages are additionally fused into ONE launch behind a
//     deadlock-free, self-resetting grid barrier (all blocks co-resident:
//     288 x 256 threads, 0 LDS, <=64 VGPRs = 768 resident slots >= 288), which
//     removes the ~2 us/node dispatch gap on all 44 NVFP4 layers.
//
// ----------------------------------------------------------------------------
// BIT-EXACTNESS
//
//  * down (nvfp4 + q8) keeps the EXACT per-lane weight set
//    ({g*32..+32}, {512+g*32..+32}, {1024+8g..+8}), the exact per-lane FMA order,
//    the exact 16-lane butterfly and the exact 8-expert summation tree, so it is
//    bit-identical to kernels_gemv_soa.hip's down_wsum + sconv epilogue.
//    (The only difference is where the activation is read from: L1 instead of
//    VGPR/LDS - the values are identical, so the result is identical.)
//  * wo keeps LPR = 32 / 3 blocks per lane / the same 32-lane butterfly, so it
//    is bit-identical to soa_q8_gemv (K = 3072) + sconv_post_res; only the ring
//    depth changes (PD 1 -> 3), which does not reorder any FMA.
//  * gate_up (nvfp4 + q8) changes the per-lane weight assignment (4 lanes/pair
//    with 64-weight blocks -> 8 lanes/pair with 32-weight units, and the 4-lane
//    ALT walks 16-weight units).  The per-lane partial sums, and therefore the
//    final ULP, differ by ~1e-7 relative - the same class of deviation as the
//    B/C fused kernels that were previously reverted, so gate_up MUST be
//    validated against the token oracle before it is made the default.  The
//    bit-for-bit reference stays the shipped moe_*_gate_up_situ kernel.
//
// ----------------------------------------------------------------------------
// LAUNCH SHAPES (every figure below was extracted from the built object with
// llvm-readobj --notes; see the .hip header for the full table)
//
//   kernel                  grid x thr   VGPR  spills  warps/CU  waves/SIMD  LDS
//   moe_nvfp4_gate_up_v3    288 x 256     70    0/0       24         6      10,368 B
//   moe_nvfp4_down_v3       384 x 256     50    0/0       32         8         128 B
//   moe_nvfp4_fused_v3      288 x 256     64    0/0       24         6      10,496 B
//   moe_q8_gate_up_v3       288 x 256     61    0/0       24         6      10,368 B
//   moe_q8_down_v3          384 x 256     56    0/0       32         8         128 B
//   moe_q8_fused_v3         288 x 256     62    0/0       24         6      10,496 B
//   soa_q8_gemv_wo_v3 W1    288 x 256     69    0/0       24         6      13,824 B
//   soa_q8_gemv_wo_v3 W2    576 x 128     78    0/0       24         6           0 B
//
// Before: gate_up 131 VGPR / 3 waves, down 105 VGPR + 12.4 KB LDS / 4 waves,
// wo 51 VGPR but PD = 1 / 4.5 waves.  Every new kernel is >= 6 waves/SIMD with
// a 2-3 deep weight ring, and no kernel spills a single VGPR.
//
// ----------------------------------------------------------------------------
// MEASURED A/B ON GPU 0 (2026-11-23, RX 7900 XTX, q4_1 + sparse-96)
//
// The offline analysis above was WRONG about which term dominates: on this MoE
// model the v3 re-tilings all LOSE to the shipped kernels, because the shipped
// 4-lane/64-weight tiling reads ONE 128 B activation chunk broadcast across the
// lane group, while the 8-lane or 16-weight-unit re-tilings make each lane read
// a DIFFERENT chunk (8x / 2x more LDS bytes for the same weights).  LDS
// bandwidth costs more than the missing waves/SIMD.  rocprofv3 per-call
// (d = 0, 40 decode steps):
//
//   kernel                 shipped     v3 AUTO    v3 ALT    best v3 tuning
//   moe_nvfp4_gate_up      31.10 us   38.49 us   38.9 us    33.07 us (SHIP PD1)
//   moe_nvfp4_down         17.46 us   21.18 us   21.5 us    (R=2 also slower)
//   moe_q8_gate_up         55.41 us   84.23 us     --       55.38 us (PD=2, ties)
//   moe_q8_down            28.68 us   32.06 us     --       28.7  us
//   wo (q8, K=3072)        11.63 us   11.55 us     --       (wall: ties)
//
// Wall clock at d = 0 / 8k (-p 0 -n 128 -r 3): shipped 140.72 / 132.68 tok/s;
// gate_up AUTO 130.18 / 123.34; gate_up SHIP PD1 139.21 / 131.41; down AUTO
// 137.07 / 129.12; wo AUTO 140.92 / 132.10; down R=2 138.28 / 129.19; fused
// (356 -> 308 nodes, phase 1 = 8-lane AUTO) 130.26 / 123.23.
//
// Everything else that was tried also fails to beat the shipped shape:
//   * deeper rings (gate_up PD=2/3, down R=2, q8 gate_up PD=2): neutral to
//     -2.6% - the 512-VGPR/SIMD register file caps in-flight weight bytes, so
//     more ring costs waves instead of buying MLP.
//   * MOE_V3_CACHE (drop the `slc` streaming hint so the ~39 MB/token decode
//     working set could live in the 96 MB Infinity Cache): 0.5-8% SLOWER -
//     no reuse materialises, cached loads only add L2 pressure.
//   * fully unrolling a 9-step ring body (4.6k instructions): -8% - it thrashes
//     the 32 KB SIMD instruction cache; the loop must stay ROLLED.
//   * direct-to-LDS weight loads (the only way to raise MLP past the register
//     file): not available on gfx1100 (clang: "needs target feature
//     vmem-to-lds-load-insts").
//   * RDNA3 has no way to interleave gate/up rows without a repack, so the two
//     1.3 MB-apart row streams per pair are unavoidable here.
//
// The per-stage split of the shipped launch (launch_moe_*_ship in
// src/kernels_gemv_soa.hip) exists so each stage can still be A/B'd in
// isolation.  DEFAULTS ARE THE SHIPPED KERNELS (variant -1): on this model the
// module is a measurement harness plus the fused-path prototype, not a win.
// The fused path is CORRECT (308 graph nodes, graph == --no-graph bit-identical,
// eval32 top-5 exact) but its gate_up phase is the losing 8-lane shape.
// ============================================================================

#include "inkling_engine.h"

namespace inkling {

// ---------------------------------------------------------------------------
// Variant selectors (runtime A/B switches; every one is a plain enum so the
// engine can expose them as CLI flags without recompiling).
//   MOE_V3_AUTO     : the tuned default above
//   MOE_V3_ALT      : gate_up keeps the 1-pair-per-4-lane mapping (bit-exact
//                     with the shipped kernel, 3 waves/SIMD) / down caches the
//                     activation in 48 VGPRs (bit-exact, 4 waves/SIMD) /
//                     wo keeps the 13.8 KB LDS staging
//   MOE_V3_ALT2     : wo reads its activation from L1 (no LDS, 4-warp blocks)
//   MOE_V3_CACHE    : AUTO/ALT shape but the weight loads are cacheable (no
//                     `slc` streaming hint).  The whole decode weight working
//                     set is ~39 MB and this GPU has a 96 MB Infinity Cache,
//                     so a repeated per-token stream can hit L2 instead of
//                     GDDR6.  Pure load-policy change; identical arithmetic.
//   MOE_V3_SHIP     : gate_up keeps the SHIPPED 4-lane / 64-weight-block tiling
//                     (4-lane LDS broadcast, bit-identical output) with a
//                     2-deep prefetch ring + branchless ue4m3.
//   MOE_V3_SHIP3    : same, 3-deep ring.
//   (SHIP/SHIP3 are gate_up-only; the other launchers map them to AUTO.)
// ---------------------------------------------------------------------------
enum MoeV3Variant {
    MOE_V3_AUTO = 0,
    MOE_V3_ALT  = 1,
    MOE_V3_ALT2 = 2,
    MOE_V3_CACHE = 3,
    MOE_V3_SHIP  = 4,
    MOE_V3_SHIP3 = 5,
};

// ---------------------------------------------------------------------------
// NVFP4 MoE decode (44 layers of the loop body), Stage 1: gate_up + SITU.
//   gate_up must be [*, N_GATE_UP_EXP=2304, K=N_EMBD=2304] NVFP4 SoA.
//   act      : [TOP_K, N_FF_EXP=1152] FP32, written (SITU applied).
// Drop-in for launch_moe_nvfp4_gate_up_v2 / the stage-1 half of
// launch_moe_nvfp4_decode.
// ---------------------------------------------------------------------------
void launch_moe_nvfp4_gate_up_v3(
    const SoaNVFP4Experts & gate_up,
    const float * x_norm, const int32_t * topk_ids, float * act,
    hipStream_t stream, int variant = MOE_V3_AUTO);

// ---------------------------------------------------------------------------
// NVFP4 MoE decode, Stage 2: down + weighted 8-expert sum + fused
// sconv_post_res epilogue.  down must be [*, N_EMBD=2304, K=N_FF_EXP=1152].
// Drop-in for launch_moe_nvfp4_down_v2.
// ---------------------------------------------------------------------------
void launch_moe_nvfp4_down_v3(
    const SoaNVFP4Experts & down,
    const float * act, const int32_t * topk_ids, const float * topk_w,
    float * out, hipStream_t stream,
    const float * post_conv_w = nullptr, float * post_sconv_h = nullptr,
    const float * post_res_add = nullptr, int variant = MOE_V3_AUTO);

// Both stages, two launches (byte-for-byte the same contract as
// launch_moe_nvfp4_decode / _v2).
void launch_moe_nvfp4_decode_v3(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w = nullptr, float * post_sconv_h = nullptr,
    const float * post_res_add = nullptr, int variant = MOE_V3_AUTO);

// Both stages fused into ONE kernel behind an in-kernel grid barrier (-1 graph
// node per MoE layer).  `scratch` is the engine's existing 512-float
// buf_router_scratch_; slots [400..408] are used and left at 0 (self-resetting).
// If scratch == nullptr the call transparently falls back to the two-launch
// path, so this is also safe inside the static HIP graph.
void launch_moe_nvfp4_fused_v3(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * scratch, int variant = MOE_V3_AUTO);

// ---------------------------------------------------------------------------
// Q8_0 MoE decode (physical layers 0/1 and the two loop replays, 4 calls/layer).
// Exactly the same contracts, for SoaQ8Experts ([*, 2304, 2304] / [*, 2304, 1152]).
// ---------------------------------------------------------------------------
void launch_moe_q8_gate_up_v3(
    const SoaQ8Experts & gate_up,
    const float * x_norm, const int32_t * topk_ids, float * act,
    hipStream_t stream, int variant = MOE_V3_AUTO);

void launch_moe_q8_down_v3(
    const SoaQ8Experts & down,
    const float * act, const int32_t * topk_ids, const float * topk_w,
    float * out, hipStream_t stream,
    const float * post_conv_w = nullptr, float * post_sconv_h = nullptr,
    const float * post_res_add = nullptr, int variant = MOE_V3_AUTO);

void launch_moe_q8_decode_v3(
    const SoaQ8Experts & gate_up, const SoaQ8Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w = nullptr, float * post_sconv_h = nullptr,
    const float * post_res_add = nullptr, int variant = MOE_V3_AUTO);

void launch_moe_q8_fused_v3(
    const SoaQ8Experts & gate_up, const SoaQ8Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * scratch, int variant = MOE_V3_AUTO);

// ---------------------------------------------------------------------------
// wo (K = N_Q = 3072, N = 2304 rows, SoA Q8_0) + fused sconv_post_res.
// Same contract as launch_soa_q8_gemv_sconv_post_res / _wo_v2.
//   MOE_V3_AUTO : W1, 288 x 256, 3-deep ring, 13.8 KB LDS activation
//   MOE_V3_ALT2 : W2, 384 x 192, 3-deep ring, activation from L1 (no LDS/barrier)
// ---------------------------------------------------------------------------
void launch_soa_q8_gemv_wo_v3(
    const SoaQ8Matrix & W, const float * x,
    const float * conv_w, float * sconv_hist, const float * res, float * out,
    hipStream_t stream, int variant = MOE_V3_AUTO);

// Prime the fused-kernel occupancy queries BEFORE the static HIP graph is
// captured: hipOccupancyMaxActiveBlocksPerMultiprocessor is a host-only query
// and must not run inside stream capture.  Call once after device/weights init
// when the fused path may be selected (no kernel is launched; the answer is
// cached in a static, so capture sees a pure return).
void moe_v3_prime_occupancy();

} // namespace inkling
