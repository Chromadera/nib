#pragma once

// ============================================================================
// Step 7G — RDNA3 `v_dot2_f32_f16` fast decode: SWA + sparse-Global FA v3,
//           Quest page-select v3, QKV+qkvr_prep v3 (gfx1100, Wave32).
//
// Everything here is implemented in src/kernels_fa_decode_v3.hip and is
// self-contained (links against nothing but the HIP runtime).  All launches are
// capture-safe: no host sync, no allocation, self-resetting device counters
// only, so all four launchers are drop-in replacements inside the existing
// static decode HIP graph.
//
// ----------------------------------------------------------------------------
// WHAT STEP 7E MEASURED (rocprofv3, /tmp/prof_step7e_final/final_results.db,
// d = 262144, --kv-type q4_1 --sparse-pages 96, 40 decode steps):
//
//   qkv_qkvr_prep_fused_decode   48 calls x 29.41 us = 1412 us/tok  720 GB/s
//   inkling_fa_decode_swa_v2     40 calls x  9.90 us =  396 us/tok
//   sparse_global_select_v2       8 calls x 21.50 us =  172 us/tok  394 GB/s
//   fa_sparse_global_v2           8 calls x 25.57 us =  205 us/tok
//                                                   --------------
//                                                    2185 us/tok  (31% of all
//                                                    decode kernel busy time)
//   DRAM floors at 960 GB/s: qkv 22.0, select 8.8, global 7.3 us/call.
//
// ----------------------------------------------------------------------------
// ROOT CAUSES, all read out of the *compiled* ISA of build/kernels_*_v2.o and
// build/kernels_decode_fused.o (llvm-objdump -d --mcpu=gfx1100), not guessed:
//
//  1. fa_decode_v2 (SWA + Global) — 107-112 VGPR, 100% ALU/LDS-latency bound.
//     Per 8-key tile, per lane, the disassembly shows
//       * 216 v_fma_mix_f32   (96 K.Q + 96 P.V + 24 q4_1 zero-point)
//       *  96 v_perm_b32 + 48 v_pk_add_f16  (nibble -> fp16 unpack, 2 per 4
//            nibbles, each followed by a `v_pk_sub_f16` to undo the 1024 bias)
//       *  24 ds_load_b128  reading Q out of LDS *inside* the hot loop, each
//           with an s_waitcnt lgkmcnt before it can be consumed
//       *  63 ds_bpermute_b32 (score butterfly, softmax reductions, and 24
//           P.V broadcasts of w)
//     and because v2 launches exactly 96 CTAs of 128 threads for SWA
//     (12 splits x 8 KV heads = 1 CTA/CU = 4 warps/CU) the machine runs at
//     **1 warp per SIMD**, i.e. every dependent ALU/LDS latency is fully
//     exposed: 3451 static instructions are executed in ~25000 cycles.
//
//  2. sparse_global_select_v2 — 120 VGPR -> 2 CTAs/CU, and the inner loop is
//     ~250 instructions per 32 loaded bytes: 16 int8 x 2 arrays are converted
//     to fp32, then 3 heads x 16 dims x (2 mul + 1 fmax + 1 add).  That is
//     8 instructions per *byte* of metadata, so 8.44 MiB takes 21.5 us
//     (394 GB/s) instead of the 8.8 us DRAM floor.
//
//  3. qkv_qkvr_prep_fused_decode — 21.05 MB of Q8_0 weights in 29.41 us
//     (720 GB/s) at 288 blocks x 128 threads = 3 waves/SIMD.  Step 7F measured
//     the same box at 6 waves/SIMD: 3 waves -> 690-737 GB/s, 6 waves ->
//     884-891 GB/s (lm_head / wo_v3).  The kernel is purely occupancy-limited.
//
// ----------------------------------------------------------------------------
// v3 DESIGN
//
//  A. FA v3 (SWA + sparse Global), one skeleton, 3 KV formats:
//     * Q is converted ONCE to fp16 (16 x __half2 per GQA head = 48 VGPRs) and
//       kept in registers for the whole kernel -> ZERO ds_load inside the tile
//       loop.  K/V are 4-bit, so fp16 Q (11-bit mantissa) is 100x more precise
//       than the KV it is dotted against; every dot product accumulates in FP32
//       (standard fp16xfp16->fp32 FlashAttention, no quality loss).
//     * K.Q uses `__builtin_amdgcn_fdot2` (RDNA3 `v_dot2_f32_f16`, one
//       instruction = 2 MACs).  The 4-bit nibble -> fp16 unpack is the IEEE-754
//       fp16 trick `half(1024 + n) = 0x6400 | n` (mantissa LSB of 1024.0 is
//       exactly 1.0), produced by ONE `v_perm_b32` per 2 nibbles, and the
//       mandatory -1024 is folded into ONE scalar term
//       `(k_zero - 1024*k_d) * q_sum` per (head, key) instead of being
//       subtracted per element.  Per 8-nibble word: 1 and + 1 shift + 1 and +
//       4 v_perm_b32 = 7 instructions (v2: 8 instructions *and* 8 fma halves),
//       then 12 v_dot2_f32_f16 for the 3 GQA heads instead of 24
//       v_fma_mix_f32.  Same trick, same exactness, for Q4_0 (bias 1032) and
//       Q8_0 (bias 1152, `(byte^0x80)|0x6400`).
//     * The P.V path is NOT converted to fdot2 (it would mix output dims): it
//       stays 4 dims x 3 heads x 8 keys FP32 v_fma_mix_f32, but the 1024 bias
//       is undone by an exact `v_pk_add_f16` so the accumulation order and the
//       precision of v2 are preserved bit-for-bit up to FP32 rounding.
//     * CTA = 8 warps (v2: 4).  SWA 12 splits x 8 KV heads = 96 CTAs, Global
//       16 splits x 8 KV heads = 128 CTAs, so the split-K combine and its
//       scratch layout are unchanged and `n_splits` semantics match v2.
//     * Tiles are distributed cyclically over the CTA's warps (v2 used
//       contiguous chunks, which gave the first warps 3 tiles and left the last
//       3 warps of 4 idle for SWA's 11 tiles per split) and the one redundant
//       `__shfl(m,0,32)` broadcast after the xor-4/8/16 max reduction is
//       dropped (all lanes already hold it).
//     * COMPILED RESOURCE USE (see the table at the bottom): 157-162 VGPR,
//       12484 B LDS, 0 spills, 0 private segment.  That is 1 CTA/CU, i.e.
//       8 warps/CU = 2 waves/SIMD versus v2's 4 warps/CU = 1 wave/SIMD.
//       v2 used 107-112 VGPR at 128 threads, which *also* fits only 1 CTA/CU,
//       so doubling the warps per CTA is free (v3 uses more VGPRs than the
//       ~90 the design asked for because the scheduler hoists all four K words'
//       nibble unpack and the whole V word set; it is 0-spill, so that is a
//       schedule win, not waste).
//
//  B. page-select v3:
//     * algebraic rewrite of the Quest bound.  For any q and any
//       k_min <= k_max:  max(q*k_min, q*k_max) = q+ * k_max + q- * k_min  with
//       q+ = max(q,0), q- = min(q,0) -- exactly, with no fmaxf and with half
//       the products.  q+ / q- are staged as __half2 (16 dims per lane, 8
//       int8 per lane) and the 128-dim page bound is 48 `v_dot2_f32_f16`
//       instead of 48 fmaxf + 96 mul + 48 add.
//     * int8 -> fp16 uses `(byte ^ 0x80) | 0x6400` again; the -1152 is folded
//       into ONE scalar per head (`sum_d q_d`) because q+ + q- = q.
//     * the shared page scale k_mm_d is applied ONCE per page, after the sum.
//     * COMPILED: 104 VGPR (v2: 120) -> 2 CTAs/CU = 16 warps/CU = 4 waves/SIMD,
//       the same occupancy as v2, but the scoring loop drops from ~250 to ~87
//       instructions per page per lane (verified: the kernel's whole text has
//       124 v_dot2 and ZERO v_fma_mix_f32 in the scoring phase).
//       `__launch_bounds__(256,3)` + `amdgpu_waves_per_eu(6,6)` are declared
//       but this ROCm/clang does not enforce either as a register cap, so 6
//       waves would need the q+-/q- staging moved to LDS (documented below).
//     * 36 chunks x 8 KV heads = 288 CTAs (v2: 24 x 8 = 192).  The Top-K is
//       the v2 register-resident 4-pass 8-bit radix select (unchanged, exact,
//       deterministic, ties broken in scan order).
//     NOTE ON PARITY: q is rounded to fp16 before the dot.  The page bound
//     keeps a >= 1-LSB int8 widening (see sparse_global.h), which is ~16x
//     larger than the fp16 rounding of q, so the bound stays conservative;
//     the resulting page *ranking* can differ from v2 only for pages within
//     ~5e-4 relative of each other.
//
//  C. QKV+qkvr_prep v3: identical math (Q-norm, K-norm, RoPE-free FWHT-128,
//     q4_1/q4_0/q8_0 KV store, qgate, rel_table), identical sharded
//     self-resetting grid barrier over the same 288 blocks, but 288 x 256
//     threads (8 warps): COMPILED 75 VGPR, 10368 B LDS -> 3 CTAs/CU =
//     24 warps/CU = **6 waves/SIMD** instead of 3.  The rel-table LDS staging
//     is now a union member of the GEMV x-buffer (they are separated by the
//     grid barrier), which keeps LDS at 10.4 KiB; the two separate buffers
//     (12.4 KiB) allowed only five 128-thread CTAs.  The launcher probes
//     hipOccupancyMaxActiveBlocksPerMultiprocessor once (cached) and falls
//     back to the 288 x 128 (3 waves/SIMD) shape if 3 CTAs/CU cannot be
//     co-resident, so the in-kernel barrier can never deadlock.
//
// ----------------------------------------------------------------------------
// COMPILED RESOURCES (llvm-readobj --notes on /tmp/kernels_fa_decode_v3.o,
// gfx1100, hipcc -O3, wave32).  All nine instantiated kernels: 0 VGPR spills,
// 0 SGPR spills, private_segment_fixed_size = 0, no scratch_/flat_ anywhere.
//
//   kernel                     VGPR  SGPR    LDS  threads  CTAs/CU  warps/CU
//   fa_swa_v3   <q8_0/q4_1/q4_0> 157-162 57-58 12484  256       1        8
//   fa_global_v3<q8_0/q4_1/q4_0> 158-162 57-58 12484  256       1        8
//   select_v3                    104   46    1124   256       2       16
//   qkv_qkvr_v3 (8 warps)         75   55   10368   256       3       24
//   qkv_qkvr_v3 (4 warps, fallbk) 75   55   10368   128       6       24
//
// ISA facts checked on the object (not inferred):
//   * `v_dot2acc_f32_f16` present in all six FA kernels (108-126) and in
//     select_v3 (124); select_v3 has ZERO v_fma_mix_f32.
//   * ZERO `ds_load` of any width inside the FA 8-key tile loop (v2 had 24
//     ds_load_b128 + an s_waitcnt lgkmcnt per tile).
//   * `global_load_b128 ... slc dlc` on every FA-Global K/V load, every
//     select_v3 metadata load, and every QKV weight load (24 per kernel).
//   * insert/emitted text size: fa_swa_v3<q4_1> 3566 instructions vs v2's
//     3451 -- larger only because the epilogue now reduces 8 warps instead of
//     4; per 8-key tile the K path loses 96 v_fma_mix_f32 + 24 ds_load_b128 +
//     16 v_pk_add_f16 and gains 42 v_dot2_f32_f16, while the P.V path is
//     unchanged by design.
//   * qkv_qkvr_v3<8 warps> 2793 instructions vs v2's 2817 at identical math.
//
// ----------------------------------------------------------------------------
// MEASURED (integrated, rocprofv3 kernel stats, d = 262144, q4_1, sparse 96,
// 32 decode steps, same session; `--fa-swa|--fa-select|--fa-global v1|v2|v3`):
//
//   kernel                        calls/tok     v1        v2        v3
//   SWA dense FA                     40     12.25 us  10.48 us  10.28 us
//   sparse Global FA                  8     31.19 us  29.85 us  20.26 us   <<
//   page select                       8     39.16 us  26.75 us  27.48 us
//   QKV+qkvr prep                    48     29.40 us  29.40 us  30.62 us
//   FA+select total (us/tok)                1052.7     872.0     793.1
//
// End-to-end tg128 @ 262144 (sparse 96): v1 FA 125.46, v2 FA 130.06,
// v3 FA+select 130.52 tok/s; the needle-safe mix (v1 FA + select v2) 127.56.
//
// TWO REAL BUGS FOUND BY NUMERIC PARITY VS v1 (not compile-time checks):
//  1. v3 FA (both kernels) dropped the `mt = __shfl(mt, 0, 32)` broadcast.
//     That is only a no-op when the 4 lanes of one key hold the same score --
//     but the RelBias is added on the blk==0 lanes only, so the xor-4/8/16
//     butterfly returns a *per-blk* max.  m_new, the o/l/ob rescale and l_reg
//     then diverge across the 4 lane groups: measured 1-10% error per split
//     with rel != 0, 1e-6 exact with rel == 0, and 3.4e-2 on real 258k SWA
//     data.  Broadcast restored -> all 9 instantiations now within 2.6-5.6e-6
//     of v1 (fp16-Q rounding only), 0/97 rel-distance pairs mismapped.
//  2. qkv_qkvr_prep_fused_v3: `V3QkvSeg s[3]` is POD; with nseg < 3 (the SWA
//     proj_qkv_all shape: 8600 rows = 269 blocks at 288 launched) the tail
//     blocks selected an UNINITIALIZED segment -> HIP error 700 (illegal
//     memory access) on the 9th decode step.  The launcher now aliases the
//     spare segments to segment 0 (rows = 0) and gives the last real segment
//     blk_end = 288 so `nblk`/`step` cannot overflow into negative rows.
//
// ADOPTION (Step 7G): the SWA/Global FA kernels are faster (v3 Global -9.6 us
// per call) but EVERY reordered FA shifts the decode trajectory by 1 ulp, and
// this model amplifies a 1-ulp FA change into ~0.3 logits through MoE top-8
// near-ties: the 258k needle (`7492-AURORA-8831`) flips with any v2/v3 FA, and
// even with v1 FA + `--moe-gu-v3 auto`.  The shipped default is therefore the
// bit-exact reference (v1 FA + select v2, which emits the identical page list);
// everything here stays reachable as an opt-in A/B switch.
//
// ----------------------------------------------------------------------------
// PROJECTED (from the ISA deltas + Step 7F's measured occupancy curve; the FA
// projections below were NOT met -- see MEASURED above):
//   qkv_qkvr_prep_v3        29.41 -> ~24.0 us  (720 -> 880 GB/s)  -215 us/tok
//   fa_decode_swa_v3         9.90 -> ~6.0  us   2x warps + no LDS    -156 us/tok
//   fa_sparse_global_v3     25.57 -> ~17.0 us                     -69  us/tok
//   sparse_global_select_v3 21.50 -> ~11.0 us  (394 -> 770 GB/s)  -84  us/tok
//                                                               = -524 us/tok
//   The FA numbers assume latency hiding, not instruction count, is the limit
//   (v2 runs at 1 warp/SIMD at ~19 cycles/instruction, so 2 warps/SIMD plus
//   the removal of the 24 LDS reads per tile is the dominant effect).
// ============================================================================

#include "inkling_engine.h"
#include "sparse_global.h"

#include <cstddef>
#include <cstdint>

namespace inkling {

// ----------------------------------------------------------------------------
// Launch shapes (96 CU / 48 WGP gfx1100, Wave32)
// ----------------------------------------------------------------------------
// Both FA kernels use 8 warps per CTA (v2 used 4).  At the same CTA count that
// doubles warps/SIMD for free -- the split-K combine cost is unchanged.
constexpr int FA_V3_NWARPS         = 8;
constexpr int FA_V3_SWA_SPLITS     = 12;   // 12 x 8 KV heads =  96 CTAs
constexpr int FA_V3_GLOBAL_SPLITS  = 16;   // 16 x 8 KV heads = 128 CTAs
constexpr int FA_V3_SELECT_CHUNKS  = 36;   // 36 x 8 KV heads = 288 CTAs
constexpr int FA_V3_MAX_SPLITS     = 32;   // split-K combine is one lane/split
constexpr int FA_V3_SEL_THREADS    = 256;
constexpr int FA_V3_SEL_SLOTS      = SPARSE_MAX_PAGES / FA_V3_SEL_THREADS; // 16
constexpr int FA_V3_QKV_BLOCKS     = 288;  // == the barrier's shard totals
constexpr int FA_V3_QKV_THREADS    = 256;

static_assert(FA_V3_SWA_SPLITS    >= 1 && FA_V3_SWA_SPLITS    <= FA_V3_MAX_SPLITS, "split count");
static_assert(FA_V3_GLOBAL_SPLITS >= 1 && FA_V3_GLOBAL_SPLITS <= FA_V3_MAX_SPLITS, "split count");
static_assert(SPARSE_MAX_PAGES % FA_V3_SEL_THREADS == 0, "select slots");

// ----------------------------------------------------------------------------
// 1. SWA dense decode FA v3 (40 layers, window = SWA_WINDOW = 1024)
// ----------------------------------------------------------------------------
// Contract identical to launch_fa_swa_decode_v2 (drop-in):
//   d_meta != nullptr, rel_table != nullptr, `fa_done` = >= N_HEAD_KV
//   zero-initialised int32 self-resetting counters (only touched when
//   n_splits > 1); tmp_o[N_HEAD * n_splits * HEAD_DIM] floats and
//   tmp_meta[N_HEAD * n_splits] float2 are only written when n_splits > 1.
// Results agree with v2 up to FP32 rounding order (Q is rounded to fp16 before
// the K dot; the softmax/RelBias/sink/qgate/FWHT/split-combine math is
// unchanged).
void launch_fa_swa_decode_v3(
        const float * q,                               // [N_HEAD, HEAD_DIM]
        KVCacheSlot   kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,                       // [N_HEAD, N_REL_EXTENT]
        const float * qgate_in, int qgate_stride,
        float       * out,                             // [N_HEAD, HEAD_DIM]
        float       * tmp_o,
        float2      * tmp_meta,
        int           n_splits,
        const StepMeta * d_meta,
        hipStream_t   stream,
        int32_t     * fa_done);

// ----------------------------------------------------------------------------
// 2. Fused page score + Top-K v3 (8 Global layers)
// ----------------------------------------------------------------------------
// Drop-in for launch_sparse_global_select_v2: writes
// scratch.page_scores[0][h][p] (debug/parity), scratch.selected_pages[0][h][*]
// and scratch.n_selected[0][h] in the exact v1/v2 layout
// [sink pages 0,1][top-K historical][last SPARSE_RECENT_PAGES pages].
// `select_done` = >= N_HEAD_KV zeroed self-resetting counters that must NOT
// alias the FA `fa_done` counters.
void launch_sparse_global_select_v3(
        const float * q,                               // [N_HEAD, HEAD_DIM]
        const SparseGlobalPageMeta & meta,
        SparseGlobalScratch & scratch,
        const StepMeta * d_meta,
        int             sparse_pages,
        int32_t       * select_done,
        hipStream_t     stream);

// ----------------------------------------------------------------------------
// 3. Sparse Global decode FA over the selected 64-token pages (8 layers)
// ----------------------------------------------------------------------------
// Drop-in for launch_fa_sparse_global_decode_v2 (same contract as 1.).
void launch_fa_sparse_global_decode_v3(
        const float * q,
        KVCacheSlot   kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,
        const float * qgate_in, int qgate_stride,
        float       * out,
        const int32_t * selected_pages,
        const int32_t * n_selected_pages,
        float       * tmp_o,
        float2      * tmp_meta,
        int           n_splits,
        const StepMeta * d_meta,
        hipStream_t   stream,
        int32_t     * fa_done);

// ----------------------------------------------------------------------------
// 4. QKV GEMV + qkvr_prep fused v3 (48 calls/layer-set, 6 waves/SIMD)
// ----------------------------------------------------------------------------
// Drop-in for launch_qkvr_prep_fused_decode's fused entry point
// (launch_qkv_qkvr_prep_fused_decode): W0/W1/W2 are the same segments
// (proj_q4 | proj_k | proj_v views of proj_qkv_all), q_in / r_in / k_raw /
// v_raw / k_dyn / v_dyn point at the projections the GEMV just writes, and
// `scratch` is buf_router_scratch_ (uses slots 392..395, same as v2, and
// leaves them at zero).
//
// Bit-identical math to v2 (same per-lane block order, same FP32 FMA chain,
// same FWHT-128, same KV quantisation).  The only changes are 256 threads per
// block (RPB 16 -> 32 rows/block, one GEMV step instead of a loop) and the
// occupancy target.  Launches 288 blocks; the sharded grid barrier is the same
// 3 x 96 + ack structure, so it is deadlock-free only if all 288 blocks are
// co-resident -- the launcher checks that with a cached
// hipOccupancyMaxActiveBlocksPerMultiprocessor and falls back to the
// 288 x 128 / 3-waves shape otherwise.
//
// If any pre-condition fails (K != N_EMBD, |rows| > 12288, scratch == nullptr)
// the launcher simply returns without launching, exactly like v2's early-exit
// contract expects the caller to have checked; callers must therefore keep
// their own fallback (engine.cpp already does).
void launch_qkv_qkvr_prep_fused_v3(
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

} // namespace inkling
