#pragma once

// ============================================================================
// Step 7E — Standalone decode FA v2 (SWA dense + Quest sparse Global) and the
//             fused page-score / Top-K selector.
//
// Why this exists (measured at d = 262144, `--kv-type q4_1 --sparse-pages 96`,
// from /tmp/prof_step7c_256k/decode_32steps_kernel_stats.csv):
//
//   inkling_fa_decode            (SWA, 40 layers)   12.08 us/call  = 483.0 us/tok
//   sparse_score_pages_decode    (Global, 8 layers) 22.16 us/call  = 177.3 us/tok
//   sparse_topk_pages_decode     (Global, 8 layers) 13.47 us/call  = 107.7 us/tok
//   fa_decode_sparse_global      (Global, 8 layers) 23.55 us/call  = 188.4 us/tok
//                                                    -------------  = 956.4 us/tok
//
// Real root causes (all confirmed in the compiled ISA of build/*.o, not guessed):
//
//  1. SWA dense FA — 108 GB/s effective, not 185 GB/s.  The kernel is NOT
//     bandwidth bound: one SWA layer only touches 8 hd x 1024 keys x 160 B =
//     1.31 MiB (q4_1), i.e. a 1.54 us DRAM floor.  It is (a) instruction bound
//     (the v1 fast path pays 3 shfl + ~3 shuffle-rounds of `exp2f` — the *precise*
//     OpenCL exp2, which the ISA expands into v_exp_f32 + v_ldexp_f32 + a
//     v_div_scale_f32 exception path, 49 of each per kernel, whereas the sibling
//     combine kernel already uses the 1-cycle `v_exp_f32`), (b) it sweeps the
//     whole 1536-slot ring even though only SWA_WINDOW = 1024 slots can be
//     valid (1.5x wasted K loads + k_pos probe + ballot per key), (c) its q4_1
//     V path folds the zero-point into every output dim with an extra FADD
//     (96 extra FADDs per 8-key iteration), and (d) it has no cross-iteration
//     software pipeline.  Its 192 CTAs = 2 CTAs/CU = only 4 waves/SIMD.
//
//  2. sparse_score_pages_decode — 380 GB/s.  One warp scores ONE page with 4 B
//     per lane (`global_load_b32 ... slc dlc`, 128 B per wave-instruction) and
//     then emits three `s_waitcnt vmcnt(N)` one after another, i.e. one load
//     latency exposed per page with no cross-page prefetch; the 3 GQA heads are
//     reduced with 5 butterfly shuffles *per page* (15 ds_bpermute_b32 + ~10
//     ds waitcnts per page in the ISA).  It is a latency/MLP problem, not ALU:
//     the byte stream is 8.4 MiB per layer (0.98 MiB of which is real KV).
//
//  3. sparse_topk_pages_decode — 8 CTAs of 256 threads (8 of 96 CUs busy!),
//     4 radix passes with a *block-shared* LDS histogram atomic per candidate
//     plus a 3-pass re-scan of `page_scores` from global memory (112 loads per
//     thread).  13.47 us of pure latency for 4 KB of L2-hot data.
//
//  4. fa_decode_sparse_global — 334 GB/s.  16 splits x 8 heads = 128 CTAs =
//     1.33 waves over 96 CUs, so a third of the GPU idles in the tail wave
//     (0.67 CTAs/CU in wave 2); 116 VGPRs -> 2.67 waves/SIMD; same precise
//     `exp2f`, same 96-FADD q4_1 V fold, a redundant `k_pos[]` probe + ballot
//     per key (for Global slots `k_pos[j] == j` by construction), and no
//     cross-iteration prefetch (`s_waitcnt vmcnt(0)` immediately after the K/V
//     loads in the ISA).
//
// v2 design (all three kernels share the same skeleton):
//   * 4 lanes own one 128-dim token (16 B = 32 nibbles / 32 int8 per lane), so
//     one Wave32 streams 8 consecutive tokens = 4 full 128 B lines with a single
//     128-bit `global_load_b128` per K and one coalesced 32-bit group per V.
//   * 2-deep software pipeline: the K+V of group g+1 are issued before the
//     arithmetic of group g (v1 waited on every load immediately).
//   * only *valid* keys are enumerated (SWA: exactly min(pos+1, 1024) keys,
//     no k_pos[], no ballot; Global: `j < kv_len_global` because Global slots
//     store position == slot index).
//   * `__builtin_amdgcn_exp2f` (1-cycle v_exp_f32), the same substitution
//     Step 5 already validated for the combine kernel and prefill FA.
//   * q4_1 zero-point folded into ONE scalar accumulator per (head, lane)
//     instead of 4 extra FADDs per key (V += w*d*q4 and ob += w*m at the end;
//     identical math, different FP rounding order).
//   * exactly one wave of CTAs: SWA 12 splits x 8 heads = 96 CTAs = 1 CTA/CU,
//     Global 24 splits x 8 heads = 192 CTAs = exactly 2 CTAs/CU (zero tail).
//   * page score + Top-K fused into ONE kernel (192 CTAs) with a per-head
//     self-resetting arrival counter; the last chunk per head runs a
//     register-resident 4-pass radix select directly out of L2-hot scores.
//
// Everything else (Q scaling, q4_1 zero-point handling of K, RelBias, sinks,
// qgate sigmoid, orthogonal FWHT-128 on the output, split-K combine) is the
// same formula as src/kernels_fa_relbias.hip / src/kernels_sparse_global.hip.
// Results agree with v1 up to FP32 rounding order (the split boundaries and the
// q4_1 V zero-point accumulation order differ by design).
// ============================================================================

#include "inkling_engine.h"
#include "sparse_global.h"

#include <cstddef>
#include <cstdint>

namespace inkling {

// ----------------------------------------------------------------------------
// Launch shapes (defaults tuned for 96 CU / 48 WGP gfx1100, Wave32)
// ----------------------------------------------------------------------------
// SWA:   12 KV splits x 8 KV heads =  96 CTAs = exactly 1 CTA per CU
//        (1024 valid keys / 48 warps = 3 groups of 8 keys per warp, which is
//        the sweet spot: the 2-deep pipeline is full over the whole chunk and
//        the per-CTA prologue/epilogue stays at ~6% of the layer).
// Global: 24 KV splits x 8 KV heads = 192 CTAs = exactly 2 CTAs per CU = 2
//        full waves over 96 CUs (v1 used 16 = 128 CTAs = 1.33 waves, which
//        idles a third of the machine in the tail wave).
// Select: 24 page chunks x 8 KV heads = 192 CTAs; chunks are 4 warps of
//        scoring each, the last arriving chunk per head runs the Top-K.
// ----------------------------------------------------------------------------
// ----------------------------------------------------------------------------
// MEASURED ON THIS BOX (integrated in src/engine.cpp, 256k, q4_1, sparse 96,
// rocprofv3 kernel times, 32 decode steps):
//   inkling_fa_decode_swa_v2_kernel      10.37 us/call (v1 12.08)  -> 415 us/tok
//   inkling_fa_decode_sparse_global_v2   28.37 us/call (v1 23.55) -> 227 us/tok
//   sparse_global_select_v2_kernel       26.62 us/call (v1 22.16 + 13.47 = 35.6)
//   => FA total 864 us/tok vs 956 us/tok for the v1 kernels (-92 us), plus
//      -8 HIP-graph nodes (-26 us).  Net +3.4 tok/s (124.40 -> 127.78).
//   The kernels are ALU-bound, not bandwidth-bound, so they land well above the
//   optimistic targets in the block comment below; the Global FA is actually
//   4.8 us/call SLOWER than v1 at 24 splits (try 16 if you revisit it).
//   A/B of FA_V2_SWA_SPLITS at 256k: 12 -> 127.78 tok/s, 24 -> 127.47 tok/s
//   (d=0: 135.19 vs 135.41).  Kept 12.
//   A/B of FA_V2_GLOBAL_SPLITS with B+C on, 256k: 12 -> 129.10, 16 -> 130.10,
//   24 -> 129.97.  Kept 16 (the v1 grid; 24 CTAs/split-count loses to it).
// ----------------------------------------------------------------------------
constexpr int FA_V2_SWA_SPLITS    = 12;
constexpr int FA_V2_GLOBAL_SPLITS = 16;
constexpr int FA_V2_SELECT_CHUNKS = 24;

// In-kernel split-K combine uses one lane per split, so it needs n_splits <= 32.
constexpr int FA_V2_MAX_SPLITS    = 32;

// KV splits a page-list split owns at `--sparse-pages 96` (>= 96/24 = 4 pages
// => 32 groups of 8 keys => 8 pipelined groups per warp).
static_assert(FA_V2_GLOBAL_SPLITS >= 1 && FA_V2_GLOBAL_SPLITS <= FA_V2_MAX_SPLITS, "split count");
static_assert(FA_V2_SWA_SPLITS    >= 1 && FA_V2_SWA_SPLITS    <= FA_V2_MAX_SPLITS, "split count");

// ----------------------------------------------------------------------------
// 1. SWA dense decode FA (40 layers, window = SWA_WINDOW = 1024)
// ----------------------------------------------------------------------------
// Requires d_meta != nullptr (decode, n_tokens == 1) and rel_table != nullptr.
// `fa_done` must point at >= N_HEAD_KV zero-initialised int32 counters that are
// left at zero after every launch (self-resetting); it is only touched when
// n_splits > 1.  Pass n_splits = 1 for short contexts (no combine at all).
//
// tmp_o / tmp_meta are only written when n_splits > 1:
//   tmp_o   [N_HEAD * n_splits * HEAD_DIM] floats
//   tmp_meta[N_HEAD * n_splits]            float2
void launch_fa_swa_decode_v2(
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
// 2. Fused page score + Top-K (replaces sparse_score_pages_decode +
//    sparse_topk_pages_decode + 8 extra HIP-graph nodes)
// ----------------------------------------------------------------------------
// Writes scratch.page_scores[0][h][p] (kept for debugging / parity checks),
// scratch.selected_pages[0][h][*] and scratch.n_selected[0][h] with the exact
// v1 layout: [sink pages 0,1][top-K historical][last SPARSE_RECENT_PAGES pages].
// `select_done` must point at >= N_HEAD_KV zero-initialised int32 counters
// (self-resetting).  They must NOT alias the FA `fa_done` counters; the engine
// already owns 64 of them, so pass `buf_fa_done_ + 16`.
void launch_sparse_global_select_v2(
        const float * q,                               // [N_HEAD, HEAD_DIM]
        const SparseGlobalPageMeta & meta,
        SparseGlobalScratch & scratch,
        const StepMeta * d_meta,
        int             sparse_pages,
        int32_t       * select_done,
        hipStream_t     stream);

// Step 8D: bit-exact batched m-row Quest page selection (m <= SPARSE_MAX_QB = 8).
// Streams page min/max metadata from HBM ONCE for all `nrows` verify queries
// (vs m separate scans) and runs all `nrows * N_HEAD_KV` Top-K selects in
// parallel without atomic barriers.  Writes row r into
// scratch.selected_pages[r][h][*] and scratch.n_selected[r][h].
void launch_sparse_global_select_v2_mrows(
        const float * q,                               // [nrows, N_HEAD, HEAD_DIM]
        const SparseGlobalPageMeta & meta,
        SparseGlobalScratch & scratch,
        const StepMeta * d_meta,                       // [nrows]
        int             sparse_pages,
        hipStream_t     stream,
        int             nrows);

// ----------------------------------------------------------------------------
// 3. Sparse Global decode FA over the selected 64-token pages (8 layers)
// ----------------------------------------------------------------------------
// Same contract as launch_fa_swa_decode_v2: d_meta != nullptr, rel_table
// non-null, `fa_done` = >= N_HEAD_KV zeroed self-resetting counters.
// `selected_pages` / `n_selected_pages` are the outputs of
// launch_sparse_global_select_v2 (or of the v1 selector, or of the prefill
// selector's q_block 0 slice).
void launch_fa_sparse_global_decode_v2(
        const float * q,                               // [N_HEAD, HEAD_DIM]
        KVCacheSlot   kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,                       // [N_HEAD, N_REL_EXTENT]
        const float * qgate_in, int qgate_stride,
        float       * out,                             // [N_HEAD, HEAD_DIM]
        const int32_t * selected_pages,                // [N_HEAD_KV, SPARSE_MAX_SELECTED]
        const int32_t * n_selected_pages,              // [N_HEAD_KV]
        float       * tmp_o,
        float2      * tmp_meta,
        int           n_splits,
        const StepMeta * d_meta,
        hipStream_t   stream,
        int32_t     * fa_done);

} // namespace inkling
