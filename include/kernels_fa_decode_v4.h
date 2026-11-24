#pragma once

// ============================================================================
// Step 7K — low-VGPR, L3-aware `fa_swa_v4` / `fa_global_v4` and the
//           4-bit outward-rounded Quest metadata (`Q4_MM`) + `select_v4`.
//
// Everything here is implemented in src/kernels_fa_decode_v4.hip and is
// self-contained (links against nothing but the HIP runtime).  All launches are
// capture-safe (no host sync, no allocation, self-resetting device counters
// only), so every entry point is a drop-in replacement inside the existing
// static decode HIP graph.
//
// ----------------------------------------------------------------------------
// WHY (Step 7J rocprofv3 @ d = 262144, defaults + FA v3):
//
//   SWA FA v3              40 calls x 10.2 us = 408 us/tok   (52.4 MB q4_1 /
//                                                            89.1 MB q8_0 KV
//                                                            -- both <= 96 MB L3)
//   sparse_global_select   8 calls x 21.4 us = 171 us/tok   (67.5 MiB of int8
//                                                            metadata @ 395 GB/s)
//   Global FA v3           8 calls x 18.1 us = 145 us/tok   (63 MB)
//                                                          --------------
//                                                           724 us/tok
//
// Two structural causes, both read out of the compiled ISA / resource notes:
//
//  1. fa_*_v3 compiles to 157-162 VGPR -> ONE 256-thread CTA per CU, i.e.
//     2 waves/SIMD (a SIMD holds 512 VGPRs of Wave32 state, so waves/SIMD =
//     512 / VGPR).  48 of those VGPRs are `V3Q<3>`: fp16 Q (3 heads x 16
//     __half2) held in registers *identically on every one of the 8 k_slot
//     lanes*, and the scheduler additionally hoists all four K words and the
//     whole V word set inside the tile loop.
//  2. fa_swa_v3 loads its KV with `__builtin_nontemporal_load` (`slc dlc`),
//     which streams past the caches.  The whole 1024-token SWA window of all
//     40 SWA layers is 36.7 MB (q4_1) / 55.1 MB (q8_0) and every token revisits
//     exactly the same bytes, so the 96 MB Infinity Cache is the right home for
//     it -- and 10.2 us for 0.92 MB/call is 90 GB/s, i.e. pure untouched
//     latency, not bandwidth.
//
// v4 DESIGN (A/B-able through the CLI, all combinations are bit-exact copies of
// the v3 arithmetic -- identical fdot2 order, identical shuffles, identical
// softmax/RelBias/sink/qgate/FWHT/split-combine -- so every variant differs
// from v3 only in *scheduling* and cache policy):
//
//   * `q_lds == 1` stages Q as `__shared__ __half2 s_q[GQA][64]` (768 B) and
//     reads the 4 half2 of one 8-dim group with ONE `ds_read_b128` inside the
//     K dot.  All 8 k_slot lanes that share `blk` read the SAME 16 bytes, which
//     is an LDS hardware broadcast (no bank conflict), so the 48 pinned Q VGPRs
//     become 4 transient ones.
//   * `cached == 1` replaces every `__builtin_nontemporal_load` in the KV path
//     with a normal (L1/L2-backed) 128-bit vector load.
//   * `n_splits` is doubled for SWA (12 -> 24) and raised for Global
//     (16 -> 24): a 96-CU part needs >= 192 CTAs before a 2-CTA/CU occupancy is
//     reachable at all, and SWA/Global are latency bound (0.92 MB per SWA call,
//     5.5 MB per Global call).
//
// Q4_MM (`SPARSE_META_Q4`): one page/head keeps `k_qmm[128]` (nibble-packed
// outward-rounded min/max + one shared fp16 scale) = 130 B instead of
// 258 B, i.e. exactly half the DRAM traffic and 32 MiB less VRAM at 256k.
// The bound stays strictly conservative: `d = __float2half_ru(amax/7 * (1+2^-9))`,
// `q_min = floor(k_min/d)`, `q_max = ceil(k_max/d)` (both clamped to +-7, which
// is safe because |k| <= amax = 7*d), so `q_min*d <= k_min <= k_max <= q_max*d`
// for every dimension.  The nibble layout lets the scorer unpack one byte into
// the packed half2 pair `half(1024 + n) = 0x6400 | n` with the same ONE
// `v_perm_b32` per two values that the int8 path uses, with the -1032 bias
// folded into one scalar per head (`q+ + q- == q`).
// ----------------------------------------------------------------------------
// MEASURED (gfx1100 GPU 0, all numbers end-to-end `bench -p 0 -n 128 -r 3`
// unless stated otherwise; every variant is the SAME arithmetic, so any
// difference below is scheduling + cache policy only).
//
// 1. COMPILED RESOURCES (`llvm-readobj --notes`, 0 VGPR/SGPR spills, 0 private
//    segment on all 32 instantiations):
//
//      kernel                     QLDS=1 (Q in LDS)   QLDS=0 (Q in regs)
//      fa_swa_v4   q4_1/q4_0/q8_0   117-128 VGPR        161-162 VGPR
//      fa_global_v4 q4_1/q4_0/q8_0  121-131 VGPR        157-161 VGPR
//      select_v4 q8 (int8 meta)     123 VGPR, 2660 B    106 VGPR, 1124 B
//      select_v4 q4 (Q4_MM)         112 VGPR, 2660 B    103 VGPR, 1124 B
//
//    A SIMD holds 512 Wave32 VGPRs, so 161 VGPR = 2 waves/SIMD and 117 = 4
//    waves/SIMD.  THE OCCUPANCY DOUBLING LOSES: at d = 2048 (SWA window full,
//    cheap prefill, so the FA is a large share of the step) register-Q +
//    nontemporal measures 136.51 tok/s against 135.54 for the LDS-Q build and
//    135.56 for v3.  The 12 extra `ds_read_b128` + their `s_waitcnt lgkmcnt`
//    per 8-key tile inside the K dot cost more than a third wave/SIMD buys.
//    Register Q + nontemporal is therefore the shipped `--fa-q-lds 0`.
//
// 2. CACHE POLICY: `__builtin_nontemporal_load` (`slc dlc`) WINS for the SWA
//    ring even though the whole window (36.7 MB q4_1 / 55.1 MB q8_0) fits in
//    the 96 MB Infinity Cache.  Measured at the same split count:
//      136.51 (nontemporal) vs 135.54 (L1/L2-backed)  d = 2048, q4_1
//      132.99 (nontemporal) vs 132.59 (cached)        d = 131072, q4_1
//    The window is not re-read *within* a call (each byte is touched once), so
//    caching only evicts the weight streams -- the exact opposite of the
//    hypothesis in the Step 7K brief, which is why it is measured per call.
//
// 3. SPLITS.  SWA is latency-, not bandwidth-bound (0.92 MB per call at
//    90 GB/s effective), so split count dominates:
//      d = 2048:  6 -> 135.56, 12 -> 136.07, 16 -> 135.74, 24 -> 136.51, 32 -> 133.02
//      d = 131072:      12 -> 132.70, 24 -> 132.99
//    but PPL depends on the SWA split partition (1-ulp combine reorder, and
//    this model amplifies it): 12 -> 3.5664, 16 -> 3.6124, 20 -> 3.5931,
//    24 -> 3.5852.  **12 splits (PPL 3.5664, better than the 3.5788 v1
//    reference) is the shipped default**; 24 is ~0.3 tok/s faster and stays
//    behind `--fa-swa-splits 24`.
//
// 4. Q4_MM: half the metadata DRAM and 32 MiB less VRAM, +1.0 tok/s
//    (133.29 vs 132.34 at d = 262144 / sparse 96), but the 15-level outward
//    bound is ~18x looser than int8 and the 258k needle page falls out of the
//    top-K at BOTH 96 and 112 pages (with the bit-exact v1 FA too), so it ships
//    as `--sparse-meta q4` only.  Int8 metadata + `select_v4` is also slower
//    than the shipped `launch_sparse_global_select_v2` (132.53 vs 133.22 at
//    d = 131072), so `--fa-select v4` exists to serve Q4_MM, not as a default.
//
// 5. END-TO-END (d = 262144, -p 0 -n 128 -r 3):
//      q4_1 sparse 96   129.32 (v1 FA)  -> 131.19 (FA v3) -> 132.34 (v4)
//      q4_1 sparse 112                                 -> 131.88 (v4, needle-safe)
//      q8_0 sparse 96   128.51 (v1 FA)                  -> 131.31 (v4)
//      q8_0 sparse 112                                 -> 130.84 (v4, needle-safe)
//      q8_0 sparse 0 (dense exact)  78.35               -> 78.90 (v4)
//    Profiled decode cost of the whole attention stack (rocprofv3, d = 262144,
//    sparse 112): SWA 400 + select 178 + global 161 + minmax 10 = 749 us/tok,
//    against 724 us/tok in Step 7J at sparse 96 (i.e. 112 pages costs ~17% more
//    global traffic).  The remaining gap to the DRAM floors (SWA 0.95, select
//    8.8, global 5.7 us/call) is dependent-chain latency inside one 8-key tile
//    (667 `ds_bpermute_b32` + 802 `s_waitcnt` in the compiled SWA body), which
//    neither occupancy nor cache policy moves.
// ============================================================================

#include "inkling_engine.h"
#include "sparse_global.h"

#include <cstddef>
#include <cstdint>

namespace inkling {

// ----------------------------------------------------------------------------
// Launch shapes (96 CU / 48 WGP gfx1100, Wave32)
// ----------------------------------------------------------------------------
constexpr int FA_V4_NWARPS          = 8;
constexpr int FA_V4_SWA_SPLITS      = 12;   // 12 x 8 KV heads =  96 CTAs (PPL-optimal)
constexpr int FA_V4_GLOBAL_SPLITS   = 16;   // 16 x 8 KV heads = 128 CTAs
constexpr int FA_V4_MAX_SPLITS      = 32;
constexpr int FA_V4_SEL_THREADS     = 256;
constexpr int FA_V4_SEL_SLOTS       = SPARSE_MAX_PAGES / FA_V4_SEL_THREADS; // 16
constexpr int FA_V4_SELECT_CHUNKS   = 16;   // 16 x 8 KV heads = 128 CTAs

static_assert(FA_V4_SWA_SPLITS    >= 1 && FA_V4_SWA_SPLITS    <= FA_V4_MAX_SPLITS, "split count");
static_assert(FA_V4_GLOBAL_SPLITS >= 1 && FA_V4_GLOBAL_SPLITS <= FA_V4_MAX_SPLITS, "split count");
static_assert(SPARSE_MAX_PAGES % FA_V4_SEL_THREADS == 0, "select slots");

// ----------------------------------------------------------------------------
// 1. SWA dense decode FA v4 (40 layers, window = SWA_WINDOW = 1024)
// ----------------------------------------------------------------------------
// Contract identical to launch_fa_swa_decode_v3 (drop-in).
//   q_lds  : 1 = Q staged in LDS (frees 48 VGPRs), 0 = Q in registers (= v3)
//   cached : 1 = normal cached KV loads, 0 = `slc dlc` streaming (= v3)
void launch_fa_swa_decode_v4(
        const float * q,                               // [N_HEAD, HEAD_DIM]
        KVCacheSlot   kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,
        const float * qgate_in, int qgate_stride,
        float       * out,
        float       * tmp_o,
        float2      * tmp_meta,
        int           n_splits,
        const StepMeta * d_meta,
        hipStream_t   stream,
        int32_t     * fa_done,
        int           q_lds,
        int           cached);

// Step 8C: all `nrows` speculative-verify rows in ONE launch (grid.z = row).
// `nrows == 1` forwards to the single-row launcher.  Base pointers are the row-0
// ones; each row's q/out/sinks/qgate/rel_table/tmp_o/tmp_meta/d_meta and its
// N_HEAD_KV arrival counters (fa_done + row*N_HEAD_KV) are derived in-kernel.
void launch_fa_swa_decode_v4_mrows(
        const float * q,
        KVCacheSlot   kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,
        const float * qgate_in, int qgate_stride,
        float       * out,
        float       * tmp_o,
        float2      * tmp_meta,
        int           n_splits,
        const StepMeta * d_meta,
        hipStream_t   stream,
        int32_t     * fa_done,
        int           q_lds,
        int           cached,
        int           nrows);

// ----------------------------------------------------------------------------
// 2. Sparse Global decode FA v4 over the selected 64-token pages (8 layers)
// ----------------------------------------------------------------------------
void launch_fa_sparse_global_decode_v4(
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
        int32_t     * fa_done,
        int           q_lds,
        int           cached);

// Step 8D: all `nrows` speculative-verify rows in ONE launch (grid.x = row,
// grid.y = split, grid.z = grp_id so all m rows of a split/head hit L2 together).
// Each row reads its own selected_pages[r] / n_selected_pages[r] slice and
// owns its own tmp_o / tmp_meta / fa_done[r*N_HEAD_KV] slice (100% bit-identical).
void launch_fa_sparse_global_decode_v4_mrows(
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
        int32_t     * fa_done,
        int           q_lds,
        int           cached,
        int           nrows);

// ----------------------------------------------------------------------------
// 3. Fused page score + Top-K v4 (8 Global layers) for both metadata formats
// ----------------------------------------------------------------------------
// `meta.fmt` selects SPARSE_META_Q8 (int8 min/max, v3 layout) or
// SPARSE_META_Q4 (`k_qmm`).  Writes scratch.page_scores / selected_pages /
// n_selected in the exact v1/v2/v3 layout, same self-resetting `select_done`
// counters (8 of them at buf_fa_done_ + 16).
//   q_lds : 1 = q+ / q- staged in LDS (1.5 KB, 6 waves/SIMD), 0 = registers
void launch_sparse_global_select_v4(
        const float * q,
        const SparseGlobalPageMeta & meta,
        SparseGlobalScratch & scratch,
        const StepMeta * d_meta,
        int             sparse_pages,
        int32_t       * select_done,
        hipStream_t     stream,
        int             q_lds,
        int             cached,
        int             chunks);

} // namespace inkling
