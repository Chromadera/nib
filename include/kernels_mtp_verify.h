#pragma once

// Step 8A: target-side support kernels for DSpark MTP speculative decoding.
//
// These are only ever launched when MTP I/O is enabled; the non-MTP decode and
// prefill graphs never reference them, so their existence costs the baseline
// nothing (no VRAM, no dispatches, no different numerics).
//
//  * launch_mtp_capture_tap    — bf16 copy of the hidden entering visit 5 / 23 /
//                                41 and of the final normed hidden (tap 3) into
//                                d_taps[r * 4 + tap] (*_h, N_EMBD), exactly the
//                                [n, 4, H] layout the drafter's commit() reads.
//  * launch_mtp_copy_raw_pair  — the raw pre-conv K/V projection rows of one
//                                verify row (they are still in the projection
//                                buffer when the fused prep epilogue returns).
//  * launch_mtp_save_hist      — save every slot's short-conv [6656][3] history
//                                before a speculative verify call.
//  * launch_mtp_rebuild_hist   — restore it to the exact state after `commit`
//                                rows from the saved state + the m recorded raw
//                                pre-conv inputs (the history holds the LAST 3
//                                raw inputs per channel, oldest first, so it is
//                                a pure 3-element window reconstruction).

#include "inkling_engine.h"

namespace inkling {

void launch_mtp_capture_tap(const float * src, uint16_t * taps, int n_rows, int tap, hipStream_t stream);

void launch_mtp_copy_raw_pair(const float * k_raw, int k_stride, const float * v_raw, int v_stride,
                              float * dst, hipStream_t stream);

// All `nrows` verify rows in ONE launch: row r copies k_raw[r*k_stride ..] and
// v_raw[r*v_stride ..] into dst[r*dst_stride ..].
void launch_mtp_copy_raw_pair_mrows(const float * k_raw, int k_stride, const float * v_raw,
                                    int v_stride, float * dst, int dst_stride, int nrows,
                                    hipStream_t stream);

void launch_mtp_save_hist(float * const * hist_ptrs, float * bak, hipStream_t stream);

void launch_mtp_rebuild_hist(float * const * hist_ptrs, const float * bak, const float * raw,
                             int commit, hipStream_t stream);

constexpr int MTP_VERIFY_TOPK = 256;

// Exact GPU top-256 extraction + full-vocab max_l and sum(exp((l - max_l)*inv_t))
// for all `m` verify rows in parallel (one 256-thread CTA per row).
void launch_mtp_verify_topk(const float * logits, int m, int vocab_size, float inv_t,
                            float * out_vals, int32_t * out_ids,
                            float * out_max_l, float * out_sum_exp, hipStream_t stream);

// ---------------------------------------------------------------------------
// Exact full-vocab categorical sampling on the GPU (issue #4: unbounded
// top_k = 0 / top_p = 1 sampling used to copy all 200,058 logits of every
// verify row to the host and std::sort them).
//
// For every row r (0 <= r < m, m <= 8) with M = max_x l_x and
//   e_x  = expf((l_x - M) * inv_t)                (unnormalised target weight)
//   keep = { x : e_x >= min_p }                   (min_p <= 0 keeps every x;
//                                                  e_argmax = 1 is always kept)
// the launch returns, with ZERO per-logit host traffic:
//   out_tok[r]       = a sample from p(x) = e_x / Z over keep \ {excl[r]}
//                      (excl[r] < 0: over all of keep), or -1 if that set is
//                      empty.  Drawn with the exponential race
//                      argmax_x [ (l_x - M) * inv_t - log E_x ],  E_x ~ Exp(1)
//                      (Gumbel-max), i.e. P(out = x) = e_x / sum_{y in set} e_y
//                      exactly, for any set -- which is what both the bonus draw
//                      (excl = -1) and the speculative residual
//                      norm(max(0, p - delta_draft)) = p(x)/(1 - p(draft)), x != draft,
//                      need.  E_x comes from Philox4x32-10(counter = {x, r, 0, 0},
//                      key = seed), so rows / calls are independent for distinct seeds.
//   out_stats[4r+0]  = M
//   out_stats[4r+1]  = Z = sum_{x in keep} e_x     (fixed-order, deterministic)
//   out_stats[4r+2]  = e_excl = expf((l_excl - M) * inv_t)  (0 if excl < 0)
//   out_stats[4r+3]  = 1 if excl[r] is in keep, else 0
// so the host's acceptance probability is p(draft) = keep ? e_excl / Z : 0.
// `scratch` must hold 4 * MTP_SAMPLE_MAX_ROWS * MTP_SAMPLE_BLOCKS words
// (MTP_SAMPLE_SCRATCH_WORDS).  Stream-ordered; no host traffic, no atomics.
// ---------------------------------------------------------------------------
constexpr int MTP_SAMPLE_BLOCKS  = 32;   // CTAs per row (one vocab chunk each)
constexpr int MTP_SAMPLE_MAX_ROWS = 8;
constexpr int MTP_SAMPLE_SCRATCH_WORDS = 4 * MTP_SAMPLE_MAX_ROWS * MTP_SAMPLE_BLOCKS;

struct MtpSampleExcl {
    int32_t tok[MTP_SAMPLE_MAX_ROWS];
};

struct MtpSampleBatchParams {
    int32_t  row_idx[MTP_SAMPLE_MAX_ROWS]; // source row index in logits[]
    int32_t  excl[MTP_SAMPLE_MAX_ROWS];    // excluded token (-1 = none)
    float    inv_t[MTP_SAMPLE_MAX_ROWS];   // 1/T (> 0)
    float    min_p[MTP_SAMPLE_MAX_ROWS];   // min_p (>= 0)
    uint32_t k0[MTP_SAMPLE_MAX_ROWS];      // Philox key lo
    uint32_t k1[MTP_SAMPLE_MAX_ROWS];      // Philox key hi
    uint32_t salt[MTP_SAMPLE_MAX_ROWS];    // Philox counter row salt
};

void launch_mtp_sample_full_batch(const float * logits, int m, int vocab_size, int row_stride,
                                  const MtpSampleBatchParams & params, float * scratch,
                                  int32_t * out_tok, float * out_stats, hipStream_t stream);

void launch_mtp_sample_full(const float * logits, int m, int vocab_size, int row_stride,
                            float inv_t, float min_p, const MtpSampleExcl & excl,
                            uint64_t seed, float * scratch,
                            int32_t * out_tok, float * out_stats, hipStream_t stream);

} // namespace inkling
