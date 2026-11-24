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

} // namespace inkling
