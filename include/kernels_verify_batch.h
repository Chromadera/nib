#pragma once

// Step 8B: target-side batched (m-token) speculative-verify kernels.
//
// Every kernel here is only ever launched from record_verify_pass() when --mtp
// is active; the non-MTP decode path never references them, so their existence
// costs the baseline nothing (no VRAM, no dispatches, no different numerics).
//
//  * launch_verify_gemv_mrows   — exact m-row SoA Q8 GEMV (1..3 segments, one
//                                 weight stream for all m rows, FP32 x staged
//                                 in LDS; per-row math identical to the engine's
//                                 single-row GEMV).
//  * launch_verify_moe_dedup    — topk_ids[m][8] -> unique experts + per-token
//                                 masks/slots/weights.
//  * launch_verify_moe_gu_*     — unique-expert gate_up + SITU (NVFP4 / Q8_0),
//                                 each expert's weights read ONCE for all of its
//                                 active tokens.
//  * launch_verify_moe_dn_*     — unique-expert down, writing per-(row, slot)
//                                 partial sums into buf_moe_slot.
//  * launch_verify_ffn_post     — fixed-order 8-slot tree sum + causal 4-tap
//                                 shortconv + residual + raw dump (bit-exact to
//                                 m sequential decode steps).
//  * launch_verify_save_raw     — copy one [m, N_EMBD] slab of raw pre-conv
//                                 values into the per-slot rollback buffer.

#include "inkling_engine.h"

#include <cstdint>

namespace inkling {

// m-row SoA Q8 GEMV over up to 3 independent (W, x, y) segments sharing K.
// y[r * ystride + row] = sum_k W_s[row, k] * x_s[r * xstride + k], m = 1..8.
struct VerifyGemvSeg {
    SoaQ8Matrix W;    // by value: the kernel must never see a host pointer
    const float * x;
    float * y;
    int blk_end;      // exclusive end of this segment's block range
};

// Each segment's output row stride is its own W.rows.
void launch_verify_gemv_mrows(const VerifyGemvSeg seg[3], int n_seg, int m,
                              int xstride, hipStream_t stream);

// wo projection with the exact fused 4-tap shortconv + residual epilogue
// (bit-identical to the decode wo kernel), causal across the m rows, plus the
// raw pre-conv dump for the speculative rollback.  ONE 3072-row weight stream
// for all m rows (the shipped per-row path re-reads 7.1 MB per row).
void launch_verify_wo_mrows(const SoaQ8Matrix & W, const float * x, int xstride,
                            const float * conv_w, float * hist, const float * res,
                            float * y, int m, float * raw_out, int raw_stride,
                            hipStream_t stream);

// Expert dedup: unique_exp[u], exp_tok_mask[u] (bit r set = token r routed to
// expert u), exp_tok_slot[u*8+r] (slot k of token r), exp_tok_w[u*8+r],
// exp_tok_cnt[u] (active tokens of u), exp_tok_list[u*8+i] (token id of the i-th
// active token, ascending), *d_num_unique.  max_unique = min(128, m*8).
void launch_verify_moe_dedup(const int32_t * topk_ids, const float * topk_w, int m,
                             int32_t * unique_exp, uint32_t * exp_tok_mask,
                             int32_t * exp_tok_slot, float * exp_tok_w,
                             int32_t * exp_tok_cnt, int32_t * exp_tok_list,
                             int32_t * d_num_unique, hipStream_t stream);

// gate_up + SITU into act[(r*TOP_K + slot) * N_FF_EXP + row].
void launch_verify_moe_gu_nvfp4(const SoaNVFP4Experts & gate_up,
                                const float * x_norm, int xstride,
                                const int32_t * unique_exp, const uint32_t * exp_tok_mask,
                                const int32_t * exp_tok_slot, const float * exp_tok_w,
                                const int32_t * exp_tok_cnt, const int32_t * exp_tok_list,
                                const int32_t * d_num_unique, int m,
                                float * act, hipStream_t stream);
void launch_verify_moe_gu_q8(const SoaQ8Experts & gate_up,
                             const float * x_norm, int xstride,
                             const int32_t * unique_exp, const uint32_t * exp_tok_mask,
                             const int32_t * exp_tok_slot, const float * exp_tok_w,
                             const int32_t * exp_tok_cnt, const int32_t * exp_tok_list,
                             const int32_t * d_num_unique, int m,
                             float * act, hipStream_t stream);

// down: buf_slot[(r*TOP_K + slot) * N_EMBD + row] = dot(down_row, act[r][slot]) * w
void launch_verify_moe_dn_nvfp4(const SoaNVFP4Experts & down, const float * act,
                                const int32_t * unique_exp, const uint32_t * exp_tok_mask,
                                const int32_t * exp_tok_slot, const float * exp_tok_w,
                                const int32_t * exp_tok_cnt, const int32_t * exp_tok_list,
                                const int32_t * d_num_unique, int m,
                                float * buf_slot, hipStream_t stream);
void launch_verify_moe_dn_q8(const SoaQ8Experts & down, const float * act,
                             const int32_t * unique_exp, const uint32_t * exp_tok_mask,
                             const int32_t * exp_tok_slot, const float * exp_tok_w,
                             const int32_t * exp_tok_cnt, const int32_t * exp_tok_list,
                             const int32_t * d_num_unique, int m,
                             float * buf_slot, hipStream_t stream);

// Router-only top-8 over an ALREADY normalised x_norm (the exact decode
// mix_rms_norm reduction), one CTA per token.
void launch_verify_router_topk(const float * x_norm, const __half * router_w,
                               const float * exp_probs_b, int32_t * topk_ids,
                               float * topk_w, int m, hipStream_t stream);

// Fixed-order tree sum of the 8 slots + causal 4-tap shortconv + residual.
// out/res are [m, N_EMBD]; raw_out (optional) receives the pre-conv sum.
void launch_verify_ffn_post(const float * buf_slot, int m, const float * conv_w,
                            float * hist, const float * res, float * out,
                            float * raw_out, int raw_stride, hipStream_t stream);

// raw[t * raw_stride + 0..width) = src[t * N_EMBD + 0..width)
void launch_verify_save_raw(const float * src, int m, int width, float * raw,
                            int raw_stride, hipStream_t stream);

// token ids of the m verify rows: d_verify_meta_[r].token_id -> d_token_ids_[r]
void launch_verify_gather_tokens(const StepMeta * d_meta, int32_t * d_token_ids, int m,
                                 hipStream_t stream);

// By-value per-row short-conv history pointers for independent multi-stream decode.
struct BatchHistPtrs {
    float * p[MAX_CONCURRENT_SLOTS];
};

// Multi-stream independent-slot wo projection (1x weight stream for all m rows,
// each row updates its own independent sconv history in hists.p[t]).
void launch_batch_wo_mrows(const SoaQ8Matrix & W, const float * x, int xstride,
                           const float * conv_w, BatchHistPtrs hists, const float * res,
                           float * y, int m, hipStream_t stream);

// Multi-stream independent-slot FFN 8-slot tree sum + 4-tap shortconv + residual
// (1 parallel launch across all m rows, each row updates hists.p[t]).
void launch_batch_ffn_post(const float * buf_slot, int m, const float * conv_w,
                           BatchHistPtrs hists, const float * res, float * out,
                           hipStream_t stream);

} // namespace inkling
