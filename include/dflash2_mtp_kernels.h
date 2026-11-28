#pragma once

// Native RDNA3/RDNA4 DFlash 2 speculative drafter kernel API
// (src/kernels_dflash2_mtp.hip).
// Every function is a raw-pointer stream-ordered launcher; none allocates or syncs.

#include "inkling_engine.h"

#include <cstdint>

namespace inkling {

// Per-call dynamic parameters for static HIP graph replay of DFlash 2 draft() / commit().
struct DFlash2DraftParams {
    int32_t anchor;   // draft(): anchor token id
    int32_t pos;      // draft(): P (anchor position); commit(): start position
    int32_t ncommit;  // commit(): rows 1..8
    int32_t _pad;
};

// Stream-ordered by-value upload of DFlash2DraftParams.
void launch_dflash2_set_params(DFlash2DraftParams * d_dp, DFlash2DraftParams val,
                               hipStream_t stream);

// ---- Load-time helpers -----------------------------------------------------
// BF16 [rows, K] -> 16-byte-aligned SoA Q8_0 (K must be a multiple of 32).
void launch_dflash2_repack_q8(const uint16_t * bf16, int rows, int K,
                              int8_t * qs, __half * d, hipStream_t stream);

// Precompute bf16(RMSNorm(dequant(tok_embd[mask_token_id]), tok_norm, 1e-6)) [2304].
void launch_dflash2_precompute_mask_embed(const SoaQ8Matrix & tok_embd,
                                          const float * tok_norm,
                                          int32_t mask_token_id,
                                          uint16_t * mask_embed_out,
                                          hipStream_t stream);

// ---- m-row SoA Q8_0 GEMV (RDNA3/4 fdot2 + half2 LDS) -----------------------
// y[r * ystride + row] = sum_k W[row_offset + row, k] * x[r * xstride + k], m = 1..8.
// K must be a multiple of 256 (all DFlash 2 projections: 2304, 4096, 6912, 9216).
// mode 0 = static m; mode 2 = dynamic m from dp->ncommit (1..8).
void launch_dflash2_gemv(const SoaQ8Matrix & W, int row_offset, int rows,
                         const uint16_t * x_bf16, int xstride,
                         int m, float * y, int ystride, hipStream_t stream,
                         const DFlash2DraftParams * dp = nullptr, int mode = 0);

// ---- Commit (target taps -> draft sliding-window KV) -----------------------
// out[r] = bf16(RMSNorm(x[r], w, 1e-6)), x is f32 [n, 2304].
// mode 0 = static n; mode 1 = dynamic n from dp->ncommit.
void launch_dflash2_rmsnorm_f32_to_bf16(const float * x, int n, const uint16_t * w,
                                        uint16_t * out, hipStream_t stream,
                                        const DFlash2DraftParams * dp = nullptr,
                                        int mode = 0);

// Store committed context K/V into BF16 sliding-window ring (WINDOW = 2048).
// kv is f32 [n, 2048] = (k[1024] | v[1024]); K gets head-RMSNorm + NeoX RoPE (theta=1e4).
void launch_dflash2_ctx_kv_store(const float * kv, int n, int start,
                                 const uint16_t * k_norm,
                                 uint16_t * k_ctx, uint16_t * v_ctx,
                                 int32_t * ctx_pos, hipStream_t stream,
                                 const DFlash2DraftParams * dp = nullptr);

// ---- Draft backbone --------------------------------------------------------
// res[0] = bf16(RMSNorm(tok_embd[anchor], tok_norm)), res[1..7] = mask_embed;
// xn[r] = bf16(RMSNorm(res[r], in_norm)).
void launch_dflash2_embed_block(const SoaQ8Matrix & tok_embd, const float * tok_norm,
                                const uint16_t * mask_embed, const uint16_t * in_norm,
                                int anchor, uint16_t * res, uint16_t * xn,
                                hipStream_t stream,
                                const DFlash2DraftParams * dp = nullptr);

// Dynamic 2-tap grouped depthwise convolution (input side = 0).
// xn: bf16 [8, 2304], coeff: f32 [8, 576], base: bf16 [2, 2, 2304] -> out: bf16 [8, 2304].
void launch_dflash2_conv_in(const uint16_t * xn, const float * coeff,
                            const uint16_t * base_kernel, uint16_t * out,
                            hipStream_t stream);

// Fused dynamic 2-tap grouped depthwise convolution (output side = 1) + residual add + RMSNorm.
// sub_out: f32 [8, 2304], coeff: f32 [8, 576], base: bf16 [2, 2, 2304],
// res: bf16 [8, 2304] (updated in-place), norm_w: bf16 [2304] (nullable),
// xn_out: bf16 [8, 2304] (written when norm_w != nullptr).
void launch_dflash2_conv_out_add_rmsnorm(const float * sub_out, const float * coeff,
                                         const uint16_t * base_kernel, uint16_t * res,
                                         const uint16_t * norm_w, uint16_t * xn_out,
                                         hipStream_t stream);

// Final norm on block prediction rows 1..7 -> both BF16 [7, 2304] (selector hidden_proj)
// and FP32 [7, 2304] (target lm_head GEMV).
void launch_dflash2_final_norm_pred(const uint16_t * res, const uint16_t * norm_w,
                                    uint16_t * h_bf16, float * h_f32, hipStream_t stream);

// Block Q/K/V prep: qkv f32 [8, 6144] -> q_out bf16 [8, 32, 128] (q_norm + RoPE theta=1e4),
// k/v written to slots WINDOW..WINDOW+7 of k_ctx/v_ctx (k gets k_norm + RoPE theta=1e4).
void launch_dflash2_qkv_rope(const float * qkv, const uint16_t * q_norm,
                             const uint16_t * k_norm, int P, uint16_t * q_out,
                             uint16_t * k_ctx, uint16_t * v_ctx, hipStream_t stream,
                             const DFlash2DraftParams * dp = nullptr);

// Bidirectional block attention + sliding-window (2048) context attention.
// 32 Q heads, 8 KV heads (GQA=4), head_dim=128, scale=1/sqrt(128).
// out: bf16 [8, 4096].
void launch_dflash2_attention(const uint16_t * q, const uint16_t * k_ctx,
                              const uint16_t * v_ctx, const int32_t * ctx_pos,
                              int P, float * ws, uint16_t * out, hipStream_t stream,
                              const DFlash2DraftParams * dp = nullptr);
int64_t dflash2_attention_ws_floats();

// out = bf16(silu(gate) * up), gu f32 [8, 2*inter] -> out bf16 [8, inter].
void launch_dflash2_silu_mul(const float * gu, int rows, int inter, uint16_t * out,
                             hipStream_t stream);

// ---- Full-vocab Top-16 & Bilinear Candidate Path Selector ------------------
// Exact Top-16 per row over logits f32 [7, V] (V = VOCAB_UNPADDED = 200058).
// part_val / part_idx workspace: [7 * 128 * 16].
// Outputs: cand_val f32 [7, 16], cand_idx int32 [7, 16].
void launch_dflash2_topk16(const float * logits, int V,
                           float * part_val, int32_t * part_idx,
                           float * cand_val, int32_t * cand_idx,
                           hipStream_t stream);
int64_t dflash2_topk16_part_elems();

// Greedy bilinear Candidate Path Selector (rank=256, top_k=16, softcap=30 after correction).
// Computes tokens[7] and conf[7] = max softmax prob over the 16 scored candidates at each step.
void launch_dflash2_selector_walk(const SoaQ8Matrix & pred_cb,
                                  const SoaQ8Matrix & succ_cb,
                                  const float * sel_h,
                                  const float * cand_val,
                                  const int32_t * cand_idx,
                                  int anchor,
                                  int32_t * tokens_out,
                                  float * conf_out,
                                  hipStream_t stream,
                                  const DFlash2DraftParams * dp = nullptr);

} // namespace inkling
