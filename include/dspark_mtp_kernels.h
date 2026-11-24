#pragma once

// Internal kernel API of the Step 8A DSpark MTP drafter (src/kernels_dspark_mtp.hip).
// Every function is a plain raw-pointer launcher on `stream`; none of them
// allocates or synchronises.

#include "inkling_engine.h"

#include <cstdint>

namespace inkling {

// ---- Step 8B: per-call dynamic values for the captured draft/commit graphs --
// Every kernel of draft()/commit() that used to take a host scalar now reads it
// from this 32-byte device struct, so the whole pipeline can be a static
// hipGraphExec_t: one tiny pinned H2D copy before each graph launch is the only
// per-call host work.
struct DSparkDraftParams {
    int32_t anchor;       // draft(): anchor token id (the embed row)
    int32_t pos;          // draft(): P (anchor position); commit(): start position
    int32_t nkeys_global; // draft(): min(P + BLOCK, ctx_)
    int32_t head_rows;    // draft(): short_K_ + short_E_ (shortlist head rows)
    int32_t ncommit;      // commit(): rows 1..8
    int32_t _pad[3];
};

// Stream-ordered by-value upload of DSparkDraftParams (zero host-buffer race
// when commit() and draft() are queued back-to-back without a host sync).
void launch_dspark_set_params(DSparkDraftParams * d_dp, DSparkDraftParams val,
                              hipStream_t stream);

// ---- load time -------------------------------------------------------------
// BF16 [rows, K] -> this engine's SoA Q8_0 (d = amax/127, round to nearest).
void launch_dspark_repack_q8(const uint16_t * bf16, int rows, int K,
                             int8_t * qs, __half * d, hipStream_t stream);

// Gather n rows of `src` (SoA Q8_0) into dst rows [dst_row0, dst_row0+n) and
// set map[i] = ids[i].
void launch_dspark_gather_q8(const SoaQ8Matrix & src, const int32_t * ids, int n,
                             int8_t * dst_qs, __half * dst_d, int dst_row0,
                             int32_t * map, hipStream_t stream);

// fold_w = bf16(U @ hw), fold_b = bf16(U @ hb) computed in FP64; U = the
// lm_head rows dequantized to bf16(float(d) * float(qs)) (the engine's own Q8_0
// inverse of the SoA GEMV).
void launch_dspark_fold_head(const SoaQ8Matrix & lm, const uint16_t * hw,
                             const uint16_t * hb, int rows,
                             uint16_t * fold_w, uint16_t * fold_b, hipStream_t stream);

// ---- m-row SoA Q8 GEMV -----------------------------------------------------
// y[r * ystride + row] = sum_k W[row, k] * x[r * xstride + k],  m = 1..8.
// x is bf16 when x_bf16 is true (draft activations), else f32 (commit rows).
// Step 8B: the bf16 path stages the m x rows in LDS (FP32, exact) and streams
// every weight block ONCE for all m rows.  `dyn_rows` (optional) overrides W.rows
// with dp->head_rows so the shortlist-head launch is graph-static.
void launch_dspark_gemv(const SoaQ8Matrix & W, const void * x_rows, int xstride,
                        bool x_bf16, int m, float * y, int ystride, hipStream_t stream,
                        const DSparkDraftParams * dp = nullptr, int mode = 0,
                        int rows_static = 0);

// ---- draft block -----------------------------------------------------------
// out[r] = bf16(rms(sum_j feats[r,j] * alpha[j]) * w), feats bf16 [n, 4, H].
// Step 8B: `dp` selects commit-mode row counts: mode 0 = the static `n`,
// mode 1 = dp->ncommit rows, mode 2 = the single row dp->ncommit - 1.
void launch_dspark_mix_rmsnorm(const uint16_t * feats, int n, const uint16_t * alpha,
                               const uint16_t * w, uint16_t * out, hipStream_t stream,
                               const DSparkDraftParams * dp = nullptr, int mode = 0);

// res[0] = bf16(rms(embed[anchor]) * embed_norm), res[1..7] = mask_embed,
// xn[r] = bf16(rms(res[r]) * in_norm); also clears the GAMMA picks.
void launch_dspark_embed_block(const SoaQ8Matrix & tok_embd, const float * embed_norm,
                               const uint16_t * mask_embed, const uint16_t * in_norm,
                               int anchor, uint16_t * res, uint16_t * xn,
                               uint64_t * picks, hipStream_t stream,
                               const DSparkDraftParams * dp = nullptr);

// Commit of context keys/values: kv f32 [n, 2048] = (c Wk^T | c Wv^T).  Row r is
// at position start + r; k = bf16(rope(bf16(rms_head(k) * k_norm))), v as is.
// local rings keep BF16 rows at slot pos % WINDOW and tag ctx_pos; the global
// ring keeps Q8_0 rows at slot pos % size.
void launch_dspark_ctx_kv_store(const float * kv, int n, int start, int size,
                                const uint16_t * k_norm, bool local,
                                int8_t * k_qs, __half * k_d, int8_t * v_qs, __half * v_d,
                                int32_t * ctx_pos, hipStream_t stream,
                                const DSparkDraftParams * dp = nullptr);

// Block rows' q/k/v: q -> bf16 [8, 24, 128] (RMSNorm+RoPE at P+i), keys/values
// into the cache (local: slots WINDOW+i; global: slot (P+i) % size).
void launch_dspark_qkv_rope(const float * qkv, const uint16_t * q_norm,
                            const uint16_t * k_norm, int P, uint16_t * q_out,
                            bool local, int size,
                            int8_t * k_qs, __half * k_d, int8_t * v_qs, __half * v_d,
                            hipStream_t stream, const DSparkDraftParams * dp = nullptr);

// Bidirectional 8-row block attention (scale 1/sqrt(128), GQA h -> h/3).
// out bf16 [8, 3072].  ws: layout from dspark_attention_ws_floats().
// Step 8B: 16-key LDS-staged K/V tiles, coalesced int4 loads (bit-exact math).
void launch_dspark_attention(const uint16_t * q, int nkeys, bool local, int size,
                             const int8_t * k_qs, const __half * k_d,
                             const int8_t * v_qs, const __half * v_d,
                             const int32_t * ctx_pos, int P, float * ws,
                             uint16_t * out, hipStream_t stream,
                             const DSparkDraftParams * dp = nullptr,
                             int n_chunks_static = 0, int chunk_static = 0);
int64_t dspark_attention_ws_floats(int nkeys);
int dspark_attention_chunks(int nkeys);
int dspark_attention_chunk_len(int nkeys);

// res = bf16(res + delta); xn = bf16(rms(res) * w).
void launch_dspark_add_rmsnorm(uint16_t * res, const float * delta, const uint16_t * w,
                               uint16_t * xn, int rows, hipStream_t stream);

// out = bf16(silu(gate) * up), gu f32 [rows, 2*inter].
void launch_dspark_silu_mul(const float * gu, int rows, int inter, uint16_t * out,
                            hipStream_t stream);

// fill
void launch_dspark_fill(float * x, int64_t n, float v, hipStream_t stream);

// ---- Markov head -----------------------------------------------------------
// Top-1024 candidates per row of base f32 [rows, V], deterministic order:
// elements above the threshold by ascending index (32 slices, slice-major),
// then the ties needed by ascending index.  map maps the row -> token id.
void launch_dspark_topk_candidates(const float * base, int rows, int V,
                                   const int32_t * map, float * cand_val,
                                   int32_t * cand_idx, float * ws, hipStream_t stream,
                                   const DSparkDraftParams * dp = nullptr, int stride = 0);

// part[i, r] = sum_{c >= 512} rnn_w[r, c] z_i[c], z_i[512:] = (h[i], f_rnn).
void launch_dspark_markov_pre(const uint16_t * rnn_w, const uint16_t * h,
                              const uint16_t * f_rnn, int gamma, float * part,
                              hipStream_t stream);

// Step i's GRU-like cell -> state[i+1], latent.
void launch_dspark_markov_step(int i, const uint16_t * rnn_w, const uint16_t * rnn_b,
                               const uint16_t * w1, const float * part,
                               const uint64_t * picks, const int32_t * cand_idx,
                               int anchor, uint16_t * state, uint16_t * latent,
                               hipStream_t stream, const DSparkDraftParams * dp = nullptr);

// Step i's candidate logits (softcap 30 * tanh(x / 30)) and the greedy pick.
void launch_dspark_markov_logits(int i, const uint16_t * fold_w, const uint16_t * fold_b,
                                 const float * cand_val, const int32_t * cand_idx,
                                 const uint16_t * latent, int gamma,
                                 float * cand_logits, uint64_t * picks, hipStream_t stream);

// Emits tokens[7] and conf[7] = max softmax probability per row.
void launch_dspark_markov_finish(const int32_t * cand_idx, const float * cand_logits,
                                 const uint64_t * picks, int32_t * tokens, float * conf,
                                 hipStream_t stream);

} // namespace inkling
