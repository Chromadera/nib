#pragma once

// Step 8A: Native RDNA3 DSpark MTP drafter for Inkling-2-Mini.
//
// Ports the reference DSpark drafter (ingine_ref/native/src/model/drafter.cpp +
// kernels/draft/*) to this engine's data structures:
//
//  * mtp.safetensors (54 BF16 tensors) parsed by a tiny header-only reader;
//  * the 3 draft layers' qkv / wo / gate_up / down / k|v_ctx projections repacked
//    on the GPU into this engine's 16-byte-aligned SoA Q8_0 layout (~278 MiB,
//    half the BF16 bytes) and driven by one m-row (<= 8) SoA Q8 GEMV that reads
//    every weight once for the whole drafted block;
//  * the shortlist head = `short_K_` ranked/special token rows of the target's
//    lm_head (SoA Q8_0 gather) + up to CONTEXT_ROWS dynamic context tokens,
//    plus fold_w/fold_b (the Markov head's hidden_out folded into the
//    unembedding) computed in FP64 on the GPU;
//  * draft context caches: layers 0 and 2 keep a BF16 sliding window
//    (WINDOW = 1024) + the 8 block rows, layer 1 a Q8_0 ring of
//    ctx = min(max_ctx, 65536) slots (exact same per-32-element format as the
//    engine's KV cache);
//  * exact DSpark math: NeoX RoPE (theta 1e6), RMSNorm eps 1e-6, the BF16
//    rounding points, bidirectional 8-row block attention, the GRU-like Markov
//    cell and the softcapped (30 * tanh(x/30)) candidate logits.
//
// Only *draft quality* depends on this module — the target verify is what makes
// generation exact — so the projections use Q8_0 weights (the reference uses
// FP8 rows), which keeps drafts well inside the reference's acceptance band.

#include "inkling_engine.h"
#include "dspark_mtp_kernels.h"

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace inkling {

struct DSparkDraft {
    const int32_t * tokens = nullptr;  // device [GAMMA]: drafted tokens
    const float   * conf   = nullptr;  // device [GAMMA]: max softmax probability per step
};

class DSparkDrafter {
public:
    static constexpr int BLOCK     = 8;      // drafted block: anchor + 7 mask rows
    static constexpr int GAMMA     = BLOCK - 1;
    static constexpr int WINDOW    = 1024;   // local layers' context ring
    static constexpr int DRAFT_CTX = 65536;  // global layer's maximum ring slots
    static constexpr int DEFAULT_DRAFT_CTX = 8192; // Step 8D: fast 256k drafting (1.2 ms vs 3.6 ms)
    static constexpr int RANK      = 256;    // Markov head state width
    static constexpr int MAX_CTX_ROWS = 16384;

    DSparkDrafter() = default;
    ~DSparkDrafter();

    // mtp_path: models/mtp.safetensors.  `w` supplies the target's tok_embd /
    // tok_norm (draft input embedding) and lm_head (shortlist head + fold), and
    // must outlive the drafter.  max_ctx and draft_ctx_cap bound the global ring.
    bool init(const std::string & mtp_path, const ModelWeights & w, int max_ctx,
              int vocab, hipStream_t stream, int draft_ctx_cap = DEFAULT_DRAFT_CTX);

    // Empty draft context (rings invalidated, length 0, no context tokens).
    void reset();

    // Committed target rows: taps bf16 [n, 4, N_EMBD] on the device at positions
    // start..start+n-1 become draft context.  n <= 8 replays the small commit;
    // larger n (prefill chunks) runs in pieces of 8.  Stream-ordered.
    void commit(const uint16_t * d_feats, int n, int start);

    // Context tokens join the shortlist head (up to MAX_CTX_ROWS until reset()).
    // Stream-ordered; `ids` is a host array of n token ids.
    void note_tokens(const int32_t * ids, int n);

    // Drafts for positions P+1..P+7 after the anchor token at P (greedy).
    DSparkDraft draft(int32_t anchor, int P);

    int length() const { return length_; }
    int ctx() const { return ctx_; }
    int vocab() const { return vocab_; }
    size_t vram_bytes() const { return vram_bytes_; }
    int short_rows() const { return short_K_ + short_E_; }

private:
    struct Layer {
        const uint16_t * in_norm = nullptr;
        const uint16_t * post_norm = nullptr;
        const uint16_t * ctx_norm = nullptr;
        const uint16_t * q_norm = nullptr;
        const uint16_t * k_norm = nullptr;
        const uint16_t * k_norm_ctx = nullptr;
        SoaQ8Matrix qkv;      // [5120, 2304]
        SoaQ8Matrix wo;       // [2304, 3072]
        SoaQ8Matrix gate_up;  // [18432, 2304]
        SoaQ8Matrix w_down;   // [2304, 9216]
        SoaQ8Matrix wkv_ctx;  // [2048, 2304] = k_proj_ctx | v_proj_ctx
        int8_t  * k_ctx = nullptr; // local: bf16 [WINDOW+BLOCK, 8, 128]; global: q8 qs [ctx, 8, 128]
        __half  * d_ctx = nullptr; // global only: [ctx, 8, 4]
        int8_t  * v_ctx = nullptr;
        __half  * vd_ctx = nullptr;
        int32_t * ctx_pos = nullptr; // local rings only
        bool      local = true;
    };

    void run_small_commit();
    void run_commit_piece(const uint16_t * feats, int n, int row0, bool last,
                          const DSparkDraftParams * dp);
    void run_layers();
    void run_draft_body(const DSparkDraftParams * dp, int anchor, int P);
    void capture_draft_graphs();
    void set_context_rows(int rows);

    const ModelWeights * w_ = nullptr;
    hipStream_t stream_ = nullptr;
    int ctx_ = 0;
    int vocab_ = 0;
    int inter_ = 0;          // 9216
    int length_ = 0;
    size_t vram_bytes_ = 0;
    std::vector<void *> allocs_;

    Layer layers_[3];
    const uint16_t * alpha_ = nullptr;     // [4, 4] bf16 softmax(fuse_weight)
    const uint16_t * final_norm_ = nullptr;
    const uint16_t * mask_embed_ = nullptr;
    const uint16_t * f_norm_ = nullptr;
    const uint16_t * w1_ = nullptr;        // markov_w1 [V, 256]
    const uint16_t * rnn_w_ = nullptr;     // [768, 5120]
    const uint16_t * rnn_b_ = nullptr;     // [768]
    uint16_t * fold_w_ = nullptr;          // [V, 256]
    uint16_t * fold_b_ = nullptr;          // [V]

    // Shortlist head: rows [0, short_K_) = ranked tokens + specials (ascending
    // ids), then context tokens.  short_head_ is SoA Q8_0 [rows, 2304].
    SoaQ8Matrix short_head_;
    int32_t * short_map_ = nullptr;
    int32_t * short_ids_ = nullptr;        // staging of ids to gather
    int short_K_ = 0;
    int short_E_ = 0;
    std::vector<uint8_t>  short_in_;       // host: token has a row
    std::vector<int32_t>  short_extra_;    // host: context tokens, in row order
    std::vector<int32_t>  short_base_list_; // host: the short_K_ static token ids

    // per-call scratch
    uint16_t * res_ = nullptr;      // [BLOCK, H]
    uint16_t * xn_ = nullptr;       // [BLOCK, H]
    uint16_t * q_ = nullptr;        // [BLOCK, 24, 128]
    uint16_t * attn_o_ = nullptr;   // [BLOCK, 3072]
    float    * delta_ = nullptr;    // [BLOCK, H]
    uint16_t * act_ = nullptr;      // [BLOCK, inter]
    float    * qkv_out_ = nullptr;  // [BLOCK, 5120]
    float    * gu_out_ = nullptr;   // [BLOCK, 2*inter]
    uint16_t * f_rnn_ = nullptr;    // [H]
    uint16_t * xin_ = nullptr;      // [8, H] bf16 (commit mix)
    float    * kv_commit_ = nullptr;// [8, 2048]
    float    * base_ = nullptr;     // [GAMMA, short rows]
    float    * cand_val_ = nullptr; // [GAMMA, 1024]
    int32_t  * cand_idx_ = nullptr; // [GAMMA, 1024]
    float    * cand_logits_ = nullptr;
    uint16_t * state_ = nullptr;    // [GAMMA+1, RANK]
    uint16_t * latent_ = nullptr;   // [RANK]
    uint64_t * picks_ = nullptr;    // [GAMMA]
    float    * part_ = nullptr;     // [GAMMA, 768]
    float    * attn_ws_ = nullptr;  // split-K attention partials
    float    * topk_ws_ = nullptr;  // radix-select workspace
    int32_t  * tokens_ = nullptr;   // [GAMMA]
    float    * conf_ = nullptr;     // [GAMMA]

    // Step 8B: static HIP graphs (one draft graph per nkeys bucket + one commit)
    DSparkDraftParams * h_dp_ = nullptr;  // pinned host staging
    DSparkDraftParams * d_dp_ = nullptr;  // device copy read by every kernel
    const uint16_t    * d_taps_base_ = nullptr; // commit() source (engine tap buffer)
    std::vector<hipGraphExec_t> draft_execs_;
    std::vector<int>            draft_buckets_; // nkeys cap of each draft graph
    hipGraphExec_t              commit_graph_exec_ = nullptr;

    // lightweight per-call statistics (printed by the CLI)
    struct {
        double commit_ms = 0.0;
        double draft_ms = 0.0;
    } stats_;
public:
    // Selftest: differential check of the m-row SoA Q8 GEMV against the engine's
    // single-row GEMV on the same matrix (returns the max |difference|).
    double debug_gemv_check();
    double debug_shortlist_check();
    const int32_t * debug_cand_idx() const { return cand_idx_; }
    const float * debug_cand_val() const { return cand_val_; }
    void debug_base_copy(float * host_out) const;

    const void * stats() const { return &stats_; }
    double last_draft_ms() const { return stats_.draft_ms; }
    double last_commit_ms() const { return stats_.commit_ms; }
};

} // namespace inkling
