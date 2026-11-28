#pragma once

// Native RDNA3/RDNA4 DFlash 2 speculative drafter for Inkling-2-Mini
// (`incoai/Inkling-2-Mini-DFlash2`), plus a unified SpecDrafter facade that
// auto-detects DFlash 2 vs DSpark from the safetensors header behind `--mtp`.
//
// Architecture (verified from incoai/Inkling-2-Mini-DFlash2):
//  * 4 target hidden taps [5, 23, 41, 48] (identical to InklingEngine::d_taps_)
//    concatenated [n, 9216] -> fc [2304, 9216] -> hidden_norm [2304];
//  * 5-layer Qwen3-style draft decoder (hidden=2304, inter=6912, 32 Q heads,
//    8 KV heads, head_dim=128, NeoX RoPE theta=10000, RMSNorm eps=1e-6);
//  * all 5 layers use sliding_window=2048 BF16 context KV rings + 8-row
//    bidirectional block attention (is_causal=false) -> O(1) VRAM (~42 MiB KV)
//    and O(1) draft latency at any context up to 256k+;
//  * 2-tap grouped dynamic depthwise convolutions (kernel=2, group_size=16,
//    num_groups=144) wrapping both attention and MLP sublayers;
//  * full-vocabulary target lm_head projection (VOCAB_UNPADDED=200058) via the
//    engine's existing fdot2 m-row SoA Q8_0 GEMV + exact 2-stage Top-16;
//  * low-rank bilinear Candidate Path Selector (rank=256, top_k=16) with
//    softcap=30*tanh(x/30) applied after correction (`after_correction`).
//
// All large projections and codebooks are repacked to 16-byte-aligned SoA Q8_0
// on GPU at load time (~550 MiB total steady-state VRAM; temporary staging freed).

#include "dspark_mtp.h"
#include "dflash2_mtp_kernels.h"

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace inkling {

class DFlash2Drafter {
public:
    static constexpr int BLOCK       = 8;      // anchor + 7 mask rows
    static constexpr int GAMMA       = BLOCK - 1;
    static constexpr int N_LAYERS    = 5;
    static constexpr int N_HEAD_D    = 32;
    static constexpr int N_KV_HEAD_D = 8;
    static constexpr int Q_DIM_D     = N_HEAD_D * HEAD_DIM;     // 4096
    static constexpr int KV_DIM_D    = N_KV_HEAD_D * HEAD_DIM;  // 1024
    static constexpr int QKV_DIM_D   = Q_DIM_D + 2 * KV_DIM_D;  // 6144
    static constexpr int INTER_D     = 6912;
    static constexpr int WINDOW      = 2048;
    static constexpr int CONV_GROUPS = N_EMBD / 16;             // 144
    static constexpr int CONV_PROJ   = 2 * 2 * CONV_GROUPS;     // 576
    static constexpr int SEL_RANK    = 256;
    static constexpr int SEL_TOP_K   = 16;
    static constexpr int MASK_TOK_ID = 200056;

    DFlash2Drafter() = default;
    ~DFlash2Drafter();

    // Auto-detection helper: returns true if `path` (file or directory containing
    // model.safetensors) is a DFlash 2 safetensors checkpoint.
    static bool is_dflash2_checkpoint(const std::string & path,
                                      std::string * resolved_safetensors = nullptr);

    // Accurate post-repack VRAM estimate for apply_vram_profile().
    static size_t estimate_vram_bytes();

    bool init(const std::string & mtp_path, const ModelWeights & w, int max_ctx,
              int vocab, hipStream_t stream, int draft_ctx_cap = 0);

    void reset();
    void commit(const uint16_t * d_feats, int n, int start);
    void note_tokens(const int32_t * ids, int n) { (void)ids; (void)n; }
    DSparkDraft draft(int32_t anchor, int P);

    int length() const { return length_; }
    int ctx() const { return WINDOW; }
    int vocab() const { return vocab_; }
    size_t vram_bytes() const { return vram_bytes_; }
    int short_rows() const { return vocab_; }

    struct Checkpoint {
        int length = 0;
        std::vector<uint8_t> layers_blob;
    };
    bool save_checkpoint(Checkpoint & out) const;
    bool restore_checkpoint(const Checkpoint & ckpt);

    // Selftest / debug helpers
    double debug_gemv_check();
    double debug_shortlist_check() { return 0.0; }
    const int32_t * debug_cand_idx() const { return cand_idx_; }
    const float   * debug_cand_val() const { return cand_val_; }
    void debug_base_copy(float * host_out) const;

private:
    struct Layer {
        const uint16_t * in_norm        = nullptr; // [2304]
        const uint16_t * post_norm      = nullptr; // [2304]
        const uint16_t * q_norm         = nullptr; // [128]
        const uint16_t * k_norm         = nullptr; // [128]
        const uint16_t * attn_conv_base = nullptr; // [2, 2, 2304]
        const uint16_t * mlp_conv_base  = nullptr; // [2, 2, 2304]
        SoaQ8Matrix      attn_conv_proj;           // [576, 2304]
        SoaQ8Matrix      mlp_conv_proj;            // [576, 2304]
        SoaQ8Matrix      qkv;                      // [6144, 2304] (k,v at rows 4096..6143)
        SoaQ8Matrix      wo;                       // [2304, 4096]
        SoaQ8Matrix      gate_up;                  // [13824, 2304]
        SoaQ8Matrix      w_down;                   // [2304, 6912]
        uint16_t       * k_ctx          = nullptr; // [(WINDOW + BLOCK), 8, 128] bf16
        uint16_t       * v_ctx          = nullptr; // [(WINDOW + BLOCK), 8, 128] bf16
        int32_t        * ctx_pos        = nullptr; // [WINDOW]
    };

    void run_commit_piece(const uint16_t * feats, int n, int start,
                          const DFlash2DraftParams * dp);
    void run_draft_body(const DFlash2DraftParams * dp, int anchor, int P);
    void capture_draft_graph();

    const ModelWeights * w_      = nullptr;
    hipStream_t          stream_ = nullptr;
    int                  vocab_  = 0;
    int                  length_ = 0;
    size_t               vram_bytes_ = 0;
    std::vector<void *>  allocs_;

    SoaQ8Matrix          fc_;                      // [2304, 9216]
    const uint16_t     * hidden_norm_ = nullptr;   // [2304]
    const uint16_t     * final_norm_  = nullptr;   // [2304]
    uint16_t           * mask_embed_  = nullptr;   // [2304] precomputed
    Layer                layers_[N_LAYERS];

    SoaQ8Matrix          sel_hidden_proj_;         // [256, 2304]
    SoaQ8Matrix          sel_pred_cb_;             // [vocab_, 256]
    SoaQ8Matrix          sel_succ_cb_;             // [vocab_, 256]

    // Per-call scratch
    uint16_t * res_        = nullptr; // [8, 2304]
    uint16_t * xn_         = nullptr; // [8, 2304]
    uint16_t * xconv_      = nullptr; // [8, 2304]
    float    * conv_coeff_ = nullptr; // [8, 576]
    float    * qkv_out_    = nullptr; // [8, 6144]
    uint16_t * q_          = nullptr; // [8, 32, 128]
    uint16_t * attn_o_     = nullptr; // [8, 4096]
    float    * delta_      = nullptr; // [8, 2304]
    float    * gu_out_     = nullptr; // [8, 2 * 6912]
    uint16_t * act_        = nullptr; // [8, 6912]
    uint16_t * h_final_bf_ = nullptr; // [7, 2304]
    float    * h_final_f32_= nullptr; // [7, 2304]
    float    * logits_     = nullptr; // [7, vocab_]
    float    * sel_h_      = nullptr; // [7, 256]
    float    * topk_pv_    = nullptr; // [7 * 128 * 16]
    int32_t  * topk_pi_    = nullptr; // [7 * 128 * 16]
    float    * cand_val_   = nullptr; // [7, 16]
    int32_t  * cand_idx_   = nullptr; // [7, 16]
    int32_t  * tokens_     = nullptr; // [7]
    float    * conf_       = nullptr; // [7]
    float    * attn_ws_    = nullptr; // split-K attention partials
    float    * ctx_proj_   = nullptr; // [8, 2304] commit fc output
    uint16_t * ctx_h_      = nullptr; // [8, 2304] commit hidden_norm output
    float    * kv_commit_  = nullptr; // [8, 2048] commit k|v output

    // Static HIP graphs
    DFlash2DraftParams * d_dp_              = nullptr;
    const uint16_t     * d_taps_base_       = nullptr;
    hipGraphExec_t       draft_graph_exec_  = nullptr;
    hipGraphExec_t       commit_graph_exec_ = nullptr;
};

// Unified speculative drafter facade: auto-detects DFlash 2 vs DSpark from the
// `--mtp` path without modifying DSparkDrafter.
class SpecDrafter {
public:
    enum Kind { NONE = 0, DSPARK = 1, DFLASH2 = 2 };

    struct Checkpoint {
        Kind kind = NONE;
        DSparkDrafter::Checkpoint  dspark;
        DFlash2Drafter::Checkpoint dflash2;
    };

    static size_t estimate_vram_bytes(const std::string & mtp_path);

    bool init(const std::string & mtp_path, const ModelWeights & w, int max_ctx,
              int vocab, hipStream_t stream,
              int draft_ctx_cap = DSparkDrafter::DEFAULT_DRAFT_CTX);

    void reset();
    void commit(const uint16_t * d_feats, int n, int start);
    void note_tokens(const int32_t * ids, int n);
    DSparkDraft draft(int32_t anchor, int P);

    Kind kind() const { return kind_; }
    const char * kind_name() const {
        return kind_ == DFLASH2 ? "dflash2" : (kind_ == DSPARK ? "dspark" : "none");
    }
    int length() const;
    int ctx() const;
    int vocab() const;
    size_t vram_bytes() const;
    int short_rows() const;
    int cand_k() const { return kind_ == DFLASH2 ? DFlash2Drafter::SEL_TOP_K : 1024; }

    bool save_checkpoint(Checkpoint & out) const;
    bool restore_checkpoint(const Checkpoint & ckpt);

    double debug_gemv_check();
    double debug_shortlist_check();
    const int32_t * debug_cand_idx() const;
    const float   * debug_cand_val() const;
    void debug_base_copy(float * host_out) const;

private:
    Kind kind_ = NONE;
    std::unique_ptr<DSparkDrafter>  dspark_;
    std::unique_ptr<DFlash2Drafter> dflash2_;
};

} // namespace inkling
