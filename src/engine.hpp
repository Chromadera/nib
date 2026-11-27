#pragma once

#include "inkling_engine.h"
#include "kernels_wmma_fp8.h"
#include "sparse_global.h"

#include <cstdint>
#include <cstring>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>

namespace inkling {

// Step 8A: widest speculative verify call (anchor + 7 drafts).
static constexpr int MAX_VERIFY_TOKENS = 8;

struct EngineConfig {
    int32_t max_ctx       = 262144; // 256k default (global KV cache slots)
    int32_t swa_cap       = SWA_WINDOW + MAX_UBATCH; // 1536 ring buffer for SWA layers
    int32_t n_slots       = 1;      // concurrent stream slots (1..MAX_CONCURRENT_SLOTS); server defaults to 4
    KVType  kv_type       = KV_Q8_0;
    bool    use_hip_graph = true;
    int32_t sparse_pages  = 0; // 0 = dense Global FA; >= SPARSE_MIN_PAGES = Quest block-sparse

    // Step 8A: buf_logits_ is [MAX_UBATCH, VOCAB] (391 MiB) only where every row's
    // logits are needed (ppl / oracle).  gen / bench / vision / selftest only ever
    // read the last (or a single) row plus the 8 verify rows, so 8 rows (6.1 MiB)
    // is enough and frees ~385 MiB of VRAM for 256k q8_0 + Vision + MTP.
    bool    need_all_logits_buf = false;

    // Step 8B: batched m-token verify (one weight/expert stream per matrix for all
    // m rows).  1.8x faster per row than the per-row path, but the engine's
    // n_tokens>1 mix/ctrl/norm kernel variants differ from the fused single-token
    // decode kernels by ~1 ulp, and this model amplifies that into real token
    // flips.  Default OFF: speculative decoding stays 100% greedy-exact.
    int32_t verify_batch  = 0;

    // Step 7F A/B switches (kernels_moe_v3). -1 = shipped kernel; 0 = MOE_V3_AUTO;
    // 1 = MOE_V3_ALT; 2 = MOE_V3_ALT2 (wo only). Defaults are the A/B winners.
    int32_t wo_v3         = -1; // launch_soa_q8_gemv_wo_v2 vs _wo_v3
    int32_t moe_gu_v3     = -1; // MoE stage 1 (gate_up + SITU)
    int32_t moe_dn_v3     = -1; // MoE stage 2 (down + weighted sum + sconv epilogue)
    int32_t moe_fused_v3  = -1; // -1 = two-launch; 0 = fused single launch

    // Step 7G/7K A/B switches (decode attention + QKV prep). -1 = shipped default.
    //   0 = v1 (launch_fa_relbias / launch_sparse_global_select_pages_decode /
    //           launch_fa_sparse_global_decode)
    //   1 = v2 (kernels_fa_decode_v2)
    //   2 = v3 (kernels_fa_decode_v3)
    //   3 = v4 (kernels_fa_decode_v4: low-VGPR / L3-aware)
    // For qkv_v3: 0 = v2 fused (kernels_decode_fused), 1 = v3 fused.
    int32_t fa_swa_v3     = -1;
    int32_t fa_select_v3  = -1;
    int32_t fa_global_v3  = -1;
    int32_t qkv_super_v3  = -1;
    // Step 7G A/B: split-count overrides (<=0 = use each kernel's default).
    int32_t fa_swa_splits  = 0;
    int32_t fa_glob_splits = 0;
    int32_t fa_sel_chunks  = 0;

    // Step 7K: Quest page-metadata format + the two fa_*_v4 policy knobs.
    //   sparse_meta : SPARSE_META_Q8 (258 B/page/head) | SPARSE_META_Q4 (130 B)
    //   fa_q_lds    : 1 = v4 stages Q / q+ / q- in LDS (frees 48 / 96 VGPRs)
    //   fa_cached   : 1 = v4 uses L1/L2-backed KV loads instead of `slc dlc`
    // Q4_MM halves the metadata DRAM (-32 MiB VRAM) and is ~+0.8 tok/s, but its
    // 15-level outward bound is ~18x looser than int8 and drops the 258k needle
    // page (measured), so the shipped default stays int8 (`--sparse-meta q4`).
    int32_t sparse_meta    = 0;
    int32_t fa_q_lds       = 0;   // register Q beats LDS Q (measured)
    int32_t fa_cached      = 0;   // `slc dlc` streaming beats L1/L2 caching (measured)

    // Step 7H A/B switches.  Each one replaces a shipped kernel with its Step 7H
    // counterpart; -1/0 keeps the shipped path so any knob can be measured alone.
    int32_t moe_ship_fused = 2;  // 0 = off, 1 = 288-block fused, 2 = 192-block fused (NVFP4)
    int32_t moe_q8_fused   = 0;  // 0 = off, 1 = fused Q8 MoE
    int32_t b_v3           = 0;  // Pass-0 pre-MoE: 0 = B_v2, 1 = B_v3 (128 blocks)
    int32_t c_v3           = 0;  // Pass-1/2 pre-MoE: 0 = C_v2, 1 = C_v3 (96 blocks)
    int32_t mix_fast       = 0;  // 0 = shipped mix_rms_norm, 1 = n_states-specialised
    int32_t qkv_grid       = 288; // Step 7H: QKV-prep grid (288 shipped / 384 A/B)

    // Step 7I-1 super-fusion switches (see include/kernels_step7i_fused.h).
    int32_t fuse_wob      = 0;  // wo + B: -24 nodes but ~2 tok/s SLOWER at d=0 (flag only)
    int32_t fuse_woc      = 1;  // wo + C (Pass-1/2 pre-MoE)    -> 1 node, 192x288
    // Not bit-exact on this model (<=1 ulp in the distributed mix, amplified by
    // the MoE near-ties: PPL 3.6815 vs 3.5788), so flag-only.
    int32_t fuse_mix_qkv  = 0;  // Pass-0 attn norm + qkv prep  -> 1 node, 192x192
    // 0 = shipped A + QKV (fastest, bit-exact); 1 = standalone A_v3;
    // 2 = super-fused A+QKV.  Measured: A_v3 is ~4.6 tok/s SLOWER at d=0.
    int32_t fuse_a_qkv    = 0;
    int32_t fuse_minmax   = 0;  // fold page_minmax into the fused K-store epilogue

    // Step 7J: RDNA3 v_dot2_f32_f16 + __half2 LDS weight-stream kernels
    // (see include/kernels_step7j_fdot2.h).  0 = shipped kernel in every case.
    // Measured on GPU 0 (PPL gate <= 3.600, baseline 3.5788):
    //   lm_head   fdot2 : 553.2 -> 548.8 us/call, PPL 3.5788  -> DEFAULT ON
    //   wo/wo_C   fdot2 : -29 us/tok,           PPL 3.6036  -> flag only (gate)
    //   NVFP4 MoE fdot2 : +1 us/call,           PPL 3.6020  -> flag only (gate)
    //   QKV       fdot2 : +0.5 us/call,         PPL 3.5960  -> flag only (slower)
    //   Q8 MoE    fdot2 : +0.6 us/call,         PPL 3.5790  -> flag only (slower)
    // The four losers are DRAM-bound (84-95% of the practical roof once the
    // per-block x_norm reload is counted), so halving their LDS/ALU does not
    // move them; the two residual-feeding FP16-x variants also shift PPL.
    int32_t fdot2_moe     = 0;  // 1 = fused 192, 2 = fused 288, 3 = fused 384, 4 = split
    int32_t fdot2_qkv     = 0;  // 1 = 288x256, 2 = 384x256, 3 = 288x128
    int32_t fdot2_wo      = 0;  // 1 = wo_fdot2 + wo_C_fdot2, 2 = wo_fdot2 + unfused C
    int32_t fdot2_wo_shape = 0; // 0 = 6w LPR32 (384 blk), 1 = 8w LPR32, 2 = 4w LPR16, 3 = 4w LPR32
    int32_t fdot2_lm      = 1;  // 1 = lm_head fdot2 (8 warps LPR 8)
    int32_t fdot2_q8moe   = 0;  // 1 = fused Q8 MoE fdot2

    // --cpu-embd: keep token_embd.weight (~469 MiB) in pinned GPU-mapped host RAM
    // instead of VRAM.  Off by default (and never auto-enabled on >= 22 GiB GPUs).
    bool    cpu_embd      = false;
    // Host-side tight-VRAM profile was applied (main.cpp): only switches on the
    // estimated-vs-actual state-bytes report at the end of init().
    bool    report_state_estimate = false;

    // RDNA4 FP8 WMMA prefill (src/kernels_wmma_fp8_rdna4.hip).  FP8_PREFILL_MOE runs the NVFP4
    // expert GEMMs of every prefill micro-batch with more than fp8_min_m tokens on FP8 WMMA (E4M3
    // operands, FP32 accumulation); everything else stays on FP16 WMMA.  Default off.  RDNA4
    // only: on another GPU, or in a binary built without a gfx12 target, init() warns once and
    // forces it off.
    int32_t fp8_prefill    = FP8_PREFILL_OFF;
    int32_t fp8_min_m      = FP8_PREFILL_DEFAULT_MIN_M;
    // Runtime prefill micro-batch size (512 default; up to FP8_MAX_UBATCH=1024).
    // When raised to 1024, init() automatically ensures swa_cap >= SWA_WINDOW + prefill_ubatch.
    int32_t prefill_ubatch = MAX_UBATCH;
};

class InklingEngine {
public:
    InklingEngine() = default;
    ~InklingEngine();

    bool init(const std::string & model_path, const EngineConfig & cfg, bool verify_repack = false);
    void reset_state(); // resets pos=0, clears KV k_pos to -1 and sconv hist to 0

    // Prefill a batch of tokens (chunked automatically by cfg_.prefill_ubatch).
    // If compute_all_logits == true, copies [n_tokens, VOCAB_UNPADDED] logits into out_logits (for PPL).
    // If compute_all_logits == false, computes logits ONLY for the last token (writes [VOCAB_UNPADDED] if out_logits != nullptr).
    bool prefill(const std::vector<int32_t> & tokens, bool compute_all_logits, std::vector<float> * out_logits = nullptr);

    // Prefill raw FP32 device embeddings [n_tokens, N_EMBD] (e.g. VisionEncoder output).
    // IMPORTANT: bypasses tok_norm, matching llama.cpp/src/models/inkling2.cpp!
    bool prefill_embeds(const float * d_embeds, int n_tokens, bool compute_all_logits = false, std::vector<float> * out_logits = nullptr);

    // Fast prefill benchmark helper (advances state by n_tokens using dummy token IDs, returns elapsed ms)
    double benchmark_prefill_chunk(const std::vector<int32_t> & tokens);

    // Single-token decode step (uses static captured HIP Graph when enabled!).
    // Returns argmax token_id; optionally copies full logits [VOCAB_UNPADDED] to host.
    int32_t decode_step(int32_t token_id, std::vector<float> * out_logits = nullptr);

    // Fast decode loop benchmark (stays on GPU except 4-byte argmax or StepMeta update)
    double benchmark_decode_steps(int32_t start_token, int n_steps, std::vector<int32_t> * out_tokens = nullptr);

    int32_t current_pos()    const { return cur_pos_; }
    int32_t max_ctx()        const { return cfg_.max_ctx; }
    int32_t prefill_ubatch() const { return cfg_.prefill_ubatch; }
    int32_t num_slots()      const { return cfg_.n_slots; }
    int32_t slot_ctx()       const { return slot_ctx_; }
    int32_t slot_pos(int slot) const {
        return (slot >= 0 && slot < cfg_.n_slots) ? slot_pos_[slot] : 0;
    }
    int32_t slot_span(int slot) const {
        return (slot >= 0 && slot < cfg_.n_slots) ? slot_span_[slot] : 0;
    }
    int32_t slot_ctx_limit(int slot, int span) const;
    int32_t slot_ctx_limit(int slot) const {
        return slot_ctx_limit(slot, slot_span(slot));
    }
    void rewind_pos_for_bench(int32_t target_pos);
    const TokenizerData & tokenizer() const { return tok_; }
    const ModelWeights  & weights()   const { return weights_; }
    hipStream_t           stream()    const { return stream_; }
    size_t total_vram_bytes() const { return total_vram_bytes_; }
    // --cpu-embd: bytes of token_embd held in pinned, GPU-mapped host RAM (not in total_vram_bytes()).
    size_t host_pinned_bytes() const { return weights_.host_pinned_bytes; }
    bool   cpu_embd() const { return weights_.tok_embd_on_host; }

    // FP8 WMMA prefill (EngineConfig::fp8_prefill).  set_fp8_prefill() switches it at runtime
    // (selftest / A-B measurements); it returns false, leaving FP8 off, when this binary or GPU
    // cannot run the FP8 kernels.
    bool fp8_prefill_active() const { return fp8_moe_prefill_; }
    bool set_fp8_prefill(int32_t mode);

    // Host-only mirror of alloc_state_and_scratch() (+ alloc_batch_scratch() and
    // enable_mtp_io() when with_mtp_io): VRAM bytes the engine allocates on top of
    // the weights for `cfg`.  Used by the tight-VRAM auto profile to size -c
    // before anything is loaded.
    static size_t estimate_state_bytes(const EngineConfig & cfg, bool with_mtp_io);

    // ========================================================================
    // Multi-slot continuous batching (zero-regression on single-stream slot 0)
    // ========================================================================
    void reset_slot(int slot, int span = 1);
    bool expand_slot_span(int slot, int new_span);
    bool shrink_slot_span(int slot, int new_span);
    bool prefill_slot_chunk(int slot, const int32_t * tokens, int n_tokens,
                            bool want_logits, std::vector<float> * out_logits = nullptr);
    bool prefill_embeds_slot_chunk(int slot, const float * d_embeds, int n_tokens,
                                   bool want_logits, std::vector<float> * out_logits = nullptr);
    int32_t decode_slot_single(int slot, int32_t token_id, std::vector<float> * out_logits = nullptr);
    // Runs ONE batched decode step across `batch_size` (2..n_slots) independent slots.
    // If out_logits_buf != nullptr, writes [batch_size, VOCAB_UNPADDED] softcapped logits.
    // Always writes per-row argmaxes into out_argmax[0..batch_size-1] if non-null.
    bool decode_batch(const int32_t * slots, const int32_t * token_ids, int batch_size,
                      int32_t * out_argmax, float * out_logits_buf = nullptr);

    // Host-RAM snapshot of slot 0's mutable non-Global state (40 SWA rings +
    // 48 short-conv histories) at a prompt boundary. Zero extra VRAM.
    // Global KV [0..pos) is immutable during subsequent slot-0 decode, so
    // restore rewinds Global k_pos to `pos`, recomputes the partial Quest
    // boundary page (if sparse_pages > 0), and uploads SWA + short-conv.
    struct Slot0Checkpoint {
        int32_t pos = 0;
        int32_t swa_cap = 0;
        KVType  kv_type = KV_Q8_0;
        std::vector<uint8_t> swa_blob;   // packed SWA slots: k_qs, k_d, v_qs, v_d, k_pos
        std::vector<float>   sconv_blob; // packed sconv_slots_[0..N_LAYER_SLOTS-1].hist
    };
    bool save_slot0_checkpoint(Slot0Checkpoint & out) const;
    bool restore_slot0_checkpoint(const Slot0Checkpoint & ckpt);

    // ========================================================================
    // Step 8A: DSpark MTP speculative decoding support
    // ========================================================================
    // Allocates the 4-tap BF16 buffer ([MAX_UBATCH, 4, N_EMBD] = 9.0 MiB) plus the
    // verify save/rollback state.  Only called when --mtp is given: the non-MTP
    // path never allocates anything and its launches are bit-identical.
    bool enable_mtp_io();

    // Called by prefill()/prefill_embeds() after every chunk with that chunk's
    // taps (bf16 [n, 4, N_EMBD] on the device) and, for prefill(), its token ids
    // (host pointer, nullptr for raw-embedding chunks).  Stream-ordered: the
    // consumer must launch on stream() so it reads the taps before the next
    // chunk overwrites them.
    void set_tap_hook(std::function<void(const uint16_t *, const int32_t *, int, int)> hook);
    void clear_tap_hook();

    // Speculative verify of `m` (1..8) tokens at pos_start: saves the per-slot
    // short-conv state, runs `m` exact single-token decode rows (same kernels as
    // the non-MTP path, so each row's greedy argmax is bit-identical to a plain
    // decode step), captures the 4 tap vectors per row and the raw pre-conv
    // inputs needed by rollback, then computes the m softcapped argmaxes into
    // target_ids().  cur_pos_ becomes pos_start + m.
    bool verify_step(const int32_t * tokens, int m, int pos_start);

    // Drops rows [commit, m): rebuilds every slot's short-conv history to the
    // state after `commit` rows, invalidates the rejected Global KV rows and
    // updates the Quest page metadata for exactly the committed tokens.
    // cur_pos_ becomes pos_start + commit.
    void verify_rollback(int pos_start, int commit, int m);

    // Debug: host copy of one slot's short-conv history (selftest gate B diagnostics).
    void debug_dump_hist(int slot, std::vector<float> & out);

    // Host copy of the first m verify rows' softcapped logits (selftest gate).
    void verify_logits_to_host(int m, std::vector<float> & out) const;
    // Host copy of a single verify row's softcapped logits (rare high-temp top_k=0 fallback).
    void verify_row_logits_to_host(int row, std::vector<float> & out) const;
    // Fast exact GPU top-256 + full-vocab (max_l, sum_exp) for lossless temp>0 MTP verify.
    bool verify_topk_to_host(int m, float inv_t,
                             float * out_vals, int32_t * out_ids,
                             float * out_max_l, float * out_sum_exp) const;
    // Issue #4: exact full-vocab GPU sampling of the first m (<= 8) softcapped
    // logit rows in buf_logits_ (the m verify rows, or row 0 after a decode
    // step).  Row r draws from p(x) ~ exp((l_x - max) / T) restricted to
    // { x : p(x) >= min_p * p_max } \ { excl[r] } (excl[r] < 0: no exclusion)
    // and reports Z and e_excl so the caller can form p(excl) = e_excl / Z.
    // See launch_mtp_sample_full.  Only 8 ints + 32 floats cross the bus.
    bool sample_rows_full_to_host(int m, float inv_t, float min_p,
                                  const int32_t * excl, uint64_t seed,
                                  int32_t * out_tok, float * out_stats) const;
    // Heterogeneous batched decode variant: samples `n` (1..MAX_CONCURRENT_SLOTS)
    // selected rows in buf_logits_, each with its own (row_idx, inv_t, min_p, seed),
    // in a single GPU launch + sync.  Uses salt=0 so seeded output is batch-invariant.
    bool sample_batch_rows_full_to_host(int n, const int32_t * row_idx,
                                        const float * inv_t, const float * min_p,
                                        const uint64_t * seeds,
                                        int32_t * out_tok, float * out_stats) const;

    const uint16_t * taps()       const { return d_taps_; }
    const int32_t  * target_ids() const { return d_target_; }   // device [MAX_VERIFY_TOKENS]
    const int32_t  * host_target_ids() const { return h_target_pinned_; }
    bool mtp_io_enabled() const { return mtp_io_; }

private:
    void alloc_state_and_scratch();
    void alloc_batch_scratch();
    KVCacheSlot          slot_kv_view(int slot_id, int layer_slot) const;
    SparseGlobalPageMeta slot_sparse_meta_view(int slot_id, int layer_slot) const;
    float *              slot_sconv_hist(int slot_id, int layer_slot) const;
    void record_forward_pass(int n_tokens, int pos_start, const StepMeta * d_meta, bool compute_all_logits, const float * d_raw_embeds = nullptr,
                             int row_slot = 0, bool skip_head = false);
    // Step 8B: m-token batched verify (m in {2,4,8}); see verify_step.
    void record_verify_pass(int m);
    void capture_verify_graph(int idx, int m);
    void record_layer(
        int slot, int ip,
        const float * cur_in, float * cur_out,
        std::vector<const float *> & blocks,
        const float * messages,
        const LoopLayerWeights * loop,
        int n_tokens, int pos_start, const StepMeta * d_meta,
        bool attn_already_mixed = false,
        const float * const * mix_states0 = nullptr, int n_mix_states0 = 0,
        const float * const * mix_states1 = nullptr, int n_mix_states1 = 0,
        int row_slot = 0);
    void capture_decode_graph();

    EngineConfig  cfg_;
    bool          fp8_moe_prefill_ = false; // resolved EngineConfig::fp8_prefill == FP8_PREFILL_MOE
    ModelWeights  weights_;
    TokenizerData tok_;
    hipStream_t   stream_ = nullptr;

    // Persistent per-slot KV and short-conv states (48 virtual slots)
    KVCacheSlot    kv_slots_[N_LAYER_SLOTS];
    SConvStateSlot sconv_slots_[N_LAYER_SLOTS];

    // Per-Global-slot Quest block-sparse page metadata + shared selection scratch
    // (allocated ONLY when cfg_.sparse_pages > 0, so the dense path uses 0 extra VRAM)
    SparseGlobalPageMeta sparse_meta_[N_LAYER_SLOTS];
    SparseGlobalScratch  sparse_scratch_{};

    // Device-side decode StepMeta & argmax output
    StepMeta * d_meta_       = nullptr;
    int32_t  * d_argmax_id_  = nullptr;
    int32_t  * d_token_ids_  = nullptr; // [MAX_UBATCH]

    // Dedicated non-aliasing residual/block buffers ([MAX_UBATCH, N_EMBD] each)
    float * buf_a_           = nullptr;
    float * buf_b_           = nullptr;
    float * buf_ffn_inp_     = nullptr;
    float * buf_blk0_        = nullptr; // inpSA at il=0
    float * buf_prelude_     = nullptr; // out after il=5
    float * buf_pass_b6_[3]  = {};      // inpSA at ip=6 for pass 0,1,2
    float * buf_pass_b12_[3] = {};      // inpSA at ip=12 for pass 0,1,2
    float * buf_pass_out_[2] = {};      // out after ip=17 for pass 0,1 (bank_out)
    float * buf_b18_         = nullptr; // inpSA at il=18

    // Layer workspace scratch buffers
    float * buf_mixed_       = nullptr; // [4 * MAX_UBATCH * N_EMBD] (max n_q = 4)
    float * buf_messages_    = nullptr; // [4 * MAX_UBATCH * N_EMBD]
    float * buf_ctrl_[3]     = {};      // [MAX_UBATCH * N_EMBD] each
    float * buf_ctrl_gate_   = nullptr; // [3 * MAX_UBATCH * LOOP_GATE_RANK]
    float * buf_norm_[3]     = {};      // [MAX_UBATCH * N_EMBD] each
    float * buf_proj_qkv_    = nullptr; // [MAX_UBATCH * 8600]
    float * buf_proj_k_      = nullptr; // [MAX_UBATCH * 1056]
    float * buf_proj_v_      = nullptr; // [MAX_UBATCH * 1056]
    float * buf_q_prep_      = nullptr; // [MAX_UBATCH * N_HEAD * HEAD_DIM]
    float * buf_rel_table_   = nullptr; // [MAX_UBATCH * N_HEAD * N_REL_EXTENT]
    float * buf_fa_out_      = nullptr; // [MAX_UBATCH * N_Q]
    float * buf_wo_out_      = nullptr; // [MAX_UBATCH * N_EMBD]
    float * buf_x_norm_      = nullptr; // [MAX_UBATCH * N_EMBD]
    int32_t * buf_topk_ids_  = nullptr; // [MAX_UBATCH * TOP_K]
    float * buf_topk_w_      = nullptr; // [MAX_UBATCH * TOP_K]
    float * buf_router_scratch_ = nullptr; // [512] floats (zeroed once; self-resetting atomic counters)
    float * buf_mix_scratch_ = nullptr;    // [16384] floats (Step 7I-1 partials/barriers; zeroed once)
    float * buf_moe_act_dec_ = nullptr; // [TOP_K * N_FF_EXP]
    float * buf_moe_out_     = nullptr; // [MAX_UBATCH * N_EMBD]
    float * buf_fa_tmp_o_    = nullptr; // [max(64, 4*MAX_UBATCH) * N_HEAD * HEAD_DIM]
    float2 * buf_fa_tmp_m_   = nullptr; // [max(64, 4*MAX_UBATCH) * N_HEAD]
    int32_t * buf_fa_done_   = nullptr; // [64] zeroed once; self-resetting atomic counter for decode FA combine
    float * buf_logits_      = nullptr; // [MAX_UBATCH * VOCAB_UNPADDED]

    MoePrefillScratch moe_scratch_;

    // Pinned (hipHostMalloc) host staging: zero pageable staging stalls on the decode hot path.
    StepMeta * h_meta_pinned_     = nullptr;
    int32_t  * h_argmax_pinned_   = nullptr;
    int32_t  * h_tok_hist_pinned_ = nullptr;
    // Device copy of the token-history written by the in-graph meta-advance epilogue.
    int32_t  * d_tok_hist_        = nullptr;
    int32_t    tok_hist_cap_      = 0;
    // Host-built exact tau(pos) table (bit-identical to make_step_meta) for device-side advance.
    float    * d_tau_tab_         = nullptr;
    int32_t    tau_tab_cap_       = 0;

    // Single static captured HIP Graph for decode (0 -> 256k without recapture!)
    hipGraph_t     decode_graph_      = nullptr;
    hipGraphExec_t decode_graph_exec_ = nullptr;
    bool           graph_captured_    = false;

    int32_t cur_pos_ = 0;
    std::vector<void *> state_allocs_;
    size_t total_vram_bytes_ = 0;

    // ---- Step 8B: batched verify scratch + static graphs ----
    int         verify_m_    = 0;     // >0 while record_verify_pass records
    int32_t   * d_verify_unique_ = nullptr;  // [64]
    uint32_t  * d_verify_mask_   = nullptr;  // [64]
    int32_t   * d_verify_slot_   = nullptr;  // [64 * 8]
    float     * d_verify_w_      = nullptr;  // [64 * 8]
    int32_t   * d_verify_cnt_    = nullptr;  // [64]  active tokens per unique expert
    int32_t   * d_verify_list_   = nullptr;  // [64 * 8] token ids of each expert's actives
    int32_t   * d_num_unique_    = nullptr;  // [1]
    float     * buf_moe_slot_    = nullptr;  // [MAX_VERIFY_TOKENS * TOP_K * N_EMBD]
    float     * buf_verify_act_  = nullptr;  // [MAX_VERIFY_TOKENS * TOP_K * N_FF_EXP]
    // Step 8B: token-major mix/ctrl buffers.  The decode's fused A/B/C kernels are
    // single-token (out[q] is contiguous), so the batched verify calls them per row
    // and stores row r's slices at [r][q][N_EMBD]; the m-row GEMVs then use a
    // 3*N_EMBD row stride.  This keeps every mix/norm/ctrl/router bit-identical to
    // a plain decode step (the engine's n_tokens>1 variants are 1-ulp different,
    // which this model amplifies into real token flips).
    float     * buf_verify_mixed_ = nullptr; // [MAX_VERIFY_TOKENS][3][N_EMBD]
    float     * buf_verify_msg_   = nullptr; // [MAX_VERIFY_TOKENS][4][N_EMBD]
    float     * buf_verify_gate_  = nullptr; // [3][LOOP_GATE_RANK]
    hipGraphExec_t verify_graph_exec_[MAX_VERIFY_TOKENS] = {}; // m = 1..8 (idx = m - 1)
    bool           verify_graph_ready_[MAX_VERIFY_TOKENS] = {};

    // ---- Step 8A: MTP tap capture / speculative verify state ----
    bool        mtp_io_      = false;
    bool        verify_mode_ = false; // while true record_layer dumps raw pre-conv inputs
    bool        graph_capturing_ = false;
    int         verify_row_  = 0;
    uint16_t  * d_taps_      = nullptr; // [MAX_UBATCH, 4, N_EMBD] bf16
    float     * d_verify_raw_ = nullptr; // [N_LAYER_SLOTS, MAX_VERIFY_TOKENS, N_SCONV_CHANNELS]
    float     * d_hist_bak_  = nullptr; // [N_LAYER_SLOTS, N_SCONV_CHANNELS * N_CONV_HIST]
    StepMeta  * d_verify_meta_ = nullptr; // [MAX_VERIFY_TOKENS]
    StepMeta  * h_verify_meta_ = nullptr; // pinned [MAX_VERIFY_TOKENS]
    int32_t   * d_target_    = nullptr; // [MAX_VERIFY_TOKENS]
    int32_t   * h_target_pinned_ = nullptr; // pinned [MAX_VERIFY_TOKENS]
    float     * d_verify_topk_vals_ = nullptr; // [MAX_VERIFY_TOKENS * 256]
    int32_t   * d_verify_topk_ids_  = nullptr; // [MAX_VERIFY_TOKENS * 256]
    float     * d_verify_stats_     = nullptr; // [MAX_VERIFY_TOKENS * 2] (max_l, sum_exp)
    float     * h_verify_topk_vals_ = nullptr; // pinned [MAX_VERIFY_TOKENS * 256]
    int32_t   * h_verify_topk_ids_  = nullptr; // pinned [MAX_VERIFY_TOKENS * 256]
    float     * h_verify_stats_     = nullptr; // pinned [MAX_VERIFY_TOKENS * 2]
    float     * d_sample_scratch_   = nullptr; // [MTP_SAMPLE_SCRATCH_WORDS] (issue #4 GPU sampler)
    int32_t   * d_sample_tok_       = nullptr; // [MAX_VERIFY_TOKENS]
    float     * d_sample_stats_     = nullptr; // [MAX_VERIFY_TOKENS * 4]
    int32_t   * h_sample_tok_       = nullptr; // pinned [MAX_VERIFY_TOKENS]
    float     * h_sample_stats_     = nullptr; // pinned [MAX_VERIFY_TOKENS * 4]
    float    ** d_hist_ptrs_ = nullptr;  // [N_LAYER_SLOTS] device array of sconv hist pointers
    bool        mtp_batched_head_ = true; // one m-row lm_head GEMV instead of m launches
    std::function<void(const uint16_t *, const int32_t *, int, int)> tap_hook_;

    // ---- Multi-slot continuous batching state ----
    int32_t        slot_ctx_ = 0;
    int32_t        slot_pos_[MAX_CONCURRENT_SLOTS] = {};
    int32_t        slot_span_[MAX_CONCURRENT_SLOTS] = {};
    // Extra per-slot SWA rings & short-conv histories for slots 1..n_slots-1
    // (slot 0 aliases kv_slots_ and sconv_slots_ directly, zero overhead).
    KVCacheSlot    slot_swa_kv_[MAX_CONCURRENT_SLOTS][N_LAYER_SLOTS] = {};
    SConvStateSlot slot_sconv_[MAX_CONCURRENT_SLOTS][N_LAYER_SLOTS] = {};
    bool           batch_scratch_ready_ = false;
    int            active_slot_         = 0;     // active slot for single-slot prefill/decode
    bool           batch_slots_active_  = false; // true while decode_batch records m independent slots
    int32_t        batch_slot_ids_[MAX_CONCURRENT_SLOTS] = {};
    struct BatchGraphEntry {
        hipGraphExec_t exec = nullptr;
        uint64_t       last_used = 0;
    };
    static constexpr size_t MAX_BATCH_GRAPHS = 64;
    uint64_t       batch_graph_lru_seq_ = 0;
    std::unordered_map<uint32_t, BatchGraphEntry> batch_graphs_;
};

} // namespace inkling
