#include "engine.hpp"
#include "kernels_decode_fused.h"
#include "kernels_fa_decode_v2.h"
#include "kernels_fa_decode_v3.h"
#include "kernels_fa_decode_v4.h"
#include "kernels_moe_v3.h"
#include "kernels_step7h_fused.h"
#include "kernels_step7i_fused.h"
#include "kernels_step7j_fdot2.h"
#include "kernels_mtp_verify.h"
#include "kernels_verify_batch.h"

// Step 7F: per-stage launchers of the shipped MoE decode (src/kernels_gemv_soa.hip),
// split out so each stage can be A/B benchmarked against its kernels_moe_v3
// counterpart in isolation.  They are the exact old two launches.
namespace inkling {
void launch_moe_nvfp4_gate_up_ship(
    const SoaNVFP4Experts & gate_up,
    const float * x_norm, const int32_t * topk_ids, float * act,
    hipStream_t stream);
void launch_moe_nvfp4_down_ship(
    const SoaNVFP4Experts & down,
    const float * act, const int32_t * topk_ids, const float * topk_w,
    float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * raw_out = nullptr);
void launch_moe_q8_gate_up_ship(
    const SoaQ8Experts & gate_up,
    const float * x_norm, const int32_t * topk_ids, float * act,
    hipStream_t stream);
void launch_moe_q8_down_ship(
    const SoaQ8Experts & down,
    const float * act, const int32_t * topk_ids, const float * topk_w,
    float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * raw_out = nullptr);
} // namespace inkling

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// Step 7D super-fused decode kernels: per-group switches kept so each group can
// be A/B benchmarked in isolation against the split pipeline.
namespace {
// B/C measured 2026-11-23: x_norm/out_gate differ from the split pipeline by
// <=1 ulp (FP reassociation + C's 32x72 vs 72x32 ssq partition), topk_ids are
// bit-IDENTICAL, and the 510-token PPL is 3.6139 vs 3.6159 without them (i.e.
// within the model's own FP-sensitivity band: see the note at the end of this
// block).  -72 graph nodes, -1.26 ms/32 tok of wall time.
constexpr bool FUSE_B   = true;  // B: Pass-0 pre-MoE fusion (-24 nodes)
constexpr bool FUSE_C   = true;  // C: Pass-1/2 pre-MoE fusion (-48 nodes)
constexpr bool FUSE_WO  = true;   // F: 384-block wo GEMV + sconv epilogue
constexpr bool FUSE_MOE = false; // G+H: NVFP4 MoE v2 (old split pipeline wins A/B)
constexpr bool FUSE_E   = true;  // E: QKV GEMV + qkvr_prep
constexpr bool FUSE_A   = true;  // A: dual mix + ctrl QKV3 + norm
// NOTE ON "1 ULP" SENSITIVITY (measured, not theorised): this model amplifies
// FP32 reassociation aggressively.  Changing ONLY FA_V2_SWA_SPLITS from 12 to 24
// (identical math, different split boundaries) moves the 512-token decode logits
// by 9.86 and the 510-token PPL from 3.5672 (v1 FA) to 3.6139 (v2 FA).  So a
// 1-ulp kernel difference CAN move logits by >1 and PPL by ~0.05; it is not a
// bug signal by itself.  Quality is therefore gated on determinism + eval32
// top-5 + the 258k needle, not on bit-exact PPL parity.
} // namespace

#define HIP_CHECK(expr)                                                                 \
    do {                                                                                \
        hipError_t _err = (expr);                                                       \
        if (_err != hipSuccess) {                                                       \
            std::fprintf(stderr, "HIP error %d (%s) at %s:%d: %s\n",                    \
                         (int)_err, hipGetErrorString(_err), __FILE__, __LINE__, #expr);\
            std::abort();                                                               \
        }                                                                               \
    } while (0)

namespace inkling {

InklingEngine::~InklingEngine() {
    if (decode_graph_exec_) {
        (void)hipGraphExecDestroy(decode_graph_exec_);
    }
    for (int i = 0; i < MAX_VERIFY_TOKENS; ++i) {
        if (verify_graph_exec_[i]) {
            (void)hipGraphExecDestroy(verify_graph_exec_[i]);
            verify_graph_exec_[i] = nullptr;
        }
    }
    for (auto & kv : batch_graphs_) {
        if (kv.second.exec) {
            (void)hipGraphExecDestroy(kv.second.exec);
        }
    }
    batch_graphs_.clear();
    if (decode_graph_) {
        (void)hipGraphDestroy(decode_graph_);
    }
    if (cfg_.sparse_pages > 0) {
        for (int s = 0; s < N_LAYER_SLOTS; ++s) {
            sparse_global_page_meta_free(sparse_meta_[s]);
        }
        sparse_global_scratch_free(sparse_scratch_);
    }
    if (h_meta_pinned_)     { (void)hipHostFree(h_meta_pinned_); }
    if (h_argmax_pinned_)   { (void)hipHostFree(h_argmax_pinned_); }
    if (h_tok_hist_pinned_) { (void)hipHostFree(h_tok_hist_pinned_); }
    if (h_target_pinned_)   { (void)hipHostFree(h_target_pinned_); }
    if (h_verify_meta_)     { (void)hipHostFree(h_verify_meta_); }
    if (h_verify_topk_vals_){ (void)hipHostFree(h_verify_topk_vals_); }
    if (h_verify_topk_ids_) { (void)hipHostFree(h_verify_topk_ids_); }
    if (h_verify_stats_)    { (void)hipHostFree(h_verify_stats_); }
    if (h_sample_tok_)      { (void)hipHostFree(h_sample_tok_); }
    if (h_sample_stats_)    { (void)hipHostFree(h_sample_stats_); }
    h_sample_tok_ = nullptr;
    h_sample_stats_ = nullptr;
    h_verify_meta_ = nullptr;
    h_verify_topk_vals_ = nullptr;
    h_verify_topk_ids_ = nullptr;
    h_verify_stats_ = nullptr;
    h_meta_pinned_ = nullptr;
    h_argmax_pinned_ = nullptr;
    h_tok_hist_pinned_ = nullptr;
    h_target_pinned_ = nullptr;
    for (void * p : state_allocs_) {
        if (p) {
            (void)hipFree(p);
        }
    }
    state_allocs_.clear();
    weights_.destroy();
    if (stream_) {
        (void)hipStreamDestroy(stream_);
    }
}

template <typename T>
static T * alloc_gpu(size_t count, std::vector<void *> & track, size_t & total_bytes,
                     hipStream_t stream) {
    void * ptr = nullptr;
    size_t bytes = count * sizeof(T);
    // Round up to 256-byte alignment for clean coalescing
    size_t alloc_bytes = (bytes + 255) & ~size_t(255);
    HIP_CHECK(hipMalloc(&ptr, alloc_bytes));
    // Issue #1: never hand out uninitialised VRAM.  hipMalloc does not zero
    // (whether the pages come back clean depends on the kernel driver and on
    // what this process freed before, e.g. the GGUF repack staging buffers),
    // and several decode kernels read slots they then weight by exactly 0
    // (partial 8-key FA groups, padded tiles).  0 * NaN = NaN, so a garbage
    // fp16 KV scale turned into NaN logits -> argmax INT32_MAX -> an
    // out-of-bounds embedding gather on the next step (crash / GPU hang).
    // The fill is stream-ordered on the engine stream (every consumer runs on
    // it): a legacy-null-stream hipMemset here would instantiate an extra HW
    // queue for the rest of the process and measurably slows prefill (-3.5%).
    HIP_CHECK(hipMemsetAsync(ptr, 0, alloc_bytes, stream));
    track.push_back(ptr);
    total_bytes += alloc_bytes;
    return reinterpret_cast<T *>(ptr);
}

bool InklingEngine::init(const std::string & model_path, const EngineConfig & cfg, bool verify_repack) {
    cfg_ = cfg;
    if (cfg_.n_slots < 1) cfg_.n_slots = 1;
    if (cfg_.n_slots > MAX_CONCURRENT_SLOTS) cfg_.n_slots = MAX_CONCURRENT_SLOTS;
    const int32_t pool_ctx = (cfg_.sparse_pages > 0)
        ? std::min<int32_t>(cfg_.max_ctx, SPARSE_MAX_PAGES * SPARSE_PAGE_SIZE)
        : cfg_.max_ctx;
    slot_ctx_ = (cfg_.n_slots == 1)
        ? cfg_.max_ctx
        : (((pool_ctx / cfg_.n_slots) / SPARSE_PAGE_SIZE) * SPARSE_PAGE_SIZE);
    if (slot_ctx_ < SPARSE_PAGE_SIZE * 4) {
        std::fprintf(stderr, "InklingEngine: max_ctx=%d too small for n_slots=%d\n",
                     cfg_.max_ctx, cfg_.n_slots);
        return false;
    }
    HIP_CHECK(hipSetDevice(0));
    HIP_CHECK(hipStreamCreate(&stream_));

    // Quest block-sparse Global attention: all pages that may ever be scored are
    // capped at SPARSE_MAX_PAGES; the trailing SPARSE_RECENT_PAGES pages are always
    // selected from raw KV (no min/max metadata), which sets a hard 263,680-token
    // limit for the sparse path.
    if (cfg_.sparse_pages > 0) {
        const int64_t max_pages_needed = ((int64_t)cfg_.max_ctx + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
        if (max_pages_needed > (int64_t)SPARSE_MAX_PAGES + SPARSE_RECENT_PAGES) {
            std::fprintf(stderr,
                "InklingEngine: --sparse-pages supports max_ctx <= %d "
                "(requested %d = %lld pages > %d scored + %d recent)\n",
                (SPARSE_MAX_PAGES + SPARSE_RECENT_PAGES) * SPARSE_PAGE_SIZE,
                cfg_.max_ctx, (long long)max_pages_needed,
                SPARSE_MAX_PAGES, SPARSE_RECENT_PAGES);
            return false;
        }
    }

    if (!load_inkling_gguf(model_path, weights_, tok_, verify_repack)) {
        std::fprintf(stderr, "InklingEngine: failed to load GGUF '%s'\n", model_path.c_str());
        return false;
    }
    total_vram_bytes_ = weights_.total_vram_bytes;

    alloc_state_and_scratch();
    reset_state();

    // Step 7F: the fused MoE kernels cache a host occupancy query; run it now,
    // outside stream capture (it is not capture-safe).
    if (cfg_.moe_fused_v3 >= 0) {
        moe_v3_prime_occupancy();
    }
    // Step 7H: no runtime occupancy query is needed (all grids are statically
    // co-resident), but keep the hook next to the Step 7F one.
    step7h_prime_occupancy();
    // Step 7I-1: cache the blocks/CU of every fused grid before graph capture
    // (the fused launchers fall back when a grid cannot be fully co-resident).
    step7i_prime_occupancy();
    step7j_prime_occupancy();

    const char * kv_name = (cfg_.kv_type == KV_Q4_1) ? "q4_1"
                         : (cfg_.kv_type == KV_Q4_0) ? "q4_0" : "q8_0";
    std::printf("[InklingEngine] Ready! max_ctx=%d slots=%d slot_ctx=%d swa_cap=%d kv=%s sparse_pages=%d total_vram=%.2f GiB (weights=%.2f GiB, kv+scratch=%.2f GiB)\n",
                cfg_.max_ctx, cfg_.n_slots, slot_ctx_, cfg_.swa_cap, kv_name, cfg_.sparse_pages,
                (double)total_vram_bytes_ / (1024.0 * 1024.0 * 1024.0),
                (double)weights_.total_vram_bytes / (1024.0 * 1024.0 * 1024.0),
                (double)(total_vram_bytes_ - weights_.total_vram_bytes) / (1024.0 * 1024.0 * 1024.0));
    std::fflush(stdout);
    // Step 7K: the fa_*_v4 defaults reorder the split-K combine by <= 1 ulp and
    // this model amplifies that into a MoE top-8 near-tie, so the 258k needle
    // needs the wider Quest budget (measured: exact at 112 pages for q4_1 AND
    // q8_0, truncated at 96 -- the needle spans two pages and the second one
    // falls out of K = 70).  Only a note; the run stays bit-deterministic.
    if (cfg_.sparse_pages > 0 && cfg_.sparse_pages < 112 &&
        (cfg_.fa_swa_v3 < 0 || cfg_.fa_swa_v3 == 3)) {
        std::fprintf(stderr,
            "[step7k] note: --sparse-pages %d with FA v4 passes PPL/eval32/vision but not the\n"
            "         258k needle; use --sparse-pages 112 (or --fa-swa v1 for the old default).\n",
            cfg_.sparse_pages);
    }
    return true;
}

void InklingEngine::alloc_state_and_scratch() {
    // 1. Allocate KV cache & short-conv recurrent state for all 48 virtual slots (slot 0)
    const size_t qs_bytes_per_head = (cfg_.kv_type == KV_Q8_0) ? (size_t)HEAD_DIM : (size_t)(HEAD_DIM / 2);
    const size_t d_halves_per_head = (cfg_.kv_type == KV_Q4_1) ? (size_t)(HEAD_DIM / 16) : (size_t)(HEAD_DIM / 32);
    for (int s = 0; s < N_LAYER_SLOTS; ++s) {
        const bool swa = is_swa_slot(s);
        const int32_t cap = swa ? cfg_.swa_cap : cfg_.max_ctx;
        auto & kv = kv_slots_[s];
        kv.cap     = cap;
        kv.kv_type = (int32_t)cfg_.kv_type;
        kv.is_swa  = swa;
        kv.k_qs    = alloc_gpu<int8_t>((size_t)cap * N_HEAD_KV * qs_bytes_per_head, state_allocs_, total_vram_bytes_, stream_);
        kv.k_d     = alloc_gpu<__half>((size_t)cap * N_HEAD_KV * d_halves_per_head, state_allocs_, total_vram_bytes_, stream_);
        kv.v_qs    = alloc_gpu<int8_t>((size_t)cap * N_HEAD_KV * qs_bytes_per_head, state_allocs_, total_vram_bytes_, stream_);
        kv.v_d     = alloc_gpu<__half>((size_t)cap * N_HEAD_KV * d_halves_per_head, state_allocs_, total_vram_bytes_, stream_);
        kv.k_pos   = alloc_gpu<int32_t>((size_t)cap,                                state_allocs_, total_vram_bytes_, stream_);

        sconv_slots_[s].hist = alloc_gpu<float>((size_t)N_SCONV_CHANNELS * N_CONV_HIST, state_allocs_, total_vram_bytes_, stream_);

        slot_swa_kv_[0][s] = kv;
        slot_sconv_[0][s]  = sconv_slots_[s];
    }

    // 1b. Extra per-slot SWA rings & short-conv states for concurrent slots 1..n_slots-1.
    //     Global KV is partitioned zero-copy inside slot 0's allocation (0 extra Global KV VRAM!).
    for (int c = 1; c < cfg_.n_slots; ++c) {
        for (int s = 0; s < N_LAYER_SLOTS; ++s) {
            if (is_swa_slot(s)) {
                auto & kv = slot_swa_kv_[c][s];
                kv.cap     = cfg_.swa_cap;
                kv.kv_type = (int32_t)cfg_.kv_type;
                kv.is_swa  = true;
                kv.k_qs    = alloc_gpu<int8_t>((size_t)cfg_.swa_cap * N_HEAD_KV * qs_bytes_per_head, state_allocs_, total_vram_bytes_, stream_);
                kv.k_d     = alloc_gpu<__half>((size_t)cfg_.swa_cap * N_HEAD_KV * d_halves_per_head, state_allocs_, total_vram_bytes_, stream_);
                kv.v_qs    = alloc_gpu<int8_t>((size_t)cfg_.swa_cap * N_HEAD_KV * qs_bytes_per_head, state_allocs_, total_vram_bytes_, stream_);
                kv.v_d     = alloc_gpu<__half>((size_t)cfg_.swa_cap * N_HEAD_KV * d_halves_per_head, state_allocs_, total_vram_bytes_, stream_);
                kv.k_pos   = alloc_gpu<int32_t>((size_t)cfg_.swa_cap,                                state_allocs_, total_vram_bytes_, stream_);
            }
            slot_sconv_[c][s].hist = alloc_gpu<float>((size_t)N_SCONV_CHANNELS * N_CONV_HIST, state_allocs_, total_vram_bytes_, stream_);
        }
    }

    // 2. Device metadata & token IDs
    d_meta_      = alloc_gpu<StepMeta>(1,          state_allocs_, total_vram_bytes_, stream_);
    d_argmax_id_ = alloc_gpu<int32_t>(1,           state_allocs_, total_vram_bytes_, stream_);
    d_token_ids_ = alloc_gpu<int32_t>(MAX_UBATCH,  state_allocs_, total_vram_bytes_, stream_);

    // 3. Dedicated non-aliasing residual stream buffers ([MAX_UBATCH, N_EMBD])
    const size_t embd_buf = (size_t)MAX_UBATCH * N_EMBD;
    buf_a_       = alloc_gpu<float>(embd_buf, state_allocs_, total_vram_bytes_, stream_);
    buf_b_       = alloc_gpu<float>(embd_buf, state_allocs_, total_vram_bytes_, stream_);
    buf_ffn_inp_ = alloc_gpu<float>(embd_buf, state_allocs_, total_vram_bytes_, stream_);
    buf_blk0_    = alloc_gpu<float>(embd_buf, state_allocs_, total_vram_bytes_, stream_);
    buf_prelude_ = alloc_gpu<float>(embd_buf, state_allocs_, total_vram_bytes_, stream_);
    for (int p = 0; p < 3; ++p) {
        buf_pass_b6_[p]  = alloc_gpu<float>(embd_buf, state_allocs_, total_vram_bytes_, stream_);
        buf_pass_b12_[p] = alloc_gpu<float>(embd_buf, state_allocs_, total_vram_bytes_, stream_);
    }
    for (int p = 0; p < 2; ++p) {
        buf_pass_out_[p] = alloc_gpu<float>(embd_buf, state_allocs_, total_vram_bytes_, stream_);
    }
    buf_b18_ = alloc_gpu<float>(embd_buf, state_allocs_, total_vram_bytes_, stream_);

    // 4. Layer workspace scratch
    buf_mixed_       = alloc_gpu<float>(4 * embd_buf,                       state_allocs_, total_vram_bytes_, stream_);
    buf_messages_    = alloc_gpu<float>(4 * embd_buf,                       state_allocs_, total_vram_bytes_, stream_);
    for (int j = 0; j < 3; ++j) {
        buf_ctrl_[j] = alloc_gpu<float>(embd_buf,                           state_allocs_, total_vram_bytes_, stream_);
        buf_norm_[j] = alloc_gpu<float>(embd_buf,                           state_allocs_, total_vram_bytes_, stream_);
    }
    buf_ctrl_gate_   = alloc_gpu<float>(3 * (size_t)MAX_UBATCH * LOOP_GATE_RANK, state_allocs_, total_vram_bytes_, stream_);
    buf_proj_qkv_    = alloc_gpu<float>((size_t)MAX_UBATCH * 8600,          state_allocs_, total_vram_bytes_, stream_);
    buf_proj_k_      = alloc_gpu<float>((size_t)MAX_UBATCH * 1056,          state_allocs_, total_vram_bytes_, stream_);
    buf_proj_v_      = alloc_gpu<float>((size_t)MAX_UBATCH * 1056,          state_allocs_, total_vram_bytes_, stream_);
    buf_q_prep_      = alloc_gpu<float>((size_t)MAX_UBATCH * N_Q,           state_allocs_, total_vram_bytes_, stream_);
    buf_rel_table_   = alloc_gpu<float>((size_t)MAX_UBATCH * N_HEAD * N_REL_EXTENT, state_allocs_, total_vram_bytes_, stream_);
    buf_fa_out_      = alloc_gpu<float>((size_t)MAX_UBATCH * N_Q,           state_allocs_, total_vram_bytes_, stream_);
    buf_wo_out_      = alloc_gpu<float>(embd_buf,                           state_allocs_, total_vram_bytes_, stream_);
    buf_x_norm_      = alloc_gpu<float>(embd_buf,                           state_allocs_, total_vram_bytes_, stream_);
    buf_topk_ids_    = alloc_gpu<int32_t>((size_t)MAX_UBATCH * TOP_K,       state_allocs_, total_vram_bytes_, stream_);
    buf_topk_w_      = alloc_gpu<float>((size_t)MAX_UBATCH * TOP_K,         state_allocs_, total_vram_bytes_, stream_);
    // 512 floats per verify row (Step 8C runs the A/B/C kernels with grid.y = row,
    // each row owning its own self-resetting barrier sector).
    buf_router_scratch_ = alloc_gpu<float>((size_t)MAX_VERIFY_TOKENS * 512,  state_allocs_, total_vram_bytes_, stream_);
    buf_mix_scratch_    = alloc_gpu<float>(16384,                           state_allocs_, total_vram_bytes_, stream_);
    buf_moe_act_dec_ = alloc_gpu<float>((size_t)TOP_K * N_FF_EXP,           state_allocs_, total_vram_bytes_, stream_);
    buf_moe_out_     = alloc_gpu<float>(embd_buf,                           state_allocs_, total_vram_bytes_, stream_);
    const size_t fa_tmp_rows = (size_t)std::max(64, 8 * MAX_UBATCH);
    buf_fa_tmp_o_    = alloc_gpu<float>(fa_tmp_rows * N_HEAD * HEAD_DIM,    state_allocs_, total_vram_bytes_, stream_);
    buf_fa_tmp_m_    = alloc_gpu<float2>(fa_tmp_rows * N_HEAD,              state_allocs_, total_vram_bytes_, stream_);
    buf_fa_done_     = alloc_gpu<int32_t>(128,                              state_allocs_, total_vram_bytes_, stream_);
    // Step 8A: only ppl / oracle read every row's logits.  gen / bench / vision /
    // selftest read the last row (prefill) or a single decode row plus the 8
    // speculative verify rows, so 8 * VOCAB_UNPADDED (6.1 MiB) instead of
    // MAX_UBATCH * VOCAB_UNPADDED (391 MiB) frees ~385 MiB of VRAM.
    const int logits_rows = cfg_.need_all_logits_buf ? MAX_UBATCH : MAX_VERIFY_TOKENS;
    buf_logits_      = alloc_gpu<float>((size_t)logits_rows * VOCAB_UNPADDED, state_allocs_, total_vram_bytes_, stream_);

    // 5. Grouped MoE prefill scratchpad
    const size_t max_slots = (size_t)MAX_UBATCH * TOP_K;
    moe_scratch_.exp_counts  = alloc_gpu<int32_t>(N_EXPERT,                 state_allocs_, total_vram_bytes_, stream_);
    moe_scratch_.exp_offsets = alloc_gpu<int32_t>(N_EXPERT + 1,             state_allocs_, total_vram_bytes_, stream_);
    moe_scratch_.exp_cursors = alloc_gpu<int32_t>(N_EXPERT,                 state_allocs_, total_vram_bytes_, stream_);
    moe_scratch_.perm_tok    = alloc_gpu<int32_t>(max_slots,                state_allocs_, total_vram_bytes_, stream_);
    moe_scratch_.perm_w      = alloc_gpu<float>(max_slots,                  state_allocs_, total_vram_bytes_, stream_);
    moe_scratch_.perm_x      = alloc_gpu<float>(max_slots * N_EMBD,         state_allocs_, total_vram_bytes_, stream_);
    moe_scratch_.perm_act    = alloc_gpu<float>(max_slots * N_FF_EXP,       state_allocs_, total_vram_bytes_, stream_);
    moe_scratch_.perm_down   = alloc_gpu<float>(max_slots * N_EMBD,         state_allocs_, total_vram_bytes_, stream_);
    moe_scratch_.tile_exp    = alloc_gpu<int32_t>(max_slots + N_EXPERT,     state_allocs_, total_vram_bytes_, stream_);
    moe_scratch_.tile_m0     = alloc_gpu<int32_t>(max_slots + N_EXPERT,     state_allocs_, total_vram_bytes_, stream_);
    moe_scratch_.d_num_tiles = alloc_gpu<int32_t>(1,                        state_allocs_, total_vram_bytes_, stream_);

    // 6. Device-side decode meta chaining (token history + exact host-built tau(pos) table)
    tok_hist_cap_ = cfg_.max_ctx + 2048;
    d_tok_hist_   = alloc_gpu<int32_t>(tok_hist_cap_, state_allocs_, total_vram_bytes_, stream_);
    tau_tab_cap_  = cfg_.max_ctx + 2048;
    d_tau_tab_    = alloc_gpu<float>(tau_tab_cap_, state_allocs_, total_vram_bytes_, stream_);
    {
        std::vector<float> tau((size_t)tau_tab_cap_);
        for (int32_t p = 0; p < tau_tab_cap_; ++p) {
            const float ratio = (float)(p + 1) / LOG_SCALE_FLOOR;
            tau[(size_t)p] = 1.0f + LOG_SCALE_ALPHA * std::log(ratio > 1.0f ? ratio : 1.0f);
        }
        HIP_CHECK(hipMemcpyAsync(d_tau_tab_, tau.data(), (size_t)tau_tab_cap_ * sizeof(float),
                                 hipMemcpyHostToDevice, stream_));
    }

    // 7. Quest block-sparse Global attention page metadata + shared selection scratch
    //    (allocated ONLY when --sparse-pages > 0: the dense path keeps 0 extra VRAM)
    if (cfg_.sparse_pages > 0) {
        const int32_t max_pages = std::min<int32_t>(
            (cfg_.max_ctx + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE, SPARSE_MAX_PAGES);
        for (int s = 0; s < N_LAYER_SLOTS; ++s) {
            if (is_swa_slot(s)) {
                continue;
            }
            if (!sparse_global_page_meta_alloc(sparse_meta_[s], max_pages, cfg_.sparse_meta, stream_)) {
                std::fprintf(stderr,
                    "InklingEngine: sparse Global page metadata allocation failed (%d pages)\n",
                    max_pages);
                std::abort();
            }
            total_vram_bytes_ += sparse_global_page_meta_bytes(max_pages, cfg_.sparse_meta);
        }
        if (!sparse_global_scratch_alloc(sparse_scratch_, stream_)) {
            std::fprintf(stderr, "InklingEngine: sparse Global selection scratch allocation failed\n");
            std::abort();
        }
        total_vram_bytes_ += sparse_global_scratch_bytes();
    }

    // 8. Pinned host staging (never pageable on the decode hot path)
    HIP_CHECK(hipHostMalloc((void **)&h_meta_pinned_,     sizeof(StepMeta)));
    HIP_CHECK(hipHostMalloc((void **)&h_argmax_pinned_,   sizeof(int32_t)));
    HIP_CHECK(hipHostMalloc((void **)&h_tok_hist_pinned_, (size_t)tok_hist_cap_ * sizeof(int32_t)));
    std::memset(h_meta_pinned_, 0, sizeof(StepMeta));
    *h_argmax_pinned_ = 0;

    // 8b. Issue #4: exact full-vocab GPU sampler scratch (4.1 KiB + 160 B; used by
    //     both the MTP verify rows and the plain decode row).
    d_sample_scratch_ = alloc_gpu<float>(MTP_SAMPLE_SCRATCH_WORDS, state_allocs_, total_vram_bytes_, stream_);
    d_sample_tok_     = alloc_gpu<int32_t>(MAX_VERIFY_TOKENS,     state_allocs_, total_vram_bytes_, stream_);
    d_sample_stats_   = alloc_gpu<float>((size_t)MAX_VERIFY_TOKENS * 4, state_allocs_, total_vram_bytes_, stream_);
    HIP_CHECK(hipHostMalloc((void **)&h_sample_tok_,   MAX_VERIFY_TOKENS * sizeof(int32_t)));
    HIP_CHECK(hipHostMalloc((void **)&h_sample_stats_, (size_t)MAX_VERIFY_TOKENS * 4 * sizeof(float)));

    // 9. Batched m-row decode scratch (only when n_slots > 1; also shared with MTP verify)
    if (cfg_.n_slots > 1) {
        alloc_batch_scratch();
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
}

void InklingEngine::alloc_batch_scratch() {
    if (batch_scratch_ready_) return;
    d_verify_meta_    = alloc_gpu<StepMeta>(MAX_VERIFY_TOKENS, state_allocs_, total_vram_bytes_, stream_);
    d_target_         = alloc_gpu<int32_t>(MAX_VERIFY_TOKENS, state_allocs_, total_vram_bytes_, stream_);
    d_verify_unique_  = alloc_gpu<int32_t>(64, state_allocs_, total_vram_bytes_, stream_);
    d_verify_mask_    = alloc_gpu<uint32_t>(64, state_allocs_, total_vram_bytes_, stream_);
    d_verify_slot_    = alloc_gpu<int32_t>(64 * TOP_K, state_allocs_, total_vram_bytes_, stream_);
    d_verify_w_       = alloc_gpu<float>(64 * TOP_K, state_allocs_, total_vram_bytes_, stream_);
    d_verify_cnt_     = alloc_gpu<int32_t>(64, state_allocs_, total_vram_bytes_, stream_);
    d_verify_list_    = alloc_gpu<int32_t>(64 * TOP_K, state_allocs_, total_vram_bytes_, stream_);
    d_num_unique_     = alloc_gpu<int32_t>(1, state_allocs_, total_vram_bytes_, stream_);
    buf_moe_slot_     = alloc_gpu<float>((size_t)MAX_VERIFY_TOKENS * TOP_K * N_EMBD,
                                         state_allocs_, total_vram_bytes_, stream_);
    buf_verify_act_   = alloc_gpu<float>((size_t)MAX_VERIFY_TOKENS * TOP_K * N_FF_EXP,
                                         state_allocs_, total_vram_bytes_, stream_);
    buf_verify_mixed_ = alloc_gpu<float>((size_t)MAX_VERIFY_TOKENS * 3 * N_EMBD,
                                         state_allocs_, total_vram_bytes_, stream_);
    buf_verify_msg_   = alloc_gpu<float>((size_t)MAX_VERIFY_TOKENS * 4 * N_EMBD,
                                         state_allocs_, total_vram_bytes_, stream_);
    buf_verify_gate_  = alloc_gpu<float>((size_t)MAX_VERIFY_TOKENS * 3 * LOOP_GATE_RANK,
                                         state_allocs_, total_vram_bytes_, stream_);
    HIP_CHECK(hipHostMalloc((void **)&h_target_pinned_, MAX_VERIFY_TOKENS * sizeof(int32_t)));
    HIP_CHECK(hipHostMalloc((void **)&h_verify_meta_,   MAX_VERIFY_TOKENS * sizeof(StepMeta)));
    std::memset(h_target_pinned_, 0, MAX_VERIFY_TOKENS * sizeof(int32_t));
    std::memset(h_verify_meta_,   0, MAX_VERIFY_TOKENS * sizeof(StepMeta));
    batch_scratch_ready_ = true;
}

KVCacheSlot InklingEngine::slot_kv_view(int slot_id, int layer_slot) const {
    if (slot_id <= 0 || slot_id >= cfg_.n_slots) {
        return kv_slots_[layer_slot];
    }
    if (is_swa_slot(layer_slot)) {
        return slot_swa_kv_[slot_id][layer_slot];
    }
    KVCacheSlot v = kv_slots_[layer_slot];
    const size_t base = (size_t)slot_id * (size_t)slot_ctx_;
    const size_t qs_bytes_per_head = (cfg_.kv_type == KV_Q8_0) ? (size_t)HEAD_DIM : (size_t)(HEAD_DIM / 2);
    const size_t d_halves_per_head = (cfg_.kv_type == KV_Q4_1) ? (size_t)(HEAD_DIM / 16) : (size_t)(HEAD_DIM / 32);
    v.k_qs  += base * qs_bytes_per_head;
    v.k_d   += base * d_halves_per_head;
    v.v_qs  += base * qs_bytes_per_head;
    v.v_d   += base * d_halves_per_head;
    v.k_pos += base;
    return v;
}

SparseGlobalPageMeta InklingEngine::slot_sparse_meta_view(int slot_id, int layer_slot) const {
    SparseGlobalPageMeta m = sparse_meta_[layer_slot];
    if (slot_id <= 0 || slot_id >= cfg_.n_slots || m.max_pages <= 0) {
        return m;
    }
    const size_t base_page = ((size_t)slot_id * (size_t)slot_ctx_) / SPARSE_PAGE_SIZE;
    if (m.k_min_q) m.k_min_q += base_page * HEAD_DIM;
    if (m.k_max_q) m.k_max_q += base_page * HEAD_DIM;
    if (m.k_qmm)   m.k_qmm   += base_page * HEAD_DIM;
    if (m.k_mm_d)  m.k_mm_d  += base_page;
    return m;
}

float * InklingEngine::slot_sconv_hist(int slot_id, int layer_slot) const {
    if (slot_id <= 0 || slot_id >= cfg_.n_slots) {
        return sconv_slots_[layer_slot].hist;
    }
    return slot_sconv_[slot_id][layer_slot].hist;
}

void InklingEngine::reset_state() {
    cur_pos_ = 0;
    for (int c = 0; c < MAX_CONCURRENT_SLOTS; ++c) {
        slot_pos_[c] = 0;
        slot_span_[c] = 1;
    }
    for (int s = 0; s < N_LAYER_SLOTS; ++s) {
        HIP_CHECK(hipMemsetAsync(kv_slots_[s].k_pos, 0xFF, (size_t)kv_slots_[s].cap * sizeof(int32_t), stream_));
        HIP_CHECK(hipMemsetAsync(sconv_slots_[s].hist, 0, (size_t)N_SCONV_CHANNELS * N_CONV_HIST * sizeof(float), stream_));
    }
    for (int c = 1; c < cfg_.n_slots; ++c) {
        for (int s = 0; s < N_LAYER_SLOTS; ++s) {
            if (is_swa_slot(s)) {
                HIP_CHECK(hipMemsetAsync(slot_swa_kv_[c][s].k_pos, 0xFF, (size_t)cfg_.swa_cap * sizeof(int32_t), stream_));
            }
            HIP_CHECK(hipMemsetAsync(slot_sconv_[c][s].hist, 0, (size_t)N_SCONV_CHANNELS * N_CONV_HIST * sizeof(float), stream_));
        }
    }
    // One self-resetting barrier sector per verify row (Step 8C runs A/B/C with grid.y = row).
    HIP_CHECK(hipMemsetAsync(buf_router_scratch_, 0,
                             (size_t)MAX_VERIFY_TOKENS * 512 * sizeof(float), stream_));
    HIP_CHECK(hipMemsetAsync(buf_mix_scratch_, 0, 16384 * sizeof(float), stream_));
    HIP_CHECK(hipMemsetAsync(buf_fa_done_, 0, 128 * sizeof(int32_t), stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));
}

int32_t InklingEngine::slot_ctx_limit(int slot, int span) const {
    if (slot < 0 || slot >= cfg_.n_slots || span < 1) return 0;
    if (cfg_.n_slots == 1) return cfg_.max_ctx;
    const int max_span = cfg_.n_slots - slot;
    if (span > max_span) span = max_span;
    if (slot == 0 && span == cfg_.n_slots) return cfg_.max_ctx;
    return span * slot_ctx_;
}

void InklingEngine::reset_slot(int slot, int span) {
    if (slot < 0 || slot >= cfg_.n_slots) return;
    if (cfg_.n_slots == 1 && slot == 0) {
        reset_state();
        return;
    }
    const int max_span = cfg_.n_slots - slot;
    if (span < 1) span = 1;
    if (span > max_span) span = max_span;
    slot_span_[slot] = span;
    slot_pos_[slot] = 0;
    if (slot == 0) cur_pos_ = 0;
    const size_t base = (size_t)slot * (size_t)slot_ctx_;
    const int32_t clear_global = slot_ctx_limit(slot, span);
    for (int s = 0; s < N_LAYER_SLOTS; ++s) {
        if (is_swa_slot(s)) {
            KVCacheSlot kv = slot_kv_view(slot, s);
            HIP_CHECK(hipMemsetAsync(kv.k_pos, 0xFF, (size_t)kv.cap * sizeof(int32_t), stream_));
        } else {
            HIP_CHECK(hipMemsetAsync(kv_slots_[s].k_pos + base, 0xFF,
                                     (size_t)clear_global * sizeof(int32_t), stream_));
        }
        HIP_CHECK(hipMemsetAsync(slot_sconv_hist(slot, s), 0,
                                 (size_t)N_SCONV_CHANNELS * N_CONV_HIST * sizeof(float), stream_));
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
}

bool InklingEngine::expand_slot_span(int slot, int new_span) {
    if (slot < 0 || slot >= cfg_.n_slots) return false;
    if (cfg_.n_slots == 1) return (new_span == 1);
    const int max_span = cfg_.n_slots - slot;
    if (new_span <= slot_span_[slot] || new_span > max_span) return false;
    const int32_t old_limit = slot_ctx_limit(slot, slot_span_[slot]);
    const int32_t new_limit = slot_ctx_limit(slot, new_span);
    const size_t base = (size_t)slot * (size_t)slot_ctx_;
    const int32_t clear_n = new_limit - old_limit;
    if (clear_n > 0) {
        for (int s = 0; s < N_LAYER_SLOTS; ++s) {
            if (!is_swa_slot(s)) {
                HIP_CHECK(hipMemsetAsync(kv_slots_[s].k_pos + base + (size_t)old_limit, 0xFF,
                                         (size_t)clear_n * sizeof(int32_t), stream_));
            }
        }
        HIP_CHECK(hipStreamSynchronize(stream_));
    }
    slot_span_[slot] = new_span;
    return true;
}

bool InklingEngine::shrink_slot_span(int slot, int new_span) {
    if (slot < 0 || slot >= cfg_.n_slots) return false;
    if (cfg_.n_slots == 1) return (new_span == 1);
    if (new_span < 1 || new_span > cfg_.n_slots - slot) return false;
    if (slot_pos_[slot] > slot_ctx_limit(slot, new_span)) return false;
    slot_span_[slot] = new_span;
    return true;
}

void InklingEngine::rewind_pos_for_bench(int32_t target_pos) {
    if (target_pos <= 0) {
        reset_state();
        return;
    }
    if (target_pos >= cur_pos_) return;
    // Clear global k_pos entries [target_pos .. cur_pos_-1] back to -1 so FA ignores overwritten benchmark tokens
    const int32_t clear_n = cur_pos_ - target_pos;
    for (int s = 0; s < N_LAYER_SLOTS; ++s) {
        if (!kv_slots_[s].is_swa) {
            HIP_CHECK(hipMemsetAsync(
                kv_slots_[s].k_pos + target_pos, 0xFF,
                (size_t)clear_n * sizeof(int32_t), stream_));
        }
    }
    cur_pos_ = target_pos;
    slot_pos_[0] = target_pos;
    HIP_CHECK(hipStreamSynchronize(stream_));
}

void InklingEngine::record_layer(
        int slot, int ip,
        const float * cur_in, float * cur_out,
        std::vector<const float *> & blocks,
        const float * messages,
        const LoopLayerWeights * loop,
        int n_tokens, int pos_start, const StepMeta * d_meta,
        bool attn_already_mixed,
        const float * const * mix_states0, int n_mix_states0,
        const float * const * mix_states1, int n_mix_states1,
        int row_slot) {
    const int phys = slot_to_phys(slot);
    const auto & L = weights_.layers[phys];
    const bool is_swa = L.is_swa;
    const size_t tok_embd_stride = (size_t)n_tokens * N_EMBD;

    // Step 8A: row offsets of the [MAX_UBATCH, ...] activation buffers for the
    // speculative verify path (row_slot == 0 gives 0 for every one of them).
    const size_t o_embd = (size_t)row_slot * N_EMBD;
    const size_t o_mix  = (size_t)row_slot * 4 * N_EMBD;
    const size_t o_gate = (size_t)row_slot * 3 * LOOP_GATE_RANK;
    const size_t o_qkv  = (size_t)row_slot * 8600;
    const size_t o_kv   = (size_t)row_slot * 1056;
    const size_t o_q    = (size_t)row_slot * N_Q;
    const size_t o_rel  = (size_t)row_slot * N_HEAD * N_REL_EXTENT;
    const size_t o_topk = (size_t)row_slot * TOP_K;

    float       * const norm_[3]     = { buf_norm_[0] + o_embd, buf_norm_[1] + o_embd, buf_norm_[2] + o_embd };
    float       * const mixed_       = buf_mixed_ + o_mix;
    float       * const ctrl_gate_   = buf_ctrl_gate_ + o_gate;
    float       * const proj_qkv_    = buf_proj_qkv_ + o_qkv;
    float       * const proj_k_      = buf_proj_k_ + o_kv;
    float       * const proj_v_      = buf_proj_v_ + o_kv;
    float       * const q_prep_      = buf_q_prep_ + o_q;
    float       * const rel_table_   = buf_rel_table_ + o_rel;
    float       * const fa_out_      = buf_fa_out_ + o_q;
    float       * const wo_out_      = buf_wo_out_ + o_embd;
    float       * const x_norm_      = buf_x_norm_ + o_embd;
    int32_t     * const topk_ids_    = buf_topk_ids_ + o_topk;
    float       * const topk_w_      = buf_topk_w_ + o_topk;

    // Step 8A: the DSpark taps are the hidden entering visits 5, 23, 41 (in this
    // engine: prelude layer 5, pass 1 ip 11 => slot 29, pass 2 ip 17 => slot 47);
    // the reference writes them as bf16(hid) inside the visit's mix launch.
    if (d_taps_ != nullptr && mtp_io_ && !batch_slots_active_ && active_slot_ == 0) {
        const int tap = (slot == 5) ? 0 : (slot == 29) ? 1 : (slot == 47) ? 2 : -1;
        if (tap >= 0) {
            launch_mtp_capture_tap(cur_in, d_taps_ + o_mix, n_tokens, tap, stream_);
        }
    }

    const float * inpSA = cur_in;

    // 1 + 2 + 3. Depth attention mix -> optional cross-pass loop controller -> Attention input RMSNorm
    const bool shared_swa_inp = (is_swa && messages == nullptr);
    const float * inputs[3] = { cur_in, cur_in, cur_in };

    // ---- Step 7I-1: capture the input-norm / A arguments for the fused
    //      launches (Pass-0 norm+QKV, Pass-1/2 A+QKV) and skip the split
    //      kernels when the winning fused launch takes over. --------------
    const bool s7i_decode   = (n_tokens == 1 && d_meta != nullptr);
    const bool s7i_mixqkv   = s7i_decode && messages == nullptr && !attn_already_mixed &&
                              cfg_.fuse_mix_qkv != 0 &&
                              (std::getenv("INKLING_MQKV_SWA_ONLY") == nullptr || is_swa) &&
                              (std::getenv("INKLING_MQKV_GLOBAL_ONLY") == nullptr || !is_swa);
    const bool s7i_aqkv2    = s7i_decode && messages != nullptr && FUSE_A &&
                              cfg_.fuse_a_qkv >= 2 && !attn_already_mixed &&
                              mix_states0 != nullptr && mix_states1 != nullptr;
    const bool s7i_a_v3     = s7i_decode && messages != nullptr && FUSE_A &&
                              cfg_.fuse_a_qkv == 1 && !attn_already_mixed &&
                              mix_states0 != nullptr && mix_states1 != nullptr;

    const float * s7i_query  = nullptr;
    int           s7i_nq     = 0;
    const float * s7i_states[MAX_RES_BLOCKS] = { nullptr };
    int           s7i_nstate = 0;
    const float * s7i_wn[3]  = { nullptr, nullptr, nullptr };
    float *       s7i_out    = nullptr;

    const size_t q_stride_rows = (size_t)N_EMBD;

    const float * a_q0 = nullptr, * a_q1 = nullptr;
    float * a_o0 = nullptr, * a_o1 = nullptr;
    int a_nq0 = 0, a_nq1 = 0, a_ns0 = 0, a_ns1 = 0;
    const float * a_s0[MAX_RES_BLOCKS] = { nullptr };
    const float * a_s1[MAX_RES_BLOCKS] = { nullptr };
    const __half * a_wd[3] = { nullptr, nullptr, nullptr };
    const __half * a_wg[3] = { nullptr, nullptr, nullptr };
    const __half * a_wf[3] = { nullptr, nullptr, nullptr };
    const float *  a_wn3[3] = { nullptr, nullptr, nullptr };
    float *        a_on[3]  = { nullptr, nullptr, nullptr };

    if (messages == nullptr) {
        if (L.attn_res_query != nullptr) {
            const int n_states = (int)blocks.size() + 1;
            const float * states[MAX_RES_BLOCKS];
            int idx = 0;
            for (int i = (int)blocks.size() - 1; i >= 0; --i) {
                states[idx++] = blocks[i];
            }
            states[idx++] = cur_in;
            if (s7i_mixqkv) {
                s7i_query = L.attn_res_query;
                s7i_nq    = is_swa ? 1 : 3;
                s7i_nstate = n_states;
                for (int i = 0; i < n_states; ++i) s7i_states[i] = states[i];
                if (is_swa) {
                    s7i_wn[0] = L.attn_norm;
                    s7i_out   = norm_[0];
                } else {
                    s7i_wn[0] = L.attn_q_inp_norm;
                    s7i_wn[1] = L.attn_k_inp_norm;
                    s7i_wn[2] = L.attn_v_inp_norm;
                    s7i_out   = mixed_;
                }
            }
            if (is_swa) {
                const float * w_norms[1] = { L.attn_norm };
                if (!s7i_mixqkv) {
                    if (cfg_.mix_fast > 0 && n_tokens == 1) {
                        launch_mix_rms_norm_2304_fast(L.attn_res_query, 1, states, n_states, w_norms,
                                                      norm_[0], RMS_EPS, stream_);
                    } else {
                        launch_inkling_mix_rms_norm(L.attn_res_query, 1, states, n_states, w_norms, norm_[0], n_tokens, RMS_EPS, stream_);
                    }
                }
                inputs[0] = inputs[1] = inputs[2] = norm_[0];
            } else {
                const float * w_norms[3] = { L.attn_q_inp_norm, L.attn_k_inp_norm, L.attn_v_inp_norm };
                if (!s7i_mixqkv) {
                    if (cfg_.mix_fast > 0 && n_tokens == 1) {
                        launch_mix_rms_norm_2304_fast(L.attn_res_query, 3, states, n_states, w_norms,
                                                      mixed_, RMS_EPS, stream_);
                    } else {
                        launch_inkling_mix_rms_norm(L.attn_res_query, 3, states, n_states, w_norms, mixed_, n_tokens, RMS_EPS, stream_);
                    }
                }
                inputs[0] = mixed_ + 0 * tok_embd_stride;
                inputs[1] = mixed_ + 1 * tok_embd_stride;
                inputs[2] = mixed_ + 2 * tok_embd_stride;
            }
        } else {
            // phys == 0 (always SWA, no attn_res_query)
            if (s7i_mixqkv) {
                s7i_query  = cur_in;
                s7i_nq     = 1;
                s7i_nstate = 0;
                s7i_wn[0]  = L.attn_norm;
                s7i_out    = norm_[0];
            } else {
                launch_rms_norm(cur_in, L.attn_norm, norm_[0], n_tokens, N_EMBD, RMS_EPS, stream_);
            }
            inputs[0] = inputs[1] = inputs[2] = norm_[0];
        }
    } else {
        // Looped Pass 1 & Pass 2 (layers 6..17): mix (possibly pre-fused with msg_mix!) -> fused 3-way (Q, K, V) ctrl + RMSNorm
        const __half * w_down[3]   = { loop->ctrl_down[LOOP_CTRL_Q],   loop->ctrl_down[LOOP_CTRL_K],   loop->ctrl_down[LOOP_CTRL_V] };
        const __half * w_gain[3]   = { loop->ctrl_gain[LOOP_CTRL_Q],   loop->ctrl_gain[LOOP_CTRL_K],   loop->ctrl_gain[LOOP_CTRL_V] };
        const __half * w_forget[3] = { loop->ctrl_forget[LOOP_CTRL_Q], loop->ctrl_forget[LOOP_CTRL_K], loop->ctrl_forget[LOOP_CTRL_V] };
        const float  * w_norm[3]   = {
            is_swa ? L.attn_norm : L.attn_q_inp_norm,
            is_swa ? L.attn_norm : L.attn_k_inp_norm,
            is_swa ? L.attn_norm : L.attn_v_inp_norm
        };
        float * out_norm[3] = { norm_[0], norm_[1], norm_[2] };

        bool fused_a = (verify_m_ > 0 && attn_already_mixed);
        if (L.attn_res_query != nullptr) {
            const int n_q = is_swa ? 1 : 3;
            if (!attn_already_mixed) {
                if (FUSE_A && n_tokens == 1 && d_meta != nullptr &&
                    mix_states0 != nullptr && mix_states1 != nullptr) {
                    if (s7i_aqkv2 || s7i_a_v3) {
                        a_q0 = loop->msg_query; a_nq0 = LOOP_CTRL_COUNT;
                        a_ns0 = n_mix_states0;
                        for (int i = 0; i < n_mix_states0; ++i) a_s0[i] = mix_states0[i];
                        a_o0 = const_cast<float *>(messages);
                        a_q1 = L.attn_res_query; a_nq1 = n_q;
                        a_ns1 = n_mix_states1;
                        for (int i = 0; i < n_mix_states1; ++i) a_s1[i] = mix_states1[i];
                        a_o1 = mixed_;
                        for (int i = 0; i < 3; ++i) {
                            a_wd[i] = w_down[i]; a_wg[i] = w_gain[i]; a_wf[i] = w_forget[i];
                            a_wn3[i] = w_norm[i]; a_on[i] = out_norm[i];
                        }
                    }
                    if (!s7i_aqkv2) {
                        if (s7i_a_v3) {
                            launch_a_v3_decode(
                                a_q0, a_nq0, a_s0, a_ns0, a_o0,
                                a_q1, a_nq1, a_s1, a_ns1, a_o1,
                                a_wd, a_wg, a_wf, a_wn3,
                                ctrl_gate_, a_on, RMS_EPS, stream_, buf_mix_scratch_);
                        } else {
                            // A: dual depth mix + 3-way ctrl QKV + RMSNorm in ONE launch.
                            launch_mix_dual_ctrl_qkv3_norm_decode(
                                loop->msg_query, LOOP_CTRL_COUNT, mix_states0, n_mix_states0, const_cast<float *>(messages),
                                L.attn_res_query, n_q, mix_states1, n_mix_states1, mixed_,
                                w_down, w_gain, w_forget, w_norm,
                                ctrl_gate_, out_norm, 1, RMS_EPS, stream_, buf_router_scratch_);
                        }
                        fused_a = true;
                    } else {
                        fused_a = true;   // the fused A+QKV launch happens below
                    }
                } else {
                    const int n_states = (int)blocks.size() + 1;
                    const float * states[MAX_RES_BLOCKS];
                    int idx = 0;
                    for (int i = (int)blocks.size() - 1; i >= 0; --i) {
                        states[idx++] = blocks[i];
                    }
                    states[idx++] = cur_in;
                    launch_inkling_mix(L.attn_res_query, n_q, states, n_states, mixed_, n_tokens, stream_);
                }
            }
            if (!fused_a) {
                if (is_swa) {
                    inputs[0] = inputs[1] = inputs[2] = mixed_;
                } else {
                    inputs[0] = mixed_ + 0 * tok_embd_stride;
                    inputs[1] = mixed_ + 1 * tok_embd_stride;
                    inputs[2] = mixed_ + 2 * tok_embd_stride;
                }
            }
        }
        if (!fused_a) {
            launch_inkling_ctrl_qkv_norm(
                inputs, messages + (size_t)LOOP_CTRL_Q * tok_embd_stride,
                w_down, w_gain, w_forget, w_norm,
                ctrl_gate_, out_norm, n_tokens, RMS_EPS, stream_, buf_router_scratch_);
        }
        inputs[0] = norm_[0];
        inputs[1] = norm_[1];
        inputs[2] = norm_[2];
    }

    // 4. Fused SoA Q8_0 Attention Projections (GEMV on decode, FP16 WMMA on prefill)
    auto run_proj = [&](const SoaQ8Matrix & W, const float * X, float * Y) {
        if (n_tokens == 1) {
            launch_soa_q8_gemv(W, X, Y, stream_);
        } else {
            launch_soa_q8_gemm_wmma(W, X, Y, n_tokens, stream_);
        }
    };

    // Step 8B: batched verify streams every QKV weight once for all m rows.
    const bool vsync2 = std::getenv("INKLING_VERIFY_SYNC") != nullptr;
    const bool vrow_qkv = std::getenv("INKLING_VROW_QKV") != nullptr;
    const int vq_xstride = (int)q_stride_rows;
    if (verify_m_ > 0 && vrow_qkv) {
        for (int vr = 0; vr < n_tokens; ++vr) {
            if (shared_swa_inp) {
                launch_soa_q8_gemv(L.proj_qkv_all, inputs[0] + (size_t)vr * N_EMBD,
                                   proj_qkv_ + (size_t)vr * 8600, stream_);
            } else {
                launch_soa_q8_gemv_qkv3(
                    L.proj_q4, inputs[0] + (size_t)vr * N_EMBD, proj_qkv_ + (size_t)vr * 6552,
                    L.proj_k,  inputs[1] + (size_t)vr * N_EMBD, proj_k_ + (size_t)vr * L.proj_k.rows,
                    L.proj_v,  inputs[2] + (size_t)vr * N_EMBD, proj_v_ + (size_t)vr * L.proj_v.rows,
                    stream_);
            }
        }
    } else if (verify_m_ > 0) {
        if (shared_swa_inp) {
            const VerifyGemvSeg sg[1] = {{ L.proj_qkv_all, inputs[0], proj_qkv_, 0 }};
            launch_verify_gemv_mrows(sg, 1, verify_m_, vq_xstride, stream_);
        } else {
            const VerifyGemvSeg sg[3] = {
                { L.proj_q4, inputs[0], proj_qkv_, 0 },
                { L.proj_k,  inputs[1], proj_k_,  0 },
                { L.proj_v,  inputs[2], proj_v_,  0 },
            };
            launch_verify_gemv_mrows(sg, 3, verify_m_, vq_xstride, stream_);
        }
        if (std::getenv("INKLING_QKV_CHECK") != nullptr && slot == 0 && shared_swa_inp) {
            HIP_CHECK(hipStreamSynchronize(stream_));
            double md = 0.0; int wi = -1, wr = -1;
            for (int vr = 0; vr < n_tokens; ++vr) {
                launch_soa_q8_gemv(L.proj_qkv_all, inputs[0] + (size_t)vr * N_EMBD,
                                   buf_moe_out_, stream_);
                HIP_CHECK(hipStreamSynchronize(stream_));
                std::vector<float> a(8600), b(8600);
                HIP_CHECK(hipMemcpy(a.data(), proj_qkv_ + (size_t)vr * 8600, a.size() * 4, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(b.data(), buf_moe_out_, b.size() * 4, hipMemcpyDeviceToHost));
                for (int i = 0; i < 8600; ++i) {
                    const double d = std::fabs((double)a[(size_t)i] - (double)b[(size_t)i]);
                    if (d > md) { md = d; wi = i; wr = vr; }
                }
            }
            std::printf("[qkv-check] max |d| = %.6g (row %d col %d)\n", md, wr, wi);
            std::fflush(stdout);
        }
        if (vsync2) { HIP_CHECK(hipStreamSynchronize(stream_)); std::fprintf(stderr, "[v] slot=%d qkv ok\n", slot); }
    }

    int q_stride = 0, k_stride = 0, v_stride = 0;
    const float * q_raw = nullptr;
    const float * qgate = nullptr;
    const float * r_raw = nullptr;
    const float * sinks = nullptr;
    const float * k_raw = nullptr;
    const float * k_dyn = nullptr;
    const float * v_raw = nullptr;
    const float * v_dyn = nullptr;

    // E (Step 7D): QKV GEMV + qkvr_prep in ONE launch (decode only; needs d_meta).
    const bool fuse_e = FUSE_E && n_tokens == 1 && d_meta != nullptr;

    if (shared_swa_inp) {
        // ONE launch computes all 6 projections: [q(3072) | gate(3072) | rel(384) | sink(24) | k(1024) | v(1024)] = 8600
        if (!fuse_e && verify_m_ == 0) run_proj(L.proj_qkv_all, inputs[0], proj_qkv_);
        q_stride = k_stride = v_stride = 8600;
        q_raw = proj_qkv_ + 0;
        qgate = proj_qkv_ + 3072;
        r_raw = proj_qkv_ + 6144;
        sinks = proj_qkv_ + 6528;
        k_raw = proj_qkv_ + 6552;
        v_raw = proj_qkv_ + 7576;
    } else {
        // Q-side: [q(3072) | gate(3072) | rel(384) | sink(24)] = 6552, plus separate K and V
        if (verify_m_ > 0) {
            // already launched above (m-row exact GEMV)
        } else if (fuse_e) {
            // projections are launched by launch_qkv_qkvr_prep_fused_decode below
        } else if (n_tokens == 1) {
            launch_soa_q8_gemv_qkv3(
                L.proj_q4, inputs[0], proj_qkv_,
                L.proj_k,  inputs[1], proj_k_,
                L.proj_v,  inputs[2], proj_v_,
                stream_);
        } else {
            launch_soa_q8_gemm_wmma_qkv3(
                L.proj_q4, inputs[0], proj_qkv_,
                L.proj_k,  inputs[1], proj_k_,
                L.proj_v,  inputs[2], proj_v_,
                n_tokens, stream_);
        }
        q_stride = 6552;
        q_raw = proj_qkv_ + 0;
        qgate = proj_qkv_ + 3072;
        r_raw = proj_qkv_ + 6144;
        sinks = proj_qkv_ + 6528;

        k_stride = L.proj_k.rows; // 1024 (SWA) or 1056 (Global)
        v_stride = L.proj_v.rows;
        k_raw = proj_k_;
        v_raw = proj_v_;
        if (!is_swa) {
            k_dyn = proj_k_ + N_KV;
            v_dyn = proj_v_ + N_KV;
        }
    }

    // 5 + 6. K/V shortconv+KV store AND Q/R prep+rel_table
    float * hist_base = slot_sconv_hist(active_slot_, slot);
    const KVCacheSlot cur_kv = slot_kv_view(active_slot_, slot);
    const SparseGlobalPageMeta cur_sm = slot_sparse_meta_view(active_slot_, slot);
    bool minmax_folded = false;
    if (verify_m_ > 0 && batch_slots_active_) {
        BatchKVSlotArgs kv_args{};
        for (int r = 0; r < n_tokens; ++r) {
            const int c = batch_slot_ids_[r];
            kv_args.hist_base[r] = slot_sconv_hist(c, slot);
            kv_args.kv[r]        = slot_kv_view(c, slot);
        }
        launch_batch_sconv_kv_store_q8(
            k_raw, k_stride, L.attn_k_conv, k_dyn, k_stride,
            v_raw, v_stride, L.attn_v_conv, v_dyn, v_stride,
            kv_args, n_tokens, d_meta, stream_);
        launch_q_r_prep_and_reltable(
            q_raw, q_stride, L.attn_q_clip,
            r_raw, q_stride, L.attn_rel_clip, L.attn_rel_bias,
            q_prep_, rel_table_, /*apply_tau=*/!is_swa,
            n_tokens, pos_start, d_meta, stream_);
    } else if (verify_m_ > 0) {
        // Step 8B: exact sequential 4-tap shortconv across the m rows (per-channel
        // in registers) + per-row RoPE/KV store + q/r prep; pos comes from
        // d_verify_meta_[t] so the whole pass is HIP-graph static.
        launch_sconv_kv_store_q8(
            k_raw, k_stride, L.attn_k_conv, k_dyn, k_stride,
            hist_base + 0 * N_CONV_HIST, cur_kv, /*is_v=*/false,
            n_tokens, pos_start, d_meta, stream_);
        launch_sconv_kv_store_q8(
            v_raw, v_stride, L.attn_v_conv, v_dyn, v_stride,
            hist_base + (size_t)N_KV * N_CONV_HIST, cur_kv, /*is_v=*/true,
            n_tokens, pos_start, d_meta, stream_);
        launch_q_r_prep_and_reltable(
            q_raw, q_stride, L.attn_q_clip,
            r_raw, q_stride, L.attn_rel_clip, L.attn_rel_bias,
            q_prep_, rel_table_, /*apply_tau=*/!is_swa,
            n_tokens, pos_start, d_meta, stream_);
        if (vsync2) { HIP_CHECK(hipStreamSynchronize(stream_)); std::fprintf(stderr, "[v] slot=%d prep ok\n", slot); }
    } else if (s7i_mixqkv) {
        const SoaQ8Matrix none{};
        launch_mix_rms_qkv_qkvr_prep_fused(
            s7i_query, s7i_nq, s7i_states, s7i_nstate, s7i_wn, s7i_out,
            shared_swa_inp ? L.proj_qkv_all : L.proj_q4,
            shared_swa_inp ? inputs[0] : inputs[0],
            proj_qkv_,
            shared_swa_inp ? none : L.proj_k,
            shared_swa_inp ? nullptr : inputs[1],
            shared_swa_inp ? nullptr : proj_k_,
            shared_swa_inp ? none : L.proj_v,
            shared_swa_inp ? nullptr : inputs[2],
            shared_swa_inp ? nullptr : proj_v_,
            q_raw, L.attn_q_clip,
            r_raw, L.attn_rel_clip, L.attn_rel_bias,
            k_raw, L.attn_k_conv, k_dyn,
            v_raw, L.attn_v_conv, v_dyn,
            hist_base, cur_kv,
            q_prep_, rel_table_, /*apply_tau=*/!is_swa,
            d_meta, cfg_.fuse_minmax != 0, cur_sm,
            buf_mix_scratch_, stream_);
        minmax_folded = cfg_.fuse_minmax != 0 && std::getenv("INKLING_NO_MINMAX_FOLD") == nullptr;
    } else if (s7i_aqkv2) {
        launch_a_qkv_qkvr_prep_fused(
            a_q0, a_nq0, a_s0, a_ns0, a_o0,
            a_q1, a_nq1, a_s1, a_ns1, a_o1,
            a_wd, a_wg, a_wf, a_wn3,
            ctrl_gate_, a_on, RMS_EPS,
            L.proj_q4, inputs[0], proj_qkv_,
            L.proj_k, inputs[1], proj_k_,
            L.proj_v, inputs[2], proj_v_,
            q_raw, L.attn_q_clip,
            r_raw, L.attn_rel_clip, L.attn_rel_bias,
            k_raw, L.attn_k_conv, k_dyn,
            v_raw, L.attn_v_conv, v_dyn,
            hist_base, cur_kv,
            q_prep_, rel_table_, /*apply_tau=*/!is_swa,
            d_meta, cfg_.fuse_minmax != 0, cur_sm,
            buf_mix_scratch_, stream_);
        minmax_folded = cfg_.fuse_minmax != 0;
    } else if (fuse_e) {
        const SoaQ8Matrix none{};
        // Step 7G A/B: 1 = the 6-wave/SIMD v3 super-fused prep, else the
        // shipped v2 fused prep (bit-identical math, different launch shape).
        const bool use_qkv_v3 = (cfg_.qkv_super_v3 >= 1);   // 1 = v3 (v2 parser maps "v3" -> 2)
        if (cfg_.fdot2_qkv > 0) {
            launch_qkv_qkvr_prep_fdot2(
                shared_swa_inp ? L.proj_qkv_all : L.proj_q4, inputs[0], proj_qkv_,
                shared_swa_inp ? none : L.proj_k,
                shared_swa_inp ? nullptr : inputs[1],
                shared_swa_inp ? nullptr : proj_k_,
                shared_swa_inp ? none : L.proj_v,
                shared_swa_inp ? nullptr : inputs[2],
                shared_swa_inp ? nullptr : proj_v_,
                q_raw, L.attn_q_clip,
                r_raw, L.attn_rel_clip, L.attn_rel_bias,
                k_raw, L.attn_k_conv, k_dyn,
                v_raw, L.attn_v_conv, v_dyn,
                hist_base, cur_kv,
                q_prep_, rel_table_, /*apply_tau=*/!is_swa,
                d_meta, buf_router_scratch_, stream_,
                cfg_.fdot2_qkv == 3 ? 0 : (cfg_.fdot2_qkv == 2 ? 2 : 1));
        } else if (shared_swa_inp) {
            if (use_qkv_v3) {
                launch_qkv_qkvr_prep_fused_v3(
                    L.proj_qkv_all, inputs[0], proj_qkv_,
                    none, nullptr, nullptr, none, nullptr, nullptr,
                    q_raw, L.attn_q_clip,
                    r_raw, L.attn_rel_clip, L.attn_rel_bias,
                    k_raw, L.attn_k_conv, k_dyn,
                    v_raw, L.attn_v_conv, v_dyn,
                    hist_base, cur_kv,
                    q_prep_, rel_table_, /*apply_tau=*/!is_swa,
                    d_meta, buf_router_scratch_, stream_);
            } else {
                launch_qkv_qkvr_prep_fused_decode_g(
                    L.proj_qkv_all, inputs[0], proj_qkv_,
                    none, nullptr, nullptr, none, nullptr, nullptr,
                    q_raw, L.attn_q_clip,
                    r_raw, L.attn_rel_clip, L.attn_rel_bias,
                    k_raw, L.attn_k_conv, k_dyn,
                    v_raw, L.attn_v_conv, v_dyn,
                    hist_base, cur_kv,
                    q_prep_, rel_table_, /*apply_tau=*/!is_swa,
                    d_meta, buf_router_scratch_, stream_, cfg_.qkv_grid);
            }
        } else {
            if (use_qkv_v3) {
                launch_qkv_qkvr_prep_fused_v3(
                    L.proj_q4, inputs[0], proj_qkv_,
                    L.proj_k, inputs[1], proj_k_,
                    L.proj_v, inputs[2], proj_v_,
                    q_raw, L.attn_q_clip,
                    r_raw, L.attn_rel_clip, L.attn_rel_bias,
                    k_raw, L.attn_k_conv, k_dyn,
                    v_raw, L.attn_v_conv, v_dyn,
                    hist_base, cur_kv,
                    q_prep_, rel_table_, /*apply_tau=*/!is_swa,
                    d_meta, buf_router_scratch_, stream_);
            } else {
                launch_qkv_qkvr_prep_fused_decode_g(
                    L.proj_q4, inputs[0], proj_qkv_,
                    L.proj_k, inputs[1], proj_k_,
                    L.proj_v, inputs[2], proj_v_,
                    q_raw, L.attn_q_clip,
                    r_raw, L.attn_rel_clip, L.attn_rel_bias,
                    k_raw, L.attn_k_conv, k_dyn,
                    v_raw, L.attn_v_conv, v_dyn,
                    hist_base, cur_kv,
                    q_prep_, rel_table_, /*apply_tau=*/!is_swa,
                    d_meta, buf_router_scratch_, stream_, cfg_.qkv_grid);
            }
        }
    } else if (n_tokens == 1 && d_meta != nullptr) {
        launch_qkvr_prep_fused_decode(
            q_raw, q_stride, L.attn_q_clip,
            r_raw, q_stride, L.attn_rel_clip, L.attn_rel_bias,
            k_raw, k_stride, L.attn_k_conv, k_dyn, k_stride,
            v_raw, v_stride, L.attn_v_conv, v_dyn, v_stride,
            hist_base, cur_kv,
            q_prep_, rel_table_, /*apply_tau=*/!is_swa,
            d_meta, stream_);
    } else if (n_tokens > 1 && d_meta == nullptr) {
        launch_qkvr_prep_prefill(
            q_raw, q_stride, L.attn_q_clip,
            r_raw, q_stride, L.attn_rel_clip, L.attn_rel_bias,
            k_raw, k_stride, L.attn_k_conv, k_dyn, k_stride,
            v_raw, v_stride, L.attn_v_conv, v_dyn, v_stride,
            hist_base, cur_kv,
            q_prep_, rel_table_, /*apply_tau=*/!is_swa,
            n_tokens, pos_start, stream_);
    } else {
        launch_sconv_kv_store_q8(
            k_raw, k_stride, L.attn_k_conv, k_dyn, k_stride,
            hist_base + 0 * N_CONV_HIST, cur_kv, /*is_v=*/false,
            n_tokens, pos_start, d_meta, stream_);
        launch_sconv_kv_store_q8(
            v_raw, v_stride, L.attn_v_conv, v_dyn, v_stride,
            hist_base + (size_t)N_KV * N_CONV_HIST, cur_kv, /*is_v=*/true,
            n_tokens, pos_start, d_meta, stream_);
        launch_q_r_prep_and_reltable(
            q_raw, q_stride, L.attn_q_clip,
            r_raw, q_stride, L.attn_rel_clip, L.attn_rel_bias,
            q_prep_, rel_table_, /*apply_tau=*/!is_swa,
            n_tokens, pos_start, d_meta, stream_);
    }

    // Step 8A: speculative verify records the raw pre-conv K/V inputs of this row
    // (they are still in the projection buffer; the fused epilogue only read them).
    if (verify_mode_) {
        float * raw = d_verify_raw_ + ((size_t)slot * MAX_VERIFY_TOKENS + verify_row_) * N_SCONV_CHANNELS;
        launch_mtp_copy_raw_pair(k_raw, k_stride, v_raw, v_stride, raw, stream_);
    } else if (verify_m_ > 0 && !batch_slots_active_) {
        // Step 8C: one launch for all m rows (was one per row per layer slot).
        launch_mtp_copy_raw_pair_mrows(
            k_raw, k_stride, v_raw, v_stride,
            d_verify_raw_ + (size_t)slot * MAX_VERIFY_TOKENS * N_SCONV_CHANNELS,
            N_SCONV_CHANNELS, n_tokens, stream_);
    }

    if (const char * dq = std::getenv("INKLING_DUMP_Q")) {
        HIP_CHECK(hipStreamSynchronize(stream_));
        std::vector<float> tmp(3 * N_EMBD + N_Q + N_EMBD);
        HIP_CHECK(hipMemcpy(tmp.data(), mixed_, (size_t)3 * N_EMBD * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(tmp.data() + 3 * N_EMBD, q_prep_, (size_t)N_Q * sizeof(float), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(tmp.data() + 3 * N_EMBD + N_Q, norm_[0], (size_t)N_EMBD * sizeof(float), hipMemcpyDeviceToHost));
        if (FILE * fp = std::fopen(dq, "ab")) {
            std::fwrite(tmp.data(), 4, tmp.size(), fp);
            std::fclose(fp);
        }
    }

    const char * stage_dump = std::getenv("INKLING_DUMP_STAGE");
    const int stage_slot = std::getenv("INKLING_STAGE_SLOT") ? std::atoi(std::getenv("INKLING_STAGE_SLOT")) : 0;
    const int stage_rows = (verify_m_ > 0) ? verify_m_ : ((n_tokens == 1) ? 1 : 0);
    if (stage_dump != nullptr && slot == stage_slot && stage_rows > 0) {
        HIP_CHECK(hipStreamSynchronize(stream_));
        auto dump = [&](const char * tag, const float * p, size_t n) {
            if (FILE * fp = std::fopen(stage_dump, "ab")) {
                const int32_t hdr[2] = { stage_rows, cur_pos_ };
                std::fwrite(hdr, 4, 2, fp);
                std::fwrite(tag, 1, 8, fp);
                std::fwrite(p, 4, n, fp);
                std::fclose(fp);
            }
        };
        dump("norm0   ", norm_[0], (size_t)stage_rows * N_EMBD);
        dump("projqkv ", proj_qkv_, (size_t)stage_rows * q_stride);
        dump("qprep   ", q_prep_, (size_t)stage_rows * N_Q);
        dump("reltab  ", rel_table_, (size_t)stage_rows * N_HEAD * N_REL_EXTENT);
    }

    // 7. Native RDNA3 On-The-Fly RelBias FlashAttention + fused output FWHT-128 + sigmoid(qgate)
    //    Global layers with --sparse-pages > 0 use Quest-style block-sparse attention
    //    over the query-selected 64-position KV pages; SWA layers stay dense (window <= 1536).
    if (verify_m_ > 0 && batch_slots_active_) {
        // Multi-stream independent-slot decode attention: each row vr attends over
        // its own slot's SWA ring or zero-copy shifted Global slice & Quest metadata.
        for (int vr = 0; vr < n_tokens; ++vr) {
            const int c = batch_slot_ids_[vr];
            const KVCacheSlot kv_r = slot_kv_view(c, slot);
            const SparseGlobalPageMeta sm_r = slot_sparse_meta_view(c, slot);
            const float * qv    = q_prep_ + (size_t)vr * N_Q;
            const float * sinkv = (sinks != nullptr) ? (sinks + (size_t)vr * q_stride) : nullptr;
            const float * relv  = rel_table_ + (size_t)vr * N_HEAD * N_REL_EXTENT;
            const float * gatev = (qgate != nullptr) ? (qgate + (size_t)vr * q_stride) : nullptr;
            float * outv        = fa_out_ + (size_t)vr * N_Q;
            const StepMeta * dm = d_meta + vr;
            if (!is_swa && cfg_.sparse_pages > 0) {
                launch_kv_update_page_minmax_decode(kv_r, sm_r, dm, stream_);
                SparseGlobalScratch & vs = sparse_scratch_;
                const int sel_ver  = (cfg_.fa_select_v3 < 0) ? 1 : cfg_.fa_select_v3;
                const int glob_ver = (cfg_.fa_global_v3 < 0) ? 3 : cfg_.fa_global_v3;
                if (sel_ver == 3) {
                    launch_sparse_global_select_v4(qv, sm_r, vs, dm,
                                                   cfg_.sparse_pages, buf_fa_done_ + 16, stream_,
                                                   cfg_.fa_q_lds, cfg_.fa_cached, cfg_.fa_sel_chunks);
                } else if (sel_ver == 2) {
                    launch_sparse_global_select_v3(qv, sm_r, vs, dm,
                                                   cfg_.sparse_pages, buf_fa_done_ + 16, stream_);
                } else if (sel_ver == 0) {
                    launch_sparse_global_select_pages_decode(qv, sm_r, vs, dm,
                                                             cfg_.sparse_pages, stream_);
                } else {
                    launch_sparse_global_select_v2(qv, sm_r, vs, dm,
                                                   cfg_.sparse_pages, buf_fa_done_ + 16, stream_);
                }
                if (glob_ver == 0) {
                    launch_fa_sparse_global_decode(
                        qv, kv_r, sinkv, q_stride, relv,
                        nullptr, nullptr, gatev, q_stride, outv,
                        vs.selected_pages, vs.n_selected,
                        buf_fa_tmp_o_, buf_fa_tmp_m_,
                        (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : SPARSE_DECODE_SPLITS,
                        dm, stream_, buf_fa_done_);
                } else if (glob_ver == 2) {
                    launch_fa_sparse_global_decode_v3(
                        qv, kv_r, sinkv, q_stride, relv, gatev, q_stride, outv,
                        vs.selected_pages, vs.n_selected, buf_fa_tmp_o_, buf_fa_tmp_m_,
                        (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : FA_V3_GLOBAL_SPLITS,
                        dm, stream_, buf_fa_done_);
                } else if (glob_ver == 3) {
                    launch_fa_sparse_global_decode_v4(
                        qv, kv_r, sinkv, q_stride, relv, gatev, q_stride, outv,
                        vs.selected_pages, vs.n_selected, buf_fa_tmp_o_, buf_fa_tmp_m_,
                        (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : FA_V4_GLOBAL_SPLITS,
                        dm, stream_, buf_fa_done_, cfg_.fa_q_lds, cfg_.fa_cached);
                } else {
                    launch_fa_sparse_global_decode_v2(
                        qv, kv_r, sinkv, q_stride, relv, gatev, q_stride, outv,
                        vs.selected_pages, vs.n_selected, buf_fa_tmp_o_, buf_fa_tmp_m_,
                        (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : FA_V2_GLOBAL_SPLITS,
                        dm, stream_, buf_fa_done_);
                }
            } else if (is_swa) {
                const int swa_ver = (cfg_.fa_swa_v3 < 0) ? 3 : cfg_.fa_swa_v3;
                if (swa_ver == 0) {
                    launch_fa_relbias(qv, kv_r, sinkv, q_stride, relv,
                                      gatev, q_stride, outv, buf_fa_tmp_o_, buf_fa_tmp_m_,
                                      1, 0, dm, stream_, buf_fa_done_);
                } else if (swa_ver == 2) {
                    launch_fa_swa_decode_v3(qv, kv_r, sinkv, q_stride, relv,
                                            gatev, q_stride, outv, buf_fa_tmp_o_, buf_fa_tmp_m_,
                                            (cfg_.fa_swa_splits > 0) ? cfg_.fa_swa_splits : FA_V3_SWA_SPLITS,
                                            dm, stream_, buf_fa_done_);
                } else if (swa_ver == 3) {
                    launch_fa_swa_decode_v4(qv, kv_r, sinkv, q_stride, relv,
                                            gatev, q_stride, outv, buf_fa_tmp_o_, buf_fa_tmp_m_,
                                            (cfg_.fa_swa_splits > 0) ? cfg_.fa_swa_splits : FA_V4_SWA_SPLITS,
                                            dm, stream_, buf_fa_done_, cfg_.fa_q_lds, cfg_.fa_cached);
                } else {
                    launch_fa_swa_decode_v2(qv, kv_r, sinkv, q_stride, relv,
                                            gatev, q_stride, outv, buf_fa_tmp_o_, buf_fa_tmp_m_,
                                            (cfg_.fa_swa_splits > 0) ? cfg_.fa_swa_splits : FA_V2_SWA_SPLITS,
                                            dm, stream_, buf_fa_done_);
                }
            } else {
                launch_fa_relbias(qv, kv_r, sinkv, q_stride, relv,
                                  gatev, q_stride, outv, buf_fa_tmp_o_, buf_fa_tmp_m_,
                                  1, 0, dm, stream_, buf_fa_done_);
            }
        }
    } else if (verify_m_ > 0) {
        // Step 8B: the exact per-row decode FA (identical kernels and arithmetic
        // as a plain decode step) for each verify row.  Global Quest selection is
        // per row too (each row keeps its own top-K page list).
        for (int vr = 0; vr < n_tokens; ++vr) {
            const float * qv = q_prep_ + (size_t)vr * N_Q;
            const float * sinkv = (sinks != nullptr) ? (sinks + (size_t)vr * q_stride) : nullptr;
            const float * relv = rel_table_ + (size_t)vr * N_HEAD * N_REL_EXTENT;
            const float * gatev = (qgate != nullptr) ? (qgate + (size_t)vr * q_stride) : nullptr;
            float * outv = fa_out_ + (size_t)vr * N_Q;
            const StepMeta * dm = d_meta + vr;
            if (!is_swa && cfg_.sparse_pages > 0) {
                SparseGlobalScratch & vs = sparse_scratch_;
                const int sel_ver = (cfg_.fa_select_v3 < 0) ? 1 : cfg_.fa_select_v3;
                const int glob_ver = (cfg_.fa_global_v3 < 0) ? 3 : cfg_.fa_global_v3;
                if (vr == 0 && n_tokens > 1 && sel_ver == 1 && glob_ver == 3 &&
                    sparse_meta_[slot].fmt == SPARSE_META_Q8 &&
                    n_tokens <= SPARSE_MAX_QB) {
                    // Step 8D: 1x HBM metadata scan + parallel Top-K + 1-launch
                    // L2/L3-cached m-row Global sparse FA (bit-identical to per-row).
                    launch_sparse_global_select_v2_mrows(
                        q_prep_, sparse_meta_[slot], vs, d_meta,
                        cfg_.sparse_pages, stream_, n_tokens);
                    launch_fa_sparse_global_decode_v4_mrows(
                        q_prep_, kv_slots_[slot], sinks, q_stride, rel_table_,
                        qgate, q_stride, fa_out_,
                        vs.selected_pages, vs.n_selected,
                        buf_fa_tmp_o_, buf_fa_tmp_m_,
                        (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : FA_V4_GLOBAL_SPLITS,
                        d_meta, stream_, buf_fa_done_ + 32, cfg_.fa_q_lds,
                        cfg_.fa_cached, n_tokens);
                    break;
                }
                if (sel_ver == 3) {
                    launch_sparse_global_select_v4(qv, sparse_meta_[slot], vs, dm,
                                                   cfg_.sparse_pages, buf_fa_done_ + 16, stream_,
                                                   cfg_.fa_q_lds, cfg_.fa_cached, cfg_.fa_sel_chunks);
                } else if (sel_ver == 2) {
                    launch_sparse_global_select_v3(qv, sparse_meta_[slot], vs, dm,
                                                   cfg_.sparse_pages, buf_fa_done_ + 16, stream_);
                } else if (sel_ver == 0) {
                    launch_sparse_global_select_pages_decode(qv, sparse_meta_[slot], vs, dm,
                                                             cfg_.sparse_pages, stream_);
                } else {
                    launch_sparse_global_select_v2(qv, sparse_meta_[slot], vs, dm,
                                                   cfg_.sparse_pages, buf_fa_done_ + 16, stream_);
                }
                if (glob_ver == 0) {
                    launch_fa_sparse_global_decode(
                        qv, kv_slots_[slot], sinkv, q_stride, relv,
                        nullptr, nullptr, gatev, q_stride, outv,
                        vs.selected_pages, vs.n_selected,
                        buf_fa_tmp_o_, buf_fa_tmp_m_,
                        (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : SPARSE_DECODE_SPLITS,
                        dm, stream_, buf_fa_done_);
                } else if (glob_ver == 2) {
                    launch_fa_sparse_global_decode_v3(
                        qv, kv_slots_[slot], sinkv, q_stride, relv, gatev, q_stride, outv,
                        vs.selected_pages, vs.n_selected, buf_fa_tmp_o_, buf_fa_tmp_m_,
                        (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : FA_V3_GLOBAL_SPLITS,
                        dm, stream_, buf_fa_done_);
                } else if (glob_ver == 3) {
                    launch_fa_sparse_global_decode_v4(
                        qv, kv_slots_[slot], sinkv, q_stride, relv, gatev, q_stride, outv,
                        vs.selected_pages, vs.n_selected, buf_fa_tmp_o_, buf_fa_tmp_m_,
                        (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : FA_V4_GLOBAL_SPLITS,
                        dm, stream_, buf_fa_done_, cfg_.fa_q_lds, cfg_.fa_cached);
                } else {
                    launch_fa_sparse_global_decode_v2(
                        qv, kv_slots_[slot], sinkv, q_stride, relv, gatev, q_stride, outv,
                        vs.selected_pages, vs.n_selected, buf_fa_tmp_o_, buf_fa_tmp_m_,
                        (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : FA_V2_GLOBAL_SPLITS,
                        dm, stream_, buf_fa_done_);
                }
            } else if (is_swa) {
                const int swa_ver = (cfg_.fa_swa_v3 < 0) ? 3 : cfg_.fa_swa_v3;
                if (swa_ver == 3 && vr == 0) {
                    // Step 8C/8D: all m rows in ONE launch (per-row scratch slices).
                    launch_fa_swa_decode_v4_mrows(
                        q_prep_, kv_slots_[slot], sinks, q_stride, rel_table_,
                        qgate, q_stride, fa_out_, buf_fa_tmp_o_, buf_fa_tmp_m_,
                        (cfg_.fa_swa_splits > 0) ? cfg_.fa_swa_splits : FA_V4_SWA_SPLITS,
                        d_meta, stream_, buf_fa_done_ + 32, cfg_.fa_q_lds,
                        cfg_.fa_cached, n_tokens);
                    break;
                }
                if (swa_ver == 0) {
                    launch_fa_relbias(qv, kv_slots_[slot], sinkv, q_stride, relv,
                                      gatev, q_stride, outv, buf_fa_tmp_o_, buf_fa_tmp_m_,
                                      1, 0, dm, stream_, buf_fa_done_);
                } else if (swa_ver == 2) {
                    launch_fa_swa_decode_v3(qv, kv_slots_[slot], sinkv, q_stride, relv,
                                            gatev, q_stride, outv, buf_fa_tmp_o_, buf_fa_tmp_m_,
                                            (cfg_.fa_swa_splits > 0) ? cfg_.fa_swa_splits : FA_V3_SWA_SPLITS,
                                            dm, stream_, buf_fa_done_);
                } else if (swa_ver == 3) {
                    launch_fa_swa_decode_v4(qv, kv_slots_[slot], sinkv, q_stride, relv,
                                            gatev, q_stride, outv, buf_fa_tmp_o_, buf_fa_tmp_m_,
                                            (cfg_.fa_swa_splits > 0) ? cfg_.fa_swa_splits : FA_V4_SWA_SPLITS,
                                            dm, stream_, buf_fa_done_, cfg_.fa_q_lds, cfg_.fa_cached);
                } else {
                    launch_fa_swa_decode_v2(qv, kv_slots_[slot], sinkv, q_stride, relv,
                                            gatev, q_stride, outv, buf_fa_tmp_o_, buf_fa_tmp_m_,
                                            (cfg_.fa_swa_splits > 0) ? cfg_.fa_swa_splits : FA_V2_SWA_SPLITS,
                                            dm, stream_, buf_fa_done_);
                }
            } else {
                launch_fa_relbias(qv, kv_slots_[slot], sinkv, q_stride, relv,
                                  gatev, q_stride, outv, buf_fa_tmp_o_, buf_fa_tmp_m_,
                                  1, 0, dm, stream_, buf_fa_done_);
            }
        }
        if (vsync2) { HIP_CHECK(hipStreamSynchronize(stream_)); std::fprintf(stderr, "[v] slot=%d fa ok\n", slot); }
    } else if (!is_swa && cfg_.sparse_pages > 0) {
        if (n_tokens == 1 && d_meta != nullptr) {
            // Decode: 100% static-HIP-graph safe (every decision comes from d_meta).
            // Step 7I-1: skipped when the fused norm+prep already refreshed it.
            if (!minmax_folded) {
                launch_kv_update_page_minmax_decode(
                    cur_kv, cur_sm, d_meta, stream_);
            }
            const int sel_ver = (cfg_.fa_select_v3 < 0) ? 1 : cfg_.fa_select_v3;
            (void)cfg_.fa_sel_chunks;   // chunk-count override handled in the v3 header
            if (sel_ver == 0) {
                launch_sparse_global_select_pages_decode(
                    q_prep_, cur_sm, sparse_scratch_, d_meta,
                    cfg_.sparse_pages, stream_);
            } else if (sel_ver == 2) {
                launch_sparse_global_select_v3(
                    q_prep_, cur_sm, sparse_scratch_, d_meta,
                    cfg_.sparse_pages, buf_fa_done_ + 16, stream_);
            } else if (sel_ver == 3) {
                launch_sparse_global_select_v4(
                    q_prep_, cur_sm, sparse_scratch_, d_meta,
                    cfg_.sparse_pages, buf_fa_done_ + 16, stream_,
                    cfg_.fa_q_lds, cfg_.fa_cached, cfg_.fa_sel_chunks);
            } else {
                launch_sparse_global_select_v2(
                    q_prep_, cur_sm, sparse_scratch_, d_meta,
                    cfg_.sparse_pages, buf_fa_done_ + 16, stream_);
            }
            const int glob_ver = (cfg_.fa_global_v3 < 0) ? 3 : cfg_.fa_global_v3;
            if (glob_ver == 0) {
                launch_fa_sparse_global_decode(
                    q_prep_, cur_kv,
                    sinks, q_stride, rel_table_,
                    /*r_prep=*/nullptr, /*attn_rel_bias=*/nullptr,
                    qgate, q_stride, fa_out_,
                    sparse_scratch_.selected_pages, sparse_scratch_.n_selected,
                    buf_fa_tmp_o_, buf_fa_tmp_m_,
                    (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : SPARSE_DECODE_SPLITS,
                    d_meta, stream_, buf_fa_done_);
            } else if (glob_ver == 2) {
                launch_fa_sparse_global_decode_v3(
                    q_prep_, cur_kv,
                    sinks, q_stride, rel_table_,
                    qgate, q_stride, fa_out_,
                    sparse_scratch_.selected_pages, sparse_scratch_.n_selected,
                    buf_fa_tmp_o_, buf_fa_tmp_m_,
                    (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : FA_V3_GLOBAL_SPLITS,
                    d_meta, stream_, buf_fa_done_);
            } else if (glob_ver == 3) {
                launch_fa_sparse_global_decode_v4(
                    q_prep_, cur_kv,
                    sinks, q_stride, rel_table_,
                    qgate, q_stride, fa_out_,
                    sparse_scratch_.selected_pages, sparse_scratch_.n_selected,
                    buf_fa_tmp_o_, buf_fa_tmp_m_,
                    (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : FA_V4_GLOBAL_SPLITS,
                    d_meta, stream_, buf_fa_done_,
                    cfg_.fa_q_lds, cfg_.fa_cached);
            } else {
                launch_fa_sparse_global_decode_v2(
                    q_prep_, cur_kv,
                    sinks, q_stride, rel_table_,
                    qgate, q_stride, fa_out_,
                    sparse_scratch_.selected_pages, sparse_scratch_.n_selected,
                    buf_fa_tmp_o_, buf_fa_tmp_m_,
                    (cfg_.fa_glob_splits > 0) ? cfg_.fa_glob_splits : FA_V2_GLOBAL_SPLITS,
                    d_meta, stream_, buf_fa_done_);
            }
        } else {
            launch_kv_update_page_minmax_prefill(
                cur_kv, cur_sm, pos_start, n_tokens, stream_);
            const int kv_end = pos_start + n_tokens;
            if (kv_end <= SPARSE_DENSE_PREFILL_LIMIT) {
                launch_fa_relbias(
                    q_prep_, cur_kv,
                    sinks, q_stride, rel_table_, qgate, q_stride,
                    fa_out_, buf_fa_tmp_o_, buf_fa_tmp_m_,
                    n_tokens, pos_start, d_meta, stream_,
                    (n_tokens == 1) ? buf_fa_done_ : nullptr);
            } else {
                const int prefill_pages = std::max(cfg_.sparse_pages, SPARSE_PREFILL_MIN_PAGES);
                launch_sparse_global_select_pages_prefill(
                    q_prep_, cur_sm, sparse_scratch_,
                    n_tokens, pos_start, prefill_pages, stream_);
                launch_fa_sparse_global_prefill(
                    q_prep_, cur_kv,
                    sinks, q_stride, rel_table_,
                    /*r_prep=*/nullptr, /*attn_rel_bias=*/nullptr,
                    qgate, q_stride, fa_out_,
                    sparse_scratch_.selected_pages, sparse_scratch_.n_selected,
                    n_tokens, pos_start, kv_end, stream_);
            }
        }
    } else {
        if (is_swa && n_tokens == 1 && d_meta != nullptr) {
            const int swa_ver = (cfg_.fa_swa_v3 < 0) ? 3 : cfg_.fa_swa_v3;
            if (swa_ver == 0) {
                launch_fa_relbias(
                    q_prep_, cur_kv,
                    sinks, q_stride, rel_table_, qgate, q_stride,
                    fa_out_, buf_fa_tmp_o_, buf_fa_tmp_m_,
                    n_tokens, pos_start, d_meta, stream_, buf_fa_done_);
            } else if (swa_ver == 2) {
                launch_fa_swa_decode_v3(
                    q_prep_, cur_kv,
                    sinks, q_stride, rel_table_,
                    qgate, q_stride, fa_out_,
                    buf_fa_tmp_o_, buf_fa_tmp_m_,
                    (cfg_.fa_swa_splits > 0) ? cfg_.fa_swa_splits : FA_V3_SWA_SPLITS,
                    d_meta, stream_, buf_fa_done_);
            } else if (swa_ver == 3) {
                launch_fa_swa_decode_v4(
                    q_prep_, cur_kv,
                    sinks, q_stride, rel_table_,
                    qgate, q_stride, fa_out_,
                    buf_fa_tmp_o_, buf_fa_tmp_m_,
                    (cfg_.fa_swa_splits > 0) ? cfg_.fa_swa_splits : FA_V4_SWA_SPLITS,
                    d_meta, stream_, buf_fa_done_,
                    cfg_.fa_q_lds, cfg_.fa_cached);
            } else {
                launch_fa_swa_decode_v2(
                    q_prep_, cur_kv,
                    sinks, q_stride, rel_table_,
                    qgate, q_stride, fa_out_,
                    buf_fa_tmp_o_, buf_fa_tmp_m_,
                    (cfg_.fa_swa_splits > 0) ? cfg_.fa_swa_splits : FA_V2_SWA_SPLITS,
                    d_meta, stream_, buf_fa_done_);
            }
        } else {
            launch_fa_relbias(
                q_prep_, cur_kv,
                sinks, q_stride, rel_table_, qgate, q_stride,
                fa_out_, buf_fa_tmp_o_, buf_fa_tmp_m_,
                n_tokens, pos_start, d_meta, stream_,
                (n_tokens == 1) ? buf_fa_done_ : nullptr);
        }
    }

    if (stage_dump != nullptr && slot == stage_slot && stage_rows > 0) {
        HIP_CHECK(hipStreamSynchronize(stream_));
        if (FILE * fp = std::fopen(stage_dump, "ab")) {
            const int32_t hdr[2] = { stage_rows, cur_pos_ };
            std::fwrite(hdr, 4, 2, fp);
            std::fwrite("faout   ", 1, 8, fp);
            std::fwrite(fa_out_, 4, (size_t)stage_rows * N_Q, fp);
            std::fclose(fp);
        }
    }

    // Step 7G debug: dump this slot's raw FA output (host sync; use --no-graph).
    if (const char * dslot = std::getenv("INKLING_DUMP_FA_SLOT")) {
        const int want = std::atoi(dslot);
        const char * dpath = std::getenv("INKLING_DUMP_FA_PATH");
        if ((want < 0 || slot == want) && dpath != nullptr && n_tokens == 1) {
            std::vector<float> tmp((size_t)N_HEAD * HEAD_DIM);
            HIP_CHECK(hipMemcpyAsync(tmp.data(), fa_out_, tmp.size() * 4,
                                     hipMemcpyDeviceToHost, stream_));
            HIP_CHECK(hipStreamSynchronize(stream_));
            if (FILE * fp = std::fopen(dpath, "ab")) {
                std::fwrite(tmp.data(), 4, tmp.size(), fp);
                std::fclose(fp);
            }
        }
    }

    // 8. Attention output projection wo + fused attn_post_conv + residual handling
    const float * attn_res_add = nullptr;
    if (ip % ATTN_RES_BLOCK_SIZE == 0) {
        blocks.push_back(inpSA);
    } else {
        attn_res_add = inpSA;
    }
    // Step 7I-1: B/C depth states, built here so the fused wo launch sees them.
    const int b_n_states = (int)blocks.size() + 1;
    const float * b_states[MAX_RES_BLOCKS];
    {
        int idx = 0;
        for (int i = (int)blocks.size() - 1; i >= 0; --i) {
            b_states[idx++] = blocks[i];
        }
        b_states[idx++] = buf_ffn_inp_;
    }
    const bool s7i_wob = (n_tokens == 1 && messages == nullptr && cfg_.fuse_wob != 0 &&
                          cfg_.b_v3 == 0 && cfg_.mix_fast == 0 && cfg_.wo_v3 < 0);
    const bool s7i_woc = (n_tokens == 1 && messages != nullptr && loop != nullptr &&
                          cfg_.fuse_woc != 0 && cfg_.c_v3 == 0 && cfg_.wo_v3 < 0 &&
                          cfg_.fdot2_wo != 2);
    bool norm_router_done = false;
    // Step 8A: the verify path asks the fused epilogue to dump its raw pre-conv input.
    float * const wo_raw = ((verify_mode_ || verify_m_ > 0) && !batch_slots_active_)
        ? (d_verify_raw_ + ((size_t)slot * MAX_VERIFY_TOKENS + (verify_mode_ ? verify_row_ : 0)) * N_SCONV_CHANNELS + 2 * N_KV)
        : nullptr;

    if (verify_m_ > 0 && batch_slots_active_) {
        BatchHistPtrs wo_hists{};
        for (int r = 0; r < n_tokens; ++r) {
            wo_hists.p[r] = slot_sconv_hist(batch_slot_ids_[r], slot) + (size_t)(2 * N_KV) * N_CONV_HIST;
        }
        launch_batch_wo_mrows(
            L.wo, fa_out_, N_Q, L.attn_post_conv,
            wo_hists, attn_res_add, buf_ffn_inp_, n_tokens, stream_);
    } else if (verify_m_ > 0) {
        // Step 8C: ONE weight stream for all m rows + the exact causal
        // multi-token sconv_post_res epilogue (identical per-row math to the
        // fused decode epilogue); the raw pre-conv rows feed the rollback.
        // The wo GEMV is K = N_Q, whose decode kernel uses a 32-lane-per-row
        // reduction; the m-row kernel keeps that exact lane/block order, so the
        // verify logits stay bit-identical to a plain decode step while wo is
        // read once (7.1 MB) instead of once per row.
        launch_verify_wo_mrows(
            L.wo, fa_out_, N_Q, L.attn_post_conv,
            hist_base + (size_t)(2 * N_KV) * N_CONV_HIST,
            attn_res_add, buf_ffn_inp_, n_tokens, wo_raw, N_SCONV_CHANNELS, stream_);
        if (vsync2) { HIP_CHECK(hipStreamSynchronize(stream_)); std::fprintf(stderr, "[v] slot=%d wo ok\n", slot); }
    } else if (n_tokens == 1) {
        if (s7i_wob) {
            launch_wo_b_fused(
                L.wo, fa_out_, L.attn_post_conv,
                hist_base + (size_t)(2 * N_KV) * N_CONV_HIST,
                attn_res_add, buf_ffn_inp_,
                L.ffn_res_query, b_states, b_n_states, L.ffn_norm,
                L.ffn_gate_inp, L.ffn_exp_probs_b,
                x_norm_, topk_ids_, topk_w_, buf_mix_scratch_, stream_);
            if (const char * dp = std::getenv("INKLING_DUMP_XNORM")) {
                HIP_CHECK(hipStreamSynchronize(stream_));
                std::vector<float> tmp(N_EMBD);
                HIP_CHECK(hipMemcpy(tmp.data(), x_norm_, N_EMBD * sizeof(float), hipMemcpyDeviceToHost));
                int32_t tk[TOP_K];
                HIP_CHECK(hipMemcpy(tk, topk_ids_, sizeof(tk), hipMemcpyDeviceToHost));
                if (FILE * fp = std::fopen(dp, "ab")) {
                    std::fwrite(tmp.data(), 4, tmp.size(), fp);
                    std::fwrite(tk, 4, TOP_K, fp);
                    std::fclose(fp);
                }
            }
            norm_router_done = true;
        } else if (s7i_woc && cfg_.fdot2_wo == 1) {
            const float * msg_ffn = messages + (size_t)LOOP_CTRL_FFN * tok_embd_stride;
            launch_wo_c_fused_fdot2(
                L.wo, fa_out_, L.attn_post_conv,
                hist_base + (size_t)(2 * N_KV) * N_CONV_HIST,
                attn_res_add, buf_ffn_inp_,
                L.ffn_res_query, b_states, b_n_states, msg_ffn,
                loop->ctrl_down[LOOP_CTRL_FFN], loop->ctrl_gain[LOOP_CTRL_FFN],
                loop->ctrl_forget[LOOP_CTRL_FFN], L.ffn_norm,
                L.ffn_gate_inp, L.ffn_exp_probs_b,
                mixed_, ctrl_gate_, x_norm_,
                topk_ids_, topk_w_, buf_mix_scratch_, stream_);
            norm_router_done = true;
        } else if (s7i_woc) {
            const float * msg_ffn = messages + (size_t)LOOP_CTRL_FFN * tok_embd_stride;
            launch_wo_c_fused(
                L.wo, fa_out_, L.attn_post_conv,
                hist_base + (size_t)(2 * N_KV) * N_CONV_HIST,
                attn_res_add, buf_ffn_inp_,
                L.ffn_res_query, b_states, b_n_states, msg_ffn,
                loop->ctrl_down[LOOP_CTRL_FFN], loop->ctrl_gain[LOOP_CTRL_FFN],
                loop->ctrl_forget[LOOP_CTRL_FFN], L.ffn_norm,
                L.ffn_gate_inp, L.ffn_exp_probs_b,
                mixed_, ctrl_gate_, x_norm_,
                topk_ids_, topk_w_, buf_mix_scratch_, stream_, wo_raw);
            norm_router_done = true;
        } else if (cfg_.fdot2_wo >= 1) {
            launch_soa_q8_gemv_wo_fdot2(
                L.wo, fa_out_, L.attn_post_conv,
                hist_base + (size_t)(2 * N_KV) * N_CONV_HIST,
                attn_res_add, buf_ffn_inp_, stream_, cfg_.fdot2_wo_shape);
        } else if (cfg_.wo_v3 >= 0) {
            launch_soa_q8_gemv_wo_v3(
                L.wo, fa_out_, L.attn_post_conv,
                hist_base + (size_t)(2 * N_KV) * N_CONV_HIST,
                attn_res_add, buf_ffn_inp_, stream_, cfg_.wo_v3);
        } else if (FUSE_WO) {
            launch_soa_q8_gemv_wo_v2(
                L.wo, fa_out_, L.attn_post_conv,
                hist_base + (size_t)(2 * N_KV) * N_CONV_HIST,
                attn_res_add, buf_ffn_inp_, stream_, wo_raw);
        } else {
            launch_soa_q8_gemv_sconv_post_res(
                L.wo, fa_out_, L.attn_post_conv,
                hist_base + (size_t)(2 * N_KV) * N_CONV_HIST,
                attn_res_add, buf_ffn_inp_, stream_);
        }
    } else {
        run_proj(L.wo, fa_out_, wo_out_);
        launch_sconv_post_res(
            wo_out_, L.attn_post_conv,
            hist_base + (size_t)(2 * N_KV) * N_CONV_HIST,
            attn_res_add, buf_ffn_inp_, n_tokens, stream_);
    }

    if (stage_dump != nullptr && slot == stage_slot && stage_rows > 0) {
        HIP_CHECK(hipStreamSynchronize(stream_));
        if (FILE * fp = std::fopen(stage_dump, "ab")) {
            const int32_t hdr[2] = { stage_rows, cur_pos_ };
            std::fwrite(hdr, 4, 2, fp);
            std::fwrite("ffnin   ", 1, 8, fp);
            std::fwrite(buf_ffn_inp_, 4, (size_t)stage_rows * N_EMBD, fp);
            std::fclose(fp);
        }
    }

    // 9. MoE block: depth mix -> optional loop ctrl -> fused ffn_norm+router+topk -> experts -> ffn_post_conv+residual
    if (!norm_router_done) {
        const int n_states = (int)blocks.size() + 1;
        const float * states[MAX_RES_BLOCKS];
        int idx = 0;
        for (int i = (int)blocks.size() - 1; i >= 0; --i) {
            states[idx++] = blocks[i];
        }
        states[idx++] = buf_ffn_inp_;

        if (n_tokens == 1 && messages == nullptr) {
            if (cfg_.b_v3 > 0) {
                launch_mix_ffn_norm_router_topk_decode_v3(
                    L.ffn_res_query, states, n_states,
                    L.ffn_norm, L.ffn_gate_inp, L.ffn_exp_probs_b,
                    x_norm_, topk_ids_, topk_w_,
                    buf_router_scratch_, stream_);
            } else if (FUSE_B) {
                launch_mix_ffn_norm_router_topk_decode_v2(
                    L.ffn_res_query, states, n_states,
                    L.ffn_norm, L.ffn_gate_inp, L.ffn_exp_probs_b,
                    x_norm_, topk_ids_, topk_w_,
                    buf_router_scratch_, stream_);
                if (const char * dp = std::getenv("INKLING_DUMP_XNORM")) {
                    HIP_CHECK(hipStreamSynchronize(stream_));
                    std::vector<float> tmp(N_EMBD);
                    HIP_CHECK(hipMemcpy(tmp.data(), x_norm_, N_EMBD * sizeof(float), hipMemcpyDeviceToHost));
                    int32_t tk[TOP_K];
                    HIP_CHECK(hipMemcpy(tk, topk_ids_, sizeof(tk), hipMemcpyDeviceToHost));
                    if (FILE * fp = std::fopen(dp, "ab")) {
                        std::fwrite(tmp.data(), 4, tmp.size(), fp);
                        std::fwrite(tk, 4, TOP_K, fp);
                        std::fclose(fp);
                    }
                }
            } else {
                launch_mix_ffn_norm_router_topk_decode(
                    L.ffn_res_query, states, n_states,
                    L.ffn_norm, L.ffn_gate_inp, L.ffn_exp_probs_b,
                    x_norm_, topk_ids_, topk_w_,
                    buf_router_scratch_, stream_);
            }
        } else if (verify_m_ > 0 && messages != nullptr) {
            // Step 8C: exact fused C for all verify rows in ONE launch (grid.y = row).
            const float * msg_ffn = messages + (size_t)LOOP_CTRL_FFN * N_EMBD;
            launch_mix_ctrl_ffn_norm_router_topk_decode_v2_mrows(
                L.ffn_res_query, states, n_states,
                msg_ffn,
                loop->ctrl_down[LOOP_CTRL_FFN], loop->ctrl_gain[LOOP_CTRL_FFN],
                loop->ctrl_forget[LOOP_CTRL_FFN],
                L.ffn_norm, L.ffn_gate_inp, L.ffn_exp_probs_b,
                mixed_, ctrl_gate_,
                x_norm_, topk_ids_, topk_w_,
                buf_router_scratch_, n_tokens, stream_);
            norm_router_done = true;
        } else if (n_tokens == 1 && messages != nullptr) {
            const float * msg_ffn = messages + (size_t)LOOP_CTRL_FFN * tok_embd_stride;
            if (cfg_.c_v3 > 0) {
                launch_mix_ctrl_ffn_norm_router_topk_decode_v3(
                    L.ffn_res_query, states, n_states, msg_ffn,
                    loop->ctrl_down[LOOP_CTRL_FFN],
                    loop->ctrl_gain[LOOP_CTRL_FFN],
                    loop->ctrl_forget[LOOP_CTRL_FFN],
                    L.ffn_norm, L.ffn_gate_inp, L.ffn_exp_probs_b,
                    mixed_, ctrl_gate_,
                    x_norm_, topk_ids_, topk_w_,
                    buf_router_scratch_, stream_);
            } else if (FUSE_C) {
                launch_mix_ctrl_ffn_norm_router_topk_decode_v2(
                    L.ffn_res_query, states, n_states,
                    msg_ffn,
                    loop->ctrl_down[LOOP_CTRL_FFN],
                    loop->ctrl_gain[LOOP_CTRL_FFN],
                    loop->ctrl_forget[LOOP_CTRL_FFN],
                    L.ffn_norm, L.ffn_gate_inp, L.ffn_exp_probs_b,
                    mixed_, ctrl_gate_,
                    x_norm_, topk_ids_, topk_w_,
                    buf_router_scratch_, stream_);
            } else {
                launch_mix_ctrl_down_decode(
                    L.ffn_res_query, states, n_states,
                    msg_ffn, loop->ctrl_down[LOOP_CTRL_FFN],
                    mixed_, ctrl_gate_, stream_);
                launch_ctrl_apply_ffn_norm_router_topk_decode(
                    mixed_, msg_ffn, ctrl_gate_,
                    loop->ctrl_gain[LOOP_CTRL_FFN], loop->ctrl_forget[LOOP_CTRL_FFN],
                    L.ffn_norm, L.ffn_gate_inp, L.ffn_exp_probs_b,
                    x_norm_, topk_ids_, topk_w_,
                    buf_router_scratch_, stream_);
            }
        } else if (verify_m_ > 0) {
            // Step 8C: exact fused B for all verify rows in ONE launch (grid.y =
            // row); the m-row kernel is the single-token decode kernel verbatim.
            launch_mix_ffn_norm_router_topk_decode_v2_mrows(
                L.ffn_res_query, states, n_states,
                L.ffn_norm, L.ffn_gate_inp, L.ffn_exp_probs_b,
                x_norm_, topk_ids_, topk_w_,
                buf_router_scratch_, n_tokens, stream_);
            norm_router_done = true;
        } else {
            launch_inkling_mix(L.ffn_res_query, 1, states, n_states, mixed_, n_tokens, stream_);
            const float * ffn_x = mixed_;
            if (messages != nullptr) {
                const float * msg_ffn = messages + (size_t)LOOP_CTRL_FFN * tok_embd_stride;
                launch_inkling_ctrl(
                    ffn_x, msg_ffn,
                    loop->ctrl_down[LOOP_CTRL_FFN],
                    loop->ctrl_gain[LOOP_CTRL_FFN],
                    loop->ctrl_forget[LOOP_CTRL_FFN],
                    ctrl_gate_, buf_ctrl_[0], n_tokens, stream_);
                ffn_x = buf_ctrl_[0];
            }
            launch_ffn_norm_router_topk(
                ffn_x, L.ffn_norm, L.ffn_gate_inp, L.ffn_exp_probs_b,
                x_norm_, topk_ids_, topk_w_, n_tokens, stream_, buf_router_scratch_);
        }
    }

    if (const char * dp = std::getenv("INKLING_DUMP_ROUTER")) {
        const int rows = (verify_m_ > 0) ? verify_m_ : 1;
        HIP_CHECK(hipStreamSynchronize(stream_));
        std::vector<float> xn((size_t)rows * N_EMBD);
        std::vector<int32_t> ti((size_t)rows * TOP_K);
        std::vector<float> tw((size_t)rows * TOP_K);
        HIP_CHECK(hipMemcpy(xn.data(), x_norm_, xn.size() * 4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(ti.data(), topk_ids_, ti.size() * 4, hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(tw.data(), topk_w_, tw.size() * 4, hipMemcpyDeviceToHost));
        if (FILE * fp = std::fopen(dp, "ab")) {
            const int hdr[3] = { slot, rows, cur_pos_ };
            std::fwrite(hdr, 4, 3, fp);
            std::fwrite(xn.data(), 4, xn.size(), fp);
            std::fwrite(ti.data(), 4, ti.size(), fp);
            std::fwrite(tw.data(), 4, tw.size(), fp);
            std::fclose(fp);
        }
    }
    float * ffn_hist = hist_base + (size_t)(2 * N_KV + N_EMBD) * N_CONV_HIST;
    // Step 8A: raw pre-conv FFN input for the verify rollback (see wo_raw).
    float * const ffn_raw = ((verify_mode_ || verify_m_ > 0) && !batch_slots_active_)
        ? (d_verify_raw_ + ((size_t)slot * MAX_VERIFY_TOKENS + (verify_mode_ ? verify_row_ : 0)) * N_SCONV_CHANNELS + 2 * N_KV + N_EMBD)
        : nullptr;
    if (verify_m_ > 0 && !batch_slots_active_ && std::getenv("INKLING_VROW_MOE") != nullptr) {
        // A/B fallback: the exact shipped per-row fused decode MoE + post-conv.
        for (int vr = 0; vr < n_tokens; ++vr) {
            if (L.is_nvfp4) {
                launch_moe_nvfp4_ship_fused192(L.gate_up_nvfp4, L.down_nvfp4,
                                               x_norm_ + (size_t)vr * N_EMBD,
                                               topk_ids_ + (size_t)vr * TOP_K,
                                               topk_w_ + (size_t)vr * TOP_K,
                                               buf_verify_act_, cur_out + (size_t)vr * N_EMBD, stream_,
                                               L.ffn_post_conv, ffn_hist, buf_ffn_inp_ + (size_t)vr * N_EMBD,
                                               buf_router_scratch_,
                                               (ffn_raw != nullptr) ? (ffn_raw + (size_t)vr * N_SCONV_CHANNELS) : nullptr);
            } else {
                launch_moe_q8_decode(L.gate_up_q8, L.down_q8,
                                     x_norm_ + (size_t)vr * N_EMBD,
                                     topk_ids_ + (size_t)vr * TOP_K,
                                     topk_w_ + (size_t)vr * TOP_K,
                                     buf_verify_act_, cur_out + (size_t)vr * N_EMBD, stream_,
                                     L.ffn_post_conv, ffn_hist, buf_ffn_inp_ + (size_t)vr * N_EMBD,
                                     (ffn_raw != nullptr) ? (ffn_raw + (size_t)vr * N_SCONV_CHANNELS) : nullptr);
            }
        }
    } else if (verify_m_ > 0) {
        // Step 8B: unique-expert batched MoE.  The dedup table is shared by every
        // layer (recomputed per layer from that layer's own topk_ids/topk_w).
        launch_verify_moe_dedup(topk_ids_, topk_w_, n_tokens,
                                d_verify_unique_, d_verify_mask_, d_verify_slot_,
                                d_verify_w_, d_verify_cnt_, d_verify_list_,
                                d_num_unique_, stream_);
        if (L.is_nvfp4) {
            launch_verify_moe_gu_nvfp4(L.gate_up_nvfp4, x_norm_, N_EMBD,
                                       d_verify_unique_, d_verify_mask_, d_verify_slot_,
                                       d_verify_w_, d_verify_cnt_, d_verify_list_,
                                       d_num_unique_, n_tokens,
                                       buf_verify_act_, stream_);
            if (std::getenv("INKLING_MOE_CHECK") != nullptr && slot == 0) {
                HIP_CHECK(hipStreamSynchronize(stream_));
                launch_moe_nvfp4_gate_up_ship(L.gate_up_nvfp4, x_norm_, topk_ids_,
                                              buf_verify_act_ + 8 * N_FF_EXP, stream_);
                HIP_CHECK(hipStreamSynchronize(stream_));
                std::vector<float> a((size_t)TOP_K * N_FF_EXP), b((size_t)TOP_K * N_FF_EXP);
                HIP_CHECK(hipMemcpy(a.data(), buf_verify_act_, a.size() * 4, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(b.data(), buf_verify_act_ + 8 * N_FF_EXP, b.size() * 4, hipMemcpyDeviceToHost));
                double md = 0.0; int wi = -1;
                for (size_t i = 0; i < a.size(); ++i) {
                    const double d = std::fabs((double)a[i] - (double)b[i]);
                    if (d > md) { md = d; wi = (int)i; }
                }
                std::printf("[moe-check] act max |d| = %.6g at %d (slot %d row %d) mine=%.6f ref=%.6f\n",
                            md, wi, wi / N_FF_EXP, wi % N_FF_EXP, a[(size_t)wi], b[(size_t)wi]);
                std::fflush(stdout);
            }
            launch_verify_moe_dn_nvfp4(L.down_nvfp4, buf_verify_act_,
                                       d_verify_unique_, d_verify_mask_, d_verify_slot_,
                                       d_verify_w_, d_verify_cnt_, d_verify_list_,
                                       d_num_unique_, n_tokens,
                                       buf_moe_slot_, stream_);
            if (std::getenv("INKLING_MOE_CHECK") != nullptr && slot == 0) {
                HIP_CHECK(hipStreamSynchronize(stream_));
                launch_moe_nvfp4_down_ship(L.down_nvfp4, buf_verify_act_, topk_ids_, topk_w_,
                                           buf_moe_out_, stream_, nullptr, nullptr, nullptr);
                HIP_CHECK(hipStreamSynchronize(stream_));
                std::vector<float> sl((size_t)TOP_K * N_EMBD), ref((size_t)N_EMBD);
                HIP_CHECK(hipMemcpy(sl.data(), buf_moe_slot_, sl.size() * 4, hipMemcpyDeviceToHost));
                HIP_CHECK(hipMemcpy(ref.data(), buf_moe_out_, ref.size() * 4, hipMemcpyDeviceToHost));
                double md = 0.0; int wi = -1;
                for (int i = 0; i < N_EMBD; ++i) {
                    const float s0 = sl[(size_t)0 * N_EMBD + i], s1 = sl[(size_t)1 * N_EMBD + i];
                    const float s2 = sl[(size_t)2 * N_EMBD + i], s3 = sl[(size_t)3 * N_EMBD + i];
                    const float s4 = sl[(size_t)4 * N_EMBD + i], s5 = sl[(size_t)5 * N_EMBD + i];
                    const float s6 = sl[(size_t)6 * N_EMBD + i], s7 = sl[(size_t)7 * N_EMBD + i];
                    const double d = std::fabs((double)(((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7))) - (double)ref[(size_t)i]);
                    if (d > md) { md = d; wi = i; }
                }
                std::printf("[moe-check] down sum max |d| = %.6g at %d (ref=%.6f)\n", md, wi,
                            wi >= 0 ? ref[(size_t)wi] : 0.0f);
                for (int k = 0; k < TOP_K; ++k) {
                    std::printf("  slot %d: mine=%.6f\n", k,
                                sl[(size_t)k * N_EMBD + (wi < 0 ? 0 : wi)]);
                }
                std::fflush(stdout);
            }
        } else {
            launch_verify_moe_gu_q8(L.gate_up_q8, x_norm_, N_EMBD,
                                    d_verify_unique_, d_verify_mask_, d_verify_slot_,
                                    d_verify_w_, d_verify_cnt_, d_verify_list_,
                                    d_num_unique_, n_tokens,
                                    buf_verify_act_, stream_);
            launch_verify_moe_dn_q8(L.down_q8, buf_verify_act_,
                                    d_verify_unique_, d_verify_mask_, d_verify_slot_,
                                    d_verify_w_, d_verify_cnt_, d_verify_list_,
                                    d_num_unique_, n_tokens,
                                    buf_moe_slot_, stream_);
        }
        if (batch_slots_active_) {
            BatchHistPtrs ffn_hists{};
            for (int r = 0; r < n_tokens; ++r) {
                ffn_hists.p[r] = slot_sconv_hist(batch_slot_ids_[r], slot) + (size_t)(2 * N_KV + N_EMBD) * N_CONV_HIST;
            }
            launch_batch_ffn_post(buf_moe_slot_, n_tokens, L.ffn_post_conv, ffn_hists,
                                  buf_ffn_inp_, cur_out, stream_);
        } else {
            launch_verify_ffn_post(buf_moe_slot_, n_tokens, L.ffn_post_conv, ffn_hist,
                                   buf_ffn_inp_, cur_out, ffn_raw, N_SCONV_CHANNELS, stream_);
        }
        if (vsync2) { HIP_CHECK(hipStreamSynchronize(stream_)); std::fprintf(stderr, "[v] slot=%d moe ok\n", slot); }
    } else if (n_tokens == 1) {
        // Step 7F: per-stage v3 A/B switches.  Each of gate_up / down is
        // independently selectable (so a single stage can be measured against
        // the shipped kernel with everything else fixed); the fused path
        // replaces both stages plus one graph node per MoE layer.
        if (L.is_nvfp4) {
            if (cfg_.fdot2_moe == 4) {
                launch_moe_nvfp4_fdot2_split(
                    L.gate_up_nvfp4, L.down_nvfp4, x_norm_,
                    topk_ids_, topk_w_, buf_moe_act_dec_, cur_out, stream_,
                    L.ffn_post_conv, ffn_hist, buf_ffn_inp_);
            } else if (cfg_.fdot2_moe == 2) {
                launch_moe_nvfp4_fdot2_fused288(
                    L.gate_up_nvfp4, L.down_nvfp4, x_norm_,
                    topk_ids_, topk_w_, buf_moe_act_dec_, cur_out, stream_,
                    L.ffn_post_conv, ffn_hist, buf_ffn_inp_, buf_router_scratch_);
            } else if (cfg_.fdot2_moe == 1) {
                launch_moe_nvfp4_fdot2_fused(
                    L.gate_up_nvfp4, L.down_nvfp4, x_norm_,
                    topk_ids_, topk_w_, buf_moe_act_dec_, cur_out, stream_,
                    L.ffn_post_conv, ffn_hist, buf_ffn_inp_, buf_router_scratch_);
            } else if (cfg_.moe_ship_fused == 1) {
                launch_moe_nvfp4_ship_fused(
                    L.gate_up_nvfp4, L.down_nvfp4, x_norm_,
                    topk_ids_, topk_w_, buf_moe_act_dec_, cur_out, stream_,
                    L.ffn_post_conv, ffn_hist, buf_ffn_inp_, buf_router_scratch_, ffn_raw);
            } else if (cfg_.moe_ship_fused == 2) {
                launch_moe_nvfp4_ship_fused192(
                    L.gate_up_nvfp4, L.down_nvfp4, x_norm_,
                    topk_ids_, topk_w_, buf_moe_act_dec_, cur_out, stream_,
                    L.ffn_post_conv, ffn_hist, buf_ffn_inp_, buf_router_scratch_, ffn_raw);
            } else if (cfg_.moe_fused_v3 >= 0) {
                launch_moe_nvfp4_fused_v3(
                    L.gate_up_nvfp4, L.down_nvfp4, x_norm_,
                    topk_ids_, topk_w_, buf_moe_act_dec_, cur_out, stream_,
                    L.ffn_post_conv, ffn_hist, buf_ffn_inp_,
                    buf_router_scratch_, cfg_.moe_fused_v3);
            } else if (cfg_.moe_gu_v3 >= 0 || cfg_.moe_dn_v3 >= 0) {
                if (cfg_.moe_gu_v3 >= 0) {
                    launch_moe_nvfp4_gate_up_v3(
                        L.gate_up_nvfp4, x_norm_, topk_ids_,
                        buf_moe_act_dec_, stream_, cfg_.moe_gu_v3);
                } else {
                    launch_moe_nvfp4_gate_up_ship(
                        L.gate_up_nvfp4, x_norm_, topk_ids_,
                        buf_moe_act_dec_, stream_);
                }
                if (cfg_.moe_dn_v3 >= 0) {
                    launch_moe_nvfp4_down_v3(
                        L.down_nvfp4, buf_moe_act_dec_, topk_ids_, topk_w_,
                        cur_out, stream_, L.ffn_post_conv, ffn_hist, buf_ffn_inp_,
                        cfg_.moe_dn_v3);
                } else {
                    launch_moe_nvfp4_down_ship(
                        L.down_nvfp4, buf_moe_act_dec_, topk_ids_, topk_w_,
                        cur_out, stream_, L.ffn_post_conv, ffn_hist, buf_ffn_inp_);
                }
            } else if (FUSE_MOE) {
                launch_moe_nvfp4_decode_v2(
                    L.gate_up_nvfp4, L.down_nvfp4, x_norm_,
                    topk_ids_, topk_w_, buf_moe_act_dec_, cur_out, stream_,
                    L.ffn_post_conv, ffn_hist, buf_ffn_inp_);
            } else {
                launch_moe_nvfp4_decode(
                    L.gate_up_nvfp4, L.down_nvfp4, x_norm_,
                    topk_ids_, topk_w_, buf_moe_act_dec_, cur_out, stream_,
                    L.ffn_post_conv, ffn_hist, buf_ffn_inp_);
            }
        } else {
            if (cfg_.fdot2_q8moe == 1) {
                launch_moe_q8_fdot2_fused(
                    L.gate_up_q8, L.down_q8, x_norm_,
                    topk_ids_, topk_w_, buf_moe_act_dec_, cur_out, stream_,
                    L.ffn_post_conv, ffn_hist, buf_ffn_inp_, buf_router_scratch_);
            } else if (cfg_.moe_q8_fused == 1) {
                launch_moe_q8_ship_fused(
                    L.gate_up_q8, L.down_q8, x_norm_,
                    topk_ids_, topk_w_, buf_moe_act_dec_, cur_out, stream_,
                    L.ffn_post_conv, ffn_hist, buf_ffn_inp_, buf_router_scratch_);
            } else if (cfg_.moe_fused_v3 >= 0) {
                launch_moe_q8_fused_v3(
                    L.gate_up_q8, L.down_q8, x_norm_,
                    topk_ids_, topk_w_, buf_moe_act_dec_, cur_out, stream_,
                    L.ffn_post_conv, ffn_hist, buf_ffn_inp_,
                    buf_router_scratch_, cfg_.moe_fused_v3);
            } else if (cfg_.moe_gu_v3 >= 0 || cfg_.moe_dn_v3 >= 0) {
                // SHIP/SHIP3 are NVFP4-only gate_up variants; the Q8 stage keeps
                // the shipped kernel so a NVFP4 gate_up A/B leaves Q8 untouched.
                if (cfg_.moe_gu_v3 >= 0 && cfg_.moe_gu_v3 <= MOE_V3_CACHE) {
                    launch_moe_q8_gate_up_v3(
                        L.gate_up_q8, x_norm_, topk_ids_,
                        buf_moe_act_dec_, stream_, cfg_.moe_gu_v3);
                } else {
                    launch_moe_q8_gate_up_ship(
                        L.gate_up_q8, x_norm_, topk_ids_,
                        buf_moe_act_dec_, stream_);
                }
                if (cfg_.moe_dn_v3 >= 0) {
                    launch_moe_q8_down_v3(
                        L.down_q8, buf_moe_act_dec_, topk_ids_, topk_w_,
                        cur_out, stream_, L.ffn_post_conv, ffn_hist, buf_ffn_inp_,
                        cfg_.moe_dn_v3);
                } else {
                    launch_moe_q8_down_ship(
                        L.down_q8, buf_moe_act_dec_, topk_ids_, topk_w_,
                        cur_out, stream_, L.ffn_post_conv, ffn_hist, buf_ffn_inp_);
                }
            } else {
                launch_moe_q8_decode(
                    L.gate_up_q8, L.down_q8, x_norm_,
                    topk_ids_, topk_w_, buf_moe_act_dec_, cur_out, stream_,
                    L.ffn_post_conv, ffn_hist, buf_ffn_inp_, ffn_raw);
            }
        }
    } else {
        if (L.is_nvfp4) {
            launch_moe_nvfp4_prefill_wmma(
                L.gate_up_nvfp4, L.down_nvfp4, x_norm_,
                topk_ids_, topk_w_, moe_scratch_, cur_out, n_tokens, stream_,
                L.ffn_post_conv, ffn_hist, buf_ffn_inp_);
        } else {
            launch_moe_q8_prefill_wmma(
                L.gate_up_q8, L.down_q8, x_norm_,
                topk_ids_, topk_w_, moe_scratch_, cur_out, n_tokens, stream_,
                L.ffn_post_conv, ffn_hist, buf_ffn_inp_);
        }
    }
}

void InklingEngine::record_forward_pass(
        int n_tokens, int pos_start, const StepMeta * d_meta, bool compute_all_logits,
        const float * d_raw_embeds, int row_slot, bool skip_head) {
    // Step 8A: `row_slot` shifts every per-token activation buffer by that row of
    // [MAX_UBATCH, ...] so `m` speculative verify rows can keep their own state
    // while every launch is the exact single-token decode kernel (row_slot == 0
    // in the non-MTP path: every offset below is a no-op).
    const size_t o_embd = (size_t)row_slot * N_EMBD;
    const size_t o_mix  = (size_t)row_slot * 4 * N_EMBD;
    // 1. Input embedding -> buf_blk0_ (preserved because il=0 pushes it to blocks!)
    // IMPORTANT: raw image embeddings MUST bypass tok_norm (per llama.cpp/src/models/inkling2.cpp).
    if (d_raw_embeds != nullptr) {
        HIP_CHECK(hipMemcpyAsync(
            buf_blk0_ + o_embd, d_raw_embeds,
            (size_t)n_tokens * N_EMBD * sizeof(float),
            hipMemcpyDeviceToDevice, stream_));
    } else {
        // Step 8B: the batched verify gathers every row's token id into
        // d_token_ids_ (d_meta->token_id is only for the single-token path).
        launch_embed_norm(weights_.tok_embd, weights_.tok_norm, d_token_ids_,
                          (verify_m_ > 0) ? nullptr : d_meta, buf_blk0_ + o_embd, n_tokens, stream_);
    }

    const bool dbg_sync = std::getenv("INKLING_DEBUG_SYNC") != nullptr && !graph_capturing_;

    std::vector<const float *> blocks;
    blocks.reserve(8);

    // Prelude layers 0..5:
    //   il=0 reads buf_blk0_ (pushed to blocks), writes buf_a_
    //   il=1..4 ping-pong buf_a_ <-> buf_b_
    //   il=5 writes buf_prelude_ (preserved across all 3 loop passes!)
    const float * cur = buf_blk0_ + o_embd;
    for (int il = 0; il < LOOP_START; ++il) {
        float * dst = (il == LOOP_START - 1) ? (buf_prelude_ + o_embd)
                    : ((cur == buf_a_ + o_embd) ? (buf_b_ + o_embd) : (buf_a_ + o_embd));
        record_layer(il, il, cur, dst, blocks, nullptr, nullptr, n_tokens, pos_start, d_meta,
                     /*attn_already_mixed=*/false, nullptr, 0, nullptr, 0, row_slot);
        cur = dst;
        if (dbg_sync) { HIP_CHECK(hipStreamSynchronize(stream_)); std::fprintf(stderr, "[dbg] prelude il=%d ok\n", il); }
    }

    const float * prelude = cur; // == buf_prelude_
    const auto prelude_blocks = blocks; // [buf_blk0_]

    // 3 Looped Passes over physical layers 6..17
    for (int pass = 0; pass < LOOP_COUNT; ++pass) {
        std::vector<const float *> bank;
        const float * bank_out = nullptr;
        if (pass > 0) {
            bank.assign(blocks.begin() + prelude_blocks.size() + 1, blocks.end());
            bank_out = cur;
            cur = prelude;
            blocks = prelude_blocks;
        }

        for (int ip = LOOP_START; ip < LOOP_END; ++ip) {
            const int slot = (pass == 0) ? ip : (N_LAYER_PHYS + (pass - 1) * LOOP_CORE_LEN + (ip - LOOP_START));

            // Select non-aliasing destination buffer:
            //   - When ip == 6:  cur_in will be pushed to blocks! For pass 0..2, cur_in is buf_prelude_ (safe!),
            //     wait: at ip==6, inpSA == buf_prelude_, which is never overwritten!
            //   - When ip == 11: write into buf_pass_b12_[pass] so at ip==12 inpSA is preserved in buf_pass_b12_[pass]!
            //   - When ip == 17: for pass 0,1 write into buf_pass_out_[pass] (preserved as bank_out for next pass!);
            //     for pass 2 write into buf_b18_ (so at il==18 inpSA is preserved in buf_b18_!).
            float * dst = nullptr;
            if (ip == 11) {
                dst = buf_pass_b12_[pass] + o_embd;
            } else if (ip == LOOP_END - 1) {
                dst = (pass < LOOP_COUNT - 1) ? (buf_pass_out_[pass] + o_embd) : (buf_b18_ + o_embd);
            } else {
                dst = (cur == buf_a_ + o_embd) ? (buf_b_ + o_embd) : (buf_a_ + o_embd);
            }

            if (pass == 0) {
                record_layer(slot, ip, cur, dst, blocks, nullptr, nullptr, n_tokens, pos_start, d_meta,
                             /*attn_already_mixed=*/false, nullptr, 0, nullptr, 0, row_slot);
            } else {
                const auto & loop = weights_.loop[ip - LOOP_START];
                const auto & L    = weights_.layers[ip];
                // Compute cross-pass messages AND attn_mix in ONE parallel launch!
                const int n_msg_states = (int)bank.size() + 1;
                const float * msg_states[MAX_RES_BLOCKS];
                int idx = 0;
                for (int i = (int)bank.size() - 1; i >= 0; --i) {
                    msg_states[idx++] = bank[i];
                }
                msg_states[idx++] = bank_out;

                if (L.attn_res_query != nullptr) {
                    const int n_attn_q = L.is_swa ? 1 : 3;
                    const float * w_norm3[3] = {
                        L.is_swa ? L.attn_norm : L.attn_q_inp_norm,
                        L.is_swa ? L.attn_norm : L.attn_k_inp_norm,
                        L.is_swa ? L.attn_norm : L.attn_v_inp_norm
                    };

                    const int n_cur_states = (int)blocks.size() + 1;
                    const float * cur_states[MAX_RES_BLOCKS];
                    int cidx = 0;
                    for (int i = (int)blocks.size() - 1; i >= 0; --i) {
                        cur_states[cidx++] = blocks[i];
                    }
                    cur_states[cidx++] = cur;
                    if (FUSE_A && n_tokens == 1 && d_meta != nullptr) {
                        // A launches the dual mix itself; the split mix_dual is skipped.
                        record_layer(slot, ip, cur, dst, blocks, buf_messages_ + o_mix, &loop,
                                     n_tokens, pos_start, d_meta, /*attn_already_mixed=*/false,
                                     msg_states, n_msg_states, cur_states, n_cur_states, row_slot);
                    } else if (verify_m_ > 0) {
                        // Step 8C: ONE m-row launch of the exact fused A (grid.y =
                        // row); the ctrl step inside record_layer is then skipped
                        // (attn_already_mixed = true).
                        float * on3[3] = { buf_norm_[0], buf_norm_[1], buf_norm_[2] };
                        launch_mix_dual_ctrl_qkv3_norm_decode_mrows(
                            loop.msg_query, LOOP_CTRL_COUNT, msg_states, n_msg_states,
                            buf_verify_msg_,
                            L.attn_res_query, n_attn_q, cur_states, n_cur_states,
                            buf_verify_mixed_,
                            loop.ctrl_down, loop.ctrl_gain, loop.ctrl_forget, w_norm3,
                            buf_verify_gate_, on3,
                            RMS_EPS, stream_, buf_router_scratch_, n_tokens);
                        record_layer(slot, ip, cur, dst, blocks, buf_verify_msg_, &loop, n_tokens, pos_start, d_meta,
                                     /*attn_already_mixed=*/true, nullptr, 0, nullptr, 0, row_slot);
                    } else {
                        launch_inkling_mix_dual(
                            loop.msg_query, LOOP_CTRL_COUNT, msg_states, n_msg_states, buf_messages_ + o_mix,
                            L.attn_res_query, n_attn_q, cur_states, n_cur_states, buf_mixed_ + o_mix,
                            n_tokens, stream_);
                        record_layer(slot, ip, cur, dst, blocks, buf_messages_ + o_mix, &loop, n_tokens, pos_start, d_meta, /*attn_already_mixed=*/true,
                                     nullptr, 0, nullptr, 0, row_slot);
                    }
                } else {
                    launch_inkling_mix(loop.msg_query, LOOP_CTRL_COUNT, msg_states, n_msg_states, buf_messages_ + o_mix, n_tokens, stream_);
                    record_layer(slot, ip, cur, dst, blocks, buf_messages_ + o_mix, &loop, n_tokens, pos_start, d_meta, /*attn_already_mixed=*/false,
                                 nullptr, 0, nullptr, 0, row_slot);
                }
            }
            cur = dst;
            if (dbg_sync) { HIP_CHECK(hipStreamSynchronize(stream_)); std::fprintf(stderr, "[dbg] pass=%d ip=%d slot=%d ok\n", pass, ip, slot); }
        }
    }

    // Post-loop layers 18..23 (at il=18, cur == buf_b18_ which is pushed to blocks and never overwritten!)
    for (int il = LOOP_END; il < N_LAYER_PHYS; ++il) {
        float * dst = (cur == buf_a_ + o_embd) ? (buf_b_ + o_embd) : (buf_a_ + o_embd);
        record_layer(il, il, cur, dst, blocks, nullptr, nullptr, n_tokens, pos_start, d_meta,
                     /*attn_already_mixed=*/false, nullptr, 0, nullptr, 0, row_slot);
        cur = dst;
        if (dbg_sync) { HIP_CHECK(hipStreamSynchronize(stream_)); std::fprintf(stderr, "[dbg] post il=%d ok\n", il); }
    }

    // Final output mix + RMSNorm + lm_head + softcap
    {
        const int n_states = (int)blocks.size() + 1;
        const float * states[MAX_RES_BLOCKS];
        int idx = 0;
        for (int i = (int)blocks.size() - 1; i >= 0; --i) {
            states[idx++] = blocks[i];
        }
        states[idx++] = cur;
        const float * out_norms[1] = { weights_.output_norm };
        if (cfg_.mix_fast > 0 && n_tokens == 1) {
            launch_mix_rms_norm_2304_fast(weights_.output_res_q, 1, states, n_states, out_norms,
                                          buf_x_norm_ + o_embd, RMS_EPS, stream_);
        } else {
            launch_inkling_mix_rms_norm(weights_.output_res_q, 1, states, n_states, out_norms, buf_x_norm_ + o_embd, n_tokens, RMS_EPS, stream_);
        }
    }

    // Step 8A: tap 3 = the final normed hidden (right before lm_head).
    if (d_taps_ != nullptr && mtp_io_ && !batch_slots_active_ && active_slot_ == 0) {
        launch_mtp_capture_tap(buf_x_norm_ + o_embd, d_taps_ + (size_t)row_slot * 4 * N_EMBD, n_tokens, /*tap=*/3, stream_);
    }

    if (skip_head) {
        return; // speculative verify finishes all rows with one batched lm_head
    }

    if (n_tokens == 1) {
        if (cfg_.fdot2_lm > 0 && d_meta != nullptr) {
            launch_soa_q8_gemv_fdot2(weights_.lm_head, buf_x_norm_ + o_embd, buf_logits_, stream_, 0);
        } else {
            launch_soa_q8_gemv(weights_.lm_head, buf_x_norm_ + o_embd, buf_logits_, stream_);
        }
        if (d_meta != nullptr) {
            // In-graph deterministic meta advance: lets N captured-graph executions chain
            // back-to-back with zero host round-trips (single sync for the whole batch).
            launch_logit_softcap_argmax_advance(
                buf_logits_, VOCAB_UNPADDED, LOGIT_SOFTCAP, d_argmax_id_, buf_router_scratch_,
                const_cast<StepMeta *>(d_meta), d_tau_tab_, cfg_.swa_cap,
                d_tok_hist_, tok_hist_cap_, stream_);
        } else {
            launch_logit_softcap_argmax(buf_logits_, VOCAB_UNPADDED, LOGIT_SOFTCAP, d_argmax_id_, buf_router_scratch_, stream_);
        }
    } else if (compute_all_logits) {
        launch_soa_q8_gemm_wmma(weights_.lm_head, buf_x_norm_ + o_embd, buf_logits_, n_tokens, stream_);
        launch_logit_softcap(buf_logits_, n_tokens, VOCAB_UNPADDED, LOGIT_SOFTCAP, stream_);
    } else {
        // Only compute logits for the last token in the prefill chunk!
        const float * last_x = buf_x_norm_ + o_embd + (size_t)(n_tokens - 1) * N_EMBD;
        launch_soa_q8_gemv(weights_.lm_head, last_x, buf_logits_, stream_);
        launch_logit_softcap_argmax(buf_logits_, VOCAB_UNPADDED, LOGIT_SOFTCAP, d_argmax_id_, buf_router_scratch_, stream_);
    }
}


static inline StepMeta make_step_meta(int32_t token_id, int32_t pos, int32_t swa_cap);

// ============================================================================
// Step 8A: DSpark MTP support — tap capture, speculative verify, exact rollback
// ============================================================================
bool InklingEngine::enable_mtp_io() {
    if (mtp_io_) return true;
    d_taps_        = alloc_gpu<uint16_t>((size_t)MAX_UBATCH * 4 * N_EMBD,           state_allocs_, total_vram_bytes_, stream_);
    d_verify_raw_  = alloc_gpu<float>((size_t)N_LAYER_SLOTS * MAX_VERIFY_TOKENS * N_SCONV_CHANNELS,
                                      state_allocs_, total_vram_bytes_, stream_);
    d_hist_bak_    = alloc_gpu<float>((size_t)N_LAYER_SLOTS * N_SCONV_CHANNELS * N_CONV_HIST,
                                      state_allocs_, total_vram_bytes_, stream_);
    float ** d_hist_ptrs = alloc_gpu<float *>(N_LAYER_SLOTS, state_allocs_, total_vram_bytes_, stream_);
    d_hist_ptrs_ = d_hist_ptrs;
    {
        std::vector<float *> hp((size_t)N_LAYER_SLOTS);
        for (int s = 0; s < N_LAYER_SLOTS; ++s) hp[(size_t)s] = sconv_slots_[s].hist;
        HIP_CHECK(hipMemcpy(d_hist_ptrs, hp.data(), hp.size() * sizeof(float *), hipMemcpyHostToDevice));
    }
    d_verify_topk_vals_ = alloc_gpu<float>((size_t)MAX_VERIFY_TOKENS * MTP_VERIFY_TOPK, state_allocs_, total_vram_bytes_, stream_);
    d_verify_topk_ids_  = alloc_gpu<int32_t>((size_t)MAX_VERIFY_TOKENS * MTP_VERIFY_TOPK, state_allocs_, total_vram_bytes_, stream_);
    d_verify_stats_     = alloc_gpu<float>((size_t)MAX_VERIFY_TOKENS * 2, state_allocs_, total_vram_bytes_, stream_);
    HIP_CHECK(hipHostMalloc(&h_verify_topk_vals_, (size_t)MAX_VERIFY_TOKENS * MTP_VERIFY_TOPK * sizeof(float)));
    HIP_CHECK(hipHostMalloc(&h_verify_topk_ids_,  (size_t)MAX_VERIFY_TOKENS * MTP_VERIFY_TOPK * sizeof(int32_t)));
    HIP_CHECK(hipHostMalloc(&h_verify_stats_,     (size_t)MAX_VERIFY_TOKENS * 2 * sizeof(float)));
    // Shared batched-verify / multi-slot decode scratch (idempotent if n_slots > 1 already allocated it)
    alloc_batch_scratch();
    HIP_CHECK(hipStreamSynchronize(stream_));
    mtp_io_ = true;
    std::printf("[InklingEngine] MTP I/O enabled: taps=%.1f MiB verify_raw=%.1f MiB hist_bak=%.1f MiB\n",
                (double)((size_t)MAX_UBATCH * 4 * N_EMBD * 2) / 1048576.0,
                (double)((size_t)N_LAYER_SLOTS * MAX_VERIFY_TOKENS * N_SCONV_CHANNELS * 4) / 1048576.0,
                (double)((size_t)N_LAYER_SLOTS * N_SCONV_CHANNELS * N_CONV_HIST * 4) / 1048576.0);
    std::fflush(stdout);
    return true;
}

void InklingEngine::set_tap_hook(std::function<void(const uint16_t *, const int32_t *, int, int)> hook) {
    tap_hook_ = std::move(hook);
}

void InklingEngine::clear_tap_hook() { tap_hook_ = nullptr; }

// ============================================================================
// Step 8B: m-token batched verify.  ONE weight stream per matrix / expert for
// all m rows, exact per-row arithmetic (same kernels or bit-identical copies of
// the decode ones), all state (positions, KV writes, shortconv) driven by
// d_verify_meta_[0..m-1] so the whole pass is a static HIP graph.
// ============================================================================
void InklingEngine::record_verify_pass(int m) {
    const bool vsync = std::getenv("INKLING_VERIFY_SYNC") != nullptr;
    verify_m_ = m;
    // 1. embed the m verify rows from d_verify_meta_[r].token_id
    launch_verify_gather_tokens(d_verify_meta_, d_token_ids_, m, stream_);
    // 2. the exact decode layer stack with n_tokens = m (per-row meta drives
    //    positions, RoPE, KV writes, tau and the attention windows)
    if (vsync) { HIP_CHECK(hipStreamSynchronize(stream_)); std::fprintf(stderr, "[verify] gather+embed ok\n"); }
    record_forward_pass(m, /*pos_start=*/0, d_verify_meta_, /*compute_all_logits=*/false,
                        /*d_raw_embeds=*/nullptr, /*row_slot=*/0, /*skip_head=*/true);
    verify_m_ = 0;
    if (vsync) { HIP_CHECK(hipStreamSynchronize(stream_)); std::fprintf(stderr, "[verify] layers ok\n"); }
    if (const char * dp = std::getenv("INKLING_DUMP_HID")) {
        HIP_CHECK(hipStreamSynchronize(stream_));
        std::vector<float> h((size_t)m * N_EMBD);
        HIP_CHECK(hipMemcpy(h.data(), buf_x_norm_, h.size() * 4, hipMemcpyDeviceToHost));
        if (FILE * fp = std::fopen(dp, "ab")) {
            std::fwrite(h.data(), 4, h.size(), fp);
            std::fclose(fp);
        }
    }
    // 3. one m-row lm_head weight stream + per-row softcapped argmax
    launch_soa_q8_gemv_fdot2_mrows(weights_.lm_head, buf_x_norm_, buf_logits_, m, stream_, 0);
    for (int r = 0; r < m; ++r) {
        launch_logit_softcap_argmax(buf_logits_ + (size_t)r * VOCAB_UNPADDED, VOCAB_UNPADDED,
                                    LOGIT_SOFTCAP, d_target_ + r, buf_router_scratch_, stream_);
    }
}

void InklingEngine::capture_verify_graph(int idx, int m) {
    if (idx < 0 || idx >= MAX_VERIFY_TOKENS || verify_graph_ready_[idx]) return;
    HIP_CHECK(hipStreamSynchronize(stream_));
    hipGraph_t g = nullptr;
    HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal));
    graph_capturing_ = true;
    record_verify_pass(m);
    graph_capturing_ = false;
    HIP_CHECK(hipStreamEndCapture(stream_, &g));
    HIP_CHECK(hipGraphInstantiate(&verify_graph_exec_[idx], g, nullptr, nullptr, 0));
    size_t nodes = 0;
    (void)hipGraphGetNodes(g, nullptr, &nodes);
    (void)hipGraphDestroy(g);
    verify_graph_ready_[idx] = true;
    std::fprintf(stderr, "[InklingEngine] Captured static verify(m=%d) HIP Graph: %zu kernel nodes\n",
                 m, nodes);
    std::fflush(stderr);
}

bool InklingEngine::verify_step(const int32_t * tokens, int m, int pos_start) {
    if (!mtp_io_ || tokens == nullptr || m < 1 || m > MAX_VERIFY_TOKENS) {
        std::fprintf(stderr, "[InklingEngine] verify_step: bad arguments (mtp_io=%d m=%d)\n", (int)mtp_io_, m);
        return false;
    }
    if (pos_start != cur_pos_ || pos_start + m > cfg_.max_ctx) {
        std::fprintf(stderr, "[InklingEngine] verify_step: pos mismatch (%d vs cur %d) or past max_ctx\n",
                     pos_start, cur_pos_);
        return false;
    }
    // The short-conv state must be recoverable when the round does not accept all rows.
    if (m > 1) {
        launch_mtp_save_hist(d_hist_ptrs_, d_hist_bak_, stream_);
    }
    // Per-row decode metadata (exactly what make_step_meta produces for each token).
    for (int r = 0; r < m; ++r) {
        h_verify_meta_[r] = make_step_meta(tokens[r], pos_start + r, cfg_.swa_cap);
    }
    HIP_CHECK(hipMemcpyAsync(d_verify_meta_, h_verify_meta_, (size_t)m * sizeof(StepMeta),
                             hipMemcpyHostToDevice, stream_));
    // Step 8B: batched m-token verify (one weight/expert stream per matrix for
    // all m rows) inside a static HIP graph.
    if (cfg_.verify_batch > 0) {
        const int gidx = (m >= 1 && m <= MAX_VERIFY_TOKENS) ? (m - 1) : -1;
        if (gidx >= 0 && cfg_.use_hip_graph) {
            if (!verify_graph_ready_[gidx]) capture_verify_graph(gidx, m);
            if (verify_graph_ready_[gidx]) {
                HIP_CHECK(hipGraphLaunch(verify_graph_exec_[gidx], stream_));
                HIP_CHECK(hipMemcpyAsync(h_target_pinned_, d_target_, (size_t)m * sizeof(int32_t),
                                         hipMemcpyDeviceToHost, stream_));
                HIP_CHECK(hipStreamSynchronize(stream_));
                cur_pos_ = pos_start + m;
                slot_pos_[0] = cur_pos_;
                return true;
            }
        }
        record_verify_pass(m);
        HIP_CHECK(hipMemcpyAsync(h_target_pinned_, d_target_, (size_t)m * sizeof(int32_t),
                                 hipMemcpyDeviceToHost, stream_));
        HIP_CHECK(hipStreamSynchronize(stream_));
        cur_pos_ = pos_start + m;
        slot_pos_[0] = cur_pos_;
        return true;
    }
    // Default: m exact single-token decode rows (the same launches as the non-MTP
    // path) -- bit-identical greedy verification.
    verify_mode_ = true;
    for (int r = 0; r < m; ++r) {
        verify_row_ = r;
        record_forward_pass(1, pos_start + r, d_verify_meta_ + r, /*compute_all_logits=*/false,
                            /*d_raw_embeds=*/nullptr, /*row_slot=*/r, /*skip_head=*/true);
    }
    verify_mode_ = false;
    verify_row_  = 0;
    if (mtp_batched_head_) {
        launch_soa_q8_gemv_fdot2_mrows(weights_.lm_head, buf_x_norm_, buf_logits_, m, stream_, 0);
    } else {
        for (int r = 0; r < m; ++r) {
            launch_soa_q8_gemv_fdot2(weights_.lm_head, buf_x_norm_ + (size_t)r * N_EMBD,
                                     buf_logits_ + (size_t)r * VOCAB_UNPADDED, stream_, 0);
        }
    }
    for (int r = 0; r < m; ++r) {
        launch_logit_softcap_argmax(buf_logits_ + (size_t)r * VOCAB_UNPADDED, VOCAB_UNPADDED,
                                    LOGIT_SOFTCAP, d_target_ + r, buf_router_scratch_, stream_);
    }
    HIP_CHECK(hipMemcpyAsync(h_target_pinned_, d_target_, (size_t)m * sizeof(int32_t),
                             hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));
    cur_pos_ = pos_start + m;
    slot_pos_[0] = cur_pos_;
    return true;
}

void InklingEngine::debug_dump_hist(int slot, std::vector<float> & out) {
    out.resize((size_t)N_SCONV_CHANNELS * N_CONV_HIST);
    HIP_CHECK(hipMemcpy(out.data(), sconv_slots_[slot].hist, out.size() * sizeof(float),
                        hipMemcpyDeviceToHost));
}

void InklingEngine::verify_logits_to_host(int m, std::vector<float> & out) const {
    if (m < 1) m = 1;
    out.resize((size_t)m * VOCAB_UNPADDED);
    HIP_CHECK(hipMemcpyAsync(out.data(), buf_logits_, (size_t)m * VOCAB_UNPADDED * sizeof(float),
                             hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));
}

void InklingEngine::verify_row_logits_to_host(int row, std::vector<float> & out) const {
    if (row < 0) row = 0;
    if (row >= MAX_VERIFY_TOKENS) row = MAX_VERIFY_TOKENS - 1;
    out.resize(VOCAB_UNPADDED);
    HIP_CHECK(hipMemcpyAsync(out.data(), buf_logits_ + (size_t)row * VOCAB_UNPADDED,
                             (size_t)VOCAB_UNPADDED * sizeof(float),
                             hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));
}

bool InklingEngine::verify_topk_to_host(int m, float inv_t,
                                        float * out_vals, int32_t * out_ids,
                                        float * out_max_l, float * out_sum_exp) const {
    if (!mtp_io_ || m < 1 || m > MAX_VERIFY_TOKENS ||
        out_vals == nullptr || out_ids == nullptr ||
        out_max_l == nullptr || out_sum_exp == nullptr ||
        d_verify_topk_vals_ == nullptr) {
        return false;
    }
    launch_mtp_verify_topk(buf_logits_, m, VOCAB_UNPADDED, inv_t,
                           d_verify_topk_vals_, d_verify_topk_ids_,
                           d_verify_stats_, d_verify_stats_ + MAX_VERIFY_TOKENS, stream_);
    const size_t n_k = (size_t)m * MTP_VERIFY_TOPK;
    HIP_CHECK(hipMemcpyAsync(h_verify_topk_vals_, d_verify_topk_vals_, n_k * sizeof(float),
                             hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_verify_topk_ids_, d_verify_topk_ids_, n_k * sizeof(int32_t),
                             hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_verify_stats_, d_verify_stats_,
                             (size_t)MAX_VERIFY_TOKENS * 2 * sizeof(float),
                             hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));
    std::memcpy(out_vals, h_verify_topk_vals_, n_k * sizeof(float));
    std::memcpy(out_ids,  h_verify_topk_ids_,  n_k * sizeof(int32_t));
    std::memcpy(out_max_l,   h_verify_stats_,                     (size_t)m * sizeof(float));
    std::memcpy(out_sum_exp, h_verify_stats_ + MAX_VERIFY_TOKENS, (size_t)m * sizeof(float));
    return true;
}

bool InklingEngine::sample_rows_full_to_host(int m, float inv_t, float min_p,
                                             const int32_t * excl, uint64_t seed,
                                             int32_t * out_tok, float * out_stats) const {
    static_assert(MAX_VERIFY_TOKENS <= MTP_SAMPLE_MAX_ROWS, "sampler row capacity");
    if (m < 1 || m > MAX_VERIFY_TOKENS || !(inv_t > 0.0f) || !std::isfinite(inv_t) ||
        out_tok == nullptr || out_stats == nullptr || d_sample_scratch_ == nullptr) {
        return false;
    }
    MtpSampleExcl ex{};
    for (int r = 0; r < MTP_SAMPLE_MAX_ROWS; ++r) {
        const int32_t t = (excl != nullptr && r < m) ? excl[r] : -1;
        ex.tok[r] = (t >= 0 && t < VOCAB_UNPADDED) ? t : -1;
    }
    launch_mtp_sample_full(buf_logits_, m, VOCAB_UNPADDED, VOCAB_UNPADDED, inv_t, min_p, ex, seed,
                           d_sample_scratch_, d_sample_tok_, d_sample_stats_, stream_);
    HIP_CHECK(hipMemcpyAsync(h_sample_tok_, d_sample_tok_, (size_t)m * sizeof(int32_t),
                             hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_sample_stats_, d_sample_stats_, (size_t)m * 4 * sizeof(float),
                             hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));
    std::memcpy(out_tok, h_sample_tok_, (size_t)m * sizeof(int32_t));
    std::memcpy(out_stats, h_sample_stats_, (size_t)m * 4 * sizeof(float));
    return true;
}

bool InklingEngine::sample_batch_rows_full_to_host(int n, const int32_t * row_idx,
                                                   const float * inv_t, const float * min_p,
                                                   const uint64_t * seeds,
                                                   int32_t * out_tok, float * out_stats) const {
    static_assert(MAX_CONCURRENT_SLOTS <= MAX_VERIFY_TOKENS, "batch sampler row capacity");
    static_assert(MAX_VERIFY_TOKENS <= MTP_SAMPLE_MAX_ROWS, "sampler row capacity");
    if (n < 1 || n > MAX_VERIFY_TOKENS ||
        row_idx == nullptr || inv_t == nullptr || min_p == nullptr || seeds == nullptr ||
        out_tok == nullptr || out_stats == nullptr || d_sample_scratch_ == nullptr) {
        return false;
    }
    MtpSampleBatchParams params{};
    for (int i = 0; i < n; ++i) {
        if (row_idx[i] < 0 || row_idx[i] >= MAX_VERIFY_TOKENS ||
            !(inv_t[i] > 0.0f) || !std::isfinite(inv_t[i])) {
            return false;
        }
        params.row_idx[i] = row_idx[i];
        params.excl[i]    = -1;
        params.inv_t[i]   = inv_t[i];
        params.min_p[i]   = (min_p[i] > 0.0f) ? min_p[i] : 0.0f;
        params.k0[i]      = (uint32_t)seeds[i];
        params.k1[i]      = (uint32_t)(seeds[i] >> 32);
        params.salt[i]    = 0u;
    }
    launch_mtp_sample_full_batch(buf_logits_, n, VOCAB_UNPADDED, VOCAB_UNPADDED, params,
                                 d_sample_scratch_, d_sample_tok_, d_sample_stats_, stream_);
    HIP_CHECK(hipMemcpyAsync(h_sample_tok_, d_sample_tok_, (size_t)n * sizeof(int32_t),
                             hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipMemcpyAsync(h_sample_stats_, d_sample_stats_, (size_t)n * 4 * sizeof(float),
                             hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));
    std::memcpy(out_tok, h_sample_tok_, (size_t)n * sizeof(int32_t));
    std::memcpy(out_stats, h_sample_stats_, (size_t)n * 4 * sizeof(float));
    return true;
}

void InklingEngine::verify_rollback(int pos_start, int commit, int m) {
    if (!mtp_io_) return;
    if (commit < 1) commit = 1;
    if (commit > m) commit = m;
    if (commit < m) {
        // Hist after `commit` rows = the raw pre-conv inputs of the last 3 committed
        // positions (from the verify dump) or, before the block, the saved state.
        launch_mtp_rebuild_hist(d_hist_ptrs_, d_hist_bak_, d_verify_raw_, commit, stream_);
        // Rejected Global KV rows: mark them empty (never read again, overwritten
        // by the next verify row; the SWA ring cap 1536 > window 1024 makes the
        // stale SWA slots unreachable, so no SWA restore is needed).
        for (int s = 0; s < N_LAYER_SLOTS; ++s) {
            if (kv_slots_[s].is_swa) continue;
            HIP_CHECK(hipMemsetAsync(kv_slots_[s].k_pos + pos_start + commit, 0xFF,
                                     (size_t)(m - commit) * sizeof(int32_t), stream_));
        }
    }
    cur_pos_ = pos_start + commit;
    slot_pos_[0] = cur_pos_;
    // Quest page metadata was deliberately NOT touched during verify (candidate
    // pages are >= SPARSE_RECENT_PAGES = 24 pages behind the query, so the block's
    // own page is never scored).  Recompute it now for exactly the committed
    // tokens, once per touched page, with the same kernel a decode step uses.
    if (cfg_.sparse_pages > 0) {
        for (int r = 0; r < commit; ++r) {
            const bool page_end = (r + 1 == commit) ||
                                  (((pos_start + r) / SPARSE_PAGE_SIZE) != ((pos_start + r + 1) / SPARSE_PAGE_SIZE));
            if (!page_end) continue;
            for (int s = 0; s < N_LAYER_SLOTS; ++s) {
                if (kv_slots_[s].is_swa) continue;
                launch_kv_update_page_minmax_decode(kv_slots_[s], sparse_meta_[s], d_verify_meta_ + r, stream_);
            }
        }
    }
}

void InklingEngine::capture_decode_graph() {
    if (graph_captured_) return;
    active_slot_ = 0;
    batch_slots_active_ = false;
    verify_m_ = 0;
    HIP_CHECK(hipStreamSynchronize(stream_));
    HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal));
    graph_capturing_ = true;
    record_forward_pass(/*n_tokens=*/1, /*pos_start=*/0, d_meta_, /*compute_all_logits=*/false);
    graph_capturing_ = false;
    HIP_CHECK(hipStreamEndCapture(stream_, &decode_graph_));
    HIP_CHECK(hipGraphInstantiate(&decode_graph_exec_, decode_graph_, nullptr, nullptr, 0));
    graph_captured_ = true;

    size_t num_nodes = 0;
    (void)hipGraphGetNodes(decode_graph_, nullptr, &num_nodes);
    std::fprintf(stderr, "[InklingEngine] Captured 1-shot static decode HIP Graph: %zu kernel nodes!\n", num_nodes);
    std::fflush(stderr);
}

static inline StepMeta make_step_meta(int32_t token_id, int32_t pos, int32_t swa_cap) {
    StepMeta m{};
    m.token_id        = token_id;
    m.pos             = pos;
    m.kv_len_global   = pos + 1;
    m.kv_len_swa      = std::min(pos + 1, swa_cap);
    m.kv_write_global = pos;
    m.kv_write_swa    = pos % swa_cap;
    float ratio       = (float)(pos + 1) / LOG_SCALE_FLOOR;
    m.tau             = 1.0f + LOG_SCALE_ALPHA * std::log(ratio > 1.0f ? ratio : 1.0f);
    m._pad            = 0;
    return m;
}

bool InklingEngine::prefill(
        const std::vector<int32_t> & tokens,
        bool compute_all_logits,
        std::vector<float> * out_logits) {
    const int total = (int)tokens.size();
    if (total <= 0) return true;
    if (cur_pos_ + total > cfg_.max_ctx) {
        std::fprintf(stderr, "InklingEngine::prefill: exceeds max_ctx (%d + %d > %d)\n",
                     cur_pos_, total, cfg_.max_ctx);
        return false;
    }

    if (out_logits) {
        if (compute_all_logits) {
            out_logits->resize((size_t)total * VOCAB_UNPADDED);
        } else {
            out_logits->resize(VOCAB_UNPADDED);
        }
    }

    active_slot_ = 0;
    int offset = 0;
    while (offset < total) {
        const int chunk = std::min(MAX_UBATCH, total - offset);
        const bool is_last_chunk = (offset + chunk == total);
        if (chunk == 1) {
            // Single-token chunk: use fast decode path (eager while MTP taps are on,
            // because the captured graph does not write them).
            StepMeta meta = make_step_meta(tokens[offset], cur_pos_, cfg_.swa_cap);
            HIP_CHECK(hipMemcpyAsync(d_meta_, &meta, sizeof(StepMeta), hipMemcpyHostToDevice, stream_));
            if (cfg_.use_hip_graph && !mtp_io_) {
                if (!graph_captured_) capture_decode_graph();
                HIP_CHECK(hipGraphLaunch(decode_graph_exec_, stream_));
            } else {
                record_forward_pass(1, cur_pos_, d_meta_, false);
            }
            if (out_logits && (compute_all_logits || is_last_chunk)) {
                float * dst = compute_all_logits ? (out_logits->data() + (size_t)offset * VOCAB_UNPADDED)
                                                 : out_logits->data();
                HIP_CHECK(hipMemcpyAsync(dst, buf_logits_, (size_t)VOCAB_UNPADDED * sizeof(float), hipMemcpyDeviceToHost, stream_));
            }
        } else {
            HIP_CHECK(hipMemcpyAsync(d_token_ids_, tokens.data() + offset, (size_t)chunk * sizeof(int32_t), hipMemcpyHostToDevice, stream_));
            const bool need_all = compute_all_logits;
            record_forward_pass(chunk, cur_pos_, /*d_meta=*/nullptr, need_all);
            if (out_logits && need_all) {
                HIP_CHECK(hipMemcpyAsync(
                    out_logits->data() + (size_t)offset * VOCAB_UNPADDED,
                    buf_logits_, (size_t)chunk * VOCAB_UNPADDED * sizeof(float),
                    hipMemcpyDeviceToHost, stream_));
            } else if (out_logits && is_last_chunk) {
                HIP_CHECK(hipMemcpyAsync(
                    out_logits->data(), buf_logits_,
                    (size_t)VOCAB_UNPADDED * sizeof(float),
                    hipMemcpyDeviceToHost, stream_));
            }
        }
        if (tap_hook_ && mtp_io_) {
            tap_hook_(d_taps_, tokens.data() + offset, chunk, cur_pos_);
        }
        cur_pos_ += chunk;
        slot_pos_[0] = cur_pos_;
        offset   += chunk;
    }

    HIP_CHECK(hipStreamSynchronize(stream_));
    return true;
}

bool InklingEngine::prefill_embeds(
        const float * d_embeds,
        int n_tokens,
        bool compute_all_logits,
        std::vector<float> * out_logits) {
    if (n_tokens <= 0) return true;
    if (cur_pos_ + n_tokens > cfg_.max_ctx) {
        std::fprintf(stderr, "InklingEngine::prefill_embeds: exceeds max_ctx (%d + %d > %d)\n",
                     cur_pos_, n_tokens, cfg_.max_ctx);
        return false;
    }

    if (out_logits) {
        if (compute_all_logits) {
            out_logits->resize((size_t)n_tokens * VOCAB_UNPADDED);
        } else {
            out_logits->resize(VOCAB_UNPADDED);
        }
    }

    active_slot_ = 0;
    int offset = 0;
    while (offset < n_tokens) {
        const int chunk = std::min(MAX_UBATCH, n_tokens - offset);
        const bool is_last_chunk = (offset + chunk == n_tokens);
        const float * chunk_ptr = d_embeds + (size_t)offset * N_EMBD;
        if (chunk == 1) {
            StepMeta meta = make_step_meta(0, cur_pos_, cfg_.swa_cap);
            HIP_CHECK(hipMemcpyAsync(d_meta_, &meta, sizeof(StepMeta), hipMemcpyHostToDevice, stream_));
            record_forward_pass(1, cur_pos_, d_meta_, false, chunk_ptr);
            if (out_logits && (compute_all_logits || is_last_chunk)) {
                float * dst = compute_all_logits ? (out_logits->data() + (size_t)offset * VOCAB_UNPADDED)
                                                 : out_logits->data();
                HIP_CHECK(hipMemcpyAsync(dst, buf_logits_, (size_t)VOCAB_UNPADDED * sizeof(float), hipMemcpyDeviceToHost, stream_));
            }
        } else {
            record_forward_pass(chunk, cur_pos_, /*d_meta=*/nullptr, compute_all_logits, chunk_ptr);
            if (out_logits && compute_all_logits) {
                HIP_CHECK(hipMemcpyAsync(
                    out_logits->data() + (size_t)offset * VOCAB_UNPADDED,
                    buf_logits_, (size_t)chunk * VOCAB_UNPADDED * sizeof(float),
                    hipMemcpyDeviceToHost, stream_));
            } else if (out_logits && is_last_chunk) {
                HIP_CHECK(hipMemcpyAsync(
                    out_logits->data(), buf_logits_,
                    (size_t)VOCAB_UNPADDED * sizeof(float),
                    hipMemcpyDeviceToHost, stream_));
            }
        }
        if (tap_hook_ && mtp_io_) {
            tap_hook_(d_taps_, nullptr, chunk, cur_pos_);
        }
        cur_pos_ += chunk;
        slot_pos_[0] = cur_pos_;
        offset   += chunk;
    }

    HIP_CHECK(hipStreamSynchronize(stream_));
    return true;
}

double InklingEngine::benchmark_prefill_chunk(const std::vector<int32_t> & tokens) {
    active_slot_ = 0;
    const int chunk = (int)tokens.size();
    HIP_CHECK(hipMemcpyAsync(d_token_ids_, tokens.data(), (size_t)chunk * sizeof(int32_t), hipMemcpyHostToDevice, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));

    auto t0 = std::chrono::steady_clock::now();
    record_forward_pass(chunk, cur_pos_, nullptr, /*compute_all_logits=*/false);
    HIP_CHECK(hipStreamSynchronize(stream_));
    auto t1 = std::chrono::steady_clock::now();

    cur_pos_ += chunk;
    slot_pos_[0] = cur_pos_;
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

int32_t InklingEngine::decode_step(int32_t token_id, std::vector<float> * out_logits) {
    if (cur_pos_ >= cfg_.max_ctx) {
        std::fprintf(stderr, "InklingEngine::decode_step: max_ctx reached (%d)\n", cfg_.max_ctx);
        return -1;
    }
    active_slot_ = 0;
    if (cfg_.use_hip_graph && !graph_captured_) {
        capture_decode_graph();
    }

    // Pinned staging: zero pageable-staging stalls before/after the graph.
    StepMeta * meta = h_meta_pinned_;
    *meta = make_step_meta(token_id, cur_pos_, cfg_.swa_cap);
    HIP_CHECK(hipMemcpyAsync(d_meta_, meta, sizeof(StepMeta), hipMemcpyHostToDevice, stream_));

    if (cfg_.use_hip_graph) {
        HIP_CHECK(hipGraphLaunch(decode_graph_exec_, stream_));
    } else {
        record_forward_pass(1, cur_pos_, d_meta_, false);
    }

    HIP_CHECK(hipMemcpyAsync(h_argmax_pinned_, d_argmax_id_, sizeof(int32_t), hipMemcpyDeviceToHost, stream_));
    if (out_logits) {
        out_logits->resize(VOCAB_UNPADDED);
        HIP_CHECK(hipMemcpyAsync(out_logits->data(), buf_logits_, (size_t)VOCAB_UNPADDED * sizeof(float), hipMemcpyDeviceToHost, stream_));
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
    if (const char * dp = std::getenv("INKLING_DUMP_HID")) {
        std::vector<float> h((size_t)N_EMBD);
        HIP_CHECK(hipMemcpy(h.data(), buf_x_norm_, h.size() * 4, hipMemcpyDeviceToHost));
        if (FILE * fp = std::fopen(dp, "ab")) {
            std::fwrite(h.data(), 4, h.size(), fp);
            std::fclose(fp);
        }
    }
    const int32_t next_id = *h_argmax_pinned_;
    cur_pos_ += 1;
    slot_pos_[0] = cur_pos_;
    return next_id;
}

double InklingEngine::benchmark_decode_steps(int32_t start_token, int n_steps, std::vector<int32_t> * out_tokens) {
    active_slot_ = 0;
    if (cfg_.use_hip_graph && !graph_captured_) {
        capture_decode_graph();
    }
    if (out_tokens) out_tokens->clear();
    if (n_steps <= 0) return 0.0;

    // The in-graph softcap/argmax epilogue advances d_meta_ (token_id, pos, kv len/write,
    // tau) and writes the produced token to d_tok_hist_[pos] on the device, so the whole
    // step loop needs exactly ONE host<->device round-trip (initial meta) + ONE final sync.
    const int32_t pos0 = cur_pos_;
    if (pos0 + n_steps + 2 > tok_hist_cap_) {
        std::fprintf(stderr, "InklingEngine::benchmark_decode_steps: exceeds token history capacity\n");
        return 0.0;
    }
    StepMeta * meta = h_meta_pinned_;
    *meta = make_step_meta(start_token, pos0, cfg_.swa_cap);
    HIP_CHECK(hipMemcpyAsync(d_meta_, meta, sizeof(StepMeta), hipMemcpyHostToDevice, stream_));

    // Warmup sync (before the timer, so all prior work has drained)
    HIP_CHECK(hipStreamSynchronize(stream_));
    auto t0 = std::chrono::steady_clock::now();

    for (int i = 0; i < n_steps; ++i) {
        if (cfg_.use_hip_graph) {
            HIP_CHECK(hipGraphLaunch(decode_graph_exec_, stream_));
        } else {
            record_forward_pass(1, pos0 + i, d_meta_, false);
        }
    }
    HIP_CHECK(hipMemcpyAsync(h_tok_hist_pinned_, d_tok_hist_ + pos0, (size_t)(n_steps + 1) * sizeof(int32_t),
                             hipMemcpyDeviceToHost, stream_));
    HIP_CHECK(hipStreamSynchronize(stream_));

    auto t1 = std::chrono::steady_clock::now();
    cur_pos_ += n_steps;
    slot_pos_[0] = cur_pos_;

    // Produced tokens are hist[pos] for pos = pos0+1 .. pos0+n_steps
    for (int i = 0; i < n_steps; ++i) {
        const int32_t tok = h_tok_hist_pinned_[i + 1];
        if (tok < 0 || tok >= VOCAB_UNPADDED) {
            std::fprintf(stderr, "[InklingEngine] ERROR: invalid token %d at step=%d pos=%d\n", tok, i, pos0 + i + 1);
            std::abort();
        }
        if (out_tokens) out_tokens->push_back(tok);
    }
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ============================================================================
// Multi-slot continuous batching (zero-regression on single-stream slot 0)
// ============================================================================
bool InklingEngine::prefill_slot_chunk(
        int slot, const int32_t * tokens, int n_tokens,
        bool want_logits, std::vector<float> * out_logits) {
    if (slot < 0 || slot >= cfg_.n_slots || tokens == nullptr || n_tokens <= 0 || n_tokens > MAX_UBATCH) {
        return false;
    }
    const int32_t limit = slot_ctx_limit(slot);
    const int32_t pos = slot_pos_[slot];
    if (pos + n_tokens > limit) {
        std::fprintf(stderr, "InklingEngine::prefill_slot_chunk(slot=%d, span=%d): exceeds limit (%d + %d > %d)\n",
                     slot, slot_span_[slot], pos, n_tokens, limit);
        return false;
    }

    active_slot_ = slot;
    if (n_tokens == 1) {
        StepMeta meta = make_step_meta(tokens[0], pos, cfg_.swa_cap);
        HIP_CHECK(hipMemcpyAsync(d_meta_, &meta, sizeof(StepMeta), hipMemcpyHostToDevice, stream_));
        if (slot == 0 && cfg_.use_hip_graph && !mtp_io_) {
            if (!graph_captured_) capture_decode_graph();
            HIP_CHECK(hipGraphLaunch(decode_graph_exec_, stream_));
        } else {
            record_forward_pass(1, pos, d_meta_, /*compute_all_logits=*/false,
                                /*d_raw_embeds=*/nullptr, /*row_slot=*/0, /*skip_head=*/!want_logits);
        }
    } else {
        HIP_CHECK(hipMemcpyAsync(d_token_ids_, tokens, (size_t)n_tokens * sizeof(int32_t),
                                 hipMemcpyHostToDevice, stream_));
        record_forward_pass(n_tokens, pos, /*d_meta=*/nullptr, /*compute_all_logits=*/false,
                            /*d_raw_embeds=*/nullptr, /*row_slot=*/0, /*skip_head=*/!want_logits);
    }

    if (want_logits && out_logits != nullptr) {
        out_logits->resize(VOCAB_UNPADDED);
        HIP_CHECK(hipMemcpyAsync(out_logits->data(), buf_logits_,
                                 (size_t)VOCAB_UNPADDED * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_));
    }
    if (slot == 0 && tap_hook_ && mtp_io_) {
        tap_hook_(d_taps_, tokens, n_tokens, pos);
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
    active_slot_ = 0;

    slot_pos_[slot] = pos + n_tokens;
    if (slot == 0) cur_pos_ = slot_pos_[0];
    return true;
}

bool InklingEngine::prefill_embeds_slot_chunk(
        int slot, const float * d_embeds, int n_tokens,
        bool want_logits, std::vector<float> * out_logits) {
    if (slot < 0 || slot >= cfg_.n_slots || d_embeds == nullptr || n_tokens <= 0 || n_tokens > MAX_UBATCH) {
        return false;
    }
    const int32_t limit = slot_ctx_limit(slot);
    const int32_t pos = slot_pos_[slot];
    if (pos + n_tokens > limit) {
        std::fprintf(stderr, "InklingEngine::prefill_embeds_slot_chunk(slot=%d, span=%d): exceeds limit (%d + %d > %d)\n",
                     slot, slot_span_[slot], pos, n_tokens, limit);
        return false;
    }

    active_slot_ = slot;
    if (n_tokens == 1) {
        StepMeta meta = make_step_meta(0, pos, cfg_.swa_cap);
        HIP_CHECK(hipMemcpyAsync(d_meta_, &meta, sizeof(StepMeta), hipMemcpyHostToDevice, stream_));
        record_forward_pass(1, pos, d_meta_, /*compute_all_logits=*/false,
                            d_embeds, /*row_slot=*/0, /*skip_head=*/!want_logits);
    } else {
        record_forward_pass(n_tokens, pos, /*d_meta=*/nullptr, /*compute_all_logits=*/false,
                            d_embeds, /*row_slot=*/0, /*skip_head=*/!want_logits);
    }

    if (want_logits && out_logits != nullptr) {
        out_logits->resize(VOCAB_UNPADDED);
        HIP_CHECK(hipMemcpyAsync(out_logits->data(), buf_logits_,
                                 (size_t)VOCAB_UNPADDED * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_));
    }
    if (slot == 0 && tap_hook_ && mtp_io_) {
        tap_hook_(d_taps_, nullptr, n_tokens, pos);
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
    active_slot_ = 0;

    slot_pos_[slot] = pos + n_tokens;
    if (slot == 0) cur_pos_ = slot_pos_[0];
    return true;
}

int32_t InklingEngine::decode_slot_single(int slot, int32_t token_id, std::vector<float> * out_logits) {
    if (slot < 0 || slot >= cfg_.n_slots) return -1;
    if (slot == 0) {
        const int32_t limit0 = slot_ctx_limit(0);
        if (slot_pos_[0] >= limit0) {
            std::fprintf(stderr, "InklingEngine::decode_slot_single(slot=0, span=%d): limit reached (%d)\n",
                         slot_span_[0], limit0);
            return -1;
        }
        return decode_step(token_id, out_logits);
    }
    const int32_t pos = slot_pos_[slot];
    const int32_t limit = slot_ctx_limit(slot);
    if (pos >= limit) {
        std::fprintf(stderr, "InklingEngine::decode_slot_single(slot=%d, span=%d): limit reached (%d)\n",
                     slot, slot_span_[slot], limit);
        return -1;
    }

    active_slot_ = slot;
    StepMeta * meta = h_meta_pinned_;
    *meta = make_step_meta(token_id, pos, cfg_.swa_cap);
    HIP_CHECK(hipMemcpyAsync(d_meta_, meta, sizeof(StepMeta), hipMemcpyHostToDevice, stream_));
    record_forward_pass(1, pos, d_meta_, /*compute_all_logits=*/false);

    HIP_CHECK(hipMemcpyAsync(h_argmax_pinned_, d_argmax_id_, sizeof(int32_t),
                             hipMemcpyDeviceToHost, stream_));
    if (out_logits) {
        out_logits->resize(VOCAB_UNPADDED);
        HIP_CHECK(hipMemcpyAsync(out_logits->data(), buf_logits_,
                                 (size_t)VOCAB_UNPADDED * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_));
    }
    HIP_CHECK(hipStreamSynchronize(stream_));
    active_slot_ = 0;

    const int32_t next_id = *h_argmax_pinned_;
    slot_pos_[slot] = pos + 1;
    return next_id;
}

bool InklingEngine::decode_batch(
        const int32_t * slots, const int32_t * token_ids, int batch_size,
        int32_t * out_argmax, float * out_logits_buf) {
    if (slots == nullptr || token_ids == nullptr || batch_size < 1 ||
        batch_size > cfg_.n_slots || batch_size > MAX_CONCURRENT_SLOTS) {
        return false;
    }
    if (batch_size == 1) {
        std::vector<float> tmp_logits;
        const int32_t id = decode_slot_single(slots[0], token_ids[0],
                                              out_logits_buf ? &tmp_logits : nullptr);
        if (id < 0) return false;
        if (out_argmax) out_argmax[0] = id;
        if (out_logits_buf) {
            std::memcpy(out_logits_buf, tmp_logits.data(), (size_t)VOCAB_UNPADDED * sizeof(float));
        }
        return true;
    }
    if (!batch_scratch_ready_) {
        alloc_batch_scratch();
    }

    uint32_t seen_mask = 0;
    uint32_t graph_key = 0;
    for (int r = 0; r < batch_size; ++r) {
        const int c = slots[r];
        if (c < 0 || c >= cfg_.n_slots) return false;
        if (seen_mask & (1u << c)) return false; // duplicate slot in same batch
        seen_mask |= (1u << c);
        graph_key = (graph_key << 4) | (uint32_t)(c + 1);
        const int32_t limit = slot_ctx_limit(c);
        if (slot_pos_[c] >= limit) return false;
        batch_slot_ids_[r] = c;
        h_verify_meta_[r]  = make_step_meta(token_ids[r], slot_pos_[c], cfg_.swa_cap);
    }

    HIP_CHECK(hipMemcpyAsync(d_verify_meta_, h_verify_meta_, (size_t)batch_size * sizeof(StepMeta),
                             hipMemcpyHostToDevice, stream_));

    auto record_batch_body = [&]() {
        batch_slots_active_ = true;
        verify_m_ = batch_size;
        launch_verify_gather_tokens(d_verify_meta_, d_token_ids_, batch_size, stream_);
        record_forward_pass(batch_size, /*pos_start=*/0, d_verify_meta_, /*compute_all_logits=*/false,
                            /*d_raw_embeds=*/nullptr, /*row_slot=*/0, /*skip_head=*/true);
        verify_m_ = 0;
        batch_slots_active_ = false;

        // 1x weight-streamed lm_head across all batch_size rows + per-row softcap/argmax
        launch_soa_q8_gemv_fdot2_mrows(weights_.lm_head, buf_x_norm_, buf_logits_, batch_size, stream_, 0);
        for (int r = 0; r < batch_size; ++r) {
            launch_logit_softcap_argmax(buf_logits_ + (size_t)r * VOCAB_UNPADDED, VOCAB_UNPADDED,
                                        LOGIT_SOFTCAP, d_target_ + r, buf_router_scratch_, stream_);
        }
    };

    if (cfg_.use_hip_graph) {
        auto it = batch_graphs_.find(graph_key);
        if (it == batch_graphs_.end()) {
            HIP_CHECK(hipStreamSynchronize(stream_));
            if (batch_graphs_.size() >= MAX_BATCH_GRAPHS) {
                auto victim = batch_graphs_.begin();
                for (auto cand = batch_graphs_.begin(); cand != batch_graphs_.end(); ++cand) {
                    if (cand->second.last_used < victim->second.last_used) {
                        victim = cand;
                    }
                }
                if (victim->second.exec) {
                    (void)hipGraphExecDestroy(victim->second.exec);
                }
                batch_graphs_.erase(victim);
            }
            hipGraph_t g = nullptr;
            HIP_CHECK(hipStreamBeginCapture(stream_, hipStreamCaptureModeThreadLocal));
            graph_capturing_ = true;
            record_batch_body();
            graph_capturing_ = false;
            HIP_CHECK(hipStreamEndCapture(stream_, &g));
            hipGraphExec_t exec = nullptr;
            HIP_CHECK(hipGraphInstantiate(&exec, g, nullptr, nullptr, 0));
            size_t nodes = 0;
            (void)hipGraphGetNodes(g, nullptr, &nodes);
            (void)hipGraphDestroy(g);
            it = batch_graphs_.emplace(graph_key, BatchGraphEntry{exec, ++batch_graph_lru_seq_}).first;
            std::fprintf(stderr,
                         "[InklingEngine] Captured static batch(B=%d, key=0x%x) HIP Graph: %zu kernel nodes (cached %zu/%zu)\n",
                         batch_size, graph_key, nodes, batch_graphs_.size(), MAX_BATCH_GRAPHS);
            std::fflush(stderr);
        } else {
            it->second.last_used = ++batch_graph_lru_seq_;
        }
        HIP_CHECK(hipGraphLaunch(it->second.exec, stream_));
    } else {
        record_batch_body();
    }

    HIP_CHECK(hipMemcpyAsync(h_target_pinned_, d_target_, (size_t)batch_size * sizeof(int32_t),
                             hipMemcpyDeviceToHost, stream_));
    if (out_logits_buf != nullptr) {
        HIP_CHECK(hipMemcpyAsync(out_logits_buf, buf_logits_,
                                 (size_t)batch_size * VOCAB_UNPADDED * sizeof(float),
                                 hipMemcpyDeviceToHost, stream_));
    }
    HIP_CHECK(hipStreamSynchronize(stream_));

    for (int r = 0; r < batch_size; ++r) {
        const int c = slots[r];
        slot_pos_[c] += 1;
        if (c == 0) cur_pos_ = slot_pos_[0];
        if (out_argmax) out_argmax[r] = h_target_pinned_[r];
    }
    return true;
}

} // namespace inkling
