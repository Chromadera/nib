#pragma once

#include <cstddef>
#include <cstdint>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <string>
#include <vector>

// ============================================================================
// gfx12 hardware WMMA tile gate (include/nib_mma.h, NIB_GFX12_WMMA=1)
// ============================================================================
// nib builds one code object per .hip file, so the gate flag is a private
// __device__ variable per translation unit and cannot be written through a single
// symbol. Every .hip TU that uses the tile registers its own setter callback here
// from a static initializer; nib_mma::apply_gate_appliers() runs them once, after
// the HIP runtime and all code objects are up (InklingEngine::init and
// run_wmma_tile_selftest both call it). No env var -> nothing is registered and no
// GPU call is made.
namespace nib_mma {
typedef void (*gate_applier)();
void register_gate_applier(gate_applier f);   // src/engine.cpp
void apply_gate_appliers();                   // src/engine.cpp
} // namespace nib_mma

namespace inkling {

// ============================================================================
// Exact Inkling 2 Mini (inkling2_loop) Architecture Constants
// ============================================================================
static constexpr int N_LAYER_PHYS        = 24;
static constexpr int LOOP_START          = 6;
static constexpr int LOOP_END            = 18;
static constexpr int LOOP_COUNT          = 3;
static constexpr int LOOP_CORE_LEN       = LOOP_END - LOOP_START;                 // 12
static constexpr int N_LAYER_SLOTS       = N_LAYER_PHYS + (LOOP_COUNT - 1) * LOOP_CORE_LEN; // 48
static constexpr int LOOP_GATE_RANK      = 64;
static constexpr int LOOP_CTRL_COUNT     = 4; // Q, K, V, FFN
static constexpr int LOOP_CTRL_Q         = 0;
static constexpr int LOOP_CTRL_K         = 1;
static constexpr int LOOP_CTRL_V         = 2;
static constexpr int LOOP_CTRL_FFN       = 3;

static constexpr int N_EMBD              = 2304;
static constexpr int N_HEAD              = 24;
static constexpr int N_HEAD_KV           = 8;
static constexpr int GQA_RATIO           = N_HEAD / N_HEAD_KV;                    // 3
static constexpr int HEAD_DIM            = 128;
static constexpr int N_Q                 = N_HEAD * HEAD_DIM;                     // 3072
static constexpr int N_KV                = N_HEAD_KV * HEAD_DIM;                  // 1024

static constexpr int N_EXPERT            = 128;
static constexpr int TOP_K               = 8;
static constexpr int N_FF_EXP            = 1152;
static constexpr int N_GATE_UP_EXP       = 2 * N_FF_EXP;                          // 2304
static constexpr int NVFP4_MAX_LAYER     = 19; // physical layers 0..19 NVFP4, 20..23 Q8_0

static constexpr int N_REL               = 16;
static constexpr int N_REL_EXTENT        = 1024;
static constexpr int SWA_WINDOW          = 1024;
static constexpr int N_CONV              = 4;
static constexpr int N_CONV_HIST         = N_CONV - 1;                            // 3
// sconv state channels per slot: K(1024) + V(1024) + attn_out(2304) + ffn_out(2304) = 6656
static constexpr int N_SCONV_CHANNELS    = 2 * N_KV + 2 * N_EMBD;                 // 6656
static constexpr int ATTN_RES_BLOCK_SIZE = 6;

static constexpr int VOCAB_PADDED        = 201024;
static constexpr int VOCAB_UNPADDED      = 200058;

static constexpr float RMS_EPS           = 1e-6f;
static constexpr float QK_CLIP           = 22.627416610717773f; // 16 * sqrt(2)
static constexpr float REL_CLIP          = 8.0f;
static constexpr float LOG_SCALE_FLOOR   = 8192.0f;
static constexpr float LOG_SCALE_ALPHA   = 0.1f;
static constexpr float SITU_BETA         = 4.0f;
static constexpr float SITU_LINEAR_BETA  = 25.0f;
static constexpr float EXPERT_W_SCALE    = 8.0f;
static constexpr float EXPERT_NORM_EPS   = 1e-20f;
static constexpr float LOGIT_SOFTCAP     = 30.0f;
static constexpr float KQ_SCALE          = 0.08838834764831845f; // 1 / sqrt(128)
static constexpr int   KV_FWHT_N         = 128; // Orthogonal FWHT-128 for lossless Q8_0 KV quantization

static constexpr int   MAX_UBATCH        = 512;
static constexpr int   MAX_RES_BLOCKS    = 8; // max depth states passed to mix()

// Every 6th physical layer (5, 11, 17, 23) is global; all others are SWA-1024.
static inline bool is_swa_phys(int phys) {
    return (phys % 6) != 5;
}

static inline int slot_to_phys(int slot) {
    return (slot < N_LAYER_PHYS) ? slot : (LOOP_START + (slot - N_LAYER_PHYS) % LOOP_CORE_LEN);
}

static inline bool is_swa_slot(int slot) {
    return is_swa_phys(slot_to_phys(slot));
}

// ============================================================================
// RDNA3-Native 16-Byte-Aligned Structure-of-Arrays (SoA) Weight Layouts
// ============================================================================
//
// Why SoA?
// GGML block_q8_0 is 34 bytes (unaligned!) and block_nvfp4 is 36 bytes (unaligned!).
// By repacking at startup on GPU into separate contiguous planes:
//   - Q8_0 SoA:
//       qs: int8_t[rows * K]           (K % 32 == 0 => 32B/block => 100% 128-bit aligned!)
//       d:  __half[rows * (K / 32)]    (contiguous fp16 scales)
//     Dequant: w[r, k] = __half2float(d[r*(K/32) + k/32]) * (float)qs[r*K + k]
//
//   - NVFP4 SoA (128 experts, each [rows, K], K % 64 == 0):
//       qs:     uint8_t[n_exp * rows * (K / 2)]  (32B per 64 weights => 100% 128-bit aligned!)
//               EXACT GGML NVFP4 nibble packing within each 16-elem sub-block s in {0..3}:
//                 byte j in {0..7} at offset (blk*32 + s*8 + j):
//                   low nibble  (b & 0xF) -> elem (blk*64 + s*16 + j)
//                   high nibble (b >> 4)  -> elem (blk*64 + s*16 + j + 8)
//       scales: uint8_t[n_exp * rows * (K / 16)] (4 bytes per 64 weights = 1 uint32 per block!)
//               Each byte is UE4M3 code; ggml_ue4m3_to_fp32(code) returns raw * 0.5f.
//       scale2: float[n_exp] (per-expert global FP32 multiplier)
//     Dequant: w = kvalues_mxfp4[nibble] * ue4m3_to_fp32(scales[sub]) * scale2[exp]
//              where kvalues_mxfp4 = {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12}.
//
// Note: Because qs rows are 16-byte aligned and K is a multiple of 64 (1152, 2304, 3072),
// every single row & block starts on a 128-bit boundary!

struct SoaQ8Matrix {
    int8_t * qs   = nullptr; // [rows, K] row-major
    __half * d    = nullptr; // [rows, K/32] row-major
    int      rows = 0;
    int      K    = 0;
};

struct SoaNVFP4Experts {
    uint8_t * qs     = nullptr; // [n_exp, rows, K/2]
    uint8_t * scales = nullptr; // [n_exp, rows, K/16] (UE4M3 bytes)
    float   * scale2 = nullptr; // [n_exp] (F32)
    int       n_exp  = N_EXPERT;
    int       rows   = 0;       // 2304 for gate_up, 2304 for down
    int       K      = 0;       // 2304 for gate_up, 1152 for down
};

struct SoaQ8Experts {
    int8_t * qs    = nullptr;   // [n_exp, rows, K]
    __half * d     = nullptr;   // [n_exp, rows, K/32]
    int      n_exp = N_EXPERT;
    int      rows  = 0;
    int      K     = 0;
};

// ============================================================================
// Device-Side Dynamic Decode Metadata (Enables 1 Static HIP Graph for 0..256k!)
// ============================================================================
// Updated via a single 32-byte hipMemcpyAsync before launching the captured decode graph.
struct StepMeta {
    int32_t token_id;        // current input token ID
    int32_t pos;             // absolute sequence position of current token (0, 1, 2, ...)
    int32_t kv_len_global;   // number of valid KV entries in global slots AFTER writing current token (pos + 1)
    int32_t kv_len_swa;      // number of valid KV entries in SWA ring buffer AFTER writing current token (min(pos+1, swa_cap))
    int32_t kv_write_global; // write index in global KV cache (= pos)
    int32_t kv_write_swa;    // write index in SWA ring buffer (= pos % swa_cap)
    float   tau;             // 1.0f + LOG_SCALE_ALPHA * logf(max(1.0f, (pos + 1.0f) / LOG_SCALE_FLOOR))
    int32_t _pad;
};

// ============================================================================
// KV Cache Layout (Q8_0, Contiguous per Slot, Wave32-Friendly)
// ============================================================================
// For each slot s in [0..47]:
//   cap = is_swa_slot(s) ? swa_cap : global_cap
//   (swa_cap = 1024 + MAX_UBATCH = 1536, global_cap = n_ctx e.g. 262144)
//   K/V stored in Head-Major SoA Q8_0 for 100% contiguous sequential DRAM streaming per KV head!
//   Layout: [N_HEAD_KV, cap, HEAD_DIM] for qs (int8_t) -> idx = ((size_t)h_kv * cap + j) * HEAD_DIM
//           [N_HEAD_KV, cap, 4]        for d  (__half) -> idx = ((size_t)h_kv * cap + j) * 4
//   k_pos: int32_t[cap] holding absolute token position at each slot index (or -1 if empty)
enum KVType : int32_t {
    KV_Q8_0 = 0, // 128 B qs + 8 B d (__half[4])    = 136 B/head/tok
    KV_Q4_1 = 1, //  64 B qs + 16 B dm (__half2[4]) =  80 B/head/tok
    KV_Q4_0 = 2, //  64 B qs +  8 B d (__half[4])   =  72 B/head/tok
};

struct KVCacheSlot {
    int8_t  * k_qs    = nullptr; // Q8_0: [N_HEAD_KV, cap, 128]; Q4_1/Q4_0: [N_HEAD_KV, cap, 64]
    __half  * k_d     = nullptr; // Q8_0/Q4_0: [N_HEAD_KV, cap, 4]; Q4_1: [N_HEAD_KV, cap, 8] (__half2[4])
    int8_t  * v_qs    = nullptr; // Q8_0: [N_HEAD_KV, cap, 128]; Q4_1/Q4_0: [N_HEAD_KV, cap, 64]
    __half  * v_d     = nullptr; // Q8_0/Q4_0: [N_HEAD_KV, cap, 4]; Q4_1: [N_HEAD_KV, cap, 8] (__half2[4])
    int32_t * k_pos   = nullptr; // [cap]
    int32_t   cap     = 0;
    int32_t   kv_type = KV_Q8_0;
    bool      is_swa  = true;
};

// Short-convolution recurrent state per slot:
// Layout: [N_SCONV_CHANNELS, N_CONV_HIST] (channel-major: 3 contiguous floats per channel!)
// Channel offsets:
//   K:        0
//   V:        1024 (N_KV)
//   attn_out: 2048 (2 * N_KV)
//   ffn_out:  4352 (2 * N_KV + N_EMBD)
struct SConvStateSlot {
    float * hist = nullptr; // [N_SCONV_CHANNELS * N_CONV_HIST]
};

// ============================================================================
// Per-Physical-Layer Weights (0..23) & Loop Controller Weights (6..17)
// ============================================================================
struct PhysLayerWeights {
    bool is_swa = true;
    bool is_nvfp4 = true; // true for phys 0..19, false (Q8_0) for phys 20..23

    // Fused sibling Q8_0 SoA matrices built at load time:
    //   SWA layers:
    //     proj_all6: [wq(3072) | wgate(3072) | wk(1024) | wv(1024) | wrel(384) | wsink(24)] = [8600, 2304]
    //     proj_q4:   view of top 6552 rows? NOTE: for Pass-1/2 SWA (where inputs[0,1,2] differ),
    //                we store Q-side [wq|wgate|wrel|wsink] (6552 rows) + wk (1024) + wv (1024)
    //                OR store [wq(3072)|wgate(3072)|wrel(384)|wsink(24)|wk(1024)|wv(1024)] = 8600 rows!
    //                GENIUS! Look at that ordering:
    //                  rows [0 .. 6552)    = Q-side: wq(3072) | wgate(3072) | wrel(384) | wsink(24)
    //                  rows [6552 .. 7576) = wk(1024)
    //                  rows [7576 .. 8600) = wv(1024)
    //                Then Pass-0 SWA runs ONE 8600-row GEMV on proj_qkv_all, while Pass-1/2 SWA runs
    //                ZERO-COPY views of the exact same buffer: proj_q4 (rows 0..6552), wk (rows 6552..7576), wv (rows 7576..8600)!
    //   Global layers:
    //     proj_q4: [wq(3072) | wgate(3072) | wrel(384) | wsink(24)] = [6552, 2304]
    //     proj_k2: [wk(1024) | wk_dyn(32)]                          = [1056, 2304]
    //     proj_v2: [wv(1024) | wv_dyn(32)]                          = [1056, 2304]
    SoaQ8Matrix proj_qkv_all; // SWA only: [8600, 2304] (zero-copy parent of proj_q4, proj_k, proj_v)
    SoaQ8Matrix proj_q4;      // [6552, 2304] (view on SWA, owned on Global)
    SoaQ8Matrix proj_k;       // [1024, 2304] on SWA (view), [1056, 2304] on Global (wk|wk_dyn)
    SoaQ8Matrix proj_v;       // [1024, 2304] on SWA (view), [1056, 2304] on Global (wv|wv_dyn)
    SoaQ8Matrix wo;           // [2304, 3072]

    // F32 norms, clips, convs, rel_bias, res_queries, router
    float * attn_norm       = nullptr; // [2304] (SWA only)
    float * attn_q_inp_norm = nullptr; // [2304] (Global only)
    float * attn_k_inp_norm = nullptr; // [2304] (Global only)
    float * attn_v_inp_norm = nullptr; // [2304] (Global only)

    float * attn_q_clip     = nullptr; // [24, 128] (stored [128, 24] in GGUF -> head-major [24*128])
    float * attn_rel_clip   = nullptr; // [24, 16]  (stored [16, 24] in GGUF)
    float * attn_rel_bias   = nullptr; // [1024, 16] (stored [16, 1024] in GGUF: ne0=16, ne1=1024)

    float * attn_k_conv     = nullptr; // [1024, 4] (ne0=4, ne1=1024)
    float * attn_v_conv     = nullptr; // [1024, 4]
    float * attn_post_conv  = nullptr; // [2304, 4]
    float * ffn_post_conv   = nullptr; // [2304, 4]

    float * attn_res_query  = nullptr; // [1 or 3, 2304] (nullptr for phys==0)
    float * ffn_res_query   = nullptr; // [2304]
    float * ffn_norm        = nullptr; // [2304]

    __half * ffn_gate_inp   = nullptr; // [128, 2304] FP16 on GPU (converted from F32 at load)
    float  * ffn_exp_probs_b = nullptr; // [128] (optional, present in Inkling 2 Mini)

    // Routed MoE experts (only one pair populated depending on is_nvfp4)
    SoaNVFP4Experts gate_up_nvfp4; // [128, 2304, 2304]
    SoaNVFP4Experts down_nvfp4;    // [128, 2304, 1152]
    SoaQ8Experts    gate_up_q8;    // [128, 2304, 2304]
    SoaQ8Experts    down_q8;       // [128, 2304, 1152]
};

struct LoopLayerWeights {
    float  * msg_query   = nullptr; // [4, 2304] (ne0=2304, ne1=4)
    __half * ctrl_down[LOOP_CTRL_COUNT]   = {}; // [64, 4608] FP16 on GPU
    __half * ctrl_gain[LOOP_CTRL_COUNT]   = {}; // [2304, 64] FP16 on GPU
    __half * ctrl_forget[LOOP_CTRL_COUNT] = {}; // [2304, 64] FP16 on GPU
};

struct ModelWeights {
    SoaQ8Matrix tok_embd;              // [201024, 2304] (actually only 200058 needed, keep full or unpadded)
    float *     tok_norm      = nullptr; // [2304]
    SoaQ8Matrix lm_head;               // [200058, 2304] (unpadded rows only! saves 966 rows)
    float *     output_norm   = nullptr; // [2304]
    float *     output_res_q  = nullptr; // [2304]

    PhysLayerWeights layers[N_LAYER_PHYS];
    LoopLayerWeights loop[LOOP_CORE_LEN];

    // Owned GPU backing allocations to free on destroy
    std::vector<void *> gpu_allocs;
    size_t total_vram_bytes = 0;

    // --cpu-embd: pinned, GPU-mapped host allocations (hipHostMalloc(Mapped)).
    // Stored as HOST pointers (freed with hipHostFree); the matrices hold the
    // hipHostGetDevicePointer() aliases.  Not counted in total_vram_bytes.
    std::vector<void *> host_allocs;
    size_t host_pinned_bytes = 0;
    bool   tok_embd_on_host  = false;

    void destroy();
};

// ============================================================================
// Tokenizer Metadata Loaded from GGUF
// ============================================================================
struct TokenizerData {
    std::vector<std::string> vocab;
    std::vector<float>       scores;
    std::vector<int32_t>     token_types;
    std::vector<std::string> merges;
    int32_t bos_id = -1;
    int32_t eos_id = -1;
    int32_t pad_id = -1;
    bool    add_bos = false;
    std::string chat_template;
    std::string model_type;
};

// ============================================================================
// Loader API (src/gguf_loader.cpp + src/repack_soa.hip)
// ============================================================================
// tok_embd_host = true (--cpu-embd): token_embd.weight (~469 MiB SoA Q8_0) lives in
// pinned, GPU-mapped host RAM instead of VRAM.  It is only ever gathered one row
// per token (embed + DSpark embed block), so the PCIe cost is ~2.4 KiB/token.
// lm_head (output.weight) always stays in VRAM.
bool load_inkling_gguf(
    const std::string & path,
    ModelWeights & out_weights,
    TokenizerData & out_tok,
    bool verify_repack_self_test = false,
    bool tok_embd_host = false);

// ============================================================================
// Core Wave32 Kernels API (src/kernels_core.hip)
// ============================================================================
// Embed + RMSNorm:
//   Decode: if d_meta != nullptr, reads token_id from d_meta->token_id (n_tokens == 1)
//   Prefill: reads token_ids[0..n_tokens-1] from device pointer d_token_ids
void launch_embed_norm(
    const SoaQ8Matrix & tok_embd, const float * tok_norm,
    const int32_t * d_token_ids, const StepMeta * d_meta,
    float * out, int n_tokens, hipStream_t stream);

// Depth attention mix:
//   states_ptrs: up to MAX_RES_BLOCKS pointers (first n_blocks are blocks in REVERSE order
//                [blocks.back(), ..., blocks[0]], last entry states[n_blocks] is cur!)
//   Wait — to make it crystal clear and match inkling2.cpp line 176:
//     states[0] = blocks.back(), ..., states[n_blocks-1] = blocks[0], states[n_blocks] = cur
//     Total n_states = n_blocks + 1 (1 <= n_states <= 8).
//   query: [n_q, N_EMBD] (n_q in {1, 3, 4})
//   out:   [n_q, n_tokens, N_EMBD] (each query slice q*n_tokens*N_EMBD is contiguous!)
void launch_inkling_mix(
    const float * query, int n_q,
    const float * const * states, int n_states,
    float * out, int n_tokens, hipStream_t stream);

// Dual independent depth attention mixes in ONE launch (e.g. msg_mix + attn_mix on Pass 1/2!)
void launch_inkling_mix_dual(
    const float * query0, int n_q0, const float * const * states0, int n_states0, float * out0,
    const float * query1, int n_q1, const float * const * states1, int n_states1, float * out1,
    int n_tokens, hipStream_t stream);

// Fused depth mix + per-query RMSNorm:
//   w_norms: host array of n_q device pointers (each [N_EMBD] or nullptr)
//   out:     [n_q, n_tokens, N_EMBD]
void launch_inkling_mix_rms_norm(
    const float * query, int n_q,
    const float * const * states, int n_states,
    const float * const * w_norms,
    float * out, int n_tokens, float eps, hipStream_t stream);

// Loop controller (FP16 weights, FP32 accumulate):
//   gate = down @ concat(rmsnorm(cur), rmsnorm(msg));
//   out  = (1 - tanh(forget @ gate)) * cur + tanh(gain @ gate) * msg
void launch_inkling_ctrl(
    const float * cur, const float * msg,
    const __half * w_down, const __half * w_gain, const __half * w_forget,
    float * tmp_gate, // [n_tokens, 64]
    float * out, int n_tokens, hipStream_t stream);

// Fused 3-way (Q, K, V) loop controller + per-branch RMSNorm in parallel (FP16 weights)!
//   scratch: optional >= 512 floats zero-initialized once (enables multi-WGP cooperative decode!)
void launch_inkling_ctrl_qkv_norm(
    const float * const cur[3],
    const float * msg_base,
    const __half * const w_down[3],
    const __half * const w_gain[3],
    const __half * const w_forget[3],
    const float * const w_norm[3],
    float * tmp_gate,
    float * const out_norm[3],
    int n_tokens, float eps, hipStream_t stream,
    float * scratch = nullptr);

// Single-token (n_tokens == 1) fused ffn_mix (n_q=1) + ctrl_down (FP16 w_down):
//   Writes mixed state to out_mixed [N_EMBD] AND gate to out_gate [LOOP_GATE_RANK] in ONE launch!
void launch_mix_ctrl_down_decode(
    const float * res_query,
    const float * const * states, int n_states,
    const float * msg,
    const __half * w_down,
    float * out_mixed,
    float * out_gate,
    hipStream_t stream);

// Single-token (n_tokens == 1) fused ffn_mix (n_q=1) + FFN RMSNorm + FP16 Router + Top-8:
void launch_mix_ffn_norm_router_topk_decode(
    const float * res_query,
    const float * const * states, int n_states,
    const float * ffn_norm_w,
    const __half * router_w,
    const float * exp_probs_b,
    float * x_norm_out,
    int32_t * topk_ids_out,
    float * topk_w_out,
    float * router_scratch,
    hipStream_t stream);

// Single-token (n_tokens == 1) fused ctrl_apply + FFN RMSNorm + FP16 Router + Top-8:
void launch_ctrl_apply_ffn_norm_router_topk_decode(
    const float * cur, const float * msg, const float * gate,
    const __half * w_gain, const __half * w_forget,
    const float * ffn_norm_w,
    const __half * router_w,
    const float * exp_probs_b,
    float * x_norm_out,
    int32_t * topk_ids_out,
    float * topk_w_out,
    float * router_scratch,
    hipStream_t stream);

void launch_rms_norm(
    const float * x, const float * w, float * out,
    int n_tokens, int dim, float eps, hipStream_t stream);

// Super-fused K/V preparation + KV-cache Q8_0 store (head-major [N_HEAD_KV, cap, ...]):
//   1. Causal shortconv-4 on Kraw (or Vraw) with optional dynamic taps (Global layers) + state update
//   2. If is_v == false: L2 clip(QK_CLIP) + Wave32 FWHT-128
//      If is_v == true:  RMSNorm(RMS_EPS) + Wave32 FWHT-128
//   3. Quantize directly to Q8_0 SoA into kv_slot at positions:
//        Decode (d_meta != nullptr, n_tokens == 1): writes at d_meta->kv_write_{swa|global} and updates k_pos
//        Prefill (d_meta == nullptr): writes token t at (pos_start + t) % cap (if SWA) or (pos_start + t) (if Global)
void launch_sconv_kv_store_q8(
    const float * raw_kv,        // [n_tokens, N_KV] (or strided in packed proj buffer: stride_floats)
    int           raw_stride,    // stride in floats between tokens (N_KV if contiguous, or packed_cols)
    const float * conv_w,        // [N_KV, 4]
    const float * dyn_taps,      // [n_tokens, 32] or nullptr (with dyn_stride)
    int           dyn_stride,    // stride in floats between tokens for dyn_taps
    float       * sconv_hist,    // [N_KV, 3] pointer at channel offset (0 for K, 1024 for V)
    KVCacheSlot   kv_slot,
    bool          is_v,
    int           n_tokens,
    int           pos_start,     // used when d_meta == nullptr
    const StepMeta * d_meta,     // non-null for static-graph decode (n_tokens == 1)
    hipStream_t   stream);

// Maximum concurrent independent stream slots supported by batched decode.
constexpr int MAX_CONCURRENT_SLOTS = 8;

// By-value per-row state pointers for independent multi-stream batched K+V prep/store.
struct BatchKVSlotArgs {
    float *     hist_base[MAX_CONCURRENT_SLOTS]; // per-row sconv_slots_[layer].hist
    KVCacheSlot kv[MAX_CONCURRENT_SLOTS];        // per-row KV cache view (SWA ring or Global slice)
};

// 1-launch dual K+V shortconv + clip/RMSNorm + FWHT-128 + quantized KV store across
// m independent concurrent streams (grid = (N_HEAD_KV, 2, m), block = 32).
void launch_batch_sconv_kv_store_q8(
    const float * k_raw, int k_stride, const float * k_conv_w, const float * k_dyn, int k_dyn_stride,
    const float * v_raw, int v_stride, const float * v_conv_w, const float * v_dyn, int v_dyn_stride,
    BatchKVSlotArgs args,
    int             m,
    const StepMeta * d_meta,
    hipStream_t     stream);

// Fused Q clip + optional tau + FWHT-128, AND R clip + optional tau + rel_table = attn_rel_bias @ R!
//   q_in:  [n_tokens, N_Q] (strided by q_stride) -> q_out: [n_tokens, N_HEAD, HEAD_DIM] contiguous F32
//   r_in:  [n_tokens, N_HEAD * N_REL] (strided by r_stride) -> rel_table: [n_tokens, N_HEAD, N_REL_EXTENT] F32
void launch_q_r_prep_and_reltable(
    const float * q_in, int q_stride,
    const float * q_clip_w,      // [N_HEAD, HEAD_DIM]
    const float * r_in, int r_stride,
    const float * r_clip_w,      // [N_HEAD, N_REL]
    const float * rel_bias_w,    // [N_REL_EXTENT, N_REL]
    float       * q_out,         // [n_tokens, N_HEAD, HEAD_DIM]
    float       * rel_table_out, // [n_tokens, N_HEAD, N_REL_EXTENT]
    bool          apply_tau,     // true for Global layers, false for SWA
    int           n_tokens,
    int           pos_start,
    const StepMeta * d_meta,
    hipStream_t   stream);

// Super-fused decode (n_tokens == 1, d_meta != nullptr):
// Runs Q_prep_FWHT128 + K_sconv_KV_store + V_sconv_KV_store + R_prep_reltable in ONE multi-CTA launch!
void launch_qkvr_prep_fused_decode(
    const float * q_in, int q_stride, const float * q_clip_w,
    const float * r_in, int r_stride, const float * r_clip_w, const float * rel_bias_w,
    const float * k_raw, int k_stride, const float * k_conv_w, const float * k_dyn, int k_dyn_stride,
    const float * v_raw, int v_stride, const float * v_conv_w, const float * v_dyn, int v_dyn_stride,
    float       * sconv_hist_kv, // [2 * N_KV, 3]: K at 0, V at N_KV * N_CONV_HIST
    KVCacheSlot   kv_slot,
    float       * q_out,         // [N_HEAD, HEAD_DIM]
    float       * rel_table_out, // [N_HEAD, N_REL_EXTENT]
    bool          apply_tau,
    const StepMeta * d_meta,
    hipStream_t   stream);

// Super-fused prefill (n_tokens > 1):
// Runs K_sconv_KV_store + V_sconv_KV_store + inline sconv_hist update + Q_prep_FWHT128 + R_prep_reltable
// with zero separate sconv_hist_update launches!
void launch_qkvr_prep_prefill(
    const float * q_in, int q_stride, const float * q_clip_w,
    const float * r_in, int r_stride, const float * r_clip_w, const float * rel_bias_w,
    const float * k_raw, int k_stride, const float * k_conv_w, const float * k_dyn, int k_dyn_stride,
    const float * v_raw, int v_stride, const float * v_conv_w, const float * v_dyn, int v_dyn_stride,
    float       * sconv_hist_kv, // [2 * N_KV, 3]: K at 0, V at N_KV * N_CONV_HIST
    KVCacheSlot   kv_slot,
    float       * q_out,         // [n_tokens, N_HEAD, HEAD_DIM]
    float       * rel_table_out, // [n_tokens, N_HEAD, N_REL_EXTENT]
    bool          apply_tau,
    int           n_tokens,
    int           pos_start,
    hipStream_t   stream);

// Fused post_conv (2304 channels, 4-tap) + state update + optional residual add (out = res + conv(x))
// If res == nullptr, writes out = conv(x) (used at first layer of attn_res block!)
void launch_sconv_post_res(
    const float * x,           // [n_tokens, N_EMBD]
    const float * conv_w,      // [N_EMBD, 4]
    float       * sconv_hist,  // [N_EMBD, 3] at offset 2048 (attn_post) or 4352 (ffn_post)
    const float * res,         // [n_tokens, N_EMBD] or nullptr
    float       * out,         // [n_tokens, N_EMBD]
    int           n_tokens,
    hipStream_t   stream);

// Fused FFN RMSNorm + Router FP16 GEMV/GEMM (2304->128) + sigmoid + exp_probs_b + Top-8 + normalize * 8.0
// Also writes normalized x_norm [n_tokens, N_EMBD] so MoE experts can consume it directly!
// Optional router_scratch (>= 512 floats, zero-initialized once at engine init) enables multi-WGP
// parallel decode router with self-resetting atomic completion barrier (100% HIP-Graph safe!).
void launch_ffn_norm_router_topk(
    const float  * x_in,         // [n_tokens, N_EMBD]
    const float  * ffn_norm_w,   // [N_EMBD]
    const __half * router_w,     // [N_EXPERT, N_EMBD] FP16
    const float  * exp_probs_b,  // [N_EXPERT] F32 (or nullptr)
    float        * x_norm_out,   // [n_tokens, N_EMBD]
    int32_t      * topk_ids_out, // [n_tokens, TOP_K]
    float        * topk_w_out,   // [n_tokens, TOP_K]
    int            n_tokens,
    hipStream_t    stream,
    float        * router_scratch = nullptr);

// Final logit softcapping (30 * tanh(x / 30)) in-place on [n_tokens, VOCAB_UNPADDED]
void launch_logit_softcap(
    float * logits, int n_tokens, int vocab_size, float cap, hipStream_t stream);

// Device-side argmax over [VOCAB_UNPADDED] -> writes token_id to d_out_id
void launch_argmax_f32(
    const float * logits, int vocab_size, int32_t * d_out_id, hipStream_t stream);

// Single-launch fused logit softcap (in-place on [vocab_size]) + cooperative argmax -> d_out_id
// scratch: >= 512 floats zero-initialized once (self-resetting atomic completion counter!)
void launch_logit_softcap_argmax(
    float * logits, int vocab_size, float cap, int32_t * d_out_id,
    float * scratch, hipStream_t stream);

// Same, but additionally advances the device-side StepMeta for the NEXT decode step
// (token_id=argmax, pos+1, kv len/write idx, tau from d_tau_tab[pos]) and records the
// produced token into d_tok_hist[pos]. Enables chaining N captured-graph executions
// back-to-back with a single host sync at the end.
void launch_logit_softcap_argmax_advance(
    float * logits, int vocab_size, float cap, int32_t * d_out_id,
    float * scratch, StepMeta * d_meta_next, const float * d_tau_tab,
    int32_t swa_cap, int32_t * d_tok_hist, int32_t hist_cap, hipStream_t stream);

// ============================================================================
// On-The-Fly RelBias FlashAttention API (src/kernels_fa_relbias.hip)
// ============================================================================
// Decode (n_tokens == 1, d_meta != nullptr):
//   Reads kv_len = is_swa ? d_meta->kv_len_swa : d_meta->kv_len_global
//   Reads q_pos  = d_meta->pos
//   If fa_done != nullptr (>= N_HEAD_KV int32s zero-initialized once), uses in-kernel
//   self-resetting atomic combine to eliminate the separate combine kernel launch!
// Prefill (n_tokens > 1, d_meta == nullptr):
//   Uses GQA-3 Shared-KV RDNA3 FP16 WMMA prefill kernel with fast Global interior causal path
//   + fused FWHT-128 + sigmoid(qgate)!
//   tmp_o is sized >= 8 * MAX_UBATCH * N_HEAD * HEAD_DIM floats and tmp_meta >= 8 * MAX_UBATCH * N_HEAD float2s
//   to support optional up to 8-way split-KV on long-context Global prefill.
void launch_fa_relbias(
    const float * q,             // [n_tokens, N_HEAD, HEAD_DIM]
    KVCacheSlot   kv_slot,
    const float * sinks_in, int sinks_stride,
    const float * rel_table,     // [n_tokens, N_HEAD, N_REL_EXTENT]
    const float * qgate_in, int qgate_stride,
    float       * out,           // [n_tokens, N_Q] (contiguous [n_tokens, 3072])
    float       * tmp_o,         // split-K scratch: >= max(64, 8*MAX_UBATCH) * N_HEAD * HEAD_DIM
    float2      * tmp_meta,      // split-K scratch: >= max(64, 8*MAX_UBATCH) * N_HEAD
    int           n_tokens,
    int           pos_start,
    const StepMeta * d_meta,
    hipStream_t   stream,
    int32_t     * fa_done = nullptr);

// ============================================================================
// Peak-Bandwidth SoA Decode GEMV & Fused MoE API (src/kernels_gemv_soa.hip)
// ============================================================================
// Single-token (M=1) lossless exact-FP32-activation SoA Q8_0 GEMV (128-bit coalesced loads)
void launch_soa_q8_gemv(
    const SoaQ8Matrix & W,
    const float       * x,       // [K]
    float             * y,       // [W.rows]
    hipStream_t         stream);

// Fused 3-input QKV SoA Q8_0 GEMV in ONE kernel launch (for non-shared layers: proj_q4, proj_k, proj_v)
// All 3 matrices must share the same K (N_EMBD = 2304).
void launch_soa_q8_gemv_qkv3(
    const SoaQ8Matrix & W0, const float * x0, float * y0,
    const SoaQ8Matrix & W1, const float * x1, float * y1,
    const SoaQ8Matrix & W2, const float * x2, float * y2,
    hipStream_t         stream);

// Single-token SoA Q8_0 GEMV with fused 4-tap static sconv_post_res epilogue (used for attn wo!)
// Applies y = (1/sqrt(5)) * (x3 + h0*w0 + h1*w1 + h2*w2 + x3*w3) + (res ? res[r] : 0) and updates sconv_hist[r] in place!
void launch_soa_q8_gemv_sconv_post_res(
    const SoaQ8Matrix & W,
    const float       * x,          // [K]
    const float       * conv_w,     // [W.rows, 4]
    float             * sconv_hist, // [W.rows, 3]
    const float       * res,        // [W.rows] or nullptr
    float             * out,        // [W.rows]
    hipStream_t         stream);

// Fused Top-8 MoE NVFP4 Decode:
//   Stage 1: gate_up [128, 2304, 2304] + SITU(4.0, 25.0) -> act [8, 1152]
//   Stage 2: down [128, 2304, 1152] + weighted sum across 8 experts -> optional fused sconv_post_res -> out [2304]
void launch_moe_nvfp4_decode(
    const SoaNVFP4Experts & gate_up,
    const SoaNVFP4Experts & down,
    const float           * x_norm,    // [N_EMBD]
    const int32_t         * topk_ids,  // [TOP_K]
    const float           * topk_w,    // [TOP_K]
    float                 * tmp_act,   // [TOP_K, N_FF_EXP] (8 * 1152 floats)
    float                 * out,       // [N_EMBD]
    hipStream_t             stream,
    const float           * post_conv_w  = nullptr, // [N_EMBD, 4] optional fused sconv_post_res
    float                 * post_sconv_h = nullptr, // [N_EMBD, 3]
    const float           * post_res_add = nullptr, // [N_EMBD]
    float                 * raw_out      = nullptr);// Step 8A: raw pre-conv dump [N_EMBD]

// Fused Top-8 MoE Q8_0 Decode (layers 20..23):
void launch_moe_q8_decode(
    const SoaQ8Experts & gate_up,
    const SoaQ8Experts & down,
    const float        * x_norm,       // [N_EMBD]
    const int32_t      * topk_ids,     // [TOP_K]
    const float        * topk_w,       // [TOP_K]
    float              * tmp_act,      // [TOP_K, N_FF_EXP]
    float              * out,          // [N_EMBD]
    hipStream_t          stream,
    const float        * post_conv_w  = nullptr, // [N_EMBD, 4] optional fused sconv_post_res
    float              * post_sconv_h = nullptr, // [N_EMBD, 3]
    const float        * post_res_add = nullptr, // [N_EMBD]
    float              * raw_out      = nullptr);// Step 8A: raw pre-conv dump [N_EMBD]

// ============================================================================
// Hardware RDNA3 FP16 WMMA Prefill GEMM & Grouped MoE API (src/kernels_wmma_prefill.hip)
// ============================================================================
// Batched (M >= 1) SoA Q8_0 -> FP16 WMMA GEMM: Y[M, N] = X[M, K] @ W[N, K]^T
void launch_soa_q8_gemm_wmma(
    const SoaQ8Matrix & W,
    const float       * X,       // [M, K] row-major
    float             * Y,       // [M, W.rows] row-major
    int                 M,
    hipStream_t         stream);

// Batched 3-input QKV SoA Q8_0 -> FP16 WMMA GEMM in ONE launch (proj_q4, proj_k, proj_v).
// All 3 matrices must share the same K (N_EMBD = 2304).
void launch_soa_q8_gemm_wmma_qkv3(
    const SoaQ8Matrix & W0, const float * X0, float * Y0,
    const SoaQ8Matrix & W1, const float * X1, float * Y1,
    const SoaQ8Matrix & W2, const float * X2, float * Y2,
    int                 M,
    hipStream_t         stream);

// Scratchpad needed for grouped MoE prefill (pre-allocated once in EngineScratch!)
struct MoePrefillScratch {
    int32_t * exp_counts   = nullptr; // [N_EXPERT]
    int32_t * exp_offsets  = nullptr; // [N_EXPERT + 1]
    int32_t * exp_cursors  = nullptr; // [N_EXPERT]
    int32_t * perm_tok     = nullptr; // [MAX_UBATCH * TOP_K] -> original token idx
    float   * perm_w       = nullptr; // [MAX_UBATCH * TOP_K] -> router weight
    float   * perm_x       = nullptr; // [MAX_UBATCH * TOP_K, N_EMBD]
    float   * perm_act     = nullptr; // [MAX_UBATCH * TOP_K, N_FF_EXP] (post-SITU!)
    float   * perm_down    = nullptr; // [MAX_UBATCH * TOP_K, N_EMBD]
    // Tile-to-expert work descriptor table for load-balanced grouped WMMA:
    // Each entry packs (expert_id, tile_m_offset_within_expert, tile_m_count)
    int32_t * tile_exp     = nullptr; // [MAX_UBATCH * TOP_K + N_EXPERT]
    int32_t * tile_m0      = nullptr; // [MAX_UBATCH * TOP_K + N_EXPERT]
    int32_t * d_num_tiles  = nullptr; // [1]
};

// Grouped MoE Prefill (NVFP4 layers 0..19) using RDNA3 FP16 WMMA + fused SITU + weighted combine
// + optional fused 4-tap static sconv_post_res & inline sconv_hist update!
void launch_moe_nvfp4_prefill_wmma(
    const SoaNVFP4Experts & gate_up,
    const SoaNVFP4Experts & down,
    const float           * X_norm,    // [M, N_EMBD]
    const int32_t         * topk_ids,  // [M, TOP_K]
    const float           * topk_w,    // [M, TOP_K]
    MoePrefillScratch     & scratch,
    float                 * Y_out,     // [M, N_EMBD]
    int                     M,
    hipStream_t             stream,
    const float           * post_conv_w  = nullptr, // [N_EMBD, 4] optional fused sconv_post_res
    float                 * post_sconv_h = nullptr, // [N_EMBD, 3]
    const float           * post_res_add = nullptr);// [M, N_EMBD]

// Grouped MoE Prefill (Q8_0 layers 20..23) using RDNA3 FP16 WMMA + fused SITU + weighted combine
// + optional fused 4-tap static sconv_post_res & inline sconv_hist update!
void launch_moe_q8_prefill_wmma(
    const SoaQ8Experts & gate_up,
    const SoaQ8Experts & down,
    const float        * X_norm,       // [M, N_EMBD]
    const int32_t      * topk_ids,     // [M, TOP_K]
    const float        * topk_w,       // [M, TOP_K]
    MoePrefillScratch  & scratch,
    float              * Y_out,        // [M, N_EMBD]
    int                  M,
    hipStream_t          stream,
    const float        * post_conv_w  = nullptr, // [N_EMBD, 4] optional fused sconv_post_res
    float              * post_sconv_h = nullptr, // [N_EMBD, 3]
    const float        * post_res_add = nullptr);// [M, N_EMBD]

// ============================================================================
// 16x16x16 FP16 -> FP32 tile validation (`nib wmma-selftest`, no model needed)
// ============================================================================
// Runs the emulated tile (v_dot2_f32_f16) and the hardware tile (v_wmma, gfx12:
// __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12) over random exact-in-FP16
// 16x16x16 tiles staged in LDS, and compares them element by element in the nib
// D layout. Prints max |diff| and cosine similarity; returns 0 on PASS.
// This is the gate for NIB_GFX12_WMMA=1 (see include/nib_mma.h).
int run_wmma_tile_selftest(int n_tiles);

} // namespace inkling
