#pragma once

#include "inkling_engine.h"

#include <cstdint>
#include <string>
#include <vector>

namespace inkling {

// ============================================================================
// Native RDNA3 Vision Encoder for Inkling 2 Mini (inkling2v / mmproj-Q8_0)
// ============================================================================
// Architecture (verified against mmproj-Inkling-2-Mini-Q8_0.gguf):
//   - Patch Conv2D: 20x20x3 -> 1024 (F32 weights, no bias)
//   - Bilinearly-resized learned pos_embd (32x32 -> grid_w x grid_h)
//   - Prepended CLS token (class_embd + class_position_embd) -> n_pos = n_patches + 1
//   - pre_ln (LayerNorm, eps = 1e-5)
//   - 23 ViT blocks:
//       ln1 -> fused QKV (SoA Q8_0 WMMA [3072, 1024]) + bias -> 2D RoPE (theta=10000)
//       -> Wave32 FP16 WMMA non-causal FlashAttention (H=16, D=64, scale=0.125)
//       -> attn_out (SoA Q8_0 WMMA [1024, 1024]) + bias -> * ls1 + residual
//       ln2 -> ffn_up (SoA Q8_0 WMMA [4096, 1024]) + bias -> GELU_ERF
//       -> ffn_down (SoA Q8_0 WMMA [1024, 4096]) + bias -> * ls2 + residual
//   - Drop CLS, 2x2 spatial patch merge ([grid_h/2 * grid_w/2, 4096])
//   - 2-layer MLP projector: mm.1 (4096->4096 Q8_0) -> GELU_ERF -> mm.2 (4096->2304 Q8_0)
// ============================================================================

struct VisionHParams {
    int32_t n_layer      = 23;
    int32_t n_embd       = 1024;
    int32_t n_head       = 16;
    int32_t head_dim     = 64;
    int32_t n_ff         = 4096;
    int32_t patch_size   = 20;
    int32_t image_size   = 640;
    int32_t n_merge      = 2;
    int32_t proj_dim     = N_EMBD; // 2304
    int32_t max_tokens   = 4096;   // max merged image tokens
    float   ln_eps       = 1e-5f;
    float   rope_theta   = 10000.0f;
    float   image_mean[3] = { 0.5f, 0.5f, 0.5f };
    float   image_std[3]  = { 0.5f, 0.5f, 0.5f };
};

struct VisionLayerWeights {
    float *     ln1_w      = nullptr; // [1024]
    float *     ln1_b      = nullptr; // [1024]
    SoaQ8Matrix qkv_w;                // Fused [3072, 1024] = [q(1024) | k(1024) | v(1024)]
    float *     qkv_b      = nullptr; // [3072]
    SoaQ8Matrix attn_out_w;           // [1024, 1024]
    float *     attn_out_b = nullptr; // [1024]
    float *     ls1_w      = nullptr; // [1024]

    float *     ln2_w      = nullptr; // [1024]
    float *     ln2_b      = nullptr; // [1024]
    SoaQ8Matrix ffn_up_w;             // [4096, 1024]
    float *     ffn_up_b   = nullptr; // [4096]
    SoaQ8Matrix ffn_down_w;           // [1024, 4096]
    float *     ffn_down_b = nullptr; // [1024]
    float *     ls2_w      = nullptr; // [1024]
};

struct VisionEncodeResult {
    int32_t            n_tokens  = 0;     // number of merged image tokens (grid_w/2 * grid_h/2)
    int32_t            grid_w    = 0;     // patch grid width  (W_pad / 20)
    int32_t            grid_h    = 0;     // patch grid height (H_pad / 20)
    int32_t            img_w     = 0;     // original (or resized) width
    int32_t            img_h     = 0;     // original (or resized) height
    double             preproc_ms = 0.0;  // host decode + pad/norm ms
    double             gpu_ms     = 0.0;  // pure GPU vision encoder + projector ms
    const float *      d_embeds   = nullptr; // device pointer [n_tokens, 2304] (valid until next encode)
};

class VisionEncoder {
public:
    VisionEncoder() = default;
    ~VisionEncoder();

    bool init(const std::string & mmproj_path, hipStream_t stream);
    void destroy();

    // Load PNG/JPG/BMP from disk, preprocess, and encode into device [n_tokens, 2304] FP32 embeddings.
    // Optionally copies embeddings to host if out_host_embeds != nullptr.
    bool encode_image_file(
        const std::string & image_path,
        VisionEncodeResult & out_res,
        std::vector<float> * out_host_embeds = nullptr);

    // Decode encoded PNG/JPG/BMP bytes in memory and encode into device embeddings.
    bool encode_image_bytes(
        const uint8_t * data, size_t len,
        VisionEncodeResult & out_res,
        std::vector<float> * out_host_embeds = nullptr);

    // Encode raw RGB uint8 buffer [height, width, 3].
    bool encode_rgb_u8(
        const uint8_t * rgb, int width, int height,
        VisionEncodeResult & out_res,
        std::vector<float> * out_host_embeds = nullptr);

    const VisionHParams & hparams() const { return hp_; }
    // Resident VRAM right now: weights + live transient workspace (0 after an encode
    // unless keep_scratch) + the small persistent embeddings output buffer.
    size_t vram_bytes() const { return total_vram_bytes_; }
    size_t weight_vram_bytes() const { return weight_vram_bytes_; }
    size_t workspace_bytes() const { return arena_bytes_; }            // live transient arena (0 when released)
    size_t output_bytes() const { return proj_out_bytes_; }            // persistent d_embeds buffer
    size_t last_workspace_bytes() const { return last_arena_bytes_; }  // arena size used by the last encode
    size_t peak_workspace_bytes() const { return peak_arena_bytes_; }  // largest arena ever allocated

    // Transient-workspace policy.  Default (false): the single aliased ViT arena is
    // hipFree'd right after each encode (the [n_tokens, 2304] embeddings stay valid
    // in their own small buffer until the next encode).  true: keep the arena
    // resident between encodes (no allocator churn; used by --bench-vision and
    // --vision-keep-scratch).  Switching to false releases the arena immediately.
    void set_keep_scratch(bool keep);
    bool keep_scratch() const { return keep_scratch_; }
    void release_workspace();

    // Bytes of the transient arena + persistent output buffer for an image of
    // `n_merged` merged tokens (n_pos = 4 * n_merged + 1 patches).  Host-only maths;
    // valid before init() (uses the default hparams).
    size_t estimate_encode_bytes(int n_merged) const;
    // Worst case over every accepted image size (hp_.max_tokens merged tokens).
    size_t estimate_max_encode_bytes() const { return estimate_encode_bytes(hp_.max_tokens); }

private:
    bool ensure_scratch(int n_patches, int n_merged, int w_pad, int h_pad);
    bool ensure_output(int n_merged);

    VisionHParams                   hp_;
    hipStream_t                     stream_ = nullptr;
    std::vector<VisionLayerWeights> layers_;

    float *     d_cls_tok_      = nullptr; // [1024] = class_embd + class_position_embd
    float *     d_patch_embd_w_ = nullptr; // [1024, 1200] F32 (reshaped from [20, 20, 3, 1024])
    float *     d_pos_embd_w_   = nullptr; // [32*32, 1024] F32
    float *     d_pre_ln_w_     = nullptr; // [1024]
    float *     d_pre_ln_b_     = nullptr; // [1024]
    SoaQ8Matrix mm1_w_;                    // [4096, 4096]
    SoaQ8Matrix mm2_w_;                    // [2304, 4096]

    std::vector<void *> weight_allocs_;
    size_t weight_vram_bytes_ = 0;

    // Transient ViT workspace: ONE 256-byte-aligned arena, phase-aliased:
    //   [ d_hidden_ | d_norm_ | shared pool ]
    //   pool (size = max over disjoint phases):
    //     patch     : d_img_f32_
    //     attention : d_qkv_ (== d_attn_out_, qkv is dead once split) | d_q_h_ | d_k_h_ | d_v_h_
    //     ffn       : d_ffn_mid_
    //     projector : d_merged_ | d_proj_mid_
    // Released after every encode unless keep_scratch_ (see set_keep_scratch()).
    bool    keep_scratch_       = false;
    void *  d_arena_            = nullptr;
    size_t  arena_bytes_        = 0;
    size_t  last_arena_bytes_   = 0;
    size_t  peak_arena_bytes_   = 0;
    int32_t scratch_max_pos_    = 0;
    int32_t scratch_max_pixels_ = 0;
    int32_t scratch_max_merged_ = 0;
    float * d_img_f32_          = nullptr; // [H_pad * W_pad * 3]
    float * d_hidden_           = nullptr; // [n_pos * 1024]
    float * d_norm_             = nullptr; // [n_pos * 1024]
    float * d_qkv_              = nullptr; // [n_pos * 3072]
    __half * d_q_h_             = nullptr; // [16 * n_pos * 64]
    __half * d_k_h_             = nullptr; // [16 * n_pos * 64]
    __half * d_v_h_             = nullptr; // [16 * n_pos * 64]
    float * d_attn_out_         = nullptr; // [n_pos * 1024] (aliases d_qkv_)
    float * d_ffn_mid_          = nullptr; // [n_pos * 4096]
    float * d_merged_           = nullptr; // [n_merged * 4096]
    float * d_proj_mid_         = nullptr; // [n_merged * 4096]

    // Persistent output (out_res.d_embeds): its own small allocation, grow-only,
    // valid until the next encode (server reads it across prefill chunks).
    float * d_proj_out_         = nullptr; // [n_merged * 2304]
    size_t  proj_out_bytes_     = 0;
    int32_t proj_out_cap_       = 0;

    size_t scratch_vram_bytes_  = 0;       // arena_bytes_ + proj_out_bytes_
    size_t total_vram_bytes_    = 0;
};

// ============================================================================
// Native RDNA3 Vision Kernels (src/kernels_vision.hip)
// ============================================================================
void launch_vision_patch_embed_pos_cls(
    const float * d_img_rgb,     // [H_pad, W_pad, 3] normalized F32
    const float * d_patch_w,     // [1024, 3, 20, 20] F32 (GGML [20, 20, 3, 1024])
    const float * d_pos_embd,    // [pos_grid * pos_grid, 1024] F32
    const float * d_cls_tok,     // [1024] F32
    float *       d_out,         // [n_patches + 1, 1024] F32
    int grid_w, int grid_h, int patch_size, int pos_grid,
    hipStream_t stream);

void launch_vision_layernorm(
    const float * x, const float * w, const float * b, float * y,
    int n_tokens, int D, float eps, hipStream_t stream);

void launch_vision_qkv_bias_rope2d_split(
    const float * qkv,           // [n_pos, 3072] F32
    const float * qkv_b,         // [3072] F32
    __half *      q_h,           // [16, n_pos, 64] FP16
    __half *      k_h,           // [16, n_pos, 64] FP16
    __half *      v_h,           // [16, n_pos, 64] FP16
    int n_pos, int grid_w, float rope_theta, hipStream_t stream);

void launch_vision_fa_wmma_d64(
    const __half * q_h,          // [16, n_pos, 64] FP16
    const __half * k_h,          // [16, n_pos, 64] FP16
    const __half * v_h,          // [16, n_pos, 64] FP16
    float *        out,          // [n_pos, 1024] FP32
    int n_pos, int n_head, float scale, hipStream_t stream);

void launch_vision_bias_ls_res(
    const float * x, const float * bias, const float * ls_w, float * residual,
    int n_tokens, int D, hipStream_t stream);

void launch_vision_bias_gelu_erf(
    float * x, const float * bias, int n_tokens, int D, hipStream_t stream);

void launch_vision_patch_merge(
    const float * src_with_cls,  // [n_patches + 1, 1024] F32
    float *       dst_merged,    // [(grid_h/2)*(grid_w/2), 4096] F32
    int grid_w, int grid_h, int D, int n_merge, hipStream_t stream);

} // namespace inkling
