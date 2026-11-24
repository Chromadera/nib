// ============================================================================
// kernels_vision.hip — Native RDNA3 (gfx1100, Wave32) Vision Encoder Kernels
// ============================================================================
// Implements all non-GEMM ops for Inkling 2 Mini's Vision Encoder (inkling2v):
//   1. vision_patch_embed_pos_cls_kernel : 20x20x3 Conv2D patch embedding +
//      exact bilinear pos_embd resize (pixel_offset=0.5) + CLS prepend.
//   2. vision_layernorm_kernel           : Wave32 LayerNorm (eps=1e-5).
//   3. vision_qkv_bias_rope2d_split      : Fused QKV bias + 2D RoPE (mode 0,
//      theta=10000, CLS=0, patches 1-indexed) + head-major FP16 split.
//   4. vision_fa_wmma_d64_kernel         : Wave32 FP16 WMMA non-causal
//      FlashAttention (H=16, D=64, BR=64, BC=64, register-only softmax,
//      zero LDS for P, bank-conflict-free XOR-swizzled V^T).
//   5. vision_bias_ls_res_kernel         : Fused bias + LayerScale + residual.
//   6. vision_bias_gelu_erf_kernel       : Fused optional bias + exact GELU_ERF.
//   7. vision_patch_merge_kernel         : Drop CLS + 2x2 spatial patch merge.
//
// All linear projections (QKV, attn_out, ffn_up, ffn_down, mm.1, mm.2) reuse
// launch_soa_q8_gemm_wmma() (85 TFLOPS FP16 WMMA) from kernels_wmma_prefill.hip.
// ============================================================================

#include "vision_encoder.hpp"

#include <cmath>
#include <cstdint>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

namespace inkling {
namespace {

typedef _Float16 fp16x16 __attribute__((ext_vector_type(16)));
typedef float    f32x8   __attribute__((ext_vector_type(8)));

__device__ __forceinline__ float warp_reduce_sum(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor(v, off, 32);
    }
    return v;
}

__device__ __forceinline__ float fast_exp2f(float x) {
    return __builtin_amdgcn_exp2f(x);
}

// ---------------------------------------------------------------------------
// 1. Patch Conv2D (20x20x3 -> 1024) + Bilinear Pos Embed + CLS Prepend (WMMA)
// ---------------------------------------------------------------------------
// Grid: (1 + ceil(n_patches / 16), 16)
//   blockIdx.x == 0: writes CLS token (64 channels per blockIdx.y)
//   blockIdx.x >= 1: tiles 16 patches x 64 output channels with 4 warps (16 ch/warp)
//   using RDNA3 FP16 WMMA (16x16x16), 512 B LDS for s_p[16][16], 16x weight reuse!
__global__ void __launch_bounds__(128)
vision_patch_embed_pos_cls_kernel(
    const float * __restrict__ img_rgb,   // [H_pad, W_pad, 3]
    const float * __restrict__ patch_w,   // [1024, 3, 20, 20] = [1024, 1200]
    const float * __restrict__ pos_embd,  // [pos_grid * pos_grid, 1024]
    const float * __restrict__ cls_tok,   // [1024]
    float * __restrict__       out,       // [n_patches + 1, 1024]
    int grid_w, int grid_h, int patch_size, int pos_grid)
{
    const int oc_base   = blockIdx.y * 64;
    const int tid       = threadIdx.x; // 0..127
    const int n_patches = grid_w * grid_h;
    constexpr int D     = 1024;

    if (blockIdx.x == 0) {
        if (tid < 64) {
            const int oc = oc_base + tid;
            out[oc] = cls_tok[oc];
        }
        return;
    }

    const int p0    = (blockIdx.x - 1) * 16;
    const int warp  = tid >> 5;  // 0..3
    const int lane  = tid & 31;  // 0..31
    const int l16   = lane & 15; // 0..15
    const int lhalf = lane >> 4; // 0 or 1

    const int w_pad = grid_w * patch_size;
    const int p2    = patch_size * patch_size; // 400
    const int kdim  = 3 * p2;                  // 1200

    const int oc_row = oc_base + warp * 16 + l16;
    const float * wrow = patch_w + (size_t)oc_row * kdim;

    __shared__ __align__(16) _Float16 s_p[16][16];
    f32x8 acc = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };

    for (int k0 = 0; k0 < kdim; k0 += 16) {
        // Cooperatively stage 16 patches x 16 k-values = 256 FP16 into LDS (2 elems/thread)
#pragma unroll
        for (int it = 0; it < 2; ++it) {
            const int idx   = tid + it * 128; // 0..255
            const int p_loc = idx >> 4;       // 0..15
            const int k_loc = idx & 15;       // 0..15
            const int p_idx = p0 + p_loc;
            const int k     = k0 + k_loc;
            _Float16 val = (_Float16)0.0f;
            if (p_idx < n_patches && k < kdim) {
                const int py  = p_idx / grid_w;
                const int px  = p_idx - py * grid_w;
                const int c   = k / p2;
                const int rem = k - c * p2;
                const int ky  = rem / patch_size;
                const int kx  = rem - ky * patch_size;
                const int iy  = py * patch_size + ky;
                const int ix  = px * patch_size + kx;
                val = (_Float16)img_rgb[((size_t)iy * w_pad + ix) * 3 + c];
            }
            s_p[p_loc][k_loc] = val;
        }
        __syncthreads();

        // Load A = W[oc_row, k0..k0+15] (4 x float4 -> fp16x16)
        fp16x16 w_frag;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            const float4 fw = *reinterpret_cast<const float4 *>(wrow + k0 + q * 4);
            w_frag[q * 4 + 0] = (_Float16)fw.x;
            w_frag[q * 4 + 1] = (_Float16)fw.y;
            w_frag[q * 4 + 2] = (_Float16)fw.z;
            w_frag[q * 4 + 3] = (_Float16)fw.w;
        }

        // Load B = P[p0 + l16, k0..k0+15] from LDS
        fp16x16 p_frag;
        const uint4 lo = *reinterpret_cast<const uint4 *>(&s_p[l16][0]);
        const uint4 hi = *reinterpret_cast<const uint4 *>(&s_p[l16][8]);
        const _Float16 * hp0 = reinterpret_cast<const _Float16 *>(&lo);
        const _Float16 * hp1 = reinterpret_cast<const _Float16 *>(&hi);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            p_frag[i]     = hp0[i];
            p_frag[i + 8] = hp1[i];
        }

        acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(w_frag, p_frag, acc);
        __syncthreads();
    }

    // Output: lane (lhalf, l16) owns patch p_idx = p0 + l16, channels oc = oc_base + warp*16 + 2*i + lhalf
    const int p_idx = p0 + l16;
    if (p_idx < n_patches) {
        const int py = p_idx / grid_w;
        const int px = p_idx - py * grid_w;

        // Exact ggml upscale_f32_bilinear (pixel_offset = 0.5f) for position embedding (computed ONCE per patch!)
        const float sf_x = (float)grid_w / (float)pos_grid;
        const float sf_y = (float)grid_h / (float)pos_grid;
        const float x_f  = ((float)px + 0.5f) / sf_x - 0.5f;
        const float y_f  = ((float)py + 0.5f) / sf_y - 0.5f;

        int x0 = (int)floorf(x_f);
        int x1 = x0 + 1;
        x0 = max(0, min(x0, pos_grid - 1));
        x1 = max(0, min(x1, pos_grid - 1));
        const float dx = fminf(1.0f, fmaxf(0.0f, x_f - (float)x0));

        int y0 = (int)floorf(y_f);
        int y1 = y0 + 1;
        y0 = max(0, min(y0, pos_grid - 1));
        y1 = max(0, min(y1, pos_grid - 1));
        const float dy = fminf(1.0f, fmaxf(0.0f, y_f - (float)y0));

        const float wa = (1.0f - dx) * (1.0f - dy);
        const float wb = dx * (1.0f - dy);
        const float wc = (1.0f - dx) * dy;
        const float wd = dx * dy;

        const float * pa_row = pos_embd + (size_t)(y0 * pos_grid + x0) * D;
        const float * pb_row = pos_embd + (size_t)(y0 * pos_grid + x1) * D;
        const float * pc_row = pos_embd + (size_t)(y1 * pos_grid + x0) * D;
        const float * pd_row = pos_embd + (size_t)(y1 * pos_grid + x1) * D;
        float * out_row      = out + (size_t)(p_idx + 1) * D;

#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int oc = oc_base + warp * 16 + 2 * i + lhalf;
            const float pos_val = pa_row[oc] * wa + pb_row[oc] * wb + pc_row[oc] * wc + pd_row[oc] * wd;
            out_row[oc] = acc[i] + pos_val;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. Wave32 LayerNorm (D = 1024, 1 block per token, 256 threads = 8 warps)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
vision_layernorm_kernel(
    const float * __restrict__ x,
    const float * __restrict__ w,
    const float * __restrict__ b,
    float * __restrict__       y,
    int D, float eps)
{
    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const float * xr = x + (size_t)row * D;
    float *       yr = y + (size_t)row * D;

    // Fast path for D == 1024: each thread holds 4 floats (1 float4) in registers
    if (D == 1024) {
        const float4 xv = reinterpret_cast<const float4 *>(xr)[tid];
        float sum = xv.x + xv.y + xv.z + xv.w;
        sum = warp_reduce_sum(sum);
        __shared__ float s_buf[8];
        if (lane == 0) s_buf[warp] = sum;
        __syncthreads();
        float mean = (lane < 8) ? s_buf[lane] : 0.0f;
        mean = warp_reduce_sum(mean) * (1.0f / 1024.0f);

        const float d0 = xv.x - mean;
        const float d1 = xv.y - mean;
        const float d2 = xv.z - mean;
        const float d3 = xv.w - mean;
        float vsum = d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
        vsum = warp_reduce_sum(vsum);
        if (lane == 0) s_buf[warp] = vsum;
        __syncthreads();
        float var = (lane < 8) ? s_buf[lane] : 0.0f;
        var = warp_reduce_sum(var) * (1.0f / 1024.0f);
        const float inv_std = rsqrtf(var + eps);

        const float4 wv = reinterpret_cast<const float4 *>(w)[tid];
        const float4 bv = reinterpret_cast<const float4 *>(b)[tid];
        float4 ov;
        ov.x = fmaf(d0 * inv_std, wv.x, bv.x);
        ov.y = fmaf(d1 * inv_std, wv.y, bv.y);
        ov.z = fmaf(d2 * inv_std, wv.z, bv.z);
        ov.w = fmaf(d3 * inv_std, wv.w, bv.w);
        reinterpret_cast<float4 *>(yr)[tid] = ov;
        return;
    }

    // Generic fallback
    __shared__ float s_buf[8];
    float sum = 0.0f;
    for (int i = tid; i < D; i += 256) sum += xr[i];
    sum = warp_reduce_sum(sum);
    if (lane == 0) s_buf[warp] = sum;
    __syncthreads();
    float mean = (lane < 8) ? s_buf[lane] : 0.0f;
    mean = warp_reduce_sum(mean) / (float)D;

    float vsum = 0.0f;
    for (int i = tid; i < D; i += 256) {
        float d = xr[i] - mean;
        vsum += d * d;
    }
    vsum = warp_reduce_sum(vsum);
    if (lane == 0) s_buf[warp] = vsum;
    __syncthreads();
    float var = (lane < 8) ? s_buf[lane] : 0.0f;
    var = warp_reduce_sum(var) / (float)D;
    const float inv_std = rsqrtf(var + eps);

    for (int i = tid; i < D; i += 256) {
        yr[i] = fmaf((xr[i] - mean) * inv_std, w[i], b[i]);
    }
}

// ---------------------------------------------------------------------------
// 3. Fused QKV Bias + 2D RoPE (mode 0, theta=10000) + Head-Major FP16 Split
// ---------------------------------------------------------------------------
// Input:  qkv [n_pos, 3072] FP32 (concatenated [Q(1024), K(1024), V(1024)]),
//         qkv_b [3072] FP32.
// Output: q_h, k_h, v_h each [16, n_pos, 64] FP16.
// Grid: (n_pos), Block: 256 threads (each thread handles 2 pairs = 4 dims of Q, K, V
//       across 16 heads * 32 pairs = 512 pairs per token!).
__global__ void __launch_bounds__(256)
vision_qkv_bias_rope2d_split_kernel(
    const float * __restrict__  qkv,
    const float * __restrict__  qkv_b,
    __half * __restrict__       q_h,
    __half * __restrict__       k_h,
    __half * __restrict__       v_h,
    int n_pos, int grid_w, float rope_theta)
{
    const int t = blockIdx.x;
    if (t >= n_pos) return;

    // 2D positions: CLS (t == 0) -> (0, 0); patch p = t - 1 -> (x + 1, y + 1)
    int pw = 0, ph = 0;
    if (t > 0) {
        const int p = t - 1;
        ph = p / grid_w + 1;
        pw = (p - (ph - 1) * grid_w) + 1;
    }

    const float * row = qkv + (size_t)t * 3072;
    const float theta_scale = powf(rope_theta, -2.0f / 32.0f);

    // 512 adjacent pairs across 16 heads x 32 pairs/head -> 2 pairs per thread
#pragma unroll
    for (int iter = 0; iter < 2; ++iter) {
        const int pair_global = threadIdx.x + iter * 256; // 0..511
        const int h           = pair_global >> 5;         // 0..15
        const int p_in_head   = pair_global & 31;         // 0..31
        const int d0          = p_in_head * 2;            // 0, 2, ..., 62
        const int c0          = h * 64 + d0;              // 0..1022

        const bool is_second_half = (p_in_head >= 16);
        const int  freq_idx       = is_second_half ? (p_in_head - 16) : p_in_head;
        const int  pos_val        = is_second_half ? ph : pw;
        const float angle         = (float)pos_val * powf(theta_scale, (float)freq_idx);
        float s_ang, c_ang;
        __sincosf(angle, &s_ang, &c_ang);

        // Q
        const float q0 = row[c0]     + qkv_b[c0];
        const float q1 = row[c0 + 1] + qkv_b[c0 + 1];
        const float rq0 = q0 * c_ang - q1 * s_ang;
        const float rq1 = q0 * s_ang + q1 * c_ang;

        // K
        const float k0 = row[1024 + c0]     + qkv_b[1024 + c0];
        const float k1 = row[1024 + c0 + 1] + qkv_b[1024 + c0 + 1];
        const float rk0 = k0 * c_ang - k1 * s_ang;
        const float rk1 = k0 * s_ang + k1 * c_ang;

        // V (no RoPE)
        const float v0 = row[2048 + c0]     + qkv_b[2048 + c0];
        const float v1 = row[2048 + c0 + 1] + qkv_b[2048 + c0 + 1];

        const size_t dst_off = ((size_t)h * n_pos + t) * 64 + d0;
        *reinterpret_cast<__half2 *>(q_h + dst_off) = __floats2half2_rn(rq0, rq1);
        *reinterpret_cast<__half2 *>(k_h + dst_off) = __floats2half2_rn(rk0, rk1);
        *reinterpret_cast<__half2 *>(v_h + dst_off) = __floats2half2_rn(v0, v1);
    }
}

// ---------------------------------------------------------------------------
// 4. Native RDNA3 Wave32 FP16 WMMA Non-Causal FlashAttention (H=16, D=64)
// ---------------------------------------------------------------------------
// BR = 64 (4 warps x 16 Q-rows/warp), BC = 64 (4 K/V tiles of 16), D = 64.
// Uses S^T = K @ Q^T and O^T = V^T @ P^T so every lane owns 1 single Q-row:
//   - Online softmax is 100% register + 1 __shfl_xor(16) (ZERO LDS for P!).
//   - K is staged in s_k[64][72] FP16 (9,216 B).
//   - V is staged transposed + XOR-swizzled in s_vt[64][64] FP16 (8,192 B),
//     giving bank-conflict-free ds_read_b128 for V^T fragments!
//   - Total LDS = 17,408 B -> 3 blocks/CU (12 warps/CU), 0 spills!
__global__ void __launch_bounds__(128)
vision_fa_wmma_d64_kernel(
    const __half * __restrict__ q_h,   // [n_head, n_pos, 64]
    const __half * __restrict__ k_h,   // [n_head, n_pos, 64]
    const __half * __restrict__ v_h,   // [n_head, n_pos, 64]
    float * __restrict__        out,   // [n_pos, n_head * 64]
    int n_pos, int n_head, float scale)
{
    constexpr int BR = 64;
    constexpr int BC = 64;
    constexpr int D  = 64;
    constexpr int K_STRIDE = 72; // 64 + 8 padding for bank-free b128 reads

    const int q_block = blockIdx.x;
    const int h       = blockIdx.y;
    const int tid     = threadIdx.x; // 0..127
    const int warp    = tid >> 5;    // 0..3
    const int lane    = tid & 31;    // 0..31
    const int l16     = lane & 15;   // 0..15 (owns Q-row within warp's 16 rows)
    const int lhalf   = lane >> 4;   // 0 or 1 (even/odd index)

    const int q_row_base = q_block * BR + warp * 16;
    const int q_row      = q_row_base + l16;
    const bool q_valid   = (q_row < n_pos);

    const __half * q_head = q_h + (size_t)h * n_pos * D;
    const __half * k_head = k_h + (size_t)h * n_pos * D;
    const __half * v_head = v_h + (size_t)h * n_pos * D;

    // Load this warp's 16x64 Q tile permanently into registers (4 x fp16x16, pre-scaled by scale * LOG2E)
    constexpr float LOG2E = 1.44269504088896340736f;
    fp16x16 q_frag[4];
    {
        const __half * qp = q_head + (size_t)(q_valid ? q_row : 0) * D;
        const float sc = q_valid ? (scale * LOG2E) : 0.0f;
#pragma unroll
        for (int dk = 0; dk < 4; ++dk) {
            const uint4 raw = reinterpret_cast<const uint4 *>(qp)[dk * 2];
            const uint4 raw2 = reinterpret_cast<const uint4 *>(qp)[dk * 2 + 1];
            const __half * h0 = reinterpret_cast<const __half *>(&raw);
            const __half * h1 = reinterpret_cast<const __half *>(&raw2);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                q_frag[dk][i]     = (_Float16)(__half2float(h0[i]) * sc);
                q_frag[dk][i + 8] = (_Float16)(__half2float(h1[i]) * sc);
            }
        }
    }

    __shared__ __align__(16) __half s_k[BC][K_STRIDE];
    __shared__ __align__(16) __half s_vt[D][BC]; // XOR-swizzled on 8-half chunks: k_swiz = k ^ ((d >> 3) << 3)

    // Accumulator O^T[64, 16]: 4 tiles of 16x16 -> lane holds O[q_row, dt*16 + 2*i + lhalf]
    f32x8 o_acc[4];
#pragma unroll
    for (int dt = 0; dt < 4; ++dt) {
        o_acc[dt] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
    }
    float row_m = -INFINITY;
    float row_l = 0.0f;
    const uint32_t perm0 = (lhalf == 0) ? 0x05040100u : 0x01000504u;
    const uint32_t perm1 = (lhalf == 0) ? 0x07060302u : 0x03020706u;

    // Loop over K/V blocks of BC=64
    for (int kv0 = 0; kv0 < n_pos; kv0 += BC) {
        // 1. Collaboratively load K[kv0..kv0+63, 0..63] into s_k[64][72]
        //    and V[kv0..kv0+63, 0..63] transposed into s_vt[64][64] (XOR-swizzled)
        const int r_in_tile = tid >> 3;     // 0..15
        const int c_chunk   = tid & 7;      // 0..7 (each chunk is 8 FP16 = 16 bytes)
        const int d_base    = c_chunk * 8;  // 0, 8, ..., 56

#pragma unroll
        for (int it = 0; it < 4; ++it) {
            const int r    = r_in_tile + it * 16; // 0..63
            const int g_kv = kv0 + r;
            uint4 kv_raw = { 0u, 0u, 0u, 0u };
            uint4 vv_raw = { 0u, 0u, 0u, 0u };
            if (g_kv < n_pos) {
                kv_raw = reinterpret_cast<const uint4 *>(k_head + (size_t)g_kv * D)[c_chunk];
                vv_raw = reinterpret_cast<const uint4 *>(v_head + (size_t)g_kv * D)[c_chunk];
            }
            *reinterpret_cast<uint4 *>(&s_k[r][d_base]) = kv_raw;

            const __half * vh = reinterpret_cast<const __half *>(&vv_raw);
            const int r_swiz = r ^ (c_chunk << 3);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                s_vt[d_base + j][r_swiz] = vh[j];
            }
        }
        __syncthreads();

        // 2. Compute S^T = K @ Q^T -> 4 tiles of 16x16 (covering BC=64 K-columns)
        f32x8 s_tile[4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            s_tile[kt] = { 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f };
            const int k_row = kt * 16 + l16;
#pragma unroll
            for (int dk = 0; dk < 4; ++dk) {
                fp16x16 k_frag;
                const uint4 lo = *reinterpret_cast<const uint4 *>(&s_k[k_row][dk * 16]);
                const uint4 hi = *reinterpret_cast<const uint4 *>(&s_k[k_row][dk * 16 + 8]);
                const _Float16 * p0 = reinterpret_cast<const _Float16 *>(&lo);
                const _Float16 * p1 = reinterpret_cast<const _Float16 *>(&hi);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    k_frag[i]     = p0[i];
                    k_frag[i + 8] = p1[i];
                }
                s_tile[kt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k_frag, q_frag[dk], s_tile[kt]);
            }
        }

        // 3. Pure log2-domain online softmax over BC=64 (1-cycle v_exp_f32 + conditional O rescale)
        float tile_m = -INFINITY;
        const bool is_tail = (kv0 + BC > n_pos);
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                if (is_tail && (kv0 + kt * 16 + 2 * i + lhalf) >= n_pos) {
                    s_tile[kt][i] = -INFINITY;
                }
                tile_m = fmaxf(tile_m, s_tile[kt][i]);
            }
        }
        tile_m = fmaxf(tile_m, __shfl_xor(tile_m, 16, 32));

        const float new_m = fmaxf(row_m, tile_m);
        if (new_m > row_m) {
            if (row_m > -INFINITY) {
                const float alpha = fast_exp2f(row_m - new_m);
                row_l *= alpha;
#pragma unroll
                for (int dt = 0; dt < 4; ++dt) {
#pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        o_acc[dt][i] *= alpha;
                    }
                }
            }
            row_m = new_m;
        }

        // Exponentiate S -> P using 1-cycle fast_exp2f and pack via v_perm_b32
        float tile_l = 0.0f;
        fp16x16 p_frag[4];
#pragma unroll
        for (int kt = 0; kt < 4; ++kt) {
            union {
                fp16x16  h16;
                uint32_t u[8];
            } pk;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float e0 = fast_exp2f(s_tile[kt][2 * j]     - row_m);
                const float e1 = fast_exp2f(s_tile[kt][2 * j + 1] - row_m);
                tile_l += e0 + e1;
                const uint32_t mu = __builtin_bit_cast(uint32_t, __floats2half2_rn(e0, e1));
                const uint32_t ou = __shfl_xor(mu, 16, 32);
                pk.u[2 * j + 0] = __builtin_amdgcn_perm(ou, mu, perm0);
                pk.u[2 * j + 1] = __builtin_amdgcn_perm(ou, mu, perm1);
            }
            p_frag[kt] = pk.h16;
        }
        tile_l += __shfl_xor(tile_l, 16, 32);
        row_l  += tile_l;

        // 4. O^T += V^T @ P^T (4 D-tiles x 4 K-tiles = 16 WMMAs)
#pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
            const int d_row   = dt * 16 + l16;
            const int swiz_hi = (d_row >> 3) << 3; // matches write swizzle
#pragma unroll
            for (int kt = 0; kt < 4; ++kt) {
                fp16x16 v_frag;
                const int k0 = (kt * 16 + 0) ^ swiz_hi;
                const int k1 = (kt * 16 + 8) ^ swiz_hi;
                const uint4 lo = *reinterpret_cast<const uint4 *>(&s_vt[d_row][k0]);
                const uint4 hi = *reinterpret_cast<const uint4 *>(&s_vt[d_row][k1]);
                const _Float16 * p0 = reinterpret_cast<const _Float16 *>(&lo);
                const _Float16 * p1 = reinterpret_cast<const _Float16 *>(&hi);
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    v_frag[i]     = p0[i];
                    v_frag[i + 8] = p1[i];
                }
                o_acc[dt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(v_frag, p_frag[kt], o_acc[dt]);
            }
        }
        __syncthreads();
    }

    // 5. Final normalize & write O[q_row, h*64 + d] directly to [n_pos, n_head*64]
    if (q_valid) {
        const float inv_l = (row_l > 0.0f) ? (1.0f / row_l) : 0.0f;
        float * out_row = out + (size_t)q_row * (n_head * D) + h * D;
#pragma unroll
        for (int dt = 0; dt < 4; ++dt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                out_row[dt * 16 + 2 * i + lhalf] = o_acc[dt][i] * inv_l;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 5. Fused Bias + LayerScale + Residual: residual += (x + bias) * ls_w
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
vision_bias_ls_res_kernel(
    const float * __restrict__ x,
    const float * __restrict__ bias,
    const float * __restrict__ ls_w,
    float * __restrict__       residual,
    int n_tokens, int D)
{
    const size_t idx4 = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total4 = ((size_t)n_tokens * D) >> 2;
    if (idx4 >= total4) return;

    const int d4 = (int)(idx4 % (D >> 2));
    const float4 xv = reinterpret_cast<const float4 *>(x)[idx4];
    const float4 bv = reinterpret_cast<const float4 *>(bias)[d4];
    const float4 lv = reinterpret_cast<const float4 *>(ls_w)[d4];
    float4       rv = reinterpret_cast<const float4 *>(residual)[idx4];

    rv.x = fmaf(xv.x + bv.x, lv.x, rv.x);
    rv.y = fmaf(xv.y + bv.y, lv.y, rv.y);
    rv.z = fmaf(xv.z + bv.z, lv.z, rv.z);
    rv.w = fmaf(xv.w + bv.w, lv.w, rv.w);
    reinterpret_cast<float4 *>(residual)[idx4] = rv;
}

// ---------------------------------------------------------------------------
// 6. Fused Optional Bias + Exact GELU_ERF (in-place)
// ---------------------------------------------------------------------------
__device__ __forceinline__ float gelu_erf_f32(float x) {
    constexpr float INV_SQRT2 = 0.70710678118654752440f;
    return 0.5f * x * (1.0f + erff(x * INV_SQRT2));
}

__global__ void __launch_bounds__(256)
vision_bias_gelu_erf_kernel(
    float * __restrict__       x,
    const float * __restrict__ bias,
    int n_tokens, int D)
{
    const size_t idx4   = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total4 = ((size_t)n_tokens * D) >> 2;
    if (idx4 >= total4) return;

    float4 v = reinterpret_cast<const float4 *>(x)[idx4];
    if (bias != nullptr) {
        const int d4 = (int)(idx4 % (D >> 2));
        const float4 b = reinterpret_cast<const float4 *>(bias)[d4];
        v.x += b.x; v.y += b.y; v.z += b.z; v.w += b.w;
    }
    v.x = gelu_erf_f32(v.x);
    v.y = gelu_erf_f32(v.y);
    v.z = gelu_erf_f32(v.z);
    v.w = gelu_erf_f32(v.w);
    reinterpret_cast<float4 *>(x)[idx4] = v;
}

// ---------------------------------------------------------------------------
// 7. Drop CLS + 2x2 Spatial Patch Merge (exact build_patch_merge_permute)
// ---------------------------------------------------------------------------
// Input:  src_with_cls [1 + grid_h * grid_w, D]
// Output: dst_merged   [(grid_h/m) * (grid_w/m), m*m*D]
// For each merged patch (wy, wx), concatenates (dy=0..m-1, dx=0..m-1) in row-major order.
__global__ void __launch_bounds__(256)
vision_patch_merge_kernel(
    const float * __restrict__ src_with_cls,
    float * __restrict__       dst_merged,
    int grid_w, int grid_h, int D, int m)
{
    const int mw     = grid_w / m;
    const int m_idx  = blockIdx.x; // 0 .. (grid_h/m)*(grid_w/m) - 1
    const int wy     = m_idx / mw;
    const int wx     = m_idx - wy * mw;
    const int out_D4 = (m * m * D) >> 2;
    const int D4     = D >> 2;

    const float4 * src4 = reinterpret_cast<const float4 *>(src_with_cls + D); // skip CLS!
    float4 *       dst4 = reinterpret_cast<float4 *>(dst_merged + (size_t)m_idx * (m * m * D));

    for (int i = threadIdx.x; i < out_D4; i += 256) {
        const int sub = i / D4;         // 0 .. m*m - 1
        const int d4  = i - sub * D4;
        const int dy  = sub / m;
        const int dx  = sub - dy * m;
        const int sy  = wy * m + dy;
        const int sx  = wx * m + dx;
        const int p   = sy * grid_w + sx;
        dst4[i] = src4[(size_t)p * D4 + d4];
    }
}

} // namespace

// ============================================================================
// Public Launchers
// ============================================================================
void launch_vision_patch_embed_pos_cls(
    const float * d_img_rgb, const float * d_patch_w, const float * d_pos_embd,
    const float * d_cls_tok, float * d_out,
    int grid_w, int grid_h, int patch_size, int pos_grid, hipStream_t stream)
{
    const int n_patches = grid_w * grid_h;
    const int n_pblk    = (n_patches + 15) / 16;
    vision_patch_embed_pos_cls_kernel<<<dim3(1 + n_pblk, 16), 128, 0, stream>>>(
        d_img_rgb, d_patch_w, d_pos_embd, d_cls_tok, d_out,
        grid_w, grid_h, patch_size, pos_grid);
}

void launch_vision_layernorm(
    const float * x, const float * w, const float * b, float * y,
    int n_tokens, int D, float eps, hipStream_t stream)
{
    vision_layernorm_kernel<<<n_tokens, 256, 0, stream>>>(x, w, b, y, D, eps);
}

void launch_vision_qkv_bias_rope2d_split(
    const float * qkv, const float * qkv_b,
    __half * q_h, __half * k_h, __half * v_h,
    int n_pos, int grid_w, float rope_theta, hipStream_t stream)
{
    vision_qkv_bias_rope2d_split_kernel<<<n_pos, 256, 0, stream>>>(
        qkv, qkv_b, q_h, k_h, v_h, n_pos, grid_w, rope_theta);
}

void launch_vision_fa_wmma_d64(
    const __half * q_h, const __half * k_h, const __half * v_h,
    float * out, int n_pos, int n_head, float scale, hipStream_t stream)
{
    const int n_qblk = (n_pos + 63) / 64;
    vision_fa_wmma_d64_kernel<<<dim3(n_qblk, n_head), 128, 0, stream>>>(
        q_h, k_h, v_h, out, n_pos, n_head, scale);
}

void launch_vision_bias_ls_res(
    const float * x, const float * bias, const float * ls_w, float * residual,
    int n_tokens, int D, hipStream_t stream)
{
    const size_t total4 = ((size_t)n_tokens * D) >> 2;
    const int blocks = (int)((total4 + 255) / 256);
    vision_bias_ls_res_kernel<<<blocks, 256, 0, stream>>>(
        x, bias, ls_w, residual, n_tokens, D);
}

void launch_vision_bias_gelu_erf(
    float * x, const float * bias, int n_tokens, int D, hipStream_t stream)
{
    const size_t total4 = ((size_t)n_tokens * D) >> 2;
    const int blocks = (int)((total4 + 255) / 256);
    vision_bias_gelu_erf_kernel<<<blocks, 256, 0, stream>>>(x, bias, n_tokens, D);
}

void launch_vision_patch_merge(
    const float * src_with_cls, float * dst_merged,
    int grid_w, int grid_h, int D, int n_merge, hipStream_t stream)
{
    const int n_merged = (grid_w / n_merge) * (grid_h / n_merge);
    vision_patch_merge_kernel<<<n_merged, 256, 0, stream>>>(
        src_with_cls, dst_merged, grid_w, grid_h, D, n_merge);
}

} // namespace inkling
