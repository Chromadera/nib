// Native RDNA3/RDNA4 DFlash 2 speculative drafter kernels (Wave32 HIP).
//
// Implements incoai/Inkling-2-Mini-DFlash2:
//  - Load-time BF16 -> SoA Q8_0 repacking (1 warp per 32-elem block) and
//    precomputed mask token embedding
//  - m-row (1..8) SoA Q8_0 GEMV with RDNA3/4 v_dot2_f32_f16 + padded half2 LDS
//  - Commit path: fc -> hidden_norm -> K/V projection -> k_norm + NeoX RoPE (theta=1e4)
//    -> BF16 sliding-window ring (WINDOW=2048)
//  - 5-layer bidirectional block (m=8) decoder with 2-tap grouped dynamic depthwise
//    convolutions (group_size=16, num_groups=144) wrapping attention and MLP
//  - Split-K bidirectional block + sliding-window (2048) GQA=4 attention with
//    empty-tile skipping for O(1) static-graph replay across all sequence lengths
//  - Exact 2-stage Top-16 over VOCAB_UNPADDED=200058 raw logits
//  - Single-CTA greedy bilinear Candidate Path Selector (rank=256, top_k=16,
//    softcap=30*tanh(x/30) applied after correction)

#include "dflash2_mtp_kernels.h"

#include <cmath>
#include <cstdint>

namespace inkling {

static constexpr int   DF2_H          = N_EMBD;            // 2304
static constexpr int   DF2_HEADS      = 32;
static constexpr int   DF2_KVH        = 8;
static constexpr int   DF2_GQA        = DF2_HEADS / DF2_KVH; // 4
static constexpr int   DF2_HD         = HEAD_DIM;          // 128
static constexpr int   DF2_QDIM       = DF2_HEADS * DF2_HD;  // 4096
static constexpr int   DF2_KVDIM      = DF2_KVH * DF2_HD;    // 1024
static constexpr int   DF2_QKV        = DF2_QDIM + 2 * DF2_KVDIM; // 6144
static constexpr int   DF2_WINDOW     = 2048;
static constexpr int   DF2_BLOCK      = 8;
static constexpr int   DF2_GAMMA      = DF2_BLOCK - 1;     // 7
static constexpr int   DF2_CONV_GRP   = DF2_H / 16;        // 144
static constexpr int   DF2_CONV_PROJ  = 2 * 2 * DF2_CONV_GRP; // 576
static constexpr int   DF2_SEL_RANK   = 256;
static constexpr int   DF2_SEL_TOPK   = 16;
static constexpr int   DF2_ROW_TH     = DF2_H / 8;         // 288 (9 warps)
static constexpr float DF2_ROPE_THETA = 10000.0f;
static constexpr float DF2_EPS        = 1e-6f;
static constexpr float DF2_KQ_SCALE   = 0.08838834764831845f; // 1/sqrt(128)
static constexpr float DF2_SOFTCAP    = 30.0f;
static constexpr float DF2_INV_SCAP   = 1.0f / 30.0f;

typedef int v4i_df2 __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------------------
// BF16 / reduction / argmax helpers
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float df2_bf2f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}

static __device__ __forceinline__ uint16_t df2_f2bf(float x) {
    uint32_t u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

static __device__ __forceinline__ float df2_bfr(float x) {
    return df2_bf2f(df2_f2bf(x));
}

static __device__ __forceinline__ float df2_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}

static __device__ __forceinline__ float df2_warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    return v;
}

static __device__ __forceinline__ uint64_t df2_warp_max_u64(uint64_t v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const uint32_t lo = __shfl_xor((uint32_t)v, o, 32);
        const uint32_t hi = __shfl_xor((uint32_t)(v >> 32), o, 32);
        const uint64_t other = ((uint64_t)hi << 32) | (uint64_t)lo;
        if (other > v) v = other;
    }
    return v;
}

static __device__ __forceinline__ uint32_t df2_float_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static __device__ __forceinline__ float df2_unkey_float(uint32_t k) {
    const uint32_t u = (k & 0x80000000u) ? (k & 0x7fffffffu) : ~k;
    return __uint_as_float(u);
}

static __device__ __forceinline__ uint64_t df2_pack_argmax(float score, uint32_t index) {
    return ((uint64_t)df2_float_key(score) << 32) | (uint64_t)(0xffffffffu - index);
}

static __device__ __forceinline__ uint32_t df2_unpack_argmax(uint64_t p) {
    return 0xffffffffu - (uint32_t)p;
}

struct Df2Scratch {
    float s[16];
};

static __device__ __forceinline__ float df2_block_sum(float v, Df2Scratch & sc) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nw   = blockDim.x >> 5;
    v = df2_warp_sum(v);
    if (lane == 0) sc.s[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < nw; ++w) t += sc.s[w];
    return t;
}

static __device__ __forceinline__ void df2_load8_bf16(const uint16_t * p, float (&v)[8]) {
    const uint4 u = *reinterpret_cast<const uint4 *>(p);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i]     = df2_bf2f((uint16_t)(w[i] & 0xffffu));
        v[2 * i + 1] = df2_bf2f((uint16_t)(w[i] >> 16));
    }
}

static __device__ __forceinline__ void df2_store8_bf16(uint16_t * p, const float (&v)[8]) {
    uint32_t w[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[i] = (uint32_t)df2_f2bf(v[2 * i]) | ((uint32_t)df2_f2bf(v[2 * i + 1]) << 16);
    }
    *reinterpret_cast<uint4 *>(p) = make_uint4(w[0], w[1], w[2], w[3]);
}

static __device__ __forceinline__ float df2_rms_scale(const float (&x)[8], Df2Scratch & sc) {
    float ss = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) ss += x[j] * x[j];
    return rsqrtf(df2_block_sum(ss, sc) / (float)DF2_H + DF2_EPS);
}

// Exact unbiased int8 -> half2 dot via RDNA3/4 v_dot2_f32_f16
static __device__ __forceinline__ float df2_q8_word_dot_fdot2(uint32_t w, uint32_t xa, uint32_t xb, float acc) {
    const uint32_t u   = w ^ 0x80808080u;
    const __half2  b   = __builtin_bit_cast(__half2, 0x64806480u);
    const __half2  h01 = __hsub2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u)), b);
    const __half2  h23 = __hsub2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u)), b);
    acc = __builtin_amdgcn_fdot2(h01, __builtin_bit_cast(__half2, xa), acc, false);
    acc = __builtin_amdgcn_fdot2(h23, __builtin_bit_cast(__half2, xb), acc, false);
    return acc;
}

// ---------------------------------------------------------------------------
// Graph parameter upload
// ---------------------------------------------------------------------------
static __global__ void df2_set_params_kernel(DFlash2DraftParams * dst, DFlash2DraftParams val) {
    *dst = val;
}

void launch_dflash2_set_params(DFlash2DraftParams * d_dp, DFlash2DraftParams val,
                               hipStream_t stream) {
    df2_set_params_kernel<<<1, 1, 0, stream>>>(d_dp, val);
}

// ---------------------------------------------------------------------------
// Load-time: BF16 -> SoA Q8_0 (1 warp per 32-elem block -> 0 spills, coalesced)
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(256) void df2_repack_q8_kernel(
        const uint16_t * __restrict__ src, int rows, int K,
        int8_t * __restrict__ qs, __half * __restrict__ d) {
    const int nb = K >> 5;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int blk_idx = blockIdx.x * 8 + warp;
    if (blk_idx >= rows * nb) return;
    const int r = blk_idx / nb;
    const int b = blk_idx - r * nb;
    const size_t elem = (size_t)r * K + (size_t)b * 32 + lane;
    const float v = df2_bf2f(src[elem]);
    const float amax = df2_warp_max(fabsf(v));
    const __half dh = __float2half(amax * (1.0f / 127.0f));
    const float dhf = __half2float(dh);
    const float inv = (dhf > 0.0f) ? (1.0f / dhf) : 0.0f;
    if (lane == 0) d[(size_t)r * nb + b] = dh;
    const float qf = rintf(v * inv);
    qs[elem] = (int8_t)(qf > 127.0f ? 127.0f : (qf < -127.0f ? -127.0f : qf));
}

void launch_dflash2_repack_q8(const uint16_t * bf16, int rows, int K,
                              int8_t * qs, __half * d, hipStream_t stream) {
    const int total_blks = rows * (K >> 5);
    if (total_blks <= 0) return;
    df2_repack_q8_kernel<<<(total_blks + 7) / 8, 256, 0, stream>>>(bf16, rows, K, qs, d);
}

// Precompute bf16(RMSNorm(dequant(tok_embd[mask_token_id]), tok_norm, 1e-6)) [2304]
static __global__ __launch_bounds__(DF2_ROW_TH) void df2_precompute_mask_embed_kernel(
        const int8_t * __restrict__ emb_q, const __half * __restrict__ emb_d,
        const float * __restrict__ tok_norm, int32_t mask_token_id,
        uint16_t * __restrict__ mask_embed_out) {
    __shared__ Df2Scratch sc;
    const int c = 8 * threadIdx.x;
    const int nb = DF2_H >> 5;
    const int b = c >> 5;
    const int sub = c & 31;
    const float sc_q = __half2float(emb_d[(size_t)mask_token_id * nb + b]);
    const int8_t * row = emb_q + (size_t)mask_token_id * DF2_H + (size_t)b * 32 + sub;
    float v[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) v[j] = (float)row[j] * sc_q;
    const float rs = df2_rms_scale(v, sc);
    float y[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(__fmul_rn(v[j], rs), tok_norm[c + j]);
    df2_store8_bf16(mask_embed_out + c, y);
}

void launch_dflash2_precompute_mask_embed(const SoaQ8Matrix & tok_embd,
                                          const float * tok_norm,
                                          int32_t mask_token_id,
                                          uint16_t * mask_embed_out,
                                          hipStream_t stream) {
    df2_precompute_mask_embed_kernel<<<1, DF2_ROW_TH, 0, stream>>>(
        tok_embd.qs, tok_embd.d, tok_norm, mask_token_id, mask_embed_out);
}

// ---------------------------------------------------------------------------
// m-row (1..8) SoA Q8_0 GEMV: RDNA3/4 fdot2 + bank-conflict-free half2 LDS
// ---------------------------------------------------------------------------
static constexpr int DF2_GTILE = 256; // 8 Q8_0 blocks per LDS tile
static constexpr int DF2_Q8H_STRIDE = 20; // 16 half2 + 4 half2 pad per 32-dim block

template <int MODE> // 0 = static m, 2 = m from dp->ncommit (1..8)
static __global__ __launch_bounds__(256) void df2_gemv_q8_kernel(
        const int8_t * __restrict__ qs, const __half * __restrict__ dd,
        int row_offset, int rows, int K,
        const uint16_t * __restrict__ x_bf16, int xstride,
        int m, float * __restrict__ y, int ystride,
        const DFlash2DraftParams * __restrict__ dp) {
    constexpr int LPR = 8;
    constexpr int RPW = 4;
    if constexpr (MODE == 2) {
        m = dp->ncommit;
    }

    __shared__ __half2 s_x[DF2_BLOCK][8 * DF2_Q8H_STRIDE]; // 8 * 160 * 4 = 5120 B

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int grp  = lane >> 3;  // 0..3 (row within warp)
    const int l    = lane & 7;   // 0..7 (block within tile)

    const int out_row = blockIdx.x * (8 * RPW) + warp * RPW + grp;
    const bool valid_row = (out_row < rows);
    const int w_row = row_offset + (valid_row ? out_row : 0);
    const int nb = K >> 5;
    const int8_t * wr = qs + (size_t)w_row * K;
    const __half * dr = dd + (size_t)w_row * nb;

    float acc[DF2_BLOCK];
    #pragma unroll
    for (int r = 0; r < DF2_BLOCK; ++r) acc[r] = 0.0f;

    const int n_chunks = m * (DF2_GTILE / 4); // m * 64 uint2 (4 bf16 each)
    for (int k0 = 0; k0 < K; k0 += DF2_GTILE) {
        // Stage [m, 256] BF16 -> padded __half2 in LDS (<= 2 iterations/thread).
        for (int idx = tid; idx < n_chunks; idx += 256) {
            const int r  = idx >> 6;       // 0..m-1
            const int c4 = idx & 63;       // 0..63 (4-elem chunk)
            const uint2 u = *reinterpret_cast<const uint2 *>(x_bf16 + (size_t)r * xstride + k0 + c4 * 4);
            const float f0 = df2_bf2f((uint16_t)(u.x & 0xffffu));
            const float f1 = df2_bf2f((uint16_t)(u.x >> 16));
            const float f2 = df2_bf2f((uint16_t)(u.y & 0xffffu));
            const float f3 = df2_bf2f((uint16_t)(u.y >> 16));
            const int blk = c4 >> 3;       // 0..7
            const int off = (c4 & 7) << 1; // 0,2,...,14
            s_x[r][blk * DF2_Q8H_STRIDE + off]     = __floats2half2_rn(f0, f1);
            s_x[r][blk * DF2_Q8H_STRIDE + off + 1] = __floats2half2_rn(f2, f3);
        }
        __syncthreads();

        if (valid_row) {
            const int kb = (k0 >> 5) + l;
            const float ds = __half2float(dr[kb]);
            const v4i_df2 * wp = reinterpret_cast<const v4i_df2 *>(wr + (size_t)kb * 32);
            const v4i_df2 w0 = wp[0];
            const v4i_df2 w1 = wp[1];
            const uint32_t q0 = (uint32_t)w0[0], q1 = (uint32_t)w0[1];
            const uint32_t q2 = (uint32_t)w0[2], q3 = (uint32_t)w0[3];
            const uint32_t q4 = (uint32_t)w1[0], q5 = (uint32_t)w1[1];
            const uint32_t q6 = (uint32_t)w1[2], q7 = (uint32_t)w1[3];

            #pragma unroll 1
            for (int r = 0; r < m; ++r) {
                const uint4 * xp = reinterpret_cast<const uint4 *>(&s_x[r][l * DF2_Q8H_STRIDE]);
                const uint4 x0 = xp[0];
                const uint4 x1 = xp[1];
                const uint4 x2 = xp[2];
                const uint4 x3 = xp[3];
                float p = 0.0f;
                p = df2_q8_word_dot_fdot2(q0, x0.x, x0.y, p);
                p = df2_q8_word_dot_fdot2(q1, x0.z, x0.w, p);
                p = df2_q8_word_dot_fdot2(q2, x1.x, x1.y, p);
                p = df2_q8_word_dot_fdot2(q3, x1.z, x1.w, p);
                p = df2_q8_word_dot_fdot2(q4, x2.x, x2.y, p);
                p = df2_q8_word_dot_fdot2(q5, x2.z, x2.w, p);
                p = df2_q8_word_dot_fdot2(q6, x3.x, x3.y, p);
                p = df2_q8_word_dot_fdot2(q7, x3.z, x3.w, p);
                acc[r] = fmaf(p, ds, acc[r]);
            }
        }
        __syncthreads();
    }

    if (valid_row) {
        #pragma unroll 1
        for (int r = 0; r < m; ++r) {
            float v = acc[r];
            v += __shfl_xor(v, 4, LPR);
            v += __shfl_xor(v, 2, LPR);
            v += __shfl_xor(v, 1, LPR);
            if (l == 0) y[(size_t)r * ystride + out_row] = v;
        }
    }
}

void launch_dflash2_gemv(const SoaQ8Matrix & W, int row_offset, int rows,
                         const uint16_t * x_bf16, int xstride,
                         int m, float * y, int ystride, hipStream_t stream,
                         const DFlash2DraftParams * dp, int mode) {
    if (rows <= 0 || m <= 0) return;
    const int grid = (rows + 31) / 32;
    if (mode == 2) {
        df2_gemv_q8_kernel<2><<<grid, 256, 0, stream>>>(
            W.qs, W.d, row_offset, rows, W.K, x_bf16, xstride, m, y, ystride, dp);
    } else {
        df2_gemv_q8_kernel<0><<<grid, 256, 0, stream>>>(
            W.qs, W.d, row_offset, rows, W.K, x_bf16, xstride, m, y, ystride, dp);
    }
}

// ---------------------------------------------------------------------------
// Commit: hidden_norm (f32 -> bf16) and context K/V store (BF16 ring)
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(DF2_ROW_TH) void df2_rmsnorm_f32_to_bf16_kernel(
        const float * __restrict__ x, const uint16_t * __restrict__ w,
        uint16_t * __restrict__ out, const DFlash2DraftParams * __restrict__ dp, int mode) {
    __shared__ Df2Scratch sc;
    const int r = blockIdx.x;
    if (mode == 1 && r >= dp->ncommit) return;
    const int c = 8 * threadIdx.x;
    const float4 a = *reinterpret_cast<const float4 *>(x + (size_t)r * DF2_H + c);
    const float4 b = *reinterpret_cast<const float4 *>(x + (size_t)r * DF2_H + c + 4);
    float v[8] = {df2_bfr(a.x), df2_bfr(a.y), df2_bfr(a.z), df2_bfr(a.w),
                  df2_bfr(b.x), df2_bfr(b.y), df2_bfr(b.z), df2_bfr(b.w)};
    const float rs = df2_rms_scale(v, sc);
    float wv[8];
    df2_load8_bf16(w + c, wv);
    float y[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(__fmul_rn(v[j], rs), wv[j]);
    df2_store8_bf16(out + (size_t)r * DF2_H + c, y);
}

void launch_dflash2_rmsnorm_f32_to_bf16(const float * x, int n, const uint16_t * w,
                                        uint16_t * out, hipStream_t stream,
                                        const DFlash2DraftParams * dp, int mode) {
    if (n <= 0) return;
    df2_rmsnorm_f32_to_bf16_kernel<<<n, DF2_ROW_TH, 0, stream>>>(x, w, out, dp, mode);
}

// Per-head RMSNorm (eps=1e-6) + NeoX RoPE (theta=10000.0) on HEAD_DIM=128
static __device__ __forceinline__ float df2_norm_rope_head(
        float x, const uint16_t * w, int pos, Df2Scratch & sc, float * pair) {
    const int d = threadIdx.x;
    const float ss = df2_block_sum(x * x, sc);
    const float rs = rsqrtf(ss / (float)DF2_HD + DF2_EPS);
    const float y = df2_bfr(__fmul_rn(__fmul_rn(x, rs), df2_bf2f(w[d])));
    pair[d] = y;
    __syncthreads();
    constexpr int half = DF2_HD / 2;
    const int i = d & (half - 1);
    const float inv = powf(DF2_ROPE_THETA, (-(float)i * 2.0f) / (float)DF2_HD);
    const float ang = __fmul_rn((float)pos, inv);
    const float co = cosf(ang), si = sinf(ang);
    const float x1 = pair[i], x2 = pair[i + half];
    const float o = (d < half) ? __fsub_rn(__fmul_rn(x1, co), __fmul_rn(x2, si))
                               : __fadd_rn(__fmul_rn(x2, co), __fmul_rn(x1, si));
    __syncthreads();
    return df2_bfr(o);
}

static __global__ __launch_bounds__(DF2_HD) void df2_ctx_kv_store_kernel(
        const float * __restrict__ kv, int start, const uint16_t * __restrict__ k_norm,
        uint16_t * __restrict__ k_ctx, uint16_t * __restrict__ v_ctx,
        int32_t * __restrict__ ctx_pos, const DFlash2DraftParams * __restrict__ dp) {
    __shared__ Df2Scratch sc;
    __shared__ float pair[DF2_HD];
    const int r = blockIdx.x;
    const int g = blockIdx.y;
    const int d = threadIdx.x;
    if (dp != nullptr) {
        if (r >= dp->ncommit) return;
        start = dp->pos;
    }
    const int pos  = start + r;
    const int slot = pos & (DF2_WINDOW - 1);
    const float * row = kv + (size_t)r * (2 * DF2_KVDIM);
    const float k = df2_norm_rope_head(df2_bfr(row[g * DF2_HD + d]), k_norm, pos, sc, pair);
    const float v = row[DF2_KVDIM + g * DF2_HD + d];
    const size_t at = ((size_t)slot * DF2_KVH + g) * DF2_HD + d;
    k_ctx[at] = df2_f2bf(k);
    v_ctx[at] = df2_f2bf(v);
    if (ctx_pos != nullptr && g == 0 && d == 0) ctx_pos[slot] = pos;
}

void launch_dflash2_ctx_kv_store(const float * kv, int n, int start,
                                 const uint16_t * k_norm,
                                 uint16_t * k_ctx, uint16_t * v_ctx,
                                 int32_t * ctx_pos, hipStream_t stream,
                                 const DFlash2DraftParams * dp) {
    if (n <= 0) return;
    df2_ctx_kv_store_kernel<<<dim3(n, DF2_KVH), DF2_HD, 0, stream>>>(
        kv, start, k_norm, k_ctx, v_ctx, ctx_pos, dp);
}

// ---------------------------------------------------------------------------
// Draft backbone: embed_block, conv_in, conv_out_add_rmsnorm, final_norm_pred,
// qkv_rope, silu_mul
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(DF2_ROW_TH) void df2_embed_block_kernel(
        const int8_t * __restrict__ emb_q, const __half * __restrict__ emb_d,
        const float * __restrict__ tok_norm, const uint16_t * __restrict__ mask_embed,
        const uint16_t * __restrict__ in_norm, int anchor,
        uint16_t * __restrict__ res, uint16_t * __restrict__ xn,
        const DFlash2DraftParams * __restrict__ dp) {
    __shared__ Df2Scratch sc;
    if (dp != nullptr) anchor = dp->anchor;
    const int r = blockIdx.x;
    const int c = 8 * threadIdx.x;
    float x[8];
    if (r == 0) {
        const int nb = DF2_H >> 5;
        const int b = c >> 5;
        const int sub = c & 31;
        const float sc_q = __half2float(emb_d[(size_t)anchor * nb + b]);
        const int8_t * row = emb_q + (size_t)anchor * DF2_H + (size_t)b * 32 + sub;
        float v[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = (float)row[j] * sc_q;
        const float rs = df2_rms_scale(v, sc);
        #pragma unroll
        for (int j = 0; j < 8; ++j) x[j] = df2_bfr(__fmul_rn(__fmul_rn(v[j], rs), tok_norm[c + j]));
    } else {
        df2_load8_bf16(mask_embed + c, x);
    }
    df2_store8_bf16(res + (size_t)r * DF2_H + c, x);

    const float rs2 = df2_rms_scale(x, sc);
    float w[8];
    df2_load8_bf16(in_norm + c, w);
    float y[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(__fmul_rn(x[j], rs2), w[j]);
    df2_store8_bf16(xn + (size_t)r * DF2_H + c, y);
}

void launch_dflash2_embed_block(const SoaQ8Matrix & tok_embd, const float * tok_norm,
                                const uint16_t * mask_embed, const uint16_t * in_norm,
                                int anchor, uint16_t * res, uint16_t * xn,
                                hipStream_t stream, const DFlash2DraftParams * dp) {
    df2_embed_block_kernel<<<DF2_BLOCK, DF2_ROW_TH, 0, stream>>>(
        tok_embd.qs, tok_embd.d, tok_norm, mask_embed, in_norm, anchor, res, xn, dp);
}

// Dynamic 2-tap grouped depthwise convolution (side 0 = input).
// Each thread handles 8 channels inside ONE group g = threadIdx.x >> 1 (group_size=16).
static __global__ __launch_bounds__(DF2_ROW_TH) void df2_conv_in_kernel(
        const uint16_t * __restrict__ xn, const float * __restrict__ coeff,
        const uint16_t * __restrict__ base_kernel, uint16_t * __restrict__ out) {
    const int r = blockIdx.x;
    const int c = 8 * threadIdx.x;
    const int g = threadIdx.x >> 1; // c / 16 in 0..143

    // side=0: coeff[r, 0, tap, g] -> r*576 + tap*144 + g
    const float d0 = df2_bfr(coeff[(size_t)r * DF2_CONV_PROJ + g]);
    const float d1 = df2_bfr(coeff[(size_t)r * DF2_CONV_PROJ + DF2_CONV_GRP + g]);

    float x0[8], b0[8];
    df2_load8_bf16(xn + (size_t)r * DF2_H + c, x0);
    df2_load8_bf16(base_kernel + c, b0); // side=0, tap=0

    float y[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) {
        y[j] = df2_bfr((b0[j] + d0) * x0[j]);
    }
    if (r > 0) {
        float x1[8], b1[8];
        df2_load8_bf16(xn + (size_t)(r - 1) * DF2_H + c, x1);
        df2_load8_bf16(base_kernel + DF2_H + c, b1); // side=0, tap=1
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            y[j] = df2_bfr(y[j] + df2_bfr((b1[j] + d1) * x1[j]));
        }
    }
    df2_store8_bf16(out + (size_t)r * DF2_H + c, y);
}

void launch_dflash2_conv_in(const uint16_t * xn, const float * coeff,
                            const uint16_t * base_kernel, uint16_t * out,
                            hipStream_t stream) {
    df2_conv_in_kernel<<<DF2_BLOCK, DF2_ROW_TH, 0, stream>>>(xn, coeff, base_kernel, out);
}

// Fused output conv (side 1) + residual add + optional next RMSNorm.
static __global__ __launch_bounds__(DF2_ROW_TH) void df2_conv_out_add_rmsnorm_kernel(
        const float * __restrict__ sub_out, const float * __restrict__ coeff,
        const uint16_t * __restrict__ base_kernel, uint16_t * __restrict__ res,
        const uint16_t * __restrict__ norm_w, uint16_t * __restrict__ xn_out) {
    __shared__ Df2Scratch sc;
    const int r = blockIdx.x;
    const int c = 8 * threadIdx.x;
    const int g = threadIdx.x >> 1;

    // side=1: coeff[r, 1, tap, g] -> r*576 + (2 + tap)*144 + g
    const float d0 = df2_bfr(coeff[(size_t)r * DF2_CONV_PROJ + 2 * DF2_CONV_GRP + g]);
    const float d1 = df2_bfr(coeff[(size_t)r * DF2_CONV_PROJ + 3 * DF2_CONV_GRP + g]);

    const float4 s0a = *reinterpret_cast<const float4 *>(sub_out + (size_t)r * DF2_H + c);
    const float4 s0b = *reinterpret_cast<const float4 *>(sub_out + (size_t)r * DF2_H + c + 4);
    const float x0[8] = {df2_bfr(s0a.x), df2_bfr(s0a.y), df2_bfr(s0a.z), df2_bfr(s0a.w),
                         df2_bfr(s0b.x), df2_bfr(s0b.y), df2_bfr(s0b.z), df2_bfr(s0b.w)};

    float b0[8];
    df2_load8_bf16(base_kernel + 2 * DF2_H + c, b0); // side=1, tap=0

    float cy[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) cy[j] = df2_bfr((b0[j] + d0) * x0[j]);

    if (r > 0) {
        const float4 s1a = *reinterpret_cast<const float4 *>(sub_out + (size_t)(r - 1) * DF2_H + c);
        const float4 s1b = *reinterpret_cast<const float4 *>(sub_out + (size_t)(r - 1) * DF2_H + c + 4);
        const float x1[8] = {df2_bfr(s1a.x), df2_bfr(s1a.y), df2_bfr(s1a.z), df2_bfr(s1a.w),
                             df2_bfr(s1b.x), df2_bfr(s1b.y), df2_bfr(s1b.z), df2_bfr(s1b.w)};
        float b1[8];
        df2_load8_bf16(base_kernel + 3 * DF2_H + c, b1); // side=1, tap=1
        #pragma unroll
        for (int j = 0; j < 8; ++j) cy[j] = df2_bfr(cy[j] + df2_bfr((b1[j] + d1) * x1[j]));
    }

    float rv[8];
    df2_load8_bf16(res + (size_t)r * DF2_H + c, rv);
    #pragma unroll
    for (int j = 0; j < 8; ++j) rv[j] = df2_bfr(rv[j] + cy[j]);
    df2_store8_bf16(res + (size_t)r * DF2_H + c, rv);

    if (norm_w != nullptr) {
        const float rs = df2_rms_scale(rv, sc);
        float nw[8];
        df2_load8_bf16(norm_w + c, nw);
        float y[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) y[j] = __fmul_rn(__fmul_rn(rv[j], rs), nw[j]);
        df2_store8_bf16(xn_out + (size_t)r * DF2_H + c, y);
    }
}

void launch_dflash2_conv_out_add_rmsnorm(const float * sub_out, const float * coeff,
                                         const uint16_t * base_kernel, uint16_t * res,
                                         const uint16_t * norm_w, uint16_t * xn_out,
                                         hipStream_t stream) {
    df2_conv_out_add_rmsnorm_kernel<<<DF2_BLOCK, DF2_ROW_TH, 0, stream>>>(
        sub_out, coeff, base_kernel, res, norm_w, xn_out);
}

// Final norm on prediction rows 1..7 -> BF16 [7, 2304] + FP32 [7, 2304]
static __global__ __launch_bounds__(DF2_ROW_TH) void df2_final_norm_pred_kernel(
        const uint16_t * __restrict__ res, const uint16_t * __restrict__ norm_w,
        uint16_t * __restrict__ h_bf16, float * __restrict__ h_f32) {
    __shared__ Df2Scratch sc;
    const int e = blockIdx.x;       // 0..6
    const int r = e + 1;            // block row 1..7
    const int c = 8 * threadIdx.x;
    float rv[8], nw[8];
    df2_load8_bf16(res + (size_t)r * DF2_H + c, rv);
    df2_load8_bf16(norm_w + c, nw);
    const float rs = df2_rms_scale(rv, sc);
    float y[8];
    #pragma unroll
    for (int j = 0; j < 8; ++j) y[j] = df2_bfr(__fmul_rn(__fmul_rn(rv[j], rs), nw[j]));
    df2_store8_bf16(h_bf16 + (size_t)e * DF2_H + c, y);
    *reinterpret_cast<float4 *>(h_f32 + (size_t)e * DF2_H + c)     = make_float4(y[0], y[1], y[2], y[3]);
    *reinterpret_cast<float4 *>(h_f32 + (size_t)e * DF2_H + c + 4) = make_float4(y[4], y[5], y[6], y[7]);
}

void launch_dflash2_final_norm_pred(const uint16_t * res, const uint16_t * norm_w,
                                    uint16_t * h_bf16, float * h_f32, hipStream_t stream) {
    df2_final_norm_pred_kernel<<<DF2_GAMMA, DF2_ROW_TH, 0, stream>>>(res, norm_w, h_bf16, h_f32);
}

// Block Q/K/V prep: grid (8 rows, 32 q + 8 k + 8 v = 48), block 128
static __global__ __launch_bounds__(DF2_HD) void df2_qkv_rope_kernel(
        const float * __restrict__ qkv, const uint16_t * __restrict__ q_norm,
        const uint16_t * __restrict__ k_norm, int P, uint16_t * __restrict__ q_out,
        uint16_t * __restrict__ k_ctx, uint16_t * __restrict__ v_ctx,
        const DFlash2DraftParams * __restrict__ dp) {
    __shared__ Df2Scratch sc;
    __shared__ float pair[DF2_HD];
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int d = threadIdx.x;
    if (dp != nullptr) P = dp->pos;
    const int pos = P + r;
    const int slot = DF2_WINDOW + r;
    const float * row = qkv + (size_t)r * DF2_QKV;
    if (h < DF2_HEADS) {
        const float qv = df2_norm_rope_head(df2_bfr(row[h * DF2_HD + d]), q_norm, pos, sc, pair);
        q_out[((size_t)r * DF2_HEADS + h) * DF2_HD + d] = df2_f2bf(qv);
    } else if (h < DF2_HEADS + DF2_KVH) {
        const int g = h - DF2_HEADS;
        const float kv = df2_norm_rope_head(df2_bfr(row[DF2_QDIM + g * DF2_HD + d]), k_norm, pos, sc, pair);
        k_ctx[((size_t)slot * DF2_KVH + g) * DF2_HD + d] = df2_f2bf(kv);
    } else {
        const int g = h - DF2_HEADS - DF2_KVH;
        const float vv = row[DF2_QDIM + DF2_KVDIM + g * DF2_HD + d];
        v_ctx[((size_t)slot * DF2_KVH + g) * DF2_HD + d] = df2_f2bf(vv);
    }
}

void launch_dflash2_qkv_rope(const float * qkv, const uint16_t * q_norm,
                             const uint16_t * k_norm, int P, uint16_t * q_out,
                             uint16_t * k_ctx, uint16_t * v_ctx, hipStream_t stream,
                             const DFlash2DraftParams * dp) {
    df2_qkv_rope_kernel<<<dim3(DF2_BLOCK, DF2_HEADS + 2 * DF2_KVH), DF2_HD, 0, stream>>>(
        qkv, q_norm, k_norm, P, q_out, k_ctx, v_ctx, dp);
}

// out = bf16(silu(gate) * up)
static __global__ __launch_bounds__(256) void df2_silu_mul_kernel(
        const float * __restrict__ gu, int inter, uint16_t * __restrict__ out) {
    const int r = blockIdx.y;
    const int c = blockIdx.x * 256 + threadIdx.x;
    if (c >= inter) return;
    const float * row = gu + (size_t)r * (2 * inter);
    const float g = df2_bfr(row[c]);
    const float u = df2_bfr(row[inter + c]);
    const float s = g / (1.0f + expf(-g));
    out[(size_t)r * inter + c] = df2_f2bf(s * u);
}

void launch_dflash2_silu_mul(const float * gu, int rows, int inter, uint16_t * out,
                             hipStream_t stream) {
    if (rows <= 0 || inter <= 0) return;
    df2_silu_mul_kernel<<<dim3((inter + 255) / 256, rows), 256, 0, stream>>>(gu, inter, out);
}

// ---------------------------------------------------------------------------
// Split-K bidirectional block + sliding-window (2048) attention (GQA=4)
// 9 chunks: chunks 0..7 cover 256 context ring slots each (empty-tile skipped),
//           chunk 8 covers the 8 bidirectional block slots (2048..2055).
// Each CTA handles 2 Q heads x 8 rows = 16 warps = 512 threads (blockIdx.z in 0..1).
// ---------------------------------------------------------------------------
static constexpr int DF2_ATT_CHUNKS = 9;
static constexpr int DF2_ATT_TILE   = 16;
static constexpr int DF2_ATT_NQ     = DF2_BLOCK * DF2_GQA; // 32 queries per KV head

int64_t dflash2_attention_ws_floats() {
    // o[9][8][32][128] + ml[9][8][32][2]
    return (int64_t)DF2_ATT_CHUNKS * DF2_KVH * DF2_ATT_NQ * (DF2_HD + 2);
}

static __global__ __launch_bounds__(512) void df2_attn_partial_kernel(
        const uint16_t * __restrict__ q,
        const uint16_t * __restrict__ k_ctx,
        const uint16_t * __restrict__ v_ctx,
        const int32_t  * __restrict__ ctx_pos,
        int P, float * __restrict__ ws,
        const DFlash2DraftParams * __restrict__ dp) {
    __shared__ float s_k[DF2_ATT_TILE][DF2_HD];
    __shared__ float s_v[DF2_ATT_TILE][DF2_HD];
    __shared__ int32_t s_pos[DF2_ATT_TILE];
    __shared__ int32_t s_any;

    if (dp != nullptr) P = dp->pos;

    const int ci   = blockIdx.x; // 0..8
    const int g    = blockIdx.y; // 0..7 (KV head)
    const int half = blockIdx.z; // 0..1 (2 Q heads per half)

    const int tid  = threadIdx.x;
    const int warp = tid >> 5;   // 0..15
    const int lane = tid & 31;   // 0..31

    const int hh   = half * 2 + (warp >> 3); // 0..3 within KV group
    const int row  = warp & 7;               // 0..7
    const int h    = g * DF2_GQA + hh;       // 0..31

    const uint16_t * qrow = q + ((size_t)row * DF2_HEADS + h) * DF2_HD;
    const uint2 qw = *reinterpret_cast<const uint2 *>(qrow + lane * 4);
    const float qv[4] = {
        df2_bf2f((uint16_t)(qw.x & 0xffffu)),
        df2_bf2f((uint16_t)(qw.x >> 16)),
        df2_bf2f((uint16_t)(qw.y & 0xffffu)),
        df2_bf2f((uint16_t)(qw.y >> 16)),
    };

    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m_run = -INFINITY;
    float l_run = 0.0f;

    const int k0 = ci * 256;
    const int k1 = (ci < 8) ? (k0 + 256) : (DF2_WINDOW + DF2_BLOCK);

    for (int j0 = k0; j0 < k1; j0 += DF2_ATT_TILE) {
        const int jn = min(DF2_ATT_TILE, k1 - j0);

        // Check tile validity first so empty context tiles skip DRAM K/V reads completely.
        if (warp == 0) {
            int32_t cp = -1;
            bool ok = false;
            if (lane < jn) {
                const int j = j0 + lane;
                if (ci < 8) {
                    const int32_t raw_cp = ctx_pos[j];
                    if (raw_cp >= 0 && raw_cp < P && raw_cp >= P - (DF2_WINDOW - 1)) {
                        cp = raw_cp;
                        ok = true;
                    }
                } else {
                    cp = P + (j - DF2_WINDOW);
                    ok = true;
                }
                s_pos[lane] = cp;
            }
            const uint64_t mask = __ballot(ok);
            if (lane == 0) s_any = (mask != 0ull) ? 1 : 0;
        }
        __syncthreads();

        if (!s_any) {
            __syncthreads();
            continue;
        }

        // Cooperative load of up to 16 keys x 128 dims = 512 uint2 (exact 1/thread).
        if (tid < jn * 32) {
            const int jj = tid >> 5;
            const int c4 = (tid & 31) << 2;
            const int j  = j0 + jj;
            const size_t at = ((size_t)j * DF2_KVH + g) * DF2_HD + c4;
            const uint2 kw = *reinterpret_cast<const uint2 *>(k_ctx + at);
            const uint2 vw = *reinterpret_cast<const uint2 *>(v_ctx + at);
            *reinterpret_cast<float4 *>(&s_k[jj][c4]) = make_float4(
                df2_bf2f((uint16_t)(kw.x & 0xffffu)),
                df2_bf2f((uint16_t)(kw.x >> 16)),
                df2_bf2f((uint16_t)(kw.y & 0xffffu)),
                df2_bf2f((uint16_t)(kw.y >> 16)));
            *reinterpret_cast<float4 *>(&s_v[jj][c4]) = make_float4(
                df2_bf2f((uint16_t)(vw.x & 0xffffu)),
                df2_bf2f((uint16_t)(vw.x >> 16)),
                df2_bf2f((uint16_t)(vw.y & 0xffffu)),
                df2_bf2f((uint16_t)(vw.y >> 16)));
        }
        __syncthreads();

        const int min_pos = (P + row) - (DF2_WINDOW - 1);
        for (int jj = 0; jj < jn; ++jj) {
            const int32_t cp = s_pos[jj];
            if (cp < 0 || cp < min_pos) continue;
            const float * kx = s_k[jj] + lane * 4;
            const float * vx = s_v[jj] + lane * 4;
            float s = 0.0f;
            #pragma unroll
            for (int e = 0; e < 4; ++e) s = fmaf(qv[e], kx[e], s);
            s = df2_warp_sum(s) * DF2_KQ_SCALE;
            const float m_new = fmaxf(m_run, s);
            const float al = (m_run == -INFINITY) ? 0.0f : __expf(m_run - m_new);
            const float p  = __expf(s - m_new);
            l_run = l_run * al + p;
            m_run = m_new;
            #pragma unroll
            for (int e = 0; e < 4; ++e) o[e] = fmaf(p, vx[e], o[e] * al);
        }
        __syncthreads();
    }

    const size_t base = ((size_t)ci * DF2_KVH + g) * DF2_ATT_NQ + (size_t)(hh * DF2_BLOCK + row);
    float * wo = ws + base * DF2_HD;
    *reinterpret_cast<float4 *>(wo + lane * 4) = make_float4(o[0], o[1], o[2], o[3]);
    float * wm = ws + (size_t)DF2_ATT_CHUNKS * DF2_KVH * DF2_ATT_NQ * DF2_HD + base * 2;
    if (lane == 0) {
        wm[0] = m_run;
        wm[1] = l_run;
    }
}

static __global__ __launch_bounds__(32) void df2_attn_reduce_kernel(
        const float * __restrict__ ws, uint16_t * __restrict__ out) {
    const int lane = threadIdx.x;
    const int qi   = blockIdx.x;            // 0..255 = h * 8 + row
    const int h    = qi >> 3;               // 0..31
    const int row  = qi & 7;                // 0..7
    const int g    = h / DF2_GQA;           // 0..7
    const int hh   = h - g * DF2_GQA;       // 0..3
    const int qq   = hh * DF2_BLOCK + row;  // 0..31
    const float * mlo = ws + (size_t)DF2_ATT_CHUNKS * DF2_KVH * DF2_ATT_NQ * DF2_HD;

    float m_all = -INFINITY;
    #pragma unroll
    for (int c = 0; c < DF2_ATT_CHUNKS; ++c) {
        const size_t base = ((size_t)c * DF2_KVH + g) * DF2_ATT_NQ + qq;
        m_all = fmaxf(m_all, mlo[base * 2 + 0]);
    }
    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float l_sum = 0.0f;
    #pragma unroll
    for (int c = 0; c < DF2_ATT_CHUNKS; ++c) {
        const size_t base = ((size_t)c * DF2_KVH + g) * DF2_ATT_NQ + qq;
        const float mc = mlo[base * 2 + 0];
        const float lc = mlo[base * 2 + 1];
        if (lc == 0.0f) continue;
        const float scale = __expf(mc - m_all);
        l_sum += lc * scale;
        const float4 ov = *reinterpret_cast<const float4 *>(ws + base * DF2_HD + lane * 4);
        o[0] = fmaf(scale, ov.x, o[0]);
        o[1] = fmaf(scale, ov.y, o[1]);
        o[2] = fmaf(scale, ov.z, o[2]);
        o[3] = fmaf(scale, ov.w, o[3]);
    }
    const float inv_l = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    uint16_t * dst = out + ((size_t)row * DF2_HEADS + h) * DF2_HD + lane * 4;
    const uint32_t p0 = (uint32_t)df2_f2bf(o[0] * inv_l) | ((uint32_t)df2_f2bf(o[1] * inv_l) << 16);
    const uint32_t p1 = (uint32_t)df2_f2bf(o[2] * inv_l) | ((uint32_t)df2_f2bf(o[3] * inv_l) << 16);
    *reinterpret_cast<uint2 *>(dst) = make_uint2(p0, p1);
}

void launch_dflash2_attention(const uint16_t * q, const uint16_t * k_ctx,
                              const uint16_t * v_ctx, const int32_t * ctx_pos,
                              int P, float * ws, uint16_t * out, hipStream_t stream,
                              const DFlash2DraftParams * dp) {
    df2_attn_partial_kernel<<<dim3(DF2_ATT_CHUNKS, DF2_KVH, 2), 512, 0, stream>>>(
        q, k_ctx, v_ctx, ctx_pos, P, ws, dp);
    df2_attn_reduce_kernel<<<DF2_HEADS * DF2_BLOCK, 32, 0, stream>>>(ws, out);
}

// ---------------------------------------------------------------------------
// Exact 2-stage Top-16 over [7, V] (V = VOCAB_UNPADDED = 200058)
// ---------------------------------------------------------------------------
static constexpr int DF2_TOPK_BLOCKS = 128;
static constexpr int DF2_TOPK_MAX_CHUNK = 1600; // ceil(200058 / 128) = 1563 <= 1600

int64_t dflash2_topk16_part_elems() {
    return (int64_t)DF2_GAMMA * DF2_TOPK_BLOCKS * DF2_SEL_TOPK;
}

static __global__ __launch_bounds__(256) void df2_topk16_stage1_kernel(
        const float * __restrict__ logits, int V,
        float * __restrict__ part_val, int32_t * __restrict__ part_idx) {
    __shared__ float    s_val[DF2_TOPK_MAX_CHUNK];
    __shared__ uint64_t s_wmax[8];

    const int blk = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int chunk = (V + DF2_TOPK_BLOCKS - 1) / DF2_TOPK_BLOCKS;
    const int start = blk * chunk;
    const int end   = min(start + chunk, V);
    const int len   = max(0, end - start);

    const float * lrow = logits + (size_t)row * V + start;
    for (int i = tid; i < len; i += 256) {
        s_val[i] = lrow[i];
    }
    __syncthreads();

    const size_t out_base = ((size_t)row * DF2_TOPK_BLOCKS + blk) * DF2_SEL_TOPK;
    for (int k = 0; k < DF2_SEL_TOPK; ++k) {
        uint64_t best = 0ull;
        for (int i = tid; i < len; i += 256) {
            const uint64_t cand = df2_pack_argmax(s_val[i], (uint32_t)(start + i));
            if (cand > best) best = cand;
        }
        best = df2_warp_max_u64(best);
        if (lane == 0) s_wmax[warp] = best;
        __syncthreads();

        if (warp == 0) {
            uint64_t wb = (lane < 8) ? s_wmax[lane] : 0ull;
            wb = df2_warp_max_u64(wb);
            if (lane == 0) {
                if (wb == 0ull) {
                    part_val[out_base + k] = -INFINITY;
                    part_idx[out_base + k] = 0;
                } else {
                    const uint32_t gidx = df2_unpack_argmax(wb);
                    const float val = df2_unkey_float((uint32_t)(wb >> 32));
                    part_val[out_base + k] = val;
                    part_idx[out_base + k] = (int32_t)gidx;
                    s_val[gidx - (uint32_t)start] = -INFINITY;
                }
            }
        }
        __syncthreads();
    }
}

static __global__ __launch_bounds__(256) void df2_topk16_stage2_kernel(
        const float * __restrict__ part_val, const int32_t * __restrict__ part_idx,
        float * __restrict__ cand_val, int32_t * __restrict__ cand_idx) {
    constexpr int N = DF2_TOPK_BLOCKS * DF2_SEL_TOPK; // 2048
    __shared__ float    s_val[N];
    __shared__ int32_t  s_idx[N];
    __shared__ uint64_t s_wmax[8];
    __shared__ int32_t  s_win_gid;

    const int row  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const size_t in_base = (size_t)row * N;
    for (int i = tid; i < N; i += 256) {
        s_val[i] = part_val[in_base + i];
        s_idx[i] = part_idx[in_base + i];
    }
    __syncthreads();

    const size_t out_base = (size_t)row * DF2_SEL_TOPK;
    for (int k = 0; k < DF2_SEL_TOPK; ++k) {
        uint64_t best = 0ull;
        for (int i = tid; i < N; i += 256) {
            const float v = s_val[i];
            if (v > -INFINITY) {
                const uint64_t cand = df2_pack_argmax(v, (uint32_t)s_idx[i]);
                if (cand > best) best = cand;
            }
        }
        best = df2_warp_max_u64(best);
        if (lane == 0) s_wmax[warp] = best;
        __syncthreads();

        if (warp == 0) {
            uint64_t wb = (lane < 8) ? s_wmax[lane] : 0ull;
            wb = df2_warp_max_u64(wb);
            if (lane == 0) {
                const int32_t gidx = (wb == 0ull) ? 0 : (int32_t)df2_unpack_argmax(wb);
                const float val = (wb == 0ull) ? -INFINITY : df2_unkey_float((uint32_t)(wb >> 32));
                cand_val[out_base + k] = val;
                cand_idx[out_base + k] = gidx;
                s_win_gid = gidx;
            }
        }
        __syncthreads();

        const int32_t win_gid = s_win_gid;
        for (int i = tid; i < N; i += 256) {
            if (s_idx[i] == win_gid) s_val[i] = -INFINITY;
        }
        __syncthreads();
    }
}

void launch_dflash2_topk16(const float * logits, int V,
                           float * part_val, int32_t * part_idx,
                           float * cand_val, int32_t * cand_idx,
                           hipStream_t stream) {
    df2_topk16_stage1_kernel<<<dim3(DF2_TOPK_BLOCKS, DF2_GAMMA), 256, 0, stream>>>(
        logits, V, part_val, part_idx);
    df2_topk16_stage2_kernel<<<DF2_GAMMA, 256, 0, stream>>>(
        part_val, part_idx, cand_val, cand_idx);
}

// ---------------------------------------------------------------------------
// Single-CTA greedy bilinear Candidate Path Selector
// (rank=256, top_k=16, softcap=30*tanh(x/30) applied after correction)
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(256) void df2_selector_walk_kernel(
        const int8_t  * __restrict__ pred_q, const __half * __restrict__ pred_d,
        const int8_t  * __restrict__ succ_q, const __half * __restrict__ succ_d,
        const float   * __restrict__ sel_h,
        const float   * __restrict__ cand_val,
        const int32_t * __restrict__ cand_idx,
        int anchor,
        int32_t * __restrict__ tokens_out,
        float   * __restrict__ conf_out,
        const DFlash2DraftParams * __restrict__ dp) {
    __shared__ float   s_q[DF2_SEL_RANK];
    __shared__ float   s_scores[DF2_SEL_TOPK];
    __shared__ int32_t s_prev;

    const int tid  = threadIdx.x; // 0..255
    const int warp = tid >> 5;    // 0..7
    const int lane = tid & 31;    // 0..31

    if (tid == 0) {
        s_prev = (dp != nullptr) ? dp->anchor : anchor;
    }
    __syncthreads();

    constexpr int NB = DF2_SEL_RANK / 32; // 8

    for (int e = 0; e < DF2_GAMMA; ++e) {
        const int32_t prev_tok = s_prev;
        const float p_sc = __half2float(pred_d[(size_t)prev_tok * NB + (tid >> 5)]);
        const float p_val = (float)pred_q[(size_t)prev_tok * DF2_SEL_RANK + tid] * p_sc;
        const float h_val = df2_bfr(sel_h[(size_t)e * DF2_SEL_RANK + tid]);
        s_q[tid] = p_val * h_val;
        __syncthreads();

        // 8 warps compute 2 candidates each (c = warp, warp + 8)
        #pragma unroll
        for (int pass = 0; pass < 2; ++pass) {
            const int c = warp + pass * 8;
            const int32_t tok = cand_idx[(size_t)e * DF2_SEL_TOPK + c];
            const int8_t * sq = succ_q + (size_t)tok * DF2_SEL_RANK;
            const __half * sd = succ_d + (size_t)tok * NB;
            float acc = 0.0f;
            #pragma unroll
            for (int b = 0; b < NB; ++b) {
                const float sc = __half2float(sd[b]);
                const float w  = (float)sq[b * 32 + lane] * sc;
                acc = fmaf(s_q[b * 32 + lane], w, acc);
            }
            acc = df2_warp_sum(acc);
            if (lane == 0) {
                const float raw = cand_val[(size_t)e * DF2_SEL_TOPK + c] + acc;
                s_scores[c] = DF2_SOFTCAP * tanhf(raw * DF2_INV_SCAP);
            }
        }
        __syncthreads();

        if (tid == 0) {
            int best_c = 0;
            float best_s = s_scores[0];
            for (int c = 1; c < DF2_SEL_TOPK; ++c) {
                if (s_scores[c] > best_s) {
                    best_s = s_scores[c];
                    best_c = c;
                }
            }
            float sum_exp = 0.0f;
            for (int c = 0; c < DF2_SEL_TOPK; ++c) {
                sum_exp += expf(s_scores[c] - best_s);
            }
            const int32_t picked = cand_idx[(size_t)e * DF2_SEL_TOPK + best_c];
            tokens_out[e] = picked;
            conf_out[e]   = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;
            s_prev        = picked;
        }
        __syncthreads();
    }
}

void launch_dflash2_selector_walk(const SoaQ8Matrix & pred_cb,
                                  const SoaQ8Matrix & succ_cb,
                                  const float * sel_h,
                                  const float * cand_val,
                                  const int32_t * cand_idx,
                                  int anchor,
                                  int32_t * tokens_out,
                                  float * conf_out,
                                  hipStream_t stream,
                                  const DFlash2DraftParams * dp) {
    df2_selector_walk_kernel<<<1, 256, 0, stream>>>(
        pred_cb.qs, pred_cb.d, succ_cb.qs, succ_cb.d,
        sel_h, cand_val, cand_idx, anchor, tokens_out, conf_out, dp);
}

} // namespace inkling
