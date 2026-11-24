// Step 8A: DSpark MTP drafter kernels (gfx1100, Wave32 HIP).
//
// Ported from the reference CUDA implementation (kernels/draft/{block,attention,
// markov,fold}.cu) with the same rounding points where they matter for draft
// quality (BF16 rounding of activations, NeoX RoPE theta 1e6, RMSNorm eps 1e-6,
// bidirectional 8-row block attention, softcap 30*tanh(x/30)).  The projections
// run on this engine's SoA Q8_0 layout instead of the reference's FP8 rows.

#include "dspark_mtp_kernels.h"

#include <cmath>
#include <algorithm>

namespace inkling {

static constexpr int DSG_H     = N_EMBD;      // 2304
static constexpr int DSG_HEADS = N_HEAD;      // 24
static constexpr int DSG_KVH   = N_HEAD_KV;   // 8
static constexpr int DSG_HD    = HEAD_DIM;    // 128
static constexpr int DSG_QDIM  = N_Q;         // 3072
static constexpr int DSG_KVDIM = N_KV;        // 1024
static constexpr int DSG_QKV   = DSG_QDIM + 2 * DSG_KVDIM; // 5120
static constexpr int DSG_WINDOW = 1024;   // local layers' context ring
static constexpr int DSG_BLOCK = 8;
static constexpr int DSG_GAMMA = 7;
static constexpr int DSG_RANK  = 256;
static constexpr int DSG_RNN_IN = 2 * DSG_RANK + 2 * DSG_H; // 5120
static constexpr int DSG_RNN_OUT = 3 * DSG_RANK;
static constexpr int DSG_CAND  = 1024;
static constexpr float DSG_ROPE_THETA = 1e6f;
static constexpr float DSG_EPS = 1e-6f;
static constexpr float DSG_KQ_SCALE = 0.08838834764831845f; // 1/sqrt(128)

// ---------------------------------------------------------------------------
// bf16 helpers (no OCML-backed header)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float dsg_bf2f(uint16_t h) {
    const uint32_t u = (uint32_t)h << 16;
    float f;
    __builtin_memcpy(&f, &u, 4);
    return f;
}
static __device__ __forceinline__ uint16_t dsg_f2bf(float x) {
    uint32_t u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}
static __device__ __forceinline__ float dsg_bfr(float x) { return dsg_bf2f(dsg_f2bf(x)); }

static __device__ __forceinline__ float dsg_warp_sum(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor(v, o, 32);
    return v;
}
static __device__ __forceinline__ float dsg_warp_max(float v) {
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor(v, o, 32));
    return v;
}
// (score, index) packed for a lowest-index-wins max (torch.argmax semantics).
static __device__ __forceinline__ uint32_t dsg_float_key(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}
static __device__ __forceinline__ uint64_t dsg_pack_argmax(float score, uint32_t index) {
    return ((uint64_t)dsg_float_key(score) << 32) | (uint64_t)(0xffffffffu - index);
}
static __device__ __forceinline__ uint32_t dsg_unpack_argmax(uint64_t p) {
    return 0xffffffffu - (uint32_t)p;
}

static __global__ void dsg_set_params_kernel(DSparkDraftParams * dst, DSparkDraftParams val) {
    *dst = val;
}

void launch_dspark_set_params(DSparkDraftParams * d_dp, DSparkDraftParams val,
                              hipStream_t stream) {
    dsg_set_params_kernel<<<1, 1, 0, stream>>>(d_dp, val);
}

// ---------------------------------------------------------------------------
// Load-time: BF16 -> SoA Q8_0, shortlist gather, FP64 fold
// ---------------------------------------------------------------------------
static __global__ void dsg_repack_q8_kernel(const uint16_t * __restrict__ src, int rows, int K,
                                            int8_t * __restrict__ qs, __half * __restrict__ d) {
    const int nb = K / 32;
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= rows * nb) return;
    const int r = i / nb, b = i % nb;
    const uint16_t * x = src + (size_t)r * K + (size_t)b * 32;
    float v[32];
    float amax = 0.0f;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        v[j] = dsg_bf2f(x[j]);
        amax = fmaxf(amax, fabsf(v[j]));
    }
    const float dq = amax / 127.0f;
    const __half dh = __float2half(dq);
    const float dhf = __half2float(dh);
    const float inv = (dhf > 0.0f) ? (1.0f / dhf) : 0.0f;
    d[(size_t)r * nb + b] = dh;
    int8_t * q = qs + (size_t)r * K + (size_t)b * 32;
    #pragma unroll
    for (int j = 0; j < 32; ++j) {
        const float qf = rintf(v[j] * inv);
        q[j] = (int8_t)(qf > 127.0f ? 127.0f : (qf < -127.0f ? -127.0f : qf));
    }
}

void launch_dspark_repack_q8(const uint16_t * bf16, int rows, int K,
                             int8_t * qs, __half * d, hipStream_t stream) {
    const int total = rows * (K / 32);
    dsg_repack_q8_kernel<<<(total + 255) / 256, 256, 0, stream>>>(bf16, rows, K, qs, d);
}

static __global__ void dsg_gather_q8_kernel(const int8_t * __restrict__ src_q,
                                            const __half * __restrict__ src_d,
                                            int K, const int32_t * __restrict__ ids, int n,
                                            int8_t * __restrict__ dst_q, __half * __restrict__ dst_d,
                                            int dst_row0, int32_t * __restrict__ map) {
    const int nb = K / 32;
    const int i  = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n * nb) return;
    const int r = i / nb, b = i % nb;
    const int id = ids[r];
    for (int j = 0; j < 32; ++j) {
        dst_q[(size_t)(dst_row0 + r) * K + (size_t)b * 32 + j] = src_q[(size_t)id * K + (size_t)b * 32 + j];
    }
    dst_d[(size_t)(dst_row0 + r) * nb + b] = src_d[(size_t)id * nb + b];
    if (b == 0) map[(size_t)dst_row0 + r] = id;
}

void launch_dspark_gather_q8(const SoaQ8Matrix & src, const int32_t * ids, int n,
                             int8_t * dst_qs, __half * dst_d, int dst_row0,
                             int32_t * map, hipStream_t stream) {
    if (n <= 0) return;
    const int total = n * (src.K / 32);
    dsg_gather_q8_kernel<<<(total + 255) / 256, 256, 0, stream>>>(
        src.qs, src.d, src.K, ids, n, dst_qs, dst_d, dst_row0, map);
}

// fold_w = bf16(U @ hw) in FP64 (U from the SoA Q8_0 lm_head rows).
static __global__ void dsg_fold_w_kernel(const int8_t * __restrict__ uq,
                                         const __half * __restrict__ ud,
                                         const uint16_t * __restrict__ hw, int rows,
                                         uint16_t * __restrict__ fold_w) {
    constexpr int FR = 8;   // rows per CTA
    constexpr int FK = 16;  // k tile
    constexpr int TH = 256;
    __shared__ double us[FR][FK];
    __shared__ double hs[FK][DSG_RANK];
    const int r0 = blockIdx.x * FR;
    const int t  = threadIdx.x;
    const int nb = DSG_H / 32;
    double acc[FR];
    #pragma unroll
    for (int r = 0; r < FR; ++r) acc[r] = 0.0;
    for (int k0 = 0; k0 < DSG_H; k0 += FK) {
        for (int idx = t; idx < FR * FK; idx += TH) {
            const int r = idx / FK, k = idx % FK;
            const int row = r0 + r;
            double v = 0.0;
            if (row < rows) {
                const float q = (float)uq[(size_t)row * DSG_H + k0 + k];
                const float d = __half2float(ud[(size_t)row * nb + (k0 + k) / 32]);
                v = (double)dsg_bfr(q * d);
            }
            us[r][k] = v;
        }
        #pragma unroll
        for (int k = 0; k < FK; ++k) hs[k][t] = (double)dsg_bf2f(hw[(size_t)(k0 + k) * DSG_RANK + t]);
        __syncthreads();
        #pragma unroll
        for (int k = 0; k < FK; ++k) {
            const double h = hs[k][t];
            #pragma unroll
            for (int r = 0; r < FR; ++r) acc[r] = fma(us[r][k], h, acc[r]);
        }
        __syncthreads();
    }
    #pragma unroll
    for (int r = 0; r < FR; ++r) {
        if (r0 + r < rows) fold_w[(size_t)(r0 + r) * DSG_RANK + t] = dsg_f2bf((float)acc[r]);
    }
}

// warp per row
static __global__ void dsg_fold_b_kernel(const int8_t * __restrict__ uq,
                                         const __half * __restrict__ ud,
                                         const uint16_t * __restrict__ hb, int rows,
                                         uint16_t * __restrict__ fold_b) {
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= rows) return;
    const int nb = DSG_H / 32;
    double acc = 0.0;
    for (int k = lane; k < DSG_H; k += 32) {
        const float q = (float)uq[(size_t)row * DSG_H + k];
        const float d = __half2float(ud[(size_t)row * nb + k / 32]);
        acc = fma((double)dsg_bfr(q * d), (double)dsg_bf2f(hb[k]), acc);
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) acc += __shfl_xor(acc, o, 32);
    if (lane == 0) fold_b[row] = dsg_f2bf((float)acc);
}

void launch_dspark_fold_head(const SoaQ8Matrix & lm, const uint16_t * hw, const uint16_t * hb,
                             int rows, uint16_t * fold_w, uint16_t * fold_b, hipStream_t stream) {
    if (rows <= 0) return;
    dsg_fold_w_kernel<<<(rows + 7) / 8, 256, 0, stream>>>(lm.qs, lm.d, hw, rows, fold_w);
    dsg_fold_b_kernel<<<(rows * 32 + 255) / 256, 256, 0, stream>>>(lm.qs, lm.d, hb, rows, fold_b);
}

// ---------------------------------------------------------------------------
// m-row SoA Q8 GEMV (weights streamed once for all m <= 8 rows)
// ---------------------------------------------------------------------------
// Step 8B: the x rows are staged in LDS as FP32 (bf16 -> float is exact) in
// K-tiles of DSG_GTILE dims, so the per-lane inner product keeps the ORIGINAL
// bit-exact fmaf order (p over the 32 block weights in order, then
// acc = fmaf(p, scale, acc)) while each 32-byte weight block is read from DRAM
// exactly once for all m rows.  x traffic is L2-resident (m * K * 4 B <= 74 KiB).
typedef int v4i_t __attribute__((ext_vector_type(4)));

// 4 signed int8 packed in w -> exact FP32 dot with x (4 floats).  int8 -> FP16 is
// exact (0x6400 | (q + 128) = 1152 + q) and fmaf(float(half), x, acc) compiles to
// v_fma_mix_f32, which widens the FP16 operand exactly inside the FMA:
// BIT-IDENTICAL to fmaf((float) q, x, acc) with 9 VALU ops instead of 12.
static __device__ __forceinline__ float dsg_q8_word_dot(uint32_t w, const float4 x, float acc) {
    const uint32_t u   = w ^ 0x80808080u;
    const __half2  b   = __builtin_bit_cast(__half2, 0x64806480u);
    const __half2  h01 = __hsub2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u)), b);
    const __half2  h23 = __hsub2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u)), b);
    acc = fmaf(__low2float(h01),  x.x, acc);
    acc = fmaf(__high2float(h01), x.y, acc);
    acc = fmaf(__low2float(h23),  x.z, acc);
    acc = fmaf(__high2float(h23), x.w, acc);
    return acc;
}
static constexpr int DSG_GTILE = 256;  // dims staged per LDS tile (8 blocks)
static constexpr int DSG_GT4   = DSG_GTILE / 4;      // float4 per row per tile
static constexpr int DSG_GTP   = (DSG_GT4 / 8) * 9;  // padded float4 per row (bank-conflict-free)

template <int MODE>  // 0 = static m/rows, 1 = dp->head_rows, 2 = m = dp->ncommit
static __global__ __launch_bounds__(256) void dsg_gemv_kernel(
        const int8_t * __restrict__ qs, const __half * __restrict__ dd,
        const void * __restrict__ x, int xstride, const bool x_bf16,
        int m, float * __restrict__ y, int ystride, int rows, int K,
        const DSparkDraftParams * __restrict__ dp) {
    constexpr int NW = 8, LPR = 8, RPW = 4;
    // 8 float4 per 32-dim weight block, padded to 9 -> the 8 lane groups of a
    // warp hit banks 0,4,8,...,28 (0 conflicts) instead of all hitting 0..3.
    __shared__ float4 s_x[8][DSG_GTP];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sub  = lane % LPR;          // weight block within the lane's stride
    const int rofs = warp * RPW + lane / LPR;
    const int nb = K / 32;
    const int step = gridDim.x * (NW * RPW);
    if (MODE == 1 && dp != nullptr) rows = dp->head_rows;
    if (MODE == 2) m = dp->ncommit;
    const uint16_t * xb16 = reinterpret_cast<const uint16_t *>(x);
    const float * xf = reinterpret_cast<const float *>(x);

    for (int row0 = blockIdx.x * (NW * RPW); row0 < rows; row0 += step) {
        const int row = row0 + rofs;
        const bool valid = row < rows;
        const int rc = valid ? row : rows - 1;
        float acc[8];
        #pragma unroll
        for (int r = 0; r < 8; ++r) acc[r] = 0.0f;
        const int8_t * wq = qs + (size_t)rc * K;
        const __half * wd = dd + (size_t)rc * nb;
        for (int kt = 0; kt < K; kt += DSG_GTILE) {
            __syncthreads();
            // stage the m x rows for this K tile as float4 (coalesced 16-byte reads)
            #pragma unroll 1
            for (int idx = (int)threadIdx.x; idx < m * DSG_GT4; idx += 256) {
                const int r = idx / DSG_GT4;
                const int i = idx - r * DSG_GT4;
                float4 v;
                if (x_bf16) {
                    const uint2 u = *reinterpret_cast<const uint2 *>(xb16 + (size_t)r * xstride + kt + i * 4);
                    v.x = dsg_bf2f((uint16_t)(u.x & 0xffffu));
                    v.y = dsg_bf2f((uint16_t)(u.x >> 16));
                    v.z = dsg_bf2f((uint16_t)(u.y & 0xffffu));
                    v.w = dsg_bf2f((uint16_t)(u.y >> 16));
                } else {
                    v = *reinterpret_cast<const float4 *>(xf + (size_t)r * xstride + kt + i * 4);
                }
                s_x[r][(i >> 3) * 9 + (i & 7)] = v;
            }
            __syncthreads();
            const int kb = (kt >> 5) + sub;               // this lane's block in the tile
            const v4i_t q0 = __builtin_nontemporal_load(reinterpret_cast<const v4i_t *>(wq + (size_t)kb * 32));
            const v4i_t q1 = __builtin_nontemporal_load(reinterpret_cast<const v4i_t *>(wq + (size_t)kb * 32 + 16));
            const float sc = __half2float(wd[kb]);
            const int32_t * w0 = reinterpret_cast<const int32_t *>(&q0);
            const int32_t * w1 = reinterpret_cast<const int32_t *>(&q1);
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (r >= m) break;
                const float4 * xb = s_x[r] + sub * 9;
                float p = 0.0f;
                #pragma unroll
                for (int j = 0; j < 4; ++j) p = dsg_q8_word_dot((uint32_t)w0[j], xb[j], p);
                #pragma unroll
                for (int j = 0; j < 4; ++j) p = dsg_q8_word_dot((uint32_t)w1[j], xb[4 + j], p);
                acc[r] = fmaf(p, sc, acc[r]);
            }
        }
        #pragma unroll
        for (int r = 0; r < 8; ++r) {
            acc[r] += __shfl_xor(acc[r], 4, 8);
            acc[r] += __shfl_xor(acc[r], 2, 8);
            acc[r] += __shfl_xor(acc[r], 1, 8);
        }
        if (sub == 0 && valid) {
            #pragma unroll
            for (int r = 0; r < 8; ++r) {
                if (r >= m) break;
                y[(size_t)r * ystride + row] = acc[r];
            }
        }
    }
}

void launch_dspark_gemv(const SoaQ8Matrix & W, const void * x_rows, int xstride,
                        bool x_bf16, int m, float * y, int ystride, hipStream_t stream,
                        const DSparkDraftParams * dp, int mode, int rows_static) {
    if (W.rows <= 0 || m <= 0) return;
    const int rows = (rows_static > 0) ? rows_static : W.rows;
    const int blocks = (rows + 31) / 32;
    const int grid = blocks < 96 * 16 ? blocks : 96 * 16;
    if (mode == 1) {
        dsg_gemv_kernel<1><<<grid, 256, 0, stream>>>(W.qs, W.d, x_rows, xstride, x_bf16,
                                                     m, y, ystride, rows, W.K, dp);
    } else if (mode == 2) {
        dsg_gemv_kernel<2><<<grid, 256, 0, stream>>>(W.qs, W.d, x_rows, xstride, x_bf16,
                                                     m, y, ystride, rows, W.K, dp);
    } else {
        dsg_gemv_kernel<0><<<grid, 256, 0, stream>>>(W.qs, W.d, x_rows, xstride, x_bf16,
                                                     m, y, ystride, rows, W.K, nullptr);
    }
}

// ---------------------------------------------------------------------------
// Row kernels: 1 CTA per row, 288 threads, thread owns 8 consecutive elements
// ---------------------------------------------------------------------------
static constexpr int DSG_ROW_THREADS = DSG_H / 8; // 288

struct DsgScratch {
    float s[33];
};

static __device__ __forceinline__ float dsg_block_sum(float v, DsgScratch & sc) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, nw = blockDim.x >> 5;
    v = dsg_warp_sum(v);
    __syncthreads();
    if (lane == 0) sc.s[warp] = v;
    __syncthreads();
    float t = 0.0f;
    for (int w = 0; w < nw; ++w) t += sc.s[w];
    return t;
}

static __device__ __forceinline__ void dsg_load8(const uint16_t * p, float (&v)[8]) {
    const uint4 u = *reinterpret_cast<const uint4 *>(p);
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        v[2 * i]     = dsg_bf2f((uint16_t)(w[i] & 0xffffu));
        v[2 * i + 1] = dsg_bf2f((uint16_t)(w[i] >> 16));
    }
}
static __device__ __forceinline__ void dsg_store8(uint16_t * p, const float (&v)[8]) {
    uint32_t w[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        w[i] = (uint32_t)dsg_f2bf(v[2 * i]) | ((uint32_t)dsg_f2bf(v[2 * i + 1]) << 16);
    }
    *reinterpret_cast<uint4 *>(p) = make_uint4(w[0], w[1], w[2], w[3]);
}
static __device__ __forceinline__ uint4 dsg_pack8(const float (&v)[8]) {
    uint32_t w[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) w[i] = (uint32_t)dsg_f2bf(v[2 * i]) | ((uint32_t)dsg_f2bf(v[2 * i + 1]) << 16);
    return make_uint4(w[0], w[1], w[2], w[3]);
}

static __device__ __forceinline__ float dsg_rms_scale(const float (&x)[8], DsgScratch & sc) {
    float ss = 0.0f;
    #pragma unroll
    for (int j = 0; j < 8; ++j) ss += x[j] * x[j];
    return rsqrtf(dsg_block_sum(ss, sc) / (float)DSG_H + DSG_EPS);
}

static __global__ __launch_bounds__(DSG_ROW_THREADS) void dsg_mix_rmsnorm_kernel(
        const uint16_t * __restrict__ feats, const uint16_t * __restrict__ alpha,
        const uint16_t * __restrict__ w, uint16_t * __restrict__ out,
        const int n_dyn, const DSparkDraftParams * __restrict__ dp, const int mode) {
    __shared__ DsgScratch sc;
    int r = blockIdx.x;    // destination row
    int rsrc = r;          // source row
    if (mode == 1) {
        if (r >= dp->ncommit) return;
    } else if (mode == 2) {
        // one output row (the drafter's f_rnn state) read from row ncommit-1
        if (blockIdx.x != 0) return;
        rsrc = dp->ncommit - 1;
        r = 0;
    }
    (void)n_dyn;
    const int c = 8 * threadIdx.x;
    float a[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) a[j] = dsg_bf2f(alpha[j]);
    const uint4 wv = *reinterpret_cast<const uint4 *>(w + c);
    float m[8];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float x[8];
        dsg_load8(feats + (size_t)rsrc * 4 * DSG_H + (size_t)j * DSG_H + c, x);
        #pragma unroll
        for (int i = 0; i < 8; ++i) m[i] = (j == 0) ? __fmul_rn(x[i], a[0]) : __fadd_rn(m[i], __fmul_rn(x[i], a[j]));
    }
    const float rs = dsg_rms_scale(m, sc);
    const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
    float y[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float w0 = dsg_bf2f((uint16_t)(ww[i] & 0xffffu));
        const float w1 = dsg_bf2f((uint16_t)(ww[i] >> 16));
        y[2 * i]     = __fmul_rn(__fmul_rn(m[2 * i], rs), w0);
        y[2 * i + 1] = __fmul_rn(__fmul_rn(m[2 * i + 1], rs), w1);
    }
    dsg_store8(out + (size_t)r * DSG_H + c, y);
}

void launch_dspark_mix_rmsnorm(const uint16_t * feats, int n, const uint16_t * alpha,
                               const uint16_t * w, uint16_t * out, hipStream_t stream,
                               const DSparkDraftParams * dp, int mode) {
    if (n <= 0) return;
    dsg_mix_rmsnorm_kernel<<<n, DSG_ROW_THREADS, 0, stream>>>(feats, alpha, w, out, n, dp, mode);
}

static __global__ __launch_bounds__(DSG_ROW_THREADS) void dsg_embed_block_kernel(
        const int8_t * __restrict__ emb_q, const __half * __restrict__ emb_d,
        const float * __restrict__ embed_norm, const uint16_t * __restrict__ mask_embed,
        const uint16_t * __restrict__ in_norm, int anchor,
        uint16_t * __restrict__ res, uint16_t * __restrict__ xn,
        uint64_t * __restrict__ picks,
        const DSparkDraftParams * __restrict__ dp) {
    __shared__ DsgScratch sc;
    if (dp != nullptr) anchor = dp->anchor;
    const int r = blockIdx.x, c = 8 * threadIdx.x;
    float x[8];
    if (r == 0) {
        const int nb = DSG_H / 32;
        const int b = c / 32;
        const float sc_q = __half2float(emb_d[(size_t)anchor * nb + b]);
        const int8_t * row = emb_q + (size_t)anchor * DSG_H + (size_t)b * 32;
        const float en[8] = {embed_norm[c + 0], embed_norm[c + 1], embed_norm[c + 2], embed_norm[c + 3],
                             embed_norm[c + 4], embed_norm[c + 5], embed_norm[c + 6], embed_norm[c + 7]};
        float v[8];
        #pragma unroll
        for (int j = 0; j < 8; ++j) v[j] = (float)row[c % 32 + j] * sc_q;
        const float rs = dsg_rms_scale(v, sc);
        #pragma unroll
        for (int j = 0; j < 8; ++j) x[j] = dsg_bfr(__fmul_rn(__fmul_rn(v[j], rs), en[j]));
    } else {
        dsg_load8(mask_embed + c, x);
    }
    dsg_store8(res + (size_t)r * DSG_H + c, x);
    // xn = bf16(rms(x) * in_norm)
    const float rs = dsg_rms_scale(x, sc);
    uint4 wv = *reinterpret_cast<const uint4 *>(in_norm + c);
    const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
    float y[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        y[2 * i]     = __fmul_rn(__fmul_rn(x[2 * i], rs), dsg_bf2f((uint16_t)(ww[i] & 0xffffu)));
        y[2 * i + 1] = __fmul_rn(__fmul_rn(x[2 * i + 1], rs), dsg_bf2f((uint16_t)(ww[i] >> 16)));
    }
    dsg_store8(xn + (size_t)r * DSG_H + c, y);
    if (r == 0 && threadIdx.x < DSG_GAMMA) picks[threadIdx.x] = 0ull;
}

void launch_dspark_embed_block(const SoaQ8Matrix & tok_embd, const float * embed_norm,
                               const uint16_t * mask_embed, const uint16_t * in_norm,
                               int anchor, uint16_t * res, uint16_t * xn,
                               uint64_t * picks, hipStream_t stream,
                               const DSparkDraftParams * dp) {
    dsg_embed_block_kernel<<<DSG_BLOCK, DSG_ROW_THREADS, 0, stream>>>(
        tok_embd.qs, tok_embd.d, embed_norm, mask_embed, in_norm, anchor, res, xn, picks, dp);
}

// One head row of 128 values, one per thread: bf16(rms(x) * w) then NeoX RoPE.
static __device__ __forceinline__ float dsg_norm_rope_head(float x, const uint16_t * w, int pos,
                                                           DsgScratch & sc, float * pair) {
    const int d = threadIdx.x;
    const float ss = dsg_block_sum(x * x, sc);
    const float rs = rsqrtf(ss / (float)DSG_HD + DSG_EPS);
    const float y = dsg_bfr(__fmul_rn(__fmul_rn(x, rs), dsg_bf2f(w[d])));
    pair[d] = y;
    __syncthreads();
    constexpr int half = DSG_HD / 2;
    const int i = d & (half - 1);
    const float inv = powf(DSG_ROPE_THETA, (-(float)i * 2.0f) / (float)DSG_HD);
    const float ang = __fmul_rn((float)pos, inv);
    const float co = cosf(ang), si = sinf(ang);
    const float x1 = pair[i], x2 = pair[i + half];
    const float o = (d < half) ? __fsub_rn(__fmul_rn(x1, co), __fmul_rn(x2, si))
                               : __fadd_rn(__fmul_rn(x2, co), __fmul_rn(x1, si));
    __syncthreads();
    return dsg_bfr(o);
}

// One Q8_0 element store: the 32 threads of a warp cover exactly one 32-dim
// block (blockDim.x == HEAD_DIM == 128), so the scale is a warp reduction.
static __device__ __forceinline__ void dsg_store_q8_scalar(float x, int8_t * qs, __half * d,
                                                           size_t at, size_t dat) {
    float amax = fabsf(x);
    amax = fmaxf(amax, __shfl_xor(amax, 16, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 8, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 4, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 2, 32));
    amax = fmaxf(amax, __shfl_xor(amax, 1, 32));
    const float sc = amax * (1.0f / 127.0f);
    const float inv = (sc > 0.0f) ? (1.0f / sc) : 0.0f;
    if ((threadIdx.x & 31) == 0) d[dat] = __float2half(sc);
    const float qf = rintf(x * inv);
    qs[at] = (int8_t)(qf > 127.0f ? 127.0f : (qf < -127.0f ? -127.0f : qf));
}

// Commit: kv f32 [n, 2048] rows -> cache (BF16 local / Q8_0 global), grid (n, 8)
static __global__ __launch_bounds__(DSG_HD) void dsg_ctx_kv_store_kernel(
        const float * __restrict__ kv, int start, int size, const uint16_t * __restrict__ k_norm,
        bool local, int8_t * __restrict__ k_qs, __half * __restrict__ k_d,
        int8_t * __restrict__ v_qs, __half * __restrict__ v_d, int32_t * __restrict__ ctx_pos,
        const DSparkDraftParams * __restrict__ dp) {
    __shared__ DsgScratch sc;
    __shared__ float pair[DSG_HD];
    const int r = blockIdx.x, g = blockIdx.y, d = threadIdx.x;
    if (dp != nullptr) {
        if (r >= dp->ncommit) return;
        start = dp->pos;
    }
    const int pos = start + r;
    const int slot = local ? (pos % DSG_WINDOW) : (pos % size);
    const float * row = kv + (size_t)r * 2 * DSG_KVDIM;
    const float k = dsg_norm_rope_head(row[g * DSG_HD + d], k_norm, pos, sc, pair);
    const float v = row[DSG_KVDIM + g * DSG_HD + d];
    const size_t at = ((size_t)slot * DSG_KVH + g) * DSG_HD + d;
    if (local) {
        reinterpret_cast<uint16_t *>(k_qs)[at] = dsg_f2bf(k);
        reinterpret_cast<uint16_t *>(v_qs)[at] = dsg_f2bf(v);
    } else {
        const size_t dat = ((size_t)slot * DSG_KVH + g) * (DSG_HD / 32) + (size_t)(d / 32);
        dsg_store_q8_scalar(k, k_qs, k_d, at, dat);
        dsg_store_q8_scalar(v, v_qs, v_d, at, dat);
    }
    if (ctx_pos != nullptr && g == 0 && d == 0) ctx_pos[slot] = pos;
}

void launch_dspark_ctx_kv_store(const float * kv, int n, int start, int size,
                                const uint16_t * k_norm, bool local,
                                int8_t * k_qs, __half * k_d, int8_t * v_qs, __half * v_d,
                                int32_t * ctx_pos, hipStream_t stream,
                                const DSparkDraftParams * dp) {
    if (n <= 0) return;
    dsg_ctx_kv_store_kernel<<<dim3(n, DSG_KVH), DSG_HD, 0, stream>>>(
        kv, start, size, k_norm, local, k_qs, k_d, v_qs, v_d, ctx_pos, dp);
}

// qkv_rope: grid (8 rows, 24 q + 8 k + 8 v)
static __global__ __launch_bounds__(DSG_HD) void dsg_qkv_rope_kernel(
        const float * __restrict__ qkv, const uint16_t * __restrict__ q_norm,
        const uint16_t * __restrict__ k_norm, int P, uint16_t * __restrict__ q_out,
        bool local, int size, int8_t * __restrict__ k_qs, __half * __restrict__ k_d,
        int8_t * __restrict__ v_qs, __half * __restrict__ v_d,
        const DSparkDraftParams * __restrict__ dp) {
    __shared__ DsgScratch sc;
    __shared__ float pair[DSG_HD];
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    if (dp != nullptr) P = dp->pos;
    const int pos = P + r;
    const int slot = local ? (DSG_WINDOW + r) : (pos % size);
    const float * row = qkv + (size_t)r * DSG_QKV;
    if (h < DSG_HEADS) {
        const float qv = dsg_norm_rope_head(row[h * DSG_HD + d], q_norm, pos, sc, pair);
        q_out[((size_t)r * DSG_HEADS + h) * DSG_HD + d] = dsg_f2bf(qv);
    } else if (h < DSG_HEADS + DSG_KVH) {
        const int g = h - DSG_HEADS;
        const float k = dsg_norm_rope_head(row[DSG_QDIM + g * DSG_HD + d], k_norm, pos, sc, pair);
        const size_t at = ((size_t)slot * DSG_KVH + g) * DSG_HD + d;
        if (local) reinterpret_cast<uint16_t *>(k_qs)[at] = dsg_f2bf(k);
        else dsg_store_q8_scalar(k, k_qs, k_d, at, ((size_t)slot * DSG_KVH + g) * (DSG_HD / 32) + (size_t)(d / 32));
    } else {
        const int g = h - DSG_HEADS - DSG_KVH;
        const float v = row[DSG_QDIM + DSG_KVDIM + g * DSG_HD + d];
        const size_t at = ((size_t)slot * DSG_KVH + g) * DSG_HD + d;
        if (local) reinterpret_cast<uint16_t *>(v_qs)[at] = dsg_f2bf(v);
        else dsg_store_q8_scalar(v, v_qs, v_d, at, ((size_t)slot * DSG_KVH + g) * (DSG_HD / 32) + (size_t)(d / 32));
    }
}

void launch_dspark_qkv_rope(const float * qkv, const uint16_t * q_norm,
                            const uint16_t * k_norm, int P, uint16_t * q_out,
                            bool local, int size,
                            int8_t * k_qs, __half * k_d, int8_t * v_qs, __half * v_d,
                            hipStream_t stream, const DSparkDraftParams * dp) {
    dsg_qkv_rope_kernel<<<dim3(DSG_BLOCK, DSG_HEADS + 2 * DSG_KVH), DSG_HD, 0, stream>>>(
        qkv, q_norm, k_norm, P, q_out, local, size, k_qs, k_d, v_qs, v_d, dp);
}

// ---------------------------------------------------------------------------
// Block attention: one warp per (row, query head) pair = 24 warps per kv head;
// keys split into chunks, each warp runs an online softmax over its chunk.
// ---------------------------------------------------------------------------
// Step 8B: 16-key LDS-staged K/V tiles.  The 24 (row, head) warps of a block
// share ONE coalesced int4 dequantization of each 16-key tile instead of each
// warp re-reading the same 256 B/key from DRAM, so the DRAM traffic per key
// drops from 24x to 1x.  The per-key math (dequantized FP32 K/V, 4-dim/lane
// fmaf + warp tree, online softmax) is byte-for-byte the old kernel's.
static constexpr int DSG_ATT_CHUNK = 256;    // keys per split (saturates 96 CUs at 2k..8k ctx)
static constexpr int DSG_ATT_MAX_CH = 64;    // chunks per (row, head)
static constexpr int DSG_ATT_TILE = 16;      // keys per LDS tile
static constexpr int DSG_NQ = DSG_BLOCK * (DSG_HEADS / DSG_KVH); // 24 per kv head

static __host__ __device__ __forceinline__ int dsg_att_chunks(int nkeys) {
    const int c = (nkeys + DSG_ATT_CHUNK - 1) / DSG_ATT_CHUNK;
    return c < 1 ? 1 : (c > DSG_ATT_MAX_CH ? DSG_ATT_MAX_CH : c);
}

int dspark_attention_chunks(int nkeys) { return dsg_att_chunks(nkeys); }
int dspark_attention_chunk_len(int nkeys) {
    const int c = dsg_att_chunks(nkeys);
    return (nkeys + c - 1) / c;
}

int64_t dspark_attention_ws_floats(int nkeys) {
    const int chunks = dsg_att_chunks(nkeys);
    // [chunk][kv_head][24 queries][128 + 2]
    return (int64_t)chunks * DSG_KVH * DSG_NQ * (DSG_HD + 2);
}

// ws layout: o[chunk][g][q][128], then ml[chunk][g][q][2]
static __global__ __launch_bounds__(768) void dsg_attn_partial_kernel(
        const uint16_t * __restrict__ q, int nkeys, int chunk, int n_chunks,
        bool local, int size, const int8_t * __restrict__ k_qs, const __half * __restrict__ k_d,
        const int8_t * __restrict__ v_qs, const __half * __restrict__ v_d,
        const int32_t * __restrict__ ctx_pos, int P, float * __restrict__ ws,
        const DSparkDraftParams * __restrict__ dp) {
    __shared__ float s_k[DSG_ATT_TILE][DSG_HD];
    __shared__ float s_v[DSG_ATT_TILE][DSG_HD];
    __shared__ int s_valid[DSG_ATT_TILE];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int ci = blockIdx.x, g = blockIdx.y;
    const int hh = warp / DSG_BLOCK;      // query head within the kv group (0..2)
    const int row = warp % DSG_BLOCK;
    const int h = g * (DSG_HEADS / DSG_KVH) + hh;
    const int k0 = ci * chunk;
    if (dp != nullptr) {
        P = dp->pos;
        nkeys = local ? (DSG_WINDOW + DSG_BLOCK) : dp->nkeys_global;
    }
    const int k1 = min(k0 + chunk, nkeys);

    // this lane's 4 query dims
    const uint16_t * qrow = q + ((size_t)row * DSG_HEADS + h) * DSG_HD;
    float qv[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) qv[j] = dsg_bf2f(qrow[lane * 4 + j]);

    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float m_run = -INFINITY, l_run = 0.0f;

    for (int j0 = k0; j0 < k1; j0 += DSG_ATT_TILE) {
        const int jn = min(DSG_ATT_TILE, k1 - j0);
        // Cooperative dequantization of the tile into LDS (4 dims per thread).
        #pragma unroll 1
        for (int idx = (int)threadIdx.x; idx < jn * (DSG_HD / 4); idx += 768) {
            const int jj = idx / (DSG_HD / 4);
            const int c4 = (idx - jj * (DSG_HD / 4)) * 4;
            const int j  = j0 + jj;
            const size_t at = ((size_t)j * DSG_KVH + g) * DSG_HD + c4;
            float kx[4], vx[4];
            if (local) {
                const uint2 kw = *reinterpret_cast<const uint2 *>(reinterpret_cast<const uint16_t *>(k_qs) + at);
                const uint2 vw = *reinterpret_cast<const uint2 *>(reinterpret_cast<const uint16_t *>(v_qs) + at);
                kx[0] = dsg_bf2f((uint16_t)(kw.x & 0xffffu));
                kx[1] = dsg_bf2f((uint16_t)(kw.x >> 16));
                kx[2] = dsg_bf2f((uint16_t)(kw.y & 0xffffu));
                kx[3] = dsg_bf2f((uint16_t)(kw.y >> 16));
                vx[0] = dsg_bf2f((uint16_t)(vw.x & 0xffffu));
                vx[1] = dsg_bf2f((uint16_t)(vw.x >> 16));
                vx[2] = dsg_bf2f((uint16_t)(vw.y & 0xffffu));
                vx[3] = dsg_bf2f((uint16_t)(vw.y >> 16));
            } else {
                const size_t gk = ((size_t)j * DSG_KVH + g) * (DSG_HD / 32) + (c4 >> 5);
                const float ks = __half2float(k_d[gk]);
                const float vs = __half2float(v_d[gk]);
                const uint32_t kw = *reinterpret_cast<const uint32_t *>(k_qs + at);
                const uint32_t vw = *reinterpret_cast<const uint32_t *>(v_qs + at);
                kx[0] = (float)(int8_t)(kw & 0xffu) * ks;
                kx[1] = (float)(int8_t)((kw >> 8) & 0xffu) * ks;
                kx[2] = (float)(int8_t)((kw >> 16) & 0xffu) * ks;
                kx[3] = (float)(int8_t)((kw >> 24) & 0xffu) * ks;
                vx[0] = (float)(int8_t)(vw & 0xffu) * vs;
                vx[1] = (float)(int8_t)((vw >> 8) & 0xffu) * vs;
                vx[2] = (float)(int8_t)((vw >> 16) & 0xffu) * vs;
                vx[3] = (float)(int8_t)((vw >> 24) & 0xffu) * vs;
            }
            *reinterpret_cast<float4 *>(&s_k[jj][c4]) = make_float4(kx[0], kx[1], kx[2], kx[3]);
            *reinterpret_cast<float4 *>(&s_v[jj][c4]) = make_float4(vx[0], vx[1], vx[2], vx[3]);
        }
        if ((int)threadIdx.x < jn) {
            const int j = j0 + (int)threadIdx.x;
            bool valid = true;
            if (local) {
                if (j < DSG_WINDOW) {
                    const int cp = ctx_pos[j];
                    valid = (cp >= 0) && (cp < P) && (cp >= P - DSG_WINDOW);
                }
            }
            s_valid[threadIdx.x] = valid ? 1 : 0;
        }
        __syncthreads();
        for (int jj = 0; jj < jn; ++jj) {
            if (!s_valid[jj]) continue;
            const float * kx = s_k[jj] + lane * 4;
            const float * vx = s_v[jj] + lane * 4;
            float s = 0.0f;
            #pragma unroll
            for (int e = 0; e < 4; ++e) s = fmaf(qv[e], kx[e], s);
            s = dsg_warp_sum(s) * DSG_KQ_SCALE;
            const float m_new = fmaxf(m_run, s);
            const float al = (m_run == -INFINITY) ? 0.0f : __expf(m_run - m_new);
            const float p = __expf(s - m_new);
            l_run = l_run * al + p;
            m_run = m_new;
            #pragma unroll
            for (int e = 0; e < 4; ++e) o[e] = fmaf(p, vx[e], o[e] * al);
        }
        __syncthreads();
    }

    const size_t base = ((size_t)ci * DSG_KVH + g) * DSG_NQ + (size_t)(hh * DSG_BLOCK + row);
    float * wo = ws + base * DSG_HD;
    #pragma unroll
    for (int e = 0; e < 4; ++e) wo[lane * 4 + e] = o[e];
    float * wm = ws + (size_t)n_chunks * DSG_KVH * DSG_NQ * DSG_HD + base * 2;
    if (lane == 0) { wm[0] = m_run; wm[1] = l_run; }
}

// combine: grid = 24 * 8 = 192, block 32
static __global__ __launch_bounds__(32) void dsg_attn_combine_kernel(
        const float * __restrict__ ws, int n_chunks, int chunk, int nkeys,
        uint16_t * __restrict__ out) {
    const int lane = threadIdx.x;
    const int qi = blockIdx.x;                 // 0..191 = head * 8 + row
    const int h = qi / DSG_BLOCK, row = qi % DSG_BLOCK;
    const int g = h / (DSG_HEADS / DSG_KVH);
    // partials are [chunk][g][q][dim] with q = hh*8 + row
    const int qq = (h % (DSG_HEADS / DSG_KVH)) * DSG_BLOCK + row;
    const float * mlo = ws + (size_t)n_chunks * DSG_KVH * DSG_NQ * DSG_HD;

    float m_all = -INFINITY;
    for (int c = 0; c < n_chunks; ++c) {
        const int k0 = c * chunk;
        if (k0 >= nkeys) continue;
        const size_t base = ((size_t)c * DSG_KVH + g) * DSG_NQ + qq;
        m_all = fmaxf(m_all, mlo[base * 2 + 0]);
    }
    float o[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float l_sum = 0.0f;
    for (int c = 0; c < n_chunks; ++c) {
        const int k0 = c * chunk;
        if (k0 >= nkeys) continue;
        const size_t base = ((size_t)c * DSG_KVH + g) * DSG_NQ + qq;
        const float mc = mlo[base * 2 + 0];
        const float lc = mlo[base * 2 + 1];
        if (!(mc > -INFINITY)) continue;
        const float f = __expf(mc - m_all);
        l_sum += lc * f;
        const float * oc = ws + base * DSG_HD;
        #pragma unroll
        for (int e = 0; e < 4; ++e) o[e] = fmaf(oc[lane * 4 + e], f, o[e]);
    }
    const float inv = (l_sum > 0.0f) ? (1.0f / l_sum) : 0.0f;
    uint16_t * dst = out + ((size_t)row * DSG_HEADS + h) * DSG_HD + lane * 4;
    #pragma unroll
    for (int e = 0; e < 4; ++e) dst[e] = dsg_f2bf(o[e] * inv);
}

void launch_dspark_attention(const uint16_t * q, int nkeys, bool local, int size,
                             const int8_t * k_qs, const __half * k_d,
                             const int8_t * v_qs, const __half * v_d,
                             const int32_t * ctx_pos, int P, float * ws,
                             uint16_t * out, hipStream_t stream,
                             const DSparkDraftParams * dp,
                             int n_chunks_static, int chunk_static) {
    (void)size;
    int n_chunks, chunk;
    if (dp != nullptr) {
        // graph-static split: the captured grid/chunk cover the bucket's maximum
        // nkeys; the kernel reads the actual nkeys from dp and clamps k1.
        n_chunks = n_chunks_static;
        chunk = chunk_static;
    } else {
        n_chunks = dsg_att_chunks(nkeys);
        chunk = (nkeys + n_chunks - 1) / n_chunks;
    }
    dsg_attn_partial_kernel<<<dim3(n_chunks, DSG_KVH), 768, 0, stream>>>(
        q, nkeys, chunk, n_chunks, local, size, k_qs, k_d, v_qs, v_d, ctx_pos, P, ws, dp);
    dsg_attn_combine_kernel<<<DSG_HEADS * DSG_BLOCK, 32, 0, stream>>>(ws, n_chunks, chunk, nkeys, out);
}

// ---------------------------------------------------------------------------
// residual / silu
// ---------------------------------------------------------------------------
static __global__ __launch_bounds__(DSG_ROW_THREADS) void dsg_add_rmsnorm_kernel(
        uint16_t * __restrict__ res, const float * __restrict__ delta,
        const uint16_t * __restrict__ w, uint16_t * __restrict__ xn) {
    __shared__ DsgScratch sc;
    const size_t at = (size_t)blockIdx.x * DSG_H + 8 * threadIdx.x;
    float x[8];
    dsg_load8(res + at, x);
    const float4 d0 = *reinterpret_cast<const float4 *>(delta + at);
    const float4 d1 = *reinterpret_cast<const float4 *>(delta + at + 4);
    const float d[8] = {d0.x, d0.y, d0.z, d0.w, d1.x, d1.y, d1.z, d1.w};
    #pragma unroll
    for (int j = 0; j < 8; ++j) x[j] = dsg_bfr(__fadd_rn(x[j], d[j]));
    dsg_store8(res + at, x);
    const float rs = dsg_rms_scale(x, sc);
    uint4 wv = *reinterpret_cast<const uint4 *>(w + 8 * threadIdx.x);
    const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
    float y[8];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        y[2 * i]     = __fmul_rn(__fmul_rn(x[2 * i], rs), dsg_bf2f((uint16_t)(ww[i] & 0xffffu)));
        y[2 * i + 1] = __fmul_rn(__fmul_rn(x[2 * i + 1], rs), dsg_bf2f((uint16_t)(ww[i] >> 16)));
    }
    dsg_store8(xn + at, y);
}

void launch_dspark_add_rmsnorm(uint16_t * res, const float * delta, const uint16_t * w,
                               uint16_t * xn, int rows, hipStream_t stream) {
    if (rows <= 0) return;
    dsg_add_rmsnorm_kernel<<<rows, DSG_ROW_THREADS, 0, stream>>>(res, delta, w, xn);
}

static __global__ void dsg_silu_mul_kernel(const float * __restrict__ gu, int rows, int inter,
                                           uint16_t * __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * inter) return;
    const size_t r = i / inter, j = i % inter;
    const float g = gu[r * 2 * inter + j], u = gu[r * 2 * inter + inter + j];
    const float s = __fdiv_rn(g, __fadd_rn(1.0f, __expf(-g)));
    out[i] = dsg_f2bf(__fmul_rn(s, u));
}

void launch_dspark_silu_mul(const float * gu, int rows, int inter, uint16_t * out,
                            hipStream_t stream) {
    const size_t n = (size_t)rows * inter;
    if (n == 0) return;
    dsg_silu_mul_kernel<<<(unsigned)((n + 255) / 256), 256, 0, stream>>>(gu, rows, inter, out);
}

static __global__ void dsg_fill_kernel(float * __restrict__ x, int64_t n, float v) {
    for (int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (int64_t)gridDim.x * blockDim.x) {
        x[i] = v;
    }
}

void launch_dspark_fill(float * x, int64_t n, float v, hipStream_t stream) {
    if (n <= 0) return;
    const int blocks = (int)std::min<int64_t>((n + 255) / 256, 4096);
    dsg_fill_kernel<<<blocks, 256, 0, stream>>>(x, n, v);
}

// ---------------------------------------------------------------------------
// Markov head
// ---------------------------------------------------------------------------
// One CTA per row: 32 warps, warp w owns slice w of V.  Two 8-bit radix passes
// find the 1024th-largest 16-bit key (float order), then each warp compacts its
// elements above the threshold (then its first ties) with a deterministic order.
static __global__ __launch_bounds__(1024) void dsg_topk_kernel(
        const float * __restrict__ base, int V, const int32_t * __restrict__ map,
        float * __restrict__ cand_val, int32_t * __restrict__ cand_idx,
        const DSparkDraftParams * __restrict__ dp, int stride) {
    __shared__ uint32_t hist_hi[256];
    __shared__ uint32_t hist_lo[256];
    __shared__ int s_bin, s_above, s_T, s_ngt;
    __shared__ int s_cnt_gt[32], s_cnt_tie[32];
    __shared__ int s_off_gt[32], s_off_tie[32];
    if (dp != nullptr) V = dp->head_rows;
    const int row = blockIdx.x;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const float * x = base + (size_t)row * stride;
    const int per = (V + 31) / 32;
    const int b0 = min(V, warp * per), b1 = min(V, b0 + per);
    auto key16 = [](float v) { return dsg_float_key(v) >> 16; };

    if (threadIdx.x < 256) { hist_hi[threadIdx.x] = 0u; hist_lo[threadIdx.x] = 0u; }
    __syncthreads();
    for (int i = b0 + lane; i < b1; i += 32) atomicAdd(&hist_hi[key16(x[i]) >> 8], 1u);
    __syncthreads();
    // find the high byte holding the 1024th key (bins scanned from the top)
    if (threadIdx.x == 0) {
        int above = 0;
        int b = 255;
        for (; b >= 0; --b) {
            const int c = (int)hist_hi[b];
            if (above + c >= DSG_CAND) break;
            above += c;
        }
        s_bin = b;
        s_above = above;
    }
    __syncthreads();
    const int bh = s_bin;
    for (int i = b0 + lane; i < b1; i += 32) {
        const uint32_t k = key16(x[i]);
        if ((int)(k >> 8) == bh) atomicAdd(&hist_lo[k & 255], 1u);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int above = s_above;
        int b = 255;
        for (; b >= 0; --b) {
            const int c = (int)hist_lo[b];
            if (above + c >= DSG_CAND) break;
            above += c;
        }
        s_T = (bh << 8) | b;
        s_ngt = above;
    }
    __syncthreads();
    const int T = s_T;

    // count this warp's gt / ties across ALL 32 lanes
    int ngt = 0, ntie = 0;
    for (int i = b0 + lane; i < b1; i += 32) {
        const uint32_t k = key16(x[i]);
        if ((int)k > T) ++ngt;
        else if ((int)k == T) ++ntie;
    }
    #pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        ngt  += __shfl_xor(ngt,  o, 32);
        ntie += __shfl_xor(ntie, o, 32);
    }
    if (lane == 0) { s_cnt_gt[warp] = ngt; s_cnt_tie[warp] = ntie; }
    __syncthreads();
    if (threadIdx.x == 0) {
        int ag = 0, at = 0;
        for (int w = 0; w < 32; ++w) {
            s_off_gt[w]  = ag;
            s_off_tie[w] = at;
            ag += s_cnt_gt[w];
            at += s_cnt_tie[w];
        }
        s_ngt = ag;
    }
    __syncthreads();
    const int off_gt  = s_off_gt[warp];
    const int off_tie = s_off_tie[warp];
    // compact: gt in index order, then ties in index order (ties cap at DSG_CAND)
    int a = 0, t = 0;
    for (int base_i = b0; base_i < b1; base_i += 32) {
        const int i = base_i + lane;
        const bool ok = i < b1;
        const uint32_t k = ok ? key16(x[i]) : 0u;
        const bool gt = ok && ((int)k > T);
        const bool tie = ok && ((int)k == T);
        const uint32_t bg = (uint32_t)__ballot_sync(~0ull, gt);
        const uint32_t bt = (uint32_t)__ballot_sync(~0ull, tie);
        const uint32_t mask_lo = (lane == 0) ? 0u : ((1u << lane) - 1u);
        if (gt) {
            const int pos = off_gt + a + __popc(bg & mask_lo);
            if (pos < DSG_CAND) {
                cand_val[(size_t)row * DSG_CAND + pos] = x[i];
                cand_idx[(size_t)row * DSG_CAND + pos] = map ? map[i] : i;
            }
        } else if (tie) {
            const int pos = s_ngt + off_tie + t + __popc(bt & mask_lo);
            if (pos < DSG_CAND) {
                cand_val[(size_t)row * DSG_CAND + pos] = x[i];
                cand_idx[(size_t)row * DSG_CAND + pos] = map ? map[i] : i;
            }
        }
        a += (int)__popc(bg);
        t += (int)__popc(bt);
    }
}

void launch_dspark_topk_candidates(const float * base, int rows, int V,
                                   const int32_t * map, float * cand_val,
                                   int32_t * cand_idx, float * ws, hipStream_t stream,
                                   const DSparkDraftParams * dp, int stride) {
    if (rows <= 0) return;
    (void)ws;
    dsg_topk_kernel<<<rows, 1024, 0, stream>>>(base, V, map, cand_val, cand_idx, dp,
                                               stride > 0 ? stride : V);
}

static __global__ __launch_bounds__(128) void dsg_markov_pre_kernel(
        const uint16_t * __restrict__ rnn_w, const uint16_t * __restrict__ h,
        const uint16_t * __restrict__ f_rnn, int gamma, float * __restrict__ part) {
    constexpr int PRE_COLS = 2 * DSG_H;         // 4608
    constexpr int PRE_CHUNKS = PRE_COLS / 8;    // 576 uint4 chunks
    constexpr int PER = PRE_CHUNKS / 32;        // 18
    const int row = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (row >= DSG_RNN_OUT) return;
    const uint16_t * wr = rnn_w + (size_t)row * DSG_RNN_IN + 2 * DSG_RANK;
    float acc[DSG_GAMMA];
    #pragma unroll
    for (int i = 0; i < DSG_GAMMA; ++i) acc[i] = 0.0f;
    for (int j = 0; j < PER; ++j) {
        const int col = 8 * (lane + 32 * j);
        float w[8];
        dsg_load8(wr + col, w);
        if (col < DSG_H) {
            #pragma unroll
            for (int i = 0; i < DSG_GAMMA; ++i) {
                if (i >= gamma) break;
                float z[8];
                dsg_load8(h + (size_t)i * DSG_H + col, z);
                #pragma unroll
                for (int e = 0; e < 8; ++e) acc[i] = fmaf(w[e], z[e], acc[i]);
            }
        } else {
            float z[8];
            dsg_load8(f_rnn + (col - DSG_H), z);
            float s = 0.0f;
            #pragma unroll
            for (int e = 0; e < 8; ++e) s = fmaf(w[e], z[e], s);
            #pragma unroll
            for (int i = 0; i < DSG_GAMMA; ++i) acc[i] += s;
        }
    }
    #pragma unroll
    for (int i = 0; i < DSG_GAMMA; ++i) {
        const float t = dsg_warp_sum(acc[i]);
        if (lane == 0 && i < gamma) part[i * DSG_RNN_OUT + row] = t;
    }
}

void launch_dspark_markov_pre(const uint16_t * rnn_w, const uint16_t * h,
                              const uint16_t * f_rnn, int gamma, float * part,
                              hipStream_t stream) {
    dsg_markov_pre_kernel<<<(DSG_RNN_OUT * 32 + 127) / 128, 128, 0, stream>>>(
        rnn_w, h, f_rnn, gamma, part);
}

// warp per state element j: rows j, 256 + j, 512 + j
static __global__ __launch_bounds__(256) void dsg_markov_step_kernel(
        int i, const uint16_t * __restrict__ rnn_w, const uint16_t * __restrict__ rnn_b,
        const uint16_t * __restrict__ w1, const float * __restrict__ part,
        const uint64_t * __restrict__ picks, const int32_t * __restrict__ cand_idx,
        int anchor, uint16_t * __restrict__ state, uint16_t * __restrict__ latent,
        const DSparkDraftParams * __restrict__ dp) {
    if (dp != nullptr) anchor = dp->anchor;
    const int j = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (j >= DSG_RANK) return;
    int prev = anchor;
    if (i > 0) prev = cand_idx[(size_t)(i - 1) * DSG_CAND + (int)dsg_unpack_argmax(picks[i - 1])];
    float z0[8], z1[8];
    dsg_load8(state + (size_t)i * DSG_RANK + 8 * lane, z0);
    dsg_load8(w1 + (size_t)prev * DSG_RANK + 8 * lane, z1);
    float raw[3];
    #pragma unroll
    for (int g = 0; g < 3; ++g) {
        const int row = g * DSG_RANK + j;
        const uint16_t * wr = rnn_w + (size_t)row * DSG_RNN_IN;
        float wa[8], wb[8];
        dsg_load8(wr + 8 * lane, wa);
        dsg_load8(wr + DSG_RANK + 8 * lane, wb);
        float acc = 0.0f;
        #pragma unroll
        for (int e = 0; e < 8; ++e) acc = fmaf(wa[e], z0[e], fmaf(wb[e], z1[e], acc));
        acc = dsg_warp_sum(acc);
        raw[g] = dsg_bfr(__fadd_rn(dsg_bfr(acc + part[i * DSG_RNN_OUT + row]), dsg_bf2f(rnn_b[row])));
    }
    if (lane == 0) {
        const float gate = dsg_bfr(__fdiv_rn(1.0f, __fadd_rn(1.0f, __expf(-raw[0]))));
        const float cell = dsg_bfr(tanhf(raw[1]));
        const float s = dsg_bf2f(state[(size_t)i * DSG_RANK + j]);
        const float gs = dsg_bfr(__fmul_rn(gate, s));
        const float om = dsg_bfr(__fsub_rn(1.0f, gate));
        const float oc = dsg_bfr(__fmul_rn(om, cell));
        state[(size_t)(i + 1) * DSG_RANK + j] = dsg_f2bf(__fadd_rn(gs, oc));
        latent[j] = dsg_f2bf(tanhf(raw[2]));
    }
}

void launch_dspark_markov_step(int i, const uint16_t * rnn_w, const uint16_t * rnn_b,
                               const uint16_t * w1, const float * part,
                               const uint64_t * picks, const int32_t * cand_idx,
                               int anchor, uint16_t * state, uint16_t * latent,
                               hipStream_t stream, const DSparkDraftParams * dp) {
    dsg_markov_step_kernel<<<(DSG_RANK * 32 + 255) / 256, 256, 0, stream>>>(
        i, rnn_w, rnn_b, w1, part, picks, cand_idx, anchor, state, latent, dp);
}

// warp per candidate: bias = fold_b + fold_w . latent; logit = bf16(30 tanh(x/30))
static __global__ __launch_bounds__(256) void dsg_markov_logits_kernel(
        int i, const uint16_t * __restrict__ fold_w, const uint16_t * __restrict__ fold_b,
        const float * __restrict__ cand_val, const int32_t * __restrict__ cand_idx,
        const uint16_t * __restrict__ latent, float * __restrict__ cand_logits,
        uint64_t * __restrict__ picks) {
    const int k = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (k >= DSG_CAND) return;
    const size_t at = (size_t)i * DSG_CAND + k;
    const int c = cand_idx[at];
    float w[8], l[8];
    dsg_load8(fold_w + (size_t)c * DSG_RANK + 8 * lane, w);
    dsg_load8(latent + 8 * lane, l);
    float acc = 0.0f;
    #pragma unroll
    for (int e = 0; e < 8; ++e) acc = fmaf(w[e], l[e], acc);
    acc = dsg_warp_sum(acc);
    if (lane == 0) {
        const float bias = dsg_bfr(__fadd_rn(dsg_bf2f(fold_b[c]), dsg_bfr(acc)));
        const float t = dsg_bfr(__fadd_rn(cand_val[at], bias));
        const float logit = dsg_bfr(__fmul_rn(30.0f, tanhf(__fdiv_rn(t, 30.0f))));
        cand_logits[at] = logit;
        atomicMax(&picks[i], dsg_pack_argmax(logit, (uint32_t)k));
    }
}

void launch_dspark_markov_logits(int i, const uint16_t * fold_w, const uint16_t * fold_b,
                                 const float * cand_val, const int32_t * cand_idx,
                                 const uint16_t * latent, int gamma,
                                 float * cand_logits, uint64_t * picks, hipStream_t stream) {
    (void)gamma;
    dsg_markov_logits_kernel<<<(DSG_CAND * 32 + 255) / 256, 256, 0, stream>>>(
        i, fold_w, fold_b, cand_val, cand_idx, latent, cand_logits, picks);
}

// CTA per draft row: conf = 1 / sum(exp(logit - max)), tokens = pick's candidate.
static __global__ __launch_bounds__(DSG_CAND) void dsg_markov_finish_kernel(
        const int32_t * __restrict__ cand_idx, const float * __restrict__ cand_logits,
        const uint64_t * __restrict__ picks, int32_t * __restrict__ tokens,
        float * __restrict__ conf) {
    __shared__ DsgScratch sc;
    const int i = blockIdx.x, t = threadIdx.x;
    const float l = cand_logits[(size_t)i * DSG_CAND + t];
    const float m = dsg_warp_max(l);
    const int lane = t & 31, warp = t >> 5;
    __syncthreads();
    if (lane == 0) sc.s[warp] = m;
    __syncthreads();
    float mm = -INFINITY;
    for (int w = 0; w < (int)(blockDim.x >> 5); ++w) mm = fmaxf(mm, sc.s[w]);
    const float s = dsg_block_sum(__expf(l - mm), sc);
    if (t == 0) {
        conf[i] = 1.0f / s;
        tokens[i] = cand_idx[(size_t)i * DSG_CAND + (int)dsg_unpack_argmax(picks[i])];
    }
}

void launch_dspark_markov_finish(const int32_t * cand_idx, const float * cand_logits,
                                 const uint64_t * picks, int32_t * tokens, float * conf,
                                 hipStream_t stream) {
    dsg_markov_finish_kernel<<<DSG_GAMMA, DSG_CAND, 0, stream>>>(cand_idx, cand_logits, picks, tokens, conf);
}

} // namespace inkling
