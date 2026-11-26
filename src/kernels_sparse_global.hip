#include "sparse_global.h"
#include "nib_mma.h"

#include <algorithm>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace inkling {

namespace {

constexpr float LOG2E = 1.44269504088896340736f;

typedef uint32_t v4u __attribute__((ext_vector_type(4)));

// ----------------------------------------------------------------------------
// Shared low-level helpers (identical semantics to kernels_fa_relbias.hip)
// ----------------------------------------------------------------------------

static __device__ __forceinline__ __half2 h2_of(uint32_t u) {
    return __builtin_bit_cast(__half2, u);
}

static __device__ __forceinline__ void i8x4_to_h2x2(uint32_t w, __half2 & h01, __half2 & h23) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    const uint32_t u = w ^ 0x80808080u;
    h01 = __hsub2(h2_of(__builtin_amdgcn_perm(0x64646464u, u, 0x04010400u)), h2_of(0x64806480u));
    h23 = __hsub2(h2_of(__builtin_amdgcn_perm(0x64646464u, u, 0x04030402u)), h2_of(0x64806480u));
#else
    const float f0 = (float)(int8_t)( w        & 0xFF);
    const float f1 = (float)(int8_t)((w >>  8) & 0xFF);
    const float f2 = (float)(int8_t)((w >> 16) & 0xFF);
    const float f3 = (float)(int8_t)((w >> 24) & 0xFF);
    h01 = __floats2half2_rn(f0, f1);
    h23 = __floats2half2_rn(f2, f3);
#endif
}

template <bool SIGNED_Q4_0>
static __device__ __forceinline__ void u4x4_to_h2x2(uint32_t nibs, __half2 & h01, __half2 & h23) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    constexpr uint32_t BIAS_H2 = SIGNED_Q4_0 ? 0x64086408u : 0x64006400u;
    h01 = __hsub2(h2_of(__builtin_amdgcn_perm(0x64646464u, nibs, 0x04010400u)), h2_of(BIAS_H2));
    h23 = __hsub2(h2_of(__builtin_amdgcn_perm(0x64646464u, nibs, 0x04030402u)), h2_of(BIAS_H2));
#else
    const float bias = SIGNED_Q4_0 ? 8.0f : 0.0f;
    const float f0 = (float)( nibs        & 0xF) - bias;
    const float f1 = (float)((nibs >>  8) & 0xF) - bias;
    const float f2 = (float)((nibs >> 16) & 0xF) - bias;
    const float f3 = (float)((nibs >> 24) & 0xF) - bias;
    h01 = __floats2half2_rn(f0, f1);
    h23 = __floats2half2_rn(f2, f3);
#endif
}

static __device__ __forceinline__ void wave32_fa_fwht128_4(
        float & v0, float & v1, float & v2, float & v3, const int lane) {
    const float a0 = v0 + v1, a1 = v0 - v1;
    const float a2 = v2 + v3, a3 = v2 - v3;
    float r[4] = { a0 + a2, a1 + a3, a0 - a2, a1 - a3 };

    #pragma unroll
    for (int m = 1; m <= 16; m <<= 1) {
        const bool hi = (lane & m) != 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float cur   = r[j];
            const float other = __shfl_xor(cur, m, 32);
            r[j] = hi ? (other - cur) : (cur + other);
        }
    }
    v0 = r[0]; v1 = r[1]; v2 = r[2]; v3 = r[3];
}

static __device__ __forceinline__ float fast_exp2f(float x) {
    return __builtin_amdgcn_exp2f(x);
}

static __device__ __forceinline__ float warp_max32(float x) {
    x = fmaxf(x, __shfl_xor(x, 16, 32));
    x = fmaxf(x, __shfl_xor(x,  8, 32));
    x = fmaxf(x, __shfl_xor(x,  4, 32));
    x = fmaxf(x, __shfl_xor(x,  2, 32));
    x = fmaxf(x, __shfl_xor(x,  1, 32));
    return x;
}

// Monotone float -> uint32 key: ascending float order == ascending key order.
static __device__ __forceinline__ uint32_t sparse_score_key(float v) {
    const uint32_t b = __builtin_bit_cast(uint32_t, v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// RelBias for one (query token, head, distance): either the precomputed
// rel_table row [N_REL_EXTENT] or the on-the-fly
// dot(r_prep_row[0..15], attn_rel_bias[dist][0..15]) where r_prep_row is the
// packed [n_tokens][N_HEAD][N_REL] row of this exact (query, head).
template <bool USE_R_PREP>
static __device__ __forceinline__ float rel_logit(
        const float * __restrict__ rel_row,
        const float * __restrict__ r_prep_row,
        const float * __restrict__ attn_rel_bias,
        int dist) {
    if constexpr (USE_R_PREP) {
        const float * b = attn_rel_bias + (size_t)dist * N_REL;
        float s = 0.0f;
        #pragma unroll
        for (int k = 0; k < N_REL; ++k) {
            s = fmaf(r_prep_row[k], b[k], s);
        }
        return s;
    } else {
        return rel_row[dist];
    }
}

// ----------------------------------------------------------------------------
// Page min/max metadata: read 4 stored (dequantized) K dims owned by one lane
// ----------------------------------------------------------------------------

template <int KV_FMT>
static __device__ __forceinline__ void load_k_row4(
        const KVCacheSlot & kv_slot, size_t head_slot, int lane, float4 & v) {
    if constexpr (KV_FMT == KV_Q8_0) {
        const float d = __half2float(__ldg(&kv_slot.k_d[head_slot * 4 + (size_t)(lane >> 3)]));
        const uint32_t w = *(const uint32_t *)(kv_slot.k_qs + head_slot * HEAD_DIM + (size_t)lane * 4);
        v.x = (float)(int8_t)( w        & 0xFF) * d;
        v.y = (float)(int8_t)((w >>  8) & 0xFF) * d;
        v.z = (float)(int8_t)((w >> 16) & 0xFF) * d;
        v.w = (float)(int8_t)((w >> 24) & 0xFF) * d;
    } else {
        const uint32_t w = *((const uint32_t *)(kv_slot.k_qs + head_slot * (HEAD_DIM / 2))
                             + (size_t)(lane >> 1));
        const uint32_t nibs = ((lane & 1) ? (w >> 4) : w) & 0x0F0F0F0Fu;
        const float q0 = (float)( nibs        & 0xF);
        const float q1 = (float)((nibs >>  8) & 0xF);
        const float q2 = (float)((nibs >> 16) & 0xF);
        const float q3 = (float)((nibs >> 24) & 0xF);
        if constexpr (KV_FMT == KV_Q4_1) {
            const __half2 dm = __ldg(&((const __half2 *)kv_slot.k_d)[head_slot * 4 + (size_t)(lane >> 3)]);
            const float d = __low2float(dm);
            const float m = __high2float(dm);
            v.x = fmaf(q0, d, m);
            v.y = fmaf(q1, d, m);
            v.z = fmaf(q2, d, m);
            v.w = fmaf(q3, d, m);
        } else {
            const float d = __half2float(__ldg(&kv_slot.k_d[head_slot * 4 + (size_t)(lane >> 3)]));
            v.x = (q0 - 8.0f) * d;
            v.y = (q1 - 8.0f) * d;
            v.z = (q2 - 8.0f) * d;
            v.w = (q3 - 8.0f) * d;
        }
    }
}

// Step 8G: symmetric quantization of one page/head (lo, hi) = (k_mean - k_rad, k_mean + k_rad).
// Every lane owns dims [4*lane .. 4*lane+3] of the 128-dim head.
// Because 0.5*(hi + lo) = k_mean and 0.5*(hi - lo) = k_rad, every scoring kernel
// automatically computes <q, k_mean> + alpha * <|q|, k_rad> with zero extra bandwidth.
template <int META_FMT>
static __device__ __forceinline__ void quantize_page_minmax_store(
        const float lo[4], const float hi[4], int lane,
        int8_t * min_row, int8_t * max_row, uint8_t * qmm_row,
        __half * d_slot) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        amax = fmaxf(amax, fmaxf(fabsf(lo[i]), fabsf(hi[i])));
    }
    amax = warp_max32(amax);

    if constexpr (META_FMT == SPARSE_META_Q4) {
        const float scale = fmaxf(amax * (1.0f / 7.0f),
                                  5.9604644775390625e-8f);   // 2^-24
        const __half dh = __float2half_ru(scale);
        const float de = __half2float(dh);
        const float inv = 1.0f / de;
        uint32_t acc = 0u;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            int qn = (int)fminf(7.0f, fmaxf(-7.0f, rintf(lo[i] * inv)));
            int qx = (int)fminf(7.0f, fmaxf(-7.0f, rintf(hi[i] * inv)));
            const uint32_t b = (uint32_t)((qn + 8) | ((qx + 8) << 4));
            acc |= b << (8 * i);
        }
        ((uint32_t *)qmm_row)[lane] = acc;
        if (lane == 0) {
            *d_slot = dh;
        }
        return;
    } else {
        const float scale = fmaxf(amax * (1.0f / 127.0f),
                                  5.9604644775390625e-8f);   // 2^-24
        const __half dh = __float2half_ru(scale);
        const float de = __half2float(dh);
        const float inv = 1.0f / de;

        int qmn[4], qmx[4];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            qmn[i] = (int)fminf(127.0f, fmaxf(-127.0f, rintf(lo[i] * inv)));
            qmx[i] = (int)fminf(127.0f, fmaxf(-127.0f, rintf(hi[i] * inv)));
        }
        const int32_t wmin = (int32_t)((qmn[0] & 0xFF) | ((qmn[1] & 0xFF) << 8) |
                                       ((qmn[2] & 0xFF) << 16) | ((qmn[3] & 0xFF) << 24));
        const int32_t wmax = (int32_t)((qmx[0] & 0xFF) | ((qmx[1] & 0xFF) << 8) |
                                       ((qmx[2] & 0xFF) << 16) | ((qmx[3] & 0xFF) << 24));
        ((int32_t *)min_row)[lane] = wmin;
        ((int32_t *)max_row)[lane] = wmax;
        if (lane == 0) {
            *d_slot = dh;
        }
    }
}

template <int KV_FMT, int META_FMT>
__launch_bounds__(32, 8)
static __global__ void sparse_page_minmax_prefill_kernel(
        KVCacheSlot kv_slot,
        SparseGlobalPageMeta meta,
        int pos_start,
        int n_tokens) {
    const int page = (int)blockIdx.x + pos_start / SPARSE_PAGE_SIZE;
    const int h    = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;

    const int page_start = page * SPARSE_PAGE_SIZE;
    const int kv_end     = pos_start + n_tokens;
    const int tok_end    = min(page_start + SPARSE_PAGE_SIZE, kv_end);
    if (page_start >= tok_end || page >= meta.max_pages) {
        return;
    }

    const size_t head_base = (size_t)h * (size_t)kv_slot.cap;
    float mn[4]  = {  INFINITY,  INFINITY,  INFINITY,  INFINITY };
    float mx[4]  = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    float sum[4] = {  0.0f,      0.0f,      0.0f,      0.0f     };
    for (int t = page_start; t < tok_end; ++t) {
        float4 v;
        load_k_row4<KV_FMT>(kv_slot, head_base + (size_t)t, lane, v);
        wave32_fa_fwht128_4(v.x, v.y, v.z, v.w, lane);
        v.x *= KQ_SCALE; v.y *= KQ_SCALE; v.z *= KQ_SCALE; v.w *= KQ_SCALE;
        mn[0] = fminf(mn[0], v.x); mx[0] = fmaxf(mx[0], v.x); sum[0] += v.x;
        mn[1] = fminf(mn[1], v.y); mx[1] = fmaxf(mx[1], v.y); sum[1] += v.y;
        mn[2] = fminf(mn[2], v.z); mx[2] = fmaxf(mx[2], v.z); sum[2] += v.z;
        mn[3] = fminf(mn[3], v.w); mx[3] = fmaxf(mx[3], v.w); sum[3] += v.w;
    }

    const float inv_n = 1.0f / (float)(tok_end - page_start);
    float lo[4], hi[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float mean = sum[i] * inv_n;
        const float rad  = 0.5f * (mx[i] - mn[i]);
        lo[i] = mean - rad;
        hi[i] = mean + rad;
    }

    const size_t page_off = ((size_t)h * (size_t)meta.max_pages + (size_t)page) * HEAD_DIM;
    quantize_page_minmax_store<META_FMT>(lo, hi, lane,
        meta.k_min_q + page_off,
        meta.k_max_q + page_off,
        meta.k_qmm + page_off,
        meta.k_mm_d + (size_t)h * (size_t)meta.max_pages + (size_t)page);
}

template <int KV_FMT, int META_FMT>
__launch_bounds__(32, 8)
static __global__ void sparse_page_minmax_decode_kernel(
        KVCacheSlot kv_slot,
        SparseGlobalPageMeta meta,
        const StepMeta * __restrict__ d_meta) {
    const int h    = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;

    const int pos        = d_meta->pos;
    const int page       = pos / SPARSE_PAGE_SIZE;
    const int page_start = page * SPARSE_PAGE_SIZE;
    const int tok_end    = pos + 1;
    if (page >= meta.max_pages) {
        return;
    }

    const size_t head_base = (size_t)h * (size_t)kv_slot.cap;
    float mn[4]  = {  INFINITY,  INFINITY,  INFINITY,  INFINITY };
    float mx[4]  = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    float sum[4] = {  0.0f,      0.0f,      0.0f,      0.0f     };
    for (int t = page_start; t < tok_end; ++t) {
        float4 v;
        load_k_row4<KV_FMT>(kv_slot, head_base + (size_t)t, lane, v);
        wave32_fa_fwht128_4(v.x, v.y, v.z, v.w, lane);
        v.x *= KQ_SCALE; v.y *= KQ_SCALE; v.z *= KQ_SCALE; v.w *= KQ_SCALE;
        mn[0] = fminf(mn[0], v.x); mx[0] = fmaxf(mx[0], v.x); sum[0] += v.x;
        mn[1] = fminf(mn[1], v.y); mx[1] = fmaxf(mx[1], v.y); sum[1] += v.y;
        mn[2] = fminf(mn[2], v.z); mx[2] = fmaxf(mx[2], v.z); sum[2] += v.z;
        mn[3] = fminf(mn[3], v.w); mx[3] = fmaxf(mx[3], v.w); sum[3] += v.w;
    }

    const float inv_n = 1.0f / (float)(tok_end - page_start);
    float lo[4], hi[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float mean = sum[i] * inv_n;
        const float rad  = 0.5f * (mx[i] - mn[i]);
        lo[i] = mean - rad;
        hi[i] = mean + rad;
    }

    const size_t page_off = ((size_t)h * (size_t)meta.max_pages + (size_t)page) * HEAD_DIM;
    quantize_page_minmax_store<META_FMT>(lo, hi, lane,
        meta.k_min_q + page_off,
        meta.k_max_q + page_off,
        meta.k_qmm + page_off,
        meta.k_mm_d + (size_t)h * (size_t)meta.max_pages + (size_t)page);
}

// ----------------------------------------------------------------------------
// Stage 1: page scoring
// ----------------------------------------------------------------------------

// Decode one 4-dim group of the page metadata into the (lo, hi) interval.
// Q4_MM packs min|max of one dimension into one byte, so `umin` alone carries
// both ends and `umax` is unused.
template <int META_FMT>
static __device__ __forceinline__ void sparse_meta_4(
        uint32_t umin, uint32_t umax, float d, float lo[4], float hi[4]) {
    if constexpr (META_FMT == SPARSE_META_Q4) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t b = (umin >> (8 * i)) & 0xFFu;
            lo[i] = (float)((int)(b & 0x0Fu) - 8) * d;
            hi[i] = (float)((int)(b >> 4) - 8) * d;
        }
    } else {
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            lo[i] = (float)(int8_t)((umin >> (8 * i)) & 0xFF) * d;
            hi[i] = (float)(int8_t)((umax >> (8 * i)) & 0xFF) * d;
        }
    }
}

// Decode: one warp scores one page; each lane loads 4 int8 min + 4 int8 max
// with a single 32-bit coalesced load each, for all GQA_RATIO=3 query heads.
template <int META_FMT>
__launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(12, 16)))
static __global__ void sparse_score_pages_decode_kernel(
        const float * __restrict__ q,            // [N_HEAD, HEAD_DIM]
        SparseGlobalPageMeta meta,
        const StepMeta * __restrict__ d_meta,
        float * __restrict__ page_scores) {      // [N_HEAD_KV, max_pages]
    const int h    = (int)blockIdx.y;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;

    const int n_kv    = d_meta->kv_len_global;
    const int n_pages = (n_kv + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
    const int cand_begin = SPARSE_SINK_PAGES;
    const int cand_end   = n_pages - SPARSE_RECENT_PAGES;
    if (cand_end <= cand_begin) {
        return;
    }

    // Collapse the GQA_RATIO unit-normalized queries into two 128-dim weight vectors:
    //   q_hi = sum_g 0.5 * (q_g + alpha * |q_g|)  (weights k_max)
    //   q_lo = sum_g 0.5 * (q_g - alpha * |q_g|)  (weights k_min)
    float q_lo[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float q_hi[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    #pragma unroll
    for (int g = 0; g < GQA_RATIO; ++g) {
        float4 t = ((const float4 *)(q + (size_t)(h * GQA_RATIO + g) * HEAD_DIM))[lane];
        wave32_fa_fwht128_4(t.x, t.y, t.z, t.w, lane);
        float n2 = t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
        n2 += __shfl_xor(n2, 16, 32);
        n2 += __shfl_xor(n2,  8, 32);
        n2 += __shfl_xor(n2,  4, 32);
        n2 += __shfl_xor(n2,  2, 32);
        n2 += __shfl_xor(n2,  1, 32);
        const float inv_n = rsqrtf(n2 + 1e-12f);
        const float qv[4] = { t.x * inv_n, t.y * inv_n, t.z * inv_n, t.w * inv_n };
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float a = SPARSE_QUEST_ALPHA * fabsf(qv[i]);
            q_hi[i] += 0.5f * (qv[i] + a);
            q_lo[i] += 0.5f * (qv[i] - a);
        }
    }

    const int gw      = (int)blockIdx.x * 4 + warp;
    const int n_warps = (int)gridDim.x * 4;
    const size_t head_base = (size_t)h * (size_t)meta.max_pages;

    for (int p = cand_begin + gw; p < cand_end; p += n_warps) {
        const size_t off = (head_base + (size_t)p) * HEAD_DIM + (size_t)lane * 4;
        uint32_t umin, umax;
        if constexpr (META_FMT == SPARSE_META_Q4) {
            umin = __builtin_nontemporal_load((const uint32_t *)(meta.k_qmm + off));
            umax = 0u;
        } else {
            umin = __builtin_nontemporal_load((const uint32_t *)(meta.k_min_q + off));
            umax = __builtin_nontemporal_load((const uint32_t *)(meta.k_max_q + off));
        }
        const float d = __half2float(*(const __half *)(meta.k_mm_d + head_base + (size_t)p));

        float lo[4], hi[4];
        sparse_meta_4<META_FMT>(umin, umax, d, lo, hi);
        float s = 0.0f;
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            s = fmaf(q_lo[i], lo[i], s);
            s = fmaf(q_hi[i], hi[i], s);
        }
        s += __shfl_xor(s, 16, 32);
        s += __shfl_xor(s,  8, 32);
        s += __shfl_xor(s,  4, 32);
        s += __shfl_xor(s,  2, 32);
        s += __shfl_xor(s,  1, 32);
        if (lane == 0) {
            page_scores[head_base + (size_t)p] = s;
        }
    }
}

// Prefill: damped Quest bound over a whole 64-query block.
template <int META_FMT>
__launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(12, 16)))
static __global__ void sparse_score_pages_prefill_kernel(
        const float * __restrict__ q_min,        // [qb][N_HEAD_KV][GQA][D]
        const float * __restrict__ q_max,
        SparseGlobalPageMeta meta,
        int pos_start,
        int n_tokens,
        float * __restrict__ page_scores) {      // [qb][N_HEAD_KV][max_pages]
    const int h    = (int)blockIdx.y;
    const int qb   = (int)blockIdx.z;
    const int lane = (int)threadIdx.x & 31;
    const int warp = (int)threadIdx.x >> 5;

    const int q_pos_start = pos_start + qb * SPARSE_PAGE_SIZE;
    const int local_start = sparse_local_start_page(q_pos_start);
    const int n_pages     = (pos_start + n_tokens + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
    const int cand_begin  = SPARSE_SINK_PAGES;
    const int cand_end    = min(local_start, n_pages);
    if (cand_end <= cand_begin) {
        return;
    }

    float qlo[GQA_RATIO][4], qhi[GQA_RATIO][4];
    #pragma unroll
    for (int g = 0; g < GQA_RATIO; ++g) {
        const size_t qoff = (((size_t)qb * N_HEAD_KV + (size_t)h) * GQA_RATIO + (size_t)g) * HEAD_DIM
                          + (size_t)lane * 4;
        const float4 lo = ((const float4 *)(q_min + qoff))[0];
        const float4 hi = ((const float4 *)(q_max + qoff))[0];
        qlo[g][0] = lo.x; qlo[g][1] = lo.y; qlo[g][2] = lo.z; qlo[g][3] = lo.w;
        qhi[g][0] = hi.x; qhi[g][1] = hi.y; qhi[g][2] = hi.z; qhi[g][3] = hi.w;
    }

    const int gw      = (int)blockIdx.x * 4 + warp;
    const int n_warps = (int)gridDim.x * 4;
    const size_t score_base = ((size_t)qb * N_HEAD_KV + (size_t)h) * (size_t)meta.max_pages;
    const size_t head_base  = (size_t)h * (size_t)meta.max_pages;

    for (int p = cand_begin + gw; p < cand_end; p += n_warps) {
        const size_t off = (head_base + (size_t)p) * HEAD_DIM + (size_t)lane * 4;
        uint32_t umin, umax;
        if constexpr (META_FMT == SPARSE_META_Q4) {
            umin = __builtin_nontemporal_load((const uint32_t *)(meta.k_qmm + off));
            umax = 0u;
        } else {
            umin = __builtin_nontemporal_load((const uint32_t *)(meta.k_min_q + off));
            umax = __builtin_nontemporal_load((const uint32_t *)(meta.k_max_q + off));
        }
        const float d = __half2float(*(const __half *)(meta.k_mm_d + head_base + (size_t)p));

        float lo[4], hi[4];
        sparse_meta_4<META_FMT>(umin, umax, d, lo, hi);
        float s0[GQA_RATIO] = { 0.0f, 0.0f, 0.0f };
        float s1[GQA_RATIO] = { 0.0f, 0.0f, 0.0f };
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float mid = 0.5f * (hi[i] + lo[i]);
            const float rad = (0.5f * SPARSE_QUEST_ALPHA) * (hi[i] - lo[i]);
            #pragma unroll
            for (int g = 0; g < GQA_RATIO; ++g) {
                const float q0 = qlo[g][i];
                const float q1 = qhi[g][i];
                s0[g] = fmaf(fabsf(q0), rad, fmaf(q0, mid, s0[g]));
                s1[g] = fmaf(fabsf(q1), rad, fmaf(q1, mid, s1[g]));
            }
        }
        #pragma unroll
        for (int g = 0; g < GQA_RATIO; ++g) {
            s0[g] += __shfl_xor(s0[g], 16, 32);
            s0[g] += __shfl_xor(s0[g],  8, 32);
            s0[g] += __shfl_xor(s0[g],  4, 32);
            s0[g] += __shfl_xor(s0[g],  2, 32);
            s0[g] += __shfl_xor(s0[g],  1, 32);
            s1[g] += __shfl_xor(s1[g], 16, 32);
            s1[g] += __shfl_xor(s1[g],  8, 32);
            s1[g] += __shfl_xor(s1[g],  4, 32);
            s1[g] += __shfl_xor(s1[g],  2, 32);
            s1[g] += __shfl_xor(s1[g],  1, 32);
        }
        if (lane == 0) {
            const float m0 = (s0[0] + s0[1] + s0[2]);
            const float m1 = (s1[0] + s1[1] + s1[2]);
            page_scores[score_base + (size_t)p] = fmaxf(m0, m1);
        }
    }
}

// Per-(q_block, kv_head) pair of 32-token half-block mean queries, unrotated
// back to the native NoPE basis via self-inverse FWHT-128.  Stored in the
// existing (q_min, q_max) scratch slots (half 0 -> q_min, half 1 -> q_max).
// Unlike a 64-token [q_min, q_max] bounding box (which spans negative..positive
// across 64 diverse queries and collapses scoring to page L1 norm), half-block
// mean queries preserve Q direction/sign while remaining sharp across topic
// boundaries inside a 64-token prefill tile.
__launch_bounds__(32, 8)
static __global__ void sparse_qblock_minmax_prefill_kernel(
        const float * __restrict__ q,            // [n_tokens, N_HEAD, HEAD_DIM]
        int n_tokens,
        float * __restrict__ q_min,
        float * __restrict__ q_max) {
    const int qb   = (int)blockIdx.x;
    const int h_kv = (int)blockIdx.y;
    const int lane = (int)threadIdx.x;           // 0..31 -> 4 dims/lane

    const int t0   = qb * SPARSE_PAGE_SIZE;
    const int t1   = min(t0 + SPARSE_PAGE_SIZE, n_tokens);
    if (t0 >= t1) {
        return;
    }
    const int tmid = min(t0 + (SPARSE_PAGE_SIZE / 2), t1);
    const float inv0 = 1.0f / (float)(tmid - t0);
    const bool  has1 = (t1 > tmid);
    const float inv1 = has1 ? (1.0f / (float)(t1 - tmid)) : inv0;

    #pragma unroll
    for (int g = 0; g < GQA_RATIO; ++g) {
        const int h = h_kv * GQA_RATIO + g;
        float s0[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float s1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        for (int t = t0; t < tmid; ++t) {
            const float4 v = ((const float4 *)(q + ((size_t)t * N_HEAD + (size_t)h) * HEAD_DIM))[lane];
            s0[0] += v.x; s0[1] += v.y; s0[2] += v.z; s0[3] += v.w;
        }
        wave32_fa_fwht128_4(s0[0], s0[1], s0[2], s0[3], lane);
        const float c0 = inv0 * KQ_SCALE;
        s0[0] *= c0; s0[1] *= c0; s0[2] *= c0; s0[3] *= c0;
        if (has1) {
            for (int t = tmid; t < t1; ++t) {
                const float4 v = ((const float4 *)(q + ((size_t)t * N_HEAD + (size_t)h) * HEAD_DIM))[lane];
                s1[0] += v.x; s1[1] += v.y; s1[2] += v.z; s1[3] += v.w;
            }
            wave32_fa_fwht128_4(s1[0], s1[1], s1[2], s1[3], lane);
            const float c1 = inv1 * KQ_SCALE;
            s1[0] *= c1; s1[1] *= c1; s1[2] *= c1; s1[3] *= c1;
        } else {
            s1[0] = s0[0]; s1[1] = s0[1]; s1[2] = s0[2]; s1[3] = s0[3];
        }
        const size_t out = (((size_t)qb * N_HEAD_KV + (size_t)h_kv) * GQA_RATIO + (size_t)g) * HEAD_DIM;
        ((float4 *)(q_min + out))[lane] = make_float4(s0[0], s0[1], s0[2], s0[3]);
        ((float4 *)(q_max + out))[lane] = make_float4(s1[0], s1[1], s1[2], s1[3]);
    }
}

// ----------------------------------------------------------------------------
// Stage 2: exact deterministic block-wide Top-K (radix select on the bit key)
// ----------------------------------------------------------------------------

static __device__ void sparse_block_topk(
        const float * __restrict__ scores,
        int cand_begin,
        int n_cand,
        int K,
        int32_t * __restrict__ out_pages,
        uint32_t * s_hist,     // [SPARSE_TOPK_THREADS]
        int * s_ctl,           // [24]
        int tid) {
    const int lane = tid & 31;
    const int warp = tid >> 5;

    uint32_t prefix = 0u;
    int need = K;

    if (tid == 0) {
        s_ctl[8] = 0;   // elected bucket (never read before a pass writes it)
        s_ctl[9] = 0;   // count of keys ranked above the elected bucket
    }
    __syncthreads();

    for (int shift = 24; shift >= 0; shift -= 8) {
        s_hist[tid] = 0u;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < SPARSE_TOPK_SLOTS; ++i) {
            const int idx = tid + i * SPARSE_TOPK_THREADS;
            if (idx < n_cand) {
                const uint32_t key = sparse_score_key(scores[cand_begin + idx]);
                if (shift == 24 || ((key >> (shift + 8)) == (prefix >> (shift + 8)))) {
                    atomicAdd(&s_hist[(key >> shift) & 0xFFu], 1u);
                }
            }
        }
        __syncthreads();

        const uint32_t hb = s_hist[tid];
        uint32_t suf = hb;
        #pragma unroll
        for (int off = 1; off < 32; off <<= 1) {
            const uint32_t t = __shfl_down(suf, off, 32);
            if (lane + off < 32) {
                suf += t;
            }
        }
        if (lane == 0) {
            s_ctl[warp] = (int)suf;
        }
        __syncthreads();
        if (tid == 0) {
            int acc = 0;
            for (int w = 7; w >= 0; --w) {
                const int t = s_ctl[w];
                s_ctl[w] = acc;
                acc += t;
            }
        }
        __syncthreads();

        const uint32_t above_warp = (uint32_t)s_ctl[warp];
        // Every lane MUST execute the shuffle (an inactive source lane 31 makes
        // the lane-30 result undefined on RDNA3); only the use is predicated.
        const uint32_t above_next_lane = __shfl_down(suf, 1, 32);
        const uint32_t above_within = (lane < 31) ? above_next_lane : 0u;
        const uint32_t above = above_warp + above_within;
        if (above < (uint32_t)need && above + hb >= (uint32_t)need) {
            s_ctl[8] = tid;
            s_ctl[9] = (int)above;
        }
        __syncthreads();
        prefix |= ((uint32_t)s_ctl[8]) << shift;
        need -= s_ctl[9];
    }

    // The last radix pass reads the elected bucket from s_ctl[8]/s_ctl[9];
    // the final phase below reuses those very slots for the gt/eq warp scans,
    // so every thread must finish reading the election result first.
    __syncthreads();

    int gt = 0, eq = 0;
    #pragma unroll
    for (int i = 0; i < SPARSE_TOPK_SLOTS; ++i) {
        const int idx = tid + i * SPARSE_TOPK_THREADS;
        if (idx < n_cand) {
            const uint32_t key = sparse_score_key(scores[cand_begin + idx]);
            if (key > prefix) {
                ++gt;
            } else if (key == prefix) {
                ++eq;
            }
        }
    }

    int gt_inc = gt, eq_inc = eq;
    #pragma unroll
    for (int off = 1; off < 32; off <<= 1) {
        const int tg = __shfl_up(gt_inc, off, 32);
        const int te = __shfl_up(eq_inc, off, 32);
        if (lane >= off) {
            gt_inc += tg;
            eq_inc += te;
        }
    }
    if (lane == 31) {
        s_ctl[warp]     = gt_inc;
        s_ctl[8 + warp] = eq_inc;
    }
    __syncthreads();
    if (tid == 0) {
        int ag = 0, ae = 0;
        for (int w = 0; w < 8; ++w) {
            const int tg = s_ctl[w];
            const int te = s_ctl[8 + w];
            s_ctl[w]     = ag;
            s_ctl[8 + w] = ae;
            ag += tg;
            ae += te;
        }
        s_ctl[16] = ag;
    }
    __syncthreads();

    int wgt = s_ctl[warp] + (gt_inc - gt);
    int weq = s_ctl[8 + warp] + (eq_inc - eq);
    const int n_gt = s_ctl[16];

    #pragma unroll
    for (int i = 0; i < SPARSE_TOPK_SLOTS; ++i) {
        const int idx = tid + i * SPARSE_TOPK_THREADS;
        if (idx < n_cand) {
            const uint32_t key = sparse_score_key(scores[cand_begin + idx]);
            if (key > prefix) {
                out_pages[wgt++] = cand_begin + idx;
            } else if (key == prefix && weq < need) {
                out_pages[n_gt + weq] = cand_begin + idx;
                ++weq;
            }
        }
    }
}

__launch_bounds__(SPARSE_TOPK_THREADS) __attribute__((amdgpu_waves_per_eu(8, 16)))
static __global__ void sparse_topk_pages_decode_kernel(
        const float * __restrict__ page_scores,
        const StepMeta * __restrict__ d_meta,
        int max_pages,
        int32_t * __restrict__ selected_pages,
        int32_t * __restrict__ n_selected,
        int sparse_pages) {
    const int h   = (int)blockIdx.x;
    const int tid = (int)threadIdx.x;

    const int n_kv    = d_meta->kv_len_global;
    const int n_pages = (n_kv + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;

    int32_t * out = selected_pages + (size_t)h * SPARSE_MAX_SELECTED;

    if (n_pages <= sparse_pages) {
        for (int i = tid; i < n_pages; i += SPARSE_TOPK_THREADS) {
            out[i] = i;
        }
        if (tid == 0) {
            n_selected[h] = n_pages;
        }
        return;
    }

    __shared__ uint32_t s_hist[SPARSE_TOPK_THREADS];
    __shared__ int s_ctl[24];

    const int cand_begin = SPARSE_SINK_PAGES;
    const int cand_end   = n_pages - SPARSE_RECENT_PAGES;
    const int n_cand     = cand_end - cand_begin;
    int K = sparse_pages - SPARSE_MIN_PAGES;
    K = min(K, SPARSE_MAX_SELECTED - SPARSE_SINK_PAGES - SPARSE_RECENT_PAGES);
    K = min(K, n_cand);

    if (tid == 0) {
        out[0] = 0;
        out[1] = 1;
        for (int i = 0; i < SPARSE_RECENT_PAGES; ++i) {
            out[SPARSE_SINK_PAGES + K + i] = n_pages - SPARSE_RECENT_PAGES + i;
        }
        n_selected[h] = SPARSE_SINK_PAGES + K + SPARSE_RECENT_PAGES;
    }
    __syncthreads();

    if (K > 0) {
        sparse_block_topk(page_scores + (size_t)h * (size_t)max_pages,
                          cand_begin, n_cand, K,
                          out + SPARSE_SINK_PAGES, s_hist, s_ctl, tid);
    }
}

__launch_bounds__(SPARSE_TOPK_THREADS) __attribute__((amdgpu_waves_per_eu(8, 16)))
static __global__ void sparse_topk_pages_prefill_kernel(
        const float * __restrict__ page_scores,
        int max_pages,
        int pos_start,
        int n_tokens,
        int32_t * __restrict__ selected_pages,
        int32_t * __restrict__ n_selected,
        int sparse_pages) {
    const int h   = (int)blockIdx.x;
    const int qb  = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;

    const int n_pages = (pos_start + n_tokens + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
    int32_t * out = selected_pages + ((size_t)qb * N_HEAD_KV + (size_t)h) * SPARSE_MAX_SELECTED;

    if (n_pages <= sparse_pages) {
        for (int i = tid; i < n_pages; i += SPARSE_TOPK_THREADS) {
            out[i] = i;
        }
        if (tid == 0) {
            n_selected[(size_t)qb * N_HEAD_KV + (size_t)h] = n_pages;
        }
        return;
    }

    __shared__ uint32_t s_hist[SPARSE_TOPK_THREADS];
    __shared__ int s_ctl[24];

    const int q_pos_start = pos_start + qb * SPARSE_PAGE_SIZE;
    const int q_pos_end   = pos_start + min((qb + 1) * SPARSE_PAGE_SIZE, n_tokens) - 1;
    const int local_start = min(sparse_local_start_page(q_pos_start), n_pages);
    const int local_end   = q_pos_end / SPARSE_PAGE_SIZE;
    const int n_rec       = (local_end >= local_start) ? (local_end - local_start + 1) : 0;

    const int cand_begin = SPARSE_SINK_PAGES;
    const int cand_end   = local_start;
    const int n_cand     = cand_end - cand_begin;
    int K = sparse_pages - SPARSE_MIN_PAGES;
    K = min(K, SPARSE_MAX_SELECTED - SPARSE_SINK_PAGES - n_rec);
    K = min(K, n_cand);

    if (tid == 0) {
        out[0] = 0;
        out[1] = 1;
        for (int i = 0; i < n_rec; ++i) {
            out[SPARSE_SINK_PAGES + K + i] = local_start + i;
        }
        n_selected[(size_t)qb * N_HEAD_KV + (size_t)h] = SPARSE_SINK_PAGES + K + n_rec;
    }
    __syncthreads();

    if (K > 0) {
        sparse_block_topk(page_scores + ((size_t)qb * N_HEAD_KV + (size_t)h) * (size_t)max_pages,
                          cand_begin, n_cand, K,
                          out + SPARSE_SINK_PAGES, s_hist, s_ctl, tid);
    }
}

// ----------------------------------------------------------------------------
// Decode sparse Global FlashAttention (pages partitioned across n_splits)
// ----------------------------------------------------------------------------

template <int GQA, int KV_FMT, bool USE_R_PREP>
__launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(12, 16)))
static __global__ void inkling_fa_decode_sparse_global_kernel(
        const float    * __restrict__ q,
        KVCacheSlot                   kv_slot,
        const float    * __restrict__ sinks_in,
        int                           sinks_stride,
        const float    * __restrict__ rel_table,
        const float    * __restrict__ r_prep,
        const float    * __restrict__ attn_rel_bias,
        const float    * __restrict__ qgate_in,
        int                           qgate_stride,
        float          * __restrict__ out,
        float          * __restrict__ tmp_o,
        float2         * __restrict__ tmp_meta,
        const int32_t  * __restrict__ selected_pages,
        const int32_t  * __restrict__ n_selected,
        int                           n_splits,
        const StepMeta * __restrict__ d_meta,
        int32_t        * __restrict__ fa_done) {
    constexpr int D      = HEAD_DIM;
    constexpr int NWARPS = 4;
    constexpr int TOK    = 0;   // decode is always a single token

    const int split  = (int)blockIdx.x;
    const int grp_id = (int)blockIdx.y;

    const int h_base = grp_id * GQA;
    const int h_kv   = grp_id;

    const int lane   = (int)threadIdx.x;
    const int warp   = (int)threadIdx.y;
    const int blk    = lane & 3;
    const int k_slot = lane >> 2;

    const int32_t qp_val = d_meta->pos;
    const int32_t n_kv   = d_meta->kv_len_global;

    const int n_sel = n_selected[grp_id];
    const int pages_per_split = (n_sel + n_splits - 1) / n_splits;
    const int pi_start = min(split * pages_per_split, n_sel);
    const int pi_end   = min(pi_start + pages_per_split, n_sel);

    union DecodeSmem {
        struct {
            float4 q4[GQA][36];
            float  q_sum[GQA][4];
        } q;
        struct {
            float o[NWARPS][GQA][D];
            float m[NWARPS][GQA];
            float l[NWARPS][GQA];
        } red;
    };
    __shared__ DecodeSmem smem;
    __shared__ int s_is_last;

    if (pi_start < pi_end) {
        constexpr float Q_SCALE_LOG2 = KQ_SCALE * LOG2E;
        if (warp < GQA) {
            const float * qp = q + (size_t)(h_base + warp) * D;
            const float4 fv  = ((const float4 *)qp)[lane];
            const int q_b = lane >> 3;
            const int q_i = lane & 7;
            const float4 qs4 = make_float4(
                fv.x * Q_SCALE_LOG2,
                fv.y * Q_SCALE_LOG2,
                fv.z * Q_SCALE_LOG2,
                fv.w * Q_SCALE_LOG2);
            smem.q.q4[warp][q_b * 9 + q_i] = qs4;
            if constexpr (KV_FMT == KV_Q4_1) {
                float s32 = (qs4.x + qs4.y) + (qs4.z + qs4.w);
                s32 += __shfl_xor(s32, 4, 8);
                s32 += __shfl_xor(s32, 2, 8);
                s32 += __shfl_xor(s32, 1, 8);
                if (q_i == 0) {
                    smem.q.q_sum[warp][q_b] = s32;
                }
            }
        }
        __syncthreads();

        const int q4_off = blk * 9;

        float o_reg[GQA][4];
        float m_reg[GQA];
        float l_reg[GQA];
        #pragma unroll
        for (int g = 0; g < GQA; ++g) {
            o_reg[g][0] = 0.0f;
            o_reg[g][1] = 0.0f;
            o_reg[g][2] = 0.0f;
            o_reg[g][3] = 0.0f;
            m_reg[g] = -INFINITY;
            l_reg[g] = 0.0f;
        }

        const float * rel_base = USE_R_PREP ? nullptr
                                             : rel_table + (size_t)TOK * N_HEAD * N_REL_EXTENT
                                                         + (size_t)h_base * N_REL_EXTENT;
        // One 64-bit base for the GQA r_prep rows (+g*N_REL folds into immediate offsets) instead
        // of three separately sign-extended row pointers: clang 22 ran out of SGPRs (106 + VCC)
        // in the USE_R_PREP Q4 variants and spilled qgate_in to a VGPR lane (issue #6).
        const float * rp_base = USE_R_PREP ? r_prep + (size_t)h_base * N_REL : nullptr;
        const int v_blk = lane >> 3;
        const int32_t * sel = selected_pages + (size_t)grp_id * SPARSE_MAX_SELECTED;
        const size_t head_base = (size_t)h_kv * (size_t)kv_slot.cap;

        for (int pi = pi_start; pi < pi_end; ++pi) {
            const int page_base = (int)sel[pi] * SPARSE_PAGE_SIZE;
            for (int j0 = warp * 8; j0 < SPARSE_PAGE_SIZE; j0 += NWARPS * 8) {
                const int j_base = page_base + j0;
                const int j = j_base + k_slot;
                const bool in_range = (j < n_kv);
                int32_t kp_val = -1;
                __half d_kh = __float2half(0.0f);
                __half2 dm_kh = h2_of(0u);
                v4u u0 = {0, 0, 0, 0}, u1 = {0, 0, 0, 0};
                if (in_range) {
                    kp_val = kv_slot.k_pos[j];
                    const size_t head_slot = head_base + (size_t)j;
                    if constexpr (KV_FMT == KV_Q8_0) {
                        d_kh = __ldg(&kv_slot.k_d[head_slot * 4 + (size_t)blk]);
                        const int8_t * k_ptr = kv_slot.k_qs + head_slot * D + (size_t)blk * 32;
                        u0 = __builtin_nontemporal_load((const v4u *)(k_ptr + 0));
                        u1 = __builtin_nontemporal_load((const v4u *)(k_ptr + 16));
                    } else {
                        if constexpr (KV_FMT == KV_Q4_1) {
                            dm_kh = __ldg(&((const __half2 *)kv_slot.k_d)[head_slot * 4 + (size_t)blk]);
                        } else {
                            d_kh = __ldg(&kv_slot.k_d[head_slot * 4 + (size_t)blk]);
                        }
                        const int8_t * k_ptr = kv_slot.k_qs + head_slot * (D / 2) + (size_t)blk * 16;
                        u0 = __builtin_nontemporal_load((const v4u *)k_ptr);
                    }
                }

                const bool valid = (kp_val >= 0) && (kp_val <= qp_val);
                const uint32_t valid_ballot = __ballot(valid);
                if (valid_ballot == 0u) {
                    continue;
                }

                if (valid_ballot == 0xFFFFFFFFu) {
                    const size_t base_slot = head_base + (size_t)j_base;
                    float vd_lane = 0.0f;
                    uint32_t vd_u_lane = 0u;
                    uint32_t vq8[8];
                    uint32_t vq4[4];
                    if constexpr (KV_FMT == KV_Q8_0) {
                        vd_lane = __half2float(__ldg(&kv_slot.v_d[base_slot * 4 + (size_t)lane]));
                        #pragma unroll
                        for (int kk = 0; kk < 8; ++kk) {
                            vq8[kk] = __builtin_nontemporal_load(
                                ((const uint32_t *)(kv_slot.v_qs + (base_slot + (size_t)kk) * D)) + lane);
                        }
                    } else {
                        if constexpr (KV_FMT == KV_Q4_0) {
                            vd_lane = __half2float(__ldg(&kv_slot.v_d[base_slot * 4 + (size_t)lane]));
                        }
                        const int v_half_lane = lane >> 1;
                        const int v_odd_lane  = lane & 1;
                        #pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            const size_t v_slot = base_slot + (size_t)(2 * p + v_odd_lane);
                            const uint32_t * vp = ((const uint32_t *)(kv_slot.v_qs + v_slot * (D / 2)))
                                                + v_half_lane;
                            vq4[p] = __builtin_nontemporal_load(vp);
                        }
                    }

                    float sc_k[GQA] = {0.0f, 0.0f, 0.0f};
                    if constexpr (KV_FMT == KV_Q8_0) {
                        const uint32_t ku[8] = { u0[0], u0[1], u0[2], u0[3], u1[0], u1[1], u1[2], u1[3] };
                        #pragma unroll
                        for (int i = 0; i < 8; ++i) {
                            __half2 kh01, kh23;
                            i8x4_to_h2x2(ku[i], kh01, kh23);
                            #pragma unroll
                            for (int g = 0; g < GQA; ++g) {
                                const float4 qv = smem.q.q4[g][q4_off + i];
                                sc_k[g] = fmaf(__low2float(kh01),  qv.x, sc_k[g]);
                                sc_k[g] = fmaf(__high2float(kh01), qv.y, sc_k[g]);
                                sc_k[g] = fmaf(__low2float(kh23),  qv.z, sc_k[g]);
                                sc_k[g] = fmaf(__high2float(kh23), qv.w, sc_k[g]);
                            }
                        }
                    } else {
                        const uint32_t ku[4] = { u0[0], u0[1], u0[2], u0[3] };
                        #pragma unroll
                        for (int i = 0; i < 4; ++i) {
                            __half2 kh01, kh23;
                            u4x4_to_h2x2<KV_FMT == KV_Q4_0>(ku[i] & 0x0F0F0F0Fu, kh01, kh23);
                            #pragma unroll
                            for (int g = 0; g < GQA; ++g) {
                                const float4 qv0 = smem.q.q4[g][q4_off + 2 * i + 0];
                                sc_k[g] = fmaf(__low2float(kh01),  qv0.x, sc_k[g]);
                                sc_k[g] = fmaf(__high2float(kh01), qv0.y, sc_k[g]);
                                sc_k[g] = fmaf(__low2float(kh23),  qv0.z, sc_k[g]);
                                sc_k[g] = fmaf(__high2float(kh23), qv0.w, sc_k[g]);
                            }
                            u4x4_to_h2x2<KV_FMT == KV_Q4_0>((ku[i] >> 4) & 0x0F0F0F0Fu, kh01, kh23);
                            #pragma unroll
                            for (int g = 0; g < GQA; ++g) {
                                const float4 qv1 = smem.q.q4[g][q4_off + 2 * i + 1];
                                sc_k[g] = fmaf(__low2float(kh01),  qv1.x, sc_k[g]);
                                sc_k[g] = fmaf(__high2float(kh01), qv1.y, sc_k[g]);
                                sc_k[g] = fmaf(__low2float(kh23),  qv1.z, sc_k[g]);
                                sc_k[g] = fmaf(__high2float(kh23), qv1.w, sc_k[g]);
                            }
                        }
                    }
                    #pragma unroll
                    for (int g = 0; g < GQA; ++g) {
                        float p = (KV_FMT == KV_Q4_1)
                            ? fmaf(sc_k[g], __low2float(dm_kh), __high2float(dm_kh) * smem.q.q_sum[g][blk])
                            : (sc_k[g] * __half2float(d_kh));
                        p += __shfl_xor(p, 1, 32);
                        p += __shfl_xor(p, 2, 32);
                        sc_k[g] = p;
                    }

                    if (blk == 0) {
                        const int64_t d64 = (int64_t)qp_val - (int64_t)kp_val;
                        if (d64 >= 0 && d64 < N_REL_EXTENT) {
                            const int dist = (int)d64;
                            #pragma unroll
                            for (int g = 0; g < GQA; ++g) {
                                const float * rp_row = USE_R_PREP
                                    ? rp_base + (size_t)g * N_REL : nullptr;
                                sc_k[g] += rel_logit<USE_R_PREP>(
                                    rel_base + (size_t)g * N_REL_EXTENT, rp_row,
                                    attn_rel_bias, dist) * LOG2E;
                            }
                        }
                    }

                    float w_k[GQA];
                    #pragma unroll
                    for (int g = 0; g < GQA; ++g) {
                        float mt = sc_k[g];
                        mt = fmaxf(mt, __shfl_xor(mt,  4, 32));
                        mt = fmaxf(mt, __shfl_xor(mt,  8, 32));
                        mt = fmaxf(mt, __shfl_xor(mt, 16, 32));
                        mt = __shfl(mt, 0, 32);

                        const float m_old = m_reg[g];
                        const float m_new = fmaxf(m_old, mt);
                        if (m_new > m_old) {
                            if (m_old > -INFINITY) {
                                const float scale_old = exp2f(m_old - m_new);
                                o_reg[g][0] *= scale_old;
                                o_reg[g][1] *= scale_old;
                                o_reg[g][2] *= scale_old;
                                o_reg[g][3] *= scale_old;
                                l_reg[g]    *= scale_old;
                            }
                            m_reg[g] = m_new;
                        }

                        const float wk = (blk == 0) ? exp2f(sc_k[g] - m_new) : 0.0f;
                        w_k[g] = wk;

                        float lt = wk;
                        lt += __shfl_xor(lt,  4, 32);
                        lt += __shfl_xor(lt,  8, 32);
                        lt += __shfl_xor(lt, 16, 32);
                        l_reg[g] += lt;
                    }

                    if constexpr (KV_FMT == KV_Q8_0) {
                        #pragma unroll
                        for (int kk = 0; kk < 8; ++kk) {
                            const float d_v = __shfl(vd_lane, kk * 4 + v_blk, 32);
                            __half2 vh01, vh23;
                            i8x4_to_h2x2(vq8[kk], vh01, vh23);
                            #pragma unroll
                            for (int g = 0; g < GQA; ++g) {
                                const float w_dv = __shfl(w_k[g], kk * 4, 32) * d_v;
                                o_reg[g][0] = fmaf(__low2float(vh01),  w_dv, o_reg[g][0]);
                                o_reg[g][1] = fmaf(__high2float(vh01), w_dv, o_reg[g][1]);
                                o_reg[g][2] = fmaf(__low2float(vh23),  w_dv, o_reg[g][2]);
                                o_reg[g][3] = fmaf(__high2float(vh23), w_dv, o_reg[g][3]);
                            }
                        }
                    } else {
                        if constexpr (KV_FMT == KV_Q4_1) {
                            vd_u_lane = __ldg(&((const uint32_t *)kv_slot.v_d)[base_slot * 4 + (size_t)lane]);
                        }
                        #pragma unroll
                        for (int p = 0; p < 4; ++p) {
                            const uint32_t w_me = vq4[p];
                            const uint32_t w_ot = __shfl_xor(w_me, 1, 32);
                            #pragma unroll
                            for (int s = 0; s < 2; ++s) {
                                const int kk = 2 * p + s;
                                const uint32_t w_cur = (((lane ^ s) & 1) ? w_ot : w_me);
                                const uint32_t nibs = ((lane & 1) ? (w_cur >> 4) : w_cur) & 0x0F0F0F0Fu;
                                __half2 vh01, vh23;
                                u4x4_to_h2x2<KV_FMT == KV_Q4_0>(nibs, vh01, vh23);
                                if constexpr (KV_FMT == KV_Q4_1) {
                                    const __half2 dm_v = h2_of(__shfl(vd_u_lane, kk * 4 + v_blk, 32));
                                    #pragma unroll
                                    for (int g = 0; g < GQA; ++g) {
                                        const float wg   = __shfl(w_k[g], kk * 4, 32);
                                        const float w_dv = wg * __low2float(dm_v);
                                        const float w_mv = wg * __high2float(dm_v);
                                        o_reg[g][0] = fmaf(__low2float(vh01),  w_dv, o_reg[g][0] + w_mv);
                                        o_reg[g][1] = fmaf(__high2float(vh01), w_dv, o_reg[g][1] + w_mv);
                                        o_reg[g][2] = fmaf(__low2float(vh23),  w_dv, o_reg[g][2] + w_mv);
                                        o_reg[g][3] = fmaf(__high2float(vh23), w_dv, o_reg[g][3] + w_mv);
                                    }
                                } else {
                                    const float d_v = __shfl(vd_lane, kk * 4 + v_blk, 32);
                                    #pragma unroll
                                    for (int g = 0; g < GQA; ++g) {
                                        const float w_dv = __shfl(w_k[g], kk * 4, 32) * d_v;
                                        o_reg[g][0] = fmaf(__low2float(vh01),  w_dv, o_reg[g][0]);
                                        o_reg[g][1] = fmaf(__high2float(vh01), w_dv, o_reg[g][1]);
                                        o_reg[g][2] = fmaf(__low2float(vh23),  w_dv, o_reg[g][2]);
                                        o_reg[g][3] = fmaf(__high2float(vh23), w_dv, o_reg[g][3]);
                                    }
                                }
                            }
                        }
                    }
                } else {
                    float sc_k[GQA] = {0.0f, 0.0f, 0.0f};
                    if (valid) {
                        if constexpr (KV_FMT == KV_Q8_0) {
                            const uint32_t ku[8] = { u0[0], u0[1], u0[2], u0[3], u1[0], u1[1], u1[2], u1[3] };
                            #pragma unroll
                            for (int i = 0; i < 8; ++i) {
                                __half2 kh01, kh23;
                                i8x4_to_h2x2(ku[i], kh01, kh23);
                                #pragma unroll
                                for (int g = 0; g < GQA; ++g) {
                                    const float4 qv = smem.q.q4[g][q4_off + i];
                                    sc_k[g] = fmaf(__low2float(kh01),  qv.x, sc_k[g]);
                                    sc_k[g] = fmaf(__high2float(kh01), qv.y, sc_k[g]);
                                    sc_k[g] = fmaf(__low2float(kh23),  qv.z, sc_k[g]);
                                    sc_k[g] = fmaf(__high2float(kh23), qv.w, sc_k[g]);
                                }
                            }
                        } else {
                            const uint32_t ku[4] = { u0[0], u0[1], u0[2], u0[3] };
                            #pragma unroll
                            for (int i = 0; i < 4; ++i) {
                                __half2 kh01, kh23;
                                u4x4_to_h2x2<KV_FMT == KV_Q4_0>(ku[i] & 0x0F0F0F0Fu, kh01, kh23);
                                #pragma unroll
                                for (int g = 0; g < GQA; ++g) {
                                    const float4 qv0 = smem.q.q4[g][q4_off + 2 * i + 0];
                                    sc_k[g] = fmaf(__low2float(kh01),  qv0.x, sc_k[g]);
                                    sc_k[g] = fmaf(__high2float(kh01), qv0.y, sc_k[g]);
                                    sc_k[g] = fmaf(__low2float(kh23),  qv0.z, sc_k[g]);
                                    sc_k[g] = fmaf(__high2float(kh23), qv0.w, sc_k[g]);
                                }
                                u4x4_to_h2x2<KV_FMT == KV_Q4_0>((ku[i] >> 4) & 0x0F0F0F0Fu, kh01, kh23);
                                #pragma unroll
                                for (int g = 0; g < GQA; ++g) {
                                    const float4 qv1 = smem.q.q4[g][q4_off + 2 * i + 1];
                                    sc_k[g] = fmaf(__low2float(kh01),  qv1.x, sc_k[g]);
                                    sc_k[g] = fmaf(__high2float(kh01), qv1.y, sc_k[g]);
                                    sc_k[g] = fmaf(__low2float(kh23),  qv1.z, sc_k[g]);
                                    sc_k[g] = fmaf(__high2float(kh23), qv1.w, sc_k[g]);
                                }
                            }
                        }
                        const float d_k = (KV_FMT == KV_Q4_1) ? __low2float(dm_kh) : __half2float(d_kh);
                        const float m_k = (KV_FMT == KV_Q4_1) ? __high2float(dm_kh) : 0.0f;
                        #pragma unroll
                        for (int g = 0; g < GQA; ++g) {
                            sc_k[g] = (KV_FMT == KV_Q4_1)
                                ? fmaf(sc_k[g], d_k, m_k * smem.q.q_sum[g][blk])
                                : (sc_k[g] * d_k);
                        }
                    }

                    #pragma unroll
                    for (int g = 0; g < GQA; ++g) {
                        float p = sc_k[g];
                        p += __shfl_xor(p, 1, 32);
                        p += __shfl_xor(p, 2, 32);
                        sc_k[g] = valid ? p : -INFINITY;
                    }

                    if (blk == 0 && valid) {
                        const int64_t d64 = (int64_t)qp_val - (int64_t)kp_val;
                        if (d64 >= 0 && d64 < N_REL_EXTENT) {
                            const int dist = (int)d64;
                            #pragma unroll
                            for (int g = 0; g < GQA; ++g) {
                                const float * rp_row = USE_R_PREP
                                    ? rp_base + (size_t)g * N_REL : nullptr;
                                sc_k[g] += rel_logit<USE_R_PREP>(
                                    rel_base + (size_t)g * N_REL_EXTENT, rp_row,
                                    attn_rel_bias, dist) * LOG2E;
                            }
                        }
                    }

                    float w_k[GQA];
                    #pragma unroll
                    for (int g = 0; g < GQA; ++g) {
                        float mt = sc_k[g];
                        mt = fmaxf(mt, __shfl_xor(mt,  4, 32));
                        mt = fmaxf(mt, __shfl_xor(mt,  8, 32));
                        mt = fmaxf(mt, __shfl_xor(mt, 16, 32));
                        mt = __shfl(mt, 0, 32);

                        const float m_old = m_reg[g];
                        const float m_new = fmaxf(m_old, mt);
                        if (m_new > m_old) {
                            if (m_old > -INFINITY) {
                                const float scale_old = exp2f(m_old - m_new);
                                o_reg[g][0] *= scale_old;
                                o_reg[g][1] *= scale_old;
                                o_reg[g][2] *= scale_old;
                                o_reg[g][3] *= scale_old;
                                l_reg[g]    *= scale_old;
                            }
                            m_reg[g] = m_new;
                        }

                        const float wk = (blk == 0 && valid) ? exp2f(sc_k[g] - m_new) : 0.0f;
                        w_k[g] = wk;

                        float lt = wk;
                        lt += __shfl_xor(lt,  4, 32);
                        lt += __shfl_xor(lt,  8, 32);
                        lt += __shfl_xor(lt, 16, 32);
                        l_reg[g] += lt;
                    }

                    #pragma unroll
                    for (int kk = 0; kk < 8; ++kk) {
                        if (!((valid_ballot >> (kk * 4)) & 1u)) {
                            continue;
                        }
                        const size_t head_slot = head_base + (size_t)(j_base + kk);
                        __half2 vh01, vh23;
                        if constexpr (KV_FMT == KV_Q8_0) {
                            const float d_v = __half2float(__ldg(&kv_slot.v_d[head_slot * 4 + (size_t)v_blk]));
                            const uint32_t q4 = *((const uint32_t *)(kv_slot.v_qs + head_slot * D) + lane);
                            i8x4_to_h2x2(q4, vh01, vh23);
                            #pragma unroll
                            for (int g = 0; g < GQA; ++g) {
                                const float w_dv = __shfl(w_k[g], kk * 4, 32) * d_v;
                                o_reg[g][0] = fmaf(__low2float(vh01),  w_dv, o_reg[g][0]);
                                o_reg[g][1] = fmaf(__high2float(vh01), w_dv, o_reg[g][1]);
                                o_reg[g][2] = fmaf(__low2float(vh23),  w_dv, o_reg[g][2]);
                                o_reg[g][3] = fmaf(__high2float(vh23), w_dv, o_reg[g][3]);
                            }
                        } else {
                            const uint32_t w = *((const uint32_t *)(kv_slot.v_qs + head_slot * (D / 2))
                                                 + (size_t)(lane >> 1));
                            const uint32_t nibs = ((lane & 1) ? (w >> 4) : w) & 0x0F0F0F0Fu;
                            u4x4_to_h2x2<KV_FMT == KV_Q4_0>(nibs, vh01, vh23);
                            if constexpr (KV_FMT == KV_Q4_1) {
                                const __half2 dm_v = __ldg(&((const __half2 *)kv_slot.v_d)[head_slot * 4 + (size_t)v_blk]);
                                const float d_v = __low2float(dm_v);
                                const float m_v = __high2float(dm_v);
                                #pragma unroll
                                for (int g = 0; g < GQA; ++g) {
                                    const float wg   = __shfl(w_k[g], kk * 4, 32);
                                    const float w_dv = wg * d_v;
                                    const float w_mv = wg * m_v;
                                    o_reg[g][0] = fmaf(__low2float(vh01),  w_dv, o_reg[g][0] + w_mv);
                                    o_reg[g][1] = fmaf(__high2float(vh01), w_dv, o_reg[g][1] + w_mv);
                                    o_reg[g][2] = fmaf(__low2float(vh23),  w_dv, o_reg[g][2] + w_mv);
                                    o_reg[g][3] = fmaf(__high2float(vh23), w_dv, o_reg[g][3] + w_mv);
                                }
                            } else {
                                const float d_v = __half2float(__ldg(&kv_slot.v_d[head_slot * 4 + (size_t)v_blk]));
                                #pragma unroll
                                for (int g = 0; g < GQA; ++g) {
                                    const float w_dv = __shfl(w_k[g], kk * 4, 32) * d_v;
                                    o_reg[g][0] = fmaf(__low2float(vh01),  w_dv, o_reg[g][0]);
                                    o_reg[g][1] = fmaf(__high2float(vh01), w_dv, o_reg[g][1]);
                                    o_reg[g][2] = fmaf(__low2float(vh23),  w_dv, o_reg[g][2]);
                                    o_reg[g][3] = fmaf(__high2float(vh23), w_dv, o_reg[g][3]);
                                }
                            }
                        }
                    }
                }
            }
        }

        __syncthreads();

        #pragma unroll
        for (int g = 0; g < GQA; ++g) {
            if (lane == 0) {
                smem.red.m[warp][g] = m_reg[g];
                smem.red.l[warp][g] = l_reg[g];
            }
            *(float4 *)(&smem.red.o[warp][g][lane * 4]) =
                make_float4(o_reg[g][0], o_reg[g][1], o_reg[g][2], o_reg[g][3]);
        }
        __syncthreads();

        if (warp == 0) {
            #pragma unroll
            for (int g = 0; g < GQA; ++g) {
                float m_blk = -INFINITY;
                #pragma unroll
                for (int w = 0; w < NWARPS; ++w) {
                    m_blk = fmaxf(m_blk, smem.red.m[w][g]);
                }

                float l_blk = 0.0f;
                float oa0 = 0.0f, oa1 = 0.0f, oa2 = 0.0f, oa3 = 0.0f;
                if (m_blk > -INFINITY) {
                    #pragma unroll
                    for (int w = 0; w < NWARPS; ++w) {
                        const float mw = smem.red.m[w][g];
                        if (mw > -INFINITY) {
                            const float sw = exp2f(mw - m_blk);
                            l_blk += smem.red.l[w][g] * sw;
                            const float4 ov = *(const float4 *)(&smem.red.o[w][g][lane * 4]);
                            oa0 = fmaf(ov.x, sw, oa0);
                            oa1 = fmaf(ov.y, sw, oa1);
                            oa2 = fmaf(ov.z, sw, oa2);
                            oa3 = fmaf(ov.w, sw, oa3);
                        }
                    }
                }

                const int h = h_base + g;
                if (n_splits == 1) {
                    const float sink_val_log2 = sinks_in[(size_t)TOK * sinks_stride + h] * LOG2E;
                    const float m_final  = fmaxf(m_blk, sink_val_log2);
                    const float scale_o  = (m_blk > -INFINITY) ? exp2f(m_blk - m_final) : 0.0f;
                    const float l_final  = l_blk * scale_o + exp2f(sink_val_log2 - m_final);
                    const float norm     = (scale_o / l_final) * KQ_SCALE;

                    float v0 = oa0 * norm;
                    float v1 = oa1 * norm;
                    float v2 = oa2 * norm;
                    float v3 = oa3 * norm;
                    wave32_fa_fwht128_4(v0, v1, v2, v3, lane);

                    if (qgate_in != nullptr) {
                        const float * g_ptr = qgate_in + (size_t)TOK * qgate_stride
                                            + (size_t)h * D + (size_t)lane * 4;
                        const float4 gv = *(const float4 *)g_ptr;
                        v0 *= 1.0f / (1.0f + exp2f(-gv.x * LOG2E));
                        v1 *= 1.0f / (1.0f + exp2f(-gv.y * LOG2E));
                        v2 *= 1.0f / (1.0f + exp2f(-gv.z * LOG2E));
                        v3 *= 1.0f / (1.0f + exp2f(-gv.w * LOG2E));
                    }
                    *(float4 *)(out + (size_t)h * D + (size_t)lane * 4) = make_float4(v0, v1, v2, v3);
                } else {
                    const int64_t meta_idx = ((int64_t)TOK * N_HEAD + h) * n_splits + split;
                    if (lane == 0) {
                        tmp_meta[meta_idx] = make_float2(m_blk, l_blk);
                    }
                    *(float4 *)(tmp_o + meta_idx * D + (size_t)lane * 4) = make_float4(oa0, oa1, oa2, oa3);
                }
            }
        }
    } else {
        if (warp == 0 && lane == 0) {
            #pragma unroll
            for (int g = 0; g < GQA; ++g) {
                const int h = h_base + g;
                const int64_t meta_idx = ((int64_t)TOK * N_HEAD + h) * n_splits + split;
                tmp_meta[meta_idx] = make_float2(-INFINITY, 0.0f);
            }
        }
    }

    if (fa_done != nullptr && n_splits > 1) {
        __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
        asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#endif
        __syncthreads();
        if (warp == 0 && lane == 0) {
            __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
            asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#endif
            const int prev = __hip_atomic_fetch_add(
                &fa_done[grp_id], 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            s_is_last = (prev == n_splits - 1) ? 1 : 0;
        }
        __syncthreads();

        if (s_is_last) {
            __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
            asm volatile("buffer_gl1_inv\n\tbuffer_gl0_inv" ::: "memory");
#endif
            if (warp == 0 && lane == 0) {
                __hip_atomic_store(&fa_done[grp_id], 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
            if (warp < GQA) {
                const int h = h_base + warp;
                const int64_t base_idx = ((int64_t)TOK * N_HEAD + h) * n_splits;
                const float sink_val_log2 = sinks_in[(size_t)TOK * sinks_stride + h] * LOG2E;

                const float2 ml_lane = (lane < n_splits)
                    ? tmp_meta[base_idx + lane]
                    : make_float2(-INFINITY, 0.0f);
                float m_all = ml_lane.x;
                m_all = fmaxf(m_all, __shfl_xor(m_all, 16, 32));
                m_all = fmaxf(m_all, __shfl_xor(m_all,  8, 32));
                m_all = fmaxf(m_all, __shfl_xor(m_all,  4, 32));
                m_all = fmaxf(m_all, __shfl_xor(m_all,  2, 32));
                m_all = fmaxf(m_all, __shfl_xor(m_all,  1, 32));
                const float m_final = fmaxf(m_all, sink_val_log2);

                const float sc_lane = (lane < n_splits && ml_lane.x > -INFINITY)
                    ? exp2f(ml_lane.x - m_final) : 0.0f;
                float l_sum = ml_lane.y * sc_lane;
                const uint32_t valid_mask = __ballot(sc_lane > 0.0f);

                float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
                int sp = 0;
                for (; sp + 4 <= n_splits; sp += 4) {
                    const uint32_t m4 = (valid_mask >> sp) & 0xFu;
                    if (m4 == 0xFu) {
                        const float sc0 = __shfl(sc_lane, sp + 0, 32);
                        const float sc1 = __shfl(sc_lane, sp + 1, 32);
                        const float sc2 = __shfl(sc_lane, sp + 2, 32);
                        const float sc3 = __shfl(sc_lane, sp + 3, 32);
                        const float4 pv0 = *(const float4 *)(tmp_o + (base_idx + sp + 0) * D + lane * 4);
                        const float4 pv1 = *(const float4 *)(tmp_o + (base_idx + sp + 1) * D + lane * 4);
                        const float4 pv2 = *(const float4 *)(tmp_o + (base_idx + sp + 2) * D + lane * 4);
                        const float4 pv3 = *(const float4 *)(tmp_o + (base_idx + sp + 3) * D + lane * 4);
                        o0 = fmaf(pv0.x, sc0, o0); o1 = fmaf(pv0.y, sc0, o1); o2 = fmaf(pv0.z, sc0, o2); o3 = fmaf(pv0.w, sc0, o3);
                        o0 = fmaf(pv1.x, sc1, o0); o1 = fmaf(pv1.y, sc1, o1); o2 = fmaf(pv1.z, sc1, o2); o3 = fmaf(pv1.w, sc1, o3);
                        o0 = fmaf(pv2.x, sc2, o0); o1 = fmaf(pv2.y, sc2, o1); o2 = fmaf(pv2.z, sc2, o2); o3 = fmaf(pv2.w, sc2, o3);
                        o0 = fmaf(pv3.x, sc3, o0); o1 = fmaf(pv3.y, sc3, o1); o2 = fmaf(pv3.z, sc3, o2); o3 = fmaf(pv3.w, sc3, o3);
                    } else if (m4 != 0u) {
                        #pragma unroll
                        for (int k = 0; k < 4; ++k) {
                            if ((m4 >> k) & 1u) {
                                const float sc = __shfl(sc_lane, sp + k, 32);
                                const float4 pv = *(const float4 *)(tmp_o + (base_idx + sp + k) * D + lane * 4);
                                o0 = fmaf(pv.x, sc, o0); o1 = fmaf(pv.y, sc, o1); o2 = fmaf(pv.z, sc, o2); o3 = fmaf(pv.w, sc, o3);
                            }
                        }
                    }
                }
                for (; sp < n_splits; ++sp) {
                    if ((valid_mask >> sp) & 1u) {
                        const float sc = __shfl(sc_lane, sp, 32);
                        const float4 pv = *(const float4 *)(tmp_o + (base_idx + sp) * D + lane * 4);
                        o0 = fmaf(pv.x, sc, o0); o1 = fmaf(pv.y, sc, o1); o2 = fmaf(pv.z, sc, o2); o3 = fmaf(pv.w, sc, o3);
                    }
                }

                l_sum += __shfl_xor(l_sum, 16, 32);
                l_sum += __shfl_xor(l_sum,  8, 32);
                l_sum += __shfl_xor(l_sum,  4, 32);
                l_sum += __shfl_xor(l_sum,  2, 32);
                l_sum += __shfl_xor(l_sum,  1, 32);
                const float l_final = l_sum + exp2f(sink_val_log2 - m_final);

                const float inv_l = (1.0f / l_final) * KQ_SCALE;
                float v0 = o0 * inv_l;
                float v1 = o1 * inv_l;
                float v2 = o2 * inv_l;
                float v3 = o3 * inv_l;
                wave32_fa_fwht128_4(v0, v1, v2, v3, lane);

                if (qgate_in != nullptr) {
                    const float4 gv = *(const float4 *)(qgate_in + (size_t)TOK * qgate_stride
                                                       + (size_t)h * D + (size_t)lane * 4);
                    v0 *= 1.0f / (1.0f + exp2f(-gv.x * LOG2E));
                    v1 *= 1.0f / (1.0f + exp2f(-gv.y * LOG2E));
                    v2 *= 1.0f / (1.0f + exp2f(-gv.z * LOG2E));
                    v3 *= 1.0f / (1.0f + exp2f(-gv.w * LOG2E));
                }
                *(float4 *)(out + (size_t)h * D + (size_t)lane * 4) = make_float4(v0, v1, v2, v3);
            }
        }
    }
}

__launch_bounds__(96, 8)
static __global__ void inkling_fa_decode_sparse_reduce_kernel(
        const float  * __restrict__ tmp_o,
        const float2 * __restrict__ tmp_meta,
        const float  * __restrict__ sinks_in,
        const float  * __restrict__ qgate_in,
        float        * __restrict__ out,
        int           n_splits) {
    constexpr int D  = HEAD_DIM;
    constexpr int TOK = 0;
    const int h_kv = (int)blockIdx.x;
    const int g    = (int)threadIdx.y;
    const int h    = h_kv * GQA_RATIO + g;
    const int lane = (int)threadIdx.x;

    const int64_t base_idx = ((int64_t)TOK * N_HEAD + h) * n_splits;
    const float sink_val_log2 = sinks_in[(size_t)h] * LOG2E;

    const float2 ml_lane = (lane < n_splits)
        ? tmp_meta[base_idx + lane]
        : make_float2(-INFINITY, 0.0f);
    float m_all = ml_lane.x;
    m_all = fmaxf(m_all, __shfl_xor(m_all, 16, 32));
    m_all = fmaxf(m_all, __shfl_xor(m_all,  8, 32));
    m_all = fmaxf(m_all, __shfl_xor(m_all,  4, 32));
    m_all = fmaxf(m_all, __shfl_xor(m_all,  2, 32));
    m_all = fmaxf(m_all, __shfl_xor(m_all,  1, 32));
    const float m_final = fmaxf(m_all, sink_val_log2);

    const float sc_lane = (lane < n_splits && ml_lane.x > -INFINITY)
        ? fast_exp2f(ml_lane.x - m_final) : 0.0f;
    float l_sum = ml_lane.y * sc_lane;
    const uint32_t valid_mask = __ballot(sc_lane > 0.0f);

    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    int sp = 0;
    for (; sp + 4 <= n_splits; sp += 4) {
        const uint32_t m4 = (valid_mask >> sp) & 0xFu;
        if (m4 == 0xFu) {
            const float sc0 = __shfl(sc_lane, sp + 0, 32);
            const float sc1 = __shfl(sc_lane, sp + 1, 32);
            const float sc2 = __shfl(sc_lane, sp + 2, 32);
            const float sc3 = __shfl(sc_lane, sp + 3, 32);
            const float4 pv0 = *(const float4 *)(tmp_o + (base_idx + sp + 0) * D + lane * 4);
            const float4 pv1 = *(const float4 *)(tmp_o + (base_idx + sp + 1) * D + lane * 4);
            const float4 pv2 = *(const float4 *)(tmp_o + (base_idx + sp + 2) * D + lane * 4);
            const float4 pv3 = *(const float4 *)(tmp_o + (base_idx + sp + 3) * D + lane * 4);
            o0 = fmaf(pv0.x, sc0, o0); o1 = fmaf(pv0.y, sc0, o1); o2 = fmaf(pv0.z, sc0, o2); o3 = fmaf(pv0.w, sc0, o3);
            o0 = fmaf(pv1.x, sc1, o0); o1 = fmaf(pv1.y, sc1, o1); o2 = fmaf(pv1.z, sc1, o2); o3 = fmaf(pv1.w, sc1, o3);
            o0 = fmaf(pv2.x, sc2, o0); o1 = fmaf(pv2.y, sc2, o1); o2 = fmaf(pv2.z, sc2, o2); o3 = fmaf(pv2.w, sc2, o3);
            o0 = fmaf(pv3.x, sc3, o0); o1 = fmaf(pv3.y, sc3, o1); o2 = fmaf(pv3.z, sc3, o2); o3 = fmaf(pv3.w, sc3, o3);
        } else if (m4 != 0u) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                if ((m4 >> k) & 1u) {
                    const float sc = __shfl(sc_lane, sp + k, 32);
                    const float4 pv = *(const float4 *)(tmp_o + (base_idx + sp + k) * D + lane * 4);
                    o0 = fmaf(pv.x, sc, o0); o1 = fmaf(pv.y, sc, o1); o2 = fmaf(pv.z, sc, o2); o3 = fmaf(pv.w, sc, o3);
                }
            }
        }
    }
    for (; sp < n_splits; ++sp) {
        if ((valid_mask >> sp) & 1u) {
            const float sc = __shfl(sc_lane, sp, 32);
            const float4 pv = *(const float4 *)(tmp_o + (base_idx + sp) * D + lane * 4);
            o0 = fmaf(pv.x, sc, o0); o1 = fmaf(pv.y, sc, o1); o2 = fmaf(pv.z, sc, o2); o3 = fmaf(pv.w, sc, o3);
        }
    }

    l_sum += __shfl_xor(l_sum, 16, 32);
    l_sum += __shfl_xor(l_sum,  8, 32);
    l_sum += __shfl_xor(l_sum,  4, 32);
    l_sum += __shfl_xor(l_sum,  2, 32);
    l_sum += __shfl_xor(l_sum,  1, 32);
    const float l_final = l_sum + fast_exp2f(sink_val_log2 - m_final);

    const float inv_l = (1.0f / l_final) * KQ_SCALE;
    float v0 = o0 * inv_l;
    float v1 = o1 * inv_l;
    float v2 = o2 * inv_l;
    float v3 = o3 * inv_l;
    wave32_fa_fwht128_4(v0, v1, v2, v3, lane);

    if (qgate_in != nullptr) {
        const float * g_ptr = qgate_in + (size_t)h * D + (size_t)lane * 4;
        const float4 gv = *(const float4 *)g_ptr;
        v0 *= 1.0f / (1.0f + fast_exp2f(-gv.x * LOG2E));
        v1 *= 1.0f / (1.0f + fast_exp2f(-gv.y * LOG2E));
        v2 *= 1.0f / (1.0f + fast_exp2f(-gv.z * LOG2E));
        v3 *= 1.0f / (1.0f + fast_exp2f(-gv.w * LOG2E));
    }
    *(float4 *)(out + (size_t)h * D + (size_t)lane * 4) = make_float4(v0, v1, v2, v3);
}

// ----------------------------------------------------------------------------
// Prefill sparse Global FA: RDNA3 Wave32 FP16 WMMA, BM = 64, BC = 64 (one page)
// ----------------------------------------------------------------------------

#if NIB_MMA_TILE_TARGET
using halfx16_t = __attribute__((ext_vector_type(16))) _Float16;
using floatx8_t = __attribute__((ext_vector_type(8)))  float;

// Materialize WMMA accumulators in VGPRs at this program point (zero instructions emitted).
// The K*Q^T WMMAs are pure arithmetic, so nothing ties them to the inline-asm s_barrier that
// follows them. AMD clang 22 (ROCm 7.2.x) sinks all of them past that barrier and the whole
// V dequant/transpose phase, keeping every K fragment plus O_acc live across it -> ~200 VGPR
// spills per prefill kernel (issue #6). The volatile asm cannot move across the volatile barrier
// asm, so S must be complete before it. clang 24 already schedules them here: no-op for it.
static __device__ __forceinline__ void pin_wmma_acc(floatx8_t & acc) {
    asm volatile("" : "+v"(acc));
}
template <int N>
static __device__ __forceinline__ void pin_wmma_acc(floatx8_t (&acc)[N]) {
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        pin_wmma_acc(acc[i]);
    }
}

static __device__ __forceinline__ halfx16_t load_halfx16_b128(const _Float16 * ptr) {
    const uint4 v0 = *(const uint4 *)(ptr + 0);
    const uint4 v1 = *(const uint4 *)(ptr + 8);
    halfx16_t res;
    memcpy(&res, &v0, 16);
    memcpy(((char *)&res) + 16, &v1, 16);
    return res;
}

static __device__ __forceinline__ halfx16_t pack_p_frag_rdna3(const floatx8_t & p, const int half_w) {
    const uint32_t perm0 = (half_w == 0) ? 0x05040100u : 0x01000504u;
    const uint32_t perm1 = (half_w == 0) ? 0x07060302u : 0x03020706u;
    union {
        halfx16_t h16;
        uint32_t  u[8];
    } res;
    #pragma unroll
    for (int t = 0; t < 4; ++t) {
        const half2 m = __floats2half2_rn(p[2 * t], p[2 * t + 1]);
        const uint32_t mu = __builtin_bit_cast(uint32_t, m);
        const uint32_t ou = __shfl_xor(mu, 16, 32);
        res.u[2 * t + 0] = __builtin_amdgcn_perm(ou, mu, perm0);
        res.u[2 * t + 1] = __builtin_amdgcn_perm(ou, mu, perm1);
    }
    return res.h16;
}

static __device__ __forceinline__ uint32_t bits_of(__half2 h) {
    return __builtin_bit_cast(uint32_t, h);
}

static __device__ __forceinline__ halfx16_t abs_halfx16(halfx16_t x) {
    union {
        halfx16_t h16;
        uint32_t  u[8];
    } v;
    v.h16 = x;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        v.u[i] &= 0x7FFF7FFFu;
    }
    return v.h16;
}

static __device__ __forceinline__ void dequant_16i8_to_8h2(
        uint4 qv, __half scale_h, uint32_t out_h2[8]) {
    const __half2 s2 = __half2half2(scale_h);
    const uint32_t words[4] = { qv.x, qv.y, qv.z, qv.w };
    #pragma unroll
    for (int w = 0; w < 4; ++w) {
        __half2 h01, h23;
        i8x4_to_h2x2(words[w], h01, h23);
        h01 = __hmul2(h01, s2);
        h23 = __hmul2(h23, s2);
        out_h2[2 * w + 0] = bits_of(h01);
        out_h2[2 * w + 1] = bits_of(h23);
    }
}

static __device__ __forceinline__ void dequant_16q4_1_to_8h2(
        uint2 qv, __half2 dm_h, uint32_t out_h2[8]) {
    const __half2 d2 = __half2half2(__low2half(dm_h));
    const __half2 m2 = __half2half2(__high2half(dm_h));
    const uint32_t words[2] = { qv.x, qv.y };
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        const uint32_t lo =  words[w]       & 0x0F0F0F0Fu;
        const uint32_t hi = (words[w] >> 4) & 0x0F0F0F0Fu;
        __half2 h01, h23, h45, h67;
        u4x4_to_h2x2<false>(lo, h01, h23);
        u4x4_to_h2x2<false>(hi, h45, h67);
        out_h2[4 * w + 0] = bits_of(__hfma2(h01, d2, m2));
        out_h2[4 * w + 1] = bits_of(__hfma2(h23, d2, m2));
        out_h2[4 * w + 2] = bits_of(__hfma2(h45, d2, m2));
        out_h2[4 * w + 3] = bits_of(__hfma2(h67, d2, m2));
    }
}

static __device__ __forceinline__ void dequant_16q4_0_to_8h2(
        uint2 qv, __half scale_h, uint32_t out_h2[8]) {
    const __half2 s2 = __half2half2(scale_h);
    const uint32_t words[2] = { qv.x, qv.y };
    #pragma unroll
    for (int w = 0; w < 2; ++w) {
        const uint32_t lo =  words[w]       & 0x0F0F0F0Fu;
        const uint32_t hi = (words[w] >> 4) & 0x0F0F0F0Fu;
        __half2 h01, h23, h45, h67;
        u4x4_to_h2x2<true>(lo, h01, h23);
        u4x4_to_h2x2<true>(hi, h45, h67);
        out_h2[4 * w + 0] = bits_of(__hmul2(h01, s2));
        out_h2[4 * w + 1] = bits_of(__hmul2(h23, s2));
        out_h2[4 * w + 2] = bits_of(__hmul2(h45, s2));
        out_h2[4 * w + 3] = bits_of(__hmul2(h67, s2));
    }
}

template <int KV_FMT>
static __device__ __forceinline__ void load_dequant_16_kv(
        const int8_t * tile_qs, const __half * tile_d, int rel_j, int l_c16, uint32_t out_h2[8]) {
    const uint32_t d_off = (uint32_t)rel_j * 4 + (uint32_t)(l_c16 >> 1);
    if constexpr (KV_FMT == KV_Q8_0) {
        const uint32_t qs_off = (uint32_t)rel_j * HEAD_DIM + (uint32_t)l_c16 * 16;
        const __half d_h = __ldg(tile_d + d_off);
        const uint4  qv  = *(const uint4 *)(tile_qs + qs_off);
        dequant_16i8_to_8h2(qv, d_h, out_h2);
    } else if constexpr (KV_FMT == KV_Q4_1) {
        const uint32_t qs_off = (uint32_t)rel_j * (HEAD_DIM / 2) + (uint32_t)l_c16 * 8;
        const __half2 dm_h = __ldg(((const __half2 *)tile_d) + d_off);
        const uint2   qv   = *(const uint2 *)(tile_qs + qs_off);
        dequant_16q4_1_to_8h2(qv, dm_h, out_h2);
    } else {
        const uint32_t qs_off = (uint32_t)rel_j * (HEAD_DIM / 2) + (uint32_t)l_c16 * 8;
        const __half d_h = __ldg(tile_d + d_off);
        const uint2  qv  = *(const uint2 *)(tile_qs + qs_off);
        dequant_16q4_0_to_8h2(qv, d_h, out_h2);
    }
}

template <int STRIDE_V, bool SWIZZLE_V>
static __device__ __forceinline__ void warp_transpose_4keys_write_v(
        const uint32_t vh2[8], _Float16 tile_V_T[][STRIDE_V], int kk_base, int lane) {
    const int sub_k = lane & 3;
    const int c16   = lane >> 2;
    const int pair  = sub_k & 1;
    const int hi2   = (sub_k >> 1) & 1;
    const uint32_t perm_sel = (pair == 0) ? 0x05040100u : 0x03020706u;

    int kk_mod7 = 0;
    if constexpr (!SWIZZLE_V) {
        const int d8     = 2 * c16 + hi2;
        const int d8_m7  = (d8 >= 14) ? (d8 - 14) : ((d8 >= 7) ? (d8 - 7) : d8);
        const int chunk8 = kk_base >> 3;
        int slot7 = chunk8 + d8_m7;
        if (slot7 >= 7) slot7 -= 7;
        kk_mod7 = slot7 * 8 + (kk_base & 4);
    }

    #pragma unroll
    for (int q = 0; q < 4; ++q) {
        const uint32_t a  = vh2[q];
        const uint32_t b  = vh2[4 + q];
        const uint32_t pa = __shfl_xor(a, 1, 32);
        const uint32_t pb = __shfl_xor(b, 1, 32);
        const uint32_t ta = __builtin_amdgcn_perm(pa, a, perm_sel);
        const uint32_t tb = __builtin_amdgcn_perm(pb, b, perm_sel);
        const uint32_t send_val = hi2 ? ta : tb;
        const uint32_t recv_val = __shfl_xor(send_val, 2, 32);
        const uint32_t k01 = hi2 ? recv_val : ta;
        const uint32_t k23 = hi2 ? tb : recv_val;
        const int d_write = c16 * 16 + 2 * (hi2 ? (4 + q) : q) + pair;
        const int kk_col  = SWIZZLE_V ? (kk_base ^ (((d_write >> 1) & 1) * 16)) : kk_mod7;
        *reinterpret_cast<uint2 *>(&tile_V_T[d_write][kk_col]) = make_uint2(k01, k23);
    }
}

template <int STRIDE_V, bool SWIZ_V>
static __device__ __forceinline__ halfx16_t load_vt_halfx16(
        const _Float16 tile_V_T[][STRIDE_V], int c_blk, int nk, int l_qlocal) {
    const int d_row = c_blk * 16 + l_qlocal;
    if constexpr (SWIZ_V) {
        const int col = (nk ^ ((d_row >> 1) & 1)) * 16;
        return load_halfx16_b128(&tile_V_T[d_row][col]);
    } else {
        const int half_hi = (l_qlocal >> 3) & 1;
        const int c0_0 = ((2 * nk + 2 * c_blk + 0) % 7) * 8;
        const int c0_1 = ((2 * nk + 2 * c_blk + 1) % 7) * 8;
        const int c1_0 = ((2 * nk + 1 + 2 * c_blk + 0) % 7) * 8;
        const int c1_1 = ((2 * nk + 1 + 2 * c_blk + 1) % 7) * 8;
        const int col0 = half_hi ? c0_1 : c0_0;
        const int col1 = half_hi ? c1_1 : c1_0;
        const uint4 v0 = *(const uint4 *)(&tile_V_T[d_row][col0]);
        const uint4 v1 = *(const uint4 *)(&tile_V_T[d_row][col1]);
        halfx16_t res;
        memcpy(&res, &v0, 16);
        memcpy(((char *)&res) + 16, &v1, 16);
        return res;
    }
}
#endif // gfx1100..gfx1103

// Exact per-token Quest prefill page scoring via RDNA3 FP16 WMMA:
//   S(t, g, p) = sum_d max(q_{t,g,d}*k_min_{p,d}, q_{t,g,d}*k_max_{p,d})
//              = <q_{t,g}, k_mid_p> + <|q_{t,g}|, k_rad_p>
// where k_mid = 0.5*(k_max + k_min), k_rad = 0.5*(k_max - k_min) >= 0.
// Pass 1 computes each token's causal peak M(t, g) = max_{p <= q_page(t)} S(t, g, p).
// Pass 2 scores candidate historical pages p in [cand_begin, cand_end) by
//   page_scores[qb][h_kv][p] = max_{t in qb, g in [0, 3)} (S(t, g, p) - M(t, g)),
// placing all 64 tokens and 3 GQA heads on the exact relative log-attention scale.
template <int META_FMT>
__launch_bounds__(384)
static __global__ void sparse_score_pages_prefill_wmma_kernel(
        const float * __restrict__ q,            // [n_tokens, N_HEAD, HEAD_DIM]
        SparseGlobalPageMeta meta,
        int pos_start,
        int n_tokens,
        int sparse_pages,
        float * __restrict__ page_scores) {      // [qb][N_HEAD_KV][max_pages]
#if NIB_MMA_TILE_TARGET
    constexpr int NWARPS = 12;                   // 4 Q-tiles (16 tok each) x 3 GQA heads
    constexpr int STRIDE = 136;                  // 128 + 8 halfs -> bank-conflict-free & 16B aligned
    constexpr float Q_NORM = KQ_SCALE * KQ_SCALE; // inverse FWHT scale * 1/sqrt(128) logit scale

    const int h_kv = (int)blockIdx.x;
    const int qb   = (int)blockIdx.y;

    const int n_pages = (pos_start + n_tokens + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
    if (n_pages <= sparse_pages) {
        return;
    }

    const int q_pos_start = pos_start + qb * SPARSE_PAGE_SIZE;
    const int q_pos_end   = pos_start + min((qb + 1) * SPARSE_PAGE_SIZE, n_tokens) - 1;
    const int local_start = sparse_local_start_page(q_pos_start);
    const int cand_begin  = SPARSE_SINK_PAGES;
    const int cand_end    = min(local_start, n_pages);
    if (cand_end <= cand_begin) {
        return;
    }
    // Pages >= meta.max_pages (SPARSE_MAX_PAGES = 4096, i.e. positions >= 262,144
    // at -c 262848) have no metadata: they are always inside the mandatory recent
    // window, launch_kv_update_page_minmax_* never writes them, and reading them
    // here walked past the end of the per-head metadata (the next head's pages or,
    // for the last KV head, out of the allocation: GPU memory fault).
    const int pass1_end = min(min(q_pos_end / SPARSE_PAGE_SIZE + 1, n_pages), meta.max_pages);

    const int lane    = (int)threadIdx.x;        // 0..31
    const int warp    = (int)threadIdx.y;        // 0..11
    const int qt      = warp / GQA_RATIO;        // 0..3
    const int g       = warp % GQA_RATIO;        // 0..2
    const int h_q     = h_kv * GQA_RATIO + g;
    const int q_local = lane & 15;               // 0..15
    const int half_w  = lane >> 4;               // 0..1

    struct PrefillScoreSmem {
        alignas(16) _Float16 Q[NWARPS][16][STRIDE];
        alignas(16) _Float16 K_mid[16][STRIDE];
        alignas(16) _Float16 K_rad[16][STRIDE];
        float tile_max[NWARPS][16];
    };
    __shared__ PrefillScoreSmem smem;

    // Load and unrotate Q for this Q-block into native NoPE basis (FP16 in LDS).
    const int qb_t0 = qb * SPARSE_PAGE_SIZE + qt * 16;
    for (int r = 0; r < 16; ++r) {
        const int t = qb_t0 + r;
        if (t < n_tokens) {
            float4 v = ((const float4 *)(q + ((size_t)t * N_HEAD + (size_t)h_q) * HEAD_DIM))[lane];
            wave32_fa_fwht128_4(v.x, v.y, v.z, v.w, lane);
            const __half2 h01 = __floats2half2_rn(v.x * Q_NORM, v.y * Q_NORM);
            const __half2 h23 = __floats2half2_rn(v.z * Q_NORM, v.w * Q_NORM);
            *reinterpret_cast<uint2 *>(&smem.Q[warp][r][lane * 4]) =
                make_uint2(bits_of(h01), bits_of(h23));
        } else {
            *reinterpret_cast<uint2 *>(&smem.Q[warp][r][lane * 4]) = make_uint2(0u, 0u);
        }
    }

    const int t_owned  = qb_t0 + q_local;
    const bool q_valid = (t_owned < n_tokens);
    const int q_page   = (pos_start + t_owned) / SPARSE_PAGE_SIZE;
    const size_t head_base  = (size_t)h_kv * (size_t)meta.max_pages;
    const size_t score_base = ((size_t)qb * N_HEAD_KV + (size_t)h_kv) * (size_t)meta.max_pages;

    auto load_page_tile = [&](int p0, int p_limit) {
        for (int pi = warp; pi < 16; pi += NWARPS) {
            const int p = p0 + pi;
            if (p < p_limit) {
                const size_t off = (head_base + (size_t)p) * HEAD_DIM + (size_t)lane * 4;
                uint32_t umin, umax;
                if constexpr (META_FMT == SPARSE_META_Q4) {
                    umin = __builtin_nontemporal_load((const uint32_t *)(meta.k_qmm + off));
                    umax = 0u;
                } else {
                    umin = __builtin_nontemporal_load((const uint32_t *)(meta.k_min_q + off));
                    umax = __builtin_nontemporal_load((const uint32_t *)(meta.k_max_q + off));
                }
                const float d = __half2float(*(const __half *)(meta.k_mm_d + head_base + (size_t)p));
                float lo[4], hi[4];
                sparse_meta_4<META_FMT>(umin, umax, d, lo, hi);
                const __half2 m01 = __floats2half2_rn(0.5f * (hi[0] + lo[0]), 0.5f * (hi[1] + lo[1]));
                const __half2 m23 = __floats2half2_rn(0.5f * (hi[2] + lo[2]), 0.5f * (hi[3] + lo[3]));
                constexpr float R_SCALE = 0.5f * SPARSE_QUEST_ALPHA;
                const __half2 r01 = __floats2half2_rn(R_SCALE * (hi[0] - lo[0]), R_SCALE * (hi[1] - lo[1]));
                const __half2 r23 = __floats2half2_rn(R_SCALE * (hi[2] - lo[2]), R_SCALE * (hi[3] - lo[3]));
                *reinterpret_cast<uint2 *>(&smem.K_mid[pi][lane * 4]) =
                    make_uint2(bits_of(m01), bits_of(m23));
                *reinterpret_cast<uint2 *>(&smem.K_rad[pi][lane * 4]) =
                    make_uint2(bits_of(r01), bits_of(r23));
            } else {
                *reinterpret_cast<uint2 *>(&smem.K_mid[pi][lane * 4]) = make_uint2(0u, 0u);
                *reinterpret_cast<uint2 *>(&smem.K_rad[pi][lane * 4]) = make_uint2(0u, 0u);
            }
        }
    };

    // always_inline: clang 22 (ROCm 7.2.x) otherwise outlines this twice-used lambda into a real
    // call (s_swappc + 16 B private stack frame for the captures, issue #6).
    auto score_tile_wmma = [&]() __attribute__((always_inline)) -> floatx8_t {
        floatx8_t S = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };
#if NIB_MMA_NATIVE
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int d_off = c * 16;
            const halfx16_t Qc  = load_halfx16_b128(&smem.Q[warp][q_local][d_off]);
            const halfx16_t Km  = load_halfx16_b128(&smem.K_mid[q_local][d_off]);
            S = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(Km, Qc, S);
            const halfx16_t Qca = abs_halfx16(Qc);
            const halfx16_t Kr  = load_halfx16_b128(&smem.K_rad[q_local][d_off]);
            S = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(Kr, Qca, S);
        }
#else
        // gfx12 (include/nib_mma.h): A = K rows (key), B = this lane's Q row (q = lane&15).
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int d_off = c * 16;
            const halfx16_t Qc  = load_halfx16_b128(&smem.Q[warp][q_local][d_off]);
            nib_mma::mma_stride(&smem.K_mid[0][d_off], STRIDE, Qc, S, lane);
            const halfx16_t Qca = abs_halfx16(Qc);
            nib_mma::mma_stride(&smem.K_rad[0][d_off], STRIDE, Qca, S, lane);
        }
#endif
        // Keep the WMMAs here: clang 22 otherwise sinks them into the caller's `if (q_valid)`
        // block, keeping all 16 K/Q fragments (~190 VGPRs) live at once.
        pin_wmma_acc(S);
        return S;
    };

    // Pass 1: per-token, per-head causal maximum Quest score across [0, pass1_end).
    float rmax = -INFINITY;
    for (int p0 = 0; p0 < pass1_end; p0 += 16) {
        load_page_tile(p0, pass1_end);
        __syncthreads();

        const floatx8_t S = score_tile_wmma();
        if (q_valid) {
            #pragma unroll
            for (int idx = 0; idx < 8; ++idx) {
                const int p = p0 + 2 * idx + half_w;
                if (p < pass1_end && p <= q_page) {
                    rmax = fmaxf(rmax, S[idx]);
                }
            }
        }
        __syncthreads();
    }
    rmax = fmaxf(rmax, __shfl_xor(rmax, 16, 32));

    // Pass 2: relative Quest score max_{t, g} (S(t, g, p) - rmax(t, g)) on [cand_begin, cand_end).
    for (int p0 = cand_begin; p0 < cand_end; p0 += 16) {
        load_page_tile(p0, cand_end);
        __syncthreads();

        const floatx8_t S = score_tile_wmma();
        #pragma unroll
        for (int idx = 0; idx < 8; ++idx) {
            float v = q_valid ? (S[idx] - rmax) : -INFINITY;
            v = fmaxf(v, __shfl_xor(v, 8, 32));
            v = fmaxf(v, __shfl_xor(v, 4, 32));
            v = fmaxf(v, __shfl_xor(v, 2, 32));
            v = fmaxf(v, __shfl_xor(v, 1, 32));
            if (q_local == 0) {
                smem.tile_max[warp][2 * idx + half_w] = v;
            }
        }
        __syncthreads();

        if (warp == 0 && lane < 16) {
            const int p = p0 + lane;
            if (p < cand_end) {
                float best = smem.tile_max[0][lane];
                #pragma unroll
                for (int w = 1; w < NWARPS; ++w) {
                    best = fmaxf(best, smem.tile_max[w][lane]);
                }
                page_scores[score_base + (size_t)p] = best;
            }
        }
        __syncthreads();
    }
#else
    (void)q; (void)meta; (void)pos_start; (void)n_tokens; (void)sparse_pages; (void)page_scores;
#endif
}

template <int NWARPS, int BC, int KV_FMT, bool USE_R_PREP>
__launch_bounds__(NWARPS * 32) __attribute__((amdgpu_waves_per_eu(5, 12)))
static __global__ void inkling_fa_prefill_gqa_wmma_sparse_global_kernel(
        const float * __restrict__ q,
        KVCacheSlot                kv_slot,
        const float * __restrict__ sinks_in,
        int                        sinks_stride,
        const float * __restrict__ rel_table,
        const float * __restrict__ r_prep,
        const float * __restrict__ attn_rel_bias,
        const float * __restrict__ qgate_in,
        int                        qgate_stride,
        float       * __restrict__ out,
        const int32_t * __restrict__ selected_pages,
        const int32_t * __restrict__ n_selected,
        int                        n_tokens,
        int                        pos_start,
        int                        n_kv) {
#if NIB_MMA_TILE_TARGET
    static_assert(NWARPS % GQA_RATIO == 0, "NWARPS must be a multiple of GQA_RATIO=3");
    static_assert(BC == SPARSE_PAGE_SIZE, "BC must equal SPARSE_PAGE_SIZE");

    constexpr int D        = HEAD_DIM;
    constexpr int NQ_TILES = NWARPS / GQA_RATIO;
    constexpr int Br       = NQ_TILES * 16;
    constexpr int NK16     = BC / 16;
    constexpr int STRIDE_K = 136;
    constexpr int STRIDE_V = BC;
    constexpr bool SWIZ_V  = true;
    constexpr float Q_SCALE_LOG2 = KQ_SCALE * LOG2E;

    // Heaviest tile first: deepest Q-blocks leave the launch queue first.
    const int qb      = ((int)gridDim.x - 1) - (int)blockIdx.x;
    const int h_kv    = (int)blockIdx.y;

    const int lane    = (int)threadIdx.x;
    const int warp    = __builtin_amdgcn_readfirstlane((int)threadIdx.y);
    const int qt      = warp / GQA_RATIO;
    const int g       = warp % GQA_RATIO;

    const int q_local = lane & 15;
    const int half_w  = lane >> 4;

    const int q_block       = qb * Br;
    const int32_t q_pos_min = pos_start + q_block;
    const int32_t q_pos_lim = pos_start + n_tokens;
    const int32_t qp_val    = pos_start + q_block + qt * 16 + q_local;
    const bool    q_val     = (qp_val < q_pos_lim);

    struct KVShared {
        alignas(16) _Float16 Q_lo[NWARPS][16][32];
        union {
            alignas(16) _Float16 K[BC][STRIDE_K];
            alignas(16) _Float16 V_T[D][STRIDE_V];
        };
    };
    __shared__ KVShared smem;

    // Q dims 0..31 live in bank-conflict-free LDS for the whole tile loop.
    // Dims 32..127 stay in global memory (L1-hot) and are converted on the fly,
    // which keeps 24 VGPRs free versus holding them in registers.
    const int qi_init = qp_val - pos_start;
    const int qi_safe = max(0, min(qi_init, n_tokens - 1));
    const int h_init  = h_kv * GQA_RATIO + g;
    const float * q_ptr = q + ((size_t)qi_safe * N_HEAD + (size_t)h_init) * D;
    if (q_val) {
        const int c_lds = half_w;
        uint32_t qh2[8];
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float4 fv = *(const float4 *)(q_ptr + c_lds * 16 + i * 4);
            const __half2 h01 = __floats2half2_rn(fv.x * Q_SCALE_LOG2, fv.y * Q_SCALE_LOG2);
            const __half2 h23 = __floats2half2_rn(fv.z * Q_SCALE_LOG2, fv.w * Q_SCALE_LOG2);
            qh2[2 * i + 0] = bits_of(h01);
            qh2[2 * i + 1] = bits_of(h23);
        }
        const int col_swiz = (c_lds ^ ((q_local >> 1) & 1)) * 16;
        *reinterpret_cast<uint4 *>(&smem.Q_lo[warp][q_local][col_swiz + 0]) =
            make_uint4(qh2[0], qh2[1], qh2[2], qh2[3]);
        *reinterpret_cast<uint4 *>(&smem.Q_lo[warp][q_local][col_swiz + 8]) =
            make_uint4(qh2[4], qh2[5], qh2[6], qh2[7]);
    } else {
        const int col_swiz = (half_w ^ ((q_local >> 1) & 1)) * 16;
        *reinterpret_cast<uint4 *>(&smem.Q_lo[warp][q_local][col_swiz + 0]) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4 *>(&smem.Q_lo[warp][q_local][col_swiz + 8]) = make_uint4(0, 0, 0, 0);
    }

    floatx8_t O_acc[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        O_acc[c] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    float m_q = -INFINITY;
    float l_q = 0.0f;

    constexpr int N_CHUNKS = BC / 4;
    const int32_t * sel = selected_pages + ((size_t)qb * N_HEAD_KV + (size_t)h_kv) * SPARSE_MAX_SELECTED;
    const int n_sel = n_selected[(size_t)qb * N_HEAD_KV + (size_t)h_kv];

    const uint64_t kv_head_base = (uint64_t)h_kv * (uint64_t)(uint32_t)kv_slot.cap;
    const int h_rel = h_kv * GQA_RATIO + g;

    for (int pi = 0; pi < n_sel; ++pi) {
        const int k0 = (int)sel[pi] * SPARSE_PAGE_SIZE;
        if (k0 >= n_kv) {
            continue;
        }

        int l_tmp = lane;
        asm volatile("" : "+v"(l_tmp));
        const int l_qlocal = l_tmp & 15;
        const int l_halfw  = l_tmp >> 4;

        const bool is_fast_interior = (k0 + BC <= n_kv) && (k0 + BC <= q_pos_min);
        const int32_t tile_min_kp = k0;
        const int32_t tile_max_kp = (int32_t)min(k0 + BC - 1, n_kv - 1);

        uint64_t tile_slot_base = kv_head_base + (uint64_t)(uint32_t)k0;
        asm volatile("" : "+s"(tile_slot_base));

        constexpr uint32_t QS_STRIDE = (KV_FMT == KV_Q8_0) ? (uint32_t)D : (uint32_t)(D / 2);
        constexpr uint32_t D_STRIDE  = (KV_FMT == KV_Q4_1) ? 8u : 4u;
        const int8_t * tile_v_qs = kv_slot.v_qs + tile_slot_base * QS_STRIDE;
        const __half * tile_v_d  = kv_slot.v_d  + tile_slot_base * D_STRIDE;
        {
            const int8_t * tile_k_qs = kv_slot.k_qs + tile_slot_base * QS_STRIDE;
            const __half * tile_k_d  = kv_slot.k_d  + tile_slot_base * D_STRIDE;
            const int l_subk = l_tmp & 3;
            const int l_c16  = l_tmp >> 2;
            if (k0 + BC <= n_kv) {
                for (int chunk4 = warp; chunk4 < N_CHUNKS; chunk4 += NWARPS) {
                    const int rel_j = chunk4 * 4 + l_subk;
                    uint32_t kh2[8];
                    load_dequant_16_kv<KV_FMT>(tile_k_qs, tile_k_d, rel_j, l_c16, kh2);
                    *reinterpret_cast<uint4 *>(&smem.K[rel_j][l_c16 * 16 + 0]) =
                        make_uint4(kh2[0], kh2[1], kh2[2], kh2[3]);
                    *reinterpret_cast<uint4 *>(&smem.K[rel_j][l_c16 * 16 + 8]) =
                        make_uint4(kh2[4], kh2[5], kh2[6], kh2[7]);
                }
            } else {
                for (int chunk4 = warp; chunk4 < N_CHUNKS; chunk4 += NWARPS) {
                    const int rel_j = chunk4 * 4 + l_subk;
                    uint32_t kh2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    if (k0 + rel_j < n_kv) {
                        load_dequant_16_kv<KV_FMT>(tile_k_qs, tile_k_d, rel_j, l_c16, kh2);
                    }
                    *reinterpret_cast<uint4 *>(&smem.K[rel_j][l_c16 * 16 + 0]) =
                        make_uint4(kh2[0], kh2[1], kh2[2], kh2[3]);
                    *reinterpret_cast<uint4 *>(&smem.K[rel_j][l_c16 * 16 + 8]) =
                        make_uint4(kh2[4], kh2[5], kh2[6], kh2[7]);
                }
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");

        floatx8_t S[NK16];
        #pragma unroll
        for (int nk = 0; nk < NK16; ++nk) {
            S[nk] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }

        halfx16_t kv_buf[2];
#if NIB_MMA_NATIVE
        kv_buf[0] = load_halfx16_b128(&smem.K[l_qlocal][0]);
#endif

        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int d_off = c * 16;
            halfx16_t Qc;
            if (c < 2) {
                Qc = load_halfx16_b128(&smem.Q_lo[warp][l_qlocal][(c ^ ((l_qlocal >> 1) & 1)) * 16]);
            } else {
                uint32_t qh[8];
                const float4 f0 = *(const float4 *)(q_ptr + c * 16 + 0);
                const float4 f1 = *(const float4 *)(q_ptr + c * 16 + 4);
                const float4 f2 = *(const float4 *)(q_ptr + c * 16 + 8);
                const float4 f3 = *(const float4 *)(q_ptr + c * 16 + 12);
                const __half2 h01 = __floats2half2_rn(f0.x * Q_SCALE_LOG2, f0.y * Q_SCALE_LOG2);
                const __half2 h23 = __floats2half2_rn(f0.z * Q_SCALE_LOG2, f0.w * Q_SCALE_LOG2);
                const __half2 h45 = __floats2half2_rn(f1.x * Q_SCALE_LOG2, f1.y * Q_SCALE_LOG2);
                const __half2 h67 = __floats2half2_rn(f1.z * Q_SCALE_LOG2, f1.w * Q_SCALE_LOG2);
                const __half2 h89 = __floats2half2_rn(f2.x * Q_SCALE_LOG2, f2.y * Q_SCALE_LOG2);
                const __half2 hab = __floats2half2_rn(f2.z * Q_SCALE_LOG2, f2.w * Q_SCALE_LOG2);
                const __half2 hcd = __floats2half2_rn(f3.x * Q_SCALE_LOG2, f3.y * Q_SCALE_LOG2);
                const __half2 hef = __floats2half2_rn(f3.z * Q_SCALE_LOG2, f3.w * Q_SCALE_LOG2);
                qh[0] = bits_of(h01); qh[1] = bits_of(h23);
                qh[2] = bits_of(h45); qh[3] = bits_of(h67);
                qh[4] = bits_of(h89); qh[5] = bits_of(hab);
                qh[6] = bits_of(hcd); qh[7] = bits_of(hef);
                union {
                    halfx16_t h16;
                    uint32_t  u[8];
                } qc;
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    qc.u[w] = qh[w];
                }
                Qc = qc.h16;
            }
#if NIB_MMA_NATIVE
            #pragma unroll
            for (int nk = 0; nk < NK16; ++nk) {
                const int cur_b = (c * NK16 + nk) & 1;
                const int nxt_b = cur_b ^ 1;
                if (nk + 1 < NK16) {
                    kv_buf[nxt_b] = load_halfx16_b128(&smem.K[(nk + 1) * 16 + l_qlocal][d_off]);
                } else if (c + 1 < 8) {
                    kv_buf[nxt_b] = load_halfx16_b128(&smem.K[l_qlocal][(c + 1) * 16]);
                }
                S[nk] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kv_buf[cur_b], Qc, S[nk]);
                __builtin_amdgcn_sched_barrier(0);
            }
#else
            // gfx12 (include/nib_mma.h): same tile from the same LDS K tile.
            #pragma unroll
            for (int nk = 0; nk < NK16; ++nk) {
                nib_mma::mma_stride(&smem.K[nk * 16][d_off], STRIDE_K, Qc, S[nk], lane);
            }
#endif
        }
        pin_wmma_acc(S);
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n\ts_waitcnt vmcnt(63)" ::: "memory");

        {
            const int l_subk = l_tmp & 3;
            const int l_c16  = l_tmp >> 2;
            if (k0 + BC <= n_kv) {
                for (int chunk4 = warp; chunk4 < N_CHUNKS; chunk4 += NWARPS) {
                    const int kk_base = chunk4 * 4;
                    const int rel_j = kk_base + l_subk;
                    uint32_t vh2[8];
                    load_dequant_16_kv<KV_FMT>(tile_v_qs, tile_v_d, rel_j, l_c16, vh2);
                    warp_transpose_4keys_write_v<STRIDE_V, SWIZ_V>(vh2, smem.V_T, kk_base, l_tmp);
                }
            } else {
                for (int chunk4 = warp; chunk4 < N_CHUNKS; chunk4 += NWARPS) {
                    const int kk_base = chunk4 * 4;
                    const int rel_j = kk_base + l_subk;
                    uint32_t vh2[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                    if (k0 + rel_j < n_kv) {
                        load_dequant_16_kv<KV_FMT>(tile_v_qs, tile_v_d, rel_j, l_c16, vh2);
                    }
                    warp_transpose_4keys_write_v<STRIDE_V, SWIZ_V>(vh2, smem.V_T, kk_base, l_tmp);
                }
            }
        }
        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");

        float m_tile = m_q;
        const bool q_in = (qp_val < q_pos_lim);
        if (is_fast_interior) {
            const bool far_no_rel = ((q_pos_min - (k0 + BC - 1)) >= N_REL_EXTENT);
            if (!far_no_rel) {
                const bool lane_has_rel = q_in && ((qp_val - (k0 + BC - 1)) < N_REL_EXTENT);
                if (__any(lane_has_rel) && q_in) {
                    int32_t qi_rel = qp_val - pos_start;
                    asm volatile("" : "+v"(qi_rel));
                    const float * rel_row = USE_R_PREP ? nullptr
                        : rel_table + ((int64_t)qi_rel * N_HEAD + h_rel) * N_REL_EXTENT;
                    const float * rp_row = USE_R_PREP
                        ? r_prep + ((size_t)qi_rel * N_HEAD + (size_t)h_rel) * N_REL : nullptr;
                    #pragma unroll
                    for (int nk = 0; nk < NK16; ++nk) {
                        #pragma unroll
                        for (int idx = 0; idx < 8; ++idx) {
                            const int32_t dist = qp_val - (k0 + nk * 16 + 2 * idx + l_halfw);
                            if (dist < N_REL_EXTENT) {
                                S[nk][idx] += rel_logit<USE_R_PREP>(rel_row, rp_row, attn_rel_bias,
                                                                    dist) * LOG2E;
                            }
                        }
                    }
                }
            }
            if (q_in) {
                #pragma unroll
                for (int nk = 0; nk < NK16; ++nk) {
                    #pragma unroll
                    for (int idx = 0; idx < 8; ++idx) {
                        m_tile = fmaxf(m_tile, S[nk][idx]);
                    }
                }
            }
        } else {
            uint32_t valid_bits[NK16];
            #pragma unroll
            for (int nk = 0; nk < NK16; ++nk) {
                valid_bits[nk] = 0u;
            }
            if (q_in) {
                #pragma unroll
                for (int nk = 0; nk < NK16; ++nk) {
                    #pragma unroll
                    for (int idx = 0; idx < 8; ++idx) {
                        const int32_t kp = k0 + nk * 16 + 2 * idx + l_halfw;
                        if (kp < n_kv && kp <= qp_val) {
                            valid_bits[nk] |= (1u << idx);
                        }
                    }
                }
            }

            const bool lane_has_rel = q_in &&
                                      (qp_val >= tile_min_kp) &&
                                      ((qp_val - tile_max_kp) < N_REL_EXTENT);
            if (__any(lane_has_rel)) {
                int32_t qi_rel = qp_val - pos_start;
                asm volatile("" : "+v"(qi_rel));
                const float * rel_row = USE_R_PREP ? nullptr
                    : rel_table + ((int64_t)qi_rel * N_HEAD + h_rel) * N_REL_EXTENT;
                const float * rp_row = USE_R_PREP
                    ? r_prep + ((size_t)qi_rel * N_HEAD + (size_t)h_rel) * N_REL : nullptr;
                #pragma unroll
                for (int nk = 0; nk < NK16; ++nk) {
                    #pragma unroll
                    for (int idx = 0; idx < 8; ++idx) {
                        if ((valid_bits[nk] >> idx) & 1u) {
                            const int32_t kp = k0 + nk * 16 + 2 * idx + l_halfw;
                            const int32_t dist = qp_val - kp;
                            if (dist >= 0 && dist < N_REL_EXTENT) {
                                S[nk][idx] += rel_logit<USE_R_PREP>(rel_row, rp_row, attn_rel_bias,
                                                                    dist) * LOG2E;
                            }
                            m_tile = fmaxf(m_tile, S[nk][idx]);
                        } else {
                            S[nk][idx] = -INFINITY;
                        }
                    }
                }
            } else {
                #pragma unroll
                for (int nk = 0; nk < NK16; ++nk) {
                    #pragma unroll
                    for (int idx = 0; idx < 8; ++idx) {
                        if ((valid_bits[nk] >> idx) & 1u) {
                            m_tile = fmaxf(m_tile, S[nk][idx]);
                        } else {
                            S[nk][idx] = -INFINITY;
                        }
                    }
                }
            }
        }

        m_tile = fmaxf(m_tile, __shfl_xor(m_tile, 16, 32));

        if (q_in && m_tile > m_q) {
            if (m_q > -INFINITY) {
                const float scale_old = fast_exp2f(m_q - m_tile);
                l_q *= scale_old;
                #pragma unroll
                for (int c = 0; c < 8; ++c) {
                    #pragma unroll
                    for (int idx = 0; idx < 8; ++idx) {
                        O_acc[c][idx] *= scale_old;
                    }
                }
            }
            m_q = m_tile;
        }

        float l_add = 0.0f;
        if (q_in && m_tile > -INFINITY) {
            #pragma unroll
            for (int nk = 0; nk < NK16; ++nk) {
                #pragma unroll
                for (int idx = 0; idx < 8; ++idx) {
                    const float p = fast_exp2f(S[nk][idx] - m_tile);
                    S[nk][idx] = p;
                    l_add += p;
                }
            }
        } else {
            #pragma unroll
            for (int nk = 0; nk < NK16; ++nk) {
                #pragma unroll
                for (int idx = 0; idx < 8; ++idx) {
                    S[nk][idx] = 0.0f;
                }
            }
        }
        l_add += __shfl_xor(l_add, 16, 32);
        l_q += l_add;

        kv_buf[0] = load_vt_halfx16<STRIDE_V, SWIZ_V>(smem.V_T, 0, 0, l_qlocal);
#if NIB_MMA_NATIVE
        #pragma unroll
        for (int nk = 0; nk < NK16; ++nk) {
            const halfx16_t Pnk = pack_p_frag_rdna3(S[nk], l_halfw);
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                const int cur_b = (nk * 8 + c) & 1;
                const int nxt_b = cur_b ^ 1;
                if (c + 1 < 8) {
                    kv_buf[nxt_b] = load_vt_halfx16<STRIDE_V, SWIZ_V>(smem.V_T, c + 1, nk, l_qlocal);
                } else if (nk + 1 < NK16) {
                    kv_buf[nxt_b] = load_vt_halfx16<STRIDE_V, SWIZ_V>(smem.V_T, 0, nk + 1, l_qlocal);
                }
                O_acc[c] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(kv_buf[cur_b], Pnk, O_acc[c]);
                __builtin_amdgcn_sched_barrier(0);
            }
        }
#else
        // gfx12 (include/nib_mma.h): B row = P[q = lane&15][key 0..15] ascending; the odd/even key
        // parities live in the two halves of the wave (S[nk][idx] is key 2*idx + l_halfw).
        #pragma unroll
        for (int nk = 0; nk < NK16; ++nk) {
            halfx16_t Prow;
            {
                union { halfx16_t h16; uint32_t u[8]; } pk;
                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const float own = S[nk][t];
                    const float oth = __shfl_xor(own, 16, 32);
                    pk.u[t] = bits_of((l_halfw == 0) ? __floats2half2_rn(own, oth)
                                                     : __floats2half2_rn(oth, own));
                }
                Prow = pk.h16;
            }
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                nib_mma::mma_rows(
                    [&](int m) { return load_vt_halfx16<STRIDE_V, SWIZ_V>(smem.V_T, c, nk, m); },
                    Prow, O_acc[c], lane);
            }
        }
#endif

        asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
    }

    if (qp_val >= q_pos_lim) {
        return;
    }
    const int qi = qp_val - pos_start;
    const int h  = h_kv * GQA_RATIO + g;

    const float sink_val_log2 = sinks_in[(size_t)qi * sinks_stride + h] * LOG2E;
    const float m_final  = fmaxf(m_q, sink_val_log2);
    const float scale_o  = (m_q > -INFINITY) ? fast_exp2f(m_q - m_final) : 0.0f;
    const float l_final  = l_q * scale_o + fast_exp2f(sink_val_log2 - m_final);
    const float norm     = (scale_o / l_final) * KQ_SCALE;

    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int idx = 0; idx < 8; ++idx) {
            O_acc[c][idx] *= norm;
        }
    }

    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int idx = 0; idx < 8; ++idx) {
            const float cur   = O_acc[c][idx];
            const float other = __shfl_xor(cur, 16, 32);
            O_acc[c][idx] = (half_w == 0) ? (cur + other) : (other - cur);
        }
    }
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const float a = O_acc[c][i], b = O_acc[c][i + 1];
            O_acc[c][i] = a + b; O_acc[c][i + 1] = a - b;
        }
    }
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int i = 0; i < 8; i += 4) {
            #pragma unroll
            for (int k = 0; k < 2; ++k) {
                const float a = O_acc[c][i + k], b = O_acc[c][i + k + 2];
                O_acc[c][i + k] = a + b; O_acc[c][i + k + 2] = a - b;
            }
        }
    }
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float a = O_acc[c][k], b = O_acc[c][k + 4];
            O_acc[c][k] = a + b; O_acc[c][k + 4] = a - b;
        }
    }
    #pragma unroll
    for (int c = 0; c < 8; c += 2) {
        #pragma unroll
        for (int idx = 0; idx < 8; ++idx) {
            const float a = O_acc[c][idx], b = O_acc[c + 1][idx];
            O_acc[c][idx] = a + b; O_acc[c + 1][idx] = a - b;
        }
    }
    #pragma unroll
    for (int c = 0; c < 8; c += 4) {
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            #pragma unroll
            for (int idx = 0; idx < 8; ++idx) {
                const float a = O_acc[c + k][idx], b = O_acc[c + k + 2][idx];
                O_acc[c + k][idx] = a + b; O_acc[c + k + 2][idx] = a - b;
            }
        }
    }
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        #pragma unroll
        for (int idx = 0; idx < 8; ++idx) {
            const float a = O_acc[k][idx], b = O_acc[k + 4][idx];
            O_acc[k][idx] = a + b; O_acc[k + 4][idx] = a - b;
        }
    }

    float * dst_ptr = out + ((size_t)qi * N_HEAD + (size_t)h) * D;
    if (qgate_in != nullptr) {
        const float * gate_ptr = qgate_in + (size_t)qi * qgate_stride + (size_t)h * D;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            #pragma unroll
            for (int idx = 0; idx < 8; ++idx) {
                const int d = c * 16 + 2 * idx + half_w;
                const float g_val = gate_ptr[d];
                dst_ptr[d] = O_acc[c][idx] * (1.0f / (1.0f + fast_exp2f(-g_val * LOG2E)));
            }
        }
    } else {
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            #pragma unroll
            for (int idx = 0; idx < 8; ++idx) {
                dst_ptr[c * 16 + 2 * idx + half_w] = O_acc[c][idx];
            }
        }
    }
    (void)NK16;
#else
    (void)q; (void)kv_slot; (void)sinks_in; (void)sinks_stride; (void)rel_table;
    (void)r_prep; (void)attn_rel_bias;
    (void)qgate_in; (void)qgate_stride; (void)out;
    (void)selected_pages; (void)n_selected;
    (void)n_tokens; (void)pos_start; (void)n_kv;
#endif // gfx1100..gfx1103
}

} // anonymous namespace

// ============================================================================
// Host launchers
// ============================================================================

static inline void sparse_hip_free(void * p) {
    if (p != nullptr) {
        const hipError_t err = hipFree(p);
        (void)err;
    }
}

// hipMalloc + zero fill: hipMalloc'd VRAM is not guaranteed to be clean, and a
// page that has never been written must score/select as a finite value (issue #1).
// Stream-ordered on the engine stream (never the legacy null stream).
static inline hipError_t sparse_hip_malloc_zero(void ** p, size_t bytes, hipStream_t stream) {
    hipError_t err = hipMalloc(p, bytes);
    if (err != hipSuccess) {
        return err;
    }
    err = hipMemsetAsync(*p, 0, bytes, stream);
    if (err != hipSuccess) {
        (void)hipFree(*p);
        *p = nullptr;
    }
    return err;
}

bool sparse_global_page_meta_alloc(SparseGlobalPageMeta & m, int max_pages, int fmt, hipStream_t stream) {
    if (max_pages <= 0) {
        return false;
    }
    const size_t rows = (size_t)N_HEAD_KV * (size_t)max_pages * (size_t)HEAD_DIM;
    void * p = nullptr;

    if (fmt == SPARSE_META_Q4) {
        if (sparse_hip_malloc_zero(&p, rows, stream) != hipSuccess) {
            return false;
        }
        m.k_qmm = (uint8_t *)p;
    } else {
        if (sparse_hip_malloc_zero(&p, rows, stream) != hipSuccess) {
            return false;
        }
        m.k_min_q = (int8_t *)p;

        p = nullptr;
        if (sparse_hip_malloc_zero(&p, rows, stream) != hipSuccess) {
            sparse_hip_free(m.k_min_q);
            m = SparseGlobalPageMeta{};
            return false;
        }
        m.k_max_q = (int8_t *)p;
    }

    p = nullptr;
    const size_t d_bytes = (size_t)N_HEAD_KV * (size_t)max_pages * sizeof(__half);
    if (sparse_hip_malloc_zero(&p, d_bytes, stream) != hipSuccess) {
        sparse_hip_free(m.k_min_q);
        sparse_hip_free(m.k_max_q);
        sparse_hip_free(m.k_qmm);
        m = SparseGlobalPageMeta{};
        return false;
    }
    m.k_mm_d    = (__half *)p;
    m.max_pages = max_pages;
    m.fmt       = fmt;
    return true;
}

void sparse_global_page_meta_free(SparseGlobalPageMeta & m) {
    sparse_hip_free(m.k_min_q);
    sparse_hip_free(m.k_max_q);
    sparse_hip_free(m.k_qmm);
    sparse_hip_free(m.k_mm_d);
    m = SparseGlobalPageMeta{};
}

bool sparse_global_scratch_alloc(SparseGlobalScratch & s, hipStream_t stream) {
    void * p = nullptr;
    const size_t page_scores_bytes = (size_t)SPARSE_MAX_QB * N_HEAD_KV * SPARSE_MAX_PAGES * sizeof(float);
    const size_t qb_bytes = (size_t)SPARSE_MAX_QB * N_HEAD_KV * GQA_RATIO * HEAD_DIM * sizeof(float);
    const size_t sel_bytes = (size_t)SPARSE_MAX_QB * N_HEAD_KV * SPARSE_MAX_SELECTED * sizeof(int32_t);
    const size_t ns_bytes = (size_t)SPARSE_MAX_QB * N_HEAD_KV * sizeof(int32_t);

    if (sparse_hip_malloc_zero(&p, page_scores_bytes, stream) != hipSuccess) {
        return false;
    }
    s.page_scores = (float *)p;

    p = nullptr;
    if (sparse_hip_malloc_zero(&p, qb_bytes, stream) != hipSuccess) {
        sparse_global_scratch_free(s);
        return false;
    }
    s.q_min = (float *)p;

    p = nullptr;
    if (sparse_hip_malloc_zero(&p, qb_bytes, stream) != hipSuccess) {
        sparse_global_scratch_free(s);
        return false;
    }
    s.q_max = (float *)p;

    p = nullptr;
    if (sparse_hip_malloc_zero(&p, sel_bytes, stream) != hipSuccess) {
        sparse_global_scratch_free(s);
        return false;
    }
    s.selected_pages = (int32_t *)p;

    p = nullptr;
    if (sparse_hip_malloc_zero(&p, ns_bytes, stream) != hipSuccess) {
        sparse_global_scratch_free(s);
        return false;
    }
    s.n_selected = (int32_t *)p;
    return true;
}

void sparse_global_scratch_free(SparseGlobalScratch & s) {
    sparse_hip_free(s.page_scores);
    sparse_hip_free(s.q_min);
    sparse_hip_free(s.q_max);
    sparse_hip_free(s.selected_pages);
    sparse_hip_free(s.n_selected);
    s = SparseGlobalScratch{};
}

namespace {

template <typename Fn>
inline void sparse_dispatch_kv(int kv_type, Fn && fn) {
    if (kv_type == KV_Q4_1) {
        fn(std::integral_constant<int, KV_Q4_1>{});
    } else if (kv_type == KV_Q4_0) {
        fn(std::integral_constant<int, KV_Q4_0>{});
    } else {
        fn(std::integral_constant<int, KV_Q8_0>{});
    }
}

template <int KV_FMT, int META_FMT>
static inline void sparse_launch_minmax_prefill(
        const KVCacheSlot & kv_slot, const SparseGlobalPageMeta & meta,
        const dim3 & grid, hipStream_t stream, int pos_start, int n_tokens) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(sparse_page_minmax_prefill_kernel<KV_FMT, META_FMT>),
        grid, dim3(32, 1, 1), 0, stream, kv_slot, meta, pos_start, n_tokens);
}

template <int KV_FMT, int META_FMT>
static inline void sparse_launch_minmax_decode(
        const KVCacheSlot & kv_slot, const SparseGlobalPageMeta & meta,
        const dim3 & grid, hipStream_t stream, const StepMeta * d_meta) {
    hipLaunchKernelGGL(HIP_KERNEL_NAME(sparse_page_minmax_decode_kernel<KV_FMT, META_FMT>),
        grid, dim3(32, 1, 1), 0, stream, kv_slot, meta, d_meta);
}

inline int sparse_clamp_pages(int sparse_pages) {
    if (sparse_pages < SPARSE_MIN_PAGES) {
        return SPARSE_MIN_PAGES;
    }
    if (sparse_pages > SPARSE_MAX_SELECTED - 2) {
        return SPARSE_MAX_SELECTED - 2;
    }
    return sparse_pages;
}

} // anonymous namespace

void launch_kv_update_page_minmax_prefill(
        const KVCacheSlot & kv_slot,
        const SparseGlobalPageMeta & meta,
        int pos_start, int n_tokens,
        hipStream_t stream) {
    if (n_tokens <= 0 || meta.max_pages <= 0 ||
        (meta.k_min_q == nullptr && meta.k_qmm == nullptr)) {
        return;
    }
    const int first_page = pos_start / SPARSE_PAGE_SIZE;
    const int last_page  = (pos_start + n_tokens - 1) / SPARSE_PAGE_SIZE;
    const dim3 grid(last_page - first_page + 1, N_HEAD_KV, 1);
    const bool q4 = (meta.fmt == SPARSE_META_Q4);

    sparse_dispatch_kv(kv_slot.kv_type, [&](auto kv_tag) {
        constexpr int KV_FMT = decltype(kv_tag)::value;
        if (q4) {
            sparse_launch_minmax_prefill<KV_FMT, SPARSE_META_Q4>(
                kv_slot, meta, grid, stream, pos_start, n_tokens);
        } else {
            sparse_launch_minmax_prefill<KV_FMT, SPARSE_META_Q8>(
                kv_slot, meta, grid, stream, pos_start, n_tokens);
        }
    });
}

void launch_kv_update_page_minmax_decode(
        const KVCacheSlot & kv_slot,
        const SparseGlobalPageMeta & meta,
        const StepMeta * d_meta,
        hipStream_t stream) {
    if (meta.max_pages <= 0 ||
        (meta.k_min_q == nullptr && meta.k_qmm == nullptr)) {
        return;
    }
    const dim3 grid(N_HEAD_KV, 1, 1);
    const bool q4 = (meta.fmt == SPARSE_META_Q4);

    sparse_dispatch_kv(kv_slot.kv_type, [&](auto kv_tag) {
        constexpr int KV_FMT = decltype(kv_tag)::value;
        if (q4) {
            sparse_launch_minmax_decode<KV_FMT, SPARSE_META_Q4>(
                kv_slot, meta, grid, stream, d_meta);
        } else {
            sparse_launch_minmax_decode<KV_FMT, SPARSE_META_Q8>(
                kv_slot, meta, grid, stream, d_meta);
        }
    });
}

void launch_sparse_global_select_pages_decode(
        const float * q,
        const SparseGlobalPageMeta & meta,
        SparseGlobalScratch & scratch,
        const StepMeta * d_meta,
        int sparse_pages,
        hipStream_t stream) {
    const int budget = sparse_clamp_pages(sparse_pages);
    const dim3 sgrid(SPARSE_DECODE_SCORE_CHUNKS, N_HEAD_KV, 1);
    const dim3 sblock(128, 1, 1);
    if (meta.fmt == SPARSE_META_Q4) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(sparse_score_pages_decode_kernel<SPARSE_META_Q4>),
            sgrid, sblock, 0, stream,
            q, meta, d_meta, scratch.page_scores);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(sparse_score_pages_decode_kernel<SPARSE_META_Q8>),
            sgrid, sblock, 0, stream,
            q, meta, d_meta, scratch.page_scores);
    }

    const dim3 tgrid(N_HEAD_KV, 1, 1);
    const dim3 tblock(SPARSE_TOPK_THREADS, 1, 1);
    hipLaunchKernelGGL(sparse_topk_pages_decode_kernel,
        tgrid, tblock, 0, stream,
        scratch.page_scores, d_meta, meta.max_pages,
        scratch.selected_pages, scratch.n_selected, budget);
}

void launch_sparse_global_select_pages_prefill(
        const float * q,
        const SparseGlobalPageMeta & meta,
        SparseGlobalScratch & scratch,
        int n_tokens, int pos_start,
        int sparse_pages,
        hipStream_t stream) {
    if (n_tokens <= 0) {
        return;
    }
    const int budget = sparse_clamp_pages(sparse_pages);
    const int n_qb   = (n_tokens + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
    if (n_qb <= 0 || n_qb > SPARSE_MAX_QB) {
        return;
    }

    if (meta.fmt == SPARSE_META_Q4) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(sparse_score_pages_prefill_wmma_kernel<SPARSE_META_Q4>),
            dim3(N_HEAD_KV, n_qb, 1), dim3(32, 12, 1), 0, stream,
            q, meta, pos_start, n_tokens, budget, scratch.page_scores);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(sparse_score_pages_prefill_wmma_kernel<SPARSE_META_Q8>),
            dim3(N_HEAD_KV, n_qb, 1), dim3(32, 12, 1), 0, stream,
            q, meta, pos_start, n_tokens, budget, scratch.page_scores);
    }

    hipLaunchKernelGGL(sparse_topk_pages_prefill_kernel,
        dim3(N_HEAD_KV, n_qb, 1), dim3(SPARSE_TOPK_THREADS, 1, 1), 0, stream,
        scratch.page_scores, meta.max_pages, pos_start, n_tokens,
        scratch.selected_pages, scratch.n_selected, budget);
}

void launch_fa_sparse_global_decode(
        const float * q,
        KVCacheSlot kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,
        const float * r_prep,
        const float * attn_rel_bias,
        const float * qgate_in, int qgate_stride,
        float * out,
        const int32_t * selected_pages,
        const int32_t * n_selected_pages,
        float * tmp_o, float2 * tmp_meta,
        int n_splits,
        const StepMeta * d_meta,
        hipStream_t stream,
        int32_t * fa_done) {
    int splits = n_splits;
    if (splits < 1) {
        splits = 1;
    }
    if (splits > 32) {
        splits = 32;
    }
    const bool use_r_prep = (rel_table == nullptr);

    const dim3 grid(splits, N_HEAD_KV, 1);
    const dim3 block(32, 4, 1);

    sparse_dispatch_kv(kv_slot.kv_type, [&](auto kv_tag) {
        constexpr int KV_FMT = decltype(kv_tag)::value;
        if (use_r_prep) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(inkling_fa_decode_sparse_global_kernel<GQA_RATIO, KV_FMT, true>),
                grid, block, 0, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, r_prep, attn_rel_bias,
                qgate_in, qgate_stride, out, tmp_o, tmp_meta,
                selected_pages, n_selected_pages, splits, d_meta, fa_done);
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(inkling_fa_decode_sparse_global_kernel<GQA_RATIO, KV_FMT, false>),
                grid, block, 0, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, r_prep, attn_rel_bias,
                qgate_in, qgate_stride, out, tmp_o, tmp_meta,
                selected_pages, n_selected_pages, splits, d_meta, fa_done);
        }
    });

    if (splits > 1 && fa_done == nullptr) {
        hipLaunchKernelGGL(inkling_fa_decode_sparse_reduce_kernel,
            dim3(N_HEAD_KV, 1, 1), dim3(32, GQA_RATIO, 1), 0, stream,
            tmp_o, tmp_meta, sinks_in, qgate_in, out, splits);
    }
}

void launch_fa_sparse_global_prefill(
        const float * q,
        KVCacheSlot kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,
        const float * r_prep,
        const float * attn_rel_bias,
        const float * qgate_in, int qgate_stride,
        float * out,
        const int32_t * selected_pages,
        const int32_t * n_selected_pages,
        int n_tokens, int pos_start, int n_kv,
        hipStream_t stream) {
    const int n_qb = (n_tokens + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
    if (n_qb <= 0 || n_qb > SPARSE_MAX_QB) {
        return;
    }
    const bool use_r_prep = (rel_table == nullptr);

    constexpr int NWARPS = 12;
    const dim3 grid(n_qb, N_HEAD_KV, 1);
    const dim3 block(32, NWARPS, 1);

    sparse_dispatch_kv(kv_slot.kv_type, [&](auto kv_tag) {
        constexpr int KV_FMT = decltype(kv_tag)::value;
        if (use_r_prep) {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(inkling_fa_prefill_gqa_wmma_sparse_global_kernel<
                    NWARPS, SPARSE_PAGE_SIZE, KV_FMT, true>),
                grid, block, 0, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, r_prep, attn_rel_bias,
                qgate_in, qgate_stride, out, selected_pages, n_selected_pages,
                n_tokens, pos_start, n_kv);
        } else {
            hipLaunchKernelGGL(
                HIP_KERNEL_NAME(inkling_fa_prefill_gqa_wmma_sparse_global_kernel<
                    NWARPS, SPARSE_PAGE_SIZE, KV_FMT, false>),
                grid, block, 0, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, r_prep, attn_rel_bias,
                qgate_in, qgate_stride, out, selected_pages, n_selected_pages,
                n_tokens, pos_start, n_kv);
        }
    });
}

} // namespace inkling
