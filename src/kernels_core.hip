#include "inkling_engine.h"
#include "nib_mma.h"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace inkling {

namespace {

constexpr int WARP_SIZE = 32;

[[maybe_unused]] static __device__ __forceinline__ uint32_t bits_of(__half2 h) {
    return __builtin_bit_cast(uint32_t, h);
}

static __device__ __forceinline__ __half2 h2_of(uint32_t u) {
    return __builtin_bit_cast(__half2, u);
}

static __device__ __forceinline__ void rdna3_flush_stores() {
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#endif
}

static __device__ __forceinline__ void rdna3_inv_l0_l1() {
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("buffer_gl1_inv\n\tbuffer_gl0_inv" ::: "memory");
#endif
}

static __device__ __forceinline__ float dot_h8_f8(uint4 w, float4 x0, float4 x1, float acc) {
    const __half2 h0 = h2_of(w.x);
    const __half2 h1 = h2_of(w.y);
    const __half2 h2 = h2_of(w.z);
    const __half2 h3 = h2_of(w.w);
    acc = fmaf(__low2float(h0),  x0.x, acc);
    acc = fmaf(__high2float(h0), x0.y, acc);
    acc = fmaf(__low2float(h1),  x0.z, acc);
    acc = fmaf(__high2float(h1), x0.w, acc);
    acc = fmaf(__low2float(h2),  x1.x, acc);
    acc = fmaf(__high2float(h2), x1.y, acc);
    acc = fmaf(__low2float(h3),  x1.z, acc);
    acc = fmaf(__high2float(h3), x1.w, acc);
    return acc;
}

#if NIB_MMA_TILE_TARGET
typedef int      v4i      __attribute__((ext_vector_type(4)));
typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float    float8_t __attribute__((ext_vector_type(8)));

static __device__ __forceinline__ half16_t lds_frag(const _Float16 * p) {
    const v4i a = *reinterpret_cast<const v4i *>(p);
    const v4i b = *reinterpret_cast<const v4i *>(p + 8);
    half16_t r;
    __builtin_memcpy(&r, &a, 16);
    __builtin_memcpy(reinterpret_cast<char *>(&r) + 16, &b, 16);
    return r;
}
#endif

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = width >> 1; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, width);
    }
    return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
    #pragma unroll
    for (int offset = width >> 1; offset > 0; offset >>= 1) {
        x = fmaxf(x, __shfl_xor(x, offset, width));
    }
    return x;
}

static __device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

// Wave32 in-register unnormalized Sylvester-Walsh-Hadamard transform (N = 128, 4 elems/lane, ZERO LDS).
// Multiply by KQ_SCALE (1 / sqrt(128)) before or after for orthonormal transform.
static __device__ __forceinline__ void wave32_fwht_128_float4(float4 & v, const int lane) {
    // Stage 0 (h = 1, bit 0: local)
    const float a0 = v.x + v.y, a1 = v.x - v.y;
    const float a2 = v.z + v.w, a3 = v.z - v.w;
    // Stage 1 (h = 2, bit 1: local)
    float r[4] = { a0 + a2, a1 + a3, a0 - a2, a1 - a3 };

    // Stages 2..6 (h = 4, 8, 16, 32, 64 -> lane bits 0..4)
    #pragma unroll
    for (int m = 1; m <= 16; m <<= 1) {
        const bool hi = (lane & m) != 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float cur   = r[j];
            const float other = __shfl_xor(cur, m, WARP_SIZE);
            r[j] = hi ? (other - cur) : (cur + other);
        }
    }

    v = make_float4(r[0], r[1], r[2], r[3]);
}

// ============================================================================
// 1. Embed + RMSNorm (SoA Q8_0 tok_embd -> F32 RMSNorm * tok_norm)
// ============================================================================
// 576 threads = 18 Wave32s per token, 1 float4 (4 contiguous elements) per thread.
// All 4 elements in a thread belong to the same 32-element Q8_0 block (tid >> 3).
__launch_bounds__(576, 1)
static __global__ void embed_norm_2304_kernel(
        const int8_t   * __restrict__ embd_qs,
        const __half   * __restrict__ embd_d,
        const float    * __restrict__ tok_norm,
        const int32_t  * __restrict__ d_token_ids,
        const StepMeta * __restrict__ d_meta,
        float          * __restrict__ out,
        int n_tokens) {
    constexpr int N_WARPS = 18;
    const int t = blockIdx.x;
    if (t >= n_tokens) {
        return;
    }

    const int tid     = threadIdx.x; // 0..575
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    int32_t tok_id = (d_meta != nullptr) ? d_meta->token_id : d_token_ids[t];
    if ((uint32_t)tok_id >= (uint32_t)VOCAB_UNPADDED) {
        tok_id = 0;   // never gather outside the embedding table (issue #1)
    }

    const int32_t * row_qs = (const int32_t *)(embd_qs + (size_t)tok_id * N_EMBD);
    const __half  * row_d  = embd_d + (size_t)tok_id * (N_EMBD / 32);

    const int32_t q4 = row_qs[tid];
    const float   d  = __half2float(row_d[tid >> 3]);

    const float v0 = d * (float)(int8_t)( q4        & 0xFF);
    const float v1 = d * (float)(int8_t)((q4 >>  8) & 0xFF);
    const float v2 = d * (float)(int8_t)((q4 >> 16) & 0xFF);
    const float v3 = d * (float)(int8_t)((q4 >> 24) & 0xFF);

    float ssq = v0 * v0 + v1 * v1 + v2 * v2 + v3 * v3;
    ssq = warp_reduce_sum<WARP_SIZE>(ssq);

    __shared__ float smem[N_WARPS];
    if (lane_id == 0) {
        smem[warp_id] = ssq;
    }
    __syncthreads();

    __shared__ float s_inv_rms;
    if (tid == 0) {
        float total = 0.0f;
        #pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            total += smem[w];
        }
        s_inv_rms = rsqrtf(total * (1.0f / (float)N_EMBD) + RMS_EPS);
    }
    __syncthreads();

    const float inv_rms = s_inv_rms;
    const float4 nw = tok_norm ? ((const float4 *)tok_norm)[tid] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    float4 o;
    o.x = v0 * inv_rms * nw.x;
    o.y = v1 * inv_rms * nw.y;
    o.z = v2 * inv_rms * nw.z;
    o.w = v3 * inv_rms * nw.w;

    ((float4 *)(out + (size_t)t * N_EMBD))[tid] = o;
}

// ============================================================================
// 2. Depth Attention Mix (Block Attention Residuals)
// ============================================================================
struct StatePtrs {
    const float * p[MAX_RES_BLOCKS];
};

// Single-pass VGPR-resident kernel for N_EMBD == 2304 (576 threads = 18 Wave32s).
// Output layout: [n_q, n_tokens, N_EMBD] (q_outer = true).
template <int MAX_M, int MAX_Q>
__launch_bounds__(576, 1)
static __global__ void inkling_mix_2304_kernel(
        const float * __restrict__ query,
        StatePtrs     states,
        float       * __restrict__ dst,
        int n_states,
        int n_q,
        int n_tokens) {
    constexpr int N_WARPS = 18;
    const int t = blockIdx.x;
    if (t >= n_tokens) {
        return;
    }

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    float4 s_vec[MAX_M];
    float  sumsq[MAX_M];

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < n_states) {
            const float4 v = ((const float4 *)(states.p[m] + (size_t)t * N_EMBD))[tid];
            s_vec[m] = v;
            sumsq[m] = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }

    __shared__ float smem_sumsq[MAX_M][N_WARPS];
    __shared__ float smem_dot[MAX_Q][MAX_M][N_WARPS];
    __shared__ float smem_effw[MAX_Q][MAX_M];

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < n_states) {
            const float ws = warp_reduce_sum<WARP_SIZE>(sumsq[m]);
            if (lane_id == 0) {
                smem_sumsq[m][warp_id] = ws;
            }
        }
    }

    for (int q = 0; q < n_q; ++q) {
        const float4 qv = ((const float4 *)(query + (size_t)q * N_EMBD))[tid];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_states) {
                const float4 v = s_vec[m];
                const float d  = v.x * qv.x + v.y * qv.y + v.z * qv.z + v.w * qv.w;
                const float wd = warp_reduce_sum<WARP_SIZE>(d);
                if (lane_id == 0) {
                    smem_dot[q][m][warp_id] = wd;
                }
            }
        }
    }

    __syncthreads();

    if (tid == 0) {
        constexpr float inv_n    = 1.0f / (float)N_EMBD;
        const float     inv_sq_n = rsqrtf((float)N_EMBD);

        float inv_rms[MAX_M];
        for (int m = 0; m < n_states; ++m) {
            float total = 0.0f;
            #pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                total += smem_sumsq[m][w];
            }
            inv_rms[m] = rsqrtf(total * inv_n + RMS_EPS);
        }

        for (int q = 0; q < n_q; ++q) {
            float sc[MAX_M];
            float max_s = -INFINITY;
            for (int m = 0; m < n_states; ++m) {
                float total_d = 0.0f;
                #pragma unroll
                for (int w = 0; w < N_WARPS; ++w) {
                    total_d += smem_dot[q][m][w];
                }
                sc[m] = total_d * inv_rms[m] * inv_sq_n;
                max_s = fmaxf(max_s, sc[m]);
            }
            float sum_e = 0.0f;
            for (int m = 0; m < n_states; ++m) {
                sc[m] = expf(sc[m] - max_s);
                sum_e += sc[m];
            }
            const float inv_e = 1.0f / sum_e;
            for (int m = 0; m < n_states; ++m) {
                smem_effw[q][m] = sc[m] * inv_e * inv_rms[m];
            }
        }
    }

    __syncthreads();

    for (int q = 0; q < n_q; ++q) {
        float4 out_v = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_states) {
                const float w = smem_effw[q][m];
                out_v.x += w * s_vec[m].x;
                out_v.y += w * s_vec[m].y;
                out_v.z += w * s_vec[m].z;
                out_v.w += w * s_vec[m].w;
            }
        }
        float * out_ptr = dst + ((size_t)q * n_tokens + t) * N_EMBD;
        ((float4 *)out_ptr)[tid] = out_v;
    }
}

template <int MAX_M, int MAX_Q>
__launch_bounds__(192, 4)
static __global__ void inkling_mix_dual_2304_kernel(
        const float * __restrict__ query0, int n_q0, StatePtrs states0, int n_states0, float * __restrict__ dst0,
        const float * __restrict__ query1, int n_q1, StatePtrs states1, int n_states1, float * __restrict__ dst1,
        int n_tokens) {
    constexpr int N_WARPS = 6;
    constexpr int NT      = 192;
    const int t = blockIdx.x;
    const int b = blockIdx.y;
    if (t >= n_tokens || b >= 2) {
        return;
    }

    const float     * query    = (b == 0) ? query0    : query1;
    const StatePtrs & states   = (b == 0) ? states0   : states1;
    float           * dst      = (b == 0) ? dst0      : dst1;
    const int         n_states = (b == 0) ? n_states0 : n_states1;
    const int         n_q      = (b == 0) ? n_q0      : n_q1;

    const int tid     = threadIdx.x; // 0..191
    const int warp_id = tid >> 5;    // 0..5
    const int lane_id = tid & 31;    // 0..31

    __shared__ float smem_sumsq[MAX_M][N_WARPS];
    __shared__ float smem_dot[MAX_Q][MAX_M][N_WARPS];

    // Keep states in registers (3 float4s/thread * MAX_M): for MAX_M=5 -> 60 VGPRs, 0 spills!
    float4 s_vec[MAX_M][3];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < n_states) {
            const float4 * sp = (const float4 *)(states.p[m] + (size_t)t * N_EMBD);
            const float4 v0 = sp[tid + 0 * NT];
            const float4 v1 = sp[tid + 1 * NT];
            const float4 v2 = sp[tid + 2 * NT];
            s_vec[m][0] = v0;
            s_vec[m][1] = v1;
            s_vec[m][2] = v2;
            float ssq = (v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w)
                      + (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w)
                      + (v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w);
            ssq = warp_reduce_sum<WARP_SIZE>(ssq);
            if (lane_id == 0) {
                smem_sumsq[m][warp_id] = ssq;
            }
        }
    }

    for (int q = 0; q < n_q; ++q) {
        const float4 * qp = (const float4 *)(query + (size_t)q * N_EMBD);
        const float4 q0 = qp[tid + 0 * NT];
        const float4 q1 = qp[tid + 1 * NT];
        const float4 q2 = qp[tid + 2 * NT];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_states) {
                const float4 v0 = s_vec[m][0];
                const float4 v1 = s_vec[m][1];
                const float4 v2 = s_vec[m][2];
                float d = (v0.x * q0.x + v0.y * q0.y + v0.z * q0.z + v0.w * q0.w)
                        + (v1.x * q1.x + v1.y * q1.y + v1.z * q1.z + v1.w * q1.w)
                        + (v2.x * q2.x + v2.y * q2.y + v2.z * q2.z + v2.w * q2.w);
                d = warp_reduce_sum<WARP_SIZE>(d);
                if (lane_id == 0) {
                    smem_dot[q][m][warp_id] = d;
                }
            }
        }
    }

    // SINGLE __syncthreads() for the entire kernel! Every warp reduces the 6 partials in parallel!
    __syncthreads();

    constexpr float inv_n    = 1.0f / (float)N_EMBD;
    const float     inv_sq_n = rsqrtf((float)N_EMBD);

    float my_inv_rms = 0.0f;
    if (lane_id < n_states) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            tot += smem_sumsq[lane_id][w];
        }
        my_inv_rms = rsqrtf(tot * inv_n + RMS_EPS);
    }

    for (int q = 0; q < n_q; ++q) {
        float sc = -INFINITY;
        if (lane_id < n_states) {
            float tot_d = 0.0f;
            #pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                tot_d += smem_dot[q][lane_id][w];
            }
            sc = tot_d * my_inv_rms * inv_sq_n;
        }
        float max_s = sc;
        max_s = fmaxf(max_s, __shfl_xor(max_s, 1, 8));
        max_s = fmaxf(max_s, __shfl_xor(max_s, 2, 8));
        max_s = fmaxf(max_s, __shfl_xor(max_s, 4, 8));
        const float ex = (lane_id < n_states) ? expf(sc - max_s) : 0.0f;
        float sum_e = ex;
        sum_e += __shfl_xor(sum_e, 1, 8);
        sum_e += __shfl_xor(sum_e, 2, 8);
        sum_e += __shfl_xor(sum_e, 4, 8);
        const float my_effw = (lane_id < n_states) ? (ex * (1.0f / sum_e) * my_inv_rms) : 0.0f;

        float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 o1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 o2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_states) {
                const float w = __shfl(my_effw, m, WARP_SIZE);
                o0.x = fmaf(w, s_vec[m][0].x, o0.x); o0.y = fmaf(w, s_vec[m][0].y, o0.y);
                o0.z = fmaf(w, s_vec[m][0].z, o0.z); o0.w = fmaf(w, s_vec[m][0].w, o0.w);
                o1.x = fmaf(w, s_vec[m][1].x, o1.x); o1.y = fmaf(w, s_vec[m][1].y, o1.y);
                o1.z = fmaf(w, s_vec[m][1].z, o1.z); o1.w = fmaf(w, s_vec[m][1].w, o1.w);
                o2.x = fmaf(w, s_vec[m][2].x, o2.x); o2.y = fmaf(w, s_vec[m][2].y, o2.y);
                o2.z = fmaf(w, s_vec[m][2].z, o2.z); o2.w = fmaf(w, s_vec[m][2].w, o2.w);
            }
        }
        float4 * out4 = (float4 *)(dst + ((size_t)q * n_tokens + t) * N_EMBD);
        out4[tid + 0 * NT] = o0;
        out4[tid + 1 * NT] = o1;
        out4[tid + 2 * NT] = o2;
    }
}

struct NormPtrs {
    const float * p[4];
};

struct Ptr3 {
    const float * p[3];
};

struct HalfPtr3 {
    const __half * p[3];
};

struct MutPtr3 {
    float * p[3];
};

template <int MAX_M, int MAX_Q>
__launch_bounds__(192, 4)
static __global__ void inkling_mix_rms_norm_2304_kernel(
        const float * __restrict__ query,
        StatePtrs     states,
        NormPtrs      w_norms,
        float       * __restrict__ dst,
        int n_states,
        int n_q,
        int n_tokens,
        float eps) {
    constexpr int N_WARPS = 6;
    constexpr int NT      = 192;
    const int t = blockIdx.x;
    if (t >= n_tokens) {
        return;
    }

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __shared__ float smem_sumsq[MAX_M][N_WARPS];
    __shared__ float smem_dot[MAX_Q][MAX_M][N_WARPS];
    __shared__ float smem_out_ssq[MAX_Q][N_WARPS];

    float4 s_vec[MAX_M][3];
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < n_states) {
            const float4 * sp = (const float4 *)(states.p[m] + (size_t)t * N_EMBD);
            const float4 v0 = sp[tid + 0 * NT];
            const float4 v1 = sp[tid + 1 * NT];
            const float4 v2 = sp[tid + 2 * NT];
            s_vec[m][0] = v0;
            s_vec[m][1] = v1;
            s_vec[m][2] = v2;
            float ssq = (v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w)
                      + (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w)
                      + (v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w);
            ssq = warp_reduce_sum<WARP_SIZE>(ssq);
            if (lane_id == 0) {
                smem_sumsq[m][warp_id] = ssq;
            }
        }
    }

    #pragma unroll
    for (int q = 0; q < MAX_Q; ++q) {
        if (q < n_q) {
            const float4 * qp = (const float4 *)(query + (size_t)q * N_EMBD);
            const float4 q0 = qp[tid + 0 * NT];
            const float4 q1 = qp[tid + 1 * NT];
            const float4 q2 = qp[tid + 2 * NT];
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_states) {
                    const float4 v0 = s_vec[m][0];
                    const float4 v1 = s_vec[m][1];
                    const float4 v2 = s_vec[m][2];
                    float d = (v0.x * q0.x + v0.y * q0.y + v0.z * q0.z + v0.w * q0.w)
                            + (v1.x * q1.x + v1.y * q1.y + v1.z * q1.z + v1.w * q1.w)
                            + (v2.x * q2.x + v2.y * q2.y + v2.z * q2.z + v2.w * q2.w);
                    d = warp_reduce_sum<WARP_SIZE>(d);
                    if (lane_id == 0) {
                        smem_dot[q][m][warp_id] = d;
                    }
                }
            }
        }
    }

    // Barrier 1: every warp computes inv_rms[m] & effw[q][m] in parallel!
    __syncthreads();

    constexpr float inv_n    = 1.0f / (float)N_EMBD;
    const float     inv_sq_n = rsqrtf((float)N_EMBD);

    float my_inv_rms = 0.0f;
    if (lane_id < n_states) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            tot += smem_sumsq[lane_id][w];
        }
        my_inv_rms = rsqrtf(tot * inv_n + RMS_EPS);
    }

    // For MAX_Q == 1 (80% of calls), keep ov[3] in registers across Barrier 2 (12 VGPRs, 0 recompute!).
    // For MAX_Q > 1, save effw in registers (1 float/lane) to keep VGPRs <= 100 (0 spills!).
    float4 ov_q0[3];
    float  saved_effw[MAX_Q];
    #pragma unroll
    for (int q = 0; q < MAX_Q; ++q) {
        saved_effw[q] = 0.0f;
        if (q < n_q) {
            float sc = -INFINITY;
            if (lane_id < n_states) {
                float tot_d = 0.0f;
                #pragma unroll
                for (int w = 0; w < N_WARPS; ++w) {
                    tot_d += smem_dot[q][lane_id][w];
                }
                sc = tot_d * my_inv_rms * inv_sq_n;
            }
            float max_s = sc;
            max_s = fmaxf(max_s, __shfl_xor(max_s, 1, 8));
            max_s = fmaxf(max_s, __shfl_xor(max_s, 2, 8));
            max_s = fmaxf(max_s, __shfl_xor(max_s, 4, 8));
            const float ex = (lane_id < n_states) ? expf(sc - max_s) : 0.0f;
            float sum_e = ex;
            sum_e += __shfl_xor(sum_e, 1, 8);
            sum_e += __shfl_xor(sum_e, 2, 8);
            sum_e += __shfl_xor(sum_e, 4, 8);
            const float my_effw = (lane_id < n_states) ? (ex * (1.0f / sum_e) * my_inv_rms) : 0.0f;
            saved_effw[q] = my_effw;

            float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 o1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 o2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_states) {
                    const float w = __shfl(my_effw, m, WARP_SIZE);
                    o0.x = fmaf(w, s_vec[m][0].x, o0.x); o0.y = fmaf(w, s_vec[m][0].y, o0.y);
                    o0.z = fmaf(w, s_vec[m][0].z, o0.z); o0.w = fmaf(w, s_vec[m][0].w, o0.w);
                    o1.x = fmaf(w, s_vec[m][1].x, o1.x); o1.y = fmaf(w, s_vec[m][1].y, o1.y);
                    o1.z = fmaf(w, s_vec[m][1].z, o1.z); o1.w = fmaf(w, s_vec[m][1].w, o1.w);
                    o2.x = fmaf(w, s_vec[m][2].x, o2.x); o2.y = fmaf(w, s_vec[m][2].y, o2.y);
                    o2.z = fmaf(w, s_vec[m][2].z, o2.z); o2.w = fmaf(w, s_vec[m][2].w, o2.w);
                }
            }
            if constexpr (MAX_Q == 1) {
                ov_q0[0] = o0; ov_q0[1] = o1; ov_q0[2] = o2;
            }
            float ssq_o = (o0.x * o0.x + o0.y * o0.y + o0.z * o0.z + o0.w * o0.w)
                        + (o1.x * o1.x + o1.y * o1.y + o1.z * o1.z + o1.w * o1.w)
                        + (o2.x * o2.x + o2.y * o2.y + o2.z * o2.z + o2.w * o2.w);
            ssq_o = warp_reduce_sum<WARP_SIZE>(ssq_o);
            if (lane_id == 0) {
                smem_out_ssq[q][warp_id] = ssq_o;
            }
        }
    }

    // Barrier 2 (FINAL barrier — cut from 4 down to 2!): every warp computes inv_out[q] in parallel!
    __syncthreads();

    #pragma unroll
    for (int q = 0; q < MAX_Q; ++q) {
        if (q < n_q) {
            float tot = 0.0f;
            #pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                tot += smem_out_ssq[q][w];
            }
            const float inv = rsqrtf(tot * inv_n + eps);

            float4 o0, o1, o2;
            if constexpr (MAX_Q == 1) {
                o0 = ov_q0[0]; o1 = ov_q0[1]; o2 = ov_q0[2];
            } else {
                o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                o1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                o2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                #pragma unroll
                for (int m = 0; m < MAX_M; ++m) {
                    if (m < n_states) {
                        const float w = __shfl(saved_effw[q], m, WARP_SIZE);
                        o0.x = fmaf(w, s_vec[m][0].x, o0.x); o0.y = fmaf(w, s_vec[m][0].y, o0.y);
                        o0.z = fmaf(w, s_vec[m][0].z, o0.z); o0.w = fmaf(w, s_vec[m][0].w, o0.w);
                        o1.x = fmaf(w, s_vec[m][1].x, o1.x); o1.y = fmaf(w, s_vec[m][1].y, o1.y);
                        o1.z = fmaf(w, s_vec[m][1].z, o1.z); o1.w = fmaf(w, s_vec[m][1].w, o1.w);
                        o2.x = fmaf(w, s_vec[m][2].x, o2.x); o2.y = fmaf(w, s_vec[m][2].y, o2.y);
                        o2.z = fmaf(w, s_vec[m][2].z, o2.z); o2.w = fmaf(w, s_vec[m][2].w, o2.w);
                    }
                }
            }
            const float4 * wn4 = (const float4 *)w_norms.p[q];
            const float4 w0 = wn4 ? wn4[tid + 0 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float4 w1 = wn4 ? wn4[tid + 1 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float4 w2 = wn4 ? wn4[tid + 2 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            o0.x *= inv * w0.x; o0.y *= inv * w0.y; o0.z *= inv * w0.z; o0.w *= inv * w0.w;
            o1.x *= inv * w1.x; o1.y *= inv * w1.y; o1.z *= inv * w1.z; o1.w *= inv * w1.w;
            o2.x *= inv * w2.x; o2.y *= inv * w2.y; o2.z *= inv * w2.z; o2.w *= inv * w2.w;
            float4 * out4 = (float4 *)(dst + ((size_t)q * n_tokens + t) * N_EMBD);
            out4[tid + 0 * NT] = o0;
            out4[tid + 1 * NT] = o1;
            out4[tid + 2 * NT] = o2;
        }
    }
}

// ============================================================================
// 3. Loop Controller: ctrl_down + ctrl_apply (FP16 weights, FP32 accumulate)
// ============================================================================
// Decode ctrl_down (n_tokens <= 16): grid = (64, n_tokens), block = 288 (9 warps x 8 elems = 2304)
__launch_bounds__(288, 3)
static __global__ void inkling_ctrl_down_decode_kernel(
        const float  * __restrict__ cur,
        const float  * __restrict__ msg,
        const __half * __restrict__ w_down,
        float        * __restrict__ gate,
        int n_tokens) {
    constexpr int N_WARPS = 9;
    const int r = blockIdx.x; // 0..63
    const int t = blockIdx.y;
    if (r >= LOOP_GATE_RANK || t >= n_tokens) {
        return;
    }

    const int tid     = threadIdx.x; // 0..287
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const float  * ct  = cur + (size_t)t * N_EMBD;
    const float  * mt  = msg + (size_t)t * N_EMBD;
    const __half * wr0 = w_down + (size_t)r * (2 * N_EMBD);
    const __half * wr1 = wr0 + N_EMBD;

    const float4 c0 = ((const float4 *)ct)[tid * 2 + 0];
    const float4 c1 = ((const float4 *)ct)[tid * 2 + 1];
    const float4 m0 = ((const float4 *)mt)[tid * 2 + 0];
    const float4 m1 = ((const float4 *)mt)[tid * 2 + 1];
    const uint4  w0 = ((const uint4 *)wr0)[tid];
    const uint4  w1 = ((const uint4 *)wr1)[tid];

    float ssq_c = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z + c0.w * c0.w +
                  c1.x * c1.x + c1.y * c1.y + c1.z * c1.z + c1.w * c1.w;
    float ssq_m = m0.x * m0.x + m0.y * m0.y + m0.z * m0.z + m0.w * m0.w +
                  m1.x * m1.x + m1.y * m1.y + m1.z * m1.z + m1.w * m1.w;
    float dot0  = dot_h8_f8(w0, c0, c1, 0.0f);
    float dot1  = dot_h8_f8(w1, m0, m1, 0.0f);

    ssq_c = warp_reduce_sum<WARP_SIZE>(ssq_c);
    ssq_m = warp_reduce_sum<WARP_SIZE>(ssq_m);
    dot0  = warp_reduce_sum<WARP_SIZE>(dot0);
    dot1  = warp_reduce_sum<WARP_SIZE>(dot1);

    __shared__ float smem[4][N_WARPS];
    if (lane_id == 0) {
        smem[0][warp_id] = ssq_c;
        smem[1][warp_id] = ssq_m;
        smem[2][warp_id] = dot0;
        smem[3][warp_id] = dot1;
    }
    __syncthreads();

    if (tid == 0) {
        float tc = 0.0f, tm = 0.0f, td0 = 0.0f, td1 = 0.0f;
        #pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            tc  += smem[0][w];
            tm  += smem[1][w];
            td0 += smem[2][w];
            td1 += smem[3][w];
        }
        const float ic = rsqrtf(tc * (1.0f / (float)N_EMBD) + RMS_EPS);
        const float im = rsqrtf(tm * (1.0f / (float)N_EMBD) + RMS_EPS);
        gate[(size_t)t * LOOP_GATE_RANK + r] = td0 * ic + td1 * im;
    }
}

// Batched 3-way QKV decode/small ctrl_down: grid = (16, n_tokens, 3), block = 288 (4 rank rows/block)
__launch_bounds__(288, 3)
static __global__ void inkling_ctrl_down_qkv3_decode_kernel(
        Ptr3          cur,
        const float * __restrict__ msg_base,
        HalfPtr3      w_down,
        float       * __restrict__ gate,
        int n_tokens) {
    constexpr int N_WARPS = 9;
    const int r0 = blockIdx.x * 4;
    const int t  = blockIdx.y;
    const int j  = blockIdx.z;
    if (r0 >= LOOP_GATE_RANK || t >= n_tokens || j >= 3) {
        return;
    }

    const int tid     = threadIdx.x; // 0..287
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const float  * ct = cur.p[j] + (size_t)t * N_EMBD;
    const float  * mt = msg_base + ((size_t)j * n_tokens + t) * N_EMBD;
    const __half * wd = w_down.p[j];

    const float4 c0 = ((const float4 *)ct)[tid * 2 + 0];
    const float4 c1 = ((const float4 *)ct)[tid * 2 + 1];
    const float4 m0 = ((const float4 *)mt)[tid * 2 + 0];
    const float4 m1 = ((const float4 *)mt)[tid * 2 + 1];

    float ssq_c = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z + c0.w * c0.w +
                  c1.x * c1.x + c1.y * c1.y + c1.z * c1.z + c1.w * c1.w;
    float ssq_m = m0.x * m0.x + m0.y * m0.y + m0.z * m0.z + m0.w * m0.w +
                  m1.x * m1.x + m1.y * m1.y + m1.z * m1.z + m1.w * m1.w;

    float dot0[4], dot1[4];
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        const __half * wr0 = wd + (size_t)(r0 + k) * (2 * N_EMBD);
        const __half * wr1 = wr0 + N_EMBD;
        const uint4 w0 = ((const uint4 *)wr0)[tid];
        const uint4 w1 = ((const uint4 *)wr1)[tid];
        dot0[k] = dot_h8_f8(w0, c0, c1, 0.0f);
        dot1[k] = dot_h8_f8(w1, m0, m1, 0.0f);
    }

    ssq_c = warp_reduce_sum<WARP_SIZE>(ssq_c);
    ssq_m = warp_reduce_sum<WARP_SIZE>(ssq_m);
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        dot0[k] = warp_reduce_sum<WARP_SIZE>(dot0[k]);
        dot1[k] = warp_reduce_sum<WARP_SIZE>(dot1[k]);
    }

    __shared__ float smem[10][N_WARPS];
    if (lane_id == 0) {
        smem[0][warp_id] = ssq_c;
        smem[1][warp_id] = ssq_m;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            smem[2 + k][warp_id] = dot0[k];
            smem[6 + k][warp_id] = dot1[k];
        }
    }
    __syncthreads();

    if (tid == 0) {
        float tc = 0.0f, tm = 0.0f;
        float td0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float td1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            tc += smem[0][w];
            tm += smem[1][w];
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                td0[k] += smem[2 + k][w];
                td1[k] += smem[6 + k][w];
            }
        }
        const float ic = rsqrtf(tc * (1.0f / (float)N_EMBD) + RMS_EPS);
        const float im = rsqrtf(tm * (1.0f / (float)N_EMBD) + RMS_EPS);
        float * g_out = gate + ((size_t)j * n_tokens + t) * LOOP_GATE_RANK + r0;
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            g_out[k] = td0[k] * ic + td1[k] * im;
        }
    }
}

// Single-token fused ffn_mix (n_q=1) + FP16 ctrl_down:
// grid = 32 blocks (2 gate rows/block), block = 288 threads (9 warps x 8 elems = 2304).
// Block 0 writes out_mixed[0..2303], all blocks compute out_gate[0..63].
template <int MAX_M>
__launch_bounds__(288, 3)
static __global__ void mix_ctrl_down_decode_kernel(
        const float  * __restrict__ res_query,
        StatePtrs                   states,
        int                         n_states,
        const float  * __restrict__ msg,
        const __half * __restrict__ w_down,
        float        * __restrict__ out_mixed,
        float        * __restrict__ out_gate) {
    constexpr int N_WARPS = 9;
    const int bid     = blockIdx.x; // 0..31 -> rows 2*bid, 2*bid+1
    const int tid     = threadIdx.x; // 0..287
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const float4 rq0 = ((const float4 *)res_query)[tid * 2 + 0];
    const float4 rq1 = ((const float4 *)res_query)[tid * 2 + 1];

    float4 s_vec0[MAX_M];
    float4 s_vec1[MAX_M];

    __shared__ float smem_ssq[MAX_M][N_WARPS];
    __shared__ float smem_dot[MAX_M][N_WARPS];
    __shared__ float s_effw[MAX_M];

    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < n_states) {
            const float4 v0 = ((const float4 *)states.p[m])[tid * 2 + 0];
            const float4 v1 = ((const float4 *)states.p[m])[tid * 2 + 1];
            s_vec0[m] = v0;
            s_vec1[m] = v1;
            const float ssq = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w +
                              v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w;
            const float dot = v0.x * rq0.x + v0.y * rq0.y + v0.z * rq0.z + v0.w * rq0.w +
                              v1.x * rq1.x + v1.y * rq1.y + v1.z * rq1.z + v1.w * rq1.w;
            const float ws = warp_reduce_sum<WARP_SIZE>(ssq);
            const float wd = warp_reduce_sum<WARP_SIZE>(dot);
            if (lane_id == 0) {
                smem_ssq[m][warp_id] = ws;
                smem_dot[m][warp_id] = wd;
            }
        }
    }
    __syncthreads();

    if (tid == 0) {
        constexpr float inv_n    = 1.0f / (float)N_EMBD;
        const float     inv_sq_n = rsqrtf((float)N_EMBD);
        float inv_rms[MAX_M];
        float sc[MAX_M];
        float max_s = -INFINITY;
        for (int m = 0; m < n_states; ++m) {
            float ts = 0.0f, td = 0.0f;
            #pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                ts += smem_ssq[m][w];
                td += smem_dot[m][w];
            }
            inv_rms[m] = rsqrtf(ts * inv_n + RMS_EPS);
            sc[m] = td * inv_rms[m] * inv_sq_n;
            max_s = fmaxf(max_s, sc[m]);
        }
        float sum_e = 0.0f;
        for (int m = 0; m < n_states; ++m) {
            sc[m] = expf(sc[m] - max_s);
            sum_e += sc[m];
        }
        const float inv_e = 1.0f / sum_e;
        for (int m = 0; m < n_states; ++m) {
            s_effw[m] = sc[m] * inv_e * inv_rms[m];
        }
    }
    __syncthreads();

    float4 c0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 c1 = {0.0f, 0.0f, 0.0f, 0.0f};
    #pragma unroll
    for (int m = 0; m < MAX_M; ++m) {
        if (m < n_states) {
            const float w = s_effw[m];
            c0.x += w * s_vec0[m].x; c0.y += w * s_vec0[m].y; c0.z += w * s_vec0[m].z; c0.w += w * s_vec0[m].w;
            c1.x += w * s_vec1[m].x; c1.y += w * s_vec1[m].y; c1.z += w * s_vec1[m].z; c1.w += w * s_vec1[m].w;
        }
    }
    if (bid == 0) {
        ((float4 *)out_mixed)[tid * 2 + 0] = c0;
        ((float4 *)out_mixed)[tid * 2 + 1] = c1;
    }

    const float4 m0 = ((const float4 *)msg)[tid * 2 + 0];
    const float4 m1 = ((const float4 *)msg)[tid * 2 + 1];

    float ssq_c = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z + c0.w * c0.w +
                  c1.x * c1.x + c1.y * c1.y + c1.z * c1.z + c1.w * c1.w;
    float ssq_m = m0.x * m0.x + m0.y * m0.y + m0.z * m0.z + m0.w * m0.w +
                  m1.x * m1.x + m1.y * m1.y + m1.z * m1.z + m1.w * m1.w;

    float dot0[2], dot1[2];
    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int r = bid * 2 + k;
        const __half * wr0 = w_down + (size_t)r * (2 * N_EMBD);
        const __half * wr1 = wr0 + N_EMBD;
        const uint4 w0 = ((const uint4 *)wr0)[tid];
        const uint4 w1 = ((const uint4 *)wr1)[tid];
        dot0[k] = dot_h8_f8(w0, c0, c1, 0.0f);
        dot1[k] = dot_h8_f8(w1, m0, m1, 0.0f);
    }

    ssq_c = warp_reduce_sum<WARP_SIZE>(ssq_c);
    ssq_m = warp_reduce_sum<WARP_SIZE>(ssq_m);
    dot0[0] = warp_reduce_sum<WARP_SIZE>(dot0[0]);
    dot0[1] = warp_reduce_sum<WARP_SIZE>(dot0[1]);
    dot1[0] = warp_reduce_sum<WARP_SIZE>(dot1[0]);
    dot1[1] = warp_reduce_sum<WARP_SIZE>(dot1[1]);

    __shared__ float smem_g[6][N_WARPS];
    if (lane_id == 0) {
        smem_g[0][warp_id] = ssq_c;
        smem_g[1][warp_id] = ssq_m;
        smem_g[2][warp_id] = dot0[0];
        smem_g[3][warp_id] = dot0[1];
        smem_g[4][warp_id] = dot1[0];
        smem_g[5][warp_id] = dot1[1];
    }
    __syncthreads();

    if (tid == 0) {
        float tc = 0.0f, tm = 0.0f, td00 = 0.0f, td01 = 0.0f, td10 = 0.0f, td11 = 0.0f;
        #pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            tc   += smem_g[0][w];
            tm   += smem_g[1][w];
            td00 += smem_g[2][w];
            td01 += smem_g[3][w];
            td10 += smem_g[4][w];
            td11 += smem_g[5][w];
        }
        const float ic = rsqrtf(tc * (1.0f / (float)N_EMBD) + RMS_EPS);
        const float im = rsqrtf(tm * (1.0f / (float)N_EMBD) + RMS_EPS);
        out_gate[bid * 2 + 0] = td00 * ic + td10 * im;
        out_gate[bid * 2 + 1] = td01 * ic + td11 * im;
    }
}

// Prefill WMMA ctrl_down (n_tokens > 16): supports 1 or 3 branches via blockIdx.z, FP16 w_down.
// 100% deterministic (ZERO float atomicAdd, ZERO hipMemsetAsync)!
// Tile: BM = 16, BN = 16, BK = 64 -> 4 blocks along N=64, 4 warps/block each compute 1 kc=0..3 slice of BK=64,
// then deterministically reduce the 4 warps' [16,16] partials in LDS.
__launch_bounds__(128, 8)
static __global__ void inkling_ctrl_down_prefill_wmma_kernel(
        Ptr3          cur,
        const float * __restrict__ msg_base,
        HalfPtr3      w_down,
        float       * __restrict__ gate,
        int n_tokens) {
#if NIB_MMA_TILE_TARGET
    constexpr int BM     = 16;
    constexpr int BN     = 16;
    constexpr int BK     = 64;
    constexpr int LDK    = BK + 8; // 72 halves
    constexpr int NWARPS = 4;
    constexpr int NTH    = NWARPS * 32; // 128

    const int r0 = blockIdx.x * BN; // 0, 16, 32, 48
    const int t0 = blockIdx.y * BM;
    const int j  = blockIdx.z;
    if (r0 >= LOOP_GATE_RANK || t0 >= n_tokens) {
        return;
    }

    const int tid    = threadIdx.x; // 0..127
    const int warp   = tid >> 5;    // 0..3 -> handles kc = warp
    const int lane   = tid & 31;    // 0..31
    const int l16    = lane & 15;   // 0..15
    const int half_w = lane >> 4;   // 0..1

    const float  * cur_j    = cur.p[j];
    const float  * msg_j    = msg_base + (size_t)j * n_tokens * N_EMBD;
    const __half * w_down_j = w_down.p[j];
    float        * gate_j   = gate + (size_t)j * n_tokens * LOOP_GATE_RANK;

    __shared__ float s_inv_c[BM];
    __shared__ float s_inv_m[BM];
    union CtrlDownSmem {
        struct {
            alignas(16) _Float16 a[BM * LDK]; // 16 * 72 * 2 = 2,304 B
            alignas(16) _Float16 b[BN * LDK]; // 16 * 72 * 2 = 2,304 B
        } gemm;
        float red[NWARPS][BM][BN];            // 4 * 16 * 16 * 4 = 4,096 B
    };
    __shared__ CtrlDownSmem smem;

    // Prologue: 4 warps compute inv_rms for BM=16 tokens (4 tokens/warp)
    #pragma unroll
    for (int mm = 0; mm < 4; ++mm) {
        const int m = warp * 4 + mm;
        if (t0 + m < n_tokens) {
            const float4 * cp = (const float4 *)(cur_j + (size_t)(t0 + m) * N_EMBD);
            const float4 * mp = (const float4 *)(msg_j + (size_t)(t0 + m) * N_EMBD);
            float sc = 0.0f, sm = 0.0f;
            for (int i = lane; i < N_EMBD / 4; i += 32) {
                const float4 cv = cp[i];
                const float4 mv = mp[i];
                sc += cv.x * cv.x + cv.y * cv.y + cv.z * cv.z + cv.w * cv.w;
                sm += mv.x * mv.x + mv.y * mv.y + mv.z * mv.z + mv.w * mv.w;
            }
            sc = warp_reduce_sum<32>(sc);
            sm = warp_reduce_sum<32>(sm);
            if (lane == 0) {
                s_inv_c[m] = rsqrtf(sc * (1.0f / (float)N_EMBD) + RMS_EPS);
                s_inv_m[m] = rsqrtf(sm * (1.0f / (float)N_EMBD) + RMS_EPS);
            }
        } else if (lane == 0) {
            s_inv_c[m] = 0.0f;
            s_inv_m[m] = 0.0f;
        }
    }
    __syncthreads();

    float8_t acc = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    constexpr int TOTAL_K_STEPS = N_EMBD / BK; // 36
    #pragma unroll 1
    for (int ks = 0; ks < TOTAL_K_STEPS; ++ks) {
        const int k0 = ks * BK;
        // 1. Stage cur_norm (2 float4/thread) + w0 (1 uint4/thread)
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int idx4 = tid + p * NTH; // 0..255 -> 16 rows x 16 float4s
            const int m    = idx4 >> 4;
            const int k4   = idx4 & 15;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (t0 + m < n_tokens) {
                v = ((const float4 *)(cur_j + (size_t)(t0 + m) * N_EMBD + k0))[k4];
                const float ic = s_inv_c[m];
                v.x *= ic; v.y *= ic; v.z *= ic; v.w *= ic;
            }
            const __half2 h01 = __floats2half2_rn(v.x, v.y);
            const __half2 h23 = __floats2half2_rn(v.z, v.w);
            *reinterpret_cast<uint2 *>(&smem.gemm.a[m * LDK + k4 * 4]) = make_uint2(bits_of(h01), bits_of(h23));
        }
        {
            const int r  = tid >> 3; // 0..15
            const int k8 = tid & 7;  // 0..7
            *reinterpret_cast<uint4 *>(&smem.gemm.b[r * LDK + k8 * 8]) =
                ((const uint4 *)(w_down_j + (size_t)(r0 + r) * (2 * N_EMBD) + k0))[k8];
        }
        __syncthreads();

#if NIB_MMA_NATIVE
        {
            const half16_t af = lds_frag(&smem.gemm.a[l16 * LDK + warp * 16]);
            const half16_t bf = lds_frag(&smem.gemm.b[l16 * LDK + warp * 16]);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc);
        }
#else
        {
            const half16_t bf = lds_frag(&smem.gemm.b[l16 * LDK + warp * 16]);
            nib_mma::mma_stride(&smem.gemm.a[warp * 16], LDK, bf, acc, lane);
        }
#endif
        __syncthreads();

        // 2. Stage msg_norm (2 float4/thread) + w1 (1 uint4/thread)
        #pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int idx4 = tid + p * NTH;
            const int m    = idx4 >> 4;
            const int k4   = idx4 & 15;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (t0 + m < n_tokens) {
                v = ((const float4 *)(msg_j + (size_t)(t0 + m) * N_EMBD + k0))[k4];
                const float im = s_inv_m[m];
                v.x *= im; v.y *= im; v.z *= im; v.w *= im;
            }
            const __half2 h01 = __floats2half2_rn(v.x, v.y);
            const __half2 h23 = __floats2half2_rn(v.z, v.w);
            *reinterpret_cast<uint2 *>(&smem.gemm.a[m * LDK + k4 * 4]) = make_uint2(bits_of(h01), bits_of(h23));
        }
        {
            const int r  = tid >> 3;
            const int k8 = tid & 7;
            *reinterpret_cast<uint4 *>(&smem.gemm.b[r * LDK + k8 * 8]) =
                ((const uint4 *)(w_down_j + (size_t)(r0 + r) * (2 * N_EMBD) + N_EMBD + k0))[k8];
        }
        __syncthreads();

#if NIB_MMA_NATIVE
        {
            const half16_t af = lds_frag(&smem.gemm.a[l16 * LDK + warp * 16]);
            const half16_t bf = lds_frag(&smem.gemm.b[l16 * LDK + warp * 16]);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc);
        }
#else
        {
            const half16_t bf = lds_frag(&smem.gemm.b[l16 * LDK + warp * 16]);
            nib_mma::mma_stride(&smem.gemm.a[warp * 16], LDK, bf, acc, lane);
        }
#endif
        __syncthreads();
    }

    // Deterministic 4-warp reduction in LDS + coalesced 64B global store
    #pragma unroll
    for (int idx = 0; idx < 8; ++idx) {
        const int t_loc = 2 * idx + half_w;
        smem.red[warp][t_loc][l16] = acc[idx];
    }
    __syncthreads();

    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int idx   = tid + p * NTH; // 0..255
        const int t_loc = idx >> 4;      // 0..15
        const int r_loc = idx & 15;      // 0..15
        const int t     = t0 + t_loc;
        if (t < n_tokens) {
            const float s01 = smem.red[0][t_loc][r_loc] + smem.red[1][t_loc][r_loc];
            const float s23 = smem.red[2][t_loc][r_loc] + smem.red[3][t_loc][r_loc];
            gate_j[(size_t)t * LOOP_GATE_RANK + r0 + r_loc] = s01 + s23;
        }
    }
#else
    (void)cur; (void)msg_base; (void)w_down; (void)gate; (void)n_tokens;
#endif
}

// Decode ctrl_apply (n_tokens <= 8): 8 lanes per output row (uint4 = 8 halves), 32 rows/block (256 threads)
__launch_bounds__(256, 4)
static __global__ void inkling_ctrl_apply_decode_kernel(
        const float  * __restrict__ cur,
        const float  * __restrict__ msg,
        const float  * __restrict__ gate,
        const __half * __restrict__ w_gain,
        const __half * __restrict__ w_forget,
        float        * __restrict__ out,
        int n_tokens) {
    const int c = blockIdx.x * 32 + (threadIdx.x >> 3);
    const int t = blockIdx.y;
    if (c >= N_EMBD || t >= n_tokens) {
        return;
    }

    const int lane8  = threadIdx.x & 7;
    const float  * gt = gate + (size_t)t * LOOP_GATE_RANK;
    const __half * wg = w_gain   + (size_t)c * LOOP_GATE_RANK;
    const __half * wf = w_forget + (size_t)c * LOOP_GATE_RANK;

    const float4 g0  = ((const float4 *)gt)[lane8 * 2 + 0];
    const float4 g1  = ((const float4 *)gt)[lane8 * 2 + 1];
    const uint4  wg8 = ((const uint4 *)wg)[lane8];
    const uint4  wf8 = ((const uint4 *)wf)[lane8];

    float dg = dot_h8_f8(wg8, g0, g1, 0.0f);
    float df = dot_h8_f8(wf8, g0, g1, 0.0f);

    dg = warp_reduce_sum<8>(dg);
    df = warp_reduce_sum<8>(df);

    if (lane8 == 0) {
        const float gain   = tanhf(dg);
        const float forget = tanhf(df);
        const size_t idx   = (size_t)t * N_EMBD + c;
        const float cv     = cur[idx];
        const float mv     = msg[idx];
        out[idx] = cv - forget * cv + gain * mv;
    }
}

// Batched 3-way QKV small-batch ctrl_apply: grid = (72, n_tokens, 3), block = 256 (32 rows/block, 8 lanes/row)
__launch_bounds__(256, 4)
static __global__ void inkling_ctrl_apply_qkv3_small_kernel(
        Ptr3          cur,
        const float * __restrict__ msg_base,
        const float * __restrict__ gate,
        HalfPtr3      w_gain,
        HalfPtr3      w_forget,
        MutPtr3       out,
        int n_tokens) {
    const int c = blockIdx.x * 32 + (threadIdx.x >> 3);
    const int t = blockIdx.y;
    const int j = blockIdx.z;
    if (c >= N_EMBD || t >= n_tokens || j >= 3) {
        return;
    }

    const int lane8 = threadIdx.x & 7;
    const float  * gt = gate + ((size_t)j * n_tokens + t) * LOOP_GATE_RANK;
    const __half * wg = w_gain.p[j]   + (size_t)c * LOOP_GATE_RANK;
    const __half * wf = w_forget.p[j] + (size_t)c * LOOP_GATE_RANK;

    const float4 g0  = ((const float4 *)gt)[lane8 * 2 + 0];
    const float4 g1  = ((const float4 *)gt)[lane8 * 2 + 1];
    const uint4  wg8 = ((const uint4 *)wg)[lane8];
    const uint4  wf8 = ((const uint4 *)wf)[lane8];

    float dg = dot_h8_f8(wg8, g0, g1, 0.0f);
    float df = dot_h8_f8(wf8, g0, g1, 0.0f);

    dg = warp_reduce_sum<8>(dg);
    df = warp_reduce_sum<8>(df);

    if (lane8 == 0) {
        const float gain   = tanhf(dg);
        const float forget = tanhf(df);
        const size_t idx   = (size_t)t * N_EMBD + c;
        const float cv     = cur.p[j][idx];
        const float mv     = msg_base[(size_t)j * n_tokens * N_EMBD + idx];
        out.p[j][idx] = cv - forget * cv + gain * mv;
    }
}

// Multi-WGP Cooperative Single-Token 3-way QKV ctrl_apply + RMSNorm (n_tokens == 1, scratch != nullptr):
// grid = (32, 3) = 96 blocks across 96 CUs; block = 192 threads (6 warps = 24 groups of 8 lanes).
// Each block handles 72 channels (3 steps x 24 channels).
// Scratch slots used: scratch[260..262] (float ssq_accum[3]) & scratch[264..266] (int32 done[3]).
// Last block per branch resets BOTH its ssq_accum and done counter to 0 before scaling out_norm[j] in-place!
__launch_bounds__(192, 4)
static __global__ void inkling_ctrl_apply_norm_qkv3_multiwgp_kernel(
        Ptr3          cur,
        const float * __restrict__ msg_base,
        const float * __restrict__ gate,
        HalfPtr3      w_gain,
        HalfPtr3      w_forget,
        Ptr3          w_norm,
        MutPtr3       out_norm,
        float         eps,
        float       * __restrict__ scratch) {
    constexpr int N_WARPS = 6;
    constexpr int N_BLK   = 32;
    const int b = blockIdx.x; // 0..31
    const int j = blockIdx.y; // 0..2
    if (b >= N_BLK || j >= 3) {
        return;
    }

    const int tid     = threadIdx.x; // 0..191
    const int warp_id = tid >> 5;    // 0..5
    const int lane    = tid & 31;    // 0..31
    const int grp     = tid >> 3;    // 0..23
    const int k8      = tid & 7;     // 0..7

    __shared__ alignas(16) float s_gate[LOOP_GATE_RANK];
    __shared__ float smem_ssq[N_WARPS];
    __shared__ bool  s_is_last;
    __shared__ float s_inv;

    if (tid < LOOP_GATE_RANK) {
        s_gate[tid] = gate[(size_t)j * LOOP_GATE_RANK + tid];
    }
    __syncthreads();

    const float4 g0 = ((const float4 *)s_gate)[k8 * 2 + 0];
    const float4 g1 = ((const float4 *)s_gate)[k8 * 2 + 1];
    const float  * cur_j = cur.p[j];
    const float  * msg_j = msg_base + (size_t)j * N_EMBD;
    const __half * wg_j  = w_gain.p[j];
    const __half * wf_j  = w_forget.p[j];
    float        * out_j = out_norm.p[j];

    float ssq = 0.0f;
    const int c_base = b * 72 + grp;
    #pragma unroll
    for (int step = 0; step < 3; ++step) {
        const int c = c_base + step * 24;
        const uint4 wg8 = ((const uint4 *)(wg_j + (size_t)c * LOOP_GATE_RANK))[k8];
        const uint4 wf8 = ((const uint4 *)(wf_j + (size_t)c * LOOP_GATE_RANK))[k8];

        float dg = dot_h8_f8(wg8, g0, g1, 0.0f);
        float df = dot_h8_f8(wf8, g0, g1, 0.0f);
        dg = warp_reduce_sum<8>(dg);
        df = warp_reduce_sum<8>(df);

        if (k8 == 0) {
            const float gain   = tanhf(dg);
            const float forget = tanhf(df);
            const float cv     = cur_j[c];
            const float mv     = msg_j[c];
            const float val    = cv - forget * cv + gain * mv;
            out_j[c] = val;
            ssq += val * val;
        }
    }

    ssq = warp_reduce_sum<WARP_SIZE>(ssq);
    if (lane == 0) {
        smem_ssq[warp_id] = ssq;
    }
    rdna3_flush_stores();
    __syncthreads();

    if (tid == 0) {
        float blk_ssq = 0.0f;
        #pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            blk_ssq += smem_ssq[w];
        }
        scratch[280 + j * N_BLK + b] = blk_ssq;
        rdna3_flush_stores();
        int32_t * done_ptr = reinterpret_cast<int32_t *>(scratch + 264 + j);
        const int prev = __hip_atomic_fetch_add(done_ptr, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_is_last = (prev == N_BLK - 1);
    }
    __syncthreads();

    if (s_is_last) {
        rdna3_inv_l0_l1();
        if (warp_id == 0) {
            const float my_ssq = scratch[280 + j * N_BLK + lane];
            const float tot_ssq = warp_reduce_sum<WARP_SIZE>(my_ssq);
            if (lane == 0) {
                int32_t * done_ptr = reinterpret_cast<int32_t *>(scratch + 264 + j);
                __hip_atomic_store(done_ptr, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                s_inv = rsqrtf(tot_ssq * (1.0f / (float)N_EMBD) + eps);
            }
        }
        __syncthreads();

        const float inv = s_inv;
        const float * wn_j = w_norm.p[j];
        // 576 float4s / 192 threads = 3 exact float4s/thread
        #pragma unroll
        for (int p = 0; p < 3; ++p) {
            const int idx4 = tid + p * 192;
            const float4 xv = ((const float4 *)out_j)[idx4];
            const float4 wv = wn_j ? ((const float4 *)wn_j)[idx4] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float4 ov;
            ov.x = xv.x * inv * wv.x;
            ov.y = xv.y * inv * wv.y;
            ov.z = xv.z * inv * wv.z;
            ov.w = xv.w * inv * wv.w;
            ((float4 *)out_j)[idx4] = ov;
        }
    }
}

// Prefill WMMA ctrl_apply (n_tokens > 8): supports 1 or 3 branches via blockIdx.z
// Tile: BM = 64, BN = 64, K = 64 (4 WMMA K-steps), 8 warps (256 threads), 16x32 per warp.
// Direct 128-bit uint4 FP16 weight loads into LDS (ZERO F32->FP16 weight conversions!).
__launch_bounds__(256, 2)
static __global__ void inkling_ctrl_apply_prefill_wmma_kernel(
        Ptr3          cur,
        const float * __restrict__ msg_base,
        const float * __restrict__ gate,
        HalfPtr3      w_gain,
        HalfPtr3      w_forget,
        MutPtr3       out,
        int n_tokens) {
#if NIB_MMA_TILE_TARGET
    constexpr int BM  = 64;
    constexpr int BN  = 64;
    constexpr int K   = LOOP_GATE_RANK; // 64
    constexpr int LDK = K + 8;          // 72 halves

    const int c0 = blockIdx.x * BN;
    const int t0 = blockIdx.y * BM;
    const int j  = blockIdx.z;
    if (c0 >= N_EMBD || t0 >= n_tokens) {
        return;
    }

    const int tid    = threadIdx.x; // 0..255
    const int warp   = tid >> 5;    // 0..7
    const int lane   = tid & 31;    // 0..31
    const int wm     = warp >> 1;   // 0..3
    const int wn     = warp & 1;    // 0..1
    const int l16    = lane & 15;
    const int half_w = lane >> 4;

    const float  * cur_j  = cur.p[j];
    const float  * msg_j  = msg_base + (size_t)j * n_tokens * N_EMBD;
    const float  * gate_j = gate + (size_t)j * n_tokens * LOOP_GATE_RANK;
    const __half * wg_j   = w_gain.p[j];
    const __half * wf_j   = w_forget.p[j];
    float        * out_j  = out.p[j];

    __shared__ alignas(16) _Float16 s_gate[BM * LDK];
    __shared__ alignas(16) _Float16 s_w[BN * LDK];

    // 1. Load gate -> s_gate (4 float4/thread), w_gain -> s_w (2 uint4/thread), prefetch w_forget (2 uint4/thread)
    #pragma unroll
    for (int p = 0; p < 4; ++p) {
        const int idx4 = tid + p * 256; // 0..1023 -> 64 rows x 16 float4s
        const int row  = idx4 >> 4;     // 0..63
        const int k4   = idx4 & 15;     // 0..15
        float4 g4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (t0 + row < n_tokens) {
            g4 = ((const float4 *)(gate_j + (size_t)(t0 + row) * K))[k4];
        }
        const __half2 gh01 = __floats2half2_rn(g4.x, g4.y);
        const __half2 gh23 = __floats2half2_rn(g4.z, g4.w);
        *reinterpret_cast<uint2 *>(&s_gate[row * LDK + k4 * 4]) = make_uint2(bits_of(gh01), bits_of(gh23));
    }

    uint4 wf_reg[2];
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int idx8 = tid + p * 256; // 0..511 -> 64 rows x 8 uint4s
        const int row  = idx8 >> 3;     // 0..63
        const int k8   = idx8 & 7;      // 0..7
        uint4 wg8 = make_uint4(0, 0, 0, 0);
        if (c0 + row < N_EMBD) {
            wg8       = ((const uint4 *)(wg_j + (size_t)(c0 + row) * K))[k8];
            wf_reg[p] = ((const uint4 *)(wf_j + (size_t)(c0 + row) * K))[k8];
        } else {
            wf_reg[p] = make_uint4(0, 0, 0, 0);
        }
        *reinterpret_cast<uint4 *>(&s_w[row * LDK + k8 * 8]) = wg8;
    }
    __syncthreads();

    // 2. Compute dg = gate @ w_gain^T
    float8_t acc_g[2] = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
    };
    #pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
#if NIB_MMA_NATIVE
        const half16_t af = lds_frag(&s_gate[(wm * 16 + l16) * LDK + kc * 16]);
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            const half16_t bf = lds_frag(&s_w[(wn * 32 + f * 16 + l16) * LDK + kc * 16]);
            acc_g[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc_g[f]);
        }
#else
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            const half16_t bf = lds_frag(&s_w[(wn * 32 + f * 16 + l16) * LDK + kc * 16]);
            nib_mma::mma_stride(&s_gate[(wm * 16) * LDK + kc * 16], LDK, bf, acc_g[f], lane);
        }
#endif
    }
    __syncthreads();

    // 3. Stage prefetched w_forget -> s_w (direct uint4 copy!)
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int idx8 = tid + p * 256;
        const int row  = idx8 >> 3;
        const int k8   = idx8 & 7;
        *reinterpret_cast<uint4 *>(&s_w[row * LDK + k8 * 8]) = wf_reg[p];
    }
    __syncthreads();

    // 4. Compute df = gate @ w_forget^T
    float8_t acc_f[2] = {
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
        {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}
    };
    #pragma unroll
    for (int kc = 0; kc < 4; ++kc) {
#if NIB_MMA_NATIVE
        const half16_t af = lds_frag(&s_gate[(wm * 16 + l16) * LDK + kc * 16]);
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            const half16_t bf = lds_frag(&s_w[(wn * 32 + f * 16 + l16) * LDK + kc * 16]);
            acc_f[f] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf, acc_f[f]);
        }
#else
        #pragma unroll
        for (int f = 0; f < 2; ++f) {
            const half16_t bf = lds_frag(&s_w[(wn * 32 + f * 16 + l16) * LDK + kc * 16]);
            nib_mma::mma_stride(&s_gate[(wm * 16) * LDK + kc * 16], LDK, bf, acc_f[f], lane);
        }
#endif
    }

    // 5. Epilogue: tanh + blend (100% coalesced 64B half-warp global stores!)
    #pragma unroll
    for (int f = 0; f < 2; ++f) {
        const int c = c0 + wn * 32 + f * 16 + l16;
        if (c < N_EMBD) {
            #pragma unroll
            for (int idx = 0; idx < 8; ++idx) {
                const int t = t0 + wm * 16 + 2 * idx + half_w;
                if (t < n_tokens) {
                    const float gain   = tanhf(acc_g[f][idx]);
                    const float forget = tanhf(acc_f[f][idx]);
                    const size_t gidx  = (size_t)t * N_EMBD + c;
                    const float cv     = cur_j[gidx];
                    const float mv     = msg_j[gidx];
                    out_j[gidx] = cv - forget * cv + gain * mv;
                }
            }
        }
    }
#else
    (void)cur; (void)msg_base; (void)gate; (void)w_gain; (void)w_forget; (void)out; (void)n_tokens;
#endif
}

// Batched 3-way RMSNorm (in-place safe) for prefill QKV loop controllers
__launch_bounds__(576, 1)
static __global__ void rms_norm_2304_qkv3_kernel(
        MutPtr3 x_inout,
        Ptr3    w,
        int n_tokens,
        float eps) {
    constexpr int N_WARPS = 18;
    const int t = blockIdx.x;
    const int j = blockIdx.y;
    if (t >= n_tokens || j >= 3) {
        return;
    }
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    float * ptr = x_inout.p[j] + (size_t)t * N_EMBD;
    const float4 xv = ((const float4 *)ptr)[tid];
    float ssq = xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    ssq = warp_reduce_sum<WARP_SIZE>(ssq);

    __shared__ float smem[N_WARPS];
    if (lane_id == 0) {
        smem[warp_id] = ssq;
    }
    __syncthreads();

    __shared__ float s_inv;
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int i = 0; i < N_WARPS; ++i) {
            tot += smem[i];
        }
        s_inv = rsqrtf(tot * (1.0f / (float)N_EMBD) + eps);
    }
    __syncthreads();

    const float inv = s_inv;
    const float4 wv = w.p[j] ? ((const float4 *)w.p[j])[tid] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    float4 o;
    o.x = xv.x * inv * wv.x;
    o.y = xv.y * inv * wv.y;
    o.z = xv.z * inv * wv.z;
    o.w = xv.w * inv * wv.w;
    ((float4 *)ptr)[tid] = o;
}

// ============================================================================
// 4. RMSNorm
// ============================================================================
// Fast single-pass kernel for dim == 2304 (576 threads = 18 Wave32s, 1 float4/thread)
__launch_bounds__(576, 1)
static __global__ void rms_norm_2304_kernel(
        const float * __restrict__ x,
        const float * __restrict__ w,
        float       * __restrict__ out,
        int n_tokens,
        float eps) {
    constexpr int N_WARPS = 18;
    const int t = blockIdx.x;
    if (t >= n_tokens) {
        return;
    }
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const float4 xv = ((const float4 *)(x + (size_t)t * N_EMBD))[tid];
    float ssq = xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
    ssq = warp_reduce_sum<WARP_SIZE>(ssq);

    __shared__ float smem[N_WARPS];
    if (lane_id == 0) {
        smem[warp_id] = ssq;
    }
    __syncthreads();

    __shared__ float s_inv;
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int i = 0; i < N_WARPS; ++i) {
            tot += smem[i];
        }
        s_inv = rsqrtf(tot * (1.0f / (float)N_EMBD) + eps);
    }
    __syncthreads();

    const float inv = s_inv;
    const float4 wv = w ? ((const float4 *)w)[tid] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
    float4 o;
    o.x = xv.x * inv * wv.x;
    o.y = xv.y * inv * wv.y;
    o.z = xv.z * inv * wv.z;
    o.w = xv.w * inv * wv.w;
    ((float4 *)(out + (size_t)t * N_EMBD))[tid] = o;
}

__launch_bounds__(256, 4)
static __global__ void rms_norm_generic_kernel(
        const float * __restrict__ x,
        const float * __restrict__ w,
        float       * __restrict__ out,
        int n_tokens,
        int dim,
        float eps) {
    constexpr int N_WARPS = 8;
    const int t = blockIdx.x;
    if (t >= n_tokens) {
        return;
    }
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    const float * xt = x + (size_t)t * dim;
    float * ot       = out + (size_t)t * dim;

    float ssq = 0.0f;
    for (int i = tid; i < dim; i += 256) {
        const float v = xt[i];
        ssq += v * v;
    }
    ssq = warp_reduce_sum<WARP_SIZE>(ssq);

    __shared__ float smem[N_WARPS];
    if (lane_id == 0) {
        smem[warp_id] = ssq;
    }
    __syncthreads();

    __shared__ float s_inv;
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int i = 0; i < N_WARPS; ++i) {
            tot += smem[i];
        }
        s_inv = rsqrtf(tot / (float)dim + eps);
    }
    __syncthreads();

    const float inv = s_inv;
    for (int i = tid; i < dim; i += 256) {
        ot[i] = xt[i] * inv * (w ? w[i] : 1.0f);
    }
}

// ============================================================================
// 5. Super-Fused SConv-4 + Clip/RMSNorm + FWHT-128 + Q8_0/Q4_1/Q4_0 SoA KV Cache Store
// ============================================================================
static __device__ __forceinline__ uint32_t pack_u4x4_lo(int q0, int q1, int q2, int q3) {
    return ((uint32_t)(q0 & 0xF)) |
           ((uint32_t)(q1 & 0xF) << 8) |
           ((uint32_t)(q2 & 0xF) << 16) |
           ((uint32_t)(q3 & 0xF) << 24);
}

// Quantizes 128 FWHT-rotated floats (4 contiguous floats per lane across Wave32) directly into
// Head-Major SoA KV cache in Q8_0, Q4_1 (unsigned 0..15 + __half2(d, min)), or Q4_0 (signed -8..+7 + __half d).
// Word-interleaved 4-bit packing (Option C): each uint32_t holds 8 consecutive elements:
//   low nibbles of bytes 0..3 = elements 0..3, high nibbles of bytes 0..3 = elements 4..7.
static __device__ __forceinline__ void store_kv_quantized(
        float4 v, const KVCacheSlot & kv_slot, bool is_v, int h, int idx, int lane) {
    int8_t * dst_qs = is_v ? kv_slot.v_qs : kv_slot.k_qs;
    __half * dst_d  = is_v ? kv_slot.v_d  : kv_slot.k_d;
    const size_t head_slot = (size_t)h * (size_t)kv_slot.cap + (size_t)idx;
    const int grp = lane >> 3; // 0..3 (32-elem block)
    const int sub = lane & 7;  // 0..7 (8 lanes per 32-elem block)

    if (kv_slot.kv_type == KV_Q8_0) {
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        amax = fmaxf(amax, __shfl_xor(amax, 4, 8));
        amax = fmaxf(amax, __shfl_xor(amax, 2, 8));
        amax = fmaxf(amax, __shfl_xor(amax, 1, 8));

        const float  d_q  = amax * (1.0f / 127.0f);
        const float  id_q = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        const __half d_h  = __float2half(d_q);

        const int q0 = __float2int_rn(v.x * id_q);
        const int q1 = __float2int_rn(v.y * id_q);
        const int q2 = __float2int_rn(v.z * id_q);
        const int q3 = __float2int_rn(v.w * id_q);

        ((int32_t *)(dst_qs + head_slot * HEAD_DIM))[lane] = pack_i8x4(q0, q1, q2, q3);
        if (sub == 0) {
            dst_d[head_slot * 4 + grp] = d_h;
        }
    } else if (kv_slot.kv_type == KV_Q4_1) {
        float vmin = fminf(fminf(v.x, v.y), fminf(v.z, v.w));
        float vmax = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        vmin = fminf(vmin, __shfl_xor(vmin, 4, 8));
        vmin = fminf(vmin, __shfl_xor(vmin, 2, 8));
        vmin = fminf(vmin, __shfl_xor(vmin, 1, 8));
        vmax = fmaxf(vmax, __shfl_xor(vmax, 4, 8));
        vmax = fmaxf(vmax, __shfl_xor(vmax, 2, 8));
        vmax = fmaxf(vmax, __shfl_xor(vmax, 1, 8));

        const float span = vmax - vmin;
        const float d_q  = span * (1.0f / 15.0f);
        const float id_q = (span > 0.0f) ? (15.0f / span) : 0.0f;

        const int q0 = min(15, max(0, __float2int_rn((v.x - vmin) * id_q)));
        const int q1 = min(15, max(0, __float2int_rn((v.y - vmin) * id_q)));
        const int q2 = min(15, max(0, __float2int_rn((v.z - vmin) * id_q)));
        const int q3 = min(15, max(0, __float2int_rn((v.w - vmin) * id_q)));

        const uint32_t lo4 = pack_u4x4_lo(q0, q1, q2, q3);
        const uint32_t hi4 = __shfl_xor(lo4, 1, 32);
        if ((lane & 1) == 0) {
            ((uint32_t *)(dst_qs + head_slot * (HEAD_DIM / 2)))[lane >> 1] = lo4 | (hi4 << 4);
        }
        if (sub == 0) {
            ((__half2 *)dst_d)[head_slot * 4 + grp] = __floats2half2_rn(d_q, vmin);
        }
    } else {
        // KV_Q4_0: symmetric signed 4-bit [-8..+7] stored biased +8 in [0..15]
        float max_v = v.x;
        float amax  = fabsf(v.x);
        if (fabsf(v.y) > amax) { amax = fabsf(v.y); max_v = v.y; }
        if (fabsf(v.z) > amax) { amax = fabsf(v.z); max_v = v.z; }
        if (fabsf(v.w) > amax) { amax = fabsf(v.w); max_v = v.w; }

        #pragma unroll
        for (int mask = 4; mask >= 1; mask >>= 1) {
            const float o_amax = __shfl_xor(amax,  mask, 8);
            const float o_maxv = __shfl_xor(max_v, mask, 8);
            if (o_amax > amax || (o_amax == amax && o_maxv < max_v)) {
                amax  = o_amax;
                max_v = o_maxv;
            }
        }

        const float  d_q  = max_v * (-1.0f / 8.0f);
        const float  id_q = (d_q != 0.0f) ? (1.0f / d_q) : 0.0f;
        const __half d_h  = __float2half(d_q);

        const int q0 = min(15, max(0, __float2int_rn(v.x * id_q) + 8));
        const int q1 = min(15, max(0, __float2int_rn(v.y * id_q) + 8));
        const int q2 = min(15, max(0, __float2int_rn(v.z * id_q) + 8));
        const int q3 = min(15, max(0, __float2int_rn(v.w * id_q) + 8));

        const uint32_t lo4 = pack_u4x4_lo(q0, q1, q2, q3);
        const uint32_t hi4 = __shfl_xor(lo4, 1, 32);
        if ((lane & 1) == 0) {
            ((uint32_t *)(dst_qs + head_slot * (HEAD_DIM / 2)))[lane >> 1] = lo4 | (hi4 << 4);
        }
        if (sub == 0) {
            dst_d[head_slot * 4 + grp] = d_h;
        }
    }
}

// 1 Wave32 (32 lanes) per KV head (128 channels = 4 contiguous channels/lane).
// Single-pass channel-owned streaming across all t = 0..n_tokens-1 with inline sconv_hist update!
template <bool HAS_DYN, bool IS_V>
__launch_bounds__(32, 8)
static __global__ void sconv_kv_store_q8_kernel(
        const float    * __restrict__ raw_kv,
        int                           raw_stride,
        const float    * __restrict__ conv_w,
        const float    * __restrict__ dyn_taps,
        int                           dyn_stride,
        float          * __restrict__ sconv_hist,
        KVCacheSlot                   kv_slot,
        int                           n_tokens,
        int                           pos_start,
        const StepMeta * __restrict__ d_meta) {
    const int h    = blockIdx.x; // 0..7
    const int lane = threadIdx.x; // 0..31
    if (n_tokens <= 0) {
        return;
    }

    const int c_base = h * HEAD_DIM + lane * 4;

    // Load 4-tap conv weights for 4 channels
    float4 w[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j] = ((const float4 *)conv_w)[c_base + j];
    }

    // Load initial 3-tap history for 4 channels once into registers
    float x0[4], x1[4], x2[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float * hs = sconv_hist + (c_base + j) * N_CONV_HIST;
        x0[j] = hs[0];
        x1[j] = hs[1];
        x2[j] = hs[2];
    }

    const float conv_scale = HAS_DYN ? (1.0f / 3.0f) : 0.4472135954999579f; // 1/sqrt(9) vs 1/sqrt(5)

    for (int t = 0; t < n_tokens; ++t) {
        const float4 rt4 = *(const float4 *)(raw_kv + (size_t)t * raw_stride + c_base);
        const float x3[4] = { rt4.x, rt4.y, rt4.z, rt4.w };

        float4 d_dyn;
        if constexpr (HAS_DYN) {
            d_dyn = ((const float4 *)(dyn_taps + (size_t)t * dyn_stride))[h];
        }

        float y[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float acc = x3[j] + x0[j] * w[j].x + x1[j] * w[j].y + x2[j] * w[j].z + x3[j] * w[j].w;
            if constexpr (HAS_DYN) {
                acc += x3[j] * d_dyn.x + x2[j] * d_dyn.y + x1[j] * d_dyn.z + x0[j] * d_dyn.w;
            }
            y[j] = acc * conv_scale;
            x0[j] = x1[j];
            x1[j] = x2[j];
            x2[j] = x3[j];
        }

        // Step B: K -> L2-clip(QK_CLIP) + FWHT_128; V -> RMSNorm(RMS_EPS) + FWHT_128
        float4 v = make_float4(y[0], y[1], y[2], y[3]);
        float sumsq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        sumsq = warp_reduce_sum<WARP_SIZE>(sumsq);

        float factor;
        if constexpr (IS_V) {
            factor = rsqrtf(sumsq * (1.0f / (float)HEAD_DIM) + RMS_EPS) * KQ_SCALE;
        } else {
            factor = fminf(1.0f, QK_CLIP / (sqrtf(sumsq) + RMS_EPS)) * KQ_SCALE;
        }
        v.x *= factor;
        v.y *= factor;
        v.z *= factor;
        v.w *= factor;

        wave32_fwht_128_float4(v, lane);

        // Step C: Direct Q8_0 / Q4_1 / Q4_0 SoA quantization
        int32_t pos, idx;
        if (d_meta != nullptr) {
            pos = d_meta[t].pos;
            idx = kv_slot.is_swa ? d_meta[t].kv_write_swa : d_meta[t].kv_write_global;
        } else {
            pos = pos_start + t;
            idx = kv_slot.is_swa ? (pos % kv_slot.cap) : pos;
        }

        store_kv_quantized(v, kv_slot, IS_V, h, idx, lane);
        if constexpr (!IS_V) {
            if (h == 0 && lane == 0) {
                kv_slot.k_pos[idx] = pos;
            }
        }
    }

    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        float * hs = sconv_hist + (c_base + j) * N_CONV_HIST;
        hs[0] = x0[j];
        hs[1] = x1[j];
        hs[2] = x2[j];
    }
}

// Single-launch Dual K+V Time-Chunked ShortConv + Clip/RMSNorm + FWHT-128 + Q8_0/Q4_1/Q4_0 SoA KV Store:
// grid = (N_HEAD_KV=8, 2, n_tchunks), block = 32 (1 Wave32 per (KV head, K/V, time chunk)).
// At M=512 with T_CHUNK=16: 8 * 2 * 32 = 512 blocks = 5.33 warps/CU across all 96 CUs!
// Race-free history update: ONLY t_blk == 0 reads and updates sconv_hist (for n_tchunks > 1,
// t_blk == 0 reads raw_kv[n_tokens-3..n_tokens-1] after its loop and writes them to sconv_hist).
constexpr int SCONV_KV_TCHUNK = 16;

__launch_bounds__(32, 8)
static __global__ void sconv_kv_dual_prefill_kernel(
        const float * __restrict__ k_raw, int k_stride, const float * __restrict__ k_conv_w, const float * __restrict__ k_dyn, int k_dyn_stride,
        const float * __restrict__ v_raw, int v_stride, const float * __restrict__ v_conv_w, const float * __restrict__ v_dyn, int v_dyn_stride,
        float       * __restrict__ sconv_hist_kv,
        KVCacheSlot                kv_slot,
        int                        n_tokens,
        int                        pos_start) {
    const int  h     = blockIdx.x; // 0..7
    const bool is_v  = (blockIdx.y != 0);
    const int  t_blk = blockIdx.z;
    const int  lane  = threadIdx.x; // 0..31
    const int  t0    = t_blk * SCONV_KV_TCHUNK;
    if (t0 >= n_tokens) {
        return;
    }
    const int t1 = min(t0 + SCONV_KV_TCHUNK, n_tokens);

    const float * raw_kv     = is_v ? v_raw : k_raw;
    const int     raw_stride = is_v ? v_stride : k_stride;
    const float * conv_w     = is_v ? v_conv_w : k_conv_w;
    const float * dyn_taps   = is_v ? v_dyn : k_dyn;
    const int     dyn_stride = is_v ? v_dyn_stride : k_dyn_stride;
    float       * sconv_hist = is_v ? (sconv_hist_kv + (size_t)N_KV * N_CONV_HIST) : sconv_hist_kv;
    const bool    has_dyn    = (dyn_taps != nullptr);

    const int c_base = h * HEAD_DIM + lane * 4;

    float4 w[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j] = ((const float4 *)conv_w)[c_base + j];
    }

    float x0[4], x1[4], x2[4];
    if (t_blk == 0) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float * hs = sconv_hist + (c_base + j) * N_CONV_HIST;
            x0[j] = hs[0];
            x1[j] = hs[1];
            x2[j] = hs[2];
        }
    } else {
        // t0 >= 16 >= 3, so t0-3, t0-2, t0-1 are strictly in [0, n_tokens-1]
        const float4 r0 = *(const float4 *)(raw_kv + (size_t)(t0 - 3) * raw_stride + c_base);
        const float4 r1 = *(const float4 *)(raw_kv + (size_t)(t0 - 2) * raw_stride + c_base);
        const float4 r2 = *(const float4 *)(raw_kv + (size_t)(t0 - 1) * raw_stride + c_base);
        x0[0] = r0.x; x0[1] = r0.y; x0[2] = r0.z; x0[3] = r0.w;
        x1[0] = r1.x; x1[1] = r1.y; x1[2] = r1.z; x1[3] = r1.w;
        x2[0] = r2.x; x2[1] = r2.y; x2[2] = r2.z; x2[3] = r2.w;
    }

    const float conv_scale = has_dyn ? (1.0f / 3.0f) : 0.4472135954999579f;

    for (int t = t0; t < t1; ++t) {
        const float4 rt4 = *(const float4 *)(raw_kv + (size_t)t * raw_stride + c_base);
        const float x3[4] = { rt4.x, rt4.y, rt4.z, rt4.w };

        float4 d_dyn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (has_dyn) {
            d_dyn = ((const float4 *)(dyn_taps + (size_t)t * dyn_stride))[h];
        }

        float y[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float acc = x3[j] + x0[j] * w[j].x + x1[j] * w[j].y + x2[j] * w[j].z + x3[j] * w[j].w;
            if (has_dyn) {
                acc += x3[j] * d_dyn.x + x2[j] * d_dyn.y + x1[j] * d_dyn.z + x0[j] * d_dyn.w;
            }
            y[j] = acc * conv_scale;
            x0[j] = x1[j];
            x1[j] = x2[j];
            x2[j] = x3[j];
        }

        float4 v = make_float4(y[0], y[1], y[2], y[3]);
        float sumsq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        sumsq = warp_reduce_sum<WARP_SIZE>(sumsq);

        const float factor = is_v
            ? (rsqrtf(sumsq * (1.0f / (float)HEAD_DIM) + RMS_EPS) * KQ_SCALE)
            : (fminf(1.0f, QK_CLIP / (sqrtf(sumsq) + RMS_EPS)) * KQ_SCALE);
        v.x *= factor;
        v.y *= factor;
        v.z *= factor;
        v.w *= factor;

        wave32_fwht_128_float4(v, lane);

        const int32_t pos = pos_start + t;
        const int32_t idx = kv_slot.is_swa ? (pos % kv_slot.cap) : pos;

        store_kv_quantized(v, kv_slot, is_v, h, idx, lane);
        if (!is_v && h == 0 && lane == 0) {
            kv_slot.k_pos[idx] = pos;
        }
    }

    if (t_blk == 0) {
        if (t1 < n_tokens) {
            const float4 r0 = *(const float4 *)(raw_kv + (size_t)(n_tokens - 3) * raw_stride + c_base);
            const float4 r1 = *(const float4 *)(raw_kv + (size_t)(n_tokens - 2) * raw_stride + c_base);
            const float4 r2 = *(const float4 *)(raw_kv + (size_t)(n_tokens - 1) * raw_stride + c_base);
            x0[0] = r0.x; x0[1] = r0.y; x0[2] = r0.z; x0[3] = r0.w;
            x1[0] = r1.x; x1[1] = r1.y; x1[2] = r1.z; x1[3] = r1.w;
            x2[0] = r2.x; x2[1] = r2.y; x2[2] = r2.z; x2[3] = r2.w;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float * hs = sconv_hist + (c_base + j) * N_CONV_HIST;
            hs[0] = x0[j];
            hs[1] = x1[j];
            hs[2] = x2[j];
        }
    }
}

// ============================================================================
// 6. Q Clip + Tau + FWHT-128 AND R Clip + Tau + RelBias Table + Fused Decode
// ============================================================================
// 1 Wave32 per (token, head) -> 128 elements (4 per lane)
__launch_bounds__(128, 4)
static __global__ void q_prep_fwht128_kernel(
        const float    * __restrict__ q_in,
        int                           q_stride,
        const float    * __restrict__ q_clip_w,
        float          * __restrict__ q_out,
        bool                          apply_tau,
        int                           n_tokens,
        int                           pos_start,
        const StepMeta * __restrict__ d_meta) {
    const int warp_global = (blockIdx.x * (blockDim.x >> 5)) + (threadIdx.x >> 5);
    const int total_rows  = n_tokens * N_HEAD;
    if (warp_global >= total_rows) {
        return;
    }

    const int lane = threadIdx.x & 31;
    const int t    = warp_global / N_HEAD;
    const int h    = warp_global - t * N_HEAD;

    const float * qp = q_in + (size_t)t * q_stride + (size_t)h * HEAD_DIM;
    float4 v = ((const float4 *)qp)[lane];
    if (q_clip_w) {
        const float4 w = ((const float4 *)(q_clip_w + (size_t)h * HEAD_DIM))[lane];
        v.x *= w.x; v.y *= w.y; v.z *= w.z; v.w *= w.w;
    }

    float sumsq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    sumsq = warp_reduce_sum<WARP_SIZE>(sumsq);

    float tv = 1.0f;
    if (apply_tau) {
        if (d_meta != nullptr) {
            tv = d_meta[t].tau;
        } else {
            const float p = (float)(pos_start + t + 1) / LOG_SCALE_FLOOR;
            tv = 1.0f + LOG_SCALE_ALPHA * logf(fmaxf(1.0f, p));
        }
    }

    const float factor = fminf(1.0f, QK_CLIP / (sqrtf(sumsq) + RMS_EPS)) * tv * KQ_SCALE;
    v.x *= factor;
    v.y *= factor;
    v.z *= factor;
    v.w *= factor;

    wave32_fwht_128_float4(v, lane);

    ((float4 *)(q_out + (size_t)warp_global * HEAD_DIM))[lane] = v;
}

// Fast decode r_prep_reltable: grid = 32 blocks, block = 128 threads.
// Each block owns 32 consecutive distances, loads R[24][16] & rel_bias[32][16] with 1 sync,
// and computes all 24 heads x 32 distances with ZERO inner __syncthreads()!
__launch_bounds__(128, 4)
static __global__ void r_prep_reltable_decode_kernel(
        const float    * __restrict__ r_in,
        const float    * __restrict__ r_clip_w,
        const float    * __restrict__ rel_bias_w,
        float          * __restrict__ rel_table_out,
        bool                          apply_tau,
        int                           pos_start,
        const StepMeta * __restrict__ d_meta) {
    const int d0      = blockIdx.x * 32;
    const int tid     = threadIdx.x; // 0..127
    const int warp_id = tid >> 5;    // 0..3
    const int lane    = tid & 31;    // 0..31

    __shared__ alignas(16) float s_r[N_HEAD][N_REL];
    __shared__ alignas(16) float s_b[32][N_REL + 1];

    float tv = 1.0f;
    if (apply_tau) {
        if (d_meta != nullptr) {
            tv = d_meta->tau;
        } else {
            const float p = (float)(pos_start + 1) / LOG_SCALE_FLOOR;
            tv = 1.0f + LOG_SCALE_ALPHA * logf(fmaxf(1.0f, p));
        }
    }

    #pragma unroll
    for (int step = 0; step < 3; ++step) {
        const int h = step * 8 + (tid >> 4);
        const int k = tid & 15;
        float rv = r_in[(size_t)h * N_REL + k];
        if (r_clip_w) {
            rv *= r_clip_w[(size_t)h * N_REL + k];
        }
        const float sumsq = warp_reduce_sum<N_REL>(rv * rv);
        const float factor = fminf(1.0f, REL_CLIP / (sqrtf(sumsq) + RMS_EPS)) * tv;
        s_r[h][k] = rv * factor;
    }

    {
        const int dl = tid >> 2; // 0..31
        const int k4 = tid & 3;  // 0..3
        const float4 b4 = ((const float4 *)(rel_bias_w + (size_t)(d0 + dl) * N_REL))[k4];
        s_b[dl][k4 * 4 + 0] = b4.x;
        s_b[dl][k4 * 4 + 1] = b4.y;
        s_b[dl][k4 * 4 + 2] = b4.z;
        s_b[dl][k4 * 4 + 3] = b4.w;
    }
    __syncthreads();

    float b[N_REL];
    #pragma unroll
    for (int k = 0; k < N_REL; ++k) {
        b[k] = s_b[lane][k];
    }

    #pragma unroll
    for (int hh = 0; hh < 6; ++hh) {
        const int h = warp_id * 6 + hh;
        float dot = 0.0f;
        #pragma unroll
        for (int k = 0; k < N_REL; ++k) {
            dot = fmaf(b[k], s_r[h][k], dot);
        }
        rel_table_out[(size_t)h * N_REL_EXTENT + d0 + lane] = dot;
    }
}

// Fast 4x-vectorized prefill r_prep_reltable:
// Clamps distances to max_dist = min(pos_start + n_tokens, N_REL_EXTENT).
// Each block of 128 threads (4 warps) handles BR = 64 (t, h) rows x BD = 128 distances:
//   - Warp w (0..3) owns 16 rows (r_base = w * 16 .. w * 16 + 15).
//   - Lane l (0..31) owns 4 contiguous distances (d_loc = l * 4 .. l * 4 + 3) in registers!
//   - Inner loop reads s_r[r][k] via uniform warp broadcast (0 bank conflicts!) and writes 128-bit float4 coalesced stores!
__launch_bounds__(128, 4)
static __global__ void r_prep_reltable_prefill_kernel(
        const float    * __restrict__ r_in,
        int                           r_stride,
        const float    * __restrict__ r_clip_w,
        const float    * __restrict__ rel_bias_w,
        float          * __restrict__ rel_table_out,
        bool                          apply_tau,
        int                           n_tokens,
        int                           pos_start,
        int                           max_dist,
        const StepMeta * __restrict__ d_meta) {
    constexpr int BR = 64;
    constexpr int BD = 128;

    const int d0         = blockIdx.x * BD;
    const int row0       = blockIdx.y * BR;
    const int total_rows = n_tokens * N_HEAD;
    if (d0 >= max_dist || row0 >= total_rows) {
        return;
    }

    const int tid     = threadIdx.x; // 0..127
    const int warp_id = tid >> 5;    // 0..3
    const int lane    = tid & 31;    // 0..31

    __shared__ alignas(16) float s_r[BR][N_REL + 1];

    // 1. Load & clip 64 (t, h) rows of R (8 rows/step x 8 steps = 64 rows)
    #pragma unroll
    for (int step = 0; step < 8; ++step) {
        const int r_loc = step * 8 + (tid >> 4);
        const int k     = tid & 15;
        const int row   = row0 + r_loc;
        if (row < total_rows) {
            const int t = row / N_HEAD;
            const int h = row - t * N_HEAD;
            float rv = r_in[(size_t)t * r_stride + (size_t)h * N_REL + k];
            if (r_clip_w) {
                rv *= r_clip_w[(size_t)h * N_REL + k];
            }
            const float sumsq = warp_reduce_sum<N_REL>(rv * rv);
            float tv = 1.0f;
            if (apply_tau) {
                if (d_meta != nullptr) {
                    tv = d_meta[t].tau;
                } else {
                    const float p = (float)(pos_start + t + 1) / LOG_SCALE_FLOOR;
                    tv = 1.0f + LOG_SCALE_ALPHA * logf(fmaxf(1.0f, p));
                }
            }
            const float factor = fminf(1.0f, REL_CLIP / (sqrtf(sumsq) + RMS_EPS)) * tv;
            s_r[r_loc][k] = rv * factor;
        }
    }
    __syncthreads();

    // 2. Each lane loads its 4 contiguous distances' rel_bias_w[d0 + lane*4 + j][0..15] directly into registers!
    const int dl0 = lane * 4;
    const int d_base = d0 + dl0;
    if (d_base >= max_dist) {
        return;
    }

    float4 b_k[N_REL]; // b_k[k] holds (b[d+0][k], b[d+1][k], b[d+2][k], b[d+3][k])
    #pragma unroll
    for (int k = 0; k < N_REL; ++k) {
        b_k[k] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int d = d_base + j;
        if (d < max_dist) {
            const float4 * bp = (const float4 *)(rel_bias_w + (size_t)d * N_REL);
            const float4 v0 = bp[0];
            const float4 v1 = bp[1];
            const float4 v2 = bp[2];
            const float4 v3 = bp[3];
            (&b_k[ 0].x)[j] = v0.x; (&b_k[ 1].x)[j] = v0.y; (&b_k[ 2].x)[j] = v0.z; (&b_k[ 3].x)[j] = v0.w;
            (&b_k[ 4].x)[j] = v1.x; (&b_k[ 5].x)[j] = v1.y; (&b_k[ 6].x)[j] = v1.z; (&b_k[ 7].x)[j] = v1.w;
            (&b_k[ 8].x)[j] = v2.x; (&b_k[ 9].x)[j] = v2.y; (&b_k[10].x)[j] = v2.z; (&b_k[11].x)[j] = v2.w;
            (&b_k[12].x)[j] = v3.x; (&b_k[13].x)[j] = v3.y; (&b_k[14].x)[j] = v3.z; (&b_k[15].x)[j] = v3.w;
        }
    }

    // 3. Warp warp_id computes its 16 rows (or up to total_rows) with 128-bit float4 coalesced stores!
    const int r_start = warp_id * 16;
    const int r_end   = min(r_start + 16, total_rows - row0);
    if (d_base + 3 < max_dist) {
        for (int r_loc = r_start; r_loc < r_end; ++r_loc) {
            float4 dot = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int k = 0; k < N_REL; ++k) {
                const float rv = s_r[r_loc][k];
                dot.x = fmaf(b_k[k].x, rv, dot.x);
                dot.y = fmaf(b_k[k].y, rv, dot.y);
                dot.z = fmaf(b_k[k].z, rv, dot.z);
                dot.w = fmaf(b_k[k].w, rv, dot.w);
            }
            *(float4 *)(rel_table_out + (size_t)(row0 + r_loc) * N_REL_EXTENT + d_base) = dot;
        }
    } else {
        for (int r_loc = r_start; r_loc < r_end; ++r_loc) {
            float4 dot = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int k = 0; k < N_REL; ++k) {
                const float rv = s_r[r_loc][k];
                dot.x = fmaf(b_k[k].x, rv, dot.x);
                dot.y = fmaf(b_k[k].y, rv, dot.y);
                dot.z = fmaf(b_k[k].z, rv, dot.z);
                dot.w = fmaf(b_k[k].w, rv, dot.w);
            }
            float * dst_row = rel_table_out + (size_t)(row0 + r_loc) * N_REL_EXTENT;
            if (d_base + 0 < max_dist) dst_row[d_base + 0] = dot.x;
            if (d_base + 1 < max_dist) dst_row[d_base + 1] = dot.y;
            if (d_base + 2 < max_dist) dst_row[d_base + 2] = dot.z;
        }
    }
}

// Super-Fused Multi-CTA Decode Kernel (72 blocks x 128 threads):
//   0..23:  Q prep (q_clip_w -> L2 clip -> tau -> FWHT-128 -> q_out)
//   24..31: K sconv_kv_store_q8 (Head-Major KV + k_pos + sconv_hist)
//   32..39: V sconv_kv_store_q8 (Head-Major KV + sconv_hist)
//   40..71: R prep + rel_table (32 blocks x 32 distances, 1 sync, 0 inner syncs)
__launch_bounds__(128, 4)
static __global__ void qkvr_prep_fused_decode_kernel(
        const float    * __restrict__ q_in,
        const float    * __restrict__ q_clip_w,
        const float    * __restrict__ r_in,
        const float    * __restrict__ r_clip_w,
        const float    * __restrict__ rel_bias_w,
        const float    * __restrict__ k_raw,
        const float    * __restrict__ k_conv_w,
        const float    * __restrict__ k_dyn,
        const float    * __restrict__ v_raw,
        const float    * __restrict__ v_conv_w,
        const float    * __restrict__ v_dyn,
        float          * __restrict__ sconv_hist_kv,
        KVCacheSlot                   kv_slot,
        float          * __restrict__ q_out,
        float          * __restrict__ rel_table_out,
        bool                          apply_tau,
        const StepMeta * __restrict__ d_meta) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    if (bid < 24) {
        // --- Q prep (1 Wave32 active) ---
        if (tid >= 32) {
            return;
        }
        const int h    = bid;
        const int lane = tid;
        float4 v = ((const float4 *)(q_in + (size_t)h * HEAD_DIM))[lane];
        if (q_clip_w) {
            const float4 w = ((const float4 *)(q_clip_w + (size_t)h * HEAD_DIM))[lane];
            v.x *= w.x; v.y *= w.y; v.z *= w.z; v.w *= w.w;
        }
        float sumsq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        sumsq = warp_reduce_sum<WARP_SIZE>(sumsq);
        const float tv = (apply_tau && d_meta != nullptr) ? d_meta->tau : 1.0f;
        const float factor = fminf(1.0f, QK_CLIP / (sqrtf(sumsq) + RMS_EPS)) * tv * KQ_SCALE;
        v.x *= factor; v.y *= factor; v.z *= factor; v.w *= factor;
        wave32_fwht_128_float4(v, lane);
        ((float4 *)(q_out + (size_t)h * HEAD_DIM))[lane] = v;
    } else if (bid < 40) {
        // --- K (24..31) and V (32..39) sconv + KV store (1 Wave32 active) ---
        if (tid >= 32) {
            return;
        }
        const bool is_v = (bid >= 32);
        const int  h    = is_v ? (bid - 32) : (bid - 24);
        const int  lane = tid;
        const int  c_base = h * HEAD_DIM + lane * 4;

        const float * raw_ptr  = is_v ? v_raw : k_raw;
        const float * cw_ptr   = is_v ? v_conv_w : k_conv_w;
        const float * dyn_ptr  = is_v ? v_dyn : k_dyn;
        float       * hist_ptr = is_v ? (sconv_hist_kv + (size_t)N_KV * N_CONV_HIST) : sconv_hist_kv;
        const bool    has_dyn  = (dyn_ptr != nullptr);

        float4 w[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            w[j] = ((const float4 *)cw_ptr)[c_base + j];
        }

        const float * rt = raw_ptr + c_base;
        float x3[4] = { rt[0], rt[1], rt[2], rt[3] };

        float4 d_dyn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (has_dyn) {
            d_dyn = ((const float4 *)dyn_ptr)[h];
        }
        const float conv_scale = has_dyn ? (1.0f / 3.0f) : 0.4472135954999579f;

        float y[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float * hs = hist_ptr + (size_t)(c_base + j) * N_CONV_HIST;
            const float x0 = hs[0];
            const float x1 = hs[1];
            const float x2 = hs[2];
            float acc = x3[j] + x0 * w[j].x + x1 * w[j].y + x2 * w[j].z + x3[j] * w[j].w;
            if (has_dyn) {
                acc += x3[j] * d_dyn.x + x2 * d_dyn.y + x1 * d_dyn.z + x0 * d_dyn.w;
            }
            y[j] = acc * conv_scale;
            hs[0] = x1;
            hs[1] = x2;
            hs[2] = x3[j];
        }

        float4 v = make_float4(y[0], y[1], y[2], y[3]);
        float sumsq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        sumsq = warp_reduce_sum<WARP_SIZE>(sumsq);

        const float factor = is_v
            ? (rsqrtf(sumsq * (1.0f / (float)HEAD_DIM) + RMS_EPS) * KQ_SCALE)
            : (fminf(1.0f, QK_CLIP / (sqrtf(sumsq) + RMS_EPS)) * KQ_SCALE);
        v.x *= factor; v.y *= factor; v.z *= factor; v.w *= factor;

        wave32_fwht_128_float4(v, lane);

        const int32_t pos = (d_meta != nullptr) ? d_meta->pos : 0;
        const int32_t idx = (d_meta != nullptr)
            ? (kv_slot.is_swa ? d_meta->kv_write_swa : d_meta->kv_write_global)
            : 0;

        store_kv_quantized(v, kv_slot, is_v, h, idx, lane);
        if (!is_v && h == 0 && lane == 0) {
            kv_slot.k_pos[idx] = pos;
        }
    } else {
        // --- R prep + rel_table (40..71, all 128 threads active) ---
        const int d0      = (bid - 40) * 32;
        const int warp_id = tid >> 5;
        const int lane    = tid & 31;

        __shared__ alignas(16) float s_r[N_HEAD][N_REL];
        __shared__ alignas(16) float s_b[32][N_REL + 1];

        const float tv = (apply_tau && d_meta != nullptr) ? d_meta->tau : 1.0f;

        #pragma unroll
        for (int step = 0; step < 3; ++step) {
            const int h = step * 8 + (tid >> 4);
            const int k = tid & 15;
            float rv = r_in[(size_t)h * N_REL + k];
            if (r_clip_w) {
                rv *= r_clip_w[(size_t)h * N_REL + k];
            }
            const float sumsq = warp_reduce_sum<N_REL>(rv * rv);
            const float factor = fminf(1.0f, REL_CLIP / (sqrtf(sumsq) + RMS_EPS)) * tv;
            s_r[h][k] = rv * factor;
        }

        {
            const int dl = tid >> 2;
            const int k4 = tid & 3;
            const float4 b4 = ((const float4 *)(rel_bias_w + (size_t)(d0 + dl) * N_REL))[k4];
            s_b[dl][k4 * 4 + 0] = b4.x;
            s_b[dl][k4 * 4 + 1] = b4.y;
            s_b[dl][k4 * 4 + 2] = b4.z;
            s_b[dl][k4 * 4 + 3] = b4.w;
        }
        __syncthreads();

        float b[N_REL];
        #pragma unroll
        for (int k = 0; k < N_REL; ++k) {
            b[k] = s_b[lane][k];
        }

        #pragma unroll
        for (int hh = 0; hh < 6; ++hh) {
            const int h = warp_id * 6 + hh;
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < N_REL; ++k) {
                dot = fmaf(b[k], s_r[h][k], dot);
            }
            rel_table_out[(size_t)h * N_REL_EXTENT + d0 + lane] = dot;
        }
    }
}

// ============================================================================
// 7. SConv-4 Static + Optional Residual Add (2304 channels = 576 float4s)
//    Single-pass 128-bit float4 channel-owned streaming across t = 0..n_tokens-1
//    with inline history update (eliminates sconv_hist_update_kernel!).
// ============================================================================
constexpr int SCONV_POST_TCHUNK = 16;

template <bool HAS_RES>
__launch_bounds__(32, 8)
static __global__ void sconv_post_res_kernel(
        const float * __restrict__ x,
        const float * __restrict__ conv_w,
        float       * __restrict__ sconv_hist,
        const float * __restrict__ res,
        float       * __restrict__ out,
        int n_tokens,
        int t_chunk) {
    const int i4 = blockIdx.x * blockDim.x + threadIdx.x; // 0..575
    if (i4 >= (N_EMBD / 4) || n_tokens <= 0) {
        return;
    }
    const int t_blk = blockIdx.y;
    const int t0    = t_blk * t_chunk;
    if (t0 >= n_tokens) {
        return;
    }
    const int t1     = min(t0 + t_chunk, n_tokens);
    const int c_base = i4 * 4;

    float4 w[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        w[j] = ((const float4 *)conv_w)[c_base + j];
    }

    float x0[4], x1[4], x2[4];
    if (t_blk == 0) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float * hs = sconv_hist + (size_t)(c_base + j) * N_CONV_HIST;
            x0[j] = hs[0];
            x1[j] = hs[1];
            x2[j] = hs[2];
        }
    } else {
        // t0 >= t_chunk >= 16 >= 3, so t0-3, t0-2, t0-1 are strictly in [0, n_tokens-1]
        const float4 r0 = ((const float4 *)x)[(size_t)(t0 - 3) * (N_EMBD / 4) + i4];
        const float4 r1 = ((const float4 *)x)[(size_t)(t0 - 2) * (N_EMBD / 4) + i4];
        const float4 r2 = ((const float4 *)x)[(size_t)(t0 - 1) * (N_EMBD / 4) + i4];
        x0[0] = r0.x; x0[1] = r0.y; x0[2] = r0.z; x0[3] = r0.w;
        x1[0] = r1.x; x1[1] = r1.y; x1[2] = r1.z; x1[3] = r1.w;
        x2[0] = r2.x; x2[1] = r2.y; x2[2] = r2.z; x2[3] = r2.w;
    }

    constexpr float scale = 0.4472135954999579f; // 1 / sqrt(5)

    for (int t = t0; t < t1; ++t) {
        const size_t off4 = (size_t)t * (N_EMBD / 4) + i4;
        const float4 x3v  = ((const float4 *)x)[off4];
        const float x3[4] = { x3v.x, x3v.y, x3v.z, x3v.w };
        float y[4];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float acc = x3[j] + x0[j] * w[j].x + x1[j] * w[j].y + x2[j] * w[j].z + x3[j] * w[j].w;
            y[j]  = acc * scale;
            x0[j] = x1[j];
            x1[j] = x2[j];
            x2[j] = x3[j];
        }
        float4 ov = make_float4(y[0], y[1], y[2], y[3]);
        if constexpr (HAS_RES) {
            const float4 rv = ((const float4 *)res)[off4];
            ov.x += rv.x;
            ov.y += rv.y;
            ov.z += rv.z;
            ov.w += rv.w;
        }
        ((float4 *)out)[off4] = ov;
    }

    if (t_blk == 0) {
        if (t1 < n_tokens) {
            const float4 r0 = ((const float4 *)x)[(size_t)(n_tokens - 3) * (N_EMBD / 4) + i4];
            const float4 r1 = ((const float4 *)x)[(size_t)(n_tokens - 2) * (N_EMBD / 4) + i4];
            const float4 r2 = ((const float4 *)x)[(size_t)(n_tokens - 1) * (N_EMBD / 4) + i4];
            x0[0] = r0.x; x0[1] = r0.y; x0[2] = r0.z; x0[3] = r0.w;
            x1[0] = r1.x; x1[1] = r1.y; x1[2] = r1.z; x1[3] = r1.w;
            x2[0] = r2.x; x2[1] = r2.y; x2[2] = r2.z; x2[3] = r2.w;
        }
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            float * hs = sconv_hist + (size_t)(c_base + j) * N_CONV_HIST;
            hs[0] = x0[j];
            hs[1] = x1[j];
            hs[2] = x2[j];
        }
    }
}

// ============================================================================
// 8. Fused FFN RMSNorm + Router F32 GEMV/GEMM + Sigmoid + Top-8 + Normalize
// ============================================================================
// Deterministic Top-8 selection across 128 experts by one Wave32 (4 experts/lane)
static __device__ __forceinline__ void wave32_topk8_128(
        const float * s_sel,
        const float * s_p,
        int32_t * topk_ids_out,
        float   * topk_w_out,
        const int lane) {
    float my_sel[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        my_sel[j] = s_sel[lane + j * 32];
    }

    float   top_p[TOP_K];
    int32_t top_i[TOP_K];

    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        float   best_v = -INFINITY;
        int32_t best_i = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + j * 32;
            const float v = my_sel[j];
            if (v > best_v || (v == best_v && idx < best_i)) {
                best_v = v;
                best_i = idx;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float   ov = __shfl_xor(best_v, off, WARP_SIZE);
            const int32_t oi = __shfl_xor(best_i, off, WARP_SIZE);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        // All lanes now hold the exact same best_i
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (lane + j * 32 == best_i) {
                my_sel[j] = -INFINITY;
            }
        }
        if (lane == 0) {
            top_i[k] = best_i;
            top_p[k] = s_p[best_i];
        }
    }

    if (lane == 0) {
        float sum_p = 0.0f;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            sum_p += top_p[k];
        }
        const float scale = EXPERT_W_SCALE / (sum_p + EXPERT_NORM_EPS);
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            topk_ids_out[k] = top_i[k];
            topk_w_out[k]   = top_p[k] * scale;
        }
    }
}

[[maybe_unused]] static __device__ __forceinline__ void wave32_topk8_128_bias(
        const float * s_p,
        const float * exp_probs_b,
        int32_t * topk_ids_out,
        float   * topk_w_out,
        const int lane) {
    float my_p[4];
    float my_sel[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int idx = lane + j * 32;
        const float p = s_p[idx];
        my_p[j]   = p;
        my_sel[j] = exp_probs_b ? (p + exp_probs_b[idx]) : p;
    }

    float   top_p[TOP_K];
    int32_t top_i[TOP_K];

    #pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        float   best_v = -INFINITY;
        int32_t best_i = 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + j * 32;
            const float v = my_sel[j];
            if (v > best_v || (v == best_v && idx < best_i)) {
                best_v = v;
                best_i = idx;
            }
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float   ov = __shfl_xor(best_v, off, WARP_SIZE);
            const int32_t oi = __shfl_xor(best_i, off, WARP_SIZE);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
        float chosen_p = 0.0f;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (lane + j * 32 == best_i) {
                chosen_p  = my_p[j];
                my_sel[j] = -INFINITY;
            }
        }
        const float p_bcast = __shfl(chosen_p, best_i & 31, WARP_SIZE);
        if (lane == 0) {
            top_i[k] = best_i;
            top_p[k] = p_bcast;
        }
    }

    if (lane == 0) {
        float sum_p = 0.0f;
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            sum_p += top_p[k];
        }
        const float scale = EXPERT_W_SCALE / (sum_p + EXPERT_NORM_EPS);
        #pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            topk_ids_out[k] = top_i[k];
            topk_w_out[k]   = top_p[k] * scale;
        }
    }
}

// Decode / small-batch fused kernel (n_tokens <= 16):
// 1 block of 512 threads (16 Wave32s) per token.
// Reads x_in once, writes x_norm_out & LDS, computes 128 FP16 router logits (8 experts/warp), Top-8.
__launch_bounds__(512, 1)
static __global__ void ffn_norm_router_topk_decode_kernel(
        const float  * __restrict__ x_in,
        const float  * __restrict__ ffn_norm_w,
        const __half * __restrict__ router_w,
        const float  * __restrict__ exp_probs_b,
        float        * __restrict__ x_norm_out,
        int32_t      * __restrict__ topk_ids_out,
        float        * __restrict__ topk_w_out,
        int n_tokens) {
    constexpr int N_WARPS = 16;
    constexpr int N4      = N_EMBD / 4; // 576
    constexpr int N8      = N_EMBD / 8; // 288
    const int t = blockIdx.x;
    if (t >= n_tokens) {
        return;
    }

    const int tid     = threadIdx.x; // 0..511
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    __shared__ float  smem_ssq[N_WARPS];
    __shared__ float4 s_xnorm4[N4];
    __shared__ float  s_sel[N_EXPERT];
    __shared__ float  s_p[N_EXPERT];

    const float * xt = x_in + (size_t)t * N_EMBD;
    float       * xo = x_norm_out + (size_t)t * N_EMBD;

    float4 xv_reg[2];
    float ssq = 0.0f;
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int i = tid + p * 512;
        if (i < N4) {
            const float4 xv = ((const float4 *)xt)[i];
            xv_reg[p] = xv;
            ssq += xv.x * xv.x + xv.y * xv.y + xv.z * xv.z + xv.w * xv.w;
        }
    }
    ssq = warp_reduce_sum<WARP_SIZE>(ssq);
    if (lane == 0) {
        smem_ssq[warp_id] = ssq;
    }
    __syncthreads();

    __shared__ float s_inv;
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            tot += smem_ssq[w];
        }
        s_inv = rsqrtf(tot * (1.0f / (float)N_EMBD) + RMS_EPS);
    }
    __syncthreads();

    const float inv = s_inv;
    #pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int i = tid + p * 512;
        if (i < N4) {
            const float4 wv = ((const float4 *)ffn_norm_w)[i];
            float4 nv;
            nv.x = xv_reg[p].x * inv * wv.x;
            nv.y = xv_reg[p].y * inv * wv.y;
            nv.z = xv_reg[p].z * inv * wv.z;
            nv.w = xv_reg[p].w * inv * wv.w;
            s_xnorm4[i] = nv;
            ((float4 *)xo)[i] = nv;
        }
    }
    __syncthreads();

    // 16 warps, 8 experts per warp, 288 uint4s / 32 lanes = 9 exact uint4s/lane
    const int e_base = warp_id * 8;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint4 * rw8 = (const uint4 *)router_w;

    #pragma unroll 1
    for (int step = 0; step < 9; ++step) {
        const int i8 = lane + step * WARP_SIZE;
        const float4 x0 = s_xnorm4[i8 * 2 + 0];
        const float4 x1 = s_xnorm4[i8 * 2 + 1];
        #pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint4 wv = rw8[(size_t)(e_base + k) * N8 + i8];
            acc[k] = dot_h8_f8(wv, x0, x1, acc[k]);
        }
        __builtin_amdgcn_sched_barrier(0);
    }

    #pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float sum = warp_reduce_sum<WARP_SIZE>(acc[k]);
        if (lane == 0) {
            const int e   = e_base + k;
            const float p = 1.0f / (1.0f + expf(-sum));
            s_p[e]   = p;
            s_sel[e] = exp_probs_b ? (p + exp_probs_b[e]) : p;
        }
    }
    __syncthreads();

    if (warp_id == 0) {
        wave32_topk8_128(s_sel, s_p, topk_ids_out + (size_t)t * TOP_K, topk_w_out + (size_t)t * TOP_K, lane);
    }
}

// Multi-WGP Single-Launch Decode Router (n_tokens == 1, router_scratch != nullptr):
// 32 blocks x 128 threads (4 Wave32s/block, 1 expert/warp -> 128 experts across 32 WGPs).
// Uses self-resetting atomic completion counter at router_scratch[256] (100% HIP-Graph safe!).
__launch_bounds__(128, 4)
static __global__ void ffn_norm_router_topk_multiwgp_kernel(
        const float  * __restrict__ x_in,
        const float  * __restrict__ ffn_norm_w,
        const __half * __restrict__ router_w,
        const float  * __restrict__ exp_probs_b,
        float        * __restrict__ x_norm_out,
        int32_t      * __restrict__ topk_ids_out,
        float        * __restrict__ topk_w_out,
        float        * __restrict__ router_scratch) {
    constexpr int N_WARPS = 4;
    constexpr int N4      = N_EMBD / 4; // 576

    const int bid     = blockIdx.x;  // 0..31
    const int tid     = threadIdx.x; // 0..127
    const int warp_id = tid >> 5;    // 0..3
    const int lane    = tid & 31;    // 0..31

    __shared__ float  smem_ssq[N_WARPS];
    __shared__ float4 s_xnorm4[N4];
    __shared__ float  s_inv;

    // 1. Load x_in once into registers & compute RMSNorm sum-of-squares
    float4 xv[5];
    float ssq = 0.0f;
    #pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int i = tid + k * 128;
        if (i < N4) {
            const float4 v = ((const float4 *)x_in)[i];
            xv[k] = v;
            ssq += v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
        }
    }
    ssq = warp_reduce_sum<WARP_SIZE>(ssq);
    if (lane == 0) {
        smem_ssq[warp_id] = ssq;
    }
    __syncthreads();

    if (tid == 0) {
        float tot = smem_ssq[0] + smem_ssq[1] + smem_ssq[2] + smem_ssq[3];
        s_inv = rsqrtf(tot * (1.0f / (float)N_EMBD) + RMS_EPS);
    }
    __syncthreads();

    // 2. Normalize into LDS s_xnorm4 (and block 0 writes x_norm_out)
    const float inv = s_inv;
    #pragma unroll
    for (int k = 0; k < 5; ++k) {
        const int i = tid + k * 128;
        if (i < N4) {
            const float4 wv = ((const float4 *)ffn_norm_w)[i];
            float4 nv;
            nv.x = xv[k].x * inv * wv.x;
            nv.y = xv[k].y * inv * wv.y;
            nv.z = xv[k].z * inv * wv.z;
            nv.w = xv[k].w * inv * wv.w;
            s_xnorm4[i] = nv;
            if (bid == 0) {
                ((float4 *)x_norm_out)[i] = nv;
            }
        }
    }
    __syncthreads();

    // 3. Each warp dots 1 expert row of FP16 router_w (288 uint4s / 32 lanes = 9 exact steps)
    const int e = bid * 4 + warp_id; // 0..127
    const uint4 * rw8 = (const uint4 *)(router_w + (size_t)e * N_EMBD);
    float acc = 0.0f;
    #pragma unroll
    for (int step = 0; step < 9; ++step) {
        const int i8 = lane + step * WARP_SIZE;
        const float4 x0 = s_xnorm4[i8 * 2 + 0];
        const float4 x1 = s_xnorm4[i8 * 2 + 1];
        const uint4  w  = rw8[i8];
        acc = dot_h8_f8(w, x0, x1, acc);
    }
    const float sum = warp_reduce_sum<WARP_SIZE>(acc);
    if (lane == 0) {
        const float p   = 1.0f / (1.0f + expf(-sum));
        const float sel = exp_probs_b ? (p + exp_probs_b[e]) : p;
        __hip_atomic_store(&router_scratch[e],            p,   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&router_scratch[N_EXPERT + e], sel, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    rdna3_flush_stores();
    __syncthreads();

    // 4. Last-block self-resetting atomic completion barrier + Top-8
    __shared__ bool s_is_last;
    if (tid == 0) {
        rdna3_flush_stores();
        int32_t * done_ptr = reinterpret_cast<int32_t *>(router_scratch + 256);
        const int prev = __hip_atomic_fetch_add(done_ptr, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_is_last = (prev == 31);
        if (s_is_last) {
            atomicExch(done_ptr, 0);
        }
    }
    __syncthreads();

    if (s_is_last && warp_id == 0) {
        rdna3_inv_l0_l1();
        __shared__ float lds_p[N_EXPERT];
        __shared__ float lds_sel[N_EXPERT];
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + j * 32;
            lds_p[idx]   = __hip_atomic_load(&router_scratch[idx],            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            lds_sel[idx] = __hip_atomic_load(&router_scratch[N_EXPERT + idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        wave32_topk8_128(lds_sel, lds_p, topk_ids_out, topk_w_out, lane);
    }
}

// ============================================================================
// Split decode FFN prep: (A) fused ctrl_apply + FFN RMSNorm (one node)
//                        (B) standalone FP16 router + sigmoid + Top-8 (one node)
// Both avoid the pathological 128-block atomic spin-wait barrier of the old fused
// kernels (was 16-18 us/call). Node count is unchanged (2 nodes per call).
// ============================================================================

// (A) ctrl_apply (loop layers) + ffn RMSNorm, one launch.
// grid = 72 blocks (32 channels each), block = 256 (8 lanes per channel, uint4 = 8 halves).
// Two-phase deterministic block-ssq reduction with a 72-block self-resetting atomic counter
// (72 arrivals instead of 128, 18x fewer than one-per-expert and no full-tensor re-reads).
__launch_bounds__(256, 4)
static __global__ void ctrl_apply_ffn_norm_decode_kernel(
        const float  * __restrict__ cur,
        const float  * __restrict__ msg,
        const float  * __restrict__ gate,
        const __half * __restrict__ w_gain,
        const __half * __restrict__ w_forget,
        const float  * __restrict__ ffn_norm_w,
        float        * __restrict__ out,
        float        * __restrict__ router_scratch) {
    constexpr int N_BLK = N_EMBD / 32;              // 72 blocks
    constexpr int N_WARPS = 8;
    const int bid  = blockIdx.x;                    // 0..71
    const int c    = bid * 32 + (threadIdx.x >> 3);
    const int lane8 = threadIdx.x & 7;
    const int tid   = threadIdx.x;
    if (c >= N_EMBD) {
        return;
    }

    const float4 g0  = ((const float4 *)gate)[lane8 * 2 + 0];
    const float4 g1  = ((const float4 *)gate)[lane8 * 2 + 1];
    const uint4  wg8 = ((const uint4 *)(w_gain   + (size_t)c * LOOP_GATE_RANK))[lane8];
    const uint4  wf8 = ((const uint4 *)(w_forget + (size_t)c * LOOP_GATE_RANK))[lane8];

    float dg = dot_h8_f8(wg8, g0, g1, 0.0f);
    float df = dot_h8_f8(wf8, g0, g1, 0.0f);
    dg = warp_reduce_sum<8>(dg);
    df = warp_reduce_sum<8>(df);

    float v = 0.0f;
    if (lane8 == 0) {
        const float gain   = tanhf(dg);
        const float forget = tanhf(df);
        const float cv     = cur[c];
        const float mv     = msg[c];
        v = cv - forget * cv + gain * mv;
    }

    // Block ssq (only 32 lane8==0 lanes contribute)
    float ssq = (lane8 == 0) ? v * v : 0.0f;
    ssq = warp_reduce_sum<WARP_SIZE>(ssq);
    __shared__ float s_warp[N_WARPS];
    if ((tid & 31) == 0) {
        s_warp[tid >> 5] = ssq;
    }
    __syncthreads();
    if (tid == 0) {
        float tot = 0.0f;
        #pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            tot += s_warp[w];
        }
        __hip_atomic_store(&router_scratch[280 + bid], tot, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    // Single-thread 72-block completion barrier (deterministic)
    int32_t * done_ptr  = reinterpret_cast<int32_t *>(router_scratch + 264);
    int32_t * done2_ptr = reinterpret_cast<int32_t *>(router_scratch + 268);
    if (tid == 0) {
        rdna3_flush_stores();
        (void)__hip_atomic_fetch_add(done_ptr, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        while (__hip_atomic_load(done_ptr, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) < N_BLK) {
            __builtin_amdgcn_s_sleep(1);
        }
    }
    __syncthreads();
    rdna3_inv_l0_l1();

    // Deterministic total: every thread sums the 72 partials in a fixed order
    float partial = 0.0f;
    #pragma unroll
    for (int j = 0; j < 3; ++j) {
        const int idx = (tid & 31) + j * 32;
        if (idx < N_BLK) {
            partial += __hip_atomic_load(&router_scratch[280 + idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
    partial = warp_reduce_sum<WARP_SIZE>(partial);
    const float inv = rsqrtf(partial * (1.0f / (float)N_EMBD) + RMS_EPS);

    if (lane8 == 0) {
        const float wv = ffn_norm_w[c];
        const float nv = v * inv * wv;
        out[c] = nv;
    }

    // Self-resetting epilogue barrier: the last of the 72 blocks (which must be after every
    // block passed the ssq barrier) zeroes both counters for the next launch.
    if (tid == 0) {
        rdna3_flush_stores();
        const int prev2 = __hip_atomic_fetch_add(done2_ptr, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        if (prev2 == N_BLK - 1) {
            __hip_atomic_store(done_ptr,  0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(done2_ptr, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
}

// (B) Standalone decode FP16 router + sigmoid(+bias) + deterministic Top-8.
// grid = N_EXPERT/EPB blocks, block = 32*EPB threads (one Wave32 per expert).
// Fewer blocks => far fewer same-address AGENT-scope atomic arrivals in the Top-8
// completion barrier (each arrival costs ~170ns of L2 serialisation on gfx1100).
// EPB = 16 -> 8 blocks x 512 threads; x_norm (9.2 KB) is staged once per block.
// Scratch layout (512 floats of buf_router_scratch_):
//   [0..127] p, [128..255] sel, [256] int32 done counter (self-resetting)
template <int EPB>
__launch_bounds__(32 * EPB, (EPB >= 16 ? 2 : 4))
static __global__ void router_topk_decode_kernel(
        const float  * __restrict__ x_norm,
        const __half * __restrict__ router_w,
        const float  * __restrict__ exp_probs_b,
        int32_t      * __restrict__ topk_ids_out,
        float        * __restrict__ topk_w_out,
        float        * __restrict__ router_scratch) {
    constexpr int N4    = N_EMBD / 4;            // 576 float4
    constexpr int NT    = 32 * EPB;
    constexpr int N_BLK = N_EXPERT / EPB;

    const int bid  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int warp = tid >> 5;
    const int lane = tid & 31;

    // 1. Stage x_norm in LDS (9.2 KB), then prefetch the full router row of this expert.
    __shared__ float4 s_x[N4];
    for (int i = tid; i < N4; i += NT) {
        s_x[i] = ((const float4 *)x_norm)[i];
    }
    const int e = bid * EPB + warp;
    const uint4 * rw8 = (const uint4 *)(router_w + (size_t)e * N_EMBD);
    uint4 rw[9];
    #pragma unroll
    for (int s = 0; s < 9; ++s) {
        rw[s] = rw8[lane + s * WARP_SIZE];
    }
    __shared__ float lds_p[N_EXPERT];
    __shared__ float lds_sel[N_EXPERT];
    __shared__ bool  s_last;
    __syncthreads();
    __builtin_amdgcn_sched_barrier(0);

    float acc = 0.0f;
    #pragma unroll
    for (int s = 0; s < 9; ++s) {
        const int i8 = lane + s * WARP_SIZE;
        acc = dot_h8_f8(rw[s], s_x[i8 * 2 + 0], s_x[i8 * 2 + 1], acc);
    }
    const float sum = warp_reduce_sum<WARP_SIZE>(acc);

    if (lane == 0) {
        const float p   = 1.0f / (1.0f + expf(-sum));
        const float sel = exp_probs_b ? (p + exp_probs_b[e]) : p;
        __hip_atomic_store(&router_scratch[e],            p,   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        __hip_atomic_store(&router_scratch[N_EXPERT + e], sel, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    // Exactly ONE arrival per block, posted only after every warp of the block has
    // published its p/sel. The old per-warp arrival (EPB arrivals vs N_BLK) both
    // raced the Top-8 against experts that had not been written yet and left the
    // self-resetting counter non-zero (Top-8 then silently stopped firing).
    rdna3_flush_stores();
    __syncthreads();
    if (tid == 0) {
        rdna3_flush_stores();
        int32_t * done_ptr = reinterpret_cast<int32_t *>(router_scratch + 256);
        const int prev = __hip_atomic_fetch_add(done_ptr, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = (prev == N_BLK - 1);
        if (s_last) {
            __hip_atomic_store(done_ptr, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
    __syncthreads();

    if (s_last && warp == 0) {
        rdna3_inv_l0_l1();
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + j * 32;
            lds_p[idx]   = __hip_atomic_load(&router_scratch[idx],            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            lds_sel[idx] = __hip_atomic_load(&router_scratch[N_EXPERT + idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        wave32_topk8_128(lds_sel, lds_p, topk_ids_out, topk_w_out, lane);
    }
}

// Prefill Hardware WMMA Router GEMM + Sigmoid + exp_probs_b + Top-8 (n_tokens > 16):
// Direct 128-bit uint4 FP16 router_w loads into LDS (ZERO F32->FP16 weight conversions!).
// Uses BK = 32 (LDK = 40) and single-array topk LDS (11,520 B total LDS -> 5 blocks/CU = 10 waves/EU!).
template <int BM>
__launch_bounds__((BM == 16 ? 128 : 256)) __attribute__((amdgpu_waves_per_eu(10, 12)))
static __global__ void router_gemm_topk_prefill_wmma_kernel(
        const float  * __restrict__ x_norm,
        const __half * __restrict__ router_w,
        const float  * __restrict__ exp_probs_b,
        int32_t      * __restrict__ topk_ids_out,
        float        * __restrict__ topk_w_out,
        int n_tokens) {
#if NIB_MMA_TILE_TARGET
    static_assert(BM == 16 || BM == 32, "BM must be 16 or 32");
    constexpr int BK     = 32;
    constexpr int LDK    = BK + 8; // 40 halves (80 B -> 16B aligned)
    constexpr int NWARPS = (BM == 16) ? 4 : 8;
    constexpr int NTH    = NWARPS * 32;

    const int t0 = blockIdx.x * BM;
    if (t0 >= n_tokens) {
        return;
    }

    const int tid    = threadIdx.x;
    const int warp   = tid >> 5;
    const int lane   = tid & 31;
    const int l16    = lane & 15;
    const int half_w = lane >> 4;
    const int wm     = (BM == 16) ? 0 : (warp >> 2);
    const int wn     = (BM == 16) ? warp : (warp & 3);

    union RouterSmem {
        struct {
            alignas(16) _Float16 a[BM * LDK];       // 16*40*2 = 1,280 B
            alignas(16) _Float16 b[N_EXPERT * LDK]; // 128*40*2 = 10,240 B -> 11,520 B total
        } gemm;
        struct {
            float p[BM][N_EXPERT];                  // 16*128*4 = 8,192 B
        } topk;
    };
    __shared__ RouterSmem smem;

    constexpr int A_ITERS = (BM * 8) / NTH;       // 1
    constexpr int B_ITERS = (N_EXPERT * 4) / NTH; // 4 (BM=16) or 2 (BM=32)

    float8_t acc0 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float8_t acc1 = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};

    #pragma unroll 1
    for (int k0 = 0; k0 < N_EMBD; k0 += BK) {
        #pragma unroll
        for (int p = 0; p < A_ITERS; ++p) {
            const int idx4 = tid + p * NTH;
            const int m    = idx4 >> 3;
            const int k4   = idx4 & 7;
            float4 v = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (t0 + m < n_tokens) {
                v = ((const float4 *)(x_norm + (size_t)(t0 + m) * N_EMBD + k0))[k4];
            }
            const __half2 h01 = __floats2half2_rn(v.x, v.y);
            const __half2 h23 = __floats2half2_rn(v.z, v.w);
            *reinterpret_cast<uint2 *>(&smem.gemm.a[m * LDK + k4 * 4]) = make_uint2(bits_of(h01), bits_of(h23));
        }
        #pragma unroll
        for (int p = 0; p < B_ITERS; ++p) {
            const int idx8 = tid + p * NTH;
            const int e    = idx8 >> 2;
            const int k8   = idx8 & 3;
            *reinterpret_cast<uint4 *>(&smem.gemm.b[e * LDK + k8 * 8]) =
                ((const uint4 *)(router_w + (size_t)e * N_EMBD + k0))[k8];
        }
        __syncthreads();

        #pragma unroll
        for (int kc = 0; kc < 2; ++kc) {
#if NIB_MMA_NATIVE
            const half16_t af  = lds_frag(&smem.gemm.a[(wm * 16 + l16) * LDK + kc * 16]);
            const half16_t bf0 = lds_frag(&smem.gemm.b[(wn * 32 +      l16) * LDK + kc * 16]);
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf0, acc0);
            const half16_t bf1 = lds_frag(&smem.gemm.b[(wn * 32 + 16 + l16) * LDK + kc * 16]);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af, bf1, acc1);
#else
            const half16_t bf0 = lds_frag(&smem.gemm.b[(wn * 32 +      l16) * LDK + kc * 16]);
            nib_mma::mma_stride(&smem.gemm.a[(wm * 16) * LDK + kc * 16], LDK, bf0, acc0, lane);
            const half16_t bf1 = lds_frag(&smem.gemm.b[(wn * 32 + 16 + l16) * LDK + kc * 16]);
            nib_mma::mma_stride(&smem.gemm.a[(wm * 16) * LDK + kc * 16], LDK, bf1, acc1, lane);
#endif
        }
        __syncthreads();
    }

    const int e0 = wn * 32 + l16;
    const int e1 = e0 + 16;
    #pragma unroll
    for (int idx = 0; idx < 8; ++idx) {
        const int t_loc = wm * 16 + 2 * idx + half_w;
        smem.topk.p[t_loc][e0] = 1.0f / (1.0f + expf(-acc0[idx]));
        smem.topk.p[t_loc][e1] = 1.0f / (1.0f + expf(-acc1[idx]));
    }
    __syncthreads();

    for (int t_off = warp; t_off < BM; t_off += NWARPS) {
        const int t = t0 + t_off;
        if (t < n_tokens) {
            wave32_topk8_128_bias(smem.topk.p[t_off], exp_probs_b,
                                  topk_ids_out + (size_t)t * TOP_K,
                                  topk_w_out   + (size_t)t * TOP_K,
                                  lane);
        }
    }
#else
    (void)x_norm; (void)router_w; (void)exp_probs_b; (void)topk_ids_out; (void)topk_w_out; (void)n_tokens;
#endif
}

// ============================================================================
// 9. Logit Softcap & Fast Single-Launch Cooperative Argmax
// ============================================================================
__launch_bounds__(256, 4)
static __global__ void logit_softcap_kernel(
        float * __restrict__ logits,
        size_t total,
        float cap,
        float inv_cap) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        logits[idx] = cap * tanhf(logits[idx] * inv_cap);
    }
}

__launch_bounds__(256, 4)
static __global__ void logit_softcap_vec2_kernel(
        float2 * __restrict__ logits2,
        size_t total2,
        float cap,
        float inv_cap) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total2) {
        float2 v = logits2[idx];
        v.x = cap * tanhf(v.x * inv_cap);
        v.y = cap * tanhf(v.y * inv_cap);
        logits2[idx] = v;
    }
}

__device__ float   g_argmax_val[64];
__device__ int32_t g_argmax_idx[64];

__launch_bounds__(256, 4)
static __global__ void argmax_f32_stage1_kernel(
        const float * __restrict__ logits,
        int                        vocab_size) {
    constexpr int N_WARPS = 8;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = blockIdx.x * 256 + tid;
    const int stride  = 64 * 256;

    float   best_v = -INFINITY;
    int32_t best_i = INT32_MAX;

    if (((uintptr_t)logits & 15) == 0) {
        const int n4 = vocab_size >> 2;
        const float4 * l4 = (const float4 *)logits;
        for (int i4 = gid; i4 < n4; i4 += stride) {
            const float4 v4 = l4[i4];
            const int base = i4 << 2;
            if (v4.x > best_v || (v4.x == best_v && (base + 0) < best_i)) { best_v = v4.x; best_i = base + 0; }
            if (v4.y > best_v || (v4.y == best_v && (base + 1) < best_i)) { best_v = v4.y; best_i = base + 1; }
            if (v4.z > best_v || (v4.z == best_v && (base + 2) < best_i)) { best_v = v4.z; best_i = base + 2; }
            if (v4.w > best_v || (v4.w == best_v && (base + 3) < best_i)) { best_v = v4.w; best_i = base + 3; }
        }
        const int tail = (n4 << 2) + gid;
        if (tail < vocab_size) {
            const float v = logits[tail];
            if (v > best_v || (v == best_v && tail < best_i)) {
                best_v = v;
                best_i = tail;
            }
        }
    } else {
        for (int i = gid; i < vocab_size; i += stride) {
            const float v = logits[i];
            if (v > best_v || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float   ov = __shfl_xor(best_v, off, WARP_SIZE);
        const int32_t oi = __shfl_xor(best_i, off, WARP_SIZE);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    __shared__ float   s_val[N_WARPS];
    __shared__ int32_t s_idx[N_WARPS];
    if (lane == 0) {
        s_val[warp_id] = best_v;
        s_idx[warp_id] = best_i;
    }
    __syncthreads();

    if (warp_id == 0) {
        float   bv = (lane < N_WARPS) ? s_val[lane] : -INFINITY;
        int32_t bi = (lane < N_WARPS) ? s_idx[lane] : INT32_MAX;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            const float   ov = __shfl_xor(bv, off, WARP_SIZE);
            const int32_t oi = __shfl_xor(bi, off, WARP_SIZE);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            g_argmax_val[blockIdx.x] = bv;
            g_argmax_idx[blockIdx.x] = bi;
        }
    }
}

__launch_bounds__(32, 1)
static __global__ void argmax_f32_stage2_kernel(int32_t * __restrict__ d_out_id) {
    const int lane = threadIdx.x;
    float   bv = g_argmax_val[lane];
    int32_t bi = g_argmax_idx[lane];
    const float   v2 = g_argmax_val[lane + 32];
    const int32_t i2 = g_argmax_idx[lane + 32];
    if (v2 > bv || (v2 == bv && i2 < bi)) {
        bv = v2;
        bi = i2;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float   ov = __shfl_xor(bv, off, WARP_SIZE);
        const int32_t oi = __shfl_xor(bi, off, WARP_SIZE);
        if (ov > bv || (ov == bv && oi < bi)) {
            bv = ov;
            bi = oi;
        }
    }
    if (lane == 0) {
        // All-NaN input never wins a comparison: never emit the INT32_MAX sentinel.
        *d_out_id = (bi == INT32_MAX) ? 0 : bi;
    }
}

// Single-launch fused logit softcap (in-place) + cooperative argmax with self-resetting atomic counter.
// grid = 64, block = 256.
// Scratch layout: scratch[0..63] = float vals, scratch[64..127] = int32 idxs, scratch[272] = int32 done.
__launch_bounds__(256, 4)
static __global__ void logit_softcap_argmax_kernel(
        float   * __restrict__ logits,
        int                    vocab_size,
        float                  cap,
        float                  inv_cap,
        int32_t * __restrict__ d_out_id,
        float   * __restrict__ scratch,
        StepMeta * __restrict__ d_meta_next,     // nullable: advance decode meta for the next step
        const float * __restrict__ d_tau_tab,    // nullable: host-built exact tau(pos) table
        int32_t                swa_cap,
        int32_t * __restrict__ d_tok_hist,       // nullable: d_tok_hist[pos] = produced token
        int32_t                hist_cap) {
    constexpr int N_WARPS = 8;
    const int bid     = blockIdx.x; // 0..63
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int gid     = bid * 256 + tid;
    const int stride  = 64 * 256;

    float   best_v = -INFINITY;
    int32_t best_i = INT32_MAX;

    if (((uintptr_t)logits & 15) == 0) {
        const int n4 = vocab_size >> 2;
        float4 * l4 = reinterpret_cast<float4 *>(logits);
        for (int i4 = gid; i4 < n4; i4 += stride) {
            float4 v4 = l4[i4];
            if (cap != 0.0f) {
                v4.x = cap * tanhf(v4.x * inv_cap);
                v4.y = cap * tanhf(v4.y * inv_cap);
                v4.z = cap * tanhf(v4.z * inv_cap);
                v4.w = cap * tanhf(v4.w * inv_cap);
                l4[i4] = v4;
            }
            const int base = i4 << 2;
            if (v4.x > best_v || (v4.x == best_v && (base + 0) < best_i)) { best_v = v4.x; best_i = base + 0; }
            if (v4.y > best_v || (v4.y == best_v && (base + 1) < best_i)) { best_v = v4.y; best_i = base + 1; }
            if (v4.z > best_v || (v4.z == best_v && (base + 2) < best_i)) { best_v = v4.z; best_i = base + 2; }
            if (v4.w > best_v || (v4.w == best_v && (base + 3) < best_i)) { best_v = v4.w; best_i = base + 3; }
        }
        const int tail = (n4 << 2) + gid;
        if (tail < vocab_size) {
            float v = logits[tail];
            if (cap != 0.0f) {
                v = cap * tanhf(v * inv_cap);
                logits[tail] = v;
            }
            if (v > best_v || (v == best_v && tail < best_i)) {
                best_v = v;
                best_i = tail;
            }
        }
    } else {
        for (int i = gid; i < vocab_size; i += stride) {
            float v = logits[i];
            if (cap != 0.0f) {
                v = cap * tanhf(v * inv_cap);
                logits[i] = v;
            }
            if (v > best_v || (v == best_v && i < best_i)) {
                best_v = v;
                best_i = i;
            }
        }
    }

    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        const float   ov = __shfl_xor(best_v, off, WARP_SIZE);
        const int32_t oi = __shfl_xor(best_i, off, WARP_SIZE);
        if (ov > best_v || (ov == best_v && oi < best_i)) {
            best_v = ov;
            best_i = oi;
        }
    }

    __shared__ float   s_val[N_WARPS];
    __shared__ int32_t s_idx[N_WARPS];
    if (lane == 0) {
        s_val[warp_id] = best_v;
        s_idx[warp_id] = best_i;
    }
    __syncthreads();

    __shared__ bool s_is_last;
    if (warp_id == 0) {
        float   bv = (lane < N_WARPS) ? s_val[lane] : -INFINITY;
        int32_t bi = (lane < N_WARPS) ? s_idx[lane] : INT32_MAX;
        #pragma unroll
        for (int off = 4; off > 0; off >>= 1) {
            const float   ov = __shfl_xor(bv, off, WARP_SIZE);
            const int32_t oi = __shfl_xor(bi, off, WARP_SIZE);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            int32_t * s_iptr = reinterpret_cast<int32_t *>(scratch + 64);
            int32_t * done   = reinterpret_cast<int32_t *>(scratch + 272);
            __hip_atomic_store(&scratch[bid],  bv, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&s_iptr[bid],   bi, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            rdna3_flush_stores();
            const int prev = __hip_atomic_fetch_add(done, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
            s_is_last = (prev == 63);
            if (s_is_last) {
                __hip_atomic_store(done, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
    }
    __syncthreads();

    if (s_is_last && warp_id == 0) {
        rdna3_inv_l0_l1();
        const int32_t * s_iptr = reinterpret_cast<const int32_t *>(scratch + 64);
        float   bv = __hip_atomic_load(&scratch[lane],       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        int32_t bi = __hip_atomic_load(&s_iptr[lane],        __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const float   v2 = __hip_atomic_load(&scratch[lane + 32], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        const int32_t i2 = __hip_atomic_load(&s_iptr[lane + 32],  __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        if (v2 > bv || (v2 == bv && i2 < bi)) {
            bv = v2;
            bi = i2;
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float   ov = __shfl_xor(bv, off, WARP_SIZE);
            const int32_t oi = __shfl_xor(bi, off, WARP_SIZE);
            if (ov > bv || (ov == bv && oi < bi)) {
                bv = ov;
                bi = oi;
            }
        }
        if (lane == 0) {
            // NaN logits never win a comparison, so an all-NaN row would leave the
            // INT32_MAX sentinel here and the next graph step would gather embedding
            // row 2^31-1 (issue #1: GPU fault / hang).  Always emit a valid id.
            if (bi < 0 || bi >= vocab_size) {
                bi = 0;
            }
            *d_out_id = bi;
            if (d_meta_next != nullptr) {
                // Single-thread deterministic meta advance for the next decode step:
                // token_id = produced token, pos += 1, kv lengths/write idx follow, tau from host table.
                const int32_t tok = bi;
                const int32_t p   = d_meta_next->pos + 1;
                d_meta_next->token_id        = tok;
                d_meta_next->pos             = p;
                d_meta_next->kv_len_global   = p + 1;
                d_meta_next->kv_len_swa      = (p + 1 < swa_cap) ? (p + 1) : swa_cap;
                d_meta_next->kv_write_global = p;
                d_meta_next->kv_write_swa    = (swa_cap > 0) ? (p % swa_cap) : p;
                d_meta_next->tau             = (d_tau_tab != nullptr) ? d_tau_tab[p] : d_meta_next->tau;
                if (d_tok_hist != nullptr && p >= 0 && p < hist_cap) {
                    d_tok_hist[p] = tok;
                }
            }
        }
    }
}

} // anonymous namespace

// ============================================================================
// Host Launchers
// ============================================================================

void launch_embed_norm(
        const SoaQ8Matrix & tok_embd, const float * tok_norm,
        const int32_t * d_token_ids, const StepMeta * d_meta,
        float * out, int n_tokens, hipStream_t stream) {
    if (n_tokens <= 0) {
        return;
    }
    hipLaunchKernelGGL(
        embed_norm_2304_kernel,
        dim3(n_tokens), dim3(576), 0, stream,
        tok_embd.qs, tok_embd.d, tok_norm, d_token_ids, d_meta, out, n_tokens);
}

void launch_inkling_mix(
        const float * query, int n_q,
        const float * const * states, int n_states,
        float * out, int n_tokens, hipStream_t stream) {
    if (n_tokens <= 0 || n_states <= 0 || n_q <= 0) {
        return;
    }
    StatePtrs sptrs = {};
    for (int i = 0; i < n_states && i < MAX_RES_BLOCKS; ++i) {
        sptrs.p[i] = states[i];
    }
    if (n_states <= 5 && n_q <= 4) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inkling_mix_2304_kernel<5, 4>),
            dim3(n_tokens), dim3(576), 0, stream,
            query, sptrs, out, n_states, n_q, n_tokens);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inkling_mix_2304_kernel<MAX_RES_BLOCKS, 4>),
            dim3(n_tokens), dim3(576), 0, stream,
            query, sptrs, out, n_states, n_q, n_tokens);
    }
}

void launch_inkling_mix_dual(
        const float * query0, int n_q0, const float * const * states0, int n_states0, float * out0,
        const float * query1, int n_q1, const float * const * states1, int n_states1, float * out1,
        int n_tokens, hipStream_t stream) {
    if (n_tokens <= 0 || n_states0 <= 0 || n_q0 <= 0 || n_states1 <= 0 || n_q1 <= 0) {
        return;
    }
    StatePtrs s0 = {}, s1 = {};
    for (int i = 0; i < n_states0 && i < MAX_RES_BLOCKS; ++i) {
        s0.p[i] = states0[i];
    }
    for (int i = 0; i < n_states1 && i < MAX_RES_BLOCKS; ++i) {
        s1.p[i] = states1[i];
    }
    if (n_states0 <= 5 && n_states1 <= 5 && n_q0 <= 1 && n_q1 <= 1) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inkling_mix_dual_2304_kernel<5, 1>),
            dim3(n_tokens, 2), dim3(192), 0, stream,
            query0, n_q0, s0, n_states0, out0,
            query1, n_q1, s1, n_states1, out1,
            n_tokens);
    } else if (n_states0 <= 5 && n_states1 <= 5) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inkling_mix_dual_2304_kernel<5, 4>),
            dim3(n_tokens, 2), dim3(192), 0, stream,
            query0, n_q0, s0, n_states0, out0,
            query1, n_q1, s1, n_states1, out1,
            n_tokens);
    } else if (n_q0 <= 1 && n_q1 <= 1) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inkling_mix_dual_2304_kernel<MAX_RES_BLOCKS, 1>),
            dim3(n_tokens, 2), dim3(192), 0, stream,
            query0, n_q0, s0, n_states0, out0,
            query1, n_q1, s1, n_states1, out1,
            n_tokens);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inkling_mix_dual_2304_kernel<MAX_RES_BLOCKS, 4>),
            dim3(n_tokens, 2), dim3(192), 0, stream,
            query0, n_q0, s0, n_states0, out0,
            query1, n_q1, s1, n_states1, out1,
            n_tokens);
    }
}

void launch_inkling_mix_rms_norm(
        const float * query, int n_q,
        const float * const * states, int n_states,
        const float * const * w_norms,
        float * out, int n_tokens, float eps, hipStream_t stream) {
    if (n_tokens <= 0 || n_states <= 0 || n_q <= 0) {
        return;
    }
    StatePtrs sptrs = {};
    for (int i = 0; i < n_states && i < MAX_RES_BLOCKS; ++i) {
        sptrs.p[i] = states[i];
    }
    NormPtrs nptrs = {};
    for (int q = 0; q < n_q && q < 4; ++q) {
        nptrs.p[q] = w_norms ? w_norms[q] : nullptr;
    }
    if (n_states <= 5 && n_q <= 1) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inkling_mix_rms_norm_2304_kernel<5, 1>),
            dim3(n_tokens), dim3(192), 0, stream,
            query, sptrs, nptrs, out, n_states, n_q, n_tokens, eps);
    } else if (n_states <= 5 && n_q <= 3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inkling_mix_rms_norm_2304_kernel<5, 3>),
            dim3(n_tokens), dim3(192), 0, stream,
            query, sptrs, nptrs, out, n_states, n_q, n_tokens, eps);
    } else if (n_states <= 5) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inkling_mix_rms_norm_2304_kernel<5, 4>),
            dim3(n_tokens), dim3(192), 0, stream,
            query, sptrs, nptrs, out, n_states, n_q, n_tokens, eps);
    } else if (n_q <= 1) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inkling_mix_rms_norm_2304_kernel<MAX_RES_BLOCKS, 1>),
            dim3(n_tokens), dim3(192), 0, stream,
            query, sptrs, nptrs, out, n_states, n_q, n_tokens, eps);
    } else if (n_q <= 3) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inkling_mix_rms_norm_2304_kernel<MAX_RES_BLOCKS, 3>),
            dim3(n_tokens), dim3(192), 0, stream,
            query, sptrs, nptrs, out, n_states, n_q, n_tokens, eps);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(inkling_mix_rms_norm_2304_kernel<MAX_RES_BLOCKS, 4>),
            dim3(n_tokens), dim3(192), 0, stream,
            query, sptrs, nptrs, out, n_states, n_q, n_tokens, eps);
    }
}

void launch_inkling_ctrl(
        const float * cur, const float * msg,
        const __half * w_down, const __half * w_gain, const __half * w_forget,
        float * tmp_gate,
        float * out, int n_tokens, hipStream_t stream) {
    if (n_tokens <= 0) {
        return;
    }
    Ptr3 cur3     = { { cur, nullptr, nullptr } };
    HalfPtr3 wd3  = { { w_down, nullptr, nullptr } };
    HalfPtr3 wg3  = { { w_gain, nullptr, nullptr } };
    HalfPtr3 wf3  = { { w_forget, nullptr, nullptr } };
    MutPtr3 out3  = { { out, nullptr, nullptr } };

    if (n_tokens <= 16) {
        hipLaunchKernelGGL(
            inkling_ctrl_down_decode_kernel,
            dim3(LOOP_GATE_RANK, n_tokens), dim3(288), 0, stream,
            cur, msg, w_down, tmp_gate, n_tokens);
    } else {
        hipLaunchKernelGGL(
            inkling_ctrl_down_prefill_wmma_kernel,
            dim3(4, (n_tokens + 15) / 16, 1), dim3(128), 0, stream,
            cur3, msg, wd3, tmp_gate, n_tokens);
    }

    if (n_tokens <= 8) {
        hipLaunchKernelGGL(
            inkling_ctrl_apply_decode_kernel,
            dim3((N_EMBD + 31) / 32, n_tokens), dim3(256), 0, stream,
            cur, msg, tmp_gate, w_gain, w_forget, out, n_tokens);
    } else {
        hipLaunchKernelGGL(
            inkling_ctrl_apply_prefill_wmma_kernel,
            dim3(N_EMBD / 64, (n_tokens + 63) / 64, 1), dim3(256), 0, stream,
            cur3, msg, tmp_gate, wg3, wf3, out3, n_tokens);
    }
}

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
        float * scratch) {
    if (n_tokens <= 0) {
        return;
    }
    Ptr3 cur3     = { { cur[0], cur[1], cur[2] } };
    HalfPtr3 wd3  = { { w_down[0], w_down[1], w_down[2] } };
    HalfPtr3 wg3  = { { w_gain[0], w_gain[1], w_gain[2] } };
    HalfPtr3 wf3  = { { w_forget[0], w_forget[1], w_forget[2] } };
    Ptr3 wn3      = { { w_norm[0], w_norm[1], w_norm[2] } };
    MutPtr3 out3  = { { out_norm[0], out_norm[1], out_norm[2] } };

    if (n_tokens == 1 && scratch != nullptr) {
        hipLaunchKernelGGL(
            inkling_ctrl_down_qkv3_decode_kernel,
            dim3(16, 1, 3), dim3(288), 0, stream,
            cur3, msg_base, wd3, tmp_gate, 1);
        hipLaunchKernelGGL(
            inkling_ctrl_apply_norm_qkv3_multiwgp_kernel,
            dim3(32, 3), dim3(192), 0, stream,
            cur3, msg_base, tmp_gate, wg3, wf3, wn3, out3, eps, scratch);
        return;
    }

    if (n_tokens <= 16) {
        hipLaunchKernelGGL(
            inkling_ctrl_down_qkv3_decode_kernel,
            dim3(16, n_tokens, 3), dim3(288), 0, stream,
            cur3, msg_base, wd3, tmp_gate, n_tokens);
    } else {
        hipLaunchKernelGGL(
            inkling_ctrl_down_prefill_wmma_kernel,
            dim3(4, (n_tokens + 15) / 16, 3), dim3(128), 0, stream,
            cur3, msg_base, wd3, tmp_gate, n_tokens);
    }

    if (n_tokens <= 8) {
        hipLaunchKernelGGL(
            inkling_ctrl_apply_qkv3_small_kernel,
            dim3((N_EMBD + 31) / 32, n_tokens, 3), dim3(256), 0, stream,
            cur3, msg_base, tmp_gate, wg3, wf3, out3, n_tokens);
    } else {
        hipLaunchKernelGGL(
            inkling_ctrl_apply_prefill_wmma_kernel,
            dim3(N_EMBD / 64, (n_tokens + 63) / 64, 3), dim3(256), 0, stream,
            cur3, msg_base, tmp_gate, wg3, wf3, out3, n_tokens);
    }

    hipLaunchKernelGGL(
        rms_norm_2304_qkv3_kernel,
        dim3(n_tokens, 3), dim3(576), 0, stream,
        out3, wn3, n_tokens, eps);
}

void launch_mix_ctrl_down_decode(
        const float * res_query,
        const float * const * states, int n_states,
        const float * msg,
        const __half * w_down,
        float * out_mixed,
        float * out_gate,
        hipStream_t stream) {
    if (n_states <= 0) {
        return;
    }
    StatePtrs sptrs = {};
    for (int i = 0; i < n_states && i < MAX_RES_BLOCKS; ++i) {
        sptrs.p[i] = states[i];
    }
    if (n_states <= 5) {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(mix_ctrl_down_decode_kernel<5>),
            dim3(32), dim3(288), 0, stream,
            res_query, sptrs, n_states, msg, w_down, out_mixed, out_gate);
    } else {
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(mix_ctrl_down_decode_kernel<MAX_RES_BLOCKS>),
            dim3(32), dim3(288), 0, stream,
            res_query, sptrs, n_states, msg, w_down, out_mixed, out_gate);
    }
}

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
        hipStream_t stream) {
    if (n_states <= 0) {
        return;
    }
    if (router_scratch == nullptr) {
        launch_inkling_mix(res_query, 1, states, n_states, x_norm_out, 1, stream);
        launch_ffn_norm_router_topk(x_norm_out, ffn_norm_w, router_w, exp_probs_b,
                                    x_norm_out, topk_ids_out, topk_w_out, 1, stream, nullptr);
        return;
    }
    // Split path (no cross-block spin-wait for the mix/norm part):
    //   (1) mix + FFN RMSNorm in one 192-thread block (all states register-cached)
    //   (2) standalone 32-block router + Top-8.
    const float * w_norms[1] = { ffn_norm_w };
    launch_inkling_mix_rms_norm(res_query, 1, states, n_states, w_norms,
                                x_norm_out, 1, RMS_EPS, stream);
    hipLaunchKernelGGL(
        HIP_KERNEL_NAME(router_topk_decode_kernel<4>),
        dim3(N_EXPERT / 4), dim3(128), 0, stream,
        x_norm_out, router_w, exp_probs_b, topk_ids_out, topk_w_out, router_scratch);
}

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
        hipStream_t stream) {
    if (router_scratch != nullptr) {
        // Split path (no 128-block spin-wait): apply+norm (72 blocks) then router+Top-8 (32 blocks).
        hipLaunchKernelGGL(
            ctrl_apply_ffn_norm_decode_kernel,
            dim3(N_EMBD / 32), dim3(256), 0, stream,
            cur, msg, gate, w_gain, w_forget, ffn_norm_w, x_norm_out, router_scratch);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(router_topk_decode_kernel<4>),
            dim3(N_EXPERT / 4), dim3(128), 0, stream,
            x_norm_out, router_w, exp_probs_b, topk_ids_out, topk_w_out, router_scratch);
    } else {
        hipLaunchKernelGGL(
            inkling_ctrl_apply_decode_kernel,
            dim3((N_EMBD + 31) / 32, 1), dim3(256), 0, stream,
            cur, msg, gate, w_gain, w_forget, x_norm_out, 1);
        hipLaunchKernelGGL(
            ffn_norm_router_topk_decode_kernel,
            dim3(1), dim3(512), 0, stream,
            x_norm_out, ffn_norm_w, router_w, exp_probs_b, x_norm_out, topk_ids_out, topk_w_out, 1);
    }
}

void launch_rms_norm(
        const float * x, const float * w, float * out,
        int n_tokens, int dim, float eps, hipStream_t stream) {
    if (n_tokens <= 0) {
        return;
    }
    const bool aligned4 = (dim == N_EMBD) &&
                          (((uintptr_t)x & 15) == 0) &&
                          (((uintptr_t)out & 15) == 0) &&
                          (!w || (((uintptr_t)w & 15) == 0));
    if (aligned4) {
        hipLaunchKernelGGL(
            rms_norm_2304_kernel,
            dim3(n_tokens), dim3(576), 0, stream,
            x, w, out, n_tokens, eps);
    } else {
        hipLaunchKernelGGL(
            rms_norm_generic_kernel,
            dim3(n_tokens), dim3(256), 0, stream,
            x, w, out, n_tokens, dim, eps);
    }
}

void launch_sconv_kv_store_q8(
        const float * raw_kv,
        int           raw_stride,
        const float * conv_w,
        const float * dyn_taps,
        int           dyn_stride,
        float       * sconv_hist,
        KVCacheSlot   kv_slot,
        bool          is_v,
        int           n_tokens,
        int           pos_start,
        const StepMeta * d_meta,
        hipStream_t   stream) {
    if (n_tokens <= 0) {
        return;
    }
    const bool has_dyn = (dyn_taps != nullptr);
    const dim3 grid(N_HEAD_KV, 1);
    const dim3 block(32);
    if (has_dyn) {
        if (is_v) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(sconv_kv_store_q8_kernel<true, true>),
                grid, block, 0, stream,
                raw_kv, raw_stride, conv_w, dyn_taps, dyn_stride, sconv_hist, kv_slot, n_tokens, pos_start, d_meta);
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(sconv_kv_store_q8_kernel<true, false>),
                grid, block, 0, stream,
                raw_kv, raw_stride, conv_w, dyn_taps, dyn_stride, sconv_hist, kv_slot, n_tokens, pos_start, d_meta);
        }
    } else {
        if (is_v) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(sconv_kv_store_q8_kernel<false, true>),
                grid, block, 0, stream,
                raw_kv, raw_stride, conv_w, dyn_taps, dyn_stride, sconv_hist, kv_slot, n_tokens, pos_start, d_meta);
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(sconv_kv_store_q8_kernel<false, false>),
                grid, block, 0, stream,
                raw_kv, raw_stride, conv_w, dyn_taps, dyn_stride, sconv_hist, kv_slot, n_tokens, pos_start, d_meta);
        }
    }
}

__launch_bounds__(32, 8)
static __global__ void batch_sconv_kv_store_q8_kernel(
        const float * __restrict__ k_raw, int k_stride, const float * __restrict__ k_conv_w,
        const float * __restrict__ k_dyn, int k_dyn_stride,
        const float * __restrict__ v_raw, int v_stride, const float * __restrict__ v_conv_w,
        const float * __restrict__ v_dyn, int v_dyn_stride,
        BatchKVSlotArgs            args,
        int                        m,
        const StepMeta * __restrict__ d_meta) {
    const int  h    = blockIdx.x;        // 0..7
    const bool is_v = (blockIdx.y != 0); // 0=K, 1=V
    const int  t    = blockIdx.z;        // 0..m-1
    const int  lane = threadIdx.x;       // 0..31
    if (t >= m) {
        return;
    }

    const float * raw_kv     = is_v ? v_raw : k_raw;
    const int     raw_stride = is_v ? v_stride : k_stride;
    const float * conv_w     = is_v ? v_conv_w : k_conv_w;
    const float * dyn_taps   = is_v ? v_dyn : k_dyn;
    const int     dyn_stride = is_v ? v_dyn_stride : k_dyn_stride;
    const bool    has_dyn    = (dyn_taps != nullptr);

    float * sconv_hist = args.hist_base[t] + (is_v ? (size_t)N_KV * N_CONV_HIST : 0);
    const KVCacheSlot kv_slot = args.kv[t];
    const int c_base = h * HEAD_DIM + lane * 4;

    const float4 rt4 = *(const float4 *)(raw_kv + (size_t)t * raw_stride + c_base);
    const float x3[4] = { rt4.x, rt4.y, rt4.z, rt4.w };

    float4 d_dyn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (has_dyn) {
        d_dyn = ((const float4 *)(dyn_taps + (size_t)t * dyn_stride))[h];
    }

    const float conv_scale = has_dyn ? (1.0f / 3.0f) : 0.4472135954999579f;
    float y[4];
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 w = ((const float4 *)conv_w)[c_base + j];
        float * hs = sconv_hist + (size_t)(c_base + j) * N_CONV_HIST;
        const float h0 = hs[0], h1 = hs[1], h2 = hs[2];
        float acc = x3[j] + h0 * w.x + h1 * w.y + h2 * w.z + x3[j] * w.w;
        if (has_dyn) {
            acc += x3[j] * d_dyn.x + h2 * d_dyn.y + h1 * d_dyn.z + h0 * d_dyn.w;
        }
        y[j] = acc * conv_scale;
        hs[0] = h1;
        hs[1] = h2;
        hs[2] = x3[j];
    }

    float4 v = make_float4(y[0], y[1], y[2], y[3]);
    float sumsq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
    sumsq = warp_reduce_sum<WARP_SIZE>(sumsq);

    const float factor = is_v
        ? (rsqrtf(sumsq * (1.0f / (float)HEAD_DIM) + RMS_EPS) * KQ_SCALE)
        : (fminf(1.0f, QK_CLIP / (sqrtf(sumsq) + RMS_EPS)) * KQ_SCALE);
    v.x *= factor;
    v.y *= factor;
    v.z *= factor;
    v.w *= factor;

    wave32_fwht_128_float4(v, lane);

    const int32_t pos = d_meta[t].pos;
    const int32_t idx = kv_slot.is_swa ? d_meta[t].kv_write_swa : d_meta[t].kv_write_global;
    store_kv_quantized(v, kv_slot, is_v, h, idx, lane);
    if (!is_v && h == 0 && lane == 0) {
        kv_slot.k_pos[idx] = pos;
    }
}

void launch_batch_sconv_kv_store_q8(
        const float * k_raw, int k_stride, const float * k_conv_w, const float * k_dyn, int k_dyn_stride,
        const float * v_raw, int v_stride, const float * v_conv_w, const float * v_dyn, int v_dyn_stride,
        BatchKVSlotArgs args,
        int             m,
        const StepMeta * d_meta,
        hipStream_t     stream) {
    if (m <= 0) {
        return;
    }
    const dim3 grid(N_HEAD_KV, 2, (unsigned)m);
    const dim3 block(32);
    hipLaunchKernelGGL(
        batch_sconv_kv_store_q8_kernel, grid, block, 0, stream,
        k_raw, k_stride, k_conv_w, k_dyn, k_dyn_stride,
        v_raw, v_stride, v_conv_w, v_dyn, v_dyn_stride,
        args, m, d_meta);
}

void launch_q_r_prep_and_reltable(
        const float * q_in, int q_stride,
        const float * q_clip_w,
        const float * r_in, int r_stride,
        const float * r_clip_w,
        const float * rel_bias_w,
        float       * q_out,
        float       * rel_table_out,
        bool          apply_tau,
        int           n_tokens,
        int           pos_start,
        const StepMeta * d_meta,
        hipStream_t   stream) {
    if (n_tokens <= 0) {
        return;
    }
    const int total_q_rows = n_tokens * N_HEAD;
    const int warps_per_blk = 4;
    hipLaunchKernelGGL(
        q_prep_fwht128_kernel,
        dim3((total_q_rows + warps_per_blk - 1) / warps_per_blk), dim3(warps_per_blk * WARP_SIZE), 0, stream,
        q_in, q_stride, q_clip_w, q_out, apply_tau, n_tokens, pos_start, d_meta);

    if (n_tokens == 1) {
        hipLaunchKernelGGL(
            r_prep_reltable_decode_kernel,
            dim3(N_REL_EXTENT / 32), dim3(128), 0, stream,
            r_in, r_clip_w, rel_bias_w, rel_table_out, apply_tau, pos_start, d_meta);
    } else {
        const int max_dist = (d_meta != nullptr) ? N_REL_EXTENT : std::min(pos_start + n_tokens, N_REL_EXTENT);
        hipLaunchKernelGGL(
            r_prep_reltable_prefill_kernel,
            dim3((max_dist + 127) / 128, (total_q_rows + 63) / 64), dim3(128), 0, stream,
            r_in, r_stride, r_clip_w, rel_bias_w, rel_table_out, apply_tau, n_tokens, pos_start, max_dist,
            d_meta);
    }
}

void launch_qkvr_prep_prefill(
        const float * q_in, int q_stride, const float * q_clip_w,
        const float * r_in, int r_stride, const float * r_clip_w, const float * rel_bias_w,
        const float * k_raw, int k_stride, const float * k_conv_w, const float * k_dyn, int k_dyn_stride,
        const float * v_raw, int v_stride, const float * v_conv_w, const float * v_dyn, int v_dyn_stride,
        float       * sconv_hist_kv,
        KVCacheSlot   kv_slot,
        float       * q_out,
        float       * rel_table_out,
        bool          apply_tau,
        int           n_tokens,
        int           pos_start,
        hipStream_t   stream) {
    if (n_tokens <= 0) {
        return;
    }
    // 1. Single-launch time-chunked K+V shortconv + Clip/RMSNorm + FWHT-128 + Q8_0 SoA KV store + inline history write
    const int n_tchunks = (n_tokens + SCONV_KV_TCHUNK - 1) / SCONV_KV_TCHUNK;
    hipLaunchKernelGGL(
        sconv_kv_dual_prefill_kernel,
        dim3(N_HEAD_KV, 2, n_tchunks), dim3(32), 0, stream,
        k_raw, k_stride, k_conv_w, k_dyn, k_dyn_stride,
        v_raw, v_stride, v_conv_w, v_dyn, v_dyn_stride,
        sconv_hist_kv, kv_slot, n_tokens, pos_start);

    // 2. Q prep + 4x-vectorized R prep & reltable
    launch_q_r_prep_and_reltable(
        q_in, q_stride, q_clip_w,
        r_in, r_stride, r_clip_w, rel_bias_w,
        q_out, rel_table_out, apply_tau, n_tokens, pos_start, nullptr, stream);
}

void launch_qkvr_prep_fused_decode(
        const float * q_in, int q_stride, const float * q_clip_w,
        const float * r_in, int r_stride, const float * r_clip_w, const float * rel_bias_w,
        const float * k_raw, int k_stride, const float * k_conv_w, const float * k_dyn, int k_dyn_stride,
        const float * v_raw, int v_stride, const float * v_conv_w, const float * v_dyn, int v_dyn_stride,
        float       * sconv_hist_kv,
        KVCacheSlot   kv_slot,
        float       * q_out,
        float       * rel_table_out,
        bool          apply_tau,
        const StepMeta * d_meta,
        hipStream_t   stream) {
    (void)q_stride; (void)r_stride; (void)k_stride; (void)k_dyn_stride; (void)v_stride; (void)v_dyn_stride;
    hipLaunchKernelGGL(
        qkvr_prep_fused_decode_kernel,
        dim3(72), dim3(128), 0, stream,
        q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
        k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
        sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta);
}

void launch_sconv_post_res(
        const float * x,
        const float * conv_w,
        float       * sconv_hist,
        const float * res,
        float       * out,
        int           n_tokens,
        hipStream_t   stream) {
    if (n_tokens <= 0) {
        return;
    }
    const bool has_res = (res != nullptr);
    const int t_chunk = (x != out && n_tokens > SCONV_POST_TCHUNK) ? SCONV_POST_TCHUNK : n_tokens;
    const int n_tchunks = (n_tokens + t_chunk - 1) / t_chunk;
    const dim3 grid((N_EMBD / 4) / 32, n_tchunks); // 18 warps * n_tchunks -> 576 blocks at M=512
    const dim3 block(32);
    if (has_res) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(sconv_post_res_kernel<true>),
            grid, block, 0, stream,
            x, conv_w, sconv_hist, res, out, n_tokens, t_chunk);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(sconv_post_res_kernel<false>),
            grid, block, 0, stream,
            x, conv_w, sconv_hist, res, out, n_tokens, t_chunk);
    }
}

void launch_ffn_norm_router_topk(
        const float  * x_in,
        const float  * ffn_norm_w,
        const __half * router_w,
        const float  * exp_probs_b,
        float        * x_norm_out,
        int32_t      * topk_ids_out,
        float        * topk_w_out,
        int            n_tokens,
        hipStream_t    stream,
        float        * router_scratch) {
    if (n_tokens <= 0) {
        return;
    }
    if (n_tokens == 1 && router_scratch != nullptr) {
        hipLaunchKernelGGL(
            ffn_norm_router_topk_multiwgp_kernel,
            dim3(32), dim3(128), 0, stream,
            x_in, ffn_norm_w, router_w, exp_probs_b, x_norm_out, topk_ids_out, topk_w_out, router_scratch);
    } else if (n_tokens <= 16) {
        hipLaunchKernelGGL(
            ffn_norm_router_topk_decode_kernel,
            dim3(n_tokens), dim3(512), 0, stream,
            x_in, ffn_norm_w, router_w, exp_probs_b, x_norm_out, topk_ids_out, topk_w_out, n_tokens);
    } else {
        launch_rms_norm(x_in, ffn_norm_w, x_norm_out, n_tokens, N_EMBD, RMS_EPS, stream);
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(router_gemm_topk_prefill_wmma_kernel<16>),
            dim3((n_tokens + 15) / 16), dim3(128), 0, stream,
            x_norm_out, router_w, exp_probs_b, topk_ids_out, topk_w_out, n_tokens);
    }
}

void launch_logit_softcap(
        float * logits, int n_tokens, int vocab_size, float cap, hipStream_t stream) {
    if (n_tokens <= 0 || vocab_size <= 0 || cap == 0.0f) {
        return;
    }
    const size_t total = (size_t)n_tokens * vocab_size;
    const float inv_cap = 1.0f / cap;
    if ((total & 1) == 0 && (((uintptr_t)logits & 7) == 0)) {
        const size_t total2 = total >> 1;
        hipLaunchKernelGGL(
            logit_softcap_vec2_kernel,
            dim3((total2 + 255) / 256), dim3(256), 0, stream,
            (float2 *)logits, total2, cap, inv_cap);
    } else {
        hipLaunchKernelGGL(
            logit_softcap_kernel,
            dim3((total + 255) / 256), dim3(256), 0, stream,
            logits, total, cap, inv_cap);
    }
}

void launch_argmax_f32(
        const float * logits, int vocab_size, int32_t * d_out_id, hipStream_t stream) {
    if (vocab_size <= 0) {
        return;
    }
    hipLaunchKernelGGL(
        argmax_f32_stage1_kernel,
        dim3(64), dim3(256), 0, stream,
        logits, vocab_size);
    hipLaunchKernelGGL(
        argmax_f32_stage2_kernel,
        dim3(1), dim3(32), 0, stream,
        d_out_id);
}

void launch_logit_softcap_argmax(
        float * logits, int vocab_size, float cap, int32_t * d_out_id,
        float * scratch, hipStream_t stream) {
    if (vocab_size <= 0) {
        return;
    }
    if (scratch == nullptr) {
        launch_logit_softcap(logits, 1, vocab_size, cap, stream);
        launch_argmax_f32(logits, vocab_size, d_out_id, stream);
        return;
    }
    const float inv_cap = (cap != 0.0f) ? (1.0f / cap) : 0.0f;
    hipLaunchKernelGGL(
        logit_softcap_argmax_kernel,
        dim3(64), dim3(256), 0, stream,
        logits, vocab_size, cap, inv_cap, d_out_id, scratch,
        (StepMeta *)nullptr, (const float *)nullptr, 0, (int32_t *)nullptr, 0);
}

// Fused softcap + argmax + device-side StepMeta advance (single-thread, deterministic).
// Used inside the static decode graph so N graph executions can be chained back-to-back
// with ZERO host round-trips (one sync for the whole batch of steps).
void launch_logit_softcap_argmax_advance(
        float * logits, int vocab_size, float cap, int32_t * d_out_id,
        float * scratch, StepMeta * d_meta_next, const float * d_tau_tab,
        int32_t swa_cap, int32_t * d_tok_hist, int32_t hist_cap, hipStream_t stream) {
    if (vocab_size <= 0) {
        return;
    }
    const float inv_cap = (cap != 0.0f) ? (1.0f / cap) : 0.0f;
    if (scratch == nullptr) {
        launch_logit_softcap(logits, 1, vocab_size, cap, stream);
        launch_argmax_f32(logits, vocab_size, d_out_id, stream);
        return;
    }
    hipLaunchKernelGGL(
        logit_softcap_argmax_kernel,
        dim3(64), dim3(256), 0, stream,
        logits, vocab_size, cap, inv_cap, d_out_id, scratch,
        d_meta_next, d_tau_tab, swa_cap, d_tok_hist, hist_cap);
}

} // namespace inkling
