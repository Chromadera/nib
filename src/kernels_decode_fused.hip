// ============================================================================
// kernels_decode_fused.hip — Step 7D super-fused decode kernels (gfx1100).
//
// Self-contained: only depends on include/inkling_engine.h (constants/structs).
// Every helper that must produce bit-identical results to the existing engine
// kernels is re-implemented here with the same instruction sequence (see
// include/kernels_decode_fused.h for the kernel inventory).
//
// RDNA3 co-residency rules obeyed by every cooperative barrier in this file:
//   blocks <= 96 * min_blocks_per_cu, all blocks resident, every counter is
//   self-resetting (gfx1100 GL1 coherence: producer does rdna3_flush_stores()
//   before the release atomic, consumer does rdna3_inv_l0_l1() after the spin).
// ============================================================================
#include "kernels_decode_fused.h"

#include <cstdio>
#include <cstdlib>

#include "nib_target.h"

namespace inkling {
namespace {

constexpr int WARP_SIZE = 32;

// ---------------------------------------------------------------------------
// Types (ABI-identical to the engine's StatePtrs / Ptr3 / HalfPtr3 / MutPtr3)
// ---------------------------------------------------------------------------
struct DecStatePtrs { const float * p[MAX_RES_BLOCKS]; };
struct DecPtr3       { const float * p[3]; };
struct DecHalfPtr3   { const __half * p[3]; };
struct DecMutPtr3    { float * p[3]; };
struct DecNormPtr3   { const float * p[3]; };

typedef int v4i __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------------------
// gfx1100 memory-ordering helpers (identical to kernels_core.hip / Step 4)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void rdna3_flush_stores() {
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    // gfx12: no equivalent instruction - on gfx1201 __scoped_atomic_thread_fence(RELEASE, AGENT)
    // above already lowers to s_waitcnt + `global_wb scope:SCOPE_DEV` (ISA-checked with
    // llvm-mc/hipcc -S), so gfx1200/gfx1201 need nothing extra here.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#endif
}

static __device__ __forceinline__ void rdna3_inv_l0_l1() {
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    // gfx12: no equivalent instruction - scope-qualified acquire atomics on gfx1201 emit
    // `global_inv scope:SCOPE_DEV` themselves (ISA-checked), which replaces the RDNA3
    // L0/L1 invalidation below and needs no hand-written cache asm.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("buffer_gl1_inv\n\tbuffer_gl0_inv" ::: "memory");
#endif
}

// ---------------------------------------------------------------------------
// Scratch counter helpers (buf_router_scratch_ is a zero-initialised 512-float
// device buffer; slot indices used by this file are documented per kernel).
// ---------------------------------------------------------------------------
static __device__ __forceinline__ int scratch_add(float * scratch, int slot) {
    return __hip_atomic_fetch_add(reinterpret_cast<int32_t *>(scratch + slot), 1,
                                  __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
}
static __device__ __forceinline__ int scratch_load(float * scratch, int slot) {
    return __hip_atomic_load(reinterpret_cast<int32_t *>(scratch + slot),
                             __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
static __device__ __forceinline__ void scratch_store(float * scratch, int slot, int v) {
    __hip_atomic_store(reinterpret_cast<int32_t *>(scratch + slot), v,
                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

static __device__ __forceinline__ void spin_until(float * scratch, int slot, int target) {
    while (scratch_load(scratch, slot) < target) {
        __builtin_amdgcn_s_sleep(1);
    }
}

// ---------------------------------------------------------------------------
// Exact helpers copied from kernels_core.hip / kernels_gemv_soa.hip
// ---------------------------------------------------------------------------
static __device__ __forceinline__ __half2 h2_of(uint32_t u) {
    return __builtin_bit_cast(__half2, u);
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

static __device__ __forceinline__ float dot_h8_f8v(v4i w, float4 x0, float4 x1, float acc) {
    const __half2 h0 = h2_of((uint32_t) w.x);
    const __half2 h1 = h2_of((uint32_t) w.y);
    const __half2 h2 = h2_of((uint32_t) w.z);
    const __half2 h3 = h2_of((uint32_t) w.w);
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

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = width >> 1; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, width);
    }
    return x;
}

// Exact int8 x FP32 dot (v_fma_mix_f32 path) from kernels_gemv_soa.hip.
static __device__ __forceinline__ float q8_word_dot_mix(uint32_t w, const float4 x, float acc) {
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

// NVFP4 decode helpers (identical to kernels_gemv_soa.hip).
static __device__ __forceinline__ float ue4m3_to_fp32(uint32_t x) {
    const uint32_t u   = (x == 0x7Fu || x == 0xFFu) ? 0u : (x & 0x7Fu);
    const uint32_t e   = u >> 3;
    const uint32_t m   = u & 7u;
    const uint32_t nrm = ((e + 119u) << 23) | (m << 20);
    return (e == 0u) ? ((float) m * (1.0f / 1024.0f)) : __uint_as_float(nrm);
}

static __device__ __forceinline__ float situ_scalar(float gate, float up, float beta, float inv_beta,
                                                    float lb, float inv_lb) {
    const float act = beta * tanhf(gate * inv_beta) / (1.0f + expf(-gate));
    const float u   = lb > 0.0f ? lb * tanhf(up * inv_lb) : up;
    return act * u;
}

static __device__ __forceinline__ float situ_default(float g, float u) {
    return situ_scalar(g, u, SITU_BETA, 1.0f / SITU_BETA, SITU_LINEAR_BETA, 1.0f / SITU_LINEAR_BETA);
}

static __device__ __forceinline__ void nvfp4_hbytes(uint32_t q, uint32_t & lo, uint32_t & hi) {
    constexpr uint32_t T0 = 0x42403C00u;
    constexpr uint32_t T1 = 0x4A484644u;
    lo = __builtin_amdgcn_perm(T1, T0, q & 0x07070707u)        | ((q & 0x08080808u) << 4);
    hi = __builtin_amdgcn_perm(T1, T0, (q >> 4) & 0x07070707u) | (q & 0x80808080u);
}

static __device__ __forceinline__ float nvfp4_hb_dot_mix(uint32_t hb, const float4 x, float acc) {
    const __half2 h01 = __builtin_bit_cast(__half2, __builtin_amdgcn_perm(hb, 0u, 0x050C040Cu));
    const __half2 h23 = __builtin_bit_cast(__half2, __builtin_amdgcn_perm(hb, 0u, 0x070C060Cu));
    acc = fmaf(__low2float(h01),  x.x, acc);
    acc = fmaf(__high2float(h01), x.y, acc);
    acc = fmaf(__low2float(h23),  x.z, acc);
    acc = fmaf(__high2float(h23), x.w, acc);
    return acc;
}

static __device__ __forceinline__ float nvfp4_sub_dot(uint32_t q0, uint32_t q1,
                                                      const float4 x0, const float4 x1,
                                                      const float4 x2, const float4 x3) {
    uint32_t lo0, hi0, lo1, hi1;
    nvfp4_hbytes(q0, lo0, hi0);
    nvfp4_hbytes(q1, lo1, hi1);
    float acc = 0.0f;
    acc = nvfp4_hb_dot_mix(lo0, x0, acc);
    acc = nvfp4_hb_dot_mix(lo1, x1, acc);
    acc = nvfp4_hb_dot_mix(hi0, x2, acc);
    acc = nvfp4_hb_dot_mix(hi1, x3, acc);
    return acc;
}

// Wave32 in-register FWHT-128 (identical to kernels_core.hip).
static __device__ __forceinline__ void wave32_fwht_128_float4(float4 & v, const int lane) {
    const float a0 = v.x + v.y, a1 = v.x - v.y;
    const float a2 = v.z + v.w, a3 = v.z - v.w;
    float r[4] = { a0 + a2, a1 + a3, a0 - a2, a1 - a3 };
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

static __device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

static __device__ __forceinline__ uint32_t pack_u4x4_lo(int q0, int q1, int q2, int q3) {
    return ((uint32_t)(q0 & 0xF)) |
           ((uint32_t)(q1 & 0xF) << 8) |
           ((uint32_t)(q2 & 0xF) << 16) |
           ((uint32_t)(q3 & 0xF) << 24);
}

static __device__ __forceinline__ v4i ld_nt_b128(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const v4i *>(p));
}

// Exact Q8_0/Q4_1/Q4_0 head-major KV store (identical to kernels_core.hip).
static __device__ __forceinline__ void store_kv_quantized(
        float4 v, const KVCacheSlot & kv_slot, bool is_v, int h, int idx, int lane) {
    int8_t * dst_qs = is_v ? kv_slot.v_qs : kv_slot.k_qs;
    __half * dst_d  = is_v ? kv_slot.v_d  : kv_slot.k_d;
    const size_t head_slot = (size_t)h * (size_t)kv_slot.cap + (size_t)idx;
    const int grp = lane >> 3;
    const int sub = lane & 7;

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

// Deterministic Top-8 over 128 expert logits by one Wave32 (identical to kernels_core.hip).
static __device__ __forceinline__ void wave32_topk8_128(
        const float * s_sel, const float * s_p,
        int32_t * topk_ids_out, float * topk_w_out, const int lane) {
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

// Fused 4-tap static short-conv epilogue (identical to kernels_gemv_soa.hip).
struct SconvEpi {
    const float * conv_w;
    float *       hist;
    const float * res;
    float *       raw_out; // Step 8A: optional dump of the raw pre-conv x3
};

static __device__ __forceinline__ float sconv_post_res_one(const SconvEpi & e, int r, float x3) {
    const float4 w  = reinterpret_cast<const float4 *>(e.conv_w)[r];
    float *      hs = e.hist + (size_t) r * N_CONV_HIST;
    const float x0 = hs[0], x1 = hs[1], x2 = hs[2];
    if (e.raw_out != nullptr) {
        e.raw_out[r] = x3;
    }
    const float acc = x3 + x0 * w.x + x1 * w.y + x2 * w.z + x3 * w.w;
    const float val = (e.res != nullptr) ? fmaf(acc, 0.4472135954999579f, e.res[r])
                                         : acc * 0.4472135954999579f;
    hs[0] = x1;
    hs[1] = x2;
    hs[2] = x3;
    return val;
}

// ============================================================================
// A. mix_dual + ctrl_down_qkv3 + ctrl_apply_norm_qkv3  -> one launch
//    grid = 98 blocks x 192 threads
//      bid 0,1     : the two depth mixes (exact inkling_mix_dual_2304<5,4>)
//      bid 2..97   : a = bid-2;  a < 48 -> ctrl_down tile (jd = a/16, 4 rows)
//                                  all   -> ctrl_apply tile (ja = a/32)
//    Scratch slots: apply uses 264..266 + 280..375 (unchanged layout),
//                   384/385 mix flag, 386..388 down flags, 389..391 down acks.
// ============================================================================
constexpr int A_S_MIX_DONE  = 384;
constexpr int A_S_MIX_ACK   = 385;
constexpr int A_S_DOWN_DONE = 386;
constexpr int A_S_DOWN_ACK  = 389;

constexpr int A_TOTAL_BLOCKS    = 96;
constexpr int A_DOWN_PRODUCERS   = 16; // down tiles per branch
constexpr int A_APPLY_TILES      = 32; // apply tiles per branch

// Exact inkling_mix_dual_2304_kernel<5,4> math. Only lanes with tid < 192 take
// part (6 warps, 3 float4 each); the kernel's block-wide barriers are executed
// by all threads at the call site (this body has exactly one internal barrier,
// which is kept outside the `active` guards).
template <int MAX_M, int MAX_Q>
static __device__ __forceinline__ void mix_dual_block(
        const float * __restrict__ query, int n_q,
        DecStatePtrs states, int n_states,
        float * __restrict__ dst, const bool active) {
    constexpr int N_WARPS = 6;
    constexpr int NT      = 192;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __shared__ float smem_sumsq[MAX_M][N_WARPS];
    __shared__ float smem_dot[MAX_Q][MAX_M][N_WARPS];

    float4 s_vec[MAX_M][3];

    if (active) {
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_states) {
                const float4 * sp = (const float4 *)(states.p[m]);
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
    }

    __syncthreads();

    if (active) {
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
            float4 * out4 = (float4 *)(dst + (size_t)q * N_EMBD);
            out4[tid + 0 * NT] = o0;
            out4[tid + 1 * NT] = o1;
            out4[tid + 2 * NT] = o2;
        }
    }
}

// 96 blocks x 288 threads: exactly one block per CU on gfx1100 (96 CUs), so
// every spin barrier below is deadlock-free by construction.
//   bid 0,1               : the two depth mixes (lanes 0..191 only)
//   ja = bid/32, tile = bid%32
//   tile < 16             : ctrl_down tile (jd = ja, ranks 4*tile..4*tile+3)
//   always                : ctrl_apply tile (lanes 0..191 only)
template <bool MROW>
__launch_bounds__(288, 1)
static __global__ void mix_dual_ctrl_qkv3_norm_decode_kernel(
        const float * __restrict__ query0, int n_q0, DecStatePtrs states0, int n_states0,
        float * __restrict__ out0,
        const float * __restrict__ query1, int n_q1, DecStatePtrs states1, int n_states1,
        float * __restrict__ out1,
        DecPtr3 cur, const float * __restrict__ msg_base,
        DecHalfPtr3 w_down, DecHalfPtr3 w_gain, DecHalfPtr3 w_forget, DecNormPtr3 w_norm,
        float * __restrict__ gate, DecMutPtr3 out_norm,
        float eps, float * __restrict__ scratch, int rstride0, int rstride1) {
    // MROW: one launch for all verify rows (grid.y = row); every per-token buffer
    // is shifted by its row, the weights and res_queries are shared.  The body is
    // the exact single-token decode kernel, so the result is bit-identical.
    // rstride0 / rstride1 are the per-row strides of the two mix outputs (4*N_EMBD
    // and 3*N_EMBD for the [row][q][N_EMBD] verify buffers).
    const int bid   = blockIdx.x;
    if constexpr (MROW) {
        const int vrow = (int)blockIdx.y;
        if (vrow > 0) {
            const size_t vof = (size_t)vrow * N_EMBD;
            const size_t o0  = (size_t)vrow * (size_t)rstride0;
            const size_t o1  = (size_t)vrow * (size_t)rstride1;
            for (int i = 0; i < MAX_RES_BLOCKS; ++i) {
                states0.p[i] += vof;
                states1.p[i] += vof;
            }
            out0 += o0;
            msg_base += o0;
            out1 += o1;
            gate += (size_t)vrow * (3 * LOOP_GATE_RANK);   // [row][jd][LOOP_GATE_RANK]
            for (int j = 0; j < 3; ++j) {
                cur.p[j] += o1;
                if (out_norm.p[j] != nullptr) out_norm.p[j] += vof;
            }
            scratch += (size_t)vrow * 512;
        }
    }
    const int tid   = threadIdx.x; // 0..287
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int grp     = tid >> 3; // 0..35 (apply phase uses 0..23)
    const int k8      = tid & 7;
    const bool apply_active = (tid < 192);

    __shared__ float sm_dn[10][9];
    __shared__ alignas(16) float s_gate[LOOP_GATE_RANK];
    __shared__ float smem_ssq[6];
    __shared__ bool  s_is_last;
    __shared__ float s_inv;

    if (bid < 2) {
        if (bid == 0) {
            mix_dual_block<5, 4>(query0, n_q0, states0, n_states0, out0, apply_active);
        } else {
            mix_dual_block<5, 4>(query1, n_q1, states1, n_states1, out1, apply_active);
        }
        rdna3_flush_stores();
        __syncthreads();
        if (tid == 0) {
            (void)scratch_add(scratch, A_S_MIX_DONE);
        }
        // Fall through: bid 0/1 also serve as apply tiles 0/1 of branch 0 so the
        // 96-block grid covers 3 x 32 apply tiles exactly (no extra blocks).
    }

    const int ja      = bid / A_APPLY_TILES;  // 0..2
    const int tile    = bid % A_APPLY_TILES;  // 0..31
    const bool do_down = (tile < A_DOWN_PRODUCERS);
    const int jd      = ja;
    const int rd0     = tile * 4;

    // ---- issue all weight streams before touching the mix barrier ----
    v4i pd0[4], pd1[4];
    if (do_down) {
        const __half * wd = w_down.p[jd];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const size_t base = (size_t)(rd0 + k) * (2 * N_EMBD) + (size_t)tid * 8;
            pd0[k] = ld_nt_b128(wd + base);
            pd1[k] = ld_nt_b128(wd + base + N_EMBD);
        }
    }
    v4i wg[3], wf[3];
    if (apply_active) {
        const __half * wgp = w_gain.p[ja];
        const __half * wfp = w_forget.p[ja];
        const int cb = tile * 72 + grp;
        #pragma unroll
        for (int s = 0; s < 3; ++s) {
            const int c = cb + s * 24;
            wg[s] = ld_nt_b128(wgp + (size_t)c * LOOP_GATE_RANK + (size_t)k8 * 8);
            wf[s] = ld_nt_b128(wfp + (size_t)c * LOOP_GATE_RANK + (size_t)k8 * 8);
        }
    }

    // All blocks (including the two mix blocks, which read each other's mix
    // output for ctrl_down) wait until BOTH depth mixes are published, and all
    // 96 blocks ack so the self-reset can never run before every block has
    // observed the flag (an early reset would spin the mix blocks forever).
    if (tid == 0) {
        spin_until(scratch, A_S_MIX_DONE, 2);
    }
    __syncthreads();
    rdna3_inv_l0_l1();
    if (tid == 0) {
        const int prev = scratch_add(scratch, A_S_MIX_ACK);
        if (prev == A_TOTAL_BLOCKS - 1) {
            scratch_store(scratch, A_S_MIX_DONE, 0);
            scratch_store(scratch, A_S_MIX_ACK, 0);
        }
    }

    // ------------------------- ctrl_down (exact) -------------------------
    if (do_down) {
        const float  * ct = cur.p[jd];
        const float  * mt = msg_base + (size_t)jd * N_EMBD;
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
            dot0[k] = dot_h8_f8v(pd0[k], c0, c1, 0.0f);
            dot1[k] = dot_h8_f8v(pd1[k], m0, m1, 0.0f);
        }

        ssq_c = warp_reduce_sum<WARP_SIZE>(ssq_c);
        ssq_m = warp_reduce_sum<WARP_SIZE>(ssq_m);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            dot0[k] = warp_reduce_sum<WARP_SIZE>(dot0[k]);
            dot1[k] = warp_reduce_sum<WARP_SIZE>(dot1[k]);
        }

        if (lane == 0) {
            sm_dn[0][warp_id] = ssq_c;
            sm_dn[1][warp_id] = ssq_m;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                sm_dn[2 + k][warp_id] = dot0[k];
                sm_dn[6 + k][warp_id] = dot1[k];
            }
        }
        __syncthreads();

        if (tid == 0) {
            float tc = 0.0f, tm = 0.0f;
            float td0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float td1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int w = 0; w < 9; ++w) {
                tc += sm_dn[0][w];
                tm += sm_dn[1][w];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    td0[k] += sm_dn[2 + k][w];
                    td1[k] += sm_dn[6 + k][w];
                }
            }
            const float ic = rsqrtf(tc * (1.0f / (float)N_EMBD) + RMS_EPS);
            const float im = rsqrtf(tm * (1.0f / (float)N_EMBD) + RMS_EPS);
            float * g_out = gate + (size_t)jd * LOOP_GATE_RANK + rd0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                g_out[k] = td0[k] * ic + td1[k] * im;
            }
            rdna3_flush_stores();
            (void)scratch_add(scratch, A_S_DOWN_DONE + jd);
        }
    }

    // ------------------------ ctrl_apply + RMSNorm -----------------------
    if (tid == 0) {
        spin_until(scratch, A_S_DOWN_DONE + ja, A_DOWN_PRODUCERS);
    }
    __syncthreads();
    rdna3_inv_l0_l1();
    if (tid == 0) {
        const int prev = scratch_add(scratch, A_S_DOWN_ACK + ja);
        if (prev == A_APPLY_TILES - 1) {
            scratch_store(scratch, A_S_DOWN_DONE + ja, 0);
            scratch_store(scratch, A_S_DOWN_ACK + ja, 0);
        }
    }
    if (tid < LOOP_GATE_RANK) {
        s_gate[tid] = gate[(size_t)ja * LOOP_GATE_RANK + tid];
    }
    __syncthreads();

    if (apply_active) {
        const float4 g0 = ((const float4 *)s_gate)[k8 * 2 + 0];
        const float4 g1 = ((const float4 *)s_gate)[k8 * 2 + 1];
        const float  * cur_j = cur.p[ja];
        const float  * msg_j = msg_base + (size_t)ja * N_EMBD;
        float        * out_j = out_norm.p[ja];

        float ssq = 0.0f;
        const int c_base = tile * 72 + grp;
        #pragma unroll
        for (int step = 0; step < 3; ++step) {
            const int c = c_base + step * 24;
            float dg = dot_h8_f8v(wg[step], g0, g1, 0.0f);
            float df = dot_h8_f8v(wf[step], g0, g1, 0.0f);
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
    }
    rdna3_flush_stores();
    __syncthreads();

    if (tid == 0) {
        float blk_ssq = 0.0f;
        #pragma unroll
        for (int w = 0; w < 6; ++w) {
            blk_ssq += smem_ssq[w];
        }
        scratch[280 + ja * A_APPLY_TILES + tile] = blk_ssq;
        rdna3_flush_stores();
        int32_t * done_ptr = reinterpret_cast<int32_t *>(scratch + 264 + ja);
        const int prev = __hip_atomic_fetch_add(done_ptr, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_is_last = (prev == A_APPLY_TILES - 1);
    }
    __syncthreads();

    if (s_is_last) {
        rdna3_inv_l0_l1();
        if (warp_id == 0) {
            const float my_ssq = scratch[280 + ja * A_APPLY_TILES + lane];
            const float tot_ssq = warp_reduce_sum<WARP_SIZE>(my_ssq);
            if (lane == 0) {
                int32_t * done_ptr = reinterpret_cast<int32_t *>(scratch + 264 + ja);
                __hip_atomic_store(done_ptr, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                s_inv = rsqrtf(tot_ssq * (1.0f / (float)N_EMBD) + eps);
            }
        }
        __syncthreads();

        if (apply_active) {
            const float inv = s_inv;
            const float * wn_j = w_norm.p[ja];
            float * out_j = out_norm.p[ja];
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
}

// ============================================================================
// B. mix_rms_norm<5,1> + router_topk_decode -> one launch
//    grid = 33 blocks x 192 threads: bid 32 = mix, bid 0..31 = 4 experts each.
//    Scratch: [0..255] p/sel, [256] topk counter, [384]/[385] mix flag/ack.
// ============================================================================
constexpr int B_S_MIX_DONE = 384;
constexpr int B_S_MIX_ACK  = 385;

template <bool MROW>
__launch_bounds__(192, 4)
static __global__ void mix_ffn_norm_router_topk_fused_v2_kernel(
        const float * __restrict__ res_query,
        DecStatePtrs states, int n_states,
        const float * __restrict__ ffn_norm_w,
        const __half * __restrict__ router_w,
        const float * __restrict__ exp_probs_b,
        float * __restrict__ x_norm_out,
        int32_t * __restrict__ topk_ids_out,
        float * __restrict__ topk_w_out,
        float * __restrict__ scratch) {
    if constexpr (MROW) {
        const int vrow = (int)blockIdx.y;
        if (vrow > 0) {
            const size_t vof = (size_t)vrow * N_EMBD;
            for (int i = 0; i < MAX_RES_BLOCKS; ++i) states.p[i] += vof;
            x_norm_out += vof;
            topk_ids_out += (size_t)vrow * TOP_K;
            topk_w_out += (size_t)vrow * TOP_K;
            scratch += (size_t)vrow * 512;
        }
    }
    constexpr int N_WARPS = 6;
    constexpr int NT      = 192;

    const int bid     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    __shared__ float4 s_x[N_EMBD / 4];
    __shared__ float  lds_p[N_EXPERT];
    __shared__ float  lds_sel[N_EXPERT];
    __shared__ bool   s_last;

    if (bid == 32) {
        // ------- exact inkling_mix_rms_norm_2304_kernel<5,1> -------
        constexpr int MAX_M = 5;
        __shared__ float smem_sumsq[MAX_M][N_WARPS];
        __shared__ float smem_dot[1][MAX_M][N_WARPS];
        __shared__ float smem_out_ssq[1][N_WARPS];

        float4 s_vec[MAX_M][3];
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_states) {
                const float4 * sp = (const float4 *)(states.p[m]);
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
                if (lane == 0) {
                    smem_sumsq[m][warp_id] = ssq;
                }
            }
        }

        {
            const float4 * qp = (const float4 *)(res_query);
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
                    if (lane == 0) {
                        smem_dot[0][m][warp_id] = d;
                    }
                }
            }
        }

        __syncthreads();

        constexpr float inv_n    = 1.0f / (float)N_EMBD;
        const float     inv_sq_n = rsqrtf((float)N_EMBD);

        float my_inv_rms = 0.0f;
        if (lane < n_states) {
            float tot = 0.0f;
            #pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                tot += smem_sumsq[lane][w];
            }
            my_inv_rms = rsqrtf(tot * inv_n + RMS_EPS);
        }

        float4 ov_q0[3];
        {
            float sc = -INFINITY;
            if (lane < n_states) {
                float tot_d = 0.0f;
                #pragma unroll
                for (int w = 0; w < N_WARPS; ++w) {
                    tot_d += smem_dot[0][lane][w];
                }
                sc = tot_d * my_inv_rms * inv_sq_n;
            }
            float max_s = sc;
            max_s = fmaxf(max_s, __shfl_xor(max_s, 1, 8));
            max_s = fmaxf(max_s, __shfl_xor(max_s, 2, 8));
            max_s = fmaxf(max_s, __shfl_xor(max_s, 4, 8));
            const float ex = (lane < n_states) ? expf(sc - max_s) : 0.0f;
            float sum_e = ex;
            sum_e += __shfl_xor(sum_e, 1, 8);
            sum_e += __shfl_xor(sum_e, 2, 8);
            sum_e += __shfl_xor(sum_e, 4, 8);
            const float my_effw = (lane < n_states) ? (ex * (1.0f / sum_e) * my_inv_rms) : 0.0f;

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
            ov_q0[0] = o0; ov_q0[1] = o1; ov_q0[2] = o2;
            float ssq_o = (o0.x * o0.x + o0.y * o0.y + o0.z * o0.z + o0.w * o0.w)
                        + (o1.x * o1.x + o1.y * o1.y + o1.z * o1.z + o1.w * o1.w)
                        + (o2.x * o2.x + o2.y * o2.y + o2.z * o2.z + o2.w * o2.w);
            ssq_o = warp_reduce_sum<WARP_SIZE>(ssq_o);
            if (lane == 0) {
                smem_out_ssq[0][warp_id] = ssq_o;
            }
        }

        __syncthreads();

        {
            float tot = 0.0f;
            #pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                tot += smem_out_ssq[0][w];
            }
            const float inv = rsqrtf(tot * inv_n + RMS_EPS);

            const float4 w0 = ffn_norm_w ? ((const float4 *)ffn_norm_w)[tid + 0 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float4 w1 = ffn_norm_w ? ((const float4 *)ffn_norm_w)[tid + 1 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float4 w2 = ffn_norm_w ? ((const float4 *)ffn_norm_w)[tid + 2 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);

            float4 o0 = ov_q0[0], o1 = ov_q0[1], o2 = ov_q0[2];
            o0.x *= inv * w0.x; o0.y *= inv * w0.y; o0.z *= inv * w0.z; o0.w *= inv * w0.w;
            o1.x *= inv * w1.x; o1.y *= inv * w1.y; o1.z *= inv * w1.z; o1.w *= inv * w1.w;
            o2.x *= inv * w2.x; o2.y *= inv * w2.y; o2.z *= inv * w2.z; o2.w *= inv * w2.w;
            float4 * out4 = (float4 *)x_norm_out;
            out4[tid + 0 * NT] = o0;
            out4[tid + 1 * NT] = o1;
            out4[tid + 2 * NT] = o2;
        }

        rdna3_flush_stores();
        __syncthreads();
        if (tid == 0) {
            (void)scratch_add(scratch, B_S_MIX_DONE);
        }
        return;
    }

    // ------------------------- router block -------------------------
    const int e0 = bid * 4;
    v4i rw[9];
    if (warp_id < 4) {
        const __half * rbase = router_w + (size_t)(e0 + warp_id) * N_EMBD;
        #pragma unroll
        for (int s = 0; s < 9; ++s) {
            rw[s] = ld_nt_b128(rbase + (size_t)(lane + s * WARP_SIZE) * 8);
        }
    }

    if (tid == 0) {
        spin_until(scratch, B_S_MIX_DONE, 1);
    }
    __syncthreads();
    rdna3_inv_l0_l1();
    if (tid == 0) {
        const int prev = scratch_add(scratch, B_S_MIX_ACK);
        if (prev == 31) {
            scratch_store(scratch, B_S_MIX_DONE, 0);
            scratch_store(scratch, B_S_MIX_ACK, 0);
        }
    }

    #pragma unroll
    for (int k = 0; k < 3; ++k) {
        s_x[tid + k * NT] = ((const float4 *)x_norm_out)[tid + k * NT];
    }
    __syncthreads();
    __builtin_amdgcn_sched_barrier(0);

    if (warp_id < 4) {
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < 9; ++s) {
            const int i8 = lane + s * WARP_SIZE;
            acc = dot_h8_f8v(rw[s], s_x[i8 * 2 + 0], s_x[i8 * 2 + 1], acc);
        }
        const float sum = warp_reduce_sum<WARP_SIZE>(acc);
        if (lane == 0) {
            const int e = e0 + warp_id;
            const float p   = 1.0f / (1.0f + expf(-sum));
            const float sel = exp_probs_b ? (p + exp_probs_b[e]) : p;
            __hip_atomic_store(&scratch[e],            p,   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&scratch[N_EXPERT + e], sel, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
    rdna3_flush_stores();
    __syncthreads();

    if (tid == 0) {
        rdna3_flush_stores();
        const int prev = scratch_add(scratch, 256);
        s_last = (prev == 31);
        if (s_last) {
            scratch_store(scratch, 256, 0);
        }
    }
    __syncthreads();

    if (s_last && warp_id == 0) {
        rdna3_inv_l0_l1();
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + j * 32;
            lds_p[idx]   = __hip_atomic_load(&scratch[idx],            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            lds_sel[idx] = __hip_atomic_load(&scratch[N_EXPERT + idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        wave32_topk8_128(lds_sel, lds_p, topk_ids_out, topk_w_out, lane);
    }
}

// ============================================================================
// C. mix_ctrl_down + ctrl_apply + FFN RMSNorm + router + Top-8 -> one launch
//    grid = 32 blocks x 288 threads.
//    Scratch: [0..255] p/sel, [256] topk, [280..311] ssq, [384..389] flags.
// ============================================================================
constexpr int C_S_GATE_DONE = 384;
constexpr int C_S_GATE_ACK  = 385;
constexpr int C_S_SSQ_DONE  = 386;
constexpr int C_S_SSQ_ACK   = 387;
constexpr int C_S_XN_DONE   = 388;
constexpr int C_S_XN_ACK    = 389;

template <int MAX_M, bool MROW>
__launch_bounds__(288, 3)
static __global__ void mix_ctrl_ffn_norm_router_topk_fused_v2_kernel(
        const float * __restrict__ res_query,
        DecStatePtrs states, int n_states,
        const float * __restrict__ msg,
        const __half * __restrict__ w_down,
        const __half * __restrict__ w_gain,
        const __half * __restrict__ w_forget,
        const float * __restrict__ ffn_norm_w,
        const __half * __restrict__ router_w,
        const float * __restrict__ exp_probs_b,
        float * __restrict__ out_mixed,
        float * __restrict__ out_gate,
        float * __restrict__ x_norm_out,
        int32_t * __restrict__ topk_ids_out,
        float * __restrict__ topk_w_out,
        float * __restrict__ scratch, int msg_rstride) {
    if constexpr (MROW) {
        const int vrow = (int)blockIdx.y;
        if (vrow > 0) {
            const size_t vof = (size_t)vrow * N_EMBD;
            for (int i = 0; i < MAX_RES_BLOCKS; ++i) states.p[i] += vof;
            msg += (size_t)vrow * (size_t)msg_rstride;
            out_mixed += vof;
            out_gate += (size_t)vrow * LOOP_GATE_RANK;
            x_norm_out += vof;
            topk_ids_out += (size_t)vrow * TOP_K;
            topk_w_out += (size_t)vrow * TOP_K;
            scratch += (size_t)vrow * 512;
        }
    }
    constexpr int N_WARPS = 9;
    const int bid     = blockIdx.x;  // 0..31
    const int tid     = threadIdx.x; // 0..287
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int grp     = tid >> 3;    // 0..35
    const int k8      = tid & 7;

    __shared__ float4 s_xn[N_EMBD / 4];
    __shared__ float  lds_p[N_EXPERT];
    __shared__ float  lds_sel[N_EXPERT];
    __shared__ bool   s_last;

    // ---------------- exact mix_ctrl_down_decode_kernel<5> ----------------
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
            if (lane == 0) {
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
    if (lane == 0) {
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
    rdna3_flush_stores();
    __syncthreads();
    if (tid == 0) {
        (void)scratch_add(scratch, C_S_GATE_DONE);
    }

    // ---- prefetch the apply weights + this block's 4 router rows ----
    v4i wg[2], wf[2];
    {
        const int cb = bid * 72 + grp;
        #pragma unroll
        for (int s = 0; s < 2; ++s) {
            const int c = cb + s * 36;
            wg[s] = ld_nt_b128(w_gain + (size_t)c * LOOP_GATE_RANK + (size_t)k8 * 8);
            wf[s] = ld_nt_b128(w_forget + (size_t)c * LOOP_GATE_RANK + (size_t)k8 * 8);
        }
    }
    const int e0 = bid * 4;
    v4i rw[9];
    if (warp_id < 4) {
        const __half * rbase = router_w + (size_t)(e0 + warp_id) * N_EMBD;
        #pragma unroll
        for (int s = 0; s < 9; ++s) {
            rw[s] = ld_nt_b128(rbase + (size_t)(lane + s * WARP_SIZE) * 8);
        }
    }

    // ---- gate barrier (32 blocks) ----
    if (tid == 0) {
        spin_until(scratch, C_S_GATE_DONE, 32);
    }
    __syncthreads();
    rdna3_inv_l0_l1();
    if (tid == 0) {
        const int prev = scratch_add(scratch, C_S_GATE_ACK);
        if (prev == 31) {
            scratch_store(scratch, C_S_GATE_DONE, 0);
            scratch_store(scratch, C_S_GATE_ACK, 0);
        }
    }

    __shared__ alignas(16) float s_gate[LOOP_GATE_RANK];
    if (tid < LOOP_GATE_RANK) {
        s_gate[tid] = out_gate[tid];
    }
    __syncthreads();

    const float4 g0 = ((const float4 *)s_gate)[k8 * 2 + 0];
    const float4 g1 = ((const float4 *)s_gate)[k8 * 2 + 1];

    // --------------------- ctrl_apply + ssq (32 slots) ---------------------
    __shared__ float s_block_ssq[N_WARPS];
    float val0 = 0.0f, val1 = 0.0f;
    {
        float ssq = 0.0f;
        const int c_base = bid * 72 + grp;
        #pragma unroll
        for (int step = 0; step < 2; ++step) {
            const int c = c_base + step * 36;
            float dg = dot_h8_f8v(wg[step], g0, g1, 0.0f);
            float df = dot_h8_f8v(wf[step], g0, g1, 0.0f);
            dg = warp_reduce_sum<8>(dg);
            df = warp_reduce_sum<8>(df);
            if (k8 == 0) {
                const float gain   = tanhf(dg);
                const float forget = tanhf(df);
                const float cv     = out_mixed[c];
                const float mv     = msg[c];
                const float val    = cv - forget * cv + gain * mv;
                if (step == 0) val0 = val; else val1 = val;
                ssq += val * val;
            }
        }
        ssq = warp_reduce_sum<WARP_SIZE>(ssq);
        if (lane == 0) {
            s_block_ssq[warp_id] = ssq;
        }
    }
    rdna3_flush_stores();
    __syncthreads();

    if (tid == 0) {
        float blk = 0.0f;
        #pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            blk += s_block_ssq[w];
        }
        scratch[280 + bid] = blk;
        rdna3_flush_stores();
        (void)scratch_add(scratch, C_S_SSQ_DONE);
        spin_until(scratch, C_S_SSQ_DONE, 32);
    }
    __syncthreads();
    rdna3_inv_l0_l1();
    if (tid == 0) {
        const int prev = scratch_add(scratch, C_S_SSQ_ACK);
        if (prev == 31) {
            scratch_store(scratch, C_S_SSQ_DONE, 0);
            scratch_store(scratch, C_S_SSQ_ACK, 0);
        }
    }

    float tot_ssq;
    {
        const int idx = tid & 31;
        const float partial = __hip_atomic_load(&scratch[280 + idx],
                                                __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        tot_ssq = warp_reduce_sum<WARP_SIZE>(partial);
    }
    const float inv = rsqrtf(tot_ssq * (1.0f / (float)N_EMBD) + RMS_EPS);

    // ---- write this block's normalized channels (own channels only) ----
    if (k8 == 0) {
        const int c_base = bid * 72 + grp;
        const float w0v = ffn_norm_w ? ffn_norm_w[c_base] : 1.0f;
        x_norm_out[c_base] = val0 * inv * w0v;
        const int c1v = c_base + 36;
        const float w1v = ffn_norm_w ? ffn_norm_w[c1v] : 1.0f;
        x_norm_out[c1v] = val1 * inv * w1v;
    }
    rdna3_flush_stores();
    __syncthreads();

    // ---- second barrier: every x_norm_out channel is visible ----
    if (tid == 0) {
        (void)scratch_add(scratch, C_S_XN_DONE);
        spin_until(scratch, C_S_XN_DONE, 32);
    }
    __syncthreads();
    rdna3_inv_l0_l1();
    if (tid == 0) {
        const int prev = scratch_add(scratch, C_S_XN_ACK);
        if (prev == 31) {
            scratch_store(scratch, C_S_XN_DONE, 0);
            scratch_store(scratch, C_S_XN_ACK, 0);
        }
    }

    #pragma unroll
    for (int k = 0; k < 2; ++k) {
        s_xn[tid + k * 288] = ((const float4 *)x_norm_out)[tid + k * 288];
    }
    __syncthreads();
    __builtin_amdgcn_sched_barrier(0);

    // ------------------------------ router ------------------------------
    if (warp_id < 4) {
        float acc = 0.0f;
        #pragma unroll
        for (int s = 0; s < 9; ++s) {
            const int i8 = lane + s * WARP_SIZE;
            acc = dot_h8_f8v(rw[s], s_xn[i8 * 2 + 0], s_xn[i8 * 2 + 1], acc);
        }
        const float sum = warp_reduce_sum<WARP_SIZE>(acc);
        if (lane == 0) {
            const int e = e0 + warp_id;
            const float p   = 1.0f / (1.0f + expf(-sum));
            const float sel = exp_probs_b ? (p + exp_probs_b[e]) : p;
            __hip_atomic_store(&scratch[e],            p,   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            __hip_atomic_store(&scratch[N_EXPERT + e], sel, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
    }
    rdna3_flush_stores();
    __syncthreads();

    if (tid == 0) {
        rdna3_flush_stores();
        const int prev = scratch_add(scratch, 256);
        s_last = (prev == 31);
        if (s_last) {
            scratch_store(scratch, 256, 0);
        }
    }
    __syncthreads();

    if (s_last && warp_id == 0) {
        rdna3_inv_l0_l1();
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + j * 32;
            lds_p[idx]   = __hip_atomic_load(&scratch[idx],            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            lds_sel[idx] = __hip_atomic_load(&scratch[N_EXPERT + idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        wave32_topk8_128(lds_sel, lds_p, topk_ids_out, topk_w_out, lane);
    }
}

// ============================================================================
// D/F. SoA Q8_0 GEMV (K = 2304 / 3072), exact same per-lane block order.
//      D: one block per 16-row group (NWARPS=4, LPR=8, PD=3, RPB=16).
//      F: one block per 6-row group (NWARPS=6, LPR=32, PD=3, RPB=6), wo only.
// ============================================================================
struct QkvSeg {
    const int8_t * qs;
    const __half * d;
    const float *  x;
    float *        y;
    int            rows;
    int            blk_end;
};

static __device__ __forceinline__ int lay_x(int i) { return (i >> 3) * 9 + (i & 7); }

template <int LPR, int NB, int PD>
struct GemvRing {
    static constexpr int ITS = NB / LPR;
    static_assert(ITS % PD == 0, "ring depth must divide the blocks per lane");
    v4i    r0[PD], r1[PD];
    __half rh[PD];

    __device__ __forceinline__ void load(int slot, const int8_t * wq, const __half * wd, int kb) {
        r0[slot] = ld_nt_b128(wq + kb * 32);
        r1[slot] = ld_nt_b128(wq + kb * 32 + 16);
        rh[slot] = wd[kb];
    }

    __device__ __forceinline__ void prime(const int8_t * __restrict__ qs,
                                          const __half * __restrict__ dd,
                                          int rowc, int sub) {
        const int8_t * wq = qs + (size_t) rowc * (size_t) (NB * 32);
        const __half * wd = dd + (size_t) rowc * (size_t) NB;
        #pragma unroll
        for (int p = 0; p < PD; ++p) {
            if (p < ITS) {
                load(p, wq, wd, sub + p * LPR);
            }
        }
    }

    __device__ __forceinline__ float run(const int8_t * __restrict__ qs,
                                         const __half * __restrict__ dd,
                                         int rowc, int nrowc, bool has_next,
                                         const float4 * s_x, int sub) {
        const int8_t * wq  = qs + (size_t) rowc * (size_t) (NB * 32);
        const __half * wd  = dd + (size_t) rowc * (size_t) NB;
        const int8_t * nwq = qs + (size_t) nrowc * (size_t) (NB * 32);
        const __half * nwd = dd + (size_t) nrowc * (size_t) NB;
        float acc = 0.0f;
        int   xo  = 0;
        #pragma unroll
        for (int it = 0; it < ITS; ++it) {
            asm volatile("" : "+v"(xo), "+v"(acc));
            const int slot = it % PD;
            const v4i w0 = r0[slot], w1 = r1[slot];
            const float sc = __half2float(rh[slot]);
            if (it + PD < ITS) {
                load(slot, wq, wd, sub + (it + PD) * LPR + xo);
            } else if (has_next) {
                load(slot, nwq, nwd, sub + (it + PD - ITS) * LPR + xo);
            }
            const float4 * xb = s_x + (sub + it * LPR + xo) * 9;
            float p = 0.0f, q = 0.0f;
            p = q8_word_dot_mix((uint32_t) w0.x, xb[0], p);
            q = q8_word_dot_mix((uint32_t) w1.x, xb[4], q);
            p = q8_word_dot_mix((uint32_t) w0.y, xb[1], p);
            q = q8_word_dot_mix((uint32_t) w1.y, xb[5], q);
            p = q8_word_dot_mix((uint32_t) w0.z, xb[2], p);
            q = q8_word_dot_mix((uint32_t) w1.z, xb[6], q);
            p = q8_word_dot_mix((uint32_t) w0.w, xb[3], p);
            q = q8_word_dot_mix((uint32_t) w1.w, xb[7], q);
            acc = fmaf(p + q, sc, acc);
        }
        return acc;
    }
};

template <int NWARPS, int LPR, int NB, int PD, bool EPI>
static __device__ __forceinline__ void q8_gemv_body(
        QkvSeg s0, QkvSeg s1, QkvSeg s2, float4 * s_x, const SconvEpi & epi) {
    constexpr int NT  = NWARPS * 32;
    constexpr int K   = NB * 32;
    constexpr int RPW = 32 / LPR;
    constexpr int RPB = NWARPS * RPW;
    constexpr int NX4 = K / 4;
    constexpr int PER = (NX4 + NT - 1) / NT;

    const int b  = blockIdx.x;
    const int si = (b >= s0.blk_end) + (b >= s1.blk_end);
    const QkvSeg & sg = si == 0 ? s0 : (si == 1 ? s1 : s2);
    const int8_t * __restrict__ qs = sg.qs;
    const __half * __restrict__ dd = sg.d;
    float * __restrict__ y = sg.y;
    const int rows = sg.rows;
    const int b0   = si == 0 ? 0 : (si == 1 ? s0.blk_end : s1.blk_end);
    const int lb   = b - b0;
    const int nblk = sg.blk_end - b0;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sub  = lane % LPR;
    const int rloc = lane / LPR;
    const int rofs = warp * RPW + rloc;
    const int step = nblk * RPB;

    const float4 * __restrict__ x4 = reinterpret_cast<const float4 *>(sg.x);
    float4 xv[PER];
    #pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * NT;
        xv[i] = x4[idx < NX4 ? idx : NX4 - 1];
    }
    int row0 = lb * RPB;
    GemvRing<LPR, NB, PD> ring;
    {
        const int row = row0 + rofs;
        ring.prime(qs, dd, row < rows ? row : rows - 1, sub);
    }
    #pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * NT;
        if (idx < NX4) {
            s_x[lay_x(idx)] = xv[i];
        }
    }
    __syncthreads();

    for (; row0 < rows; row0 += step) {
        const int  row      = row0 + rofs;
        const bool valid    = row < rows;
        const bool has_next = row0 + step < rows;
        const int  nrow     = row + step;
        float acc = ring.run(qs, dd, valid ? row : rows - 1, nrow < rows ? nrow : rows - 1,
                             has_next, s_x, sub);
        acc = warp_reduce_sum<LPR>(acc);
        if (sub == 0 && valid) {
            if (EPI && si == 0) {
                y[row] = sconv_post_res_one(epi, row, acc);
            } else {
                y[row] = acc;
            }
        }
    }
}

template <int NWARPS, int LPR, int NB, int PD>
__launch_bounds__(NWARPS * 32, 4)
static __global__ void q8_gemv_qkv_v2_kernel(QkvSeg s0, QkvSeg s1, QkvSeg s2) {
    __shared__ float4 s_x[NB * 9];
    const SconvEpi none = {nullptr, nullptr, nullptr, nullptr};
    q8_gemv_body<NWARPS, LPR, NB, PD, false>(s0, s1, s2, s_x, none);
}

template <int NWARPS, int LPR, int NB, int PD>
__launch_bounds__(NWARPS * 32, 4)
static __global__ void q8_gemv_wo_v2_kernel(QkvSeg s0, SconvEpi epi) {
    __shared__ float4 s_x[NB * 9];
    QkvSeg none;
    none.qs = nullptr; none.d = nullptr; none.x = nullptr; none.y = nullptr;
    none.rows = 0; none.blk_end = 0x7FFFFFFF;
    q8_gemv_body<NWARPS, LPR, NB, PD, true>(s0, none, none, s_x, epi);
}

// ============================================================================
// E. QKV GEMV + qkvr_prep fused (288 blocks x 128 threads, sharded grid barrier)
//    Scratch: [392..394] barrier shards, [395] barrier ack (self-resetting).
// ============================================================================
template <int NWARPS, int LPR, int NB, int PD, int GRID = 288>
__launch_bounds__(NWARPS * 32, 3)
static __global__ void qkv_qkvr_prep_fused_decode_kernel(
        QkvSeg s0, QkvSeg s1, QkvSeg s2,
        const float * __restrict__ q_in, const float * __restrict__ q_clip_w,
        const float * __restrict__ r_in, const float * __restrict__ r_clip_w,
        const float * __restrict__ rel_bias_w,
        const float * __restrict__ k_raw, const float * __restrict__ k_conv_w,
        const float * __restrict__ k_dyn,
        const float * __restrict__ v_raw, const float * __restrict__ v_conv_w,
        const float * __restrict__ v_dyn,
        float * __restrict__ sconv_hist_kv,
        KVCacheSlot kv_slot,
        float * __restrict__ q_out,
        float * __restrict__ rel_table_out,
        bool apply_tau,
        const StepMeta * __restrict__ d_meta,
        float * __restrict__ scratch) {
    constexpr int N_RT_BLOCKS = GRID - 40; // blocks 40..GRID-1
    const int bid  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;

    __shared__ float4 s_x[NB * 9];
    __shared__ alignas(16) float s_r[N_HEAD][N_REL];
    __shared__ alignas(16) float s_b[8][N_REL + 1];

    const SconvEpi none = {nullptr, nullptr, nullptr, nullptr};
    q8_gemv_body<NWARPS, LPR, NB, PD, false>(s0, s1, s2, s_x, none);

    // ------------------------- sharded grid barrier -------------------------
    rdna3_flush_stores();
    __syncthreads();
    if (tid == 0) {
        rdna3_flush_stores();
        (void)scratch_add(scratch, 392 + (bid % 3));
    }
    __syncthreads();
    if (tid == 0) {
        spin_until(scratch, 392, GRID / 3);
        spin_until(scratch, 393, GRID / 3);
        spin_until(scratch, 394, GRID / 3);
    }
    __syncthreads();
    rdna3_inv_l0_l1();

    // ------------------------------ qkvr prep ------------------------------
    if (bid < 24) {
        if (tid < 32) {
            const int h = bid;
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
        }
    } else if (bid < 40) {
        if (tid < 32) {
            const bool is_v = (bid >= 32);
            const int  h    = is_v ? (bid - 32) : (bid - 24);
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
        }
    } else {
        const int rid   = bid - 40;
        const int d_lo  = (rid * N_REL_EXTENT) / N_RT_BLOCKS;
        const int d_hi  = ((rid + 1) * N_REL_EXTENT) / N_RT_BLOCKS;
        const int d_cnt = d_hi - d_lo;

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
        for (int idx = tid; idx < d_cnt * N_REL; idx += NWARPS * 32) {
            const int dl = idx / N_REL;
            const int k  = idx % N_REL;
            s_b[dl][k] = rel_bias_w[(size_t)(d_lo + dl) * N_REL + k];
        }
        __syncthreads();

        for (int idx = tid; idx < d_cnt * N_HEAD; idx += NWARPS * 32) {
            const int dl = idx / N_HEAD;
            const int h  = idx % N_HEAD;
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < N_REL; ++k) {
                dot = fmaf(s_b[dl][k], s_r[h][k], dot);
            }
            rel_table_out[(size_t)h * N_REL_EXTENT + d_lo + dl] = dot;
        }
    }

    // ------------------------- reset the barrier -------------------------
    rdna3_flush_stores();
    __syncthreads();
    if (tid == 0) {
        const int prev = scratch_add(scratch, 395);
        if (prev == GRID - 1) {
            scratch_store(scratch, 392, 0);
            scratch_store(scratch, 393, 0);
            scratch_store(scratch, 394, 0);
            scratch_store(scratch, 395, 0);
        }
    }
}

// ============================================================================
// G. NVFP4 MoE gate_up + SITU, deeper register ring (default PD = 3).
// ============================================================================
constexpr int NVFP4_GU_LPR = 4;

static __device__ __forceinline__ void nvfp4_down_load_row(
        const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
        int ex, int g, int r,
        v4i & w0, v4i & w1, uint32_t & w2, uint16_t & sc0, uint16_t & sc1, uint8_t & sc2) {
    constexpr int QSR = N_FF_EXP / 2;
    constexpr int SCR = N_FF_EXP / 16;
    const size_t row = (size_t) ex * N_EMBD + r;
    const uint8_t  * q  = qs + row * QSR;
    const uint16_t * sc = reinterpret_cast<const uint16_t *>(scales + row * SCR);
    w0  = ld_nt_b128(q + g * 16);
    w1  = ld_nt_b128(q + (16 + g) * 16);
    w2  = __builtin_nontemporal_load(reinterpret_cast<const uint32_t *>(q + 512) + g);
    sc0 = sc[g];
    sc1 = sc[16 + g];
    sc2 = (scales + row * SCR)[64 + (g >> 1)];
}

template <int NWARPS, int PD>
__launch_bounds__(NWARPS * 32, 4) __attribute__((amdgpu_waves_per_eu(4, 4)))
static __global__ void moe_nvfp4_gate_up_v2_kernel(
        const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
        const float * __restrict__ scale2, const float * __restrict__ x_norm,
        const int32_t * __restrict__ topk_ids, float * __restrict__ act) {
    constexpr int NT  = NWARPS * 32;
    constexpr int K   = N_EMBD;
    constexpr int NB  = K / 64;              // 36
    constexpr int QSR = K / 2;
    constexpr int SCR = K / 16;
    constexpr int PPW = 32 / NVFP4_GU_LPR;   // 8
    constexpr int ITS = NB / NVFP4_GU_LPR;   // 9
    static_assert(PD >= 1 && PD <= 2, "explicit ring rotation");

    __shared__ float4 s_x[NB * 17];
    {
        constexpr int N4  = K / 4;
        constexpr int PER = (N4 + NT - 1) / NT;
        float4 v[PER];
        const float4 * src = reinterpret_cast<const float4 *>(x_norm);
        #pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int idx = threadIdx.x + i * NT;
            v[i] = src[idx < N4 ? idx : N4 - 1];
        }
        #pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int idx = threadIdx.x + i * NT;
            if (idx < N4) {
                s_x[(idx / 16) * 17 + (idx % 16)] = v[i];
            }
        }
    }
    __syncthreads();

    const int k     = blockIdx.y;
    const int expr  = topk_ids[k];
    const bool eok  = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex    = eok ? expr : 0;
    const float s2  = eok ? scale2[ex] : 0.0f;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g    = lane % NVFP4_GU_LPR;
    const int r    = blockIdx.x * (NWARPS * PPW) + warp * PPW + lane / NVFP4_GU_LPR;

    const size_t row_g = (size_t) ex * N_GATE_UP_EXP + r;
    const size_t row_u = row_g + N_FF_EXP;
    const uint8_t  * qg = qs + row_g * QSR;
    const uint8_t  * qu = qs + row_u * QSR;
    const uint32_t * sg = reinterpret_cast<const uint32_t *>(scales + row_g * SCR);
    const uint32_t * su = reinterpret_cast<const uint32_t *>(scales + row_u * SCR);

    static_assert(PD >= 1 && PD <= 2, "explicit ring rotation");
    struct Ent { v4i g0, g1, u0, u1; uint32_t sgw, suw; };
    Ent ring[PD];
    #pragma unroll
    for (int p = 0; p < PD; ++p) {
        const int b = g + p * NVFP4_GU_LPR;
        ring[p].g0 = ld_nt_b128(qg + b * 32);
        ring[p].g1 = ld_nt_b128(qg + b * 32 + 16);
        ring[p].u0 = ld_nt_b128(qu + b * 32);
        ring[p].u1 = ld_nt_b128(qu + b * 32 + 16);
        ring[p].sgw = sg[b];
        ring[p].suw = su[b];
    }

    float accg = 0.0f;
    float accu = 0.0f;
#pragma unroll 1
    for (int it = 0; it < ITS; it += PD) {
#pragma unroll
        for (int p = 0; p < PD; ++p) {
            const int b = g + (it + p) * NVFP4_GU_LPR;
            if (it + p < ITS) {
                const v4i g0 = ring[p].g0, g1 = ring[p].g1;
                const v4i u0 = ring[p].u0, u1 = ring[p].u1;
                const uint32_t sgw = ring[p].sgw, suw = ring[p].suw;
                __builtin_amdgcn_sched_barrier(0);
                if (it + p + PD < ITS) {
                    const int bn = g + (it + p + PD) * NVFP4_GU_LPR;
                    ring[p].g0 = ld_nt_b128(qg + bn * 32);
                    ring[p].g1 = ld_nt_b128(qg + bn * 32 + 16);
                    ring[p].u0 = ld_nt_b128(qu + bn * 32);
                    ring[p].u1 = ld_nt_b128(qu + bn * 32 + 16);
                    ring[p].sgw = sg[bn];
                    ring[p].suw = su[bn];
                }
                const float4 * xb = s_x + b * 17;
#pragma unroll
                for (int s = 0; s < 4; ++s) {
                    const float4 x0 = xb[s * 4 + 0];
                    const float4 x1 = xb[s * 4 + 1];
                    const float4 x2 = xb[s * 4 + 2];
                    const float4 x3 = xb[s * 4 + 3];
                    const uint32_t gq0 = (uint32_t) (s < 2 ? (s == 0 ? g0.x : g0.z) : (s == 2 ? g1.x : g1.z));
                    const uint32_t gq1 = (uint32_t) (s < 2 ? (s == 0 ? g0.y : g0.w) : (s == 2 ? g1.y : g1.w));
                    const uint32_t uq0 = (uint32_t) (s < 2 ? (s == 0 ? u0.x : u0.z) : (s == 2 ? u1.x : u1.z));
                    const uint32_t uq1 = (uint32_t) (s < 2 ? (s == 0 ? u0.y : u0.w) : (s == 2 ? u1.y : u1.w));
                    accg = fmaf(nvfp4_sub_dot(gq0, gq1, x0, x1, x2, x3), ue4m3_to_fp32((sgw >> (8 * s)) & 0xFFu), accg);
                    accu = fmaf(nvfp4_sub_dot(uq0, uq1, x0, x1, x2, x3), ue4m3_to_fp32((suw >> (8 * s)) & 0xFFu), accu);
                }
            }
        }
    }
    accg = warp_reduce_sum<NVFP4_GU_LPR>(accg);
    accu = warp_reduce_sum<NVFP4_GU_LPR>(accu);
    if (g == 0) {
        act[k * N_FF_EXP + r] = eok ? situ_default(accg * s2, accu * s2) : 0.0f;
    }
}

// ============================================================================
// H. NVFP4 MoE down + weighted 8-expert sum, 2-deep row ring.
// ============================================================================
template <int R, int GRID_BLOCKS>
__launch_bounds__(256, 2)
static __global__ void moe_nvfp4_down_v2_kernel(
        const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
        const float * __restrict__ scale2, const float * __restrict__ act,
        const int32_t * __restrict__ topk_ids, const float * __restrict__ topk_w,
        float * __restrict__ out, const SconvEpi epi) {
    constexpr int K   = N_FF_EXP;   // 1152
    constexpr int RPB = 2 * R;

    __shared__ float4 s_act1[8][4][16];
    __shared__ float4 s_act2[8][2][16];
    __shared__ float  s_red[2][8][RPB];

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k    = warp;
    const int g    = lane & 15;
    const int rloc = lane >> 4;

    const int expr   = topk_ids[k];
    const bool eok   = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex     = eok ? expr : 0;
    const float coef = eok ? topk_w[k] * scale2[ex] : 0.0f;

    const float4 * __restrict__ ak = reinterpret_cast<const float4 *>(act + (size_t) k * K);
    float4 rx0[8];
    float4 rx1[4];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        rx0[c] = ak[g * 8 + c];
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        rx1[c] = ak[(16 + g) * 8 + c];
    }
    if (rloc == 0) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            s_act1[k][c][g] = ak[(16 + g) * 8 + 4 + c];
        }
    } else {
        const int sb = g >> 1;
        const int wi = g & 1;
        s_act2[k][0][g] = ak[256 + sb * 4 + wi];
        s_act2[k][1][g] = ak[256 + sb * 4 + 2 + wi];
    }
    __syncthreads();

    struct RowLd { v4i w0, w1; uint32_t w2; uint16_t sc0, sc1; uint8_t sc2; };
    RowLd cur, nxt;

    int r0 = blockIdx.x * RPB;
    {
        const int rl = r0 + rloc;
        nvfp4_down_load_row(qs, scales, ex, g, rl, cur.w0, cur.w1, cur.w2, cur.sc0, cur.sc1, cur.sc2);
    }

    int iter = 0;
    for (; r0 < N_EMBD; r0 += GRID_BLOCKS * RPB, ++iter) {
        {
            const int rn = r0 + GRID_BLOCKS * RPB + rloc;
            const int rl = (rn < N_EMBD) ? rn : rloc;
            nvfp4_down_load_row(qs, scales, ex, g, rl, nxt.w0, nxt.w1, nxt.w2, nxt.sc0, nxt.sc1, nxt.sc2);
        }

        float acc[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            acc[rr] = nvfp4_sub_dot((uint32_t) cur.w0.x, (uint32_t) cur.w0.y, rx0[0], rx0[1], rx0[2], rx0[3]) *
                      ue4m3_to_fp32(cur.sc0 & 0xFFu);
            acc[rr] = fmaf(nvfp4_sub_dot((uint32_t) cur.w0.z, (uint32_t) cur.w0.w, rx0[4], rx0[5], rx0[6], rx0[7]),
                           ue4m3_to_fp32(cur.sc0 >> 8), acc[rr]);
        }

        __builtin_amdgcn_sched_barrier(0);
        const float4 x1_4 = s_act1[k][0][g];
        const float4 x1_5 = s_act1[k][1][g];
        const float4 x1_6 = s_act1[k][2][g];
        const float4 x1_7 = s_act1[k][3][g];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            acc[rr] = fmaf(nvfp4_sub_dot((uint32_t) cur.w1.x, (uint32_t) cur.w1.y, rx1[0], rx1[1], rx1[2], rx1[3]),
                           ue4m3_to_fp32(cur.sc1 & 0xFFu), acc[rr]);
            acc[rr] = fmaf(nvfp4_sub_dot((uint32_t) cur.w1.z, (uint32_t) cur.w1.w, x1_4, x1_5, x1_6, x1_7),
                           ue4m3_to_fp32(cur.sc1 >> 8), acc[rr]);
        }

        __builtin_amdgcn_sched_barrier(0);
        const float4 x2_0 = s_act2[k][0][g];
        const float4 x2_1 = s_act2[k][1][g];
        const int buf = iter & 1;
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            uint32_t lo, hi;
            nvfp4_hbytes(cur.w2, lo, hi);
            float p3 = 0.0f;
            p3 = nvfp4_hb_dot_mix(lo, x2_0, p3);
            p3 = nvfp4_hb_dot_mix(hi, x2_1, p3);
            acc[rr] = fmaf(p3, ue4m3_to_fp32(cur.sc2), acc[rr]);
            const float v = warp_reduce_sum<16>(acc[rr]) * coef;
            if (g == 0) {
                s_red[buf][k][rloc + 2 * rr] = v;
            }
        }
        __syncthreads();
        if (warp == 0 && lane < RPB) {
            const float sum = ((s_red[buf][0][lane] + s_red[buf][1][lane]) + (s_red[buf][2][lane] + s_red[buf][3][lane])) +
                              ((s_red[buf][4][lane] + s_red[buf][5][lane]) + (s_red[buf][6][lane] + s_red[buf][7][lane]));
            const int ro = r0 + lane;
            out[ro] = (epi.conv_w != nullptr) ? sconv_post_res_one(epi, ro, sum) : sum;
        }
        cur = nxt;
    }
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------
SconvEpi make_epi(const float * conv_w, float * hist, const float * res, float * raw_out = nullptr) {
    SconvEpi e;
    e.conv_w  = conv_w;
    e.hist    = hist;
    e.res     = res;
    e.raw_out = raw_out;
    return e;
}

template <int RPB>
int partition_segments(QkvSeg * s, int nseg, int cap) {
    long long groups[3] = {0, 0, 0};
    long long total = 0;
    for (int i = 0; i < nseg; ++i) {
        groups[i] = ((long long) s[i].rows + RPB - 1) / RPB;
        total += groups[i];
    }
    int end = 0;
    for (int i = 0; i < 3; ++i) {
        if (i < nseg) {
            long long nb = groups[i];
            if (cap > 0 && total > cap) {
                const long long share = (long long) cap * groups[i] / total;
                nb = share < 1 ? 1 : (share < groups[i] ? share : groups[i]);
            }
            end += (int) nb;
            s[i].blk_end = end;
        } else {
            s[i].blk_end = 0x7FFFFFFF;
        }
    }
    return end;
}

QkvSeg make_seg(const SoaQ8Matrix & W, const float * x, float * y) {
    QkvSeg s;
    s.qs = W.qs;
    s.d = W.d;
    s.x = x;
    s.y = y;
    s.rows = W.rows;
    s.blk_end = 0;
    return s;
}

void extract_states(const float * const * src, int n, DecStatePtrs & dst) {
    dst = {};
    for (int i = 0; i < n && i < MAX_RES_BLOCKS; ++i) {
        dst.p[i] = src[i];
    }
}

int device_wgp_count_df() {
    static int cache[64] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) {
        return 48;
    }
    if (cache[dev] == 0) {
        int n = 0;
        if (hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || n <= 0) {
            n = 48;
        }
        cache[dev] = n;
    }
    return cache[dev];
}

// ---------------------------------------------------------------------------
// Co-residency gate for the grid-barrier kernels in this file.
//
// Every fused kernel here is ONE launch whose blocks synchronise through
// self-resetting device counters, so a grid that is not fully resident deadlocks
// (blocks in a later wave spin on blocks that never start). The grids were sized
// for gfx1100 (96 CU): 96 x 288-thread blocks at __launch_bounds__(288, 1) need
// 96 CUs, 192 blocks need 192, 384 need 384.
//
// On RDNA4 (gfx1200/gfx1201, 64 CU) the 96-block A grid (and the 96 x nrows verify
// grid) does not fit, so those launchers fall back to the multi-launch
// composition, which computes the same thing without a cross-block barrier.
// gfx11 keeps the fused launch unconditionally: its behaviour is unchanged.
// ---------------------------------------------------------------------------
int g_occ_mix_dual = -1;   // blocks/CU of mix_dual_ctrl_qkv3_norm_decode_kernel
int g_occ_mix_ffn_v2 = -1; // blocks/CU of mix_ffn_norm_router_topk_fused_v2_kernel
int g_occ_mix_ctrl_v2 = -1;// blocks/CU of mix_ctrl_ffn_norm_router_topk_fused_v2_kernel

inline bool df_coresident(int occ, int grid) {
    if (occ <= 0) {
        return false;
    }
    return (long long)occ * (long long)device_wgp_count_df() >= (long long)grid;
}

// 96 blocks x 288 threads needs 2 blocks/CU; the 32/33-block kernels fit anywhere.
inline bool df_mix_dual_ok() {
#if NIB_RDNA3
    return true;  // gfx11: unchanged (statically co-resident on >=96 CU)
#else
    return df_coresident(g_occ_mix_dual, 96);
#endif
}

} // namespace

// ============================================================================
// Public API
// ============================================================================
void launch_mix_dual_ctrl_qkv3_norm_decode(
        const float * query0, int n_q0, const float * const * states0, int n_states0, float * out0,
        const float * query1, int n_q1, const float * const * states1, int n_states1, float * out1,
        const __half * const w_down[3],
        const __half * const w_gain[3],
        const __half * const w_forget[3],
        const float * const w_norm[3],
        float * tmp_gate,
        float * const out_norm[3],
        int n_tokens, float eps, hipStream_t stream,
        float * scratch) {
    const float * cur[3] = {
        out1,
        out1 + (size_t)(n_q1 >= 3 ? 1 : 0) * N_EMBD,
        out1 + (size_t)(n_q1 >= 3 ? 2 : 0) * N_EMBD
    };
    const float * msg_base = out0; // LOOP_CTRL_Q == 0

    if (n_tokens != 1 || n_states0 > 5 || n_states1 > 5 || n_q0 > 4 || n_q1 > 4 ||
        scratch == nullptr || !df_mix_dual_ok()) {
        launch_inkling_mix_dual(query0, n_q0, states0, n_states0, out0,
                                query1, n_q1, states1, n_states1, out1, n_tokens, stream);
        launch_inkling_ctrl_qkv_norm(cur, msg_base, w_down, w_gain, w_forget, w_norm,
                                     tmp_gate, out_norm, n_tokens, eps, stream, scratch);
        return;
    }

    DecStatePtrs s0, s1;
    extract_states(states0, n_states0, s0);
    extract_states(states1, n_states1, s1);

    DecPtr3 cur3;
    cur3.p[0] = cur[0]; cur3.p[1] = cur[1]; cur3.p[2] = cur[2];
    DecHalfPtr3 wd3, wg3, wf3;
    DecNormPtr3 wn3;
    DecMutPtr3 on3;
    for (int i = 0; i < 3; ++i) {
        wd3.p[i] = w_down[i];
        wg3.p[i] = w_gain[i];
        wf3.p[i] = w_forget[i];
        wn3.p[i] = w_norm ? w_norm[i] : nullptr;
        on3.p[i] = out_norm[i];
    }

    mix_dual_ctrl_qkv3_norm_decode_kernel<false><<<96, 288, 0, stream>>>(
        query0, n_q0, s0, n_states0, out0,
        query1, n_q1, s1, n_states1, out1,
        cur3, msg_base, wd3, wg3, wf3, wn3, tmp_gate, on3, eps, scratch, N_EMBD, N_EMBD);
}

void launch_mix_dual_ctrl_qkv3_norm_decode_mrows(
        const float * query0, int n_q0, const float * const * states0, int n_states0, float * out0,
        const float * query1, int n_q1, const float * const * states1, int n_states1, float * out1,
        const __half * const w_down[3],
        const __half * const w_gain[3],
        const __half * const w_forget[3],
        const float * const w_norm[3],
        float * tmp_gate,
        float * const out_norm[3],
        float eps, hipStream_t stream, float * scratch,
        int nrows) {
    if (nrows <= 0) return;
    if (nrows == 1) {
        launch_mix_dual_ctrl_qkv3_norm_decode(query0, n_q0, states0, n_states0, out0,
                                              query1, n_q1, states1, n_states1, out1,
                                              w_down, w_gain, w_forget, w_norm,
                                              tmp_gate, out_norm, 1, eps, stream, scratch);
        return;
    }
    // gfx12: this is one 96 x nrows grid-barrier launch; unless the whole grid is
    // co-resident, run the rows as separate single-token launches (same body).
    if (!df_coresident(g_occ_mix_dual, 96 * nrows)) {
        const int ns0 = (n_states0 < MAX_RES_BLOCKS) ? n_states0 : MAX_RES_BLOCKS;
        const int ns1 = (n_states1 < MAX_RES_BLOCKS) ? n_states1 : MAX_RES_BLOCKS;
        for (int r = 0; r < nrows; ++r) {
            const float * st0[MAX_RES_BLOCKS];
            const float * st1[MAX_RES_BLOCKS];
            for (int i = 0; i < ns0; ++i) st0[i] = states0[i] + (size_t)r * N_EMBD;
            for (int i = 0; i < ns1; ++i) st1[i] = states1[i] + (size_t)r * N_EMBD;
            float * on3r[3] = {
                (out_norm[0] != nullptr) ? out_norm[0] + (size_t)r * N_EMBD : nullptr,
                (out_norm[1] != nullptr) ? out_norm[1] + (size_t)r * N_EMBD : nullptr,
                (out_norm[2] != nullptr) ? out_norm[2] + (size_t)r * N_EMBD : nullptr
            };
            launch_mix_dual_ctrl_qkv3_norm_decode(
                query0, n_q0, st0, ns0, out0 + (size_t)r * 4 * N_EMBD,
                query1, n_q1, st1, ns1, out1 + (size_t)r * 3 * N_EMBD,
                w_down, w_gain, w_forget, w_norm,
                tmp_gate + (size_t)r * 3 * LOOP_GATE_RANK, on3r, 1, eps, stream,
                scratch + (size_t)r * 512);
        }
        return;
    }
    const float * cur[3] = {
        out1,
        out1 + (size_t)(n_q1 >= 3 ? 1 : 0) * N_EMBD,
        out1 + (size_t)(n_q1 >= 3 ? 2 : 0) * N_EMBD
    };
    const float * msg_base = out0;
    DecStatePtrs s0, s1;
    extract_states(states0, n_states0, s0);
    extract_states(states1, n_states1, s1);
    DecPtr3 cur3;
    cur3.p[0] = cur[0]; cur3.p[1] = cur[1]; cur3.p[2] = cur[2];
    DecHalfPtr3 wd3, wg3, wf3;
    DecNormPtr3 wn3;
    DecMutPtr3 on3;
    for (int i = 0; i < 3; ++i) {
        wd3.p[i] = w_down[i];
        wg3.p[i] = w_gain[i];
        wf3.p[i] = w_forget[i];
        wn3.p[i] = w_norm ? w_norm[i] : nullptr;
        on3.p[i] = out_norm[i];
    }
    mix_dual_ctrl_qkv3_norm_decode_kernel<true><<<dim3(96, nrows), 288, 0, stream>>>(
        query0, n_q0, s0, n_states0, out0,
        query1, n_q1, s1, n_states1, out1,
        cur3, msg_base, wd3, wg3, wf3, wn3, tmp_gate, on3, eps, scratch,
        4 * N_EMBD, 3 * N_EMBD);
}

void launch_mix_ffn_norm_router_topk_decode_v2_mrows(
        const float * res_query,
        const float * const * states, int n_states,
        const float * ffn_norm_w,
        const __half * router_w,
        const float * exp_probs_b,
        float * x_norm_out,
        int32_t * topk_ids_out,
        float * topk_w_out,
        float * router_scratch,
        int nrows, hipStream_t stream) {
    if (nrows <= 0 || n_states <= 0) return;
    if (nrows == 1) {
        launch_mix_ffn_norm_router_topk_decode_v2(res_query, states, n_states, ffn_norm_w,
                                                  router_w, exp_probs_b, x_norm_out,
                                                  topk_ids_out, topk_w_out, router_scratch, stream);
        return;
    }
    DecStatePtrs s;
    extract_states(states, n_states, s);
    // gfx12: 33 x nrows blocks in one grid-barrier launch - fall back to one row per
    // launch unless the whole grid is co-resident.
    if (!df_coresident(g_occ_mix_ffn_v2, 33 * nrows)) {
        for (int r = 0; r < nrows; ++r) {
            const float * st[MAX_RES_BLOCKS];
            for (int i = 0; i < n_states; ++i) st[i] = states[i] + (size_t)r * N_EMBD;
            launch_mix_ffn_norm_router_topk_decode_v2(
                res_query, st, n_states, ffn_norm_w, router_w, exp_probs_b,
                x_norm_out + (size_t)r * N_EMBD,
                topk_ids_out + (size_t)r * TOP_K, topk_w_out + (size_t)r * TOP_K,
                router_scratch + (size_t)r * 512, stream);
        }
        return;
    }
    mix_ffn_norm_router_topk_fused_v2_kernel<true><<<dim3(33, nrows), 192, 0, stream>>>(
        res_query, s, n_states, ffn_norm_w, router_w, exp_probs_b,
        x_norm_out, topk_ids_out, topk_w_out, router_scratch);
}

void launch_mix_ctrl_ffn_norm_router_topk_decode_v2_mrows(
        const float * res_query,
        const float * const * states, int n_states,
        const float * msg,
        const __half * w_down,
        const __half * w_gain,
        const __half * w_forget,
        const float * ffn_norm_w,
        const __half * router_w,
        const float * exp_probs_b,
        float * out_mixed,
        float * out_gate,
        float * x_norm_out,
        int32_t * topk_ids_out,
        float * topk_w_out,
        float * router_scratch,
        int nrows, hipStream_t stream) {
    if (nrows <= 0 || n_states <= 0) return;
    if (nrows == 1) {
        launch_mix_ctrl_ffn_norm_router_topk_decode_v2(res_query, states, n_states, msg,
                                                      w_down, w_gain, w_forget, ffn_norm_w,
                                                      router_w, exp_probs_b, out_mixed, out_gate,
                                                      x_norm_out, topk_ids_out, topk_w_out,
                                                      router_scratch, stream);
        return;
    }
    DecStatePtrs s;
    extract_states(states, n_states, s);
    if (n_states > 5 || router_scratch == nullptr ||
        !df_coresident(g_occ_mix_ctrl_v2, 32 * nrows)) {
        for (int r = 0; r < nrows; ++r) {
            const float * st[MAX_RES_BLOCKS];
            for (int i = 0; i < n_states; ++i) st[i] = states[i] + (size_t)r * N_EMBD;
            launch_mix_ctrl_ffn_norm_router_topk_decode_v2(
                res_query, st, n_states, msg + (size_t)r * 4 * N_EMBD,
                w_down, w_gain, w_forget, ffn_norm_w, router_w, exp_probs_b,
                out_mixed + (size_t)r * N_EMBD, out_gate + (size_t)r * LOOP_GATE_RANK,
                x_norm_out + (size_t)r * N_EMBD,
                topk_ids_out + (size_t)r * TOP_K, topk_w_out + (size_t)r * TOP_K,
                router_scratch, stream);
        }
        return;
    }
    mix_ctrl_ffn_norm_router_topk_fused_v2_kernel<5, true><<<dim3(32, nrows), 288, 0, stream>>>(
        res_query, s, n_states, msg, w_down, w_gain, w_forget, ffn_norm_w,
        router_w, exp_probs_b, out_mixed, out_gate, x_norm_out,
        topk_ids_out, topk_w_out, router_scratch, 4 * N_EMBD);
}

void launch_mix_ffn_norm_router_topk_decode_v2(
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
    if (n_states > 5 || router_scratch == nullptr) {
        launch_mix_ffn_norm_router_topk_decode(res_query, states, n_states, ffn_norm_w,
                                               router_w, exp_probs_b, x_norm_out,
                                               topk_ids_out, topk_w_out, router_scratch, stream);
        return;
    }
    DecStatePtrs s;
    extract_states(states, n_states, s);
    mix_ffn_norm_router_topk_fused_v2_kernel<false><<<33, 192, 0, stream>>>(
        res_query, s, n_states, ffn_norm_w, router_w, exp_probs_b,
        x_norm_out, topk_ids_out, topk_w_out, router_scratch);
}

void launch_mix_ctrl_ffn_norm_router_topk_decode_v2(
        const float * res_query,
        const float * const * states, int n_states,
        const float * msg,
        const __half * w_down,
        const __half * w_gain,
        const __half * w_forget,
        const float * ffn_norm_w,
        const __half * router_w,
        const float * exp_probs_b,
        float * out_mixed,
        float * out_gate,
        float * x_norm_out,
        int32_t * topk_ids_out,
        float * topk_w_out,
        float * router_scratch,
        hipStream_t stream) {
    if (n_states <= 0) {
        return;
    }
    if (n_states > 5 || router_scratch == nullptr) {
        launch_mix_ctrl_down_decode(res_query, states, n_states, msg, w_down,
                                    out_mixed, out_gate, stream);
        launch_ctrl_apply_ffn_norm_router_topk_decode(out_mixed, msg, out_gate,
                                                      w_gain, w_forget, ffn_norm_w,
                                                      router_w, exp_probs_b,
                                                      x_norm_out, topk_ids_out, topk_w_out,
                                                      router_scratch, stream);
        return;
    }
    DecStatePtrs s;
    extract_states(states, n_states, s);
    mix_ctrl_ffn_norm_router_topk_fused_v2_kernel<5, false><<<32, 288, 0, stream>>>(
        res_query, s, n_states, msg, w_down, w_gain, w_forget,
        ffn_norm_w, router_w, exp_probs_b, out_mixed, out_gate,
        x_norm_out, topk_ids_out, topk_w_out, router_scratch, N_EMBD);
}

void launch_soa_q8_gemv_v2(const SoaQ8Matrix & W, const float * x, float * y, hipStream_t stream) {
    if (W.rows <= 0) {
        return;
    }
    // Only the medium QKV-sized projections are re-tiled here; large matrices
    // (lm_head, 200058 rows) keep the engine's dedicated 8-warp config.
    if (W.K != N_EMBD || W.rows > 12288) {
        launch_soa_q8_gemv(W, x, y, stream);
        return;
    }
    QkvSeg s[3];
    s[0] = make_seg(W, x, y);
    s[1] = s[0];
    s[2] = s[0];
    constexpr int NWARPS = 4, LPR = 8, NB = N_EMBD / 32, PD = 3;
    constexpr int RPB = NWARPS * (32 / LPR);
    const int grid = partition_segments<RPB>(s, 1, 0);
    q8_gemv_qkv_v2_kernel<NWARPS, LPR, NB, PD><<<grid, NWARPS * 32, 0, stream>>>(s[0], s[1], s[2]);
}

void launch_soa_q8_gemv_qkv3_v2(
        const SoaQ8Matrix & W0, const float * x0, float * y0,
        const SoaQ8Matrix & W1, const float * x1, float * y1,
        const SoaQ8Matrix & W2, const float * x2, float * y2,
        hipStream_t stream) {
    if (W0.K != N_EMBD || (W1.rows > 0 && W1.K != N_EMBD) || (W2.rows > 0 && W2.K != N_EMBD) ||
        W0.rows > 12288 || W1.rows > 12288 || W2.rows > 12288) {
        launch_soa_q8_gemv_qkv3(W0, x0, y0, W1, x1, y1, W2, x2, y2, stream);
        return;
    }
    QkvSeg s[3];
    int nseg = 0;
    if (W0.rows > 0) s[nseg++] = make_seg(W0, x0, y0);
    if (W1.rows > 0) s[nseg++] = make_seg(W1, x1, y1);
    if (W2.rows > 0) s[nseg++] = make_seg(W2, x2, y2);
    if (nseg == 0) {
        return;
    }
    constexpr int NWARPS = 4, LPR = 8, NB = N_EMBD / 32, PD = 3;
    constexpr int RPB = NWARPS * (32 / LPR);
    const int grid = partition_segments<RPB>(s, nseg, 0);
    q8_gemv_qkv_v2_kernel<NWARPS, LPR, NB, PD><<<grid, NWARPS * 32, 0, stream>>>(s[0], s[1], s[2]);
}

namespace {
void qkv_unfused_fallback_df(
        const SoaQ8Matrix & W0, const float * x0, float * y0,
        const SoaQ8Matrix & W1, const float * x1, float * y1,
        const SoaQ8Matrix & W2, const float * x2, float * y2,
        const float * q_in, const float * q_clip_w,
        const float * r_in, const float * r_clip_w, const float * rel_bias_w,
        const float * k_raw, const float * k_conv_w, const float * k_dyn,
        const float * v_raw, const float * v_conv_w, const float * v_dyn,
        float * sconv_hist_kv, KVCacheSlot kv_slot,
        float * q_out, float * rel_table_out, bool apply_tau,
        const StepMeta * d_meta, hipStream_t stream) {
    if (W1.rows > 0 && W2.rows > 0) {
        launch_soa_q8_gemv_qkv3(W0, x0, y0, W1, x1, y1, W2, x2, y2, stream);
    } else {
        if (W0.rows > 0) launch_soa_q8_gemv(W0, x0, y0, stream);
        if (W1.rows > 0) launch_soa_q8_gemv(W1, x1, y1, stream);
        if (W2.rows > 0) launch_soa_q8_gemv(W2, x2, y2, stream);
    }
    launch_qkvr_prep_fused_decode(q_in, 0, q_clip_w, r_in, 0, r_clip_w, rel_bias_w,
                                  k_raw, 0, k_conv_w, k_dyn, 0, v_raw, 0, v_conv_w, v_dyn, 0,
                                  sconv_hist_kv, kv_slot, q_out, rel_table_out,
                                  apply_tau, d_meta, stream);
}
} // namespace

void launch_qkv_qkvr_prep_fused_decode_g(
        const SoaQ8Matrix & W0, const float * x0, float * y0,
        const SoaQ8Matrix & W1, const float * x1, float * y1,
        const SoaQ8Matrix & W2, const float * x2, float * y2,
        const float * q_in, const float * q_clip_w,
        const float * r_in, const float * r_clip_w, const float * rel_bias_w,
        const float * k_raw, const float * k_conv_w, const float * k_dyn,
        const float * v_raw, const float * v_conv_w, const float * v_dyn,
        float * sconv_hist_kv,
        KVCacheSlot kv_slot,
        float * q_out,
        float * rel_table_out,
        bool apply_tau,
        const StepMeta * d_meta,
        float * scratch,
        hipStream_t stream,
        int grid) {
    const int wgp = device_wgp_count_df();
    // Auto-downgrade grid to stay strictly co-resident on smaller gfx1100/RDNA3 GPUs
    // (128-thread kernel holds up to 7 blocks/WGP: 384 needs 48 WGPs, 288 needs 42, 192 needs 28).
    if (grid == 384 && wgp < 48) grid = 288;
    if (grid == 288 && wgp < 42) grid = 192;
    if (grid == 192 && wgp < 28) grid = 0;

    if (W0.K != N_EMBD || (W1.rows > 0 && W1.K != N_EMBD) || (W2.rows > 0 && W2.K != N_EMBD) ||
        W0.rows > 12288 || W1.rows > 12288 || W2.rows > 12288 || scratch == nullptr ||
        (grid != 192 && grid != 288 && grid != 384)) {
        qkv_unfused_fallback_df(
            W0, x0, y0, W1, x1, y1, W2, x2, y2,
            q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
            k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
            sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta, stream);
        return;
    }
    QkvSeg s[3];
    int nseg = 0;
    if (W0.rows > 0) s[nseg++] = make_seg(W0, x0, y0);
    if (W1.rows > 0) s[nseg++] = make_seg(W1, x1, y1);
    if (W2.rows > 0) s[nseg++] = make_seg(W2, x2, y2);
    if (nseg == 0) {
        return;
    }
    constexpr int NWARPS = 4, LPR = 8, NB = N_EMBD / 32, PD = 3;
    constexpr int RPB = NWARPS * (32 / LPR);
    if (grid == 384) {
        (void) partition_segments<RPB>(s, nseg, 384);
        qkv_qkvr_prep_fused_decode_kernel<NWARPS, LPR, NB, PD, 384><<<384, NWARPS * 32, 0, stream>>>(
            s[0], s[1], s[2],
            q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
            k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
            sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta, scratch);
    } else if (grid == 288) {
        (void) partition_segments<RPB>(s, nseg, 288);
        qkv_qkvr_prep_fused_decode_kernel<NWARPS, LPR, NB, PD, 288><<<288, NWARPS * 32, 0, stream>>>(
            s[0], s[1], s[2],
            q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
            k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
            sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta, scratch);
    } else {
        (void) partition_segments<RPB>(s, nseg, 192);
        qkv_qkvr_prep_fused_decode_kernel<NWARPS, LPR, NB, PD, 192><<<192, NWARPS * 32, 0, stream>>>(
            s[0], s[1], s[2],
            q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
            k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
            sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta, scratch);
    }
}

void launch_qkv_qkvr_prep_fused_decode(
        const SoaQ8Matrix & W0, const float * x0, float * y0,
        const SoaQ8Matrix & W1, const float * x1, float * y1,
        const SoaQ8Matrix & W2, const float * x2, float * y2,
        const float * q_in, const float * q_clip_w,
        const float * r_in, const float * r_clip_w, const float * rel_bias_w,
        const float * k_raw, const float * k_conv_w, const float * k_dyn,
        const float * v_raw, const float * v_conv_w, const float * v_dyn,
        float * sconv_hist_kv,
        KVCacheSlot kv_slot,
        float * q_out,
        float * rel_table_out,
        bool apply_tau,
        const StepMeta * d_meta,
        float * scratch,
        hipStream_t stream) {
    launch_qkv_qkvr_prep_fused_decode_g(
        W0, x0, y0, W1, x1, y1, W2, x2, y2,
        q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
        k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
        sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta, scratch, stream, 288);
}

void launch_soa_q8_gemv_wo_v2(
        const SoaQ8Matrix & W, const float * x,
        const float * conv_w, float * sconv_hist, const float * res, float * out,
        hipStream_t stream, float * raw_out) {
    if (W.rows <= 0) {
        return;
    }
    if (W.K != N_Q || conv_w == nullptr || sconv_hist == nullptr) {
        launch_soa_q8_gemv_sconv_post_res(W, x, conv_w, sconv_hist, res, out, stream);
        return;
    }
    QkvSeg s0 = make_seg(W, x, out);
    constexpr int NWARPS = 6, LPR = 32, NB = N_Q / 32, PD = 3;
    constexpr int RPB = NWARPS * (32 / LPR);
    const int grid = partition_segments<RPB>(&s0, 1, 0);
    q8_gemv_wo_v2_kernel<NWARPS, LPR, NB, PD><<<grid, NWARPS * 32, 0, stream>>>(
        s0, make_epi(conv_w, sconv_hist, res, raw_out));
}

void launch_moe_nvfp4_gate_up_v2(
        const SoaNVFP4Experts & gate_up,
        const float * x_norm, const int32_t * topk_ids, float * act,
        hipStream_t stream) {
    if (gate_up.rows != N_GATE_UP_EXP || gate_up.K != N_EMBD) {
        std::fprintf(stderr, "[kernels_decode_fused] gate_up shape mismatch\n");
        std::abort();
    }
    constexpr int GU_WARPS = 4;
    constexpr int GU_PAIRS = GU_WARPS * (32 / NVFP4_GU_LPR);
    static_assert(N_FF_EXP % GU_PAIRS == 0, "pair tiling");
    moe_nvfp4_gate_up_v2_kernel<GU_WARPS, 1><<<dim3(N_FF_EXP / GU_PAIRS, TOP_K), GU_WARPS * 32, 0, stream>>>(
        gate_up.qs, gate_up.scales, gate_up.scale2, x_norm, topk_ids, act);
}

void launch_moe_nvfp4_down_v2(
        const SoaNVFP4Experts & down,
        const float * act, const int32_t * topk_ids, const float * topk_w,
        float * out, hipStream_t stream,
        const float * post_conv_w, float * post_sconv_h, const float * post_res_add) {
    if (down.rows != N_EMBD || down.K != N_FF_EXP) {
        std::fprintf(stderr, "[kernels_decode_fused] down shape mismatch\n");
        std::abort();
    }
    moe_nvfp4_down_v2_kernel<1, 192><<<192, 256, 0, stream>>>(
        down.qs, down.scales, down.scale2, act, topk_ids, topk_w, out,
        make_epi(post_conv_w, post_sconv_h, post_res_add));
}

void launch_moe_nvfp4_decode_v2(
        const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
        const float * x_norm, const int32_t * topk_ids, const float * topk_w,
        float * tmp_act, float * out, hipStream_t stream,
        const float * post_conv_w, float * post_sconv_h, const float * post_res_add) {
    launch_moe_nvfp4_gate_up_v2(gate_up, x_norm, topk_ids, tmp_act, stream);
    launch_moe_nvfp4_down_v2(down, tmp_act, topk_ids, topk_w, out, stream,
                             post_conv_w, post_sconv_h, post_res_add);
}

// Cache the blocks/CU of the grid-barrier kernels of this file. Host-side query
// (no stream op), so it is capture-safe; called once from engine init.
void decode_fused_prime_occupancy() {
    int n = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &n, (const void *)mix_dual_ctrl_qkv3_norm_decode_kernel<false>, 288, 0) == hipSuccess) {
        g_occ_mix_dual = n;
    }
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &n, (const void *)mix_ffn_norm_router_topk_fused_v2_kernel<false>, 192, 0) == hipSuccess) {
        g_occ_mix_ffn_v2 = n;
    }
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &n, (const void *)mix_ctrl_ffn_norm_router_topk_fused_v2_kernel<5, false>, 288, 0) == hipSuccess) {
        g_occ_mix_ctrl_v2 = n;
    }
    if (std::getenv("NIB_DUMP_OCC") != nullptr) {
        std::fprintf(stderr, "[decode_fused] cu=%d occ/blk: mix_dual=%d mix_ffn_v2=%d mix_ctrl_v2=%d (grid 96 needs occ>=2 on 64 cu)\n",
                     device_wgp_count_df(), g_occ_mix_dual, g_occ_mix_ffn_v2, g_occ_mix_ctrl_v2);
    }
}

} // namespace inkling


