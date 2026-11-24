// ============================================================================
// src/kernels_fa_decode_v4.hip — Step 7K
//
//  * fa_swa_v4 / fa_global_v4: the Step 7G v3 arithmetic at 4-6 waves/SIMD
//    (optional LDS-staged Q) and with optional L2/L3-backed KV loads.
//  * select_v4: the v3 q+/q- page scorer for BOTH metadata formats
//    (SPARSE_META_Q8 int8 min/max and SPARSE_META_Q4 nibble-packed), with an
//    optional LDS-staged q+ / q- (frees 96 VGPRs of the 104 that capped v3 at
//    4 waves/SIMD).
//
// Every helper below that is shared with v3 is a *bit-exact copy*: the code
// base is one file per kernel family and these files do not include each other,
// so this translation unit is self-contained.  The arithmetic order of every
// dot product, shuffle tree, softmax, RelBias, sink, qgate and FWHT is
// unchanged, which is what makes the v4 A/B meaningful: only scheduling and
// cache policy differ.
// ============================================================================

#include "kernels_fa_decode_v4.h"

#include <cstdint>

namespace inkling {

namespace {

constexpr float V4_LOG2E = 1.44269504088896340736f;

typedef uint32_t v4u4 __attribute__((ext_vector_type(4)));

static_assert(sizeof(__half2) == 4, "half2 must be 4 bytes");

// ============================================================================
// gfx1100 primitives (identical to v3)
// ============================================================================
static __device__ __forceinline__ float v4_exp2f(float x) {
    return __builtin_amdgcn_exp2f(x);
}

static __device__ __forceinline__ __half2 v4_h2_of(uint32_t u) {
    return __builtin_bit_cast(__half2, u);
}

static __device__ __forceinline__ float v4_fdot2(__half2 a, __half2 b, float c) {
    return __builtin_amdgcn_fdot2(a, b, c, false);
}

static __device__ __forceinline__ __half2 v4_perm_lo(uint32_t x) {
    return v4_h2_of(__builtin_amdgcn_perm(0x64646464u, x, 0x04010400u));
}
static __device__ __forceinline__ __half2 v4_perm_hi(uint32_t x) {
    return v4_h2_of(__builtin_amdgcn_perm(0x64646464u, x, 0x04030402u));
}

static __device__ __forceinline__ void v4_u4x4_biased(uint32_t nibs, __half2 & h01, __half2 & h23) {
    h01 = v4_perm_lo(nibs);
    h23 = v4_perm_hi(nibs);
}

static __device__ __forceinline__ void v4_i8x4_biased(uint32_t w, __half2 & h01, __half2 & h23) {
    const uint32_t u = w ^ 0x80808080u;
    h01 = v4_perm_lo(u);
    h23 = v4_perm_hi(u);
}

static __device__ __forceinline__ void v4_u4x4_to_h2x2(uint32_t nibs, __half2 & h01, __half2 & h23) {
    h01 = __hsub2(v4_perm_lo(nibs), v4_h2_of(0x64006400u));
    h23 = __hsub2(v4_perm_hi(nibs), v4_h2_of(0x64006400u));
}
// q4_0's V value is d*(nibble - 8), so the fp16 bias is 1032, not 1024
// (identical to v1/v2's SIGNED_Q4_0 path; the 1024-only form silently dropped
// the -8 and broke q4_0 PPL by 5 orders of magnitude).
static __device__ __forceinline__ void v4_u4x4_to_h2x2_q40(uint32_t nibs, __half2 & h01, __half2 & h23) {
    h01 = __hsub2(v4_perm_lo(nibs), v4_h2_of(0x64086408u));
    h23 = __hsub2(v4_perm_hi(nibs), v4_h2_of(0x64086408u));
}

static __device__ __forceinline__ void v4_i8x4_to_h2x2(uint32_t w, __half2 & h01, __half2 & h23) {
    const uint32_t u = w ^ 0x80808080u;
    h01 = __hsub2(v4_perm_lo(u), v4_h2_of(0x64806480u));
    h23 = __hsub2(v4_perm_hi(u), v4_h2_of(0x64806480u));
}

static __device__ __forceinline__ void v4_fwht128_4(
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

static __device__ __forceinline__ void v4_gl_fence_release() {
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#endif
}

static __device__ __forceinline__ void v4_gl_invalidate() {
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("buffer_gl1_inv\n\tbuffer_gl0_inv" ::: "memory");
#endif
}

// ============================================================================
// FA v4 — Q either in registers (v3) or staged in LDS (768 B)
// ============================================================================
constexpr int V4_QPAIRS = 16;   // 32 dims / 2

template <int GQA>
struct V4Q {
    __half2 q[GQA][V4_QPAIRS];  // fp16 Q * Q_SCALE*LOG2E, dims [blk*32, +32)
    float   sum[GQA];           // sum of the fp16 dims (nibble-bias fold)
};

// Stage the 128-dim fp16 Q of all GQA heads into LDS as [GQA][64] __half2.
// Only the 4 k_slot==0 lanes of a warp write (lanes 0..3 = blk 0..3).
template <int GQA>
static __device__ __forceinline__ void v4_stage_q_lds(
        __half2 (*s_q)[HEAD_DIM / 2], const float * __restrict__ q,
        int h_base, int blk, int k_slot) {
    if (k_slot != 0) {
        return;
    }
    constexpr float QS = KQ_SCALE * V4_LOG2E;
    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        const float4 * qp = (const float4 *)(q + (size_t)(h_base + gq) * HEAD_DIM + (size_t)blk * 32);
        #pragma unroll
        for (int i4 = 0; i4 < 8; ++i4) {
            const float4 f = qp[i4];
            s_q[gq][blk * 16 + 2 * i4 + 0] = __floats2half2_rn(f.x * QS, f.y * QS);
            s_q[gq][blk * 16 + 2 * i4 + 1] = __floats2half2_rn(f.z * QS, f.w * QS);
        }
    }
}

template <int GQA>
static __device__ __forceinline__ void v4_stage_q(
        V4Q<GQA> & qv, const float * __restrict__ q, int h_base, int blk) {
    constexpr float QS = KQ_SCALE * V4_LOG2E;
    const __half2 ones = v4_h2_of(0x3C003C00u);
    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        const float4 * qp = (const float4 *)(q + (size_t)(h_base + gq) * HEAD_DIM + (size_t)blk * 32);
        #pragma unroll
        for (int i4 = 0; i4 < 8; ++i4) {
            const float4 f = qp[i4];
            qv.q[gq][2 * i4 + 0] = __floats2half2_rn(f.x * QS, f.y * QS);
            qv.q[gq][2 * i4 + 1] = __floats2half2_rn(f.z * QS, f.w * QS);
        }
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < V4_QPAIRS; ++j) {
            acc = v4_fdot2(qv.q[gq][j], ones, acc);
        }
        qv.sum[gq] = acc;
    }
}

// sum_d q_d of this lane's 32 dims, read back from LDS in the same order v3
// accumulated the register Q, so the nibble-bias fold stays bit-exact.
template <int GQA>
static __device__ __forceinline__ void v4_stage_q_lds_sum(
        float sum[GQA], const __half2 (*s_q)[HEAD_DIM / 2], int blk) {
    const __half2 ones = v4_h2_of(0x3C003C00u);
    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < V4_QPAIRS; ++j) {
            acc = v4_fdot2(s_q[gq][blk * 16 + j], ones, acc);
        }
        sum[gq] = acc;
    }
}

template <int KV_FMT> struct V4Nib {};
template <> struct V4Nib<KV_Q4_1> { static constexpr float BIAS = 1024.0f; };
template <> struct V4Nib<KV_Q4_0> { static constexpr float BIAS = 1032.0f; };
template <> struct V4Nib<KV_Q8_0> { static constexpr float BIAS = 1152.0f; };

template <int KV_FMT> struct V4KvGrp {};
template <> struct V4KvGrp<KV_Q8_0> {
    v4u4     k_lo, k_hi;
    float    k_d;
    uint32_t vq[8];
    float    v_d;
};
template <> struct V4KvGrp<KV_Q4_1> {
    v4u4     k_qs;
    float    k_d, k_m;
    uint32_t vq[4];
    uint32_t v_dm;
};
template <> struct V4KvGrp<KV_Q4_0> {
    v4u4     k_qs;
    float    k_d;
    uint32_t vq[4];
    float    v_d;
};

// CACHED == 1: normal L1/L2-backed vector loads.  CACHED == 0: `slc dlc`
// streaming (the v3 behaviour).
template <int CACHED>
static __device__ __forceinline__ v4u4 v4_ld128(const v4u4 * p) {
    if constexpr (CACHED) {
        return *(const v4u4 *)p;
    } else {
        return __builtin_nontemporal_load(p);
    }
}

template <int CACHED>
static __device__ __forceinline__ uint4 v4_ld128_u4(const void * p) {
    if constexpr (CACHED) {
        return *(const uint4 *)p;
    } else {
        const v4u4 t = __builtin_nontemporal_load((const v4u4 *)p);
        return make_uint4(t.x, t.y, t.z, t.w);
    }
}

template <int CACHED>
static __device__ __forceinline__ uint32_t v4_ld32(const uint32_t * p) {
    if constexpr (CACHED) {
        return *(const uint32_t *)p;
    } else {
        return __builtin_nontemporal_load(p);
    }
}

template <int KV_FMT, int CACHED>
static __device__ __forceinline__ void v4_load_kv_group(
        V4KvGrp<KV_FMT> & g, const KVCacheSlot & kv,
        size_t head_base, int cap, int ring0, int lane) {
    const int blk = lane & 3;
    const int kslot = lane >> 2;
    int jk = ring0 + kslot;
    jk -= (jk >= cap) ? cap : 0;
    const size_t ks = head_base + (size_t)jk;

    if constexpr (KV_FMT == KV_Q8_0) {
        const int8_t * kp = kv.k_qs + ks * HEAD_DIM + (size_t)blk * 32;
        g.k_lo = v4_ld128<CACHED>((const v4u4 *)(kp));
        g.k_hi = v4_ld128<CACHED>((const v4u4 *)(kp + 16));
        g.k_d  = __half2float(__ldg(&kv.k_d[ks * 4 + (size_t)blk]));
        g.v_d  = __half2float(__ldg(&kv.v_d[ks * 4 + (size_t)blk]));
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            int jt = ring0 + kk;
            jt -= (jt >= cap) ? cap : 0;
            g.vq[kk] = v4_ld32<CACHED>((const uint32_t *)(kv.v_qs + (head_base + (size_t)jt) * HEAD_DIM) + lane);
        }
    } else {
        const int8_t * kp = kv.k_qs + ks * (HEAD_DIM / 2) + (size_t)blk * 16;
        g.k_qs = v4_ld128<CACHED>((const v4u4 *)kp);
        if constexpr (KV_FMT == KV_Q4_1) {
            const __half2 dm = __ldg(&((const __half2 *)kv.k_d)[ks * 4 + (size_t)blk]);
            g.k_d  = __low2float(dm);
            g.k_m  = __high2float(dm);
            g.v_dm = __builtin_bit_cast(uint32_t, __ldg(&((const __half2 *)kv.v_d)[ks * 4 + (size_t)blk]));
        } else {
            g.k_d = __half2float(__ldg(&kv.k_d[ks * 4 + (size_t)blk]));
            g.v_d = __half2float(__ldg(&kv.v_d[ks * 4 + (size_t)blk]));
        }
        const int vhalf = lane >> 1;
        const int vodd  = lane & 1;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            int jt = ring0 + 2 * p + vodd;
            jt -= (jt >= cap) ? cap : 0;
            g.vq[p] = v4_ld32<CACHED>(((const uint32_t *)(kv.v_qs + (head_base + (size_t)jt) * (HEAD_DIM / 2))) + vhalf);
        }
    }
}

template <int KV_FMT, int CACHED>
static __device__ __forceinline__ void v4_load_kv_group_abs(
        V4KvGrp<KV_FMT> & g, const KVCacheSlot & kv,
        size_t head_base, int base, int n_kv, int lane) {
    const int blk = lane & 3;
    const int jmax = (n_kv > 0) ? (n_kv - 1) : 0;
    const int jk = ((base + (lane >> 2)) < jmax) ? (base + (lane >> 2)) : jmax;
    const size_t ks = head_base + (size_t)jk;

    if constexpr (KV_FMT == KV_Q8_0) {
        const int8_t * kp = kv.k_qs + ks * HEAD_DIM + (size_t)blk * 32;
        g.k_lo = v4_ld128<CACHED>((const v4u4 *)(kp));
        g.k_hi = v4_ld128<CACHED>((const v4u4 *)(kp + 16));
        g.k_d  = __half2float(__ldg(&kv.k_d[ks * 4 + (size_t)blk]));
        g.v_d  = __half2float(__ldg(&kv.v_d[ks * 4 + (size_t)blk]));
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int jt = ((base + kk) < jmax) ? (base + kk) : jmax;
            g.vq[kk] = v4_ld32<CACHED>(((const uint32_t *)(kv.v_qs + (head_base + (size_t)jt) * HEAD_DIM)) + lane);
        }
    } else {
        const int8_t * kp = kv.k_qs + ks * (HEAD_DIM / 2) + (size_t)blk * 16;
        g.k_qs = v4_ld128<CACHED>((const v4u4 *)kp);
        if constexpr (KV_FMT == KV_Q4_1) {
            const __half2 dm = __ldg(&((const __half2 *)kv.k_d)[ks * 4 + (size_t)blk]);
            g.k_d  = __low2float(dm);
            g.k_m  = __high2float(dm);
            g.v_dm = __builtin_bit_cast(uint32_t, __ldg(&((const __half2 *)kv.v_d)[ks * 4 + (size_t)blk]));
        } else {
            g.k_d = __half2float(__ldg(&kv.k_d[ks * 4 + (size_t)blk]));
            g.v_d = __half2float(__ldg(&kv.v_d[ks * 4 + (size_t)blk]));
        }
        const int vhalf = lane >> 1;
        const int vodd  = lane & 1;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int t  = base + 2 * p + vodd;
            const int jt = (t < jmax) ? t : jmax;
            g.vq[p] = v4_ld32<CACHED>(((const uint32_t *)(kv.v_qs + (head_base + (size_t)jt) * (HEAD_DIM / 2))) + vhalf);
        }
    }
}

// ============================================================================
// One 8-key wave tile (arithmetic identical to v3)
// ============================================================================
template <int GQA, int KV_FMT, bool FULL, bool SWA, bool QLDS>
static __device__ __forceinline__ void v4_fa_group(
        const V4Q<GQA> & qv,
        const __half2 (*__restrict__ s_q)[HEAD_DIM / 2],
        const V4KvGrp<KV_FMT> & g,
        const float * __restrict__ rel_base,
        int   dist0,
        int   blk, int lane, int v_blk, int k_slot,
        bool  kv_ok,
        float o_reg[GQA][4], float m_reg[GQA], float l_reg[GQA], float ob_reg[GQA]) {
    constexpr float KBIAS = V4Nib<KV_FMT>::BIAS;

    // ---- 1. K . Q^T ---------------------------------------------------------
    float sc_k[GQA];
    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        sc_k[gq] = 0.0f;
    }

    if constexpr (KV_FMT == KV_Q8_0) {
        const uint32_t ku[8] = { g.k_lo.x, g.k_lo.y, g.k_lo.z, g.k_lo.w,
                                 g.k_hi.x, g.k_hi.y, g.k_hi.z, g.k_hi.w };
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            __half2 k0, k1, k2, k3;
            v4_i8x4_biased(ku[2 * i + 0], k0, k1);
            v4_i8x4_biased(ku[2 * i + 1], k2, k3);
            #pragma unroll
            for (int gq = 0; gq < GQA; ++gq) {
                __half2 q0, q1, q2, q3;
                if constexpr (QLDS) {
                    const v4u4 t = *(const v4u4 *)(&s_q[gq][blk * 16 + 4 * i]);
                    q0 = v4_h2_of(t.x); q1 = v4_h2_of(t.y);
                    q2 = v4_h2_of(t.z); q3 = v4_h2_of(t.w);
                } else {
                    q0 = qv.q[gq][4 * i + 0]; q1 = qv.q[gq][4 * i + 1];
                    q2 = qv.q[gq][4 * i + 2]; q3 = qv.q[gq][4 * i + 3];
                }
                sc_k[gq] = v4_fdot2(q0, k0, sc_k[gq]);
                sc_k[gq] = v4_fdot2(q1, k1, sc_k[gq]);
                sc_k[gq] = v4_fdot2(q2, k2, sc_k[gq]);
                sc_k[gq] = v4_fdot2(q3, k3, sc_k[gq]);
            }
        }
    } else {
        const uint32_t ku[4] = { g.k_qs.x, g.k_qs.y, g.k_qs.z, g.k_qs.w };
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t w  = ku[i];
            const uint32_t lo = w & 0x0F0F0F0Fu;
            const uint32_t hi = (w >> 4) & 0x0F0F0F0Fu;
            __half2 l0, l1, h0, h1;
            v4_u4x4_biased(lo, l0, l1);
            v4_u4x4_biased(hi, h0, h1);
            #pragma unroll
            for (int gq = 0; gq < GQA; ++gq) {
                __half2 q0, q1, q2, q3;
                if constexpr (QLDS) {
                    const v4u4 t = *(const v4u4 *)(&s_q[gq][blk * 16 + 4 * i]);
                    q0 = v4_h2_of(t.x); q1 = v4_h2_of(t.y);
                    q2 = v4_h2_of(t.z); q3 = v4_h2_of(t.w);
                } else {
                    q0 = qv.q[gq][4 * i + 0]; q1 = qv.q[gq][4 * i + 1];
                    q2 = qv.q[gq][4 * i + 2]; q3 = qv.q[gq][4 * i + 3];
                }
                sc_k[gq] = v4_fdot2(q0, l0, sc_k[gq]);
                sc_k[gq] = v4_fdot2(q1, l1, sc_k[gq]);
                sc_k[gq] = v4_fdot2(q2, h0, sc_k[gq]);
                sc_k[gq] = v4_fdot2(q3, h1, sc_k[gq]);
            }
        }
    }

    // ---- 2. dequant ---------------------------------------------------------
    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        float p;
        if constexpr (KV_FMT == KV_Q4_1) {
            p = fmaf(sc_k[gq], g.k_d, (g.k_m - KBIAS * g.k_d) * qv.sum[gq]);
        } else {
            p = fmaf(sc_k[gq], g.k_d, (-KBIAS * g.k_d) * qv.sum[gq]);
        }
        p += __shfl_xor(p, 1, 32);
        p += __shfl_xor(p, 2, 32);
        if constexpr (!FULL) {
            p = kv_ok ? p : -INFINITY;
        }
        sc_k[gq] = p;
    }

    // ---- 3. RelBias (leader lane only) -------------------------------------
    if (blk == 0) {
        const int dist = dist0 - k_slot;
        bool do_rel = true;
        if constexpr (!FULL) {
            do_rel = kv_ok;
        }
        if constexpr (SWA) {
            if (do_rel) {
                #pragma unroll
                for (int gq = 0; gq < GQA; ++gq) {
                    sc_k[gq] += rel_base[(int64_t)gq * N_REL_EXTENT + dist] * V4_LOG2E;
                }
            }
        } else {
            if (do_rel && dist >= 0 && dist < N_REL_EXTENT) {
                #pragma unroll
                for (int gq = 0; gq < GQA; ++gq) {
                    sc_k[gq] += rel_base[(int64_t)gq * N_REL_EXTENT + dist] * V4_LOG2E;
                }
            }
        }
    }

    // ---- 4. tile-8 log2-domain online softmax ------------------------------
    float w_k[GQA];
    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        float mt = sc_k[gq];
        mt = fmaxf(mt, __shfl_xor(mt,  4, 32));
        mt = fmaxf(mt, __shfl_xor(mt,  8, 32));
        mt = fmaxf(mt, __shfl_xor(mt, 16, 32));
        mt = __shfl(mt, 0, 32);

        const float m_old = m_reg[gq];
        const float m_new = fmaxf(m_old, mt);
        if (m_new > m_old) {
            if (m_old > -INFINITY) {
                const float scale_old = v4_exp2f(m_old - m_new);
                o_reg[gq][0] *= scale_old;
                o_reg[gq][1] *= scale_old;
                o_reg[gq][2] *= scale_old;
                o_reg[gq][3] *= scale_old;
                ob_reg[gq]   *= scale_old;
                l_reg[gq]    *= scale_old;
            }
            m_reg[gq] = m_new;
        }

        const float wk = (blk == 0) ? v4_exp2f(sc_k[gq] - m_new) : 0.0f;
        w_k[gq] = wk;

        float lt = wk;
        lt += __shfl_xor(lt,  4, 32);
        lt += __shfl_xor(lt,  8, 32);
        lt += __shfl_xor(lt, 16, 32);
        l_reg[gq] += lt;
    }

    // ---- 5. P * V over all 8 keys, 4 output dims per lane ------------------
    if constexpr (KV_FMT == KV_Q8_0) {
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const float d_v = __shfl(g.v_d, kk * 4 + v_blk, 32);
            __half2 vh01, vh23;
            v4_i8x4_to_h2x2(g.vq[kk], vh01, vh23);
            #pragma unroll
            for (int gq = 0; gq < GQA; ++gq) {
                const float w_dv = __shfl(w_k[gq], kk * 4, 32) * d_v;
                o_reg[gq][0] = fmaf(__low2float(vh01),  w_dv, o_reg[gq][0]);
                o_reg[gq][1] = fmaf(__high2float(vh01), w_dv, o_reg[gq][1]);
                o_reg[gq][2] = fmaf(__low2float(vh23),  w_dv, o_reg[gq][2]);
                o_reg[gq][3] = fmaf(__high2float(vh23), w_dv, o_reg[gq][3]);
            }
        }
    } else {
        const int vsh = (lane & 1) << 2;
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t w_me = g.vq[p];
            const uint32_t w_ot = __shfl_xor(w_me, 1, 32);
            #pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int kk = 2 * p + s;
                const uint32_t w_cur = (((lane ^ s) & 1) ? w_ot : w_me);
                const uint32_t nibs  = (w_cur >> vsh) & 0x0F0F0F0Fu;
                __half2 vh01, vh23;
                if constexpr (KV_FMT == KV_Q4_0) {
                    v4_u4x4_to_h2x2_q40(nibs, vh01, vh23);
                } else {
                    v4_u4x4_to_h2x2(nibs, vh01, vh23);
                }
                if constexpr (KV_FMT == KV_Q4_1) {
                    const __half2 dm_v = v4_h2_of(__shfl(g.v_dm, kk * 4 + v_blk, 32));
                    const float d_v = __low2float(dm_v);
                    const float m_v = __high2float(dm_v);
                    #pragma unroll
                    for (int gq = 0; gq < GQA; ++gq) {
                        const float wg   = __shfl(w_k[gq], kk * 4, 32);
                        const float w_dv = wg * d_v;
                        ob_reg[gq] = fmaf(wg, m_v, ob_reg[gq]);
                        o_reg[gq][0] = fmaf(__low2float(vh01),  w_dv, o_reg[gq][0]);
                        o_reg[gq][1] = fmaf(__high2float(vh01), w_dv, o_reg[gq][1]);
                        o_reg[gq][2] = fmaf(__low2float(vh23),  w_dv, o_reg[gq][2]);
                        o_reg[gq][3] = fmaf(__high2float(vh23), w_dv, o_reg[gq][3]);
                    }
                } else {
                    const float d_v = __shfl(g.v_d, kk * 4 + v_blk, 32);
                    #pragma unroll
                    for (int gq = 0; gq < GQA; ++gq) {
                        const float w_dv = __shfl(w_k[gq], kk * 4, 32) * d_v;
                        o_reg[gq][0] = fmaf(__low2float(vh01),  w_dv, o_reg[gq][0]);
                        o_reg[gq][1] = fmaf(__high2float(vh01), w_dv, o_reg[gq][1]);
                        o_reg[gq][2] = fmaf(__low2float(vh23),  w_dv, o_reg[gq][2]);
                        o_reg[gq][3] = fmaf(__high2float(vh23), w_dv, o_reg[gq][3]);
                    }
                }
            }
        }
    }
}

// ============================================================================
// Split epilogue + in-kernel self-resetting combine (identical to v3)
// ============================================================================
template <int GQA, int NWARPS>
struct V4Smem {
    float o[NWARPS][GQA][HEAD_DIM];
    float m[NWARPS][GQA];
    float l[NWARPS][GQA];
};

template <int GQA, int NWARPS>
static __device__ __forceinline__ void v4_fa_epilogue(
        V4Smem<GQA, NWARPS> & smem, int & s_is_last,
        const float o_reg[GQA][4], const float m_reg[GQA], const float l_reg[GQA],
        const float ob_reg[GQA],
        int lane, int warp, int h_base, int grp_id, int split, int n_splits, int tok,
        const float * __restrict__ sinks_in, int sinks_stride,
        const float * __restrict__ qgate_in, int qgate_stride,
        float * __restrict__ out, float * __restrict__ tmp_o, float2 * __restrict__ tmp_meta,
        int32_t * __restrict__ fa_done) {
    constexpr int D = HEAD_DIM;

    __syncthreads();

    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        if (lane == 0) {
            smem.m[warp][gq] = m_reg[gq];
            smem.l[warp][gq] = l_reg[gq];
        }
        *(float4 *)(&smem.o[warp][gq][lane * 4]) =
            make_float4(o_reg[gq][0] + ob_reg[gq], o_reg[gq][1] + ob_reg[gq],
                        o_reg[gq][2] + ob_reg[gq], o_reg[gq][3] + ob_reg[gq]);
    }
    __syncthreads();

    if (warp == 0) {
        #pragma unroll
        for (int gq = 0; gq < GQA; ++gq) {
            float m_blk = -INFINITY;
            #pragma unroll
            for (int w = 0; w < NWARPS; ++w) {
                m_blk = fmaxf(m_blk, smem.m[w][gq]);
            }

            float l_blk = 0.0f;
            float oa0 = 0.0f, oa1 = 0.0f, oa2 = 0.0f, oa3 = 0.0f;
            if (m_blk > -INFINITY) {
                #pragma unroll
                for (int w = 0; w < NWARPS; ++w) {
                    const float mw = smem.m[w][gq];
                    if (mw > -INFINITY) {
                        const float sw = v4_exp2f(mw - m_blk);
                        l_blk += smem.l[w][gq] * sw;
                        const float4 ov = *(const float4 *)(&smem.o[w][gq][lane * 4]);
                        oa0 = fmaf(ov.x, sw, oa0);
                        oa1 = fmaf(ov.y, sw, oa1);
                        oa2 = fmaf(ov.z, sw, oa2);
                        oa3 = fmaf(ov.w, sw, oa3);
                    }
                }
            }

            const int h = h_base + gq;
            if (n_splits == 1) {
                const float sink_val_log2 = sinks_in[(size_t)tok * sinks_stride + h] * V4_LOG2E;
                const float m_final  = fmaxf(m_blk, sink_val_log2);
                const float scale_o  = (m_blk > -INFINITY) ? v4_exp2f(m_blk - m_final) : 0.0f;
                const float l_final  = l_blk * scale_o + v4_exp2f(sink_val_log2 - m_final);
                const float norm     = (scale_o / l_final) * KQ_SCALE;

                float v0 = oa0 * norm;
                float v1 = oa1 * norm;
                float v2 = oa2 * norm;
                float v3 = oa3 * norm;
                v4_fwht128_4(v0, v1, v2, v3, lane);

                if (qgate_in != nullptr) {
                    const float4 gv = *(const float4 *)(qgate_in + (size_t)tok * qgate_stride
                                                        + (size_t)h * D + (size_t)lane * 4);
                    v0 *= 1.0f / (1.0f + v4_exp2f(-gv.x * V4_LOG2E));
                    v1 *= 1.0f / (1.0f + v4_exp2f(-gv.y * V4_LOG2E));
                    v2 *= 1.0f / (1.0f + v4_exp2f(-gv.z * V4_LOG2E));
                    v3 *= 1.0f / (1.0f + v4_exp2f(-gv.w * V4_LOG2E));
                }
                *(float4 *)(out + ((size_t)tok * N_HEAD + h) * D + (size_t)lane * 4) =
                    make_float4(v0, v1, v2, v3);
            } else {
                const int64_t meta_idx = ((int64_t)tok * N_HEAD + h) * n_splits + split;
                if (lane == 0) {
                    tmp_meta[meta_idx] = make_float2(m_blk, l_blk);
                }
                *(float4 *)(tmp_o + meta_idx * D + (size_t)lane * 4) =
                    make_float4(oa0, oa1, oa2, oa3);
            }
        }
    }

    if (fa_done == nullptr || n_splits <= 1) {
        return;
    }

    v4_gl_fence_release();
    __syncthreads();
    if (warp == 0 && lane == 0) {
        v4_gl_fence_release();
        const int prev = __hip_atomic_fetch_add(&fa_done[grp_id], 1,
                                                __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_is_last = (prev == n_splits - 1) ? 1 : 0;
    }
    __syncthreads();

    if (!s_is_last) {
        return;
    }

    v4_gl_invalidate();
    if (warp == 0 && lane == 0) {
        __hip_atomic_store(&fa_done[grp_id], 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    if (warp < GQA) {
        const int h = h_base + warp;
        const int64_t base_idx = ((int64_t)tok * N_HEAD + h) * n_splits;
        const float sink_val_log2 = sinks_in[(size_t)tok * sinks_stride + h] * V4_LOG2E;

        const float2 ml_lane = (lane < n_splits) ? tmp_meta[base_idx + lane]
                                                 : make_float2(-INFINITY, 0.0f);
        float m_all = ml_lane.x;
        m_all = fmaxf(m_all, __shfl_xor(m_all, 16, 32));
        m_all = fmaxf(m_all, __shfl_xor(m_all,  8, 32));
        m_all = fmaxf(m_all, __shfl_xor(m_all,  4, 32));
        m_all = fmaxf(m_all, __shfl_xor(m_all,  2, 32));
        m_all = fmaxf(m_all, __shfl_xor(m_all,  1, 32));
        const float m_final = fmaxf(m_all, sink_val_log2);

        const float sc_lane = (lane < n_splits && ml_lane.x > -INFINITY)
            ? v4_exp2f(ml_lane.x - m_final) : 0.0f;
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
                const float4 pv0 = *(const float4 *)(tmp_o + (base_idx + sp + 0) * HEAD_DIM + lane * 4);
                const float4 pv1 = *(const float4 *)(tmp_o + (base_idx + sp + 1) * HEAD_DIM + lane * 4);
                const float4 pv2 = *(const float4 *)(tmp_o + (base_idx + sp + 2) * HEAD_DIM + lane * 4);
                const float4 pv3 = *(const float4 *)(tmp_o + (base_idx + sp + 3) * HEAD_DIM + lane * 4);
                o0 = fmaf(pv0.x, sc0, o0); o1 = fmaf(pv0.y, sc0, o1); o2 = fmaf(pv0.z, sc0, o2); o3 = fmaf(pv0.w, sc0, o3);
                o0 = fmaf(pv1.x, sc1, o0); o1 = fmaf(pv1.y, sc1, o1); o2 = fmaf(pv1.z, sc1, o2); o3 = fmaf(pv1.w, sc1, o3);
                o0 = fmaf(pv2.x, sc2, o0); o1 = fmaf(pv2.y, sc2, o1); o2 = fmaf(pv2.z, sc2, o2); o3 = fmaf(pv2.w, sc2, o3);
                o0 = fmaf(pv3.x, sc3, o0); o1 = fmaf(pv3.y, sc3, o1); o2 = fmaf(pv3.z, sc3, o2); o3 = fmaf(pv3.w, sc3, o3);
            } else if (m4 != 0u) {
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    if ((m4 >> k) & 1u) {
                        const float sc = __shfl(sc_lane, sp + k, 32);
                        const float4 pv = *(const float4 *)(tmp_o + (base_idx + sp + k) * HEAD_DIM + lane * 4);
                        o0 = fmaf(pv.x, sc, o0); o1 = fmaf(pv.y, sc, o1); o2 = fmaf(pv.z, sc, o2); o3 = fmaf(pv.w, sc, o3);
                    }
                }
            }
        }
        for (; sp < n_splits; ++sp) {
            if ((valid_mask >> sp) & 1u) {
                const float sc = __shfl(sc_lane, sp, 32);
                const float4 pv = *(const float4 *)(tmp_o + (base_idx + sp) * HEAD_DIM + lane * 4);
                o0 = fmaf(pv.x, sc, o0); o1 = fmaf(pv.y, sc, o1); o2 = fmaf(pv.z, sc, o2); o3 = fmaf(pv.w, sc, o3);
            }
        }

        l_sum += __shfl_xor(l_sum, 16, 32);
        l_sum += __shfl_xor(l_sum,  8, 32);
        l_sum += __shfl_xor(l_sum,  4, 32);
        l_sum += __shfl_xor(l_sum,  2, 32);
        l_sum += __shfl_xor(l_sum,  1, 32);
        const float l_final = l_sum + v4_exp2f(sink_val_log2 - m_final);

        const float inv_l = (1.0f / l_final) * KQ_SCALE;
        float v0 = o0 * inv_l;
        float v1 = o1 * inv_l;
        float v2 = o2 * inv_l;
        float v3 = o3 * inv_l;
        v4_fwht128_4(v0, v1, v2, v3, lane);

        if (qgate_in != nullptr) {
            const float4 gv = *(const float4 *)(qgate_in + (size_t)tok * qgate_stride
                                                + (size_t)h * HEAD_DIM + (size_t)lane * 4);
            v0 *= 1.0f / (1.0f + v4_exp2f(-gv.x * V4_LOG2E));
            v1 *= 1.0f / (1.0f + v4_exp2f(-gv.y * V4_LOG2E));
            v2 *= 1.0f / (1.0f + v4_exp2f(-gv.z * V4_LOG2E));
            v3 *= 1.0f / (1.0f + v4_exp2f(-gv.w * V4_LOG2E));
        }
        *(float4 *)(out + (size_t)h * HEAD_DIM + (size_t)lane * 4) = make_float4(v0, v1, v2, v3);
    }
}

// ============================================================================
// Kernel 1: SWA dense decode FA v4
// ============================================================================
template <int GQA, int KV_FMT, int NWARPS, bool QLDS, bool CACHED, bool MROW = false>
__launch_bounds__(NWARPS * 32, 2)
static __global__ void inkling_fa_decode_swa_v4_kernel(
        const float    * __restrict__ q,
        KVCacheSlot                   kv_slot,
        const float    * __restrict__ sinks_in,
        int                           sinks_stride,
        const float    * __restrict__ rel_table,
        const float    * __restrict__ qgate_in,
        int                           qgate_stride,
        float          * __restrict__ out,
        float          * __restrict__ tmp_o,
        float2         * __restrict__ tmp_meta,
        int                           n_splits,
        const StepMeta * __restrict__ d_meta,
        int32_t        * __restrict__ fa_done) {
    constexpr int TOK = 0;
    // Step 8C: one launch for all m verify rows (blockIdx.z = row).  Every
    // per-token input/output and the split-K scratch are shifted by the row; the
    // body (and therefore the arithmetic) is the exact decode kernel.
    if constexpr (MROW) {
        const int r = (int)blockIdx.x;
        if (r > 0) {
            const size_t vof = (size_t)r * N_Q;
            q += vof;
            out += vof;
            if (sinks_in != nullptr) sinks_in += (size_t)r * sinks_stride;
            if (qgate_in != nullptr) qgate_in += (size_t)r * qgate_stride;
            rel_table += (size_t)r * N_HEAD * N_REL_EXTENT;
            tmp_o    += (size_t)r * N_HEAD * n_splits * HEAD_DIM;
            tmp_meta += (size_t)r * N_HEAD * n_splits;
            d_meta   += r;
            if (fa_done != nullptr) fa_done += (size_t)r * N_HEAD_KV;
        }
    }

    const int split  = MROW ? (int)blockIdx.y : (int)blockIdx.x;
    const int grp_id = MROW ? (int)blockIdx.z : (int)blockIdx.y;
    const int lane   = (int)threadIdx.x;
    const int warp   = (int)threadIdx.y;
    const int blk    = lane & 3;
    const int k_slot = lane >> 2;
    const int v_blk  = lane >> 3;
    const int h_base = grp_id * GQA;

    const int cap     = kv_slot.cap;
    const int qp      = d_meta->pos;
    const int n_kv    = d_meta->kv_len_swa;
    const int n_valid = (n_kv < SWA_WINDOW) ? n_kv : SWA_WINDOW;
    int r_start = (qp - n_valid + 1) % cap;
    if (r_start < 0) {
        r_start += cap;
    }

    const int total_groups = (n_valid + 7) >> 3;
    const int tiles_per_split = (total_groups + n_splits - 1) / n_splits;
    const int ts0 = (split * tiles_per_split < total_groups) ? (split * tiles_per_split) : total_groups;
    const int ts1 = (ts0 + tiles_per_split < total_groups) ? (ts0 + tiles_per_split) : total_groups;

    __shared__ V4Smem<GQA, NWARPS> smem;
    __shared__ __align__(16) __half2 s_q[GQA][HEAD_DIM / 2];
    __shared__ int s_is_last;
    if (lane == 0 && warp == 0) {
        s_is_last = 0;
    }

    V4Q<GQA> qv;
    if constexpr (QLDS) {
        v4_stage_q_lds<GQA>(s_q, q, h_base, blk, k_slot);
        __syncthreads();
        v4_stage_q_lds_sum<GQA>(qv.sum, s_q, blk);
    } else {
        v4_stage_q<GQA>(qv, q, h_base, blk);
    }

    const size_t head_base = (size_t)grp_id * (size_t)cap;
    const float * rel_base = rel_table + (size_t)TOK * N_HEAD * N_REL_EXTENT
                                        + (size_t)h_base * N_REL_EXTENT;

    float o_reg[GQA][4];
    float m_reg[GQA];
    float l_reg[GQA];
    float ob_reg[GQA];
    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        o_reg[gq][0] = 0.0f;
        o_reg[gq][1] = 0.0f;
        o_reg[gq][2] = 0.0f;
        o_reg[gq][3] = 0.0f;
        m_reg[gq] = -INFINITY;
        l_reg[gq] = 0.0f;
        ob_reg[gq] = 0.0f;
    }

    int gi = ts0 + warp;
    if (gi < ts1) {
        V4KvGrp<KV_FMT> a, b;
        v4_load_kv_group<KV_FMT, CACHED ? 1 : 0>(a, kv_slot, head_base, cap, r_start + gi * 8, lane);

        for (; gi < ts1; gi += NWARPS) {
            const bool more = (gi + NWARPS < ts1);
            const int  gn   = more ? (gi + NWARPS) : gi;
            v4_load_kv_group<KV_FMT, CACHED ? 1 : 0>(b, kv_slot, head_base, cap, r_start + gn * 8, lane);

            const int dist0 = (n_valid - 1) - gi * 8;
            if (gi * 8 + 8 <= n_valid) {
                v4_fa_group<GQA, KV_FMT, true, true, QLDS>(
                    qv, s_q, a, rel_base, dist0, blk, lane, v_blk, k_slot,
                    true, o_reg, m_reg, l_reg, ob_reg);
            } else {
                v4_fa_group<GQA, KV_FMT, false, true, QLDS>(
                    qv, s_q, a, rel_base, dist0, blk, lane, v_blk, k_slot,
                    (gi * 8 + k_slot) < n_valid, o_reg, m_reg, l_reg, ob_reg);
            }
            if (more) {
                a = b;
            }
        }
    }

    v4_fa_epilogue<GQA, NWARPS>(smem, s_is_last, o_reg, m_reg, l_reg, ob_reg,
                                lane, warp, h_base, grp_id, split, n_splits, TOK,
                                sinks_in, sinks_stride, qgate_in, qgate_stride,
                                out, tmp_o, tmp_meta, fa_done);
}

// ============================================================================
// Kernel 3: sparse Global decode FA v4
// ============================================================================
template <int GQA, int KV_FMT, int NWARPS, bool QLDS, bool CACHED, bool MROW = false>
__launch_bounds__(NWARPS * 32, 2)
static __global__ void inkling_fa_decode_sparse_global_v4_kernel(
        const float    * __restrict__ q,
        KVCacheSlot                   kv_slot,
        const float    * __restrict__ sinks_in,
        int                           sinks_stride,
        const float    * __restrict__ rel_table,
        const float    * __restrict__ qgate_in,
        int                           qgate_stride,
        float          * __restrict__ out,
        const int32_t  * __restrict__ selected_pages,
        const int32_t  * __restrict__ n_selected,
        float          * __restrict__ tmp_o,
        float2         * __restrict__ tmp_meta,
        int                           n_splits,
        const StepMeta * __restrict__ d_meta,
        int32_t        * __restrict__ fa_done) {
    constexpr int TOK = 0;
    constexpr int GPG = SPARSE_PAGE_SIZE / 8;

    if constexpr (MROW) {
        const int r = (int)blockIdx.x;
        if (r > 0) {
            const size_t vof = (size_t)r * N_Q;
            q += vof;
            out += vof;
            if (sinks_in != nullptr) sinks_in += (size_t)r * sinks_stride;
            if (qgate_in != nullptr) qgate_in += (size_t)r * qgate_stride;
            rel_table      += (size_t)r * N_HEAD * N_REL_EXTENT;
            selected_pages += (size_t)r * N_HEAD_KV * SPARSE_MAX_SELECTED;
            n_selected     += (size_t)r * N_HEAD_KV;
            tmp_o          += (size_t)r * N_HEAD * n_splits * HEAD_DIM;
            tmp_meta       += (size_t)r * N_HEAD * n_splits;
            d_meta         += r;
            if (fa_done != nullptr) fa_done += (size_t)r * N_HEAD_KV;
        }
    }

    const int split  = MROW ? (int)blockIdx.y : (int)blockIdx.x;
    const int grp_id = MROW ? (int)blockIdx.z : (int)blockIdx.y;
    const int lane   = (int)threadIdx.x;
    const int warp   = (int)threadIdx.y;
    const int blk    = lane & 3;
    const int k_slot = lane >> 2;
    const int v_blk  = lane >> 3;
    const int h_base = grp_id * GQA;

    const int qp    = d_meta->pos;
    const int n_kv  = d_meta->kv_len_global;
    const int n_sel = n_selected[grp_id];
    const int pps   = (n_sel + n_splits - 1) / n_splits;
    const int pi0   = (split * pps < n_sel) ? (split * pps) : n_sel;
    const int pi1   = (pi0 + pps < n_sel) ? (pi0 + pps) : n_sel;

    __shared__ V4Smem<GQA, NWARPS> smem;
    __shared__ __align__(16) __half2 s_q[GQA][HEAD_DIM / 2];
    __shared__ int s_is_last;
    if (lane == 0 && warp == 0) {
        s_is_last = 0;
    }

    V4Q<GQA> qv;
    if constexpr (QLDS) {
        v4_stage_q_lds<GQA>(s_q, q, h_base, blk, k_slot);
        __syncthreads();
        v4_stage_q_lds_sum<GQA>(qv.sum, s_q, blk);
    } else {
        v4_stage_q<GQA>(qv, q, h_base, blk);
    }

    const size_t head_base = (size_t)grp_id * (size_t)kv_slot.cap;
    const float * rel_base = rel_table + (size_t)TOK * N_HEAD * N_REL_EXTENT
                                        + (size_t)h_base * N_REL_EXTENT;
    const int32_t * sel = selected_pages + (size_t)grp_id * SPARSE_MAX_SELECTED;

    float o_reg[GQA][4];
    float m_reg[GQA];
    float l_reg[GQA];
    float ob_reg[GQA];
    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        o_reg[gq][0] = 0.0f;
        o_reg[gq][1] = 0.0f;
        o_reg[gq][2] = 0.0f;
        o_reg[gq][3] = 0.0f;
        m_reg[gq] = -INFINITY;
        l_reg[gq] = 0.0f;
        ob_reg[gq] = 0.0f;
    }

    const int total_groups = (pi1 - pi0) * GPG;

    auto tile_pos = [&](int gg) {
        const int page = pi0 + (gg >> 3);
        return sel[page] * SPARSE_PAGE_SIZE + ((gg & (GPG - 1)) * 8);
    };

    int gg = warp;
    if (gg < total_groups) {
        V4KvGrp<KV_FMT> a, b;
        v4_load_kv_group_abs<KV_FMT, CACHED ? 1 : 0>(a, kv_slot, head_base, tile_pos(gg), n_kv, lane);

        for (; gg < total_groups; gg += NWARPS) {
            const bool more = (gg + NWARPS < total_groups);
            const int  gn   = more ? (gg + NWARPS) : gg;
            v4_load_kv_group_abs<KV_FMT, CACHED ? 1 : 0>(b, kv_slot, head_base, tile_pos(gn), n_kv, lane);

            const int s_abs = tile_pos(gg);
            const int dist0 = qp - s_abs;
            if (s_abs + 8 <= n_kv) {
                v4_fa_group<GQA, KV_FMT, true, false, QLDS>(
                    qv, s_q, a, rel_base, dist0, blk, lane, v_blk, k_slot,
                    true, o_reg, m_reg, l_reg, ob_reg);
            } else if (s_abs < n_kv) {
                v4_fa_group<GQA, KV_FMT, false, false, QLDS>(
                    qv, s_q, a, rel_base, dist0, blk, lane, v_blk, k_slot,
                    (s_abs + k_slot) < n_kv, o_reg, m_reg, l_reg, ob_reg);
            }
            if (more) {
                a = b;
            }
        }
    }

    v4_fa_epilogue<GQA, NWARPS>(smem, s_is_last, o_reg, m_reg, l_reg, ob_reg,
                                lane, warp, h_base, grp_id, split, n_splits, TOK,
                                sinks_in, sinks_stride, qgate_in, qgate_stride,
                                out, tmp_o, tmp_meta, fa_done);
}

// ============================================================================
// Kernel 2: fused page score + Top-K v4 (both metadata formats)
// ============================================================================
static __device__ __forceinline__ uint32_t v4_score_key(float v) {
    const uint32_t b = __builtin_bit_cast(uint32_t, v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

template <int GQA>
struct V4SelQ {
    __half2 pos[8];   // 16 dims, sum_g 0.5*(q_g + alpha*|q_g|)
    __half2 neg[8];   // 16 dims, sum_g 0.5*(q_g - alpha*|q_g|)
    float   sum;      // sum_d (pos + neg) (fp16-consistent)
};

// META_FMT: 0 = SPARSE_META_Q8 (int8 min/max, 2 x 16 B per lane per page),
//           1 = SPARSE_META_Q4 (nibble-packed min|max, 1 x 16 B per lane).
template <int GQA, int META_FMT, bool QLDS, bool CACHED>
__launch_bounds__(FA_V4_SEL_THREADS, 3) __attribute__((amdgpu_waves_per_eu(6, 6)))
static __global__ void sparse_global_select_v4_kernel(
        const float * __restrict__ q,
        SparseGlobalPageMeta meta,
        const StepMeta * __restrict__ d_meta,
        float * __restrict__ page_scores,
        int32_t * __restrict__ selected_pages,
        int32_t * __restrict__ n_selected,
        int32_t * __restrict__ done,
        int page_budget) {
    constexpr int SEL_SLOTS = FA_V4_SEL_SLOTS;

    const int h    = (int)blockIdx.y;
    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int n_kv    = d_meta->kv_len_global;
    const int n_pages = (n_kv + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
    int32_t * out = selected_pages + (size_t)h * SPARSE_MAX_SELECTED;

    if (n_pages <= page_budget) {
        if (blockIdx.x == 0 && tid == 0) {
            for (int i = 0; i < n_pages; ++i) {
                out[i] = i;
            }
            n_selected[h] = n_pages;
        }
        return;
    }

    const int     cand_begin = SPARSE_SINK_PAGES;
    const int     cand_end   = n_pages - SPARSE_RECENT_PAGES;
    const int     n_cand     = cand_end - cand_begin;
    const size_t  hbase      = (size_t)h * (size_t)meta.max_pages;

    // ---- Stage 1: page scores (8 lanes per page, 4 pages per warp) ---------
    {
        __shared__ float s_q_unrot[GQA][HEAD_DIM];
        if (warp < GQA) {
            float4 t = ((const float4 *)(q + (size_t)(h * GQA + warp) * HEAD_DIM))[lane];
            v4_fwht128_4(t.x, t.y, t.z, t.w, lane);
            float n2 = t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
            n2 += __shfl_xor(n2, 16, 32);
            n2 += __shfl_xor(n2,  8, 32);
            n2 += __shfl_xor(n2,  4, 32);
            n2 += __shfl_xor(n2,  2, 32);
            n2 += __shfl_xor(n2,  1, 32);
            const float inv_n = rsqrtf(n2 + 1e-12f);
            t.x *= inv_n; t.y *= inv_n; t.z *= inv_n; t.w *= inv_n;
            ((float4 *)(s_q_unrot[warp]))[lane] = t;
        }
        __syncthreads();

        const int pg  = lane >> 3;
        const int sub = lane & 7;    // 16 dims owned by this lane

        __shared__ __align__(16) __half2 s_qp[HEAD_DIM / 2];
        __shared__ __align__(16) __half2 s_qn[HEAD_DIM / 2];

        float q_hi[16], q_lo[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) { q_hi[k] = 0.0f; q_lo[k] = 0.0f; }
        #pragma unroll
        for (int gq = 0; gq < GQA; ++gq) {
            const float4 * qrow = (const float4 *)(s_q_unrot[gq] + sub * 16);
            #pragma unroll
            for (int i4 = 0; i4 < 4; ++i4) {
                const float4 t = qrow[i4];
                const float qv[4] = { t.x, t.y, t.z, t.w };
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float a = SPARSE_QUEST_ALPHA * fabsf(qv[j]);
                    q_hi[4 * i4 + j] += 0.5f * (qv[j] + a);
                    q_lo[4 * i4 + j] += 0.5f * (qv[j] - a);
                }
            }
        }
        V4SelQ<GQA> sq;
        const __half2 ones = v4_h2_of(0x3C003C00u);
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            sq.pos[j] = __floats2half2_rn(q_hi[2 * j], q_hi[2 * j + 1]);
            sq.neg[j] = __floats2half2_rn(q_lo[2 * j], q_lo[2 * j + 1]);
            acc = v4_fdot2(sq.pos[j], ones, acc);
            acc = v4_fdot2(sq.neg[j], ones, acc);
        }
        sq.sum = acc;

        if constexpr (QLDS) {
            if (pg == 0) {
                #pragma unroll
                for (int j = 0; j < 8; ++j) {
                    s_qp[sub * 8 + j] = sq.pos[j];
                    s_qn[sub * 8 + j] = sq.neg[j];
                }
            }
            __syncthreads();
        }

        const size_t lane_off = (size_t)sub * 16;
        const int n_warps = (int)gridDim.x * 8;
        const int gw      = (int)blockIdx.x * 8 + warp;
        const int mychunk = (n_cand + n_warps - 1) / n_warps;
        const int p0      = cand_begin + gw * mychunk;
        const int p1      = (p0 + mychunk < cand_end) ? (p0 + mychunk) : cand_end;
        const int phi     = cand_end - 1;

        if (p0 < p1) {
            uint4 w0, w1;
            float dv;
            auto fetch4 = [&](int pp, uint4 & o0, uint4 & o1, float & odv) {
                const int pc = (pp + pg < phi) ? (pp + pg) : phi;
                const size_t off = (size_t)pc * HEAD_DIM + lane_off;
                if constexpr (META_FMT == 1) {
                    const uint8_t * kqmm_p = meta.k_qmm + hbase * HEAD_DIM;
                    o0 = v4_ld128_u4<CACHED ? 1 : 0>(kqmm_p + off);
                    o1 = o0;
                } else {
                    const int8_t * kmin_p = meta.k_min_q + hbase * HEAD_DIM;
                    const int8_t * kmax_p = meta.k_max_q + hbase * HEAD_DIM;
                    o0 = v4_ld128_u4<CACHED ? 1 : 0>(kmin_p + off);
                    o1 = v4_ld128_u4<CACHED ? 1 : 0>(kmax_p + off);
                }
                odv = __half2float(meta.k_mm_d[hbase + (size_t)pc]);
            };
            auto store4 = [&](const uint4 & o0, const uint4 & o1, float odv, int pp) {
                float s = 0.0f;
                const uint32_t wmn[4] = { o0.x, o0.y, o0.z, o0.w };
                const uint32_t wmx[4] = { o1.x, o1.y, o1.z, o1.w };
                // 16 dims split into two 8-dim groups so each Q LDS read is a
                // single 16-byte ds_read_b128 covering 4 half2 = words 2g, 2g+1.
                #pragma unroll
                for (int i = 0; i < 2; ++i) {
                    __half2 mn0, mn1, mx0, mx1, mn2, mn3, mx2, mx3;
                    if constexpr (META_FMT == 1) {
                        // low nibble = q_min + 8, high nibble = q_max + 8
                        v4_u4x4_biased(wmn[2 * i + 0] & 0x0F0F0F0Fu, mn0, mn1);
                        v4_u4x4_biased((wmn[2 * i + 0] >> 4) & 0x0F0F0F0Fu, mx0, mx1);
                        v4_u4x4_biased(wmn[2 * i + 1] & 0x0F0F0F0Fu, mn2, mn3);
                        v4_u4x4_biased((wmn[2 * i + 1] >> 4) & 0x0F0F0F0Fu, mx2, mx3);
                    } else {
                        uint32_t umn = wmn[2 * i + 0] ^ 0x80808080u;
                        uint32_t umx = wmx[2 * i + 0] ^ 0x80808080u;
                        mn0 = v4_perm_lo(umn); mn1 = v4_perm_hi(umn);
                        mx0 = v4_perm_lo(umx); mx1 = v4_perm_hi(umx);
                        umn = wmn[2 * i + 1] ^ 0x80808080u;
                        umx = wmx[2 * i + 1] ^ 0x80808080u;
                        mn2 = v4_perm_lo(umn); mn3 = v4_perm_hi(umn);
                        mx2 = v4_perm_lo(umx); mx3 = v4_perm_hi(umx);
                    }
                    __half2 p0, p1, p2, p3, n0, n1, n2, n3;
                    if constexpr (QLDS) {
                        const uint4 tp = *(const uint4 *)(&s_qp[sub * 8 + 4 * i]);
                        const uint4 tn = *(const uint4 *)(&s_qn[sub * 8 + 4 * i]);
                        p0 = v4_h2_of(tp.x); p1 = v4_h2_of(tp.y);
                        p2 = v4_h2_of(tp.z); p3 = v4_h2_of(tp.w);
                        n0 = v4_h2_of(tn.x); n1 = v4_h2_of(tn.y);
                        n2 = v4_h2_of(tn.z); n3 = v4_h2_of(tn.w);
                    } else {
                        p0 = sq.pos[4 * i + 0]; p1 = sq.pos[4 * i + 1];
                        p2 = sq.pos[4 * i + 2]; p3 = sq.pos[4 * i + 3];
                        n0 = sq.neg[4 * i + 0]; n1 = sq.neg[4 * i + 1];
                        n2 = sq.neg[4 * i + 2]; n3 = sq.neg[4 * i + 3];
                    }
                    s = v4_fdot2(p0, mx0, s);
                    s = v4_fdot2(p1, mx1, s);
                    s = v4_fdot2(p2, mx2, s);
                    s = v4_fdot2(p3, mx3, s);
                    s = v4_fdot2(n0, mn0, s);
                    s = v4_fdot2(n1, mn1, s);
                    s = v4_fdot2(n2, mn2, s);
                    s = v4_fdot2(n3, mn3, s);
                }
                constexpr float KBIAS = (META_FMT == 1) ? 1032.0f : 1152.0f;
                s = (s - KBIAS * sq.sum) * odv;
                s += __shfl_xor(s, 1, 8);
                s += __shfl_xor(s, 2, 8);
                s += __shfl_xor(s, 4, 8);
                if (sub == 0 && (pp + pg) < p1) {
                    page_scores[hbase + (size_t)(pp + pg)] = s;
                }
            };

            int p = p0;
            fetch4(p, w0, w1, dv);
            while (p + 4 < p1) {
                uint4 n0, n1;
                float dv1;
                fetch4(p + 4, n0, n1, dv1);
                store4(w0, w1, dv, p);
                w0 = n0; w1 = n1; dv = dv1;
                p += 4;
            }
            store4(w0, w1, dv, p);
        }
    }

    // ---- Stage 2: exact Top-K of this head, run by the last chunk ----------
    __shared__ uint32_t s_hist[256];
    __shared__ int      s_ctl[24];
    __shared__ int      s_last;

    v4_gl_fence_release();
    __syncthreads();
    if (tid == 0) {
        const int prev = __hip_atomic_fetch_add(&done[h], 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = (prev == (int)gridDim.x - 1) ? 1 : 0;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    v4_gl_invalidate();
    if (tid == 0) {
        __hip_atomic_store(&done[h], 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }

    int K = page_budget - SPARSE_MIN_PAGES;
    const int kmax = SPARSE_MAX_SELECTED - SPARSE_SINK_PAGES - SPARSE_RECENT_PAGES;
    K = (K < kmax) ? K : kmax;
    K = (K < n_cand) ? K : n_cand;

    if (tid == 0) {
        out[0] = 0;
        out[1] = 1;
        #pragma unroll
        for (int i = 0; i < SPARSE_RECENT_PAGES; ++i) {
            out[SPARSE_SINK_PAGES + K + i] = n_pages - SPARSE_RECENT_PAGES + i;
        }
        n_selected[h] = SPARSE_SINK_PAGES + K + SPARSE_RECENT_PAGES;
    }
    __syncthreads();
    if (K <= 0) {
        return;
    }

    constexpr int NTHR = FA_V4_SEL_THREADS;
    uint32_t key[SEL_SLOTS];
    #pragma unroll
    for (int i = 0; i < SEL_SLOTS; ++i) {
        const int idx = tid + i * NTHR;
        key[i] = (idx < n_cand)
            ? v4_score_key(page_scores[hbase + (size_t)(cand_begin + idx)])
            : 0u;
    }

    if (tid == 0) {
        s_ctl[8] = 0;
        s_ctl[9] = 0;
    }
    __syncthreads();

    uint32_t prefix = 0u;
    int need = K;
    for (int shift = 24; shift >= 0; shift -= 8) {
        s_hist[tid] = 0u;
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < SEL_SLOTS; ++i) {
            const int idx = tid + i * NTHR;
            if (idx < n_cand) {
                const uint32_t k = key[i];
                if (shift == 24 || ((k >> (shift + 8)) == (prefix >> (shift + 8)))) {
                    atomicAdd(&s_hist[(k >> shift) & 0xFFu], 1u);
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
        const uint32_t above_next = __shfl_down(suf, 1, 32);
        const uint32_t above_within = (lane < 31) ? above_next : 0u;
        const uint32_t above = above_warp + above_within;
        if (above < (uint32_t)need && above + hb >= (uint32_t)need) {
            s_ctl[8] = tid;
            s_ctl[9] = (int)above;
        }
        __syncthreads();
        prefix |= ((uint32_t)s_ctl[8]) << shift;
        need -= s_ctl[9];
        __syncthreads();
    }

    int gt = 0, eq = 0;
    #pragma unroll
    for (int i = 0; i < SEL_SLOTS; ++i) {
        const int idx = tid + i * NTHR;
        if (idx < n_cand) {
            if (key[i] > prefix) {
                ++gt;
            } else if (key[i] == prefix) {
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
    for (int i = 0; i < SEL_SLOTS; ++i) {
        const int idx = tid + i * NTHR;
        if (idx < n_cand) {
            if (key[i] > prefix) {
                out[SPARSE_SINK_PAGES + wgt++] = cand_begin + idx;
            } else if (key[i] == prefix && weq < need) {
                out[SPARSE_SINK_PAGES + n_gt + weq] = cand_begin + idx;
                ++weq;
            }
        }
    }
}

// ============================================================================
// Host launchers
// ============================================================================
template <int GQA, int NWARPS, bool QLDS, bool CACHED, bool MROW = false>
static inline void v4_launch_swa(int kv_fmt, int q_lds, int cached, const dim3 & grid,
        hipStream_t stream,
        const float * q, KVCacheSlot kv_slot, const float * sinks_in, int sinks_stride,
        const float * rel_table, const float * qgate_in, int qgate_stride,
        float * out, float * tmp_o, float2 * tmp_meta, int n_splits,
        const StepMeta * d_meta, int32_t * fa_done) {
    const dim3 block(32, NWARPS, 1);
    (void)q_lds; (void)cached;
#define V4_SWA_CASE(FMT)                                                                  \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_swa_v4_kernel<GQA, FMT, NWARPS, QLDS, CACHED, MROW>), \
        grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,            \
        qgate_in, qgate_stride, out, tmp_o, tmp_meta, n_splits, d_meta, fa_done)
    switch (kv_fmt) {
        case KV_Q4_1: V4_SWA_CASE(KV_Q4_1); break;
        case KV_Q4_0: V4_SWA_CASE(KV_Q4_0); break;
        default:      V4_SWA_CASE(KV_Q8_0); break;
    }
#undef V4_SWA_CASE
}

template <int GQA, int NWARPS, bool QLDS, bool CACHED, bool MROW = false>
static inline void v4_launch_global(int kv_fmt, const dim3 & grid, hipStream_t stream,
        const float * q, KVCacheSlot kv_slot, const float * sinks_in, int sinks_stride,
        const float * rel_table, const float * qgate_in, int qgate_stride,
        float * out, const int32_t * selected_pages, const int32_t * n_selected,
        float * tmp_o, float2 * tmp_meta, int n_splits,
        const StepMeta * d_meta, int32_t * fa_done) {
    const dim3 block(32, NWARPS, 1);
#define V4_GLB_CASE(FMT)                                                                  \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_sparse_global_v4_kernel<GQA, FMT, NWARPS, QLDS, CACHED, MROW>), \
        grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,            \
        qgate_in, qgate_stride, out, selected_pages, n_selected, tmp_o, tmp_meta,         \
        n_splits, d_meta, fa_done)
    switch (kv_fmt) {
        case KV_Q4_1: V4_GLB_CASE(KV_Q4_1); break;
        case KV_Q4_0: V4_GLB_CASE(KV_Q4_0); break;
        default:      V4_GLB_CASE(KV_Q8_0); break;
    }
#undef V4_GLB_CASE
}

static inline int v4_clamp_splits(int n_splits) {
    int splits = n_splits;
    if (splits < 1) {
        splits = 1;
    }
    if (splits > FA_V4_MAX_SPLITS) {
        splits = FA_V4_MAX_SPLITS;
    }
    return splits;
}

} // namespace

// ----------------------------------------------------------------------------
// 1. SWA dense decode FA v4
// ----------------------------------------------------------------------------
void launch_fa_swa_decode_v4(
        const float * q,
        KVCacheSlot   kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,
        const float * qgate_in, int qgate_stride,
        float       * out,
        float       * tmp_o,
        float2      * tmp_meta,
        int           n_splits,
        const StepMeta * d_meta,
        hipStream_t   stream,
        int32_t     * fa_done,
        int           q_lds,
        int           cached) {
    if (d_meta == nullptr || q == nullptr || rel_table == nullptr) {
        return;
    }
    int splits = v4_clamp_splits(n_splits);
    if (splits > 1 && (tmp_o == nullptr || tmp_meta == nullptr || fa_done == nullptr)) {
        splits = 1;
    }

    const dim3 grid(splits, N_HEAD_KV, 1);
    if (q_lds != 0) {
        if (cached != 0) {
            v4_launch_swa<GQA_RATIO, FA_V4_NWARPS, true, true>(kv_slot.kv_type, 1, 1, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, tmp_o, tmp_meta, splits, d_meta, fa_done);
        } else {
            v4_launch_swa<GQA_RATIO, FA_V4_NWARPS, true, false>(kv_slot.kv_type, 1, 0, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, tmp_o, tmp_meta, splits, d_meta, fa_done);
        }
    } else {
        if (cached != 0) {
            v4_launch_swa<GQA_RATIO, FA_V4_NWARPS, false, true>(kv_slot.kv_type, 0, 1, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, tmp_o, tmp_meta, splits, d_meta, fa_done);
        } else {
            v4_launch_swa<GQA_RATIO, FA_V4_NWARPS, false, false>(kv_slot.kv_type, 0, 0, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, tmp_o, tmp_meta, splits, d_meta, fa_done);
        }
    }
}

// Step 8C/8D: all `nrows` verify rows in ONE launch (blockIdx.x = row,
// blockIdx.y = split, blockIdx.z = grp_id so all m rows share L2/L3 KV tiles).
void launch_fa_swa_decode_v4_mrows(
        const float * q,
        KVCacheSlot   kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,
        const float * qgate_in, int qgate_stride,
        float       * out,
        float       * tmp_o,
        float2      * tmp_meta,
        int           n_splits,
        const StepMeta * d_meta,
        hipStream_t   stream,
        int32_t     * fa_done,
        int           q_lds,
        int           cached,
        int           nrows) {
    if (d_meta == nullptr || q == nullptr || rel_table == nullptr || nrows <= 0) {
        return;
    }
    if (nrows == 1) {
        launch_fa_swa_decode_v4(q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in,
                                qgate_stride, out, tmp_o, tmp_meta, n_splits, d_meta,
                                stream, fa_done, q_lds, cached);
        return;
    }
    int splits = v4_clamp_splits(n_splits);
    if (splits > 1 && (tmp_o == nullptr || tmp_meta == nullptr || fa_done == nullptr)) {
        splits = 1;
    }
    const dim3 grid((unsigned)nrows, splits, N_HEAD_KV);
    if (q_lds != 0) {
        if (cached != 0) {
            v4_launch_swa<GQA_RATIO, FA_V4_NWARPS, true, true, true>(kv_slot.kv_type, 1, 1, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, tmp_o, tmp_meta, splits, d_meta, fa_done);
        } else {
            v4_launch_swa<GQA_RATIO, FA_V4_NWARPS, true, false, true>(kv_slot.kv_type, 1, 0, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, tmp_o, tmp_meta, splits, d_meta, fa_done);
        }
    } else {
        if (cached != 0) {
            v4_launch_swa<GQA_RATIO, FA_V4_NWARPS, false, true, true>(kv_slot.kv_type, 0, 1, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, tmp_o, tmp_meta, splits, d_meta, fa_done);
        } else {
            v4_launch_swa<GQA_RATIO, FA_V4_NWARPS, false, false, true>(kv_slot.kv_type, 0, 0, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, tmp_o, tmp_meta, splits, d_meta, fa_done);
        }
    }
}

// ----------------------------------------------------------------------------
// 2. Sparse Global decode FA v4
// ----------------------------------------------------------------------------
void launch_fa_sparse_global_decode_v4(
        const float * q,
        KVCacheSlot   kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,
        const float * qgate_in, int qgate_stride,
        float       * out,
        const int32_t * selected_pages,
        const int32_t * n_selected_pages,
        float       * tmp_o,
        float2      * tmp_meta,
        int           n_splits,
        const StepMeta * d_meta,
        hipStream_t   stream,
        int32_t     * fa_done,
        int           q_lds,
        int           cached) {
    if (d_meta == nullptr || q == nullptr || rel_table == nullptr ||
        selected_pages == nullptr || n_selected_pages == nullptr) {
        return;
    }
    int splits = v4_clamp_splits(n_splits);
    if (splits > 1 && (tmp_o == nullptr || tmp_meta == nullptr || fa_done == nullptr)) {
        splits = 1;
    }

    const dim3 grid(splits, N_HEAD_KV, 1);
    if (q_lds != 0) {
        if (cached != 0) {
            v4_launch_global<GQA_RATIO, FA_V4_NWARPS, true, true>(kv_slot.kv_type, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, selected_pages, n_selected_pages, tmp_o, tmp_meta, splits, d_meta, fa_done);
        } else {
            v4_launch_global<GQA_RATIO, FA_V4_NWARPS, true, false>(kv_slot.kv_type, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, selected_pages, n_selected_pages, tmp_o, tmp_meta, splits, d_meta, fa_done);
        }
    } else {
        if (cached != 0) {
            v4_launch_global<GQA_RATIO, FA_V4_NWARPS, false, true>(kv_slot.kv_type, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, selected_pages, n_selected_pages, tmp_o, tmp_meta, splits, d_meta, fa_done);
        } else {
            v4_launch_global<GQA_RATIO, FA_V4_NWARPS, false, false>(kv_slot.kv_type, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, selected_pages, n_selected_pages, tmp_o, tmp_meta, splits, d_meta, fa_done);
        }
    }
}

void launch_fa_sparse_global_decode_v4_mrows(
        const float * q,
        KVCacheSlot   kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,
        const float * qgate_in, int qgate_stride,
        float       * out,
        const int32_t * selected_pages,
        const int32_t * n_selected_pages,
        float       * tmp_o,
        float2      * tmp_meta,
        int           n_splits,
        const StepMeta * d_meta,
        hipStream_t   stream,
        int32_t     * fa_done,
        int           q_lds,
        int           cached,
        int           nrows) {
    if (d_meta == nullptr || q == nullptr || rel_table == nullptr ||
        selected_pages == nullptr || n_selected_pages == nullptr || nrows <= 0) {
        return;
    }
    if (nrows == 1) {
        launch_fa_sparse_global_decode_v4(q, kv_slot, sinks_in, sinks_stride, rel_table,
            qgate_in, qgate_stride, out, selected_pages, n_selected_pages, tmp_o, tmp_meta,
            n_splits, d_meta, stream, fa_done, q_lds, cached);
        return;
    }
    int splits = v4_clamp_splits(n_splits);
    if (splits > 1 && (tmp_o == nullptr || tmp_meta == nullptr || fa_done == nullptr)) {
        splits = 1;
    }

    const dim3 grid((unsigned)nrows, splits, N_HEAD_KV);
    if (q_lds != 0) {
        if (cached != 0) {
            v4_launch_global<GQA_RATIO, FA_V4_NWARPS, true, true, true>(kv_slot.kv_type, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, selected_pages, n_selected_pages, tmp_o, tmp_meta, splits, d_meta, fa_done);
        } else {
            v4_launch_global<GQA_RATIO, FA_V4_NWARPS, true, false, true>(kv_slot.kv_type, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, selected_pages, n_selected_pages, tmp_o, tmp_meta, splits, d_meta, fa_done);
        }
    } else {
        if (cached != 0) {
            v4_launch_global<GQA_RATIO, FA_V4_NWARPS, false, true, true>(kv_slot.kv_type, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, selected_pages, n_selected_pages, tmp_o, tmp_meta, splits, d_meta, fa_done);
        } else {
            v4_launch_global<GQA_RATIO, FA_V4_NWARPS, false, false, true>(kv_slot.kv_type, grid, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, selected_pages, n_selected_pages, tmp_o, tmp_meta, splits, d_meta, fa_done);
        }
    }
}

// ----------------------------------------------------------------------------
// 3. Fused page score + Top-K v4
// ----------------------------------------------------------------------------
void launch_sparse_global_select_v4(
        const float * q,
        const SparseGlobalPageMeta & meta,
        SparseGlobalScratch & scratch,
        const StepMeta * d_meta,
        int             sparse_pages,
        int32_t       * select_done,
        hipStream_t     stream,
        int             q_lds,
        int             cached,
        int             chunks) {
    if (q == nullptr || d_meta == nullptr || scratch.page_scores == nullptr ||
        scratch.selected_pages == nullptr || scratch.n_selected == nullptr ||
        select_done == nullptr) {
        return;
    }
    int budget = sparse_pages;
    if (budget < SPARSE_MIN_PAGES) {
        budget = SPARSE_MIN_PAGES;
    }
    if (budget > SPARSE_MAX_SELECTED - 2) {
        budget = SPARSE_MAX_SELECTED - 2;
    }

    if (chunks <= 0) {
        chunks = FA_V4_SELECT_CHUNKS;
    }
    if (chunks > 256) {
        chunks = 256;
    }
    const dim3 grid((unsigned)chunks, N_HEAD_KV, 1);
    const dim3 block(FA_V4_SEL_THREADS, 1, 1);

#define V4_SEL_LAUNCH(FMT, QL, CA)                                                            \
    hipLaunchKernelGGL(HIP_KERNEL_NAME(sparse_global_select_v4_kernel<GQA_RATIO, FMT, QL, CA>), \
        grid, block, 0, stream, q, meta, d_meta, scratch.page_scores,                      \
        scratch.selected_pages, scratch.n_selected, select_done, budget)

    const bool q4 = (meta.fmt == SPARSE_META_Q4);
    if (q4) {
        if (q_lds != 0) {
            if (cached != 0) { V4_SEL_LAUNCH(1, true, true); } else { V4_SEL_LAUNCH(1, true, false); }
        } else {
            if (cached != 0) { V4_SEL_LAUNCH(1, false, true); } else { V4_SEL_LAUNCH(1, false, false); }
        }
    } else {
        if (q_lds != 0) {
            if (cached != 0) { V4_SEL_LAUNCH(0, true, true); } else { V4_SEL_LAUNCH(0, true, false); }
        } else {
            if (cached != 0) { V4_SEL_LAUNCH(0, false, true); } else { V4_SEL_LAUNCH(0, false, false); }
        }
    }
#undef V4_SEL_LAUNCH
}

} // namespace inkling
