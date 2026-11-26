// ============================================================================
// src/kernels_fa_decode_v3.hip — Step 7G
//
//  * RDNA3 `v_dot2_f32_f16` SWA + sparse-Global decode FA (Q in registers, the
//    1024-bias nibble trick folded into one scalar per (head, key)).
//  * Quest page-score / Top-K with the q+ / q- algebraic rewrite.
//  * QKV+qkvr super-fused prep at 6 waves/SIMD.
//
// See include/kernels_fa_decode_v3.h for the measured before/after numbers, the
// ISA evidence and the projected savings.  Every helper that is shared with an
// existing kernel is a bit-exact copy of the shipped implementation (the code
// base is one file per kernel family and these files do not include each
// other), so this translation unit is self-contained.
// ============================================================================

#include "kernels_fa_decode_v3.h"

#include <cstdint>

namespace inkling {

namespace {

constexpr int V3_WARP_SIZE = 32;
constexpr float V3_LOG2E = 1.44269504088896340736f;

// ext_vector_type (not HIP_vector_type) so `__builtin_nontemporal_load` accepts
// it and emits global_load_b128 ... slc dlc.
typedef uint32_t v3u4 __attribute__((ext_vector_type(4)));
typedef int32_t  v3i4 __attribute__((ext_vector_type(4)));

static_assert(sizeof(__half2) == 4, "half2 must be 4 bytes");

// ============================================================================
// gfx1100 primitives
// ============================================================================

// RDNA3 native 1-cycle v_exp_f32 (2^x).  Every use is exp2(x) with x <= 0.
static __device__ __forceinline__ float v3_exp2f(float x) {
    return __builtin_amdgcn_exp2f(x);
}

static __device__ __forceinline__ __half2 v3_h2_of(uint32_t u) {
    return __builtin_bit_cast(__half2, u);
}

// RDNA3 `v_dot2_f32_f16`: 2 fp16 products accumulated into one FP32 register.
static __device__ __forceinline__ float v3_fdot2(__half2 a, __half2 b, float c) {
    return __builtin_amdgcn_fdot2(a, b, c, false);
}

// The IEEE-754 fp16 magic: 0x6400 == 1024.0 and the mantissa LSB of 1024.0 is
// exactly 1.0, so `as_half2(0x6400 | n)` == half(1024 + n) for every n in
// 0..255, bit-exactly, with no conversion instruction.  `v_perm_b32` plants the
// 0x64 high byte, so ONE perm turns two nibbles/bytes into a biased half2.
static __device__ __forceinline__ __half2 v3_perm_lo(uint32_t x) {
    return v3_h2_of(__builtin_amdgcn_perm(0x64646464u, x, 0x04010400u));
}
static __device__ __forceinline__ __half2 v3_perm_hi(uint32_t x) {
    return v3_h2_of(__builtin_amdgcn_perm(0x64646464u, x, 0x04030402u));
}

// 4 unsigned nibbles (low nibble of each of the 4 bytes) -> two biased half2.
static __device__ __forceinline__ void v3_u4x4_biased(uint32_t nibs, __half2 & h01, __half2 & h23) {
    h01 = v3_perm_lo(nibs);
    h23 = v3_perm_hi(nibs);
}

// 4 Q8 int8 -> two biased half2 (value == half(1024 + (int8 + 128))).
static __device__ __forceinline__ void v3_i8x4_biased(uint32_t w, __half2 & h01, __half2 & h23) {
    const uint32_t u = w ^ 0x80808080u;
    h01 = v3_perm_lo(u);
    h23 = v3_perm_hi(u);
}

// Unbiased variants (the P.V path keeps v2's exact values).
static __device__ __forceinline__ void v3_u4x4_to_h2x2(uint32_t nibs, __half2 & h01, __half2 & h23) {
    h01 = __hsub2(v3_perm_lo(nibs), v3_h2_of(0x64006400u));
    h23 = __hsub2(v3_perm_hi(nibs), v3_h2_of(0x64006400u));
}
// q4_0's V value is d*(nibble - 8): the fp16 bias is 1032 (v1/v2's
// SIGNED_Q4_0 path).  The 1024-only form dropped the -8 and broke q4_0 PPL.
static __device__ __forceinline__ void v3_u4x4_to_h2x2_q40(uint32_t nibs, __half2 & h01, __half2 & h23) {
    h01 = __hsub2(v3_perm_lo(nibs), v3_h2_of(0x64086408u));
    h23 = __hsub2(v3_perm_hi(nibs), v3_h2_of(0x64086408u));
}

static __device__ __forceinline__ void v3_i8x4_to_h2x2(uint32_t w, __half2 & h01, __half2 & h23) {
    const uint32_t u = w ^ 0x80808080u;
    h01 = __hsub2(v3_perm_lo(u), v3_h2_of(0x64806480u));
    h23 = __hsub2(v3_perm_hi(u), v3_h2_of(0x64806480u));
}

// Wave32 orthogonal FWHT-128 on the 4 output dims owned by each lane.
static __device__ __forceinline__ void v3_fwht128_4(
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

static __device__ __forceinline__ void v3_gl_fence_release() {
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    // gfx12: no equivalent instruction - on gfx1201 __scoped_atomic_thread_fence(RELEASE, AGENT)
    // above already lowers to s_waitcnt + `global_wb scope:SCOPE_DEV` (ISA-checked with
    // llvm-mc/hipcc -S), so gfx1200/gfx1201 need nothing extra here.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#endif
}

static __device__ __forceinline__ void v3_gl_invalidate() {
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    // gfx12: no equivalent instruction - scope-qualified acquire atomics on gfx1201 emit
    // `global_inv scope:SCOPE_DEV` themselves (ISA-checked), which replaces the RDNA3
    // L0/L1 invalidation below and needs no hand-written cache asm.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("buffer_gl1_inv\n\tbuffer_gl0_inv" ::: "memory");
#endif
}

// ============================================================================
// FA v3 — register-resident fp16 Q
// ============================================================================
//
// Lane mapping (identical to v1/v2, so the arithmetic is unchanged):
//   lane = k_slot * 4 + blk,  k_slot = lane >> 2 in [0,8), blk = lane & 3
//   K:  this lane owns dims [blk*32, blk*32+32) of key k_slot
//   V:  every lane accumulates output dims [lane*4, +4) over all 8 keys
//
// Q: this lane converts its own 32 dims of each of the GQA heads to fp16 once
// and keeps them in registers -- 16 __half2 per head, 48 VGPRs for GQA = 3.
//
// q4_*/q8_0 nibble packing (verified against store_kv_quantized): word m of a
// token covers dims 8m..8m+7, low nibbles = dims 8m+0..3, high nibbles =
// dims 8m+4..7.  So a lane's 4 words are exactly its 32 dims in order, and
// qh[g][2t], qh[g][2t+1] are the pairs (2t, 2t+1) of those 32 dims.
// ----------------------------------------------------------------------------
constexpr int V3_QPAIRS = 16;   // 32 dims / 2

template <int GQA>
struct V3Q {
    __half2 q[GQA][V3_QPAIRS];  // fp16 Q * Q_SCALE*LOG2E, dims [blk*32, +32)
    float   sum[GQA];           // sum of the fp16 dims (nibble-bias fold)
};

template <int GQA>
static __device__ __forceinline__ void v3_stage_q(
        V3Q<GQA> & qv, const float * __restrict__ q, int h_base, int blk) {
    constexpr float QS = KQ_SCALE * V3_LOG2E;
    // Summing a half2 pair in FP32 is exactly what v_dot2_f32_f16 does with a
    // (1,1) multiplier: one instruction instead of two cvt + two add.
    const __half2 ones = v3_h2_of(0x3C003C00u);
    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        const float4 * qp = (const float4 *)(q + (size_t)(h_base + gq) * HEAD_DIM + (size_t)blk * 32);
        #pragma unroll
        for (int i4 = 0; i4 < 8; ++i4) {
            const float4 f = qp[i4];
            qv.q[gq][2 * i4 + 0] = __floats2half2_rn(f.x * QS, f.y * QS);
            qv.q[gq][2 * i4 + 1] = __floats2half2_rn(f.z * QS, f.w * QS);
        }
        // Summed from the *fp16* values so the -BIAS fold is exact.
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < V3_QPAIRS; ++j) {
            acc = v3_fdot2(qv.q[gq][j], ones, acc);
        }
        qv.sum[gq] = acc;
    }
}

// Bias of the nibble/byte representation: value == half(BIAS + n) - BIAS.
template <int KV_FMT> struct V3Nib {};                                       // 1024
template <> struct V3Nib<KV_Q4_1> { static constexpr float BIAS = 1024.0f; };// 1024
template <> struct V3Nib<KV_Q4_0> { static constexpr float BIAS = 1032.0f; };// 1024 + 8
template <> struct V3Nib<KV_Q8_0> { static constexpr float BIAS = 1152.0f; };// 1024 + 128

template <int KV_FMT> struct V3KvGrp {};
template <> struct V3KvGrp<KV_Q8_0> {
    v3u4     k_lo, k_hi;  // int8 dims [blk*32 + 0, +16) / [+16, +16)
    float    k_d;
    uint32_t vq[8];
    float    v_d;
};
template <> struct V3KvGrp<KV_Q4_1> {
    v3u4     k_qs;        // 32 nibbles = dims [blk*32, +32)
    float    k_d, k_m;
    uint32_t vq[4];
    uint32_t v_dm;
};
template <> struct V3KvGrp<KV_Q4_0> {
    v3u4     k_qs;
    float    k_d;
    uint32_t vq[4];
    float    v_d;
};

template <int KV_FMT>
static __device__ __forceinline__ void v3_load_kv_group(
        V3KvGrp<KV_FMT> & g, const KVCacheSlot & kv,
        size_t head_base, int cap, int ring0, int lane, int blk) {
    const int kslot = lane >> 2;
    int jk = ring0 + kslot;
    jk -= (jk >= cap) ? cap : 0;
    const size_t ks = head_base + (size_t)jk;

    if constexpr (KV_FMT == KV_Q8_0) {
        const int8_t * kp = kv.k_qs + ks * HEAD_DIM + (size_t)blk * 32;
        g.k_lo = *(const v3u4 *)(kp);
        g.k_hi = *(const v3u4 *)(kp + 16);
        g.k_d  = __half2float(__ldg(&kv.k_d[ks * 4 + (size_t)blk]));
        g.v_d  = __half2float(__ldg(&kv.v_d[ks * 4 + (size_t)blk]));
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            int jt = ring0 + kk;
            jt -= (jt >= cap) ? cap : 0;
            g.vq[kk] = *((const uint32_t *)(kv.v_qs + (head_base + (size_t)jt) * HEAD_DIM) + lane);
        }
    } else {
        const int8_t * kp = kv.k_qs + ks * (HEAD_DIM / 2) + (size_t)blk * 16;
        g.k_qs = *(const v3u4 *)kp;
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
            g.vq[p] = *(((const uint32_t *)(kv.v_qs + (head_base + (size_t)jt) * (HEAD_DIM / 2))) + vhalf);
        }
    }
}

// Global variant: absolute position, positional clamp, no ring arithmetic.
template <int KV_FMT>
static __device__ __forceinline__ void v3_load_kv_group_abs(
        V3KvGrp<KV_FMT> & g, const KVCacheSlot & kv,
        size_t head_base, int base, int n_kv, int lane, int blk) {
    const int jmax = (n_kv > 0) ? (n_kv - 1) : 0;
    const int jk = ((base + (lane >> 2)) < jmax) ? (base + (lane >> 2)) : jmax;
    const size_t ks = head_base + (size_t)jk;

    if constexpr (KV_FMT == KV_Q8_0) {
        const int8_t * kp = kv.k_qs + ks * HEAD_DIM + (size_t)blk * 32;
        g.k_lo = __builtin_nontemporal_load((const v3u4 *)(kp));
        g.k_hi = __builtin_nontemporal_load((const v3u4 *)(kp + 16));
        g.k_d  = __half2float(__ldg(&kv.k_d[ks * 4 + (size_t)blk]));
        g.v_d  = __half2float(__ldg(&kv.v_d[ks * 4 + (size_t)blk]));
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const int jt = ((base + kk) < jmax) ? (base + kk) : jmax;
            g.vq[kk] = __builtin_nontemporal_load(
                ((const uint32_t *)(kv.v_qs + (head_base + (size_t)jt) * HEAD_DIM)) + lane);
        }
    } else {
        const int8_t * kp = kv.k_qs + ks * (HEAD_DIM / 2) + (size_t)blk * 16;
        g.k_qs = __builtin_nontemporal_load((const v3u4 *)kp);
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
            g.vq[p] = __builtin_nontemporal_load(
                ((const uint32_t *)(kv.v_qs + (head_base + (size_t)jt) * (HEAD_DIM / 2))) + vhalf);
        }
    }
}

// ============================================================================
// One 8-key wave tile
// ============================================================================
// FULL  == true : all 8 keys are valid (no masking at all).
// SWA   == true : RelBias always applies (window <= N_REL_EXTENT).
// `dist0` is the RelBias distance of key k_slot == 0.
// ----------------------------------------------------------------------------
template <int GQA, int KV_FMT, bool FULL, bool SWA>
static __device__ __forceinline__ void v3_fa_group(
        const V3Q<GQA> & qv,
        const V3KvGrp<KV_FMT> & g,
        const float * __restrict__ rel_base,
        int   dist0,
        int   blk, int lane, int v_blk, int k_slot,
        bool  kv_ok,
        float o_reg[GQA][4], float m_reg[GQA], float l_reg[GQA], float ob_reg[GQA]) {
    constexpr float KBIAS = V3Nib<KV_FMT>::BIAS;

    // ---- 1. K . Q^T for the GQA heads that share this KV head ---------------
    // Two nibbles per v_perm_b32, two MACs per v_dot2_f32_f16; the -BIAS is
    // folded into one scalar FMA per (head, key) below.
    float sc_k[GQA];
    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        sc_k[gq] = 0.0f;
    }

    if constexpr (KV_FMT == KV_Q8_0) {
        const uint32_t ku[8] = { g.k_lo.x, g.k_lo.y, g.k_lo.z, g.k_lo.w,
                                 g.k_hi.x, g.k_hi.y, g.k_hi.z, g.k_hi.w };
        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            __half2 kh01, kh23;
            v3_i8x4_biased(ku[i], kh01, kh23);
            #pragma unroll
            for (int gq = 0; gq < GQA; ++gq) {
                sc_k[gq] = v3_fdot2(qv.q[gq][2 * i + 0], kh01, sc_k[gq]);
                sc_k[gq] = v3_fdot2(qv.q[gq][2 * i + 1], kh23, sc_k[gq]);
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
            v3_u4x4_biased(lo, l0, l1);
            v3_u4x4_biased(hi, h0, h1);
            #pragma unroll
            for (int gq = 0; gq < GQA; ++gq) {
                sc_k[gq] = v3_fdot2(qv.q[gq][4 * i + 0], l0, sc_k[gq]);
                sc_k[gq] = v3_fdot2(qv.q[gq][4 * i + 1], l1, sc_k[gq]);
                sc_k[gq] = v3_fdot2(qv.q[gq][4 * i + 2], h0, sc_k[gq]);
                sc_k[gq] = v3_fdot2(qv.q[gq][4 * i + 3], h1, sc_k[gq]);
            }
        }
    }

    // ---- 2. dequant: score = k_d*dot_biased + (k_zero - BIAS*k_d)*q_sum ----
    // q4_1:  sum_d q*(d*q4 + m)      == d*dot + m*q_sum
    // q4_0:  sum_d q*(d*(q4-8))      == d*dot - 1032*d*q_sum
    // q8_0:  sum_d q*(d*i8)          == d*dot - 1152*d*q_sum
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
                    sc_k[gq] += rel_base[(int64_t)gq * N_REL_EXTENT + dist] * V3_LOG2E;
                }
            }
        } else {
            if (do_rel && dist >= 0 && dist < N_REL_EXTENT) {
                #pragma unroll
                for (int gq = 0; gq < GQA; ++gq) {
                    sc_k[gq] += rel_base[(int64_t)gq * N_REL_EXTENT + dist] * V3_LOG2E;
                }
            }
        }
    }

    // ---- 4. tile-8 log2-domain online softmax ------------------------------
    // The xor-4/8/16 butterfly alone leaves a *per-blk* max, not the tile max:
    // the RelBias below is added on the blk==0 lanes only, so the 4 lanes of a
    // key hold different scores and the 8 remainders mod 4 never mix.  The
    // lane-0 broadcast (identical to v1/v2) is therefore REQUIRED -- without it
    // m_new, the o/l/ob rescale and l_reg diverge across the 4 lane groups
    // (measured: 1-10% output error per split with rel != 0, exact at rel == 0).
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
                const float scale_old = v3_exp2f(m_old - m_new);
                o_reg[gq][0] *= scale_old;
                o_reg[gq][1] *= scale_old;
                o_reg[gq][2] *= scale_old;
                o_reg[gq][3] *= scale_old;
                ob_reg[gq]   *= scale_old;
                l_reg[gq]    *= scale_old;
            }
            m_reg[gq] = m_new;
        }

        const float wk = (blk == 0) ? v3_exp2f(sc_k[gq] - m_new) : 0.0f;
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
            v3_i8x4_to_h2x2(g.vq[kk], vh01, vh23);
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
        // Both lanes of a pair hold the same 32-bit V word (8 dims) for two
        // consecutive keys; xor-1 hands each lane the partner's key.
        const int vsh = (lane & 1) << 2;   // 0 or 4
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
                    v3_u4x4_to_h2x2_q40(nibs, vh01, vh23);
                } else {
                    v3_u4x4_to_h2x2(nibs, vh01, vh23);
                }
                if constexpr (KV_FMT == KV_Q4_1) {
                    const __half2 dm_v = v3_h2_of(__shfl(g.v_dm, kk * 4 + v_blk, 32));
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
// Split epilogue + in-kernel self-resetting combine (identical to v2)
// ============================================================================
template <int GQA, int NWARPS>
struct V3Smem {
    float o[NWARPS][GQA][HEAD_DIM];
    float m[NWARPS][GQA];
    float l[NWARPS][GQA];
};

template <int GQA, int NWARPS>
static __device__ __forceinline__ void v3_fa_epilogue(
        V3Smem<GQA, NWARPS> & smem, int & s_is_last,
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
                        const float sw = v3_exp2f(mw - m_blk);
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
                const float sink_val_log2 = sinks_in[(size_t)tok * sinks_stride + h] * V3_LOG2E;
                const float m_final  = fmaxf(m_blk, sink_val_log2);
                const float scale_o  = (m_blk > -INFINITY) ? v3_exp2f(m_blk - m_final) : 0.0f;
                const float l_final  = l_blk * scale_o + v3_exp2f(sink_val_log2 - m_final);
                const float norm     = (scale_o / l_final) * KQ_SCALE;

                float v0 = oa0 * norm;
                float v1 = oa1 * norm;
                float v2 = oa2 * norm;
                float v3 = oa3 * norm;
                v3_fwht128_4(v0, v1, v2, v3, lane);

                if (qgate_in != nullptr) {
                    const float4 gv = *(const float4 *)(qgate_in + (size_t)tok * qgate_stride
                                                        + (size_t)h * D + (size_t)lane * 4);
                    v0 *= 1.0f / (1.0f + v3_exp2f(-gv.x * V3_LOG2E));
                    v1 *= 1.0f / (1.0f + v3_exp2f(-gv.y * V3_LOG2E));
                    v2 *= 1.0f / (1.0f + v3_exp2f(-gv.z * V3_LOG2E));
                    v3 *= 1.0f / (1.0f + v3_exp2f(-gv.w * V3_LOG2E));
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

    v3_gl_fence_release();
    __syncthreads();
    if (warp == 0 && lane == 0) {
        v3_gl_fence_release();
        const int prev = __hip_atomic_fetch_add(&fa_done[grp_id], 1,
                                                __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_is_last = (prev == n_splits - 1) ? 1 : 0;
    }
    __syncthreads();

    if (!s_is_last) {
        return;
    }

    v3_gl_invalidate();
    if (warp == 0 && lane == 0) {
        __hip_atomic_store(&fa_done[grp_id], 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    if (warp < GQA) {
        const int h = h_base + warp;
        const int64_t base_idx = ((int64_t)tok * N_HEAD + h) * n_splits;
        const float sink_val_log2 = sinks_in[(size_t)tok * sinks_stride + h] * V3_LOG2E;

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
            ? v3_exp2f(ml_lane.x - m_final) : 0.0f;
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
        const float l_final = l_sum + v3_exp2f(sink_val_log2 - m_final);

        const float inv_l = (1.0f / l_final) * KQ_SCALE;
        float v0 = o0 * inv_l;
        float v1 = o1 * inv_l;
        float v2 = o2 * inv_l;
        float v3 = o3 * inv_l;
        v3_fwht128_4(v0, v1, v2, v3, lane);

        if (qgate_in != nullptr) {
            const float4 gv = *(const float4 *)(qgate_in + (size_t)tok * qgate_stride
                                                + (size_t)h * HEAD_DIM + (size_t)lane * 4);
            v0 *= 1.0f / (1.0f + v3_exp2f(-gv.x * V3_LOG2E));
            v1 *= 1.0f / (1.0f + v3_exp2f(-gv.y * V3_LOG2E));
            v2 *= 1.0f / (1.0f + v3_exp2f(-gv.z * V3_LOG2E));
            v3 *= 1.0f / (1.0f + v3_exp2f(-gv.w * V3_LOG2E));
        }
        *(float4 *)(out + (size_t)h * HEAD_DIM + (size_t)lane * 4) = make_float4(v0, v1, v2, v3);
    }
}

// ============================================================================
// Kernel 1: SWA dense decode FA v3 (window = SWA_WINDOW, 40 layers)
// ============================================================================
template <int GQA, int KV_FMT, int NWARPS>
__launch_bounds__(NWARPS * 32, 2) __attribute__((amdgpu_waves_per_eu(4, 4)))
static __global__ void inkling_fa_decode_swa_v3_kernel(
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

    const int split  = (int)blockIdx.x;
    const int grp_id = (int)blockIdx.y;
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

    __shared__ V3Smem<GQA, NWARPS> smem;
    __shared__ int s_is_last;
    if (lane == 0 && warp == 0) {
        s_is_last = 0;
    }

    V3Q<GQA> qv;
    v3_stage_q<GQA>(qv, q, h_base, blk);

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

    // Cyclic tile distribution over the CTA's warps: with 11 tiles per split and
    // 8 warps every warp gets 1-2 tiles (v2's contiguous chunking gave the first
    // warps 3 and left the last 3 warps of 4 idle).
    int gi = ts0 + warp;
    if (gi < ts1) {
        V3KvGrp<KV_FMT> a, b;
        v3_load_kv_group<KV_FMT>(a, kv_slot, head_base, cap, r_start + gi * 8, lane, blk);

        for (; gi < ts1; gi += NWARPS) {
            const bool more = (gi + NWARPS < ts1);
            const int  gn   = more ? (gi + NWARPS) : gi;
            v3_load_kv_group<KV_FMT>(b, kv_slot, head_base, cap, r_start + gn * 8, lane, blk);

            const int dist0 = (n_valid - 1) - gi * 8;
            if (gi * 8 + 8 <= n_valid) {
                v3_fa_group<GQA, KV_FMT, true, true>(
                    qv, a, rel_base, dist0, blk, lane, v_blk, k_slot,
                    true, o_reg, m_reg, l_reg, ob_reg);
            } else {
                v3_fa_group<GQA, KV_FMT, false, true>(
                    qv, a, rel_base, dist0, blk, lane, v_blk, k_slot,
                    (gi * 8 + k_slot) < n_valid, o_reg, m_reg, l_reg, ob_reg);
            }
            if (more) {
                a = b;
            }
        }
    }

    v3_fa_epilogue<GQA, NWARPS>(smem, s_is_last, o_reg, m_reg, l_reg, ob_reg,
                                lane, warp, h_base, grp_id, split, n_splits, TOK,
                                sinks_in, sinks_stride, qgate_in, qgate_stride,
                                out, tmp_o, tmp_meta, fa_done);
}

// ============================================================================
// Kernel 3: sparse Global decode FA v3 (selected 64-position pages, 8 layers)
// ============================================================================
template <int GQA, int KV_FMT, int NWARPS>
__launch_bounds__(NWARPS * 32, 2) __attribute__((amdgpu_waves_per_eu(4, 4)))
static __global__ void inkling_fa_decode_sparse_global_v3_kernel(
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
    constexpr int GPG = SPARSE_PAGE_SIZE / 8;   // 8 tiles of 8 keys per page

    const int split  = (int)blockIdx.x;
    const int grp_id = (int)blockIdx.y;
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

    __shared__ V3Smem<GQA, NWARPS> smem;
    __shared__ int s_is_last;
    if (lane == 0 && warp == 0) {
        s_is_last = 0;
    }

    V3Q<GQA> qv;
    v3_stage_q<GQA>(qv, q, h_base, blk);

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
        V3KvGrp<KV_FMT> a, b;
        v3_load_kv_group_abs<KV_FMT>(a, kv_slot, head_base, tile_pos(gg), n_kv, lane, blk);

        for (; gg < total_groups; gg += NWARPS) {
            const bool more = (gg + NWARPS < total_groups);
            const int  gn   = more ? (gg + NWARPS) : gg;
            v3_load_kv_group_abs<KV_FMT>(b, kv_slot, head_base, tile_pos(gn), n_kv, lane, blk);

            const int s_abs = tile_pos(gg);
            const int dist0 = qp - s_abs;
            if (s_abs + 8 <= n_kv) {
                v3_fa_group<GQA, KV_FMT, true, false>(
                    qv, a, rel_base, dist0, blk, lane, v_blk, k_slot,
                    true, o_reg, m_reg, l_reg, ob_reg);
            } else if (s_abs < n_kv) {
                v3_fa_group<GQA, KV_FMT, false, false>(
                    qv, a, rel_base, dist0, blk, lane, v_blk, k_slot,
                    (s_abs + k_slot) < n_kv, o_reg, m_reg, l_reg, ob_reg);
            }
            if (more) {
                a = b;
            }
        }
    }

    v3_fa_epilogue<GQA, NWARPS>(smem, s_is_last, o_reg, m_reg, l_reg, ob_reg,
                                lane, warp, h_base, grp_id, split, n_splits, TOK,
                                sinks_in, sinks_stride, qgate_in, qgate_stride,
                                out, tmp_o, tmp_meta, fa_done);
}

// ============================================================================
// Kernel 2: fused page score + Top-K v3
// ============================================================================
// Stage 1: 8 lanes per page (16 dims each), 4 pages per Wave32, two
//          global_load_b128 ... slc dlc + one half per page.
//   bound(page) = sum_d max(q_d*kmin_d, q_d*kmax_d) * scale
//               = [ sum_d (q+_d*kmax_d + q-_d*kmin_d) ] * scale     (exactly!)
//   q+ = max(q,0), q- = min(q,0)  ->  no fmaxf, half the products, and the
//   int8 -> fp16 unpack is `(byte^0x80)|0x6400` (1 v_perm_b32 per 2 values).
//   The -1152 of that bias is folded into ONE scalar per head, because
//   q+ + q- == q, so  sum_d q+ + sum_d q- == sum_d q.
//   The shared page scale is applied once per page, after the sum.
// Stage 2: last arriving chunk per head runs the v2 4-pass 8-bit radix select
//   with the 16 candidate keys per thread held in VGPRs (exact, deterministic,
//   ties broken in scan order).
// ----------------------------------------------------------------------------
static __device__ __forceinline__ uint32_t v3_score_key(float v) {
    const uint32_t b = __builtin_bit_cast(uint32_t, v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

template <int GQA>
struct V3SelQ {
    __half2 pos[8];   // 16 dims, sum_g 0.5*(q_g + alpha*|q_g|)
    __half2 neg[8];   // 16 dims, sum_g 0.5*(q_g - alpha*|q_g|)
    float   sum;      // sum_d (pos + neg) (fp16-consistent)
};

template <int GQA>
__launch_bounds__(FA_V3_SEL_THREADS, 3) __attribute__((amdgpu_waves_per_eu(6, 6)))
static __global__ void sparse_global_select_v3_kernel(
        const float * __restrict__ q,                 // [N_HEAD, HEAD_DIM]
        SparseGlobalPageMeta meta,
        const StepMeta * __restrict__ d_meta,
        float * __restrict__ page_scores,             // [N_HEAD_KV][max_pages]
        int32_t * __restrict__ selected_pages,        // [N_HEAD_KV][SPARSE_MAX_SELECTED]
        int32_t * __restrict__ n_selected,            // [N_HEAD_KV]
        int32_t * __restrict__ done,                  // [N_HEAD_KV] zeroed, self-resetting
        int page_budget) {
    constexpr int SEL_SLOTS = FA_V3_SEL_SLOTS;

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
    const int8_t * kmin_p    = meta.k_min_q + hbase * HEAD_DIM;
    const int8_t * kmax_p    = meta.k_max_q + hbase * HEAD_DIM;
    const __half * d_p       = meta.k_mm_d + hbase;

    // ---- Stage 1: page scores (8 lanes per page, 4 pages per warp) ---------
    {
        __shared__ float s_q_unrot[GQA][HEAD_DIM];
        if (warp < GQA) {
            float4 t = ((const float4 *)(q + (size_t)(h * GQA + warp) * HEAD_DIM))[lane];
            v3_fwht128_4(t.x, t.y, t.z, t.w, lane);
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

        const int pg  = lane >> 3;   // page slot inside the warp (0..3)
        const int sub = lane & 7;    // 16 dims owned by this lane

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
        V3SelQ<GQA> sq;
        const __half2 ones = v3_h2_of(0x3C003C00u);
        float acc = 0.0f;
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            sq.pos[j] = __floats2half2_rn(q_hi[2 * j], q_hi[2 * j + 1]);
            sq.neg[j] = __floats2half2_rn(q_lo[2 * j], q_lo[2 * j + 1]);
            acc = v3_fdot2(sq.pos[j], ones, acc);
            acc = v3_fdot2(sq.neg[j], ones, acc);
        }
        sq.sum = acc;

        const size_t lane_off = (size_t)sub * 16;
        const int n_warps = (int)gridDim.x * 8;
        const int gw      = (int)blockIdx.x * 8 + warp;
        const int mychunk = (n_cand + n_warps - 1) / n_warps;
        const int p0      = cand_begin + gw * mychunk;
        const int p1      = (p0 + mychunk < cand_end) ? (p0 + mychunk) : cand_end;
        const int phi     = cand_end - 1;

        if (p0 < p1) {
            v3u4 mn, mx;
            float dv;
            auto fetch4 = [&](int pp, v3u4 & omn, v3u4 & omx, float & odv) {
                const int pc = (pp + pg < phi) ? (pp + pg) : phi;
                const size_t off = (size_t)pc * HEAD_DIM + lane_off;
                omn = __builtin_nontemporal_load((const v3u4 *)(kmin_p + off));
                omx = __builtin_nontemporal_load((const v3u4 *)(kmax_p + off));
                odv = __half2float(d_p[pc]);
            };
            auto store4 = [&](const v3u4 & omn, const v3u4 & omx, float odv, int pp) {
                float s = 0.0f;
                const uint32_t wmn[4] = { omn.x, omn.y, omn.z, omn.w };
                const uint32_t wmx[4] = { omx.x, omx.y, omx.z, omx.w };
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t umn = wmn[i] ^ 0x80808080u;
                    const uint32_t umx = wmx[i] ^ 0x80808080u;
                    const __half2 mn0 = v3_perm_lo(umn);
                    const __half2 mn1 = v3_perm_hi(umn);
                    const __half2 mx0 = v3_perm_lo(umx);
                    const __half2 mx1 = v3_perm_hi(umx);
                    s = v3_fdot2(sq.pos[2 * i + 0], mx0, s);
                    s = v3_fdot2(sq.pos[2 * i + 1], mx1, s);
                    s = v3_fdot2(sq.neg[2 * i + 0], mn0, s);
                    s = v3_fdot2(sq.neg[2 * i + 1], mn1, s);
                }
                s = (s - 1152.0f * sq.sum) * odv;
                s += __shfl_xor(s, 1, 8);
                s += __shfl_xor(s, 2, 8);
                s += __shfl_xor(s, 4, 8);
                if (sub == 0 && (pp + pg) < p1) {
                    page_scores[hbase + (size_t)(pp + pg)] = s;
                }
            };

            int p = p0;
            fetch4(p, mn, mx, dv);
            while (p + 4 < p1) {
                v3u4 mn1, mx1;
                float dv1;
                fetch4(p + 4, mn1, mx1, dv1);
                store4(mn, mx, dv, p);
                mn = mn1; mx = mx1; dv = dv1;
                p += 4;
            }
            store4(mn, mx, dv, p);
        }
    }

    // ---- Stage 2: exact Top-K of this head, run by the last chunk ----------
    __shared__ uint32_t s_hist[256];
    __shared__ int      s_ctl[24];
    __shared__ int      s_last;

    v3_gl_fence_release();
    __syncthreads();
    if (tid == 0) {
        const int prev = __hip_atomic_fetch_add(&done[h], 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = (prev == (int)gridDim.x - 1) ? 1 : 0;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    v3_gl_invalidate();
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

    constexpr int NTHR = FA_V3_SEL_THREADS;
    uint32_t key[SEL_SLOTS];
    #pragma unroll
    for (int i = 0; i < SEL_SLOTS; ++i) {
        const int idx = tid + i * NTHR;
        key[i] = (idx < n_cand)
            ? v3_score_key(page_scores[hbase + (size_t)(cand_begin + idx)])
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
// Kernel 4: QKV GEMV + qkvr_prep fused v3
// ============================================================================
// Helpers below are verbatim copies of src/kernels_decode_fused.hip so the
// arithmetic (and therefore the KV cache bytes) is bit-identical; only the
// launch shape (8 warps, 32 rows/block) and the LDS layout changed.
// ----------------------------------------------------------------------------
struct V3QkvSeg {
    const int8_t * qs;
    const __half * d;
    const float *  x;
    float *        y;
    int            rows;
    int            blk_end;
};

static __device__ __forceinline__ void v3_rdna3_flush_stores() {
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    // gfx12: no equivalent instruction - on gfx1201 __scoped_atomic_thread_fence(RELEASE, AGENT)
    // above already lowers to s_waitcnt + `global_wb scope:SCOPE_DEV` (ISA-checked with
    // llvm-mc/hipcc -S), so gfx1200/gfx1201 need nothing extra here.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#endif
}

static __device__ __forceinline__ void v3_rdna3_inv_l0_l1() {
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    // gfx12: no equivalent instruction - scope-qualified acquire atomics on gfx1201 emit
    // `global_inv scope:SCOPE_DEV` themselves (ISA-checked), which replaces the RDNA3
    // L0/L1 invalidation below and needs no hand-written cache asm.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("buffer_gl1_inv\n\tbuffer_gl0_inv" ::: "memory");
#endif
}

static __device__ __forceinline__ int v3_scratch_add(float * scratch, int slot) {
    return __hip_atomic_fetch_add(reinterpret_cast<int32_t *>(scratch + slot), 1,
                                  __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
}
static __device__ __forceinline__ int v3_scratch_load(float * scratch, int slot) {
    return __hip_atomic_load(reinterpret_cast<int32_t *>(scratch + slot),
                             __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
static __device__ __forceinline__ void v3_scratch_store(float * scratch, int slot, int v) {
    __hip_atomic_store(reinterpret_cast<int32_t *>(scratch + slot), v,
                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
static __device__ __forceinline__ void v3_spin_until(float * scratch, int slot, int target) {
    while (v3_scratch_load(scratch, slot) < target) {
        __builtin_amdgcn_s_sleep(1);
    }
}

static __device__ __forceinline__ float v3_q8_word_dot_mix(uint32_t w, const float4 x, float acc) {
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

static __device__ __forceinline__ void v3_wave32_fwht_128_float4(float4 & v, const int lane) {
    const float a0 = v.x + v.y, a1 = v.x - v.y;
    const float a2 = v.z + v.w, a3 = v.z - v.w;
    float r[4] = { a0 + a2, a1 + a3, a0 - a2, a1 - a3 };
    #pragma unroll
    for (int m = 1; m <= 16; m <<= 1) {
        const bool hi = (lane & m) != 0;
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float cur   = r[j];
            const float other = __shfl_xor(cur, m, V3_WARP_SIZE);
            r[j] = hi ? (other - cur) : (cur + other);
        }
    }
    v = make_float4(r[0], r[1], r[2], r[3]);
}

static __device__ __forceinline__ int v3_pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

static __device__ __forceinline__ uint32_t v3_pack_u4x4_lo(int q0, int q1, int q2, int q3) {
    return ((uint32_t)(q0 & 0xF)) |
           ((uint32_t)(q1 & 0xF) << 8) |
           ((uint32_t)(q2 & 0xF) << 16) |
           ((uint32_t)(q3 & 0xF) << 24);
}

static __device__ __forceinline__ v3i4 v3_ld_nt_b128(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const v3i4 *>(p));
}

static __device__ __forceinline__ void v3_store_kv_quantized(
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

        ((int32_t *)(dst_qs + head_slot * HEAD_DIM))[lane] = v3_pack_i8x4(q0, q1, q2, q3);
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

        const uint32_t lo4 = v3_pack_u4x4_lo(q0, q1, q2, q3);
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

        const uint32_t lo4 = v3_pack_u4x4_lo(q0, q1, q2, q3);
        const uint32_t hi4 = __shfl_xor(lo4, 1, 32);
        if ((lane & 1) == 0) {
            ((uint32_t *)(dst_qs + head_slot * (HEAD_DIM / 2)))[lane >> 1] = lo4 | (hi4 << 4);
        }
        if (sub == 0) {
            dst_d[head_slot * 4 + grp] = d_h;
        }
    }
}

static __device__ __forceinline__ int v3_lay_x(int i) { return (i >> 3) * 9 + (i & 7); }

template <int LPR, int NB, int PD>
struct V3GemvRing {
    static constexpr int ITS = NB / LPR;
    static_assert(ITS % PD == 0, "ring depth must divide the blocks per lane");
    v3i4   r0[PD], r1[PD];
    __half rh[PD];

    __device__ __forceinline__ void load(int slot, const int8_t * wq, const __half * wd, int kb) {
        r0[slot] = v3_ld_nt_b128(wq + kb * 32);
        r1[slot] = v3_ld_nt_b128(wq + kb * 32 + 16);
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
            const v3i4 w0 = r0[slot], w1 = r1[slot];
            const float sc = __half2float(rh[slot]);
            if (it + PD < ITS) {
                load(slot, wq, wd, sub + (it + PD) * LPR + xo);
            } else if (has_next) {
                load(slot, nwq, nwd, sub + (it + PD - ITS) * LPR + xo);
            }
            const float4 * xb = s_x + (sub + it * LPR + xo) * 9;
            float p = 0.0f, q = 0.0f;
            p = v3_q8_word_dot_mix((uint32_t) w0.x, xb[0], p);
            q = v3_q8_word_dot_mix((uint32_t) w1.x, xb[4], q);
            p = v3_q8_word_dot_mix((uint32_t) w0.y, xb[1], p);
            q = v3_q8_word_dot_mix((uint32_t) w1.y, xb[5], q);
            p = v3_q8_word_dot_mix((uint32_t) w0.z, xb[2], p);
            q = v3_q8_word_dot_mix((uint32_t) w1.z, xb[6], q);
            p = v3_q8_word_dot_mix((uint32_t) w0.w, xb[3], p);
            q = v3_q8_word_dot_mix((uint32_t) w1.w, xb[7], q);
            acc = fmaf(p + q, sc, acc);
        }
        return acc;
    }
};

template <int NWARPS, int LPR, int NB, int PD>
static __device__ __forceinline__ void v3_q8_gemv_body(
        V3QkvSeg s0, V3QkvSeg s1, V3QkvSeg s2, float4 * s_x) {
    constexpr int NT  = NWARPS * 32;
    constexpr int K   = NB * 32;
    constexpr int RPW = 32 / LPR;
    constexpr int RPB = NWARPS * RPW;
    constexpr int NX4 = K / 4;
    constexpr int PER = (NX4 + NT - 1) / NT;

    const int b  = blockIdx.x;
    const int si = (b >= s0.blk_end) + (b >= s1.blk_end);
    const V3QkvSeg & sg = si == 0 ? s0 : (si == 1 ? s1 : s2);
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
    V3GemvRing<LPR, NB, PD> ring;
    {
        const int row = row0 + rofs;
        ring.prime(qs, dd, row < rows ? row : rows - 1, sub);
    }
    #pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * NT;
        if (idx < NX4) {
            s_x[v3_lay_x(idx)] = xv[i];
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
        #pragma unroll
        for (int offset = LPR >> 1; offset > 0; offset >>= 1) {
            acc += __shfl_xor(acc, offset, LPR);
        }
        if (sub == 0 && valid) {
            y[row] = acc;
        }
    }
}

template <int NWARPS, int LPR, int NB, int PD, int MINB, int NBLK>
__launch_bounds__(NWARPS * 32, MINB)
__attribute__((amdgpu_waves_per_eu((MINB * NWARPS) / 4, (MINB * NWARPS) / 4)))
static __global__ void qkv_qkvr_prep_fused_v3_kernel(
        V3QkvSeg s0, V3QkvSeg s1, V3QkvSeg s2,
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
    constexpr int N_RT_BLOCKS = 248;   // blocks 40..287
    constexpr int SHARD       = NBLK / 3;
    static_assert(NBLK % 3 == 0, "sharded barrier needs 3 equal shards");

    const int bid  = blockIdx.x;
    const int tid  = threadIdx.x;

    // s_x (GEMV x broadcast) and the rel-table staging are separated by the
    // grid barrier, so they share one LDS allocation.  That keeps the block at
    // 10.4 KiB and lets three 256-thread CTAs be co-resident (6 waves/SIMD);
    // with the two separate buffers (12.4 KiB) only five 128-thread CTAs fit.
    __shared__ union V3QkvSmem {
        float4 x[NB * 9];
        struct {
            alignas(16) float r[N_HEAD][N_REL];
            alignas(16) float b[8][N_REL + 1];
        } rel;
    } sm;

    v3_q8_gemv_body<NWARPS, LPR, NB, PD>(s0, s1, s2, sm.x);

    // ------------------------- sharded grid barrier -------------------------
    v3_rdna3_flush_stores();
    __syncthreads();
    if (tid == 0) {
        v3_rdna3_flush_stores();
        (void)v3_scratch_add(scratch, 392 + (bid % 3));
    }
    __syncthreads();
    if (tid == 0) {
        v3_spin_until(scratch, 392, SHARD);
        v3_spin_until(scratch, 393, SHARD);
        v3_spin_until(scratch, 394, SHARD);
    }
    __syncthreads();
    v3_rdna3_inv_l0_l1();

    // ------------------------------ qkvr prep ------------------------------
    if (bid < 24) {
        if (tid < 32) {
            const int h = bid;
            const int lane = tid;
            float4 v = ((const float4 *)(q_in + (size_t)h * HEAD_DIM))[lane];
            if (q_clip_w) {
                const float4 w = ((const float4 *)(q_clip_w + (size_t)h * HEAD_DIM))[lane];
                v.x *= w.x; v.y *= w.y; v.z *= w.z; v.w *= w.w;
            }
            float sumsq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            sumsq = __shfl_xor(sumsq, 16, 32);
            sumsq = __shfl_xor(sumsq,  8, 32);
            sumsq = __shfl_xor(sumsq,  4, 32);
            sumsq = __shfl_xor(sumsq,  2, 32);
            sumsq = __shfl_xor(sumsq,  1, 32);
            const float tv = (apply_tau && d_meta != nullptr) ? d_meta->tau : 1.0f;
            const float factor = fminf(1.0f, QK_CLIP / (sqrtf(sumsq) + RMS_EPS)) * tv * KQ_SCALE;
            v.x *= factor; v.y *= factor; v.z *= factor; v.w *= factor;
            v3_wave32_fwht_128_float4(v, lane);
            ((float4 *)(q_out + (size_t)h * HEAD_DIM))[lane] = v;
        }
    } else if (bid < 40) {
        if (tid < 32) {
            const int lane = tid;
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
            sumsq = __shfl_xor(sumsq, 16, 32);
            sumsq = __shfl_xor(sumsq,  8, 32);
            sumsq = __shfl_xor(sumsq,  4, 32);
            sumsq = __shfl_xor(sumsq,  2, 32);
            sumsq = __shfl_xor(sumsq,  1, 32);

            const float factor = is_v
                ? (rsqrtf(sumsq * (1.0f / (float)HEAD_DIM) + RMS_EPS) * KQ_SCALE)
                : (fminf(1.0f, QK_CLIP / (sqrtf(sumsq) + RMS_EPS)) * KQ_SCALE);
            v.x *= factor; v.y *= factor; v.z *= factor; v.w *= factor;

            v3_wave32_fwht_128_float4(v, lane);

            const int32_t pos = (d_meta != nullptr) ? d_meta->pos : 0;
            const int32_t idx = (d_meta != nullptr)
                ? (kv_slot.is_swa ? d_meta->kv_write_swa : d_meta->kv_write_global)
                : 0;

            v3_store_kv_quantized(v, kv_slot, is_v, h, idx, lane);
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

        // v2 ran 128 threads; the (tid >> 4) mapping defines exactly 8 x 16 =
        // 128 slots, so the extra warps must stay out of it.
        if (tid < 128) {
            #pragma unroll
            for (int step = 0; step < 3; ++step) {
                const int h = step * 8 + (tid >> 4);
                const int k = tid & 15;
                float rv = r_in[(size_t)h * N_REL + k];
                if (r_clip_w) {
                    rv *= r_clip_w[(size_t)h * N_REL + k];
                }
                float sumsq = rv * rv;
                #pragma unroll
                for (int off = 8; off >= 1; off >>= 1) {
                    sumsq += __shfl_xor(sumsq, off, 16);
                }
                const float factor = fminf(1.0f, REL_CLIP / (sqrtf(sumsq) + RMS_EPS)) * tv;
                sm.rel.r[h][k] = rv * factor;
            }
        }
        if (tid < d_cnt * N_REL) {
            const int dl = tid / N_REL;
            const int k  = tid % N_REL;
            sm.rel.b[dl][k] = rel_bias_w[(size_t)(d_lo + dl) * N_REL + k];
        }
        __syncthreads();

        if (tid < d_cnt * N_HEAD) {
            const int dl = tid / N_HEAD;
            const int h  = tid % N_HEAD;
            float dot = 0.0f;
            #pragma unroll
            for (int k = 0; k < N_REL; ++k) {
                dot = fmaf(sm.rel.b[dl][k], sm.rel.r[h][k], dot);
            }
            rel_table_out[(size_t)h * N_REL_EXTENT + d_lo + dl] = dot;
        }
    }

    // ------------------------- reset the barrier -------------------------
    v3_rdna3_flush_stores();
    __syncthreads();
    if (tid == 0) {
        const int prev = v3_scratch_add(scratch, 395);
        if (prev == NBLK - 1) {
            v3_scratch_store(scratch, 392, 0);
            v3_scratch_store(scratch, 393, 0);
            v3_scratch_store(scratch, 394, 0);
            v3_scratch_store(scratch, 395, 0);
        }
    }
}

// ============================================================================
// Host launchers
// ============================================================================

template <int GQA, int NWARPS>
static inline void v3_launch_swa(int kv_fmt, const dim3 & grid, hipStream_t stream,
        const float * q, KVCacheSlot kv_slot, const float * sinks_in, int sinks_stride,
        const float * rel_table, const float * qgate_in, int qgate_stride,
        float * out, float * tmp_o, float2 * tmp_meta, int n_splits,
        const StepMeta * d_meta, int32_t * fa_done) {
    const dim3 block(32, NWARPS, 1);
    switch (kv_fmt) {
        case KV_Q4_1:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_swa_v3_kernel<GQA, KV_Q4_1, NWARPS>),
                grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,
                qgate_in, qgate_stride, out, tmp_o, tmp_meta, n_splits, d_meta, fa_done);
            break;
        case KV_Q4_0:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_swa_v3_kernel<GQA, KV_Q4_0, NWARPS>),
                grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,
                qgate_in, qgate_stride, out, tmp_o, tmp_meta, n_splits, d_meta, fa_done);
            break;
        default:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_swa_v3_kernel<GQA, KV_Q8_0, NWARPS>),
                grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,
                qgate_in, qgate_stride, out, tmp_o, tmp_meta, n_splits, d_meta, fa_done);
            break;
    }
}

template <int GQA, int NWARPS>
static inline void v3_launch_global(int kv_fmt, const dim3 & grid, hipStream_t stream,
        const float * q, KVCacheSlot kv_slot, const float * sinks_in, int sinks_stride,
        const float * rel_table, const float * qgate_in, int qgate_stride,
        float * out, const int32_t * selected_pages, const int32_t * n_selected,
        float * tmp_o, float2 * tmp_meta, int n_splits,
        const StepMeta * d_meta, int32_t * fa_done) {
    const dim3 block(32, NWARPS, 1);
    switch (kv_fmt) {
        case KV_Q4_1:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_sparse_global_v3_kernel<GQA, KV_Q4_1, NWARPS>),
                grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,
                qgate_in, qgate_stride, out, selected_pages, n_selected,
                tmp_o, tmp_meta, n_splits, d_meta, fa_done);
            break;
        case KV_Q4_0:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_sparse_global_v3_kernel<GQA, KV_Q4_0, NWARPS>),
                grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,
                qgate_in, qgate_stride, out, selected_pages, n_selected,
                tmp_o, tmp_meta, n_splits, d_meta, fa_done);
            break;
        default:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_sparse_global_v3_kernel<GQA, KV_Q8_0, NWARPS>),
                grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,
                qgate_in, qgate_stride, out, selected_pages, n_selected,
                tmp_o, tmp_meta, n_splits, d_meta, fa_done);
            break;
    }
}

template <int RPB>
static inline int v3_partition_segments(V3QkvSeg * s, int nseg, int cap) {
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

static inline V3QkvSeg v3_make_seg(const SoaQ8Matrix & W, const float * x, float * y) {
    V3QkvSeg s;
    s.qs = W.qs;
    s.d = W.d;
    s.x = x;
    s.y = y;
    s.rows = W.rows;
    s.blk_end = 0;
    return s;
}

// Cached "are all 288 blocks co-resident?" probe.  hipGetDeviceProgramInfo /
// occupancy are host-side queries (no stream op), so this is capture-safe.
static inline bool v3_qkv_shape_ok() {
    static int cached = -1;
    if (cached < 0) {
        constexpr int LPR = 8;
        constexpr int NB  = N_EMBD / 32;
        int dev = 0;
        int ncu = 96;
        (void) hipGetDevice(&dev);
        (void) hipDeviceGetAttribute(&ncu, hipDeviceAttributeMultiprocessorCount, dev);
        int occ = 0;
        hipError_t e = hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &occ, qkv_qkvr_prep_fused_v3_kernel<FA_V3_NWARPS, LPR, NB, 3, 3, FA_V3_QKV_BLOCKS>,
            FA_V3_NWARPS * 32, 0);
        cached = (e == hipSuccess && occ * ncu >= FA_V3_QKV_BLOCKS) ? 1 : 0;
    }
    return cached == 1;
}

} // namespace

// ----------------------------------------------------------------------------
// 1. SWA dense decode FA v3
// ----------------------------------------------------------------------------
void launch_fa_swa_decode_v3(
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
        int32_t     * fa_done) {
    if (d_meta == nullptr || q == nullptr || rel_table == nullptr) {
        return;
    }
    int splits = n_splits;
    if (splits < 1) {
        splits = 1;
    }
    if (splits > FA_V3_MAX_SPLITS) {
        splits = FA_V3_MAX_SPLITS;
    }
    if (splits > 1 && (tmp_o == nullptr || tmp_meta == nullptr || fa_done == nullptr)) {
        splits = 1;
    }

    const dim3 grid(splits, N_HEAD_KV, 1);
    v3_launch_swa<GQA_RATIO, FA_V3_NWARPS>(kv_slot.kv_type, grid, stream,
        q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
        out, tmp_o, tmp_meta, splits, d_meta, fa_done);
}

// ----------------------------------------------------------------------------
// 2. Fused page score + Top-K v3
// ----------------------------------------------------------------------------
void launch_sparse_global_select_v3(
        const float * q,
        const SparseGlobalPageMeta & meta,
        SparseGlobalScratch & scratch,
        const StepMeta * d_meta,
        int             sparse_pages,
        int32_t       * select_done,
        hipStream_t     stream) {
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

    const dim3 grid(FA_V3_SELECT_CHUNKS, N_HEAD_KV, 1);
    const dim3 block(FA_V3_SEL_THREADS, 1, 1);
    hipLaunchKernelGGL(HIP_KERNEL_NAME(sparse_global_select_v3_kernel<GQA_RATIO>), grid, block, 0, stream,
        q, meta, d_meta, scratch.page_scores,
        scratch.selected_pages, scratch.n_selected, select_done, budget);
}

// ----------------------------------------------------------------------------
// 3. Sparse Global decode FA v3
// ----------------------------------------------------------------------------
void launch_fa_sparse_global_decode_v3(
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
        int32_t     * fa_done) {
    if (d_meta == nullptr || q == nullptr || rel_table == nullptr ||
        selected_pages == nullptr || n_selected_pages == nullptr) {
        return;
    }
    int splits = n_splits;
    if (splits < 1) {
        splits = 1;
    }
    if (splits > FA_V3_MAX_SPLITS) {
        splits = FA_V3_MAX_SPLITS;
    }
    if (splits > 1 && (tmp_o == nullptr || tmp_meta == nullptr || fa_done == nullptr)) {
        splits = 1;
    }

    const dim3 grid(splits, N_HEAD_KV, 1);
    v3_launch_global<GQA_RATIO, FA_V3_NWARPS>(kv_slot.kv_type, grid, stream,
        q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
        out, selected_pages, n_selected_pages, tmp_o, tmp_meta, splits, d_meta, fa_done);
}

// ----------------------------------------------------------------------------
// 4. QKV GEMV + qkvr_prep fused v3
// ----------------------------------------------------------------------------
void launch_qkv_qkvr_prep_fused_v3(
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
    if (W0.K != N_EMBD || (W1.rows > 0 && W1.K != N_EMBD) || (W2.rows > 0 && W2.K != N_EMBD) ||
        W0.rows > 12288 || W1.rows > 12288 || W2.rows > 12288 || scratch == nullptr) {
        return;
    }
    constexpr int LPR = 8;
    constexpr int NB  = N_EMBD / 32;
    constexpr int PD  = 3;

    constexpr int NW_HI = FA_V3_NWARPS;        // 8 warps -> 6 waves/SIMD
    constexpr int RPB_HI = NW_HI * (32 / LPR); // 32 rows/block

    constexpr int NW_LO = 4;                   // v2 shape -> 3 waves/SIMD
    constexpr int RPB_LO = NW_LO * (32 / LPR); // 16 rows/block

    V3QkvSeg s[3];
    int nseg = 0;
    if (W0.rows > 0) s[nseg++] = v3_make_seg(W0, x0, y0);
    if (W1.rows > 0) s[nseg++] = v3_make_seg(W1, x1, y1);
    if (W2.rows > 0) s[nseg++] = v3_make_seg(W2, x2, y2);
    if (nseg == 0) {
        return;
    }

    // `V3QkvSeg s[3]` is a POD array; when nseg < 3 the unused entries would hold
    // garbage pointers, and blocks past the last real segment's block range
    // (e.g. 8600 rows / 32 = 269 blocks at 288 launched) DO select them ->
    // illegal memory access (measured: HIP error 700 on the SWA shape).  Make
    // the spare segments harmless aliases of segment 0 with rows = 1: those
    // blocks stage x and join the grid barrier, but perform no row work.
    // Spare segments must be reachable-but-inert: the tail blocks (e.g. 269..287
    // of 288) are past the last real segment's rows, so `si` must resolve to a
    // segment whose blk_end covers the grid, otherwise `nblk` overflows, `step`
    // goes negative and `ring.run()` is fed a negative row -> illegal access
    // (measured: HIP error 700).  The last real segment therefore owns all
    // remaining blocks and the spare entries alias segment 0 with rows = 0.
    for (int i = nseg; i < 3; ++i) {
        s[i] = s[0];
        s[i].rows = 0;
        s[i].blk_end = 0x7FFFFFFF;
    }

    if (v3_qkv_shape_ok()) {
        (void) v3_partition_segments<RPB_HI>(s, nseg, FA_V3_QKV_BLOCKS);
        s[nseg - 1].blk_end = FA_V3_QKV_BLOCKS;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(qkv_qkvr_prep_fused_v3_kernel<NW_HI, LPR, NB, PD, 3, FA_V3_QKV_BLOCKS>),
            dim3(FA_V3_QKV_BLOCKS), dim3(NW_HI * 32), 0, stream,
            s[0], s[1], s[2],
            q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
            k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
            sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta, scratch);
    } else {
        (void) v3_partition_segments<RPB_LO>(s, nseg, FA_V3_QKV_BLOCKS);
        s[nseg - 1].blk_end = FA_V3_QKV_BLOCKS;
        hipLaunchKernelGGL(
            HIP_KERNEL_NAME(qkv_qkvr_prep_fused_v3_kernel<NW_LO, LPR, NB, PD, 3, FA_V3_QKV_BLOCKS>),
            dim3(FA_V3_QKV_BLOCKS), dim3(NW_LO * 32), 0, stream,
            s[0], s[1], s[2],
            q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
            k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
            sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta, scratch);
    }
}

} // namespace inkling
