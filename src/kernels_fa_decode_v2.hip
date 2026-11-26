#include "kernels_fa_decode_v2.h"

#include <cstdint>

namespace inkling {

namespace {

// ============================================================================
// Shared helpers (bit-exact copies of the v1 building blocks)
// ============================================================================

constexpr float V2_LOG2E = 1.44269504088896340736f;

// ext_vector_type (not HIP_vector_type) so `__builtin_nontemporal_load` accepts
// it and emits global_load_b128 ... slc dlc.
typedef uint32_t v2u4 __attribute__((ext_vector_type(4)));

// RDNA3 native 1-cycle v_exp_f32 (2^x).  The v1 *decode* FA kernels used
// `exp2f`, which the ISA expands into v_exp_f32 + v_ldexp_f32 + a
// v_div_scale_f32 denormal/overflow path (49 of each in
// inkling_fa_decode_kernel<3,true,1>).  Every use below is exp2(x) with
// x <= 0 (log2-domain softmax differences, negated sigmoid arguments), so the
// clamped/denormal path is unreachable; Step 5 already made exactly this
// substitution in the combine kernel and the prefill FA with identical PPL.
static __device__ __forceinline__ float v2_exp2f(float x) {
    return __builtin_amdgcn_exp2f(x);
}

static __device__ __forceinline__ __half2 v2_h2_of(uint32_t u) {
    return __builtin_bit_cast(__half2, u);
}

// int8 x4 -> two packed __half2 through v_perm_b32 + v_pk_sub_f16 (no cvt).
static __device__ __forceinline__ void v2_i8x4_to_h2x2(uint32_t w, __half2 & h01, __half2 & h23) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    const uint32_t u = w ^ 0x80808080u;
    h01 = __hsub2(v2_h2_of(__builtin_amdgcn_perm(0x64646464u, u, 0x04010400u)), v2_h2_of(0x64806480u));
    h23 = __hsub2(v2_h2_of(__builtin_amdgcn_perm(0x64646464u, u, 0x04030402u)), v2_h2_of(0x64806480u));
#else
    h01 = __floats2half2_rn((float)(int8_t)( w        & 0xFF),
                            (float)(int8_t)((w >>  8) & 0xFF));
    h23 = __floats2half2_rn((float)(int8_t)((w >> 16) & 0xFF),
                            (float)(int8_t)((w >> 24) & 0xFF));
#endif
}

// 4 unsigned nibbles -> two packed __half2 (q4_0 additionally subtracts 8).
template <bool SIGNED_Q4_0>
static __device__ __forceinline__ void v2_u4x4_to_h2x2(uint32_t nibs, __half2 & h01, __half2 & h23) {
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    constexpr uint32_t BIAS_H2 = SIGNED_Q4_0 ? 0x64086408u : 0x64006400u;
    h01 = __hsub2(v2_h2_of(__builtin_amdgcn_perm(0x64646464u, nibs, 0x04010400u)), v2_h2_of(BIAS_H2));
    h23 = __hsub2(v2_h2_of(__builtin_amdgcn_perm(0x64646464u, nibs, 0x04030402u)), v2_h2_of(BIAS_H2));
#else
    const float bias = SIGNED_Q4_0 ? 8.0f : 0.0f;
    h01 = __floats2half2_rn((float)( nibs        & 0xF) - bias,
                            (float)((nibs >>  8) & 0xF) - bias);
    h23 = __floats2half2_rn((float)((nibs >> 16) & 0xF) - bias,
                            (float)((nibs >> 24) & 0xF) - bias);
#endif
}

// Wave32 orthogonal FWHT-128 on the 4 output dims owned by each lane.
static __device__ __forceinline__ void v2_fwht128_4(
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

// RDNA3 GL1 coherence hardening for the self-resetting arrival counters
// (identical sequence to every other cooperative barrier in this engine).
static __device__ __forceinline__ void v2_gl_fence_release() {
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    // gfx12: no equivalent instruction - on gfx1201 __scoped_atomic_thread_fence(RELEASE, AGENT)
    // above already lowers to s_waitcnt + `global_wb scope:SCOPE_DEV` (ISA-checked with
    // llvm-mc/hipcc -S), so gfx1200/gfx1201 need nothing extra here.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#endif
}

static __device__ __forceinline__ void v2_gl_invalidate() {
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    // gfx12: no equivalent instruction - scope-qualified acquire atomics on gfx1201 emit
    // `global_inv scope:SCOPE_DEV` themselves (ISA-checked), which replaces the RDNA3
    // L0/L1 invalidation below and needs no hand-written cache asm.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("buffer_gl1_inv\n\tbuffer_gl0_inv" ::: "memory");
#endif
}

// ============================================================================
// Software-pipelined KV fetch: one register set = the full 8-key wave tile
// ============================================================================
//
// Lane mapping (identical to v1, so the arithmetic below is unchanged):
//   lane = k_slot * 4 + blk,  k_slot = lane >> 2 in [0,8), blk = lane & 3
//   K:  this lane owns 32 dims [blk*32, blk*32+32) of key k_slot
//       q4_*: 64 B/token, 4 lanes x 16 B -> ONE global_load_b128 per lane and
//             32 lanes cover 8 consecutive tokens = 512 contiguous bytes
//       q8_0: 128 B/token -> two global_load_b128 per lane
//   V:  every lane accumulates output dims [lane*4, +4) over all 8 keys
//       q4_*: 2 lanes per token, 2 tokens per 128-bit-grouped load (4 loads)
//       q8_0: 1 lane per token, 4 dims (8 x 32-bit coalesced loads)
// The scale of the lane's own 32-dim block is fetched with the tile, so its
// __half -> float conversion is also covered by the prefetch distance.
// ----------------------------------------------------------------------------
template <int KV_FMT> struct KvGrp {};

template <> struct KvGrp<KV_Q8_0> {
    v2u4     k_lo;   // int8 dims [blk*32 +  0, +16)
    v2u4     k_hi;   // int8 dims [blk*32 + 16, +16)
    float    k_d;    // K scale of this lane's 32-dim block
    uint32_t vq[8];  // 4 int8 V dims of key kk
    float    v_d;    // V scale of this lane's 32-dim block
};

template <> struct KvGrp<KV_Q4_1> {
    v2u4     k_qs;   // 32 nibbles = dims [blk*32, +32)
    float    k_d;    // zero-pointed scale d
    float    k_m;    // zero point m
    uint32_t vq[4];  // 2 keys per word (low/high nibble halves)
    uint32_t v_dm;   // packed __half2 (d, m) of this lane's V block
};

template <> struct KvGrp<KV_Q4_0> {
    v2u4     k_qs;
    float    k_d;
    uint32_t vq[4];
    float    v_d;
};

// SWA variant: `ring0` is a ring slot index; the whole tile is 8 consecutive
// ring slots, so only the tile that straddles the ring wrap pays a per-lane
// slot fix-up (one compare + one select).
template <int KV_FMT>
static __device__ __forceinline__ void v2_load_kv_group(
        KvGrp<KV_FMT> & g,
        const KVCacheSlot & kv,
        size_t head_base,   // h_kv * cap, in token units
        int    cap,
        int    ring0,
        int    lane, int blk) {
    const int kslot = lane >> 2;
    int jk = ring0 + kslot;
    jk -= (jk >= cap) ? cap : 0;
    const size_t ks = head_base + (size_t)jk;

    if constexpr (KV_FMT == KV_Q8_0) {
        const int8_t * kp = kv.k_qs + ks * HEAD_DIM + (size_t)blk * 32;
        g.k_lo = *(const v2u4 *)(kp);
        g.k_hi = *(const v2u4 *)(kp + 16);
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
        g.k_qs = *(const v2u4 *)kp;
        if constexpr (KV_FMT == KV_Q4_1) {
            const __half2 dm = __ldg(&((const __half2 *)kv.k_d)[ks * 4 + (size_t)blk]);
            g.k_d  = __low2float(dm);
            g.k_m  = __high2float(dm);
            g.v_dm = __builtin_bit_cast(uint32_t,
                        __ldg(&((const __half2 *)kv.v_d)[ks * 4 + (size_t)blk]));
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

// Global variant: `base` is an absolute position.  Global slots store
// position == slot index, so there is no ring, no wrap and no k_pos[] probe.
// `n_kv` only guards the address: a partially filled last page can point a few
// slots past the live KV, so every position is clamped into [0, n_kv).  The
// clamp is a no-op for a fully valid tile and the surplus keys are masked to
// -INF by the compute, so it never changes the result.
template <int KV_FMT>
static __device__ __forceinline__ void v2_load_kv_group_abs(
        KvGrp<KV_FMT> & g,
        const KVCacheSlot & kv,
        size_t head_base,
        int    base,
        int    n_kv,
        int    lane, int blk) {
    const int jmax = (n_kv > 0) ? (n_kv - 1) : 0;
    const int jk = ((base + (lane >> 2)) < jmax) ? (base + (lane >> 2)) : jmax;
    const size_t ks = head_base + (size_t)jk;

    if constexpr (KV_FMT == KV_Q8_0) {
        const int8_t * kp = kv.k_qs + ks * HEAD_DIM + (size_t)blk * 32;
        g.k_lo = __builtin_nontemporal_load((const v2u4 *)(kp));
        g.k_hi = __builtin_nontemporal_load((const v2u4 *)(kp + 16));
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
        g.k_qs = __builtin_nontemporal_load((const v2u4 *)kp);
        if constexpr (KV_FMT == KV_Q4_1) {
            const __half2 dm = __ldg(&((const __half2 *)kv.k_d)[ks * 4 + (size_t)blk]);
            g.k_d  = __low2float(dm);
            g.k_m  = __high2float(dm);
            g.v_dm = __builtin_bit_cast(uint32_t,
                        __ldg(&((const __half2 *)kv.v_d)[ks * 4 + (size_t)blk]));
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
// LDS state: lossless FP32 Q staging + 4-warp reduction area
// ============================================================================
// Q keeps the v1 bank-conflict-free padded layout (144 B stride per 32-dim
// block), so the 4 lanes that own one key hit 4 different LDS banks.
template <int GQA, int NWARPS>
union V2Smem {
    struct {
        float4 q4[GQA][36];
        float  q_sum[GQA][4];
    } q;
    struct {
        float o[NWARPS][GQA][HEAD_DIM];
        float m[NWARPS][GQA];
        float l[NWARPS][GQA];
    } red;
};

// Stage Q[0..GQA-1] pre-scaled by KQ_SCALE * LOG2E (warp < GQA loads one head
// each).  For q4_1 it also builds q_sum[g][blk], the per-32-dim block sum that
// folds the K zero point into a single FMA per (head, key).
template <int GQA, int KV_FMT>
static __device__ __forceinline__ void v2_stage_q(
        V2Smem<GQA, 4> & smem, const float * __restrict__ q, int h_base, int lane, int warp) {
    constexpr float Q_SCALE_LOG2 = KQ_SCALE * V2_LOG2E;
    if (warp < GQA) {
        const float4 fv  = ((const float4 *)(q + (size_t)(h_base + warp) * HEAD_DIM))[lane];
        const int q_b = lane >> 3; // 0..3
        const int q_i = lane & 7;  // 0..7
        const float4 qs4 = make_float4(fv.x * Q_SCALE_LOG2, fv.y * Q_SCALE_LOG2,
                                       fv.z * Q_SCALE_LOG2, fv.w * Q_SCALE_LOG2);
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
}

// ============================================================================
// One 8-key wave tile of the online-softmax attention body
// ============================================================================
// FULL  == true : all 8 keys of the tile are valid (no masking at all).
// FULL  == false: keys whose logical index (SWA) / absolute position (Global)
//                 reaches `n_valid` contribute nothing — their logit becomes
//                 -INF, so exp2() yields exactly 0 and l/softmax are untouched.
// SWA   == true : RelBias always applies (window <= N_REL_EXTENT), so the
//                 distance test compiles away.
// `dist0` is the RelBias distance of key k_slot == 0; the 8 keys of a tile are
// consecutive so their distances are dist0 - k_slot.
// ----------------------------------------------------------------------------
template <int GQA, int KV_FMT, bool FULL, bool SWA>
static __device__ __forceinline__ void v2_fa_group(
        V2Smem<GQA, 4> & smem,
        const KvGrp<KV_FMT> & g,
        const float * __restrict__ rel_base,   // &rel_table[tok][h_base][0]
        int   dist0,
        int   q4_off, int blk, int lane, int v_blk, int k_slot,
        bool  kv_ok,                           // ignored when FULL
        float o_reg[GQA][4], float m_reg[GQA], float l_reg[GQA], float ob_reg[GQA]) {
    // ---- 1. K . Q^T for the 3 GQA heads that share this KV head ------------
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
            v2_i8x4_to_h2x2(ku[i], kh01, kh23);
            #pragma unroll
            for (int gq = 0; gq < GQA; ++gq) {
                const float4 qv = smem.q.q4[gq][q4_off + i];
                sc_k[gq] = fmaf(__low2float(kh01),  qv.x, sc_k[gq]);
                sc_k[gq] = fmaf(__high2float(kh01), qv.y, sc_k[gq]);
                sc_k[gq] = fmaf(__low2float(kh23),  qv.z, sc_k[gq]);
                sc_k[gq] = fmaf(__high2float(kh23), qv.w, sc_k[gq]);
            }
        }
    } else {
        const uint32_t ku[4] = { g.k_qs.x, g.k_qs.y, g.k_qs.z, g.k_qs.w };
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            __half2 kh01, kh23;
            v2_u4x4_to_h2x2<KV_FMT == KV_Q4_0>(ku[i] & 0x0F0F0F0Fu, kh01, kh23);
            #pragma unroll
            for (int gq = 0; gq < GQA; ++gq) {
                const float4 qv0 = smem.q.q4[gq][q4_off + 2 * i + 0];
                sc_k[gq] = fmaf(__low2float(kh01),  qv0.x, sc_k[gq]);
                sc_k[gq] = fmaf(__high2float(kh01), qv0.y, sc_k[gq]);
                sc_k[gq] = fmaf(__low2float(kh23),  qv0.z, sc_k[gq]);
                sc_k[gq] = fmaf(__high2float(kh23), qv0.w, sc_k[gq]);
            }
            v2_u4x4_to_h2x2<KV_FMT == KV_Q4_0>((ku[i] >> 4) & 0x0F0F0F0Fu, kh01, kh23);
            #pragma unroll
            for (int gq = 0; gq < GQA; ++gq) {
                const float4 qv1 = smem.q.q4[gq][q4_off + 2 * i + 1];
                sc_k[gq] = fmaf(__low2float(kh01),  qv1.x, sc_k[gq]);
                sc_k[gq] = fmaf(__high2float(kh01), qv1.y, sc_k[gq]);
                sc_k[gq] = fmaf(__low2float(kh23),  qv1.z, sc_k[gq]);
                sc_k[gq] = fmaf(__high2float(kh23), qv1.w, sc_k[gq]);
            }
        }
    }

    // ---- 2. dequant scale (+ q4_1 zero point) and 4-lane butterfly ---------
    // q4_1: sum_d q_d * (d*q4_d + m) = d * (q . q4) + m * sum_d q_d, so the zero
    // point costs one FMA per (head, key) instead of one per element.
    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        float p;
        if constexpr (KV_FMT == KV_Q4_1) {
            p = fmaf(sc_k[gq], g.k_d, g.k_m * smem.q.q_sum[gq][blk]);
        } else {
            p = sc_k[gq] * g.k_d;
        }
        p += __shfl_xor(p, 1, 32);
        p += __shfl_xor(p, 2, 32);
        if constexpr (!FULL) {
            p = kv_ok ? p : -INFINITY;
        }
        sc_k[gq] = p;
    }

    // ---- 3. RelBias (leader lane only) ------------------------------------
    if (blk == 0) {
        const int dist = dist0 - k_slot;
        bool do_rel = true;
        if constexpr (!FULL) {
            do_rel = kv_ok;
        }
        if constexpr (SWA) {
            if (do_rel) {   // window <= N_REL_EXTENT => dist is always in range
                #pragma unroll
                for (int gq = 0; gq < GQA; ++gq) {
                    sc_k[gq] += rel_base[(int64_t)gq * N_REL_EXTENT + dist] * V2_LOG2E;
                }
            }
        } else {
            if (do_rel && dist >= 0 && dist < N_REL_EXTENT) {
                #pragma unroll
                for (int gq = 0; gq < GQA; ++gq) {
                    sc_k[gq] += rel_base[(int64_t)gq * N_REL_EXTENT + dist] * V2_LOG2E;
                }
            }
        }
    }

    // ---- 4. tile-8 log2-domain online softmax -----------------------------
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
                const float scale_old = v2_exp2f(m_old - m_new);
                o_reg[gq][0] *= scale_old;
                o_reg[gq][1] *= scale_old;
                o_reg[gq][2] *= scale_old;
                o_reg[gq][3] *= scale_old;
                ob_reg[gq]   *= scale_old;
                l_reg[gq]    *= scale_old;
            }
            m_reg[gq] = m_new;
        }

        const float wk = (blk == 0) ? v2_exp2f(sc_k[gq] - m_new) : 0.0f;
        w_k[gq] = wk;

        float lt = wk;
        lt += __shfl_xor(lt,  4, 32);
        lt += __shfl_xor(lt,  8, 32);
        lt += __shfl_xor(lt, 16, 32);
        l_reg[gq] += lt;
    }

    // ---- 5. P * V over all 8 keys, 4 output dims per lane -----------------
    if constexpr (KV_FMT == KV_Q8_0) {
        #pragma unroll
        for (int kk = 0; kk < 8; ++kk) {
            const float d_v = __shfl(g.v_d, kk * 4 + v_blk, 32);
            __half2 vh01, vh23;
            v2_i8x4_to_h2x2(g.vq[kk], vh01, vh23);
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
        #pragma unroll
        for (int p = 0; p < 4; ++p) {
            const uint32_t w_me = g.vq[p];
            const uint32_t w_ot = __shfl_xor(w_me, 1, 32);
            #pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int kk = 2 * p + s;
                const uint32_t w_cur = (((lane ^ s) & 1) ? w_ot : w_me);
                const uint32_t nibs  = ((lane & 1) ? (w_cur >> 4) : w_cur) & 0x0F0F0F0Fu;
                __half2 vh01, vh23;
                v2_u4x4_to_h2x2<KV_FMT == KV_Q4_0>(nibs, vh01, vh23);
                if constexpr (KV_FMT == KV_Q4_1) {
                    const __half2 dm_v = v2_h2_of(__shfl(g.v_dm, kk * 4 + v_blk, 32));
                    const float d_v = __low2float(dm_v);
                    const float m_v = __high2float(dm_v);
                    #pragma unroll
                    for (int gq = 0; gq < GQA; ++gq) {
                        const float wg   = __shfl(w_k[gq], kk * 4, 32);
                        const float w_dv = wg * d_v;
                        // One scalar accumulator per (head, lane) instead of 4
                        // extra FADDs per key for the zero point.
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
// Split epilogue + in-kernel self-resetting combine (shared by both FA kernels)
// ============================================================================
// 1. reduce the 4 warps of the CTA in LDS;
// 2. n_splits == 1: fold the sink, normalise, FWHT-128, sigmoid(qgate), store;
//    n_splits >  1: store (m, l) + partial O, then the LAST arriving CTA of
//                   this KV head (self-resetting arrival counter -> no extra
//                   kernel, no extra HIP-graph node) merges all splits.
// ----------------------------------------------------------------------------
template <int GQA, int NWARPS>
static __device__ __forceinline__ void v2_fa_epilogue(
        V2Smem<GQA, NWARPS> & smem, int & s_is_last,
        const float o_reg[GQA][4], const float m_reg[GQA], const float l_reg[GQA],
        const float ob_reg[GQA],
        int lane, int warp, int h_base, int grp_id, int split, int n_splits, int tok,
        const float * __restrict__ sinks_in, int sinks_stride,
        const float * __restrict__ qgate_in, int qgate_stride,
        float * __restrict__ out, float * __restrict__ tmp_o, float2 * __restrict__ tmp_meta,
        int32_t * __restrict__ fa_done) {
    constexpr int D = HEAD_DIM;

    __syncthreads();  // every warp finished reading smem.q

    #pragma unroll
    for (int gq = 0; gq < GQA; ++gq) {
        if (lane == 0) {
            smem.red.m[warp][gq] = m_reg[gq];
            smem.red.l[warp][gq] = l_reg[gq];
        }
        *(float4 *)(&smem.red.o[warp][gq][lane * 4]) =
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
                m_blk = fmaxf(m_blk, smem.red.m[w][gq]);
            }

            float l_blk = 0.0f;
            float oa0 = 0.0f, oa1 = 0.0f, oa2 = 0.0f, oa3 = 0.0f;
            if (m_blk > -INFINITY) {
                #pragma unroll
                for (int w = 0; w < NWARPS; ++w) {
                    const float mw = smem.red.m[w][gq];
                    if (mw > -INFINITY) {
                        const float sw = v2_exp2f(mw - m_blk);
                        l_blk += smem.red.l[w][gq] * sw;
                        const float4 ov = *(const float4 *)(&smem.red.o[w][gq][lane * 4]);
                        oa0 = fmaf(ov.x, sw, oa0);
                        oa1 = fmaf(ov.y, sw, oa1);
                        oa2 = fmaf(ov.z, sw, oa2);
                        oa3 = fmaf(ov.w, sw, oa3);
                    }
                }
            }

            const int h = h_base + gq;
            if (n_splits == 1) {
                const float sink_val_log2 = sinks_in[(size_t)tok * sinks_stride + h] * V2_LOG2E;
                const float m_final  = fmaxf(m_blk, sink_val_log2);
                const float scale_o  = (m_blk > -INFINITY) ? v2_exp2f(m_blk - m_final) : 0.0f;
                const float l_final  = l_blk * scale_o + v2_exp2f(sink_val_log2 - m_final);
                const float norm     = (scale_o / l_final) * KQ_SCALE;

                float v0 = oa0 * norm;
                float v1 = oa1 * norm;
                float v2 = oa2 * norm;
                float v3 = oa3 * norm;
                v2_fwht128_4(v0, v1, v2, v3, lane);

                if (qgate_in != nullptr) {
                    const float4 gv = *(const float4 *)(qgate_in + (size_t)tok * qgate_stride
                                                        + (size_t)h * D + (size_t)lane * 4);
                    v0 *= 1.0f / (1.0f + v2_exp2f(-gv.x * V2_LOG2E));
                    v1 *= 1.0f / (1.0f + v2_exp2f(-gv.y * V2_LOG2E));
                    v2 *= 1.0f / (1.0f + v2_exp2f(-gv.z * V2_LOG2E));
                    v3 *= 1.0f / (1.0f + v2_exp2f(-gv.w * V2_LOG2E));
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

    v2_gl_fence_release();
    __syncthreads();
    if (warp == 0 && lane == 0) {
        v2_gl_fence_release();
        const int prev = __hip_atomic_fetch_add(&fa_done[grp_id], 1,
                                                __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_is_last = (prev == n_splits - 1) ? 1 : 0;
    }
    __syncthreads();

    if (!s_is_last) {
        return;
    }

    v2_gl_invalidate();
    if (warp == 0 && lane == 0) {
        __hip_atomic_store(&fa_done[grp_id], 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
    if (warp < GQA) {
        const int h = h_base + warp;
        const int64_t base_idx = ((int64_t)tok * N_HEAD + h) * n_splits;
        const float sink_val_log2 = sinks_in[(size_t)tok * sinks_stride + h] * V2_LOG2E;

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
            ? v2_exp2f(ml_lane.x - m_final) : 0.0f;
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
        const float l_final = l_sum + v2_exp2f(sink_val_log2 - m_final);

        const float inv_l = (1.0f / l_final) * KQ_SCALE;
        float v0 = o0 * inv_l;
        float v1 = o1 * inv_l;
        float v2 = o2 * inv_l;
        float v3 = o3 * inv_l;
        v2_fwht128_4(v0, v1, v2, v3, lane);

        if (qgate_in != nullptr) {
            const float4 gv = *(const float4 *)(qgate_in + (size_t)tok * qgate_stride
                                                + (size_t)h * HEAD_DIM + (size_t)lane * 4);
            v0 *= 1.0f / (1.0f + v2_exp2f(-gv.x * V2_LOG2E));
            v1 *= 1.0f / (1.0f + v2_exp2f(-gv.y * V2_LOG2E));
            v2 *= 1.0f / (1.0f + v2_exp2f(-gv.z * V2_LOG2E));
            v3 *= 1.0f / (1.0f + v2_exp2f(-gv.w * V2_LOG2E));
        }
        *(float4 *)(out + (size_t)h * HEAD_DIM + (size_t)lane * 4) = make_float4(v0, v1, v2, v3);
    }
}

// ============================================================================
// Kernel 1: SWA dense decode FA v2 (window = SWA_WINDOW, 40 layers)
// ============================================================================
// Enumerates exactly the min(pos+1, SWA_WINDOW) keys that can pass the window
// test instead of sweeping the whole 1536-slot ring: no k_pos[] probe, no
// per-key ballot, no partial-tile fallback for the extra third of the ring.
// Each of the 4 warps takes one contiguous run of 8-key tiles (sequential DRAM
// per warp) and the fetch of tile t+1 is issued before tile t is consumed.
// ----------------------------------------------------------------------------
template <int GQA, int KV_FMT>
__launch_bounds__(128)
static __global__ void inkling_fa_decode_swa_v2_kernel(
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
    constexpr int NWARPS = 4;
    constexpr int TOK    = 0;

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
    // Window = [qp - (W-1), qp], W = min(n_kv, SWA_WINDOW).  swa_cap >
    // SWA_WINDOW, so the window is one contiguous ring run of W slots that
    // starts at (qp - W + 1) % cap -> no validity test at all.
    const int n_valid = (n_kv < SWA_WINDOW) ? n_kv : SWA_WINDOW;
    int r_start = (qp - n_valid + 1) % cap;
    if (r_start < 0) {
        r_start += cap;
    }

    const int total_groups = (n_valid + 7) >> 3;
    // Each split owns one contiguous slice of the window's 8-key tiles and the
    // split's 4 warps divide that slice evenly; that keeps every CTA busy (no
    // idle tail CTA) and every warp reading a sequential KV run.
    const int tiles_per_split = (total_groups + n_splits - 1) / n_splits;
    const int ts0 = (split * tiles_per_split < total_groups) ? (split * tiles_per_split) : total_groups;
    const int ts1 = (ts0 + tiles_per_split < total_groups) ? (ts0 + tiles_per_split) : total_groups;
    const int nsplit_groups = ts1 - ts0;
    const int chunk = (nsplit_groups + NWARPS - 1) / NWARPS;
    const int g0 = ts0 + warp * chunk;
    const int g1 = (g0 + chunk < ts1) ? (g0 + chunk) : ts1;

    __shared__ V2Smem<GQA, NWARPS> smem;
    __shared__ int s_is_last;

    // CTA-uniform: this split owns at least one tile.  Splits without keys must
    // still reach the epilogue so they bump the arrival counter of this KV head
    // (otherwise the in-kernel combine of a 24-split head would never fire).
    const bool split_has_keys = (ts0 < ts1);
    if (split_has_keys) {
        v2_stage_q<GQA, KV_FMT>(smem, q, h_base, lane, warp);
    }
    __syncthreads();

    const int q4_off = blk * 9;
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

    // Ring slot of the first key of tile `gi`.  Every address stays inside this
    // head's ring rows (r_start + n_valid + 7 <= cap + 1031 < 2*cap, and the
    // per-lane wrap subtract below brings it back into [0, cap)), so a tile
    // that runs past the window simply reads stale-but-mapped slots whose
    // surplus keys are masked to -INF.
    auto tile_ring = [&](int gi) {
        return r_start + gi * 8;
    };
    auto tile_full = [&](int gi) {
        return (gi * 8 + 8 <= n_valid);
    };

    // Warp-uniform guard: shuffles inside stay converged, and the loop contains
    // no barrier.
    if (g0 < g1) {
        KvGrp<KV_FMT> a, b;
        v2_load_kv_group<KV_FMT>(a, kv_slot, head_base, cap, tile_ring(g0), lane, blk);

        for (int gi = g0; gi < g1; ++gi) {
            // Issue the next tile's 512-byte KV streams before consuming this one.
            const int gn = (gi + 1 < g1) ? (gi + 1) : gi;
            v2_load_kv_group<KV_FMT>(b, kv_slot, head_base, cap, tile_ring(gn), lane, blk);

            const int dist0 = (n_valid - 1) - gi * 8;
            if (tile_full(gi)) {
                v2_fa_group<GQA, KV_FMT, true, true>(
                    smem, a, rel_base, dist0, q4_off, blk, lane, v_blk, k_slot,
                    true, o_reg, m_reg, l_reg, ob_reg);
            } else {
                v2_fa_group<GQA, KV_FMT, false, true>(
                    smem, a, rel_base, dist0, q4_off, blk, lane, v_blk, k_slot,
                    (gi * 8 + k_slot) < n_valid, o_reg, m_reg, l_reg, ob_reg);
            }
            a = b;
        }
    }

    v2_fa_epilogue<GQA, NWARPS>(smem, s_is_last, o_reg, m_reg, l_reg, ob_reg,
                                lane, warp, h_base, grp_id, split, n_splits, TOK,
                                sinks_in, sinks_stride, qgate_in, qgate_stride,
                                out, tmp_o, tmp_meta, fa_done);
}

// ============================================================================
// Kernel 2: sparse Global decode FA v2 (selected 64-position pages, 8 layers)
// ============================================================================
// Same skeleton.  KV indices are absolute positions (Global slots store
// position == slot index, so `k_pos[j] == j` and the only test is
// `j < kv_len_global`).  Pages are 64-aligned, so a tile never straddles a page
// and the prefetch carries across page boundaries.
// ----------------------------------------------------------------------------
template <int GQA, int KV_FMT>
__launch_bounds__(128)
static __global__ void inkling_fa_decode_sparse_global_v2_kernel(
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
    constexpr int NWARPS = 4;
    constexpr int TOK    = 0;
    constexpr int GPG    = SPARSE_PAGE_SIZE / 8;   // 8 tiles of 8 keys per page

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

    __shared__ V2Smem<GQA, NWARPS> smem;
    __shared__ int s_is_last;

    const int total_groups = (pi1 - pi0) * GPG;

    // A split without pages must still reach the epilogue to bump the arrival
    // counter of this KV head (see the note in the SWA kernel).
    if (total_groups > 0) {
        v2_stage_q<GQA, KV_FMT>(smem, q, h_base, lane, warp);
    }
    __syncthreads();

    const int q4_off = blk * 9;
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

    const int chunk = (total_groups + NWARPS - 1) / NWARPS;
    const int g0    = warp * chunk;
    const int g1    = (g0 + chunk < total_groups) ? (g0 + chunk) : total_groups;

    // Absolute position of the split-local tile `gg` (warp-uniform).
    auto tile_pos = [&](int gg) {
        const int page = pi0 + (gg >> 3);
        return sel[page] * SPARSE_PAGE_SIZE + ((gg & (GPG - 1)) * 8);
    };

    if (g0 < g1) {
        KvGrp<KV_FMT> a, b;
        v2_load_kv_group_abs<KV_FMT>(a, kv_slot, head_base, tile_pos(g0), n_kv, lane, blk);

        for (int gg = g0; gg < g1; ++gg) {
            const int gn = (gg + 1 < g1) ? (gg + 1) : gg;
            v2_load_kv_group_abs<KV_FMT>(b, kv_slot, head_base, tile_pos(gn), n_kv, lane, blk);

            const int s_abs = tile_pos(gg);
            const int dist0 = qp - s_abs;
            if (s_abs + 8 <= n_kv) {
                v2_fa_group<GQA, KV_FMT, true, false>(
                    smem, a, rel_base, dist0, q4_off, blk, lane, v_blk, k_slot,
                    true, o_reg, m_reg, l_reg, ob_reg);
            } else if (s_abs < n_kv) {
                // Partially valid tile (only in the last selected page).  The
                // guard is warp-uniform so every shuffle below stays converged;
                // key 0 of the tile is always valid here, hence m stays finite.
                v2_fa_group<GQA, KV_FMT, false, false>(
                    smem, a, rel_base, dist0, q4_off, blk, lane, v_blk, k_slot,
                    (s_abs + k_slot) < n_kv, o_reg, m_reg, l_reg, ob_reg);
            }
            a = b;
        }
    }

    v2_fa_epilogue<GQA, NWARPS>(smem, s_is_last, o_reg, m_reg, l_reg, ob_reg,
                                lane, warp, h_base, grp_id, split, n_splits, TOK,
                                sinks_in, sinks_stride, qgate_in, qgate_stride,
                                out, tmp_o, tmp_meta, fa_done);
}

// ============================================================================
// Kernel 3: fused page score + Top-K (Global layers, decode)
// ============================================================================
// Stage 1 (v1 22.16 us/layer): 8 lanes own one page (16 dims == one 128-bit
// k_min_q and one 128-bit k_max_q per lane), so ONE Wave32 scores FOUR pages
// with two global_load_b128 ... slc dlc and only 3 butterfly shuffles per head
// per 4 pages.  v1 used 4 B per lane (128 B per wave-instruction), serialized
// three s_waitcnt vmcnt(N) per page, and reduced with 5 shuffles per page.
// Pages are scored in warp-contiguous runs (sequential streaming) with a
// one-tile lookahead.
//
// Stage 2 (v1 13.47 us/layer in 8 single-CTA launches, 8 of 96 CUs busy): the
// last arriving chunk of each KV head runs a 4-pass 8-bit radix select with the
// 16 candidate keys per thread held in VGPRs — the scores are read from global
// exactly once.  Ranking is exact and deterministic (ties broken in scan order).
// ----------------------------------------------------------------------------
constexpr int V2_SEL_THREADS = 256;
constexpr int V2_SEL_SLOTS   = SPARSE_MAX_PAGES / V2_SEL_THREADS;  // 16
constexpr int V2_SEL_HIST    = 256;                                // 8-bit radix

// Monotone float -> uint32 key: ascending float order == ascending key order.
static __device__ __forceinline__ uint32_t v2_score_key(float v) {
    const uint32_t b = __builtin_bit_cast(uint32_t, v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__launch_bounds__(V2_SEL_THREADS)
static __global__ void sparse_global_select_v2_kernel(
        const float * __restrict__ q,                 // [N_HEAD, HEAD_DIM]
        SparseGlobalPageMeta meta,
        const StepMeta * __restrict__ d_meta,
        float * __restrict__ page_scores,             // [N_HEAD_KV][max_pages]
        int32_t * __restrict__ selected_pages,        // [N_HEAD_KV][SPARSE_MAX_SELECTED]
        int32_t * __restrict__ n_selected,            // [N_HEAD_KV]
        int32_t * __restrict__ done,                  // [N_HEAD_KV] zeroed, self-resetting
        int page_budget) {                            // clamped sparse_pages
    const int h    = (int)blockIdx.y;
    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;                        // 0..7

    const int n_kv    = d_meta->kv_len_global;
    const int n_pages = (n_kv + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
    int32_t * out = selected_pages + (size_t)h * SPARSE_MAX_SELECTED;

    if (n_pages <= page_budget) {
        // Short context: attending everything is already inside the budget.
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

    // ---- Stage 1: page scores (8 lanes per page, 4 pages per warp) --------
    {
        __shared__ float s_q_unrot[GQA_RATIO][HEAD_DIM];
        if (warp < GQA_RATIO) {
            float4 t = ((const float4 *)(q + (size_t)(h * GQA_RATIO + warp) * HEAD_DIM))[lane];
            v2_fwht128_4(t.x, t.y, t.z, t.w, lane);
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
        const int sub = lane & 7;    // 16 dims of the page owned by this lane

        // Linearly collapse the GQA_RATIO unit-normalized queries into two 16-dim
        // register arrays:
        //   q_hi[k] = sum_g 0.5 * (q_g[k] + alpha * |q_g[k]|)   (weights k_max_q)
        //   q_lo[k] = sum_g 0.5 * (q_g[k] - alpha * |q_g[k]|)   (weights k_min_q)
        float q_lo[16], q_hi[16];
        #pragma unroll
        for (int k = 0; k < 16; ++k) {
            q_lo[k] = 0.0f;
            q_hi[k] = 0.0f;
        }
        #pragma unroll
        for (int gq = 0; gq < GQA_RATIO; ++gq) {
            const float4 * qrow = (const float4 *)(s_q_unrot[gq] + sub * 16);
            #pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float4 t = qrow[i];
                const float qv[4] = { t.x, t.y, t.z, t.w };
                #pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const float a = SPARSE_QUEST_ALPHA * fabsf(qv[j]);
                    q_hi[4 * i + j] += 0.5f * (qv[j] + a);
                    q_lo[4 * i + j] += 0.5f * (qv[j] - a);
                }
            }
        }

        const size_t lane_off = (size_t)sub * 16;
        const int n_warps = (int)gridDim.x * 8;
        const int gw      = (int)blockIdx.x * 8 + warp;
        const int mychunk = (n_cand + n_warps - 1) / n_warps;
        const int p0      = cand_begin + gw * mychunk;
        const int p1      = (p0 + mychunk < cand_end) ? (p0 + mychunk) : cand_end;
        const int phi     = cand_end - 1;

        if (p0 < p1) {
            v2u4 mn, mx;
            float dv;
            // Fetch the 4 pages of tile `pp` (this lane's page is pp + pg).
            auto fetch4 = [&](int pp, v2u4 & omn, v2u4 & omx, float & odv) {
                const int pc = (pp + pg < phi) ? (pp + pg) : phi;
                const size_t off = (size_t)pc * HEAD_DIM + lane_off;
                omn = __builtin_nontemporal_load((const v2u4 *)(kmin_p + off));
                omx = __builtin_nontemporal_load((const v2u4 *)(kmax_p + off));
                odv = __half2float(d_p[pc]);
            };
            // Score this lane's page across the collapsed GQA group and store it.
            auto store4 = [&](const v2u4 & omn, const v2u4 & omx, float odv, int pp) {
                float s = 0.0f;
                #pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const uint32_t wmn = omn[i];
                    const uint32_t wmx = omx[i];
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const float lo = (float)(int8_t)((wmn >> (8 * j)) & 0xFFu);
                        const float hi = (float)(int8_t)((wmx >> (8 * j)) & 0xFFu);
                        s = fmaf(q_lo[4 * i + j], lo, s);
                        s = fmaf(q_hi[4 * i + j], hi, s);
                    }
                }
                s *= odv;
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
                v2u4 mn1, mx1;
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
    __shared__ uint32_t s_hist[V2_SEL_HIST];
    __shared__ int      s_ctl[24];
    __shared__ int      s_last;

    v2_gl_fence_release();
    __syncthreads();
    if (tid == 0) {
        const int prev = __hip_atomic_fetch_add(&done[h], 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_last = (prev == (int)gridDim.x - 1) ? 1 : 0;
    }
    __syncthreads();
    if (!s_last) {
        return;
    }
    v2_gl_invalidate();
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

    // Candidate keys: 16 per thread, ONE global read, kept in VGPRs across all
    // four radix passes (v1 re-read the score array 7 times).
    uint32_t key[V2_SEL_SLOTS];
    #pragma unroll
    for (int i = 0; i < V2_SEL_SLOTS; ++i) {
        const int idx = tid + i * V2_SEL_THREADS;
        key[i] = (idx < n_cand)
            ? v2_score_key(page_scores[hbase + (size_t)(cand_begin + idx)])
            : 0u;   // 0 is below every real key (the -inf key is 0x007FFFFF)
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
        for (int i = 0; i < V2_SEL_SLOTS; ++i) {
            const int idx = tid + i * V2_SEL_THREADS;
            if (idx < n_cand) {
                const uint32_t k = key[i];
                if (shift == 24 || ((k >> (shift + 8)) == (prefix >> (shift + 8)))) {
                    atomicAdd(&s_hist[(k >> shift) & 0xFFu], 1u);
                }
            }
        }
        __syncthreads();

        // Exclusive suffix sum over the 256 bins, then elect the bin that
        // contains the K-th largest key.
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
        // Every lane must execute the shuffle (an inactive source lane leaves
        // the result undefined on RDNA3); only the use is predicated.
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

    // Emit: all keys strictly above the elected one, then ties in scan order
    // until `need` slots are filled (deterministic for any input).
    int gt = 0, eq = 0;
    #pragma unroll
    for (int i = 0; i < V2_SEL_SLOTS; ++i) {
        const int idx = tid + i * V2_SEL_THREADS;
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
    for (int i = 0; i < V2_SEL_SLOTS; ++i) {
        const int idx = tid + i * V2_SEL_THREADS;
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
// Step 8D: Bit-exact batched m-row Quest page selection (1x metadata HBM read)
// ============================================================================
template <int M_MAX>
__launch_bounds__(V2_SEL_THREADS)
static __global__ void sparse_global_score_v2_mrows_kernel(
        const float * __restrict__ q,                 // [nrows, N_HEAD, HEAD_DIM]
        SparseGlobalPageMeta meta,
        const StepMeta * __restrict__ d_meta,         // [nrows]
        float * __restrict__ page_scores,             // [nrows][N_HEAD_KV][SPARSE_MAX_PAGES]
        int page_budget,
        int nrows) {
    const int h    = (int)blockIdx.y;
    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int n_kv_max    = d_meta[nrows - 1].kv_len_global;
    const int n_pages_max = (n_kv_max + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
    if (n_pages_max <= page_budget) {
        return;
    }

    const int cand_begin = SPARSE_SINK_PAGES;
    const int cand_end   = n_pages_max - SPARSE_RECENT_PAGES;
    const int n_cand     = cand_end - cand_begin;
    if (n_cand <= 0) {
        return;
    }

    // Stage all `nrows` collapsed damped-Quest query weights for KV head `h` into LDS
    // once (8 KB at M_MAX=8, down from 12 KB):
    //   s_qhi[r][d] = sum_g 0.5 * (q_{r,g}[d] + alpha * |q_{r,g}[d]|)   (weights k_max_q)
    //   s_qlo[r][d] = sum_g 0.5 * (q_{r,g}[d] - alpha * |q_{r,g}[d]|)   (weights k_min_q)
    __shared__ float s_qlo[M_MAX][HEAD_DIM];
    __shared__ float s_qhi[M_MAX][HEAD_DIM];
    for (int r = warp; r < nrows; r += 8) {
        float ql[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float qh[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
        #pragma unroll
        for (int gq = 0; gq < GQA_RATIO; ++gq) {
            float4 t = ((const float4 *)(q + (size_t)r * N_Q + (size_t)(h * GQA_RATIO + gq) * HEAD_DIM))[lane];
            v2_fwht128_4(t.x, t.y, t.z, t.w, lane);
            float n2 = t.x * t.x + t.y * t.y + t.z * t.z + t.w * t.w;
            n2 += __shfl_xor(n2, 16, 32);
            n2 += __shfl_xor(n2,  8, 32);
            n2 += __shfl_xor(n2,  4, 32);
            n2 += __shfl_xor(n2,  2, 32);
            n2 += __shfl_xor(n2,  1, 32);
            const float inv_n = rsqrtf(n2 + 1e-12f);
            const float qv[4] = { t.x * inv_n, t.y * inv_n, t.z * inv_n, t.w * inv_n };
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float a = SPARSE_QUEST_ALPHA * fabsf(qv[j]);
                qh[j] += 0.5f * (qv[j] + a);
                ql[j] += 0.5f * (qv[j] - a);
            }
        }
        ((float4 *)(s_qlo[r]))[lane] = make_float4(ql[0], ql[1], ql[2], ql[3]);
        ((float4 *)(s_qhi[r]))[lane] = make_float4(qh[0], qh[1], qh[2], qh[3]);
    }
    __syncthreads();

    const size_t  hbase  = (size_t)h * (size_t)meta.max_pages;
    const int8_t * kmin_p = meta.k_min_q + hbase * HEAD_DIM;
    const int8_t * kmax_p = meta.k_max_q + hbase * HEAD_DIM;
    const __half * d_p    = meta.k_mm_d + hbase;

    const int pg  = lane >> 3;   // page slot inside warp (0..3)
    const int sub = lane & 7;    // 16 dims owned by this lane
    const size_t lane_off = (size_t)sub * 16;

    const int n_warps = (int)gridDim.x * 8;
    const int gw      = (int)blockIdx.x * 8 + warp;
    const int mychunk = (n_cand + n_warps - 1) / n_warps;
    const int p0      = cand_begin + gw * mychunk;
    const int p1      = (p0 + mychunk < cand_end) ? (p0 + mychunk) : cand_end;
    const int phi     = cand_end - 1;

    if (p0 >= p1) {
        return;
    }

    auto fetch4 = [&](int pp, v2u4 & omn, v2u4 & omx, float & odv) {
        const int pc = (pp + pg < phi) ? (pp + pg) : phi;
        const size_t off = (size_t)pc * HEAD_DIM + lane_off;
        omn = __builtin_nontemporal_load((const v2u4 *)(kmin_p + off));
        omx = __builtin_nontemporal_load((const v2u4 *)(kmax_p + off));
        odv = __half2float(d_p[pc]);
    };

    auto store4 = [&](const v2u4 & omn, const v2u4 & omx, float odv, int pp) {
        float s[M_MAX];
        #pragma unroll
        for (int r = 0; r < M_MAX; ++r) {
            s[r] = 0.0f;
        }
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            const uint32_t wmn = omn[i];
            const uint32_t wmx = omx[i];
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float lo = (float)(int8_t)((wmn >> (8 * j)) & 0xFFu);
                const float hi = (float)(int8_t)((wmx >> (8 * j)) & 0xFFu);
                const int d = sub * 16 + 4 * i + j;
                #pragma unroll
                for (int r = 0; r < M_MAX; ++r) {
                    if (r < nrows) {
                        s[r] = fmaf(s_qlo[r][d], lo, s[r]);
                        s[r] = fmaf(s_qhi[r][d], hi, s[r]);
                    }
                }
            }
        }
        #pragma unroll
        for (int r = 0; r < M_MAX; ++r) {
            if (r < nrows) {
                s[r] *= odv;
                s[r] += __shfl_xor(s[r], 1, 8);
                s[r] += __shfl_xor(s[r], 2, 8);
                s[r] += __shfl_xor(s[r], 4, 8);
                if (sub == 0 && (pp + pg) < p1) {
                    const size_t sbase = ((size_t)r * N_HEAD_KV + (size_t)h) * SPARSE_MAX_PAGES;
                    page_scores[sbase + (size_t)(pp + pg)] = s[r];
                }
            }
        }
    };

    v2u4 mn, mx;
    float dv;
    int p = p0;
    fetch4(p, mn, mx, dv);
    while (p + 4 < p1) {
        v2u4 mn1, mx1;
        float dv1;
        fetch4(p + 4, mn1, mx1, dv1);
        store4(mn, mx, dv, p);
        mn = mn1; mx = mx1; dv = dv1;
        p += 4;
    }
    store4(mn, mx, dv, p);
}

// Stage 2: all `nrows * N_HEAD_KV` Top-K selects in parallel (zero atomics).
__launch_bounds__(V2_SEL_THREADS)
static __global__ void sparse_global_topk_v2_mrows_kernel(
        const StepMeta * __restrict__ d_meta,         // [nrows]
        const float * __restrict__ page_scores,       // [nrows][N_HEAD_KV][SPARSE_MAX_PAGES]
        int32_t * __restrict__ selected_pages,        // [nrows][N_HEAD_KV][SPARSE_MAX_SELECTED]
        int32_t * __restrict__ n_selected,            // [nrows][N_HEAD_KV]
        int page_budget) {
    const int r    = (int)blockIdx.x;
    const int h    = (int)blockIdx.y;
    const int tid  = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    const int n_kv    = d_meta[r].kv_len_global;
    const int n_pages = (n_kv + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
    int32_t * out     = selected_pages + ((size_t)r * N_HEAD_KV + (size_t)h) * SPARSE_MAX_SELECTED;
    int32_t * n_sel_p = n_selected + (size_t)r * N_HEAD_KV;

    if (n_pages <= page_budget) {
        if (tid == 0) {
            for (int i = 0; i < n_pages; ++i) {
                out[i] = i;
            }
            n_sel_p[h] = n_pages;
        }
        return;
    }

    const int    cand_begin = SPARSE_SINK_PAGES;
    const int    cand_end   = n_pages - SPARSE_RECENT_PAGES;
    const int    n_cand     = cand_end - cand_begin;
    const size_t sbase      = ((size_t)r * N_HEAD_KV + (size_t)h) * SPARSE_MAX_PAGES;

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
        n_sel_p[h] = SPARSE_SINK_PAGES + K + SPARSE_RECENT_PAGES;
    }
    __syncthreads();
    if (K <= 0) {
        return;
    }

    __shared__ uint32_t s_hist[V2_SEL_HIST];
    __shared__ int      s_ctl[24];

    uint32_t key[V2_SEL_SLOTS];
    #pragma unroll
    for (int i = 0; i < V2_SEL_SLOTS; ++i) {
        const int idx = tid + i * V2_SEL_THREADS;
        key[i] = (idx < n_cand)
            ? v2_score_key(page_scores[sbase + (size_t)(cand_begin + idx)])
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
        for (int i = 0; i < V2_SEL_SLOTS; ++i) {
            const int idx = tid + i * V2_SEL_THREADS;
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
    for (int i = 0; i < V2_SEL_SLOTS; ++i) {
        const int idx = tid + i * V2_SEL_THREADS;
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
    for (int i = 0; i < V2_SEL_SLOTS; ++i) {
        const int idx = tid + i * V2_SEL_THREADS;
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

template <int GQA>
static inline void v2_launch_swa(int kv_fmt, const dim3 & grid, const dim3 & block,
        hipStream_t stream,
        const float * q, KVCacheSlot kv_slot, const float * sinks_in, int sinks_stride,
        const float * rel_table, const float * qgate_in, int qgate_stride,
        float * out, float * tmp_o, float2 * tmp_meta, int n_splits,
        const StepMeta * d_meta, int32_t * fa_done) {
    switch (kv_fmt) {
        case KV_Q4_1:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_swa_v2_kernel<GQA, KV_Q4_1>),
                grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,
                qgate_in, qgate_stride, out, tmp_o, tmp_meta, n_splits, d_meta, fa_done);
            break;
        case KV_Q4_0:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_swa_v2_kernel<GQA, KV_Q4_0>),
                grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,
                qgate_in, qgate_stride, out, tmp_o, tmp_meta, n_splits, d_meta, fa_done);
            break;
        default:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_swa_v2_kernel<GQA, KV_Q8_0>),
                grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,
                qgate_in, qgate_stride, out, tmp_o, tmp_meta, n_splits, d_meta, fa_done);
            break;
    }
}

template <int GQA>
static inline void v2_launch_global(int kv_fmt, const dim3 & grid, const dim3 & block,
        hipStream_t stream,
        const float * q, KVCacheSlot kv_slot, const float * sinks_in, int sinks_stride,
        const float * rel_table, const float * qgate_in, int qgate_stride,
        float * out, const int32_t * selected_pages, const int32_t * n_selected,
        float * tmp_o, float2 * tmp_meta, int n_splits,
        const StepMeta * d_meta, int32_t * fa_done) {
    switch (kv_fmt) {
        case KV_Q4_1:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_sparse_global_v2_kernel<GQA, KV_Q4_1>),
                grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,
                qgate_in, qgate_stride, out, selected_pages, n_selected,
                tmp_o, tmp_meta, n_splits, d_meta, fa_done);
            break;
        case KV_Q4_0:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_sparse_global_v2_kernel<GQA, KV_Q4_0>),
                grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,
                qgate_in, qgate_stride, out, selected_pages, n_selected,
                tmp_o, tmp_meta, n_splits, d_meta, fa_done);
            break;
        default:
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_sparse_global_v2_kernel<GQA, KV_Q8_0>),
                grid, block, 0, stream, q, kv_slot, sinks_in, sinks_stride, rel_table,
                qgate_in, qgate_stride, out, selected_pages, n_selected,
                tmp_o, tmp_meta, n_splits, d_meta, fa_done);
            break;
    }
}

} // namespace

// ----------------------------------------------------------------------------
// 1. SWA dense decode FA v2
// ----------------------------------------------------------------------------
void launch_fa_swa_decode_v2(
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
    if (splits > FA_V2_MAX_SPLITS) {
        splits = FA_V2_MAX_SPLITS;
    }
    if (splits > 1 && (tmp_o == nullptr || tmp_meta == nullptr || fa_done == nullptr)) {
        splits = 1;   // no combine storage -> single-split fallback
    }

    const dim3 grid(splits, N_HEAD_KV, 1);
    const dim3 block(32, 4, 1);
    v2_launch_swa<GQA_RATIO>(kv_slot.kv_type, grid, block, stream,
        q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
        out, tmp_o, tmp_meta, splits, d_meta, fa_done);
}

// ----------------------------------------------------------------------------
// 2. Fused page score + Top-K
// ----------------------------------------------------------------------------
void launch_sparse_global_select_v2(
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

    const dim3 grid(FA_V2_SELECT_CHUNKS, N_HEAD_KV, 1);
    const dim3 block(V2_SEL_THREADS, 1, 1);
    hipLaunchKernelGGL(sparse_global_select_v2_kernel, grid, block, 0, stream,
        q, meta, d_meta, scratch.page_scores,
        scratch.selected_pages, scratch.n_selected, select_done, budget);
}

void launch_sparse_global_select_v2_mrows(
        const float * q,
        const SparseGlobalPageMeta & meta,
        SparseGlobalScratch & scratch,
        const StepMeta * d_meta,
        int             sparse_pages,
        hipStream_t     stream,
        int             nrows) {
    if (q == nullptr || d_meta == nullptr || scratch.page_scores == nullptr ||
        scratch.selected_pages == nullptr || scratch.n_selected == nullptr ||
        nrows <= 0 || nrows > SPARSE_MAX_QB) {
        return;
    }
    int budget = sparse_pages;
    if (budget < SPARSE_MIN_PAGES) {
        budget = SPARSE_MIN_PAGES;
    }
    if (budget > SPARSE_MAX_SELECTED - 2) {
        budget = SPARSE_MAX_SELECTED - 2;
    }

    const dim3 score_grid(FA_V2_SELECT_CHUNKS, N_HEAD_KV, 1);
    const dim3 block(V2_SEL_THREADS, 1, 1);
    if (nrows <= 2) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(sparse_global_score_v2_mrows_kernel<2>),
            score_grid, block, 0, stream, q, meta, d_meta, scratch.page_scores, budget, nrows);
    } else if (nrows <= 4) {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(sparse_global_score_v2_mrows_kernel<4>),
            score_grid, block, 0, stream, q, meta, d_meta, scratch.page_scores, budget, nrows);
    } else {
        hipLaunchKernelGGL(HIP_KERNEL_NAME(sparse_global_score_v2_mrows_kernel<8>),
            score_grid, block, 0, stream, q, meta, d_meta, scratch.page_scores, budget, nrows);
    }

    const dim3 topk_grid((unsigned)nrows, N_HEAD_KV, 1);
    hipLaunchKernelGGL(sparse_global_topk_v2_mrows_kernel, topk_grid, block, 0, stream,
        d_meta, scratch.page_scores, scratch.selected_pages, scratch.n_selected, budget);
}

// ----------------------------------------------------------------------------
// 3. Sparse Global decode FA v2
// ----------------------------------------------------------------------------
void launch_fa_sparse_global_decode_v2(
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
    if (splits > FA_V2_MAX_SPLITS) {
        splits = FA_V2_MAX_SPLITS;
    }
    if (splits > 1 && (tmp_o == nullptr || tmp_meta == nullptr || fa_done == nullptr)) {
        splits = 1;
    }

    const dim3 grid(splits, N_HEAD_KV, 1);
    const dim3 block(32, 4, 1);
    v2_launch_global<GQA_RATIO>(kv_slot.kv_type, grid, block, stream,
        q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
        out, selected_pages, n_selected_pages, tmp_o, tmp_meta, splits, d_meta, fa_done);
}

} // namespace inkling
