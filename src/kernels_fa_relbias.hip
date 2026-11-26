#include "inkling_engine.h"
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

[[maybe_unused]] static __device__ __forceinline__ __half2 h2_of(uint32_t u) {
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

// ============================================================================
// Kernel 1: Native RDNA3 Wave32 GQA-3 Split-K Decode FA (Q8_0 / Q4_1 / Q4_0)
//   - Lossless FP32 Q stored in 1.5 KB broadcast LDS (__shared__ float4 s_q4[3][36])
//     -> slashes VGPRs from 145 to <= 64 (2x wave occupancy!) with 0 bank conflicts
//   - Nontemporal 128-bit/32-bit DRAM streaming for K/V (protects L2)
//   - Zero-cast int8/u4 -> __half2 via v_perm_b32 + __hsub2, v_fma_mix_f32 dot/V
//   - Q4_1 uses precomputed q_sum[g] + scalar om_reg[g] (1 extra FMA per block/key!)
//   - Pure log2-domain online softmax (exp2f -> 1-cycle v_exp_f32)
//   - Conditional O rescale (skips 12 FMAs + exp2f whenever m_new == m_old)
//   - Optional self-resetting atomic combine (eliminates combine kernel launch!)
// ============================================================================
template <int GQA, bool IS_SWA, int KV_FMT>
__launch_bounds__(128) __attribute__((amdgpu_waves_per_eu(12, 16)))
static __global__ void inkling_fa_decode_kernel(
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
        int                           n_tokens,
        int                           pos_start,
        int                           n_splits,
        const StepMeta * __restrict__ d_meta,
        int32_t        * __restrict__ fa_done) {
    constexpr int D      = HEAD_DIM; // 128
    constexpr int NWARPS = 4;

    const int split  = blockIdx.x;
    const int grp_id = blockIdx.y; // 0..N_HEAD_KV-1
    const int tok    = blockIdx.z; // 0..n_tokens-1

    const int h_base = grp_id * GQA;
    const int h_kv   = grp_id;

    const int lane   = threadIdx.x; // 0..31
    const int warp   = threadIdx.y; // 0..3
    const int blk    = lane & 3;    // 0..3 -> 32-elem K block [blk*32 .. blk*32+31]
    const int k_slot = lane >> 2;   // 0..7 -> 8 parallel keys per warp

    const int32_t qp_val = (d_meta != nullptr) ? d_meta->pos : (pos_start + tok);
    const int32_t n_kv   = (d_meta != nullptr)
        ? (IS_SWA ? d_meta->kv_len_swa : d_meta->kv_len_global)
        : (IS_SWA ? min(pos_start + n_tokens, kv_slot.cap) : (pos_start + n_tokens));

    // Partition [0, n_kv) in 8-key chunks so splits are warp-aligned
    const int k_per_split = (n_kv > 0) ? ((((n_kv + n_splits - 1) / n_splits) + 7) & ~7) : 0;
    const int k_start     = split * k_per_split;
    const int k_end       = min(k_start + k_per_split, n_kv);

    union DecodeSmem {
        struct {
            float4 q4[GQA][36]; // 4 blocks x 9 float4s (144 B stride -> 0 LDS bank conflicts across blk=0..3!)
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

    if (n_splits == 1 || k_start < k_end) {
        // Cooperatively load lossless FP32 Q[0..GQA-1] pre-scaled by KQ_SCALE * LOG2E into bank-conflict-free LDS
        constexpr float Q_SCALE_LOG2 = KQ_SCALE * LOG2E;
        if (warp < GQA) {
            const float * qp = q + ((size_t)tok * N_HEAD + (h_base + warp)) * D;
            const float4 fv  = ((const float4 *)qp)[lane];
            const int q_b = lane >> 3; // 0..3
            const int q_i = lane & 7;  // 0..7
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

        const float * rel_base = rel_table + ((int64_t)tok * N_HEAD + h_base) * N_REL_EXTENT;
        const int v_blk = lane >> 3; // 0..3

        for (int j_base = k_start + warp * 8; j_base < k_end; j_base += NWARPS * 8) {
            const int j = j_base + k_slot;
            const bool in_range = (j < k_end);
            int32_t kp_val = -1;
            __half d_kh = __float2half(0.0f);
            __half2 dm_kh = h2_of(0u);
            v4u u0 = {0, 0, 0, 0}, u1 = {0, 0, 0, 0};
            const size_t head_base = (size_t)h_kv * (size_t)kv_slot.cap;
            if (in_range) {
                kp_val = kv_slot.k_pos[j];
                const size_t head_slot = head_base + (size_t)j;
                if constexpr (KV_FMT == KV_Q8_0) {
                    d_kh = __ldg(&kv_slot.k_d[head_slot * 4 + blk]);
                    const int8_t * k_ptr = kv_slot.k_qs + head_slot * D + blk * 32;
                    if constexpr (!IS_SWA) {
                        u0 = __builtin_nontemporal_load((const v4u *)(k_ptr + 0));
                        u1 = __builtin_nontemporal_load((const v4u *)(k_ptr + 16));
                    } else {
                        u0 = *(const v4u *)(k_ptr + 0);
                        u1 = *(const v4u *)(k_ptr + 16);
                    }
                } else {
                    if constexpr (KV_FMT == KV_Q4_1) {
                        dm_kh = __ldg(&((const __half2 *)kv_slot.k_d)[head_slot * 4 + blk]);
                    } else {
                        d_kh = __ldg(&kv_slot.k_d[head_slot * 4 + blk]);
                    }
                    const int8_t * k_ptr = kv_slot.k_qs + head_slot * (D / 2) + blk * 16;
                    if constexpr (!IS_SWA) {
                        u0 = __builtin_nontemporal_load((const v4u *)k_ptr);
                    } else {
                        u0 = *(const v4u *)k_ptr;
                    }
                }
            }

            const bool valid = (kp_val >= 0) && (kp_val <= qp_val) &&
                               (!IS_SWA || (qp_val - kp_val) < SWA_WINDOW);

            const uint32_t valid_ballot = __ballot(valid);
            if (valid_ballot == 0u) {
                continue;
            }

            if (valid_ballot == 0xFFFFFFFFu) {
                // Fast full-8-key path (single basic block):
                // Issue coalesced v_d + v_qs loads BEFORE K dot products so V DRAM
                // latency overlaps the entire 96-FMA K dot + reductions + RelBias + softmax!
                const size_t base_slot = head_base + (size_t)j_base;
                float vd_lane = 0.0f;
                uint32_t vd_u_lane = 0u;
                uint32_t vq8[8];
                uint32_t vq4[4];
                if constexpr (KV_FMT == KV_Q8_0) {
                    vd_lane = __half2float(__ldg(&kv_slot.v_d[base_slot * 4 + lane]));
                    #pragma unroll
                    for (int kk = 0; kk < 8; ++kk) {
                        if constexpr (!IS_SWA) {
                            vq8[kk] = __builtin_nontemporal_load(((const uint32_t *)(kv_slot.v_qs + (base_slot + (size_t)kk) * D)) + lane);
                        } else {
                            vq8[kk] = *((const uint32_t *)(kv_slot.v_qs + (base_slot + (size_t)kk) * D) + lane);
                        }
                    }
                } else {
                    if constexpr (KV_FMT == KV_Q4_0) {
                        vd_lane = __half2float(__ldg(&kv_slot.v_d[base_slot * 4 + lane]));
                    }
                    const int v_half_lane = lane >> 1;
                    const int v_odd_lane  = lane & 1;
                    #pragma unroll
                    for (int p = 0; p < 4; ++p) {
                        const size_t v_slot = base_slot + (size_t)(2 * p + v_odd_lane);
                        const uint32_t * vp = ((const uint32_t *)(kv_slot.v_qs + v_slot * (D / 2))) + v_half_lane;
                        if constexpr (!IS_SWA) {
                            vq4[p] = __builtin_nontemporal_load(vp);
                        } else {
                            vq4[p] = *vp;
                        }
                    }
                }

                // 1. K dot product (fast v_perm -> fp16 + v_fma_mix_f32)
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

                // On-the-fly RelBias (scaled to log2 domain) on lane blk == 0 only
                if (blk == 0) {
                    const int64_t d64 = (int64_t)qp_val - (int64_t)kp_val;
                    if (d64 >= 0 && d64 < N_REL_EXTENT) {
                        const int dist = (int)d64;
                        #pragma unroll
                        for (int g = 0; g < GQA; ++g) {
                            sc_k[g] += rel_base[(int64_t)g * N_REL_EXTENT + dist] * LOG2E;
                        }
                    }
                }

                // 2. Tile-8 log2-domain online softmax: skip O rescale when m_new == m_old!
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

                // 3. Coalesced 32-lane V accumulation over all 8 valid keys
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
                        vd_u_lane = __ldg(&((const uint32_t *)kv_slot.v_d)[base_slot * 4 + lane]);
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
                // Partial/boundary tile path
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
                            sc_k[g] += rel_base[(int64_t)g * N_REL_EXTENT + dist] * LOG2E;
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
                        const float d_v = __half2float(__ldg(&kv_slot.v_d[head_slot * 4 + v_blk]));
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
                        const uint32_t w = *((const uint32_t *)(kv_slot.v_qs + head_slot * (D / 2)) + (lane >> 1));
                        const uint32_t nibs = ((lane & 1) ? (w >> 4) : w) & 0x0F0F0F0Fu;
                        u4x4_to_h2x2<KV_FMT == KV_Q4_0>(nibs, vh01, vh23);
                        if constexpr (KV_FMT == KV_Q4_1) {
                            const __half2 dm_v = __ldg(&((const __half2 *)kv_slot.v_d)[head_slot * 4 + v_blk]);
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
                            const float d_v = __half2float(__ldg(&kv_slot.v_d[head_slot * 4 + v_blk]));
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

        // Wait for all warps to finish reading smem.q before reusing smem.red
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
                    const float sink_val_log2 = sinks_in[(size_t)tok * sinks_stride + h] * LOG2E;
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
                        const float * g_ptr = qgate_in + (size_t)tok * qgate_stride + h * D + lane * 4;
                        const float4 gv = *(const float4 *)g_ptr;
                        v0 *= 1.0f / (1.0f + exp2f(-gv.x * LOG2E));
                        v1 *= 1.0f / (1.0f + exp2f(-gv.y * LOG2E));
                        v2 *= 1.0f / (1.0f + exp2f(-gv.z * LOG2E));
                        v3 *= 1.0f / (1.0f + exp2f(-gv.w * LOG2E));
                    }
                    const size_t out_off = ((size_t)tok * N_HEAD + h) * D + lane * 4;
                    *(float4 *)(out + out_off) = make_float4(v0, v1, v2, v3);
                } else {
                    const int64_t meta_idx = ((int64_t)tok * N_HEAD + h) * n_splits + split;
                    if (lane == 0) {
                        tmp_meta[meta_idx] = make_float2(m_blk, l_blk);
                    }
                    *(float4 *)(tmp_o + meta_idx * D + lane * 4) = make_float4(oa0, oa1, oa2, oa3);
                }
            }
        }
    } else {
        // Empty split (only possible when n_splits > 1)
        if (warp == 0 && lane == 0) {
            #pragma unroll
            for (int g = 0; g < GQA; ++g) {
                const int h = h_base + g;
                const int64_t meta_idx = ((int64_t)tok * N_HEAD + h) * n_splits + split;
                tmp_meta[meta_idx] = make_float2(-INFINITY, 0.0f);
            }
        }
    }

    // In-kernel self-resetting atomic combine (when fa_done != nullptr && n_splits > 1 && n_tokens == 1)
    if (fa_done != nullptr && n_splits > 1 && n_tokens == 1) {
        __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    // gfx12: no equivalent instruction - on gfx1201 __scoped_atomic_thread_fence(RELEASE, AGENT)
    // above already lowers to s_waitcnt + `global_wb scope:SCOPE_DEV` (ISA-checked with
    // llvm-mc/hipcc -S), so gfx1200/gfx1201 need nothing extra here.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#endif
        __syncthreads();
        if (warp == 0 && lane == 0) {
            __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    // gfx12: no equivalent instruction - on gfx1201 __scoped_atomic_thread_fence(RELEASE, AGENT)
    // above already lowers to s_waitcnt + `global_wb scope:SCOPE_DEV` (ISA-checked with
    // llvm-mc/hipcc -S), so gfx1200/gfx1201 need nothing extra here.
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
    // gfx12: no equivalent instruction - scope-qualified acquire atomics on gfx1201 emit
    // `global_inv scope:SCOPE_DEV` themselves (ISA-checked), which replaces the RDNA3
    // L0/L1 invalidation below and needs no hand-written cache asm.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("buffer_gl1_inv\n\tbuffer_gl0_inv" ::: "memory");
#endif
            if (warp == 0 && lane == 0) {
                __hip_atomic_store(&fa_done[grp_id], 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
            if (warp < GQA) {
                const int h = h_base + warp;
                const int64_t base_idx = (int64_t)h * n_splits;
                const float sink_val_log2 = sinks_in[h] * LOG2E;

                float l_sum = 0.0f;
                float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
                float m_final;

                if (n_splits <= 32) {
                    const float2 ml_lane = (lane < n_splits)
                        ? tmp_meta[base_idx + lane]
                        : make_float2(-INFINITY, 0.0f);
                    float m_all = ml_lane.x;
                    m_all = fmaxf(m_all, __shfl_xor(m_all, 16, 32));
                    m_all = fmaxf(m_all, __shfl_xor(m_all,  8, 32));
                    m_all = fmaxf(m_all, __shfl_xor(m_all,  4, 32));
                    m_all = fmaxf(m_all, __shfl_xor(m_all,  2, 32));
                    m_all = fmaxf(m_all, __shfl_xor(m_all,  1, 32));
                    m_final = fmaxf(m_all, sink_val_log2);

                    const float sc_lane = (lane < n_splits && ml_lane.x > -INFINITY) ? exp2f(ml_lane.x - m_final) : 0.0f;
                    l_sum = ml_lane.y * sc_lane;
                    const uint32_t valid_mask = __ballot(sc_lane > 0.0f);

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
                } else {
                    float m_all = -INFINITY;
                    for (int base_sp = 0; base_sp < n_splits; base_sp += 32) {
                        const int sp = base_sp + lane;
                        const float m_sp = (sp < n_splits) ? tmp_meta[base_idx + sp].x : -INFINITY;
                        m_all = fmaxf(m_all, m_sp);
                    }
                    m_all = fmaxf(m_all, __shfl_xor(m_all, 16, 32));
                    m_all = fmaxf(m_all, __shfl_xor(m_all,  8, 32));
                    m_all = fmaxf(m_all, __shfl_xor(m_all,  4, 32));
                    m_all = fmaxf(m_all, __shfl_xor(m_all,  2, 32));
                    m_all = fmaxf(m_all, __shfl_xor(m_all,  1, 32));
                    m_final = fmaxf(m_all, sink_val_log2);

                    for (int base_sp = 0; base_sp < n_splits; base_sp += 32) {
                        const int sp = base_sp + lane;
                        float sc_lane = 0.0f;
                        float l_lane  = 0.0f;
                        if (sp < n_splits) {
                            const float2 ml = tmp_meta[base_idx + sp];
                            if (ml.x > -INFINITY) {
                                sc_lane = exp2f(ml.x - m_final);
                                l_lane  = ml.y * sc_lane;
                            }
                        }
                        l_sum += l_lane;
                        const uint32_t valid_mask = __ballot(sc_lane > 0.0f);
                        const int chunk_end = min(32, n_splits - base_sp);
                        int k = 0;
                        for (; k + 4 <= chunk_end; k += 4) {
                            const uint32_t m4 = (valid_mask >> k) & 0xFu;
                            if (m4 == 0xFu) {
                                const float sc0 = __shfl(sc_lane, k + 0, 32);
                                const float sc1 = __shfl(sc_lane, k + 1, 32);
                                const float sc2 = __shfl(sc_lane, k + 2, 32);
                                const float sc3 = __shfl(sc_lane, k + 3, 32);
                                const float4 pv0 = *(const float4 *)(tmp_o + (base_idx + base_sp + k + 0) * D + lane * 4);
                                const float4 pv1 = *(const float4 *)(tmp_o + (base_idx + base_sp + k + 1) * D + lane * 4);
                                const float4 pv2 = *(const float4 *)(tmp_o + (base_idx + base_sp + k + 2) * D + lane * 4);
                                const float4 pv3 = *(const float4 *)(tmp_o + (base_idx + base_sp + k + 3) * D + lane * 4);
                                o0 = fmaf(pv0.x, sc0, o0); o1 = fmaf(pv0.y, sc0, o1); o2 = fmaf(pv0.z, sc0, o2); o3 = fmaf(pv0.w, sc0, o3);
                                o0 = fmaf(pv1.x, sc1, o0); o1 = fmaf(pv1.y, sc1, o1); o2 = fmaf(pv1.z, sc1, o2); o3 = fmaf(pv1.w, sc1, o3);
                                o0 = fmaf(pv2.x, sc2, o0); o1 = fmaf(pv2.y, sc2, o1); o2 = fmaf(pv2.z, sc2, o2); o3 = fmaf(pv2.w, sc2, o3);
                                o0 = fmaf(pv3.x, sc3, o0); o1 = fmaf(pv3.y, sc3, o1); o2 = fmaf(pv3.z, sc3, o2); o3 = fmaf(pv3.w, sc3, o3);
                            } else if (m4 != 0u) {
                                #pragma unroll
                                for (int j = 0; j < 4; ++j) {
                                    if ((m4 >> j) & 1u) {
                                        const float sc = __shfl(sc_lane, k + j, 32);
                                        const float4 pv = *(const float4 *)(tmp_o + (base_idx + base_sp + k + j) * D + lane * 4);
                                        o0 = fmaf(pv.x, sc, o0); o1 = fmaf(pv.y, sc, o1); o2 = fmaf(pv.z, sc, o2); o3 = fmaf(pv.w, sc, o3);
                                    }
                                }
                            }
                        }
                        for (; k < chunk_end; ++k) {
                            if ((valid_mask >> k) & 1u) {
                                const float sc = __shfl(sc_lane, k, 32);
                                const float4 pv = *(const float4 *)(tmp_o + (base_idx + base_sp + k) * D + lane * 4);
                                o0 = fmaf(pv.x, sc, o0); o1 = fmaf(pv.y, sc, o1); o2 = fmaf(pv.z, sc, o2); o3 = fmaf(pv.w, sc, o3);
                            }
                        }
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
                    const float4 gv = *(const float4 *)(qgate_in + h * D + lane * 4);
                    v0 *= 1.0f / (1.0f + exp2f(-gv.x * LOG2E));
                    v1 *= 1.0f / (1.0f + exp2f(-gv.y * LOG2E));
                    v2 *= 1.0f / (1.0f + exp2f(-gv.z * LOG2E));
                    v3 *= 1.0f / (1.0f + exp2f(-gv.w * LOG2E));
                }
                *(float4 *)(out + h * D + lane * 4) = make_float4(v0, v1, v2, v3);
            }
        }
    }
}

static __device__ __forceinline__ float fast_exp2f(float x) {
    return __builtin_amdgcn_exp2f(x);
}

// Combine splits (fallback when fa_done == nullptr or prefill split-KV) & fold sinks + FWHT-128 + sigmoid(qgate)
// Note: tmp_meta[].x is in log2 domain!
// 4 warps/CTA (128 threads): each warp combines 1 head, giving 4 warps/CU minimum occupancy instead of 1!
__launch_bounds__(128, 8)
static __global__ void inkling_fa_decode_combine_kernel(
        const float  * __restrict__ tmp_o,
        const float2 * __restrict__ tmp_meta,
        const float  * __restrict__ sinks_in,
        int                         sinks_stride,
        const float  * __restrict__ qgate_in,
        int                         qgate_stride,
        float        * __restrict__ out,
        int                         n_splits) {
    constexpr int D = HEAD_DIM;
    const int warp = __builtin_amdgcn_readfirstlane((int)threadIdx.y);
    const int h    = blockIdx.x * 4 + warp;
    if (h >= N_HEAD) {
        return;
    }
    const int tok  = blockIdx.y;
    const int lane = threadIdx.x; // 0..31 -> 4 dims each

    const int64_t base_idx = ((int64_t)tok * N_HEAD + h) * n_splits;
    const float sink_val_log2 = sinks_in[(size_t)tok * sinks_stride + h] * LOG2E;

    float l_sum = 0.0f;
    float o0 = 0.0f, o1 = 0.0f, o2 = 0.0f, o3 = 0.0f;
    float m_final;

    if (n_splits <= 32) {
        const float2 ml_lane = (lane < n_splits)
            ? tmp_meta[base_idx + lane]
            : make_float2(-INFINITY, 0.0f);
        float m_all = ml_lane.x;
        m_all = fmaxf(m_all, __shfl_xor(m_all, 16, 32));
        m_all = fmaxf(m_all, __shfl_xor(m_all,  8, 32));
        m_all = fmaxf(m_all, __shfl_xor(m_all,  4, 32));
        m_all = fmaxf(m_all, __shfl_xor(m_all,  2, 32));
        m_all = fmaxf(m_all, __shfl_xor(m_all,  1, 32));
        m_final = fmaxf(m_all, sink_val_log2);

        const float sc_lane = (lane < n_splits && ml_lane.x > -INFINITY) ? fast_exp2f(ml_lane.x - m_final) : 0.0f;
        l_sum = ml_lane.y * sc_lane;
        const uint32_t valid_mask = __ballot(sc_lane > 0.0f);

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
    } else {
        float m_all = -INFINITY;
        for (int base_sp = 0; base_sp < n_splits; base_sp += 32) {
            const int sp = base_sp + lane;
            const float m_sp = (sp < n_splits) ? tmp_meta[base_idx + sp].x : -INFINITY;
            m_all = fmaxf(m_all, m_sp);
        }
        m_all = fmaxf(m_all, __shfl_xor(m_all, 16, 32));
        m_all = fmaxf(m_all, __shfl_xor(m_all,  8, 32));
        m_all = fmaxf(m_all, __shfl_xor(m_all,  4, 32));
        m_all = fmaxf(m_all, __shfl_xor(m_all,  2, 32));
        m_all = fmaxf(m_all, __shfl_xor(m_all,  1, 32));
        m_final = fmaxf(m_all, sink_val_log2);

        for (int base_sp = 0; base_sp < n_splits; base_sp += 32) {
            const int sp = base_sp + lane;
            float sc_lane = 0.0f;
            float l_lane  = 0.0f;
            if (sp < n_splits) {
                const float2 ml = tmp_meta[base_idx + sp];
                if (ml.x > -INFINITY) {
                    sc_lane = fast_exp2f(ml.x - m_final);
                    l_lane  = ml.y * sc_lane;
                }
            }
            l_sum += l_lane;
            const uint32_t valid_mask = __ballot(sc_lane > 0.0f);
            const int chunk_end = min(32, n_splits - base_sp);
            int k = 0;
            for (; k + 4 <= chunk_end; k += 4) {
                const uint32_t m4 = (valid_mask >> k) & 0xFu;
                if (m4 == 0xFu) {
                    const float sc0 = __shfl(sc_lane, k + 0, 32);
                    const float sc1 = __shfl(sc_lane, k + 1, 32);
                    const float sc2 = __shfl(sc_lane, k + 2, 32);
                    const float sc3 = __shfl(sc_lane, k + 3, 32);
                    const float4 pv0 = *(const float4 *)(tmp_o + (base_idx + base_sp + k + 0) * D + lane * 4);
                    const float4 pv1 = *(const float4 *)(tmp_o + (base_idx + base_sp + k + 1) * D + lane * 4);
                    const float4 pv2 = *(const float4 *)(tmp_o + (base_idx + base_sp + k + 2) * D + lane * 4);
                    const float4 pv3 = *(const float4 *)(tmp_o + (base_idx + base_sp + k + 3) * D + lane * 4);
                    o0 = fmaf(pv0.x, sc0, o0); o1 = fmaf(pv0.y, sc0, o1); o2 = fmaf(pv0.z, sc0, o2); o3 = fmaf(pv0.w, sc0, o3);
                    o0 = fmaf(pv1.x, sc1, o0); o1 = fmaf(pv1.y, sc1, o1); o2 = fmaf(pv1.z, sc1, o2); o3 = fmaf(pv1.w, sc1, o3);
                    o0 = fmaf(pv2.x, sc2, o0); o1 = fmaf(pv2.y, sc2, o1); o2 = fmaf(pv2.z, sc2, o2); o3 = fmaf(pv2.w, sc2, o3);
                    o0 = fmaf(pv3.x, sc3, o0); o1 = fmaf(pv3.y, sc3, o1); o2 = fmaf(pv3.z, sc3, o2); o3 = fmaf(pv3.w, sc3, o3);
                } else if (m4 != 0u) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if ((m4 >> j) & 1u) {
                            const float sc = __shfl(sc_lane, k + j, 32);
                            const float4 pv = *(const float4 *)(tmp_o + (base_idx + base_sp + k + j) * D + lane * 4);
                            o0 = fmaf(pv.x, sc, o0); o1 = fmaf(pv.y, sc, o1); o2 = fmaf(pv.z, sc, o2); o3 = fmaf(pv.w, sc, o3);
                        }
                    }
                }
            }
            for (; k < chunk_end; ++k) {
                if ((valid_mask >> k) & 1u) {
                    const float sc = __shfl(sc_lane, k, 32);
                    const float4 pv = *(const float4 *)(tmp_o + (base_idx + base_sp + k) * D + lane * 4);
                    o0 = fmaf(pv.x, sc, o0); o1 = fmaf(pv.y, sc, o1); o2 = fmaf(pv.z, sc, o2); o3 = fmaf(pv.w, sc, o3);
                }
            }
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
        const float * g_ptr = qgate_in + (size_t)tok * qgate_stride + h * D + lane * 4;
        const float4 gv = *(const float4 *)g_ptr;
        v0 *= 1.0f / (1.0f + fast_exp2f(-gv.x * LOG2E));
        v1 *= 1.0f / (1.0f + fast_exp2f(-gv.y * LOG2E));
        v2 *= 1.0f / (1.0f + fast_exp2f(-gv.z * LOG2E));
        v3 *= 1.0f / (1.0f + fast_exp2f(-gv.w * LOG2E));
    }
    const size_t out_off = ((size_t)tok * N_HEAD + h) * D + lane * 4;
    *(float4 *)(out + out_off) = make_float4(v0, v1, v2, v3);
}

// ============================================================================
// Kernel 2: Supercharged GQA-3 Shared-KV RDNA3 FP16 WMMA Prefill FlashAttention
//   - grid = ((n_tokens + Br - 1) / Br, N_HEAD_KV=8, split_k)
//   - All 3 GQA heads share K/V in LDS (3x DRAM bandwidth & dequant reduction!)
//   - Q_frag staged in LDS to slash VGPRs from 235 -> ~140!
//   - 128-bit coalesced K/V Q8_0 loads + v_perm int8->fp16 + v_pk_mul_f16
//   - 4-key warp-transpose ds_write_b64 for tile_V_T (ZERO ds_write_b16!)
//   - Global interior causal fast-path (skips k_pos, min/max, valid_bits & far RelBias!)
//   - Pure log2-domain online softmax (fast_exp2f -> 1-cycle v_exp_f32) + conditional O rescale
// ============================================================================
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

// Bit-exact __shfl_xor(v, MASK, 32) for a fully active wave32, via ds_swizzle BitMode
// (and_mask = 0x1F, xor_mask = MASK). Unlike __shfl_xor's ds_bpermute it needs no lane-index
// VGPR, so LICM has nothing to hoist out of the KV tile loop and keep live across it (clang 22
// spilled exactly those lane^k indices in the 6-warp kernel, issue #6).
template <int MASK>
static __device__ __forceinline__ int shfl_xor_swz(int v) {
    static_assert(MASK > 0 && MASK < 32, "ds_swizzle BitMode xor_mask is 5 bits");
    return __builtin_amdgcn_ds_swizzle(v, (MASK << 10) | 0x1F);
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

// Dequantize 16 packed int8_t (uint4) with scale_h -> 8 packed __half2 (uint32_t[8])
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

// 4-key warp-transpose: each lane holds 16 dims (8 __half2) of key sub_k for c16 = lane >> 2.
// Exchanges via 2 shuffle steps and writes 4 uint2 (64-bit ds_write_b64) into tile_V_T[d][kk_swiz].
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
        // For BC=48 (STRIDE_V=56, 7 chunks of 8 _Float16s): swizzle 8-elem chunk by (d >> 3) mod 7
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

template <int NWARPS, int BC, bool IS_SWA, int KV_FMT>
__launch_bounds__(NWARPS * 32) __attribute__((amdgpu_waves_per_eu((NWARPS == 6) ? 9 : 7, 12)))
static __global__ void inkling_fa_prefill_gqa_wmma_kernel(
        const float * __restrict__ q,
        KVCacheSlot                kv_slot,
        const float * __restrict__ sinks_in,
        int                        sinks_stride,
        const float * __restrict__ rel_table,
        const float * __restrict__ qgate_in,
        int                        qgate_stride,
        float       * __restrict__ out,
        float       * __restrict__ tmp_o,
        float2      * __restrict__ tmp_meta,
        int                        n_tokens,
        int                        pos_start,
        int                        n_kv,
        int                        split_k) {
#if NIB_MMA_TILE_TARGET
    static_assert(NWARPS % GQA_RATIO == 0, "NWARPS must be a multiple of GQA_RATIO=3");
    static_assert(BC == 32 || BC == 48 || BC == 64, "BC must be 32, 48, or 64");

    constexpr int D        = HEAD_DIM;
    constexpr int NQ_TILES = NWARPS / GQA_RATIO;
    constexpr int Br       = NQ_TILES * 16;
    constexpr int NK16     = BC / 16; // 2, 3, or 4
    constexpr int STRIDE_K = 136;     // 17 * 16B
    constexpr int STRIDE_V = (BC == 48) ? (BC + 8) : BC;
    constexpr bool SWIZ_V  = (BC == 32 || BC == 64);
    constexpr float Q_SCALE_LOG2 = KQ_SCALE * LOG2E;

    const int q_block = blockIdx.x * Br;
    const int h_kv    = blockIdx.y; // 0..N_HEAD_KV-1
    const int split   = blockIdx.z; // 0..split_k-1

    const int lane    = threadIdx.x; // 0..31
    // threadIdx.y is uniform across every Wave32: keep warp, qt, g in SGPRs!
    const int warp    = __builtin_amdgcn_readfirstlane((int)threadIdx.y); // 0..NWARPS-1
    const int qt      = warp / GQA_RATIO; // 0..NQ_TILES-1
    const int g       = warp % GQA_RATIO; // 0..2

    const int q_local = lane & 15;   // 0..15
    const int half_w  = lane >> 4;   // 0 or 1

    const int32_t q_pos_min = pos_start + q_block;
    const int32_t q_pos_max = pos_start + min(q_block + Br, n_tokens) - 1;
    const int32_t q_pos_lim = pos_start + n_tokens;
    const int32_t qp_val    = pos_start + q_block + qt * 16 + q_local;
    const bool    q_val     = (qp_val < q_pos_lim);

    // Hybrid Q storage + shared K/V_T union:
    // Chunks c=0..1 (dims 0..31) stored in LDS (smem.Q_lo[warp][16][32]).
    // Chunks c=2..4 held in Q_reg[0..2] on half_w==0 (lanes 0..15).
    // Chunks c=5..7 held in Q_reg[0..2] on half_w==1 (lanes 16..31).
    struct KVShared {
        alignas(16) _Float16 Q_lo[NWARPS][16][32];
        union {
            alignas(16) _Float16 K[BC][STRIDE_K];
            alignas(16) _Float16 V_T[D][STRIDE_V];
        };
    };
    __shared__ KVShared smem;
    __shared__ int32_t tile_kpos[BC];
    __shared__ int32_t tile_min_kp;
    __shared__ int32_t tile_max_kp;
    __shared__ int32_t tile_skip;

    halfx16_t Q_reg[3];
    if (q_val) {
        const int qi_init = qp_val - pos_start;
        const int h_init  = h_kv * GQA_RATIO + g;
        const float * q_ptr = q + ((size_t)qi_init * N_HEAD + h_init) * D;
        {
            const int c_lds = half_w; // 0 or 1
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
        }
        const float * q_hi = q_ptr + (2 + half_w * 3) * 16;
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float4 fv = *(const float4 *)(q_hi + c * 16 + k * 4);
                Q_reg[c][k * 4 + 0] = (_Float16)(fv.x * Q_SCALE_LOG2);
                Q_reg[c][k * 4 + 1] = (_Float16)(fv.y * Q_SCALE_LOG2);
                Q_reg[c][k * 4 + 2] = (_Float16)(fv.z * Q_SCALE_LOG2);
                Q_reg[c][k * 4 + 3] = (_Float16)(fv.w * Q_SCALE_LOG2);
            }
        }
    } else {
        const int col_swiz = (half_w ^ ((q_local >> 1) & 1)) * 16;
        *reinterpret_cast<uint4 *>(&smem.Q_lo[warp][q_local][col_swiz + 0]) = make_uint4(0, 0, 0, 0);
        *reinterpret_cast<uint4 *>(&smem.Q_lo[warp][q_local][col_swiz + 8]) = make_uint4(0, 0, 0, 0);
        #pragma unroll
        for (int c = 0; c < 3; ++c) {
            #pragma unroll
            for (int k = 0; k < 16; ++k) {
                Q_reg[c][k] = (_Float16)0.0f;
            }
        }
    }

    floatx8_t O_acc[8];
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        O_acc[c] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    }

    float m_q = -INFINITY;
    float l_q = 0.0f;

    constexpr int N_CHUNKS = BC / 4; // 8 (BC=32), 12 (BC=48), or 16 (BC=64)
    const int total_tiles     = (n_kv + BC - 1) / BC;
    const int tiles_per_split = (split_k > 1) ? ((total_tiles + split_k - 1) / split_k) : total_tiles;
    const int t_start         = split * tiles_per_split;
    const int t_end           = min(t_start + tiles_per_split, total_tiles);

    const uint64_t kv_head_base = (uint64_t)h_kv * (uint64_t)(uint32_t)kv_slot.cap;

    for (int t = t_start; t < t_end; ++t) {
        const int k0 = t * BC;

        // Prevent LICM from hoisting 18+ lane-derived LDS addresses/constants across the tile loop
        int l_tmp = lane;
        asm volatile("" : "+v"(l_tmp));
        const int l_qlocal = l_tmp & 15;
        const int l_halfw  = l_tmp >> 4;

        // Interior causal fast-path check (100% CTA-uniform!)
        bool is_fast_interior = false;
        if (k0 + BC <= n_kv) {
            if constexpr (!IS_SWA) {
                is_fast_interior = (k0 + BC <= q_pos_min) &&
                                   (kv_slot.k_pos[k0] == k0) &&
                                   (kv_slot.k_pos[k0 + BC - 1] == k0 + BC - 1);
            } else {
                const int32_t kp_a = kv_slot.k_pos[k0];
                const int32_t kp_b = kv_slot.k_pos[k0 + BC - 1];
                is_fast_interior = (kp_a >= 0) && (kp_b == kp_a + BC - 1) &&
                                   (kp_b <= q_pos_min) &&
                                   ((q_pos_max - kp_a) < SWA_WINDOW);
            }
        }

        if (!is_fast_interior) {
            if (warp == 0) {
                int32_t mn = INT32_MAX;
                int32_t mx = -1;
                #pragma unroll
                for (int off = 0; off < BC; off += 32) {
                    const int idx = off + l_tmp;
                    if (idx < BC) {
                        const int32_t kp = (k0 + idx < n_kv) ? kv_slot.k_pos[k0 + idx] : -1;
                        tile_kpos[idx] = kp;
                        if (kp >= 0) {
                            mn = min(mn, kp);
                            mx = max(mx, kp);
                        }
                    }
                }
                mn = min(mn, shfl_xor_swz<16>(mn));
                mn = min(mn, shfl_xor_swz<8>(mn));
                mn = min(mn, shfl_xor_swz<4>(mn));
                mn = min(mn, shfl_xor_swz<2>(mn));
                mn = min(mn, shfl_xor_swz<1>(mn));
                mx = max(mx, shfl_xor_swz<16>(mx));
                mx = max(mx, shfl_xor_swz<8>(mx));
                mx = max(mx, shfl_xor_swz<4>(mx));
                mx = max(mx, shfl_xor_swz<2>(mx));
                mx = max(mx, shfl_xor_swz<1>(mx));
                if (l_tmp == 0) {
                    tile_min_kp = mn;
                    tile_max_kp = mx;
                    const bool skip = (mx < 0) || (mn > q_pos_max) ||
                                      (IS_SWA && ((q_pos_min - mx) >= SWA_WINDOW));
                    tile_skip = skip ? 1 : 0;
                }
            }
            nib_lds_fence_barrier();
            const int32_t do_skip = tile_skip;
            if (do_skip != 0) {
                // Ensure all warps finish reading tile_skip before warp 0 overwrites it on t+1!
                nib_lds_fence_barrier();
                continue;
            }
        }

        // Pin scalar tile_slot_base in SGPRs while preserving global addrspace(1) provenance on kv_slot pointers!
        uint64_t tile_slot_base = kv_head_base + (uint64_t)(uint32_t)k0;
        asm volatile("" : "+s"(tile_slot_base));

        // Cooperatively load/commit K (coalesced global loads with 32-bit offsets)
        // When N_CHUNKS == NWARPS (12 warps, BC=48), also issue V global loads here so K and V DRAM loads overlap,
        // and V remains in flight across K dequant, K LDS write, s_barrier, and the 24 K*Q^T WMMAs!
        constexpr uint32_t QS_STRIDE = (KV_FMT == KV_Q8_0) ? (uint32_t)D : (uint32_t)(D / 2);
        constexpr uint32_t D_STRIDE  = (KV_FMT == KV_Q4_1) ? 8u : 4u;
        const int8_t * tile_v_qs = kv_slot.v_qs + tile_slot_base * QS_STRIDE;
        const __half * tile_v_d  = kv_slot.v_d  + tile_slot_base * D_STRIDE;
        uint4    vq_pf0   = {0, 0, 0, 0};
        uint2    vq_pf0_4 = {0, 0};
        uint32_t dv_pf0   = 0;
        {
            const int8_t * tile_k_qs = kv_slot.k_qs + tile_slot_base * QS_STRIDE;
            const __half * tile_k_d  = kv_slot.k_d  + tile_slot_base * D_STRIDE;
            const int l_subk = l_tmp & 3;
            const int l_c16  = l_tmp >> 2;
            if (k0 + BC <= n_kv) {
                if constexpr (N_CHUNKS == NWARPS) {
                    const int rel0 = warp * 4 + l_subk;
                    const uint32_t d_off = (uint32_t)rel0 * 4 + (uint32_t)(l_c16 >> 1);
                    uint32_t kh2[8];
                    if constexpr (KV_FMT == KV_Q8_0) {
                        const uint32_t qs_off = (uint32_t)rel0 * D + (uint32_t)l_c16 * 16;
                        const __half dk0 = __ldg(tile_k_d + d_off);
                        const uint4  kq0 = *(const uint4 *)(tile_k_qs + qs_off);
                        dv_pf0 = (uint32_t)*(const uint16_t *)(tile_v_d + d_off);
                        vq_pf0 = *(const uint4 *)(tile_v_qs + qs_off);
                        dequant_16i8_to_8h2(kq0, dk0, kh2);
                    } else if constexpr (KV_FMT == KV_Q4_1) {
                        const uint32_t qs_off = (uint32_t)rel0 * (D / 2) + (uint32_t)l_c16 * 8;
                        const __half2 dm_k0 = __ldg(((const __half2 *)tile_k_d) + d_off);
                        const uint2   kq0   = *(const uint2 *)(tile_k_qs + qs_off);
                        dv_pf0   = *(const uint32_t *)(((const __half2 *)tile_v_d) + d_off);
                        vq_pf0_4 = *(const uint2 *)(tile_v_qs + qs_off);
                        dequant_16q4_1_to_8h2(kq0, dm_k0, kh2);
                    } else {
                        const uint32_t qs_off = (uint32_t)rel0 * (D / 2) + (uint32_t)l_c16 * 8;
                        const __half dk0 = __ldg(tile_k_d + d_off);
                        const uint2  kq0 = *(const uint2 *)(tile_k_qs + qs_off);
                        dv_pf0   = (uint32_t)*(const uint16_t *)(tile_v_d + d_off);
                        vq_pf0_4 = *(const uint2 *)(tile_v_qs + qs_off);
                        dequant_16q4_0_to_8h2(kq0, dk0, kh2);
                    }
                    *reinterpret_cast<uint4 *>(&smem.K[rel0][l_c16 * 16 + 0]) =
                        make_uint4(kh2[0], kh2[1], kh2[2], kh2[3]);
                    *reinterpret_cast<uint4 *>(&smem.K[rel0][l_c16 * 16 + 8]) =
                        make_uint4(kh2[4], kh2[5], kh2[6], kh2[7]);
                } else if constexpr (N_CHUNKS == 2 * NWARPS) {
                    const int rel0 = warp * 4 + l_subk;
                    const int rel1 = (warp + NWARPS) * 4 + l_subk;
                    uint32_t kh2[8];
                    load_dequant_16_kv<KV_FMT>(tile_k_qs, tile_k_d, rel0, l_c16, kh2);
                    *reinterpret_cast<uint4 *>(&smem.K[rel0][l_c16 * 16 + 0]) =
                        make_uint4(kh2[0], kh2[1], kh2[2], kh2[3]);
                    *reinterpret_cast<uint4 *>(&smem.K[rel0][l_c16 * 16 + 8]) =
                        make_uint4(kh2[4], kh2[5], kh2[6], kh2[7]);
                    load_dequant_16_kv<KV_FMT>(tile_k_qs, tile_k_d, rel1, l_c16, kh2);
                    *reinterpret_cast<uint4 *>(&smem.K[rel1][l_c16 * 16 + 0]) =
                        make_uint4(kh2[0], kh2[1], kh2[2], kh2[3]);
                    *reinterpret_cast<uint4 *>(&smem.K[rel1][l_c16 * 16 + 8]) =
                        make_uint4(kh2[4], kh2[5], kh2[6], kh2[7]);
                } else {
                    for (int chunk4 = warp; chunk4 < N_CHUNKS; chunk4 += NWARPS) {
                        const int rel_j = chunk4 * 4 + l_subk;
                        uint32_t kh2[8];
                        load_dequant_16_kv<KV_FMT>(tile_k_qs, tile_k_d, rel_j, l_c16, kh2);
                        *reinterpret_cast<uint4 *>(&smem.K[rel_j][l_c16 * 16 + 0]) =
                            make_uint4(kh2[0], kh2[1], kh2[2], kh2[3]);
                        *reinterpret_cast<uint4 *>(&smem.K[rel_j][l_c16 * 16 + 8]) =
                            make_uint4(kh2[4], kh2[5], kh2[6], kh2[7]);
                    }
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
        nib_lds_fence_barrier();

        // Hardware WMMA: S[nk] = K[nk*16 .. nk*16+15] * Q^T with 2-stage K double-buffering
        floatx8_t S[NK16];
        #pragma unroll
        for (int nk = 0; nk < NK16; ++nk) {
            S[nk] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        }

#if NIB_MMA_NATIVE
        halfx16_t kv_buf[2];
        kv_buf[0] = load_halfx16_b128(&smem.K[l_qlocal][0]);
#endif

        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            const int d_off = c * 16;
            halfx16_t Qc;
            if (c < 2) {
                Qc = load_halfx16_b128(&smem.Q_lo[warp][l_qlocal][(c ^ ((l_qlocal >> 1) & 1)) * 16]);
            } else {
                const int src_lane = (c < 5) ? l_qlocal : (l_qlocal + 16);
                const int reg_idx  = (c < 5) ? (c - 2) : (c - 5);
                union {
                    halfx16_t h16;
                    uint32_t  u[8];
                } q_src, q_bcast;
                q_src.h16 = Q_reg[reg_idx];
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    q_bcast.u[w] = __shfl(q_src.u[w], src_lane, 32);
                }
                Qc = q_bcast.h16;
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
            // gfx12 (include/nib_mma.h): S[nk] = K[nk*16 .. nk*16+15] x Q^T, computed per row
            // from the same LDS tile. Qc is this lane's B row (q = lane&15) either way.
            #pragma unroll
            for (int nk = 0; nk < NK16; ++nk) {
                nib_mma::mma_stride(&smem.K[nk * 16][d_off], STRIDE_K, Qc, S[nk], lane);
            }
#endif
        }
        pin_wmma_acc(S);
        nib_lds_fence_barrier();

        // Cooperatively dequantize & transpose V_T into LDS
        {
            const int l_subk = l_tmp & 3;
            const int l_c16  = l_tmp >> 2;
            if (k0 + BC <= n_kv) {
                if constexpr (N_CHUNKS == NWARPS) {
                    uint32_t vh2[8];
                    if constexpr (KV_FMT == KV_Q8_0) {
                        // Prevent LLVM from hoisting V dequant VALU before K*Q^T WMMA
                        asm volatile("" : "+v"(vq_pf0.x), "+v"(vq_pf0.y), "+v"(vq_pf0.z), "+v"(vq_pf0.w), "+v"(dv_pf0));
                        const __half dv_h = __builtin_bit_cast(__half, (uint16_t)dv_pf0);
                        dequant_16i8_to_8h2(vq_pf0, dv_h, vh2);
                    } else if constexpr (KV_FMT == KV_Q4_1) {
                        asm volatile("" : "+v"(vq_pf0_4.x), "+v"(vq_pf0_4.y), "+v"(dv_pf0));
                        dequant_16q4_1_to_8h2(vq_pf0_4, h2_of(dv_pf0), vh2);
                    } else {
                        asm volatile("" : "+v"(vq_pf0_4.x), "+v"(vq_pf0_4.y), "+v"(dv_pf0));
                        const __half dv_h = __builtin_bit_cast(__half, (uint16_t)dv_pf0);
                        dequant_16q4_0_to_8h2(vq_pf0_4, dv_h, vh2);
                    }
                    warp_transpose_4keys_write_v<STRIDE_V, SWIZ_V>(vh2, smem.V_T, warp * 4, l_tmp);
                } else {
                    for (int chunk4 = warp; chunk4 < N_CHUNKS; chunk4 += NWARPS) {
                        const int kk_base = chunk4 * 4;
                        const int rel_j = kk_base + l_subk;
                        uint32_t vh2[8];
                        load_dequant_16_kv<KV_FMT>(tile_v_qs, tile_v_d, rel_j, l_c16, vh2);
                        warp_transpose_4keys_write_v<STRIDE_V, SWIZ_V>(vh2, smem.V_T, kk_base, l_tmp);
                    }
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
        nib_lds_fence_barrier();

        float m_tile = m_q;
        const bool q_in = (qp_val < q_pos_lim);
        if (is_fast_interior) {
            const int32_t kp_base = IS_SWA ? kv_slot.k_pos[k0] : k0;
            const bool far_no_rel = ((q_pos_min - (kp_base + BC - 1)) >= N_REL_EXTENT);
            if (!far_no_rel) {
                const bool lane_has_rel = q_in && ((qp_val - (kp_base + BC - 1)) < N_REL_EXTENT);
                if (__any(lane_has_rel) && q_in) {
                    int32_t qi_rel = q_block + qt * 16 + l_qlocal; // == qp_val - pos_start
                    asm volatile("" : "+v"(qi_rel));
                    const int h_rel = h_kv * GQA_RATIO + g;
                    const float * rel_row = rel_table + ((int64_t)qi_rel * N_HEAD + h_rel) * N_REL_EXTENT;
                    #pragma unroll
                    for (int nk = 0; nk < NK16; ++nk) {
                        #pragma unroll
                        for (int idx = 0; idx < 8; ++idx) {
                            const int32_t dist = qp_val - (kp_base + nk * 16 + 2 * idx + l_halfw);
                            if (dist < N_REL_EXTENT) {
                                S[nk][idx] += rel_row[dist] * LOG2E;
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
                        const int32_t kp = tile_kpos[nk * 16 + 2 * idx + l_halfw];
                        if (kp >= 0 && kp <= qp_val && (!IS_SWA || (qp_val - kp) < SWA_WINDOW)) {
                            valid_bits[nk] |= (1u << idx);
                        }
                    }
                }
            }

            const bool lane_has_rel = q_in && (tile_min_kp <= tile_max_kp) &&
                                      (qp_val >= tile_min_kp) &&
                                      ((qp_val - tile_max_kp) < N_REL_EXTENT);
            if (__any(lane_has_rel)) {
                int32_t qi_rel = q_block + qt * 16 + l_qlocal; // == qp_val - pos_start
                asm volatile("" : "+v"(qi_rel));
                const int h_rel = h_kv * GQA_RATIO + g;
                const float * rel_row = q_in ? (rel_table + ((int64_t)qi_rel * N_HEAD + h_rel) * N_REL_EXTENT) : nullptr;
                #pragma unroll
                for (int nk = 0; nk < NK16; ++nk) {
                    #pragma unroll
                    for (int idx = 0; idx < 8; ++idx) {
                        if ((valid_bits[nk] >> idx) & 1u) {
                            const int32_t kp = tile_kpos[nk * 16 + 2 * idx + l_halfw];
                            const int32_t dist = qp_val - kp;
                            if (dist >= 0 && dist < N_REL_EXTENT) {
                                S[nk][idx] += rel_row[dist] * LOG2E;
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

        // P * V_T with 2-stage V_T LDS double-buffering (overlaps ds_load_b128 under 32-cycle WMMA!)
#if NIB_MMA_NATIVE
        kv_buf[0] = load_vt_halfx16<STRIDE_V, SWIZ_V>(smem.V_T, 0, 0, l_qlocal);
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
        // gfx12 (include/nib_mma.h): B row = P[q = lane&15][key 0..15] in ascending key order.
        // This lane holds the keys of parity l_halfw (S[nk][idx] is key 2*idx + l_halfw), so the
        // other parity comes from the partner lane (lane ^ 16) — 8 shuffles, no extra LDS.
        #pragma unroll
        for (int nk = 0; nk < NK16; ++nk) {
            halfx16_t Prow;
            {
                union { halfx16_t h16; uint32_t u[8]; } pk;
                #pragma unroll
                for (int t = 0; t < 8; ++t) {
                    const float own = S[nk][t];
                    const float oth = __shfl_xor(own, 16, 32);
                    // hw == 0: (key 2t, key 2t+1) = (own, partner); hw == 1: (partner, own)
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

        nib_lds_fence_barrier();
    }

    if (qp_val >= q_pos_lim) {
        return;
    }
    // Re-derive the lane-dependent epilogue indices from a fresh opaque lane copy instead of
    // carrying qi / half_w across the whole KV loop (clang 22 spilled both at the 6-warp cap).
    int l_epi = lane;
    asm volatile("" : "+v"(l_epi));
    const int half_w_e = l_epi >> 4;
    const int qi = q_block + qt * 16 + (l_epi & 15); // == qp_val - pos_start
    const int h  = h_kv * GQA_RATIO + g;

    if (split_k > 1) {
        const int64_t meta_idx = ((int64_t)qi * N_HEAD + h) * split_k + split;
        if (half_w_e == 0) {
            tmp_meta[meta_idx] = make_float2(m_q, l_q);
        }
        if (m_q > -INFINITY) {
            float * dst = tmp_o + meta_idx * D;
            #pragma unroll
            for (int c = 0; c < 8; ++c) {
                #pragma unroll
                for (int idx = 0; idx < 8; ++idx) {
                    dst[c * 16 + 2 * idx + half_w_e] = O_acc[c][idx];
                }
            }
        }
        return;
    }

    // Finalization (split_k == 1): fold sink + fused in-register FWHT-128 & sigmoid(qgate)
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

    // 7-stage in-register FWHT-128 directly in O_acc[8]:
    // Stage 0 (h = 1, bit 0 = half_w)
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int idx = 0; idx < 8; ++idx) {
            const float cur   = O_acc[c][idx];
            const float other = __shfl_xor(cur, 16, 32);
            O_acc[c][idx] = (half_w_e == 0) ? (cur + other) : (other - cur);
        }
    }
    // Stage 1 (h = 2, bit 1 = idx & 1)
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const float a = O_acc[c][i], b = O_acc[c][i + 1];
            O_acc[c][i] = a + b; O_acc[c][i + 1] = a - b;
        }
    }
    // Stage 2 (h = 4, bit 2 = (idx >> 1) & 1)
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
    // Stage 3 (h = 8, bit 3 = (idx >> 2) & 1)
    #pragma unroll
    for (int c = 0; c < 8; ++c) {
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const float a = O_acc[c][k], b = O_acc[c][k + 4];
            O_acc[c][k] = a + b; O_acc[c][k + 4] = a - b;
        }
    }
    // Stage 4 (h = 16, bit 4 = c & 1)
    #pragma unroll
    for (int c = 0; c < 8; c += 2) {
        #pragma unroll
        for (int idx = 0; idx < 8; ++idx) {
            const float a = O_acc[c][idx], b = O_acc[c + 1][idx];
            O_acc[c][idx] = a + b; O_acc[c + 1][idx] = a - b;
        }
    }
    // Stage 5 (h = 32, bit 5 = (c >> 1) & 1)
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
    // Stage 6 (h = 64, bit 6 = (c >> 2) & 1)
    #pragma unroll
    for (int k = 0; k < 4; ++k) {
        #pragma unroll
        for (int idx = 0; idx < 8; ++idx) {
            const float a = O_acc[k][idx], b = O_acc[k + 4][idx];
            O_acc[k][idx] = a + b; O_acc[k + 4][idx] = a - b;
        }
    }

    float * dst_ptr = out + ((size_t)qi * N_HEAD + h) * D;
    if (qgate_in != nullptr) {
        const float * gate_ptr = qgate_in + (size_t)qi * qgate_stride + h * D;
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            #pragma unroll
            for (int idx = 0; idx < 8; ++idx) {
                const int d = c * 16 + 2 * idx + half_w_e;
                const float g_val = gate_ptr[d];
                dst_ptr[d] = O_acc[c][idx] * (1.0f / (1.0f + fast_exp2f(-g_val * LOG2E)));
            }
        }
    } else {
        #pragma unroll
        for (int c = 0; c < 8; ++c) {
            #pragma unroll
            for (int idx = 0; idx < 8; ++idx) {
                dst_ptr[c * 16 + 2 * idx + half_w_e] = O_acc[c][idx];
            }
        }
    }
    (void)NK16;
#else
    (void)q; (void)kv_slot; (void)sinks_in; (void)sinks_stride; (void)rel_table;
    (void)qgate_in; (void)qgate_stride; (void)out; (void)tmp_o; (void)tmp_meta;
    (void)n_tokens; (void)pos_start; (void)n_kv; (void)split_k;
#endif // gfx1100..gfx1103
}

} // anonymous namespace

// ============================================================================
// Host Launcher
// ============================================================================
void launch_fa_relbias(
        const float * q,
        KVCacheSlot   kv_slot,
        const float * sinks_in, int sinks_stride,
        const float * rel_table,
        const float * qgate_in, int qgate_stride,
        float       * out,
        float       * tmp_o,
        float2      * tmp_meta,
        int           n_tokens,
        int           pos_start,
        const StepMeta * d_meta,
        hipStream_t   stream,
        int32_t     * fa_done) {
    if (n_tokens <= 0) {
        return;
    }

    const bool is_swa = kv_slot.is_swa;

    if (n_tokens > 1 && d_meta == nullptr) {
        const int n_kv = is_swa ? std::min(pos_start + n_tokens, kv_slot.cap) : (pos_start + n_tokens);
        constexpr int BC = 48;

        // One CTA per (48-token Q-block, KV head) = 64 CTAs per split-K slice, so k slices
        // are 64*k CTAs. Fill every CU with no tail wave:
        //   gfx1100 (96 CU): 6 -> 384 CTAs = 4 CTAs/CU, 3 -> 192 = 2 CTAs/CU
        //   gfx1200/1201 (64 CU): 6 -> 384 = 6 CTAs/CU, 3 -> 192 = 3 CTAs/CU
        // Either way both are exact multiples of the CU count; the deeper split is what the
        // 96-CU build was tuned for, so keep the same numbers and only re-label them.
        int split_k = 1;
        if (n_tokens <= MAX_UBATCH && tmp_o != nullptr && tmp_meta != nullptr) {
            if (!is_swa) {
                if (n_kv >= 4096) {
                    split_k = 6; // 64 * 6 = 384 CTAs = exact 4 CTAs/CU on 96 CU, 6/CU on 64 CU
                } else if (n_kv >= 384 && n_tokens >= 256) {
                    split_k = 3; // 64 * 3 = 192 CTAs = exact 2 CTAs/CU on 96 CU, 3/CU on 64 CU
                }
            } else if (n_kv >= 512 && n_tokens >= 256) {
                split_k = 3;
            }
        }

        auto launch_prefill = [&](auto kv_tag) {
            constexpr int KV_FMT = decltype(kv_tag)::value;
            if (is_swa) {
                constexpr int NWARPS_SWA = 6;
                constexpr int Br_SWA     = (NWARPS_SWA / GQA_RATIO) * 16;
                const dim3 grid((n_tokens + Br_SWA - 1) / Br_SWA, N_HEAD_KV, split_k);
                const dim3 block(32, NWARPS_SWA, 1);
                hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_prefill_gqa_wmma_kernel<NWARPS_SWA, BC, true, KV_FMT>),
                    grid, block, 0, stream,
                    q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                    out, tmp_o, tmp_meta, n_tokens, pos_start, n_kv, split_k);
            } else {
                constexpr int NWARPS_GLOB = 12;
                constexpr int Br_GLOB     = (NWARPS_GLOB / GQA_RATIO) * 16;
                const dim3 grid((n_tokens + Br_GLOB - 1) / Br_GLOB, N_HEAD_KV, split_k);
                const dim3 block(32, NWARPS_GLOB, 1);
                hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_prefill_gqa_wmma_kernel<NWARPS_GLOB, BC, false, KV_FMT>),
                    grid, block, 0, stream,
                    q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                    out, tmp_o, tmp_meta, n_tokens, pos_start, n_kv, split_k);
            }
        };
        if (kv_slot.kv_type == KV_Q4_1) {
            launch_prefill(std::integral_constant<int, KV_Q4_1>{});
        } else if (kv_slot.kv_type == KV_Q4_0) {
            launch_prefill(std::integral_constant<int, KV_Q4_0>{});
        } else {
            launch_prefill(std::integral_constant<int, KV_Q8_0>{});
        }

        if (split_k > 1) {
            const dim3 cgrid(N_HEAD / 4, n_tokens, 1);
            const dim3 cblock(32, 4, 1);
            hipLaunchKernelGGL(
                inkling_fa_decode_combine_kernel,
                cgrid, cblock, 0, stream,
                tmp_o, tmp_meta, sinks_in, sinks_stride, qgate_in, qgate_stride, out, split_k);
        }
        return;
    }

    // Static-graph decode path (or n_tokens == 1):
    constexpr int DEC_SWA_SPLITS  = 24; // 24 * 8 = 192 CTAs = 2 CTAs/CU on 96 CU, 3/CU on 64 CU + single-wave (<=32) combine!
    constexpr int DEC_GLOB_SPLITS = 48; // 48 * 8 = 384 CTAs = 4 CTAs/CU on 96 CU, 6/CU on 64 CU (zero tail wave!)

    int n_splits = is_swa ? DEC_SWA_SPLITS : DEC_GLOB_SPLITS;
    if (d_meta == nullptr) {
        const int n_kv = is_swa ? std::min(pos_start + n_tokens, kv_slot.cap) : (pos_start + n_tokens);
        if (n_kv < 256) {
            n_splits = 1;
        } else {
            n_splits = std::min(n_splits, (n_kv + 63) / 64);
        }
    }

    int32_t * effective_fa_done = (n_tokens == 1 && n_splits > 1) ? fa_done : nullptr;
    const dim3 grid(n_splits, N_HEAD_KV, n_tokens);
    const dim3 block(32, 4, 1);

    auto launch_decode = [&](auto kv_tag) {
        constexpr int KV_FMT = decltype(kv_tag)::value;
        if (is_swa) {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_kernel<GQA_RATIO, true, KV_FMT>),
                grid, block, 0, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, tmp_o, tmp_meta, n_tokens, pos_start, n_splits, d_meta, effective_fa_done);
        } else {
            hipLaunchKernelGGL(HIP_KERNEL_NAME(inkling_fa_decode_kernel<GQA_RATIO, false, KV_FMT>),
                grid, block, 0, stream,
                q, kv_slot, sinks_in, sinks_stride, rel_table, qgate_in, qgate_stride,
                out, tmp_o, tmp_meta, n_tokens, pos_start, n_splits, d_meta, effective_fa_done);
        }
    };
    if (kv_slot.kv_type == KV_Q4_1) {
        launch_decode(std::integral_constant<int, KV_Q4_1>{});
    } else if (kv_slot.kv_type == KV_Q4_0) {
        launch_decode(std::integral_constant<int, KV_Q4_0>{});
    } else {
        launch_decode(std::integral_constant<int, KV_Q8_0>{});
    }

    if (n_splits > 1 && effective_fa_done == nullptr) {
        const dim3 cgrid(N_HEAD / 4, n_tokens, 1);
        const dim3 cblock(32, 4, 1);
        hipLaunchKernelGGL(
            inkling_fa_decode_combine_kernel,
            cgrid, cblock, 0, stream,
            tmp_o, tmp_meta, sinks_in, sinks_stride, qgate_in, qgate_stride, out, n_splits);
    }
}

} // namespace inkling
