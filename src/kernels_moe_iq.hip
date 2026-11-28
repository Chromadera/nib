// ============================================================================
// kernels_moe_iq.hip — Wave32 Decode, Batched Verify, and Grouped FP16 WMMA
// Prefill for all 9 GSQ-RCO MoE Quantization Formats
// ============================================================================
// Zero modifications to any existing NVFP4 / Q8_0 .hip file.
// Formats supported:
//   - down:    Q2_0 (QK=64), IQ4_NL (QK=32)
//   - gate_up: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S (QK=256)
//   (all 9 are instantiated for both gate_up and down if ever paired arbitrarily,
//    or dispatched cleanly via switch on MoEQuantType)
// ============================================================================

#include "kernels_iq.h"
#include "iq_dequant_device.h"
#include "nib_arch.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace inkling {
namespace {

typedef int v4i __attribute__((ext_vector_type(4)));
using wmma_hfrag_t = nib_wmma_hfrag_t;
using wmma_ffrag_t = nib_float8_t;

[[noreturn]] void iq_moe_fail(const char * msg) {
    std::fprintf(stderr, "[kernels_moe_iq] CONTRACT VIOLATION: %s\n", msg);
    std::abort();
}

// ---------------------------------------------------------------------------
// Shared helpers: SITU, dot4, warp/group reductions, sconv epilogue
// ---------------------------------------------------------------------------
__device__ __forceinline__ float situ_scalar(float gate, float up) {
    const float act = SITU_BETA * tanhf(gate * (1.0f / SITU_BETA)) / (1.0f + expf(-gate));
    const float u   = SITU_LINEAR_BETA * tanhf(up * (1.0f / SITU_LINEAR_BETA));
    return act * u;
}

__device__ __forceinline__ float dot4_acc(float4 w, float4 x, float acc) {
    acc = fmaf(w.x, x.x, acc);
    acc = fmaf(w.y, x.y, acc);
    acc = fmaf(w.z, x.z, acc);
    acc = fmaf(w.w, x.w, acc);
    return acc;
}

template <int LPR>
__device__ __forceinline__ float group_reduce_sum(float v) {
#pragma unroll
    for (int off = LPR >> 1; off > 0; off >>= 1) {
        v += __shfl_xor(v, off, 32);
    }
    return v;
}

struct IqSconvEpi {
    const float * conv_w;
    float       * hist;
    const float * res;
    float       * raw_out;
};

__device__ __forceinline__ float sconv_post_res_one(const IqSconvEpi & e, int ch, float x) {
    if (e.raw_out != nullptr) {
        e.raw_out[ch] = x;
    }
    if (e.conv_w == nullptr) {
        return x;
    }
    constexpr float kScale = 0.44721359549995793928f; // 1 / sqrt(5)
    float * hp = e.hist + (size_t) ch * N_CONV_HIST;
    const float h0 = hp[0];
    const float h1 = hp[1];
    const float h2 = hp[2];
    const float * wp = e.conv_w + (size_t) ch * N_CONV;
    const float acc = x + h0 * wp[0] + h1 * wp[1] + h2 * wp[2] + x * wp[3];
    hp[0] = h1;
    hp[1] = h2;
    hp[2] = x;
    return (e.res != nullptr) ? fmaf(acc, kScale, e.res[ch]) : (acc * kScale);
}

__device__ __forceinline__ v4i pack_f4x2_to_h8(float4 w0, float4 w1) {
    const __half2 h01 = __floats2half2_rn(w0.x, w0.y);
    const __half2 h23 = __floats2half2_rn(w0.z, w0.w);
    const __half2 h45 = __floats2half2_rn(w1.x, w1.y);
    const __half2 h67 = __floats2half2_rn(w1.z, w1.w);
    return v4i{
        __builtin_bit_cast(int, h01),
        __builtin_bit_cast(int, h23),
        __builtin_bit_cast(int, h45),
        __builtin_bit_cast(int, h67),
    };
}

__device__ __forceinline__ wmma_hfrag_t lds_frag_xor(const _Float16 * row_ptr, int rlow, int kc, int half_w) {
    const v4i * d4 = reinterpret_cast<const v4i *>(row_ptr);
#if NIB_ARCH_RDNA4
    const v4i a = d4[(2 * kc + half_w) ^ rlow];
    wmma_hfrag_t r;
    __builtin_memcpy(&r, &a, 16);
    return r;
#else
    (void) half_w;
    const v4i a = d4[(2 * kc + 0) ^ rlow];
    const v4i b = d4[(2 * kc + 1) ^ rlow];
    wmma_hfrag_t r;
    __builtin_memcpy(&r, &a, 16);
    __builtin_memcpy(reinterpret_cast<char *>(&r) + 16, &b, 16);
    return r;
#endif
}

// ============================================================================
// 1. Single-Token Decode: Stage 1 (gate_up + SITU) & Stage 2 (down + wsum)
// ============================================================================

template <MoEQuantType Q>
__global__ void __launch_bounds__(128)
moe_iq_gate_up_situ_kernel(const uint8_t * __restrict__ p0,
                           const uint8_t * __restrict__ p1,
                           const __half  * __restrict__ d_in,
                           const float   * __restrict__ x_norm,
                           const int32_t * __restrict__ topk_ids,
                           float         * __restrict__ tmp_act) {
    using T = iq_dev::IqTraits<Q>;
    constexpr int K        = N_EMBD;            // 2304
    constexpr int NB_ROW   = K / T::QK;
    constexpr int NT       = 128;               // 4 warps
    constexpr int LPR      = 8;                 // 8 lanes per (gate, up) pair
    constexpr int PPB      = NT / LPR;          // 16 pairs per block
    constexpr int NSUB8    = K >> 3;            // 288 chunks of 8 elements

    __shared__ float4 s_x[K / 4];               // 576 float4 = 9 KB
    const float4 * x4 = reinterpret_cast<const float4 *>(x_norm);
    for (int i = threadIdx.x; i < K / 4; i += NT) {
        s_x[i] = x4[i];
    }
    __syncthreads();

    const int k    = blockIdx.y;
    const int expr = topk_ids[k];
    const bool eok = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex   = eok ? expr : 0;

    const int pair_in_blk = threadIdx.x >> 3;   // 0..15
    const int g           = threadIdx.x & 7;    // 0..7
    const int r           = blockIdx.x * PPB + pair_in_blk; // 0..1151

    const size_t blk_g = ((size_t) ex * N_GATE_UP_EXP + r) * NB_ROW;
    const size_t blk_u = blk_g + (size_t) N_FF_EXP * NB_ROW;

    const uint8_t * p0_g = p0 + blk_g * T::P0B;
    const uint8_t * p0_u = p0 + blk_u * T::P0B;
    const uint8_t * p1_g = (T::P1B > 0) ? (p1 + blk_g * T::P1B) : nullptr;
    const uint8_t * p1_u = (T::P1B > 0) ? (p1 + blk_u * T::P1B) : nullptr;
    const __half  * d_g  = T::HAS_D ? (d_in + blk_g) : nullptr;
    const __half  * d_u  = T::HAS_D ? (d_in + blk_u) : nullptr;

    float accg = 0.0f;
    float accu = 0.0f;

    if constexpr (T::QK == 256) {
        // 9 super-blocks x 8 32-elem sub-blocks: lane g owns sub-block ib32 = g (4 x 8 elems)
#pragma unroll 1
        for (int b = 0; b < NB_ROW; ++b) {
            const float dg = T::HAS_D ? __half2float(d_g[b]) : 0.0f;
            const float du = T::HAS_D ? __half2float(d_u[b]) : 0.0f;
            const uint8_t * gp0 = p0_g + (size_t) b * T::P0B;
            const uint8_t * up0 = p0_u + (size_t) b * T::P0B;
            const uint8_t * gp1 = (T::P1B > 0) ? (p1_g + (size_t) b * T::P1B) : nullptr;
            const uint8_t * up1 = (T::P1B > 0) ? (p1_u + (size_t) b * T::P1B) : nullptr;
#pragma unroll
            for (int l = 0; l < 4; ++l) {
                const int sub8 = g * 4 + l;
                const int k4   = (b * 256 + sub8 * 8) >> 2;
                const float4 x0 = s_x[k4 + 0];
                const float4 x1 = s_x[k4 + 1];
                float4 wg0, wg1, wu0, wu1;
                T::dequant8(gp0, gp1, dg, sub8, wg0, wg1);
                T::dequant8(up0, up1, du, sub8, wu0, wu1);
                accg = dot4_acc(wg0, x0, accg);
                accg = dot4_acc(wg1, x1, accg);
                accu = dot4_acc(wu0, x0, accu);
                accu = dot4_acc(wu1, x1, accu);
            }
        }
    } else {
        for (int s8 = g; s8 < NSUB8; s8 += LPR) {
            const int k0 = s8 << 3;
            const float4 x0 = s_x[2 * s8 + 0];
            const float4 x1 = s_x[2 * s8 + 1];
            float4 wg0, wg1, wu0, wu1;
            iq_dev::dequant_row_k8<Q>(p0_g, p1_g, d_g, k0, wg0, wg1);
            iq_dev::dequant_row_k8<Q>(p0_u, p1_u, d_u, k0, wu0, wu1);
            accg = dot4_acc(wg0, x0, accg);
            accg = dot4_acc(wg1, x1, accg);
            accu = dot4_acc(wu0, x0, accu);
            accu = dot4_acc(wu1, x1, accu);
        }
    }

    accg = group_reduce_sum<LPR>(accg);
    accu = group_reduce_sum<LPR>(accu);
    if (g == 0) {
        tmp_act[k * N_FF_EXP + r] = eok ? situ_scalar(accg, accu) : 0.0f;
    }
}

template <MoEQuantType Q>
__global__ void __launch_bounds__(256)
moe_iq_down_wsum_kernel(const uint8_t * __restrict__ p0,
                        const uint8_t * __restrict__ p1,
                        const __half  * __restrict__ d_in,
                        const float   * __restrict__ tmp_act,
                        const int32_t * __restrict__ topk_ids,
                        const float   * __restrict__ topk_w,
                        float         * __restrict__ out,
                        const IqSconvEpi epi) {
    // 8 warps (256 threads): warp k owns expert k (0..7).
    // Grid = 288 blocks -> 8 rows per block (4 steps of 2 rows/warp; 16 lanes per row).
    using T = iq_dev::IqTraits<Q>;
    constexpr int K      = N_FF_EXP;        // 1152
    constexpr int NB_ROW = K / T::QK;       // 18 for Q2_0, 36 for IQ4_NL
    constexpr int NSUB8  = K >> 3;          // 144 chunks of 8 elements
    constexpr int RPB    = 8;               // rows per block -> 2304 / 8 = 288 blocks

    __shared__ float4 s_act[TOP_K][K / 4];  // 8 * 288 * 16 B = 36,864 B
    __shared__ float  s_part[TOP_K][2];     // 8 experts x 2 rows per step

    const int warp = threadIdx.x >> 5;      // 0..7 = expert slot k
    const int lane = threadIdx.x & 31;
    const int rloc = lane >> 4;             // 0 or 1 (2 rows per warp)
    const int g    = lane & 15;             // 0..15 (16 lanes per row -> 144/16 = 9 chunks/lane)

    const float4 * act4 = reinterpret_cast<const float4 *>(tmp_act + (size_t) warp * K);
    for (int i = lane; i < K / 4; i += 32) {
        s_act[warp][i] = act4[i];
    }
    __syncthreads();

    const int expr = topk_ids[warp];
    const bool eok = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex   = eok ? expr : 0;
    const float w_k = eok ? topk_w[warp] : 0.0f;

    const int row_base = blockIdx.x * RPB;
    const float4 * my_act = s_act[warp];

#pragma unroll 1
    for (int step = 0; step < RPB; step += 2) {
        const int row = row_base + step + rloc;
        const size_t blk_row = ((size_t) ex * N_EMBD + row) * NB_ROW;
        const uint8_t * p0_row = p0 + blk_row * T::P0B;
        const uint8_t * p1_row = (T::P1B > 0) ? (p1 + blk_row * T::P1B) : nullptr;
        const __half  * d_row  = T::HAS_D ? (d_in + blk_row) : nullptr;

        float acc = 0.0f;
#pragma unroll 1
        for (int s8 = g; s8 < NSUB8; s8 += 16) {
            float4 w0, w1;
            iq_dev::dequant_row_k8<Q>(p0_row, p1_row, d_row, s8 << 3, w0, w1);
            acc = dot4_acc(w0, my_act[2 * s8 + 0], acc);
            acc = dot4_acc(w1, my_act[2 * s8 + 1], acc);
        }
        acc = group_reduce_sum<16>(acc);
        if (g == 0) {
            s_part[warp][rloc] = acc * w_k;
        }
        __syncthreads();

        if (warp == 0 && lane < 2) {
            const int out_row = row_base + step + lane;
            const float s0 = s_part[0][lane], s1 = s_part[1][lane];
            const float s2 = s_part[2][lane], s3 = s_part[3][lane];
            const float s4 = s_part[4][lane], s5 = s_part[5][lane];
            const float s6 = s_part[6][lane], s7 = s_part[7][lane];
            const float sum = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
            out[out_row] = sconv_post_res_one(epi, out_row, sum);
        }
        __syncthreads();
    }
}

// ============================================================================
// 2. Batched Verify / Multi-Slot Decode (unique-expert deduplicated)
// ============================================================================

template <MoEQuantType Q>
__global__ void __launch_bounds__(128)
verify_moe_gu_iq_kernel(const uint8_t * __restrict__ p0,
                        const uint8_t * __restrict__ p1,
                        const __half  * __restrict__ d_in,
                        const float   * __restrict__ x_norm, int xstride,
                        const int32_t * __restrict__ unique_exp,
                        const int32_t * __restrict__ exp_tok_slot,
                        const int32_t * __restrict__ exp_tok_cnt,
                        const int32_t * __restrict__ exp_tok_list,
                        const int32_t * __restrict__ d_num_unique,
                        float         * __restrict__ act) {
    const int u = blockIdx.y;
    if (u >= *d_num_unique) return;

    using T = iq_dev::IqTraits<Q>;
    constexpr int K      = N_EMBD;
    constexpr int NB_ROW = K / T::QK;
    constexpr int LPR    = 8;
    constexpr int PPB    = 128 / LPR; // 16 pairs/block -> 72 blocks
    constexpr int NSUB8  = K >> 3;

    const int ex  = unique_exp[u];
    const int cnt = exp_tok_cnt[u];
    int toks[8];
    int slots[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t < cnt) {
            const int r = exp_tok_list[u * 8 + t];
            toks[t]  = r;
            slots[t] = exp_tok_slot[u * 8 + r];
        } else {
            toks[t]  = 0;
            slots[t] = 0;
        }
    }

    const int pair_in_blk = threadIdx.x >> 3;
    const int g           = threadIdx.x & 7;
    const int row         = blockIdx.x * PPB + pair_in_blk;

    const size_t blk_g = ((size_t) ex * N_GATE_UP_EXP + row) * NB_ROW;
    const size_t blk_u = blk_g + (size_t) N_FF_EXP * NB_ROW;
    const uint8_t * p0_g = p0 + blk_g * T::P0B;
    const uint8_t * p0_u = p0 + blk_u * T::P0B;
    const uint8_t * p1_g = (T::P1B > 0) ? (p1 + blk_g * T::P1B) : nullptr;
    const uint8_t * p1_u = (T::P1B > 0) ? (p1 + blk_u * T::P1B) : nullptr;
    const __half  * d_g  = T::HAS_D ? (d_in + blk_g) : nullptr;
    const __half  * d_u  = T::HAS_D ? (d_in + blk_u) : nullptr;

    float accg[8] = {};
    float accu[8] = {};

    for (int s8 = g; s8 < NSUB8; s8 += LPR) {
        const int k0 = s8 << 3;
        float4 wg0, wg1, wu0, wu1;
        iq_dev::dequant_row_k8<Q>(p0_g, p1_g, d_g, k0, wg0, wg1);
        iq_dev::dequant_row_k8<Q>(p0_u, p1_u, d_u, k0, wu0, wu1);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= cnt) break;
            const float4 * x4 = reinterpret_cast<const float4 *>(x_norm + (size_t) toks[t] * xstride + k0);
            const float4 x0 = x4[0];
            const float4 x1 = x4[1];
            accg[t] = dot4_acc(wg0, x0, accg[t]);
            accg[t] = dot4_acc(wg1, x1, accg[t]);
            accu[t] = dot4_acc(wu0, x0, accu[t]);
            accu[t] = dot4_acc(wu1, x1, accu[t]);
        }
    }

#pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t >= cnt) break;
        const float sg = group_reduce_sum<LPR>(accg[t]);
        const float su = group_reduce_sum<LPR>(accu[t]);
        if (g == 0) {
            act[(size_t) (toks[t] * TOP_K + slots[t]) * N_FF_EXP + row] = situ_scalar(sg, su);
        }
    }
}

template <MoEQuantType Q>
__global__ void __launch_bounds__(128)
verify_moe_dn_iq_kernel(const uint8_t * __restrict__ p0,
                        const uint8_t * __restrict__ p1,
                        const __half  * __restrict__ d_in,
                        const float   * __restrict__ act,
                        const int32_t * __restrict__ unique_exp,
                        const int32_t * __restrict__ exp_tok_slot,
                        const float   * __restrict__ exp_tok_w,
                        const int32_t * __restrict__ exp_tok_cnt,
                        const int32_t * __restrict__ exp_tok_list,
                        const int32_t * __restrict__ d_num_unique,
                        float         * __restrict__ buf_slot) {
    const int u = blockIdx.y;
    if (u >= *d_num_unique) return;

    using T = iq_dev::IqTraits<Q>;
    constexpr int K      = N_FF_EXP;
    constexpr int NB_ROW = K / T::QK;
    constexpr int LPR    = 8;
    constexpr int RPB    = 128 / LPR; // 16 rows/block -> 144 blocks
    constexpr int NSUB8  = K >> 3;

    const int ex  = unique_exp[u];
    const int cnt = exp_tok_cnt[u];
    int   toks[8];
    int   slots[8];
    float ws[8];
#pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t < cnt) {
            const int r = exp_tok_list[u * 8 + t];
            toks[t]  = r;
            slots[t] = exp_tok_slot[u * 8 + r];
            ws[t]    = exp_tok_w[u * 8 + r];
        } else {
            toks[t]  = 0;
            slots[t] = 0;
            ws[t]    = 0.0f;
        }
    }

    const int row_in_blk = threadIdx.x >> 3;
    const int g          = threadIdx.x & 7;
    const int row        = blockIdx.x * RPB + row_in_blk;

    const size_t blk_row = ((size_t) ex * N_EMBD + row) * NB_ROW;
    const uint8_t * p0_row = p0 + blk_row * T::P0B;
    const uint8_t * p1_row = (T::P1B > 0) ? (p1 + blk_row * T::P1B) : nullptr;
    const __half  * d_row  = T::HAS_D ? (d_in + blk_row) : nullptr;

    float acc[8] = {};
    for (int s8 = g; s8 < NSUB8; s8 += LPR) {
        const int k0 = s8 << 3;
        float4 w0, w1;
        iq_dev::dequant_row_k8<Q>(p0_row, p1_row, d_row, k0, w0, w1);
#pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= cnt) break;
            const float4 * a4 = reinterpret_cast<const float4 *>(
                act + (size_t) (toks[t] * TOP_K + slots[t]) * N_FF_EXP + k0);
            acc[t] = dot4_acc(w0, a4[0], acc[t]);
            acc[t] = dot4_acc(w1, a4[1], acc[t]);
        }
    }

#pragma unroll
    for (int t = 0; t < 8; ++t) {
        if (t >= cnt) break;
        const float s = group_reduce_sum<LPR>(acc[t]);
        if (g == 0) {
            buf_slot[(size_t) (toks[t] * TOP_K + slots[t]) * N_EMBD + row] = s * ws[t];
        }
    }
}

// ============================================================================
// 3. Grouped FP16 WMMA Prefill (RDNA3 + RDNA4, supports M <= 1024)
// ============================================================================
constexpr int IQ_PREFILL_MAX_M = 1024;
constexpr int IQ_ROUTE_THREADS = 1024;
constexpr int IQ_MASK_WORDS    = IQ_PREFILL_MAX_M / 32; // 32

__global__ void __launch_bounds__(IQ_ROUTE_THREADS)
iq_moe_route_kernel(const int32_t * __restrict__ topk_ids, const float * __restrict__ topk_w, int M, int TM,
                    int32_t * __restrict__ exp_counts, int32_t * __restrict__ exp_offsets,
                    int32_t * __restrict__ exp_cursors, int32_t * __restrict__ perm_tok,
                    float * __restrict__ perm_w, int32_t * __restrict__ tile_exp,
                    int32_t * __restrict__ tile_m0, int32_t * __restrict__ d_num_tiles,
                    int32_t * __restrict__ tok_slots) {
    __shared__ uint32_t s_mask[N_EXPERT][IQ_MASK_WORDS];
    __shared__ int32_t  s_pref[N_EXPERT][IQ_MASK_WORDS];
    __shared__ int32_t  s_cnt[N_EXPERT];
    __shared__ int32_t  s_off[N_EXPERT + 1];
    __shared__ int32_t  s_toff[N_EXPERT + 1];

    const int tid   = threadIdx.x;
    const int npair = M * TOP_K;
    const int nw    = (M + 31) >> 5;

    for (int i = tid; i < N_EXPERT * IQ_MASK_WORDS; i += IQ_ROUTE_THREADS) {
        (&s_mask[0][0])[i] = 0u;
    }
    __syncthreads();

    for (int p = tid; p < npair; p += IQ_ROUTE_THREADS) {
        const int e = topk_ids[p];
        if ((unsigned) e < (unsigned) N_EXPERT) {
            const int tok = p / TOP_K;
            atomicOr(&s_mask[e][tok >> 5], 1u << (tok & 31));
        }
    }
    __syncthreads();

    if (tid < N_EXPERT) {
        int run = 0;
        for (int w = 0; w < nw; ++w) {
            s_pref[tid][w] = run;
            run += __popc(s_mask[tid][w]);
        }
        s_cnt[tid] = run;
    }
    __syncthreads();

    if (tid == 0) {
        int off = 0, toff = 0;
        for (int e = 0; e < N_EXPERT; ++e) {
            s_off[e]  = off;
            s_toff[e] = toff;
            off  += s_cnt[e];
            toff += (s_cnt[e] + TM - 1) / TM;
        }
        s_off[N_EXPERT]  = off;
        s_toff[N_EXPERT] = toff;
        *d_num_tiles = toff;
    }
    __syncthreads();

    if (tid <= N_EXPERT) exp_offsets[tid] = s_off[tid];
    if (tid < N_EXPERT) {
        exp_counts[tid]  = s_cnt[tid];
        exp_cursors[tid] = s_cnt[tid];
        const int nt = (s_cnt[tid] + TM - 1) / TM;
        for (int j = 0; j < nt; ++j) {
            tile_exp[s_toff[tid] + j] = tid;
            tile_m0[s_toff[tid] + j]  = j * TM;
        }
    }

    for (int p = tid; p < npair; p += IQ_ROUTE_THREADS) {
        const int e = topk_ids[p];
        if ((unsigned) e >= (unsigned) N_EXPERT) {
            tok_slots[p] = -1;
            continue;
        }
        const int tok = p / TOP_K;
        const int k   = p - tok * TOP_K;
        bool first = true;
        float w = 0.0f;
        for (int kk = 0; kk < TOP_K; ++kk) {
            if (topk_ids[tok * TOP_K + kk] == e) {
                if (kk < k) first = false;
                w += topk_w[tok * TOP_K + kk];
            }
        }
        if (!first) {
            tok_slots[p] = -1;
            continue;
        }
        const int wi   = tok >> 5;
        const int rank = s_pref[e][wi] + __popc(s_mask[e][wi] & ((1u << (tok & 31)) - 1u));
        const int slot = s_off[e] + rank;
        perm_tok[slot] = tok;
        perm_w[slot]   = w;
        tok_slots[p]   = slot;
    }
}

__global__ void __launch_bounds__(256)
iq_x_to_f16_kernel(const float * __restrict__ X, _Float16 * __restrict__ Xh) {
    const int m = blockIdx.x;
    const float4 * src = reinterpret_cast<const float4 *>(X + (size_t) m * N_EMBD);
    uint2 * dst = reinterpret_cast<uint2 *>(Xh + (size_t) m * N_EMBD);
    for (int i = threadIdx.x; i < N_EMBD / 4; i += 256) {
        const float4 v = src[i];
        const __half2 h0 = __floats2half2_rn(v.x, v.y);
        const __half2 h1 = __floats2half2_rn(v.z, v.w);
        dst[i] = make_uint2(__builtin_bit_cast(uint32_t, h0), __builtin_bit_cast(uint32_t, h1));
    }
}

__device__ __forceinline__ float4 iq_load_tok_sum_c4(
    const int32_t * tok_slots, const float * perm_down, int t, int c4) {
    const int32_t * ts = tok_slots + (size_t) t * TOP_K;
    const float4 * pd4 = reinterpret_cast<const float4 *>(perm_down);
    float4 s = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        const int slot = ts[k];
        if (slot >= 0) {
            const float4 v = pd4[(size_t) slot * (N_EMBD / 4) + c4];
            s.x += v.x; s.y += v.y; s.z += v.z; s.w += v.w;
        }
    }
    return s;
}

__global__ void __launch_bounds__(256)
iq_moe_combine_kernel(const int32_t * __restrict__ tok_slots,
                      const float   * __restrict__ perm_down,
                      float         * __restrict__ Y_out) {
    const int t = blockIdx.x;
    float4 * y4 = reinterpret_cast<float4 *>(Y_out + (size_t) t * N_EMBD);
    for (int c4 = threadIdx.x; c4 < N_EMBD / 4; c4 += 256) {
        y4[c4] = iq_load_tok_sum_c4(tok_slots, perm_down, t, c4);
    }
}

__device__ __forceinline__ float iq_sconv_step_one(
    float x, float & h0, float & h1, float & h2, const float4 w, bool has_res, float res) {
    constexpr float kScale = 0.44721359549995793928f; // 1 / sqrt(5)
    const float acc = x + h0 * w.x + h1 * w.y + h2 * w.z + x * w.w;
    h0 = h1; h1 = h2; h2 = x;
    return has_res ? fmaf(acc, kScale, res) : (acc * kScale);
}

constexpr int IQ_COMBINE_SCONV_THREADS = 64;
constexpr int IQ_COMBINE_SCONV_TCHUNK  = 32;
constexpr int IQ_COMBINE_SCONV_BLOCKS  = (N_EMBD / 4) / IQ_COMBINE_SCONV_THREADS;

__global__ void __launch_bounds__(IQ_COMBINE_SCONV_THREADS)
iq_moe_combine_sconv_kernel(const int32_t * __restrict__ tok_slots,
                            const float   * __restrict__ perm_down,
                            const float   * __restrict__ conv_w,
                            float         * __restrict__ sconv_h,
                            const float   * __restrict__ res_add,
                            float         * __restrict__ Y_out,
                            int M) {
    const int c4 = blockIdx.x * IQ_COMBINE_SCONV_THREADS + threadIdx.x;
    const int t_blk = blockIdx.y;
    const int t0 = t_blk * IQ_COMBINE_SCONV_TCHUNK;
    const int t1 = (t0 + IQ_COMBINE_SCONV_TCHUNK < M) ? (t0 + IQ_COMBINE_SCONV_TCHUNK) : M;

    const float4 * w4 = reinterpret_cast<const float4 *>(conv_w + (size_t) c4 * 4 * N_CONV);
    const float4 w0 = w4[0], w1 = w4[1], w2 = w4[2], w3 = w4[3];
    float * h_base = sconv_h + (size_t) c4 * 4 * N_CONV_HIST;

    float4 h0, h1, h2;
    if (t0 == 0) {
        h0 = make_float4(h_base[0], h_base[3], h_base[6], h_base[9]);
        h1 = make_float4(h_base[1], h_base[4], h_base[7], h_base[10]);
        h2 = make_float4(h_base[2], h_base[5], h_base[8], h_base[11]);
    } else {
        h0 = (t0 >= 3) ? iq_load_tok_sum_c4(tok_slots, perm_down, t0 - 3, c4)
                       : make_float4(h_base[0 + (t0 == 2 ? 2 : 1)], h_base[3 + (t0 == 2 ? 2 : 1)],
                                     h_base[6 + (t0 == 2 ? 2 : 1)], h_base[9 + (t0 == 2 ? 2 : 1)]);
        h1 = (t0 >= 2) ? iq_load_tok_sum_c4(tok_slots, perm_down, t0 - 2, c4)
                       : make_float4(h_base[2], h_base[5], h_base[8], h_base[11]);
        h2 = iq_load_tok_sum_c4(tok_slots, perm_down, t0 - 1, c4);
    }

    const bool has_res = (res_add != nullptr);
    const float4 * r4_ptr = reinterpret_cast<const float4 *>(res_add);
    float4 * y4_ptr = reinterpret_cast<float4 *>(Y_out);

    for (int t = t0; t < t1; ++t) {
        const float4 x = iq_load_tok_sum_c4(tok_slots, perm_down, t, c4);
        const float4 r = has_res ? r4_ptr[(size_t) t * (N_EMBD / 4) + c4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 out;
        out.x = iq_sconv_step_one(x.x, h0.x, h1.x, h2.x, w0, has_res, r.x);
        out.y = iq_sconv_step_one(x.y, h0.y, h1.y, h2.y, w1, has_res, r.y);
        out.z = iq_sconv_step_one(x.z, h0.z, h1.z, h2.z, w2, has_res, r.z);
        out.w = iq_sconv_step_one(x.w, h0.w, h1.w, h2.w, w3, has_res, r.w);
        y4_ptr[(size_t) t * (N_EMBD / 4) + c4] = out;
    }

    if (t_blk == 0) {
        if (t1 < M) {
            h0 = (M >= 3) ? iq_load_tok_sum_c4(tok_slots, perm_down, M - 3, c4)
                          : ((M == 2) ? make_float4(h_base[2], h_base[5], h_base[8], h_base[11])
                                      : make_float4(h_base[1], h_base[4], h_base[7], h_base[10]));
            h1 = (M >= 2) ? iq_load_tok_sum_c4(tok_slots, perm_down, M - 2, c4)
                          : make_float4(h_base[2], h_base[5], h_base[8], h_base[11]);
            h2 = iq_load_tok_sum_c4(tok_slots, perm_down, M - 1, c4);
        }
        h_base[0]  = h0.x; h_base[1]  = h1.x; h_base[2]  = h2.x;
        h_base[3]  = h0.y; h_base[4]  = h1.y; h_base[5]  = h2.y;
        h_base[6]  = h0.z; h_base[7]  = h1.z; h_base[8]  = h2.z;
        h_base[9]  = h0.w; h_base[10] = h1.w; h_base[11] = h2.w;
    }
}

// Grouped FP16 WMMA Stage 1 (gate_up + SITU): BM=64, BN=128 (64 gate + 64 up), BK=64, 8 warps (4x2)
template <MoEQuantType Q>
__global__ void __launch_bounds__(256)
iq_moe_gate_up_wmma_kernel(const uint8_t * __restrict__ p0,
                           const uint8_t * __restrict__ p1,
                           const __half  * __restrict__ d_in,
                           const _Float16 * __restrict__ Xh,
                           const int32_t * __restrict__ exp_offsets,
                           const int32_t * __restrict__ perm_tok,
                           const int32_t * __restrict__ tile_exp,
                           const int32_t * __restrict__ tile_m0,
                           const int32_t * __restrict__ d_num_tiles,
                           _Float16      * __restrict__ perm_act) {
    const int tile_id = blockIdx.y;
    if (tile_id >= *d_num_tiles) return;

    using T = iq_dev::IqTraits<Q>;
    constexpr int BM      = 64;
    constexpr int GOUT    = 64;  // 64 gate + 64 up = 128 B rows
    constexpr int BN      = 128;
    constexpr int BK      = 64;
    constexpr int NB_ROW  = N_EMBD / T::QK;

    const int ex    = tile_exp[tile_id];
    const int off_e = exp_offsets[ex];
    const int cnt_e = exp_offsets[ex + 1] - off_e;
    const int m0    = tile_m0[tile_id];
    const int n0    = blockIdx.x * GOUT; // 0..1151 step 64

    __shared__ _Float16 s_a[BM * BK];   // 8 KB
    __shared__ _Float16 s_b[BN * BK];   // 16 KB (rows 0..63 = gate, 64..127 = up)
    __shared__ int32_t  s_tok[BM];

    if (threadIdx.x < BM) {
        const int li = m0 + threadIdx.x;
        s_tok[threadIdx.x] = (li < cnt_e) ? perm_tok[off_e + li] : -1;
    }
    __syncthreads();

    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int wm     = warp >> 1;       // 0..3 -> 16 M rows per warp
    const int wn     = warp & 1;        // 0..1 -> 32 gate + 32 up cols per warp
    const int l16    = lane & 15;
    const int half_w = lane >> 4;
    const int rlow   = l16 & 7;

    wmma_ffrag_t acc_g[2] = {};
    wmma_ffrag_t acc_u[2] = {};

    const size_t exp_blk_base = (size_t) ex * N_GATE_UP_EXP * NB_ROW;

    // A load: 64 rows x 8 units (16B each) = 512 units / 256 threads = 2 units/thread
    const int a_row0 = threadIdx.x >> 3;           // 0..31
    const int a_u0   = threadIdx.x & 7;            // 0..7

    // B load: 128 rows x 8 units (8 weights each) = 1024 units / 256 threads = 4 units/thread
    const int b_row0 = threadIdx.x >> 3;           // 0..31
    const int b_u0   = threadIdx.x & 7;            // 0..7

#pragma unroll 1
    for (int k0 = 0; k0 < N_EMBD; k0 += BK) {
        // Load A (2 rows per thread)
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int ar  = a_row0 + p * 32;
            const int tok = s_tok[ar];
            v4i val = v4i{0, 0, 0, 0};
            if (tok >= 0) {
                val = *reinterpret_cast<const v4i *>(Xh + (size_t) tok * N_EMBD + k0 + a_u0 * 8);
            }
            reinterpret_cast<v4i *>(&s_a[ar * BK])[a_u0 ^ (ar & 7)] = val;
        }

        // Load + dequant B (4 rows per thread: p=0,1 -> gate rows; p=2,3 -> up rows)
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int br = b_row0 + p * 32; // 0..127
            const int w_row = (br < GOUT) ? (n0 + br) : (N_FF_EXP + n0 + (br - GOUT));
            const size_t rb = exp_blk_base + (size_t) w_row * NB_ROW;
            const uint8_t * rp0 = p0 + rb * T::P0B;
            const uint8_t * rp1 = (T::P1B > 0) ? (p1 + rb * T::P1B) : nullptr;
            const __half  * rd  = T::HAS_D ? (d_in + rb) : nullptr;
            float4 w0, w1;
            iq_dev::dequant_row_k8<Q>(rp0, rp1, rd, k0 + b_u0 * 8, w0, w1);
            reinterpret_cast<v4i *>(&s_b[br * BK])[b_u0 ^ (br & 7)] = pack_f4x2_to_h8(w0, w1);
        }
        __syncthreads();

#pragma unroll
        for (int kc = 0; kc < BK / 16; ++kc) {
            const wmma_hfrag_t af = lds_frag_xor(&s_a[(wm * 16 + l16) * BK], rlow, kc, half_w);
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const int br_g = wn * 32 + j * 16 + l16;
                const wmma_hfrag_t bfg = lds_frag_xor(&s_b[br_g * BK], rlow, kc, half_w);
                const wmma_hfrag_t bfu = lds_frag_xor(&s_b[(GOUT + br_g) * BK], rlow, kc, half_w);
                acc_g[j] = nib_wmma_f32_16x16x16_f16(af, bfg, acc_g[j]);
                acc_u[j] = nib_wmma_f32_16x16x16_f16(af, bfu, acc_u[j]);
            }
        }
        __syncthreads();
    }

    // Apply SITU and store to perm_act[off_e + m, n] as _Float16
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const int col = n0 + wn * 32 + j * 16 + l16;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int mr = m0 + wm * 16 + nib_wmma_c_row(half_w, v);
            if (mr < cnt_e) {
                perm_act[(size_t) (off_e + mr) * N_FF_EXP + col] =
                    static_cast<_Float16>(situ_scalar(acc_g[j][v], acc_u[j][v]));
            }
        }
    }
}

// Grouped FP16 WMMA Stage 2 (down * perm_w): BM=64, BN=128, BK=64, 8 warps (4x2)
template <MoEQuantType Q>
__global__ void __launch_bounds__(256)
iq_moe_down_wmma_kernel(const uint8_t * __restrict__ p0,
                        const uint8_t * __restrict__ p1,
                        const __half  * __restrict__ d_in,
                        const _Float16 * __restrict__ perm_act,
                        const int32_t * __restrict__ exp_offsets,
                        const float   * __restrict__ perm_w,
                        const int32_t * __restrict__ tile_exp,
                        const int32_t * __restrict__ tile_m0,
                        const int32_t * __restrict__ d_num_tiles,
                        float         * __restrict__ perm_down) {
    const int tile_id = blockIdx.y;
    if (tile_id >= *d_num_tiles) return;

    using T = iq_dev::IqTraits<Q>;
    constexpr int BM      = 64;
    constexpr int BN      = 128;
    constexpr int BK      = 64;
    constexpr int NB_ROW  = N_FF_EXP / T::QK;

    const int ex    = tile_exp[tile_id];
    const int off_e = exp_offsets[ex];
    const int cnt_e = exp_offsets[ex + 1] - off_e;
    const int m0    = tile_m0[tile_id];
    const int n0    = blockIdx.x * BN; // 0..2303 step 128

    __shared__ _Float16 s_a[BM * BK]; // 8 KB
    __shared__ _Float16 s_b[BN * BK]; // 16 KB
    __shared__ float    s_w[BM];

    if (threadIdx.x < BM) {
        const int li = m0 + threadIdx.x;
        s_w[threadIdx.x] = (li < cnt_e) ? perm_w[off_e + li] : 0.0f;
    }
    __syncthreads();

    const int warp   = threadIdx.x >> 5;
    const int lane   = threadIdx.x & 31;
    const int wm     = warp >> 1;      // 0..3 -> 16 M rows
    const int wn     = warp & 1;       // 0..1 -> 64 N cols (4 x 16)
    const int l16    = lane & 15;
    const int half_w = lane >> 4;
    const int rlow   = l16 & 7;

    wmma_ffrag_t acc[4] = {};

    const size_t exp_blk_base = (size_t) ex * N_EMBD * NB_ROW;
    const int a_row0 = threadIdx.x >> 3;
    const int a_u0   = threadIdx.x & 7;
    const int b_row0 = threadIdx.x >> 3;
    const int b_u0   = threadIdx.x & 7;

#pragma unroll 1
    for (int k0 = 0; k0 < N_FF_EXP; k0 += BK) {
#pragma unroll
        for (int p = 0; p < 2; ++p) {
            const int ar = a_row0 + p * 32;
            const int li = m0 + ar;
            v4i val = v4i{0, 0, 0, 0};
            if (li < cnt_e) {
                val = *reinterpret_cast<const v4i *>(perm_act + (size_t) (off_e + li) * N_FF_EXP + k0 + a_u0 * 8);
            }
            reinterpret_cast<v4i *>(&s_a[ar * BK])[a_u0 ^ (ar & 7)] = val;
        }
#pragma unroll
        for (int p = 0; p < 4; ++p) {
            const int br = b_row0 + p * 32;
            const size_t rb = exp_blk_base + (size_t) (n0 + br) * NB_ROW;
            const uint8_t * rp0 = p0 + rb * T::P0B;
            const uint8_t * rp1 = (T::P1B > 0) ? (p1 + rb * T::P1B) : nullptr;
            const __half  * rd  = T::HAS_D ? (d_in + rb) : nullptr;
            float4 w0, w1;
            iq_dev::dequant_row_k8<Q>(rp0, rp1, rd, k0 + b_u0 * 8, w0, w1);
            reinterpret_cast<v4i *>(&s_b[br * BK])[b_u0 ^ (br & 7)] = pack_f4x2_to_h8(w0, w1);
        }
        __syncthreads();

#pragma unroll
        for (int kc = 0; kc < BK / 16; ++kc) {
            const wmma_hfrag_t af = lds_frag_xor(&s_a[(wm * 16 + l16) * BK], rlow, kc, half_w);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const wmma_hfrag_t bf = lds_frag_xor(&s_b[(wn * 64 + j * 16 + l16) * BK], rlow, kc, half_w);
                acc[j] = nib_wmma_f32_16x16x16_f16(af, bf, acc[j]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int col = n0 + wn * 64 + j * 16 + l16;
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int loc_m = wm * 16 + nib_wmma_c_row(half_w, v);
            const int mr = m0 + loc_m;
            if (mr < cnt_e) {
                perm_down[(size_t) (off_e + mr) * N_EMBD + col] = acc[j][v] * s_w[loc_m];
            }
        }
    }
}

// ============================================================================
// Dispatch macros over the 9 GSQ-RCO MoE types
// ============================================================================
#define IQ_DISPATCH_GU(TYPE, CALL) do {              \
    switch (TYPE) {                                  \
        case MOE_Q_IQ1_S:   CALL(MOE_Q_IQ1_S);   break; \
        case MOE_Q_IQ1_M:   CALL(MOE_Q_IQ1_M);   break; \
        case MOE_Q_IQ2_XXS: CALL(MOE_Q_IQ2_XXS); break; \
        case MOE_Q_IQ2_XS:  CALL(MOE_Q_IQ2_XS);  break; \
        case MOE_Q_IQ2_S:   CALL(MOE_Q_IQ2_S);   break; \
        case MOE_Q_IQ3_XXS: CALL(MOE_Q_IQ3_XXS); break; \
        case MOE_Q_IQ3_S:   CALL(MOE_Q_IQ3_S);   break; \
        case MOE_Q_Q2_0:    CALL(MOE_Q_Q2_0);    break; \
        case MOE_Q_IQ4_NL:  CALL(MOE_Q_IQ4_NL);  break; \
        default: iq_moe_fail("unsupported gate_up IQ MoEQuantType"); \
    }                                                \
} while (0)

#define IQ_DISPATCH_DN(TYPE, CALL) do {              \
    switch (TYPE) {                                  \
        case MOE_Q_Q2_0:    CALL(MOE_Q_Q2_0);    break; \
        case MOE_Q_IQ4_NL:  CALL(MOE_Q_IQ4_NL);  break; \
        default: iq_moe_fail("unsupported down IQ MoEQuantType (must have QK | 1152)"); \
    }                                                \
} while (0)

inline int iq_max_tiles(int M, int tm) {
    const int P = M * TOP_K;
    return P / tm + (P < N_EXPERT ? P : N_EXPERT);
}

} // namespace

// ============================================================================
// Public Launchers
// ============================================================================

void launch_moe_iq_decode(const SoaIQExperts & gate_up, const SoaIQExperts & down,
                          const float * x_norm, const int32_t * topk_ids, const float * topk_w,
                          float * tmp_act, float * out, hipStream_t stream,
                          const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
                          float * raw_out) {
    if (gate_up.rows != N_GATE_UP_EXP || gate_up.K != N_EMBD ||
        down.rows != N_EMBD || down.K != N_FF_EXP) {
        iq_moe_fail("launch_moe_iq_decode: invalid gate_up/down dimensions");
    }
    const IqSconvEpi epi{post_conv_w, post_sconv_h, post_res_add, raw_out};

#define LAUNCH_GU_DEC(Q) \
    moe_iq_gate_up_situ_kernel<Q><<<dim3(N_FF_EXP / 16, TOP_K), 128, 0, stream>>>( \
        gate_up.p0, gate_up.p1, gate_up.d, x_norm, topk_ids, tmp_act)
    IQ_DISPATCH_GU(gate_up.type, LAUNCH_GU_DEC);
#undef LAUNCH_GU_DEC

#define LAUNCH_DN_DEC(Q) \
    moe_iq_down_wsum_kernel<Q><<<N_EMBD / 8, 256, 0, stream>>>( \
        down.p0, down.p1, down.d, tmp_act, topk_ids, topk_w, out, epi)
    IQ_DISPATCH_DN(down.type, LAUNCH_DN_DEC);
#undef LAUNCH_DN_DEC
}

void launch_verify_moe_gu_iq(const SoaIQExperts & gate_up,
                             const float * x_norm, int xstride,
                             const int32_t * unique_exp, const uint32_t * /*exp_tok_mask*/,
                             const int32_t * exp_tok_slot, const float * /*exp_tok_w*/,
                             const int32_t * exp_tok_cnt, const int32_t * exp_tok_list,
                             const int32_t * d_num_unique, int m,
                             float * act, hipStream_t stream) {
    const int max_u = (m * TOP_K < N_EXPERT) ? (m * TOP_K) : N_EXPERT;
#define LAUNCH_VGU(Q) \
    verify_moe_gu_iq_kernel<Q><<<dim3(N_FF_EXP / 16, max_u), 128, 0, stream>>>( \
        gate_up.p0, gate_up.p1, gate_up.d, x_norm, xstride, unique_exp, \
        exp_tok_slot, exp_tok_cnt, exp_tok_list, d_num_unique, act)
    IQ_DISPATCH_GU(gate_up.type, LAUNCH_VGU);
#undef LAUNCH_VGU
}

void launch_verify_moe_dn_iq(const SoaIQExperts & down,
                             const float * act,
                             const int32_t * unique_exp, const uint32_t * /*exp_tok_mask*/,
                             const int32_t * exp_tok_slot, const float * exp_tok_w,
                             const int32_t * exp_tok_cnt, const int32_t * exp_tok_list,
                             const int32_t * d_num_unique, int m,
                             float * buf_slot, hipStream_t stream) {
    const int max_u = (m * TOP_K < N_EXPERT) ? (m * TOP_K) : N_EXPERT;
#define LAUNCH_VDN(Q) \
    verify_moe_dn_iq_kernel<Q><<<dim3(N_EMBD / 16, max_u), 128, 0, stream>>>( \
        down.p0, down.p1, down.d, act, unique_exp, \
        exp_tok_slot, exp_tok_w, exp_tok_cnt, exp_tok_list, d_num_unique, buf_slot)
    IQ_DISPATCH_DN(down.type, LAUNCH_VDN);
#undef LAUNCH_VDN
}

void launch_moe_iq_prefill_wmma(const SoaIQExperts & gate_up, const SoaIQExperts & down,
                                const float * X_norm, const int32_t * topk_ids, const float * topk_w,
                                MoePrefillScratch & scratch, float * Y_out, int M, hipStream_t stream,
                                const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
                                int scratch_cap) {
    if (M <= 0) return;
    if (gate_up.rows != N_GATE_UP_EXP || gate_up.K != N_EMBD ||
        down.rows != N_EMBD || down.K != N_FF_EXP) {
        iq_moe_fail("launch_moe_iq_prefill_wmma: invalid gate_up/down dimensions");
    }
    const int chunk_cap = (scratch_cap > 0 && scratch_cap < MAX_UBATCH)
                              ? scratch_cap : MAX_UBATCH;

    for (int t0 = 0; t0 < M; t0 += chunk_cap) {
        const int mc = (M - t0) < chunk_cap ? (M - t0) : chunk_cap;
        const float * chunk_res = (post_res_add != nullptr) ? (post_res_add + (size_t) t0 * N_EMBD) : nullptr;

        _Float16 * xh       = reinterpret_cast<_Float16 *>(scratch.perm_x);
        int32_t * tok_slots = reinterpret_cast<int32_t *>(xh + (size_t) mc * N_EMBD);
        _Float16 * acth     = reinterpret_cast<_Float16 *>(scratch.perm_act);

        constexpr int TM = 64;
        iq_moe_route_kernel<<<1, IQ_ROUTE_THREADS, 0, stream>>>(
            topk_ids + (size_t) t0 * TOP_K, topk_w + (size_t) t0 * TOP_K, mc, TM,
            scratch.exp_counts, scratch.exp_offsets, scratch.exp_cursors,
            scratch.perm_tok, scratch.perm_w, scratch.tile_exp, scratch.tile_m0,
            scratch.d_num_tiles, tok_slots);
        iq_x_to_f16_kernel<<<mc, 256, 0, stream>>>(X_norm + (size_t) t0 * N_EMBD, xh);

        const int tiles = iq_max_tiles(mc, TM);

#define LAUNCH_GU_WMMA(Q) \
        iq_moe_gate_up_wmma_kernel<Q><<<dim3(N_FF_EXP / 64, tiles), 256, 0, stream>>>( \
            gate_up.p0, gate_up.p1, gate_up.d, xh, scratch.exp_offsets, scratch.perm_tok, \
            scratch.tile_exp, scratch.tile_m0, scratch.d_num_tiles, acth)
        IQ_DISPATCH_GU(gate_up.type, LAUNCH_GU_WMMA);
#undef LAUNCH_GU_WMMA

#define LAUNCH_DN_WMMA(Q) \
        iq_moe_down_wmma_kernel<Q><<<dim3(N_EMBD / 128, tiles), 256, 0, stream>>>( \
            down.p0, down.p1, down.d, acth, scratch.exp_offsets, scratch.perm_w, \
            scratch.tile_exp, scratch.tile_m0, scratch.d_num_tiles, scratch.perm_down)
        IQ_DISPATCH_DN(down.type, LAUNCH_DN_WMMA);
#undef LAUNCH_DN_WMMA

        if (post_conv_w != nullptr) {
            if (post_sconv_h == nullptr) {
                iq_moe_fail("launch_moe_iq_prefill_wmma: post_sconv_h must be non-null");
            }
            const int n_tchunks = (mc + IQ_COMBINE_SCONV_TCHUNK - 1) / IQ_COMBINE_SCONV_TCHUNK;
            iq_moe_combine_sconv_kernel<<<dim3(IQ_COMBINE_SCONV_BLOCKS, n_tchunks),
                                          IQ_COMBINE_SCONV_THREADS, 0, stream>>>(
                tok_slots, scratch.perm_down, post_conv_w, post_sconv_h, chunk_res,
                Y_out + (size_t) t0 * N_EMBD, mc);
        } else {
            iq_moe_combine_kernel<<<mc, 256, 0, stream>>>(
                tok_slots, scratch.perm_down, Y_out + (size_t) t0 * N_EMBD);
        }
    }
}

} // namespace inkling
