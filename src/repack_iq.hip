// ============================================================================
// repack_iq.hip — Load-time GPU dequant+repack for GSQ-RCO GGUF tensors
// ============================================================================
// 1. Dense BF16 / Q4_K / Q5_K / Q6_K / IQ4_XS -> SoaQ8Matrix (16B-aligned Q8_0)
// 2. MoE experts (Q2_0, IQ4_NL, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
//    IQ3_XXS, IQ3_S) -> 16B-aligned SoaIQExperts + per-expert exp_amax[128]
// ============================================================================

#include "kernels_iq.h"
#include "iq_dequant_device.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

namespace inkling {
namespace {

static constexpr uint32_t GGML_T_Q4_K   = 12;
static constexpr uint32_t GGML_T_Q5_K   = 13;
static constexpr uint32_t GGML_T_Q6_K   = 14;
static constexpr uint32_t GGML_T_IQ4_XS = 23;
static constexpr uint32_t GGML_T_BF16   = 30;

[[noreturn]] void repack_iq_fail(const char * msg) {
    std::fprintf(stderr, "[repack_iq] CONTRACT VIOLATION: %s\n", msg);
    std::abort();
}

__device__ __forceinline__ float load_f16_unaligned(const uint8_t * p) {
    const uint16_t bits = (uint16_t) p[0] | ((uint16_t) p[1] << 8);
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ uint16_t load_u16_unaligned(const uint8_t * p) {
    return (uint16_t) p[0] | ((uint16_t) p[1] << 8);
}

__device__ __forceinline__ float warp_max_f32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v = fmaxf(v, __shfl_xor(v, off, 32));
    }
    return v;
}

// Quantize 32 floats held one-per-lane across a Wave32 warp into one SoA Q8_0 block.
__device__ __forceinline__ void warp_quant_q8_32(
    float val, int8_t * dst_qs, __half * dst_d, size_t blk_idx, int lane) {
    const float amax = warp_max_f32(fabsf(val));
    const float d    = amax * (1.0f / 127.0f);
    const float invd = (d > 0.0f) ? (1.0f / d) : 0.0f;
    int q = __float2int_rn(val * invd);
    if (q >  127) q =  127;
    if (q < -127) q = -127;
    dst_qs[blk_idx * 32 + lane] = (int8_t) q;
    if (lane == 0) {
        dst_d[blk_idx] = __float2half(d);
    }
}

// ---------------------------------------------------------------------------
// 1. Dense BF16 -> SoaQ8Matrix (1 warp = 1 Q8_0 block of 32 elements)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
repack_bf16_to_soa_q8_kernel(const uint16_t * __restrict__ raw,
                             int8_t * __restrict__ d_qs, __half * __restrict__ d_d,
                             int rows, int K, int dst_row_offset) {
    const int nb_row = K >> 5;
    const size_t total_blks = (size_t) rows * nb_row;
    const int warp_id = (int) ((blockIdx.x * blockDim.x + threadIdx.x) >> 5);
    const int lane    = threadIdx.x & 31;
    if ((size_t) warp_id >= total_blks) return;

    const int r = warp_id / nb_row;
    const int b = warp_id - r * nb_row;
    const uint16_t bf = raw[(size_t) warp_id * 32 + lane];
    const uint32_t f32_bits = (uint32_t) bf << 16;
    const float val = __uint_as_float(f32_bits);
    const size_t dst_blk = (size_t) (dst_row_offset + r) * nb_row + b;
    warp_quant_q8_32(val, d_qs, d_d, dst_blk, lane);
}

// ---------------------------------------------------------------------------
// 2. Dense Q4_K -> SoaQ8Matrix (1 warp = 1 QK_K=256 super-block = 8 Q8 blocks)
// ---------------------------------------------------------------------------
__device__ __forceinline__ void get_scale_min_k4_dev(int j, const uint8_t * q, uint8_t & sc, uint8_t & m) {
    if (j < 4) {
        sc = q[j] & 63u;
        m  = q[j + 4] & 63u;
    } else {
        sc = (uint8_t) ((q[j + 4] & 0xfu) | ((q[j - 4] >> 6) << 4));
        m  = (uint8_t) ((q[j + 4] >>  4)  | ((q[j - 0] >> 6) << 4));
    }
}

__global__ void __launch_bounds__(256)
repack_q4_k_to_soa_q8_kernel(const uint8_t * __restrict__ raw,
                             int8_t * __restrict__ d_qs, __half * __restrict__ d_d,
                             int rows, int K, int dst_row_offset) {
    // block_q4_K = 144 B: d(2), dmin(2), scales(12), qs(128)
    const int nb_super = K >> 8;
    const int nb_q8    = K >> 5;
    const size_t total_super = (size_t) rows * nb_super;
    const size_t sb = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane  = threadIdx.x & 31;
    if (sb >= total_super) return;

    const int r = (int) (sb / nb_super);
    const int s = (int) (sb - (size_t) r * nb_super);
    const uint8_t * blk = raw + sb * 144;
    const float d    = load_f16_unaligned(blk + 0);
    const float dmin = load_f16_unaligned(blk + 2);
    const uint8_t * scales = blk + 4;
    const uint8_t * qs     = blk + 16;
    const size_t dst_blk0  = (size_t) (dst_row_offset + r) * nb_q8 + (size_t) s * 8;

#pragma unroll
    for (int g = 0; g < 4; ++g) {
        uint8_t sc1, m1, sc2, m2;
        get_scale_min_k4_dev(2 * g + 0, scales, sc1, m1);
        get_scale_min_k4_dev(2 * g + 1, scales, sc2, m2);
        const uint8_t qb = qs[g * 32 + lane];
        const float v0 = (d * (float) sc1) * (float) (qb & 0xfu) - (dmin * (float) m1);
        const float v1 = (d * (float) sc2) * (float) (qb >>  4)  - (dmin * (float) m2);
        warp_quant_q8_32(v0, d_qs, d_d, dst_blk0 + 2 * g + 0, lane);
        warp_quant_q8_32(v1, d_qs, d_d, dst_blk0 + 2 * g + 1, lane);
    }
}

// ---------------------------------------------------------------------------
// 3. Dense Q5_K -> SoaQ8Matrix (1 warp = 1 QK_K=256 super-block = 8 Q8 blocks)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
repack_q5_k_to_soa_q8_kernel(const uint8_t * __restrict__ raw,
                             int8_t * __restrict__ d_qs, __half * __restrict__ d_d,
                             int rows, int K, int dst_row_offset) {
    // block_q5_K = 176 B: d(2), dmin(2), scales(12), qh(32), qs(128)
    const int nb_super = K >> 8;
    const int nb_q8    = K >> 5;
    const size_t total_super = (size_t) rows * nb_super;
    const size_t sb = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane  = threadIdx.x & 31;
    if (sb >= total_super) return;

    const int r = (int) (sb / nb_super);
    const int s = (int) (sb - (size_t) r * nb_super);
    const uint8_t * blk = raw + sb * 176;
    const float d    = load_f16_unaligned(blk + 0);
    const float dmin = load_f16_unaligned(blk + 2);
    const uint8_t * scales = blk + 4;
    const uint8_t   qh_b   = blk[16 + lane];
    const uint8_t * qs     = blk + 48;
    const size_t dst_blk0  = (size_t) (dst_row_offset + r) * nb_q8 + (size_t) s * 8;

#pragma unroll
    for (int g = 0; g < 4; ++g) {
        uint8_t sc1, m1, sc2, m2;
        get_scale_min_k4_dev(2 * g + 0, scales, sc1, m1);
        get_scale_min_k4_dev(2 * g + 1, scales, sc2, m2);
        const uint8_t qb = qs[g * 32 + lane];
        const uint8_t u1 = (uint8_t) (1u << (2 * g));
        const uint8_t u2 = (uint8_t) (2u << (2 * g));
        const float v0 = (d * (float) sc1) * (float) ((qb & 0xfu) + ((qh_b & u1) ? 16u : 0u)) - (dmin * (float) m1);
        const float v1 = (d * (float) sc2) * (float) ((qb >>  4)  + ((qh_b & u2) ? 16u : 0u)) - (dmin * (float) m2);
        warp_quant_q8_32(v0, d_qs, d_d, dst_blk0 + 2 * g + 0, lane);
        warp_quant_q8_32(v1, d_qs, d_d, dst_blk0 + 2 * g + 1, lane);
    }
}

// ---------------------------------------------------------------------------
// 4. Dense Q6_K -> SoaQ8Matrix (1 warp = 1 QK_K=256 super-block = 8 Q8 blocks)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
repack_q6_k_to_soa_q8_kernel(const uint8_t * __restrict__ raw,
                             int8_t * __restrict__ d_qs, __half * __restrict__ d_d,
                             int rows, int K, int dst_row_offset) {
    // block_q6_K = 210 B: ql(128), qh(64), scales(16, int8), d(2)
    const int nb_super = K >> 8;
    const int nb_q8    = K >> 5;
    const size_t total_super = (size_t) rows * nb_super;
    const size_t sb = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane  = threadIdx.x & 31;
    if (sb >= total_super) return;

    const int r = (int) (sb / nb_super);
    const int s = (int) (sb - (size_t) r * nb_super);
    const uint8_t * blk = raw + sb * 210;
    const uint8_t * ql  = blk + 0;
    const uint8_t * qh  = blk + 128;
    const int8_t  * sc  = reinterpret_cast<const int8_t *>(blk + 192);
    const float     d   = load_f16_unaligned(blk + 208);
    const size_t dst_blk0 = (size_t) (dst_row_offset + r) * nb_q8 + (size_t) s * 8;
    const int is = lane >> 4; // 0 or 1

#pragma unroll
    for (int half = 0; half < 2; ++half) {
        const uint8_t ql0 = ql[half * 64 + lane];
        const uint8_t ql1 = ql[half * 64 + 32 + lane];
        const uint8_t qhb = qh[half * 32 + lane];
        const int8_t * sch = sc + half * 8;
        const int q1 = (int) ((ql0 & 0xfu) | (((qhb >> 0) & 3u) << 4)) - 32;
        const int q2 = (int) ((ql1 & 0xfu) | (((qhb >> 2) & 3u) << 4)) - 32;
        const int q3 = (int) ((ql0 >>  4)  | (((qhb >> 4) & 3u) << 4)) - 32;
        const int q4 = (int) ((ql1 >>  4)  | (((qhb >> 6) & 3u) << 4)) - 32;
        warp_quant_q8_32(d * (float) sch[is + 0] * (float) q1, d_qs, d_d, dst_blk0 + half * 4 + 0, lane);
        warp_quant_q8_32(d * (float) sch[is + 2] * (float) q2, d_qs, d_d, dst_blk0 + half * 4 + 1, lane);
        warp_quant_q8_32(d * (float) sch[is + 4] * (float) q3, d_qs, d_d, dst_blk0 + half * 4 + 2, lane);
        warp_quant_q8_32(d * (float) sch[is + 6] * (float) q4, d_qs, d_d, dst_blk0 + half * 4 + 3, lane);
    }
}

// ---------------------------------------------------------------------------
// 5. Dense IQ4_XS -> SoaQ8Matrix (1 warp = 1 QK_K=256 super-block = 8 Q8 blocks)
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(256)
repack_iq4_xs_to_soa_q8_kernel(const uint8_t * __restrict__ raw,
                               int8_t * __restrict__ d_qs, __half * __restrict__ d_d,
                               int rows, int K, int dst_row_offset) {
    // block_iq4_xs = 136 B: d(2), scales_h(2), scales_l(4), qs(128)
    const int nb_super = K >> 8;
    const int nb_q8    = K >> 5;
    const size_t total_super = (size_t) rows * nb_super;
    const size_t sb = ((size_t) blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane  = threadIdx.x & 31;
    if (sb >= total_super) return;

    const int r = (int) (sb / nb_super);
    const int s = (int) (sb - (size_t) r * nb_super);
    const uint8_t * blk = raw + sb * 136;
    const float    d        = load_f16_unaligned(blk + 0);
    const uint16_t scales_h = load_u16_unaligned(blk + 2);
    const uint8_t * scales_l = blk + 4;
    const uint8_t * qs       = blk + 8;
    const size_t dst_blk0    = (size_t) (dst_row_offset + r) * nb_q8 + (size_t) s * 8;

    const int j  = lane & 15;
    const int hi = lane >> 4; // 0 for [0..15], 1 for [16..31]
#pragma unroll
    for (int ib = 0; ib < 8; ++ib) {
        const int ls = (int) (((scales_l[ib >> 1] >> (4 * (ib & 1))) & 0xfu) |
                              (((scales_h >> (2 * ib)) & 3u) << 4));
        const float dl = d * (float) (ls - 32);
        const uint8_t qb = qs[ib * 16 + j];
        const int nib = hi ? (qb >> 4) : (qb & 0xfu);
        const float val = dl * (float) iq_lut::kvalues_iq4nl[nib];
        warp_quant_q8_32(val, d_qs, d_d, dst_blk0 + ib, lane);
    }
}

// ---------------------------------------------------------------------------
// 6. MoE Expert SoA Repack Kernels (all 9 GSQ-RCO formats -> 16B-aligned p0)
// ---------------------------------------------------------------------------
template <int RAW_BYTES, int P0_BYTES, int P1_BYTES, bool HAS_D, bool IS_IQ3_S>
__global__ void __launch_bounds__(256)
repack_iq_blocks_kernel(const uint8_t * __restrict__ raw,
                        uint8_t * __restrict__ p0,
                        uint8_t * __restrict__ p1,
                        __half  * __restrict__ d_out,
                        size_t n_blocks) {
    const size_t b = (size_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= n_blocks) return;

    const uint8_t * src = raw + b * RAW_BYTES;
    uint8_t * dst0 = p0 + b * P0_BYTES;

    if constexpr (HAS_D) {
        const uint16_t dbits = load_u16_unaligned(src);
        d_out[b] = __ushort_as_half(dbits);
    }

    if constexpr (IS_IQ3_S) {
        // Raw IQ3_S (110 B): d(0..1), qs(2..65), qh(66..73), signs(74..105), scales(106..109)
        // SoA: p0 = qs(0..63) + signs(64..95); p1 = qh(0..7) + scales(8..11)
        for (int i = 0; i < 64; ++i) dst0[i]      = src[2 + i];
        for (int i = 0; i < 32; ++i) dst0[64 + i] = src[74 + i];
        uint8_t * dst1 = p1 + b * 12;
        for (int i = 0; i < 8; ++i)  dst1[i]      = src[66 + i];
        for (int i = 0; i < 4; ++i)  dst1[8 + i]  = src[106 + i];
    } else {
        const int off0 = HAS_D ? 2 : 0;
        for (int i = 0; i < P0_BYTES; ++i) {
            dst0[i] = src[off0 + i];
        }
        if constexpr (P1_BYTES > 0) {
            uint8_t * dst1 = p1 + b * P1_BYTES;
            for (int i = 0; i < P1_BYTES; ++i) {
                dst1[i] = src[off0 + P0_BYTES + i];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// 7. Per-expert max(|w|) reduction for RDNA4 FP8 WMMA scaling (1 CTA/expert)
// ---------------------------------------------------------------------------
template <MoEQuantType Q>
__global__ void __launch_bounds__(256)
compute_iq_expert_amax_kernel(const uint8_t * __restrict__ p0,
                              const uint8_t * __restrict__ p1,
                              const __half  * __restrict__ d_in,
                              int rows, int K,
                              float * __restrict__ exp_amax) {
    using T = iq_dev::IqTraits<Q>;
    const int ex = blockIdx.x;
    const int nb_row = K / T::QK;
    const int nsub8_row = K >> 3;
    const size_t exp_blk_base = (size_t) ex * rows * nb_row;
    const size_t total_chunks = (size_t) rows * nsub8_row;

    float local_max = 0.0f;
    for (size_t idx = threadIdx.x; idx < total_chunks; idx += blockDim.x) {
        const int r    = (int) (idx / nsub8_row);
        const int s8   = (int) (idx - (size_t) r * nsub8_row);
        const int k0   = s8 << 3;
        const size_t row_blk = exp_blk_base + (size_t) r * nb_row;
        const uint8_t * p0_row = p0 + row_blk * T::P0B;
        const uint8_t * p1_row = (T::P1B > 0) ? (p1 + row_blk * T::P1B) : nullptr;
        const __half  * d_row  = T::HAS_D ? (d_in + row_blk) : nullptr;
        float4 w0, w1;
        iq_dev::dequant_row_k8<Q>(p0_row, p1_row, d_row, k0, w0, w1);
        local_max = fmaxf(local_max, fabsf(w0.x));
        local_max = fmaxf(local_max, fabsf(w0.y));
        local_max = fmaxf(local_max, fabsf(w0.z));
        local_max = fmaxf(local_max, fabsf(w0.w));
        local_max = fmaxf(local_max, fabsf(w1.x));
        local_max = fmaxf(local_max, fabsf(w1.y));
        local_max = fmaxf(local_max, fabsf(w1.z));
        local_max = fmaxf(local_max, fabsf(w1.w));
    }

    local_max = warp_max_f32(local_max);
    __shared__ float s_wmax[8];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) s_wmax[warp] = local_max;
    __syncthreads();
    if (warp == 0) {
        float v = (lane < 8) ? s_wmax[lane] : 0.0f;
        v = warp_max_f32(v);
        if (lane == 0) {
            exp_amax[ex] = v;
        }
    }
}

// ---------------------------------------------------------------------------
// 8. Single-row GPU dequantizer for unit tests
// ---------------------------------------------------------------------------
template <MoEQuantType Q>
__global__ void __launch_bounds__(256)
dequant_iq_row_kernel(const uint8_t * __restrict__ p0,
                      const uint8_t * __restrict__ p1,
                      const __half  * __restrict__ d_in,
                      int exp, int row, int rows, int K,
                      float * __restrict__ out) {
    using T = iq_dev::IqTraits<Q>;
    const int nb_row = K / T::QK;
    const int nsub8  = K >> 3;
    const size_t row_blk = ((size_t) exp * rows + row) * nb_row;
    const uint8_t * p0_row = p0 + row_blk * T::P0B;
    const uint8_t * p1_row = (T::P1B > 0) ? (p1 + row_blk * T::P1B) : nullptr;
    const __half  * d_row  = T::HAS_D ? (d_in + row_blk) : nullptr;

    for (int s8 = threadIdx.x; s8 < nsub8; s8 += blockDim.x) {
        float4 w0, w1;
        iq_dev::dequant_row_k8<Q>(p0_row, p1_row, d_row, s8 << 3, w0, w1);
        reinterpret_cast<float4 *>(out)[2 * s8 + 0] = w0;
        reinterpret_cast<float4 *>(out)[2 * s8 + 1] = w1;
    }
}

template <MoEQuantType Q, int RAW_BYTES, bool IS_IQ3_S = false>
void launch_repack_one(const void * d_raw, const SoaIQExperts & dst, hipStream_t stream) {
    using T = iq_dev::IqTraits<Q>;
    const size_t n_blocks = (size_t) dst.n_exp * dst.rows * (dst.K / T::QK);
    const dim3 grid((unsigned) ((n_blocks + 255) / 256));
    repack_iq_blocks_kernel<RAW_BYTES, T::P0B, T::P1B, T::HAS_D, IS_IQ3_S><<<grid, 256, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(d_raw), dst.p0, dst.p1, dst.d, n_blocks);
    if (dst.exp_amax != nullptr) {
        compute_iq_expert_amax_kernel<Q><<<dst.n_exp, 256, 0, stream>>>(
            dst.p0, dst.p1, dst.d, dst.rows, dst.K, dst.exp_amax);
    }
}

} // namespace

namespace detail {

void repack_dense_to_soa_q8(
    uint32_t ggml_type, const void * d_raw,
    int8_t * d_qs, __half * d_d,
    int rows, int K, int dst_row_offset, hipStream_t stream) {
    if (rows <= 0 || K <= 0 || (K % 32) != 0) {
        repack_iq_fail("repack_dense_to_soa_q8: invalid rows/K");
    }
    constexpr int TPB = 256; // 8 warps/CTA
    constexpr int WPC = TPB / 32;
    if (ggml_type == GGML_T_BF16) {
        const size_t total_warps = (size_t) rows * (K >> 5);
        const dim3 grid((unsigned) ((total_warps + WPC - 1) / WPC));
        repack_bf16_to_soa_q8_kernel<<<grid, TPB, 0, stream>>>(
            reinterpret_cast<const uint16_t *>(d_raw), d_qs, d_d, rows, K, dst_row_offset);
        return;
    }
    if ((K % 256) != 0) {
        repack_iq_fail("repack_dense_to_soa_q8: K-quant/IQ4_XS requires K % 256 == 0");
    }
    const size_t total_warps = (size_t) rows * (K >> 8);
    const dim3 grid((unsigned) ((total_warps + WPC - 1) / WPC));
    const uint8_t * raw8 = reinterpret_cast<const uint8_t *>(d_raw);
    switch (ggml_type) {
        case GGML_T_Q4_K:
            repack_q4_k_to_soa_q8_kernel<<<grid, TPB, 0, stream>>>(raw8, d_qs, d_d, rows, K, dst_row_offset);
            break;
        case GGML_T_Q5_K:
            repack_q5_k_to_soa_q8_kernel<<<grid, TPB, 0, stream>>>(raw8, d_qs, d_d, rows, K, dst_row_offset);
            break;
        case GGML_T_Q6_K:
            repack_q6_k_to_soa_q8_kernel<<<grid, TPB, 0, stream>>>(raw8, d_qs, d_d, rows, K, dst_row_offset);
            break;
        case GGML_T_IQ4_XS:
            repack_iq4_xs_to_soa_q8_kernel<<<grid, TPB, 0, stream>>>(raw8, d_qs, d_d, rows, K, dst_row_offset);
            break;
        default:
            repack_iq_fail("repack_dense_to_soa_q8: unsupported ggml_type");
    }
}

void repack_iq_experts_to_soa(
    const void * d_raw, const SoaIQExperts & dst, hipStream_t stream) {
    switch (dst.type) {
        case MOE_Q_Q2_0:    launch_repack_one<MOE_Q_Q2_0,    18>(d_raw, dst, stream); break;
        case MOE_Q_IQ4_NL:  launch_repack_one<MOE_Q_IQ4_NL,  18>(d_raw, dst, stream); break;
        case MOE_Q_IQ1_S:   launch_repack_one<MOE_Q_IQ1_S,   50>(d_raw, dst, stream); break;
        case MOE_Q_IQ1_M:   launch_repack_one<MOE_Q_IQ1_M,   56>(d_raw, dst, stream); break;
        case MOE_Q_IQ2_XXS: launch_repack_one<MOE_Q_IQ2_XXS, 66>(d_raw, dst, stream); break;
        case MOE_Q_IQ2_XS:  launch_repack_one<MOE_Q_IQ2_XS,  74>(d_raw, dst, stream); break;
        case MOE_Q_IQ2_S:   launch_repack_one<MOE_Q_IQ2_S,   82>(d_raw, dst, stream); break;
        case MOE_Q_IQ3_XXS: launch_repack_one<MOE_Q_IQ3_XXS, 98>(d_raw, dst, stream); break;
        case MOE_Q_IQ3_S:   launch_repack_one<MOE_Q_IQ3_S,  110, true>(d_raw, dst, stream); break;
        default:
            repack_iq_fail("repack_iq_experts_to_soa: unsupported MoEQuantType");
    }
}

void dequant_iq_expert_row_gpu(
    const SoaIQExperts & src, int exp, int row, float * d_out, hipStream_t stream) {
    switch (src.type) {
        case MOE_Q_Q2_0:    dequant_iq_row_kernel<MOE_Q_Q2_0>   <<<1, 256, 0, stream>>>(src.p0, src.p1, src.d, exp, row, src.rows, src.K, d_out); break;
        case MOE_Q_IQ4_NL:  dequant_iq_row_kernel<MOE_Q_IQ4_NL> <<<1, 256, 0, stream>>>(src.p0, src.p1, src.d, exp, row, src.rows, src.K, d_out); break;
        case MOE_Q_IQ1_S:   dequant_iq_row_kernel<MOE_Q_IQ1_S>  <<<1, 256, 0, stream>>>(src.p0, src.p1, src.d, exp, row, src.rows, src.K, d_out); break;
        case MOE_Q_IQ1_M:   dequant_iq_row_kernel<MOE_Q_IQ1_M>  <<<1, 256, 0, stream>>>(src.p0, src.p1, src.d, exp, row, src.rows, src.K, d_out); break;
        case MOE_Q_IQ2_XXS: dequant_iq_row_kernel<MOE_Q_IQ2_XXS><<<1, 256, 0, stream>>>(src.p0, src.p1, src.d, exp, row, src.rows, src.K, d_out); break;
        case MOE_Q_IQ2_XS:  dequant_iq_row_kernel<MOE_Q_IQ2_XS> <<<1, 256, 0, stream>>>(src.p0, src.p1, src.d, exp, row, src.rows, src.K, d_out); break;
        case MOE_Q_IQ2_S:   dequant_iq_row_kernel<MOE_Q_IQ2_S>  <<<1, 256, 0, stream>>>(src.p0, src.p1, src.d, exp, row, src.rows, src.K, d_out); break;
        case MOE_Q_IQ3_XXS: dequant_iq_row_kernel<MOE_Q_IQ3_XXS><<<1, 256, 0, stream>>>(src.p0, src.p1, src.d, exp, row, src.rows, src.K, d_out); break;
        case MOE_Q_IQ3_S:   dequant_iq_row_kernel<MOE_Q_IQ3_S>  <<<1, 256, 0, stream>>>(src.p0, src.p1, src.d, exp, row, src.rows, src.K, d_out); break;
        default:
            repack_iq_fail("dequant_iq_expert_row_gpu: unsupported MoEQuantType");
    }
}

} // namespace detail
} // namespace inkling
