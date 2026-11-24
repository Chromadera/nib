// ============================================================================
// repack_soa.hip -- RDNA3 (gfx1100) GPU weight repacker (GGML -> 16B SoA)
// ============================================================================
//
// Converts the raw (unaligned) GGML blocks staged in a temporary device buffer
// into the 16-byte-aligned Structure-of-Arrays planes defined in
// include/inkling_engine.h:
//
//   Q8_0 : block_q8_0 { uint16_t d; int8_t qs[32]; }   (34 B / 32 weights)
//          -> qs  : int8_t[rows, K]       (32 B per block, 16-byte aligned)
//          -> d   : __half[rows, K / 32]  (contiguous fp16 scales)
//
//   NVFP4: block_nvfp4 { uint8_t d[4]; uint8_t qs[32]; } (36 B / 64 weights)
//          -> qs     : uint8_t[n_exp, rows, K/2]   (32 B per block)
//          -> scales : uint8_t[n_exp, rows, K/16]  (4 UE4M3 bytes per block)
//          Nibble ordering and sub-block scale ordering are byte-identical to
//          GGML (see the contract); only the stride/destination changes.
//
// Both repacks are pure, lossless reorganisations: the self-test dequantises
// matched samples out of the raw block and the SoA planes and reports the max
// abs error, which must be exactly 0.0f.
// ============================================================================

#include "inkling_engine.h"

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>

namespace inkling {
namespace detail {

// ---------------------------------------------------------------------------
// Q8_0 -> SoA
// ---------------------------------------------------------------------------
// One thread per 32-weight block. Because K is a multiple of 32 the flattened
// block index maps 1:1 onto a contiguous 32-byte qs segment and a contiguous
// half scale, so warp writes are fully coalesced.
__global__ void repack_q8_0_to_soa_kernel(
    const uint8_t * __restrict__ raw,
    int8_t        * __restrict__ qs,
    __half        * __restrict__ dscale,
    int rows,
    int K,
    int dst_row_offset)
{
    const size_t nb = (size_t)rows * (size_t)(K / 32);
    const size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= nb) {
        return;
    }

    const uint8_t * src = raw + tid * 34u;

    uint16_t hd;
    memcpy(&hd, src, 2);

    uint4 q0, q1;
    memcpy(&q0, src + 2,  16);
    memcpy(&q1, src + 18, 16);

    uint8_t * dst_qs = reinterpret_cast<uint8_t *>(qs)
                     + (size_t)dst_row_offset * (size_t)K
                     + tid * 32u;
    reinterpret_cast<uint4 *>(dst_qs)[0] = q0;
    reinterpret_cast<uint4 *>(dst_qs)[1] = q1;

    dscale[(size_t)dst_row_offset * (size_t)(K / 32) + tid] = __ushort_as_half(hd);
}

// ---------------------------------------------------------------------------
// NVFP4 -> SoA
// ---------------------------------------------------------------------------
// One thread per 64-weight super-block. GGML memory order is
// [n_exp, rows, K/64] blocks, which exactly matches the flattened thread id,
// so both the 32-byte qs segment and the packed uint32 of 4 UE4M3 scales land
// at contiguous SoA offsets.
__global__ void repack_nvfp4_to_soa_kernel(
    const uint8_t * __restrict__ raw,
    uint8_t       * __restrict__ qs,
    uint8_t       * __restrict__ scales,
    int n_exp,
    int rows,
    int K)
{
    const size_t nb = (size_t)n_exp * (size_t)rows * (size_t)(K / 64);
    const size_t tid = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    if (tid >= nb) {
        return;
    }

    const uint8_t * src = raw + tid * 36u;

    uint32_t sc;
    memcpy(&sc, src, 4);          // 4 UE4M3 sub-block scales (uint32 packed)

    uint4 q0, q1;
    memcpy(&q0, src + 4,  16);
    memcpy(&q1, src + 20, 16);

    reinterpret_cast<uint4 *>(qs + tid * 32u)[0] = q0;
    reinterpret_cast<uint4 *>(qs + tid * 32u)[1] = q1;

    reinterpret_cast<uint32_t *>(scales)[tid] = sc;
}

// ---------------------------------------------------------------------------
// Launch wrappers
// ---------------------------------------------------------------------------
void repack_q8_0_to_soa(
    const void * d_raw,
    int8_t     * d_qs,
    __half     * d_d,
    int          rows,
    int          K,
    int          dst_row_offset,
    hipStream_t  stream)
{
    if (rows <= 0 || K <= 0 || (K % 32) != 0) {
        throw std::runtime_error("repack_q8_0_to_soa: bad shape");
    }
    const size_t nb = (size_t)rows * (size_t)(K / 32);
    const unsigned blocks = (unsigned)((nb + 255u) / 256u);
    repack_q8_0_to_soa_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(d_raw), d_qs, d_d, rows, K, dst_row_offset);
}

void repack_nvfp4_to_soa(
    const void * d_raw,
    uint8_t    * d_qs,
    uint8_t    * d_scales,
    int          n_exp,
    int          rows,
    int          K,
    hipStream_t  stream)
{
    if (n_exp <= 0 || rows <= 0 || K <= 0 || (K % 64) != 0) {
        throw std::runtime_error("repack_nvfp4_to_soa: bad shape");
    }
    const size_t nb = (size_t)n_exp * (size_t)rows * (size_t)(K / 64);
    const unsigned blocks = (unsigned)((nb + 255u) / 256u);
    repack_nvfp4_to_soa_kernel<<<blocks, 256, 0, stream>>>(
        reinterpret_cast<const uint8_t *>(d_raw), d_qs, d_scales, n_exp, rows, K);
}

// ---------------------------------------------------------------------------
// F32 -> F16
// ---------------------------------------------------------------------------
// Converts n F32 elements sitting in device memory into FP16. When both
// pointers are suitably aligned the bulk is moved as 16-byte float4 -> 8-byte
// uint2 (two __half2 packed by __floats2half2_rn); the remainder -- or the
// whole array when the pointers are not aligned -- falls back to scalar
// __float2half_rn.
__global__ void convert_f32_to_f16_kernel(
    const float * __restrict__ src,
    __half      * __restrict__ dst,
    size_t        n)
{
    const size_t tid    = (size_t)blockIdx.x * (size_t)blockDim.x + (size_t)threadIdx.x;
    const size_t stride = (size_t)gridDim.x * (size_t)blockDim.x;

    const bool aligned =
        ((reinterpret_cast<uintptr_t>(src) & 0xFu) == 0u) &&
        ((reinterpret_cast<uintptr_t>(dst) & 0x7u) == 0u);

    if (aligned) {
        const size_t nvec = n / 4u;
        for (size_t i = tid; i < nvec; i += stride) {
            const float4 v = reinterpret_cast<const float4 *>(src)[i];
            const __half2 h01 = __floats2half2_rn(v.x, v.y);
            const __half2 h23 = __floats2half2_rn(v.z, v.w);
            uint2 o;
            memcpy(&o.x, &h01, 4);
            memcpy(&o.y, &h23, 4);
            reinterpret_cast<uint2 *>(dst)[i] = o;
        }
        const size_t tail = n - nvec * 4u;
        if (tid < tail) {
            const size_t idx = nvec * 4u + tid;
            dst[idx] = __float2half_rn(src[idx]);
        }
    } else {
        for (size_t i = tid; i < n; i += stride) {
            dst[i] = __float2half_rn(src[i]);
        }
    }
}

void convert_f32_to_f16(
    const float * d_src,
    __half      * d_dst,
    size_t        n,
    hipStream_t   stream)
{
    if (n == 0) {
        return;
    }
    size_t blocks = (n + 255u) / 256u;
    if (blocks > 65535u) {
        blocks = 65535u;
    }
    convert_f32_to_f16_kernel<<<(unsigned)blocks, 256, 0, stream>>>(d_src, d_dst, n);
}

// ---------------------------------------------------------------------------
// Host-side reference decode helpers (bit-identical on both sides => err 0)
// ---------------------------------------------------------------------------
static inline float half_bits_to_f32(uint16_t h)
{
    const uint32_t s = (uint32_t)(h & 0x8000u) << 16;
    uint32_t e = (h >> 10) & 0x1Fu;
    uint32_t m = h & 0x3FFu;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) {
            bits = s;
        } else {
            int exp = -14;
            while ((m & 0x400u) == 0) {
                m <<= 1;
                --exp;
            }
            m &= 0x3FFu;
            bits = s | ((uint32_t)(exp + 127) << 23) | (m << 13);
        }
    } else if (e == 0x1F) {
        bits = s | 0x7F800000u | (m << 13);
    } else {
        bits = s | ((e - 15u + 127u) << 23) | (m << 13);
    }
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

static inline float ue4m3_to_f32(uint8_t x)
{
    // Mirrors ggml_ue4m3_to_fp32: returns raw * 0.5f for the NVFP4 convention.
    if (x == 0 || x == 0x7F) {
        return 0.0f;
    }
    const int exp = (x >> 3) & 0xF;
    const int man = x & 0x7;
    float raw;
    if (exp == 0) {
        raw = std::ldexp((float)man, -9);
    } else {
        raw = std::ldexp(1.0f + (float)man / 8.0f, exp - 7);
    }
    return raw * 0.5f;
}

static inline float kvalue_fp4(int idx)
{
    static const int8_t kv[16] = {0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};
    return (float)kv[idx & 0xF];
}

// ---------------------------------------------------------------------------
// Self-test: dequantise matched samples from raw blocks and from the SoA
// planes, return the maximum absolute difference (must be 0.0f).
// ---------------------------------------------------------------------------
float verify_q8_0_repack(
    const void * raw_host,
    const int8_t * d_qs,
    const __half * d_d,
    int rows,
    int K)
{
    const uint8_t * raw = reinterpret_cast<const uint8_t *>(raw_host);
    const int nblk = K / 32;
    const int rs[3] = {0, rows / 2, rows - 1};
    const int bs[3] = {0, nblk / 2, nblk - 1};

    float max_err = 0.0f;

    for (int ri = 0; ri < 3; ++ri) {
        const int r = rs[ri];
        for (int bi = 0; bi < 3; ++bi) {
            const int b = bs[bi];

            const uint8_t * rb = raw + ((size_t)r * (size_t)nblk + (size_t)b) * 34u;
            uint16_t hd_raw;
            memcpy(&hd_raw, rb, 2);
            const float s_raw = half_bits_to_f32(hd_raw);

            int8_t q_dev[32];
            uint16_t hd_dev;
            (void)hipMemcpy(q_dev, d_qs + (size_t)r * (size_t)K + (size_t)b * 32, 32,
                      hipMemcpyDeviceToHost);
            (void)hipMemcpy(&hd_dev, d_d + (size_t)r * (size_t)nblk + (size_t)b, 2,
                      hipMemcpyDeviceToHost);
            const float s_dev = half_bits_to_f32(hd_dev);

            for (int j = 0; j < 32; ++j) {
                const float w_raw = s_raw * (float)(int8_t)rb[2 + j];
                const float w_dev = s_dev * (float)q_dev[j];
                const float e = fabsf(w_raw - w_dev);
                if (e > max_err) {
                    max_err = e;
                }
            }
        }
    }
    return max_err;
}

float verify_nvfp4_repack(
    const void * raw_host,
    const uint8_t * d_qs,
    const uint8_t * d_scales,
    const float * d_scale2,
    int n_exp,
    int rows,
    int K)
{
    const uint8_t * raw = reinterpret_cast<const uint8_t *>(raw_host);
    const int nblk = K / 64;
    const int es[3] = {0, n_exp / 2, n_exp - 1};
    const int rs[3] = {0, rows / 2, rows - 1};
    const int bs[3] = {0, nblk / 2, nblk - 1};

    float max_err = 0.0f;

    for (int ei = 0; ei < 3; ++ei) {
        const int e = es[ei];
        float s2;
        (void)hipMemcpy(&s2, d_scale2 + e, 4, hipMemcpyDeviceToHost);

        for (int ri = 0; ri < 3; ++ri) {
            const int r = rs[ri];
            for (int bi = 0; bi < 3; ++bi) {
                const int b = bs[bi];

                const uint8_t * rb = raw +
                    ((size_t)e * (size_t)rows + (size_t)r) * (size_t)nblk * 36u +
                    (size_t)b * 36u;

                uint8_t q_dev[32];
                uint8_t sc_dev[4];
                (void)hipMemcpy(q_dev,  d_qs     + (((size_t)e * (size_t)rows + (size_t)r) * (size_t)(K / 2)  + (size_t)b * 32u), 32, hipMemcpyDeviceToHost);
                (void)hipMemcpy(sc_dev, d_scales + (((size_t)e * (size_t)rows + (size_t)r) * (size_t)(K / 16) + (size_t)b * 4u),  4,  hipMemcpyDeviceToHost);

                for (int s = 0; s < 4; ++s) {
                    const float w_raw_scale = ue4m3_to_f32(rb[s])      * s2;
                    const float w_dev_scale = ue4m3_to_f32(sc_dev[s])  * s2;
                    for (int j = 0; j < 8; ++j) {
                        const uint8_t b_raw = rb[4 + s * 8 + j];
                        const uint8_t b_dev = q_dev[s * 8 + j];

                        const float w_raw_lo = kvalue_fp4(b_raw & 0x0F) * w_raw_scale;
                        const float w_dev_lo = kvalue_fp4(b_dev & 0x0F) * w_dev_scale;
                        const float w_raw_hi = kvalue_fp4(b_raw >> 4)   * w_raw_scale;
                        const float w_dev_hi = kvalue_fp4(b_dev >> 4)   * w_dev_scale;

                        float e0 = fabsf(w_raw_lo - w_dev_lo);
                        float e1 = fabsf(w_raw_hi - w_dev_hi);
                        if (e0 > max_err) max_err = e0;
                        if (e1 > max_err) max_err = e1;
                    }
                }
            }
        }
    }
    return max_err;
}

} // namespace detail
} // namespace inkling
