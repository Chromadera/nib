#pragma once

// ============================================================================
// iq_dequant_device.h — Canonical device dequantizers for all 9 GSQ-RCO MoE
// formats stored in SoaIQExperts (16B-aligned p0, aligned p1, fp16 d).
//
// Every format decodes 8 contiguous weights at sub8 = (k_in_block / 8) into
// (float4 w0, float4 w1), bit-exact to upstream llama.cpp ggml-quants.c.
// ============================================================================

#include "kernels_iq.h"
#include "iq_tables.h"

#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

namespace inkling {
namespace iq_dev {

__device__ __forceinline__ float sign_mul(float mag, uint32_t signs, int bit) {
    return (signs & (1u << bit)) ? -mag : mag;
}

__device__ __forceinline__ void unpack_u64_signs(uint64_t g64, uint32_t signs, float scale,
                                                 float4 & w0, float4 & w1) {
    const uint32_t lo = (uint32_t) g64;
    const uint32_t hi = (uint32_t) (g64 >> 32);
    w0.x = scale * sign_mul((float) ((lo >>  0) & 0xffu), signs, 0);
    w0.y = scale * sign_mul((float) ((lo >>  8) & 0xffu), signs, 1);
    w0.z = scale * sign_mul((float) ((lo >> 16) & 0xffu), signs, 2);
    w0.w = scale * sign_mul((float) ((lo >> 24) & 0xffu), signs, 3);
    w1.x = scale * sign_mul((float) ((hi >>  0) & 0xffu), signs, 4);
    w1.y = scale * sign_mul((float) ((hi >>  8) & 0xffu), signs, 5);
    w1.z = scale * sign_mul((float) ((hi >> 16) & 0xffu), signs, 6);
    w1.w = scale * sign_mul((float) ((hi >> 24) & 0xffu), signs, 7);
}

__device__ __forceinline__ void unpack_u32x2_signs(uint32_t g1, uint32_t g2, uint32_t signs, float scale,
                                                   float4 & w0, float4 & w1) {
    w0.x = scale * sign_mul((float) ((g1 >>  0) & 0xffu), signs, 0);
    w0.y = scale * sign_mul((float) ((g1 >>  8) & 0xffu), signs, 1);
    w0.z = scale * sign_mul((float) ((g1 >> 16) & 0xffu), signs, 2);
    w0.w = scale * sign_mul((float) ((g1 >> 24) & 0xffu), signs, 3);
    w1.x = scale * sign_mul((float) ((g2 >>  0) & 0xffu), signs, 4);
    w1.y = scale * sign_mul((float) ((g2 >>  8) & 0xffu), signs, 5);
    w1.z = scale * sign_mul((float) ((g2 >> 16) & 0xffu), signs, 6);
    w1.w = scale * sign_mul((float) ((g2 >> 24) & 0xffu), signs, 7);
}

__device__ __forceinline__ void unpack_iq1_grid(uint32_t g32, float delta, float scale,
                                                float4 & w0, float4 & w1) {
    const uint32_t lo = g32 & 0x0f0f0f0fu;
    const uint32_t hi = (g32 >> 4) & 0x0f0f0f0fu;
    w0.x = scale * ((float) ((lo >>  0) & 0xffu) + delta);
    w0.y = scale * ((float) ((lo >>  8) & 0xffu) + delta);
    w0.z = scale * ((float) ((lo >> 16) & 0xffu) + delta);
    w0.w = scale * ((float) ((lo >> 24) & 0xffu) + delta);
    w1.x = scale * ((float) ((hi >>  0) & 0xffu) + delta);
    w1.y = scale * ((float) ((hi >>  8) & 0xffu) + delta);
    w1.z = scale * ((float) ((hi >> 16) & 0xffu) + delta);
    w1.w = scale * ((float) ((hi >> 24) & 0xffu) + delta);
}

template <MoEQuantType Q>
struct IqTraits;

template <>
struct IqTraits<MOE_Q_Q2_0> {
    static constexpr int QK    = 64;
    static constexpr int P0B   = 16;
    static constexpr int P1B   = 0;
    static constexpr bool HAS_D = true;

    __device__ __forceinline__ static void dequant8(
        const uint8_t * p0_blk, const uint8_t * /*p1_blk*/, float df, int sub8,
        float4 & w0, float4 & w1) {
        const uint32_t b0 = p0_blk[2 * sub8 + 0];
        const uint32_t b1 = p0_blk[2 * sub8 + 1];
        w0.x = (float) ((int) ((b0 >> 0) & 3u) - 1) * df;
        w0.y = (float) ((int) ((b0 >> 2) & 3u) - 1) * df;
        w0.z = (float) ((int) ((b0 >> 4) & 3u) - 1) * df;
        w0.w = (float) ((int) ((b0 >> 6) & 3u) - 1) * df;
        w1.x = (float) ((int) ((b1 >> 0) & 3u) - 1) * df;
        w1.y = (float) ((int) ((b1 >> 2) & 3u) - 1) * df;
        w1.z = (float) ((int) ((b1 >> 4) & 3u) - 1) * df;
        w1.w = (float) ((int) ((b1 >> 6) & 3u) - 1) * df;
    }
};

template <>
struct IqTraits<MOE_Q_IQ4_NL> {
    static constexpr int QK    = 32;
    static constexpr int P0B   = 16;
    static constexpr int P1B   = 0;
    static constexpr bool HAS_D = true;

    __device__ __forceinline__ static void dequant8(
        const uint8_t * p0_blk, const uint8_t * /*p1_blk*/, float df, int sub8,
        float4 & w0, float4 & w1) {
        const int off   = (sub8 & 1) * 8;
        const int shift = (sub8 >> 1) * 4;
        const uint32_t q0 = *reinterpret_cast<const uint32_t *>(p0_blk + off);
        const uint32_t q1 = *reinterpret_cast<const uint32_t *>(p0_blk + off + 4);
        w0.x = df * (float) iq_lut::kvalues_iq4nl[((q0 >> ( 0 + shift)) & 0xfu)];
        w0.y = df * (float) iq_lut::kvalues_iq4nl[((q0 >> ( 8 + shift)) & 0xfu)];
        w0.z = df * (float) iq_lut::kvalues_iq4nl[((q0 >> (16 + shift)) & 0xfu)];
        w0.w = df * (float) iq_lut::kvalues_iq4nl[((q0 >> (24 + shift)) & 0xfu)];
        w1.x = df * (float) iq_lut::kvalues_iq4nl[((q1 >> ( 0 + shift)) & 0xfu)];
        w1.y = df * (float) iq_lut::kvalues_iq4nl[((q1 >> ( 8 + shift)) & 0xfu)];
        w1.z = df * (float) iq_lut::kvalues_iq4nl[((q1 >> (16 + shift)) & 0xfu)];
        w1.w = df * (float) iq_lut::kvalues_iq4nl[((q1 >> (24 + shift)) & 0xfu)];
    }
};

template <>
struct IqTraits<MOE_Q_IQ1_S> {
    static constexpr int QK    = 256;
    static constexpr int P0B   = 48; // qs[32] + qh[16]
    static constexpr int P1B   = 0;
    static constexpr bool HAS_D = true;

    __device__ __forceinline__ static void dequant8(
        const uint8_t * p0_blk, const uint8_t * /*p1_blk*/, float df, int sub8,
        float4 & w0, float4 & w1) {
        const int ib = sub8 >> 2;
        const int l  = sub8 & 3;
        const uint16_t qh16 = reinterpret_cast<const uint16_t *>(p0_blk + 32)[ib];
        const float dl    = df * (float) (2 * ((qh16 >> 12) & 7u) + 1);
        const float delta = (qh16 & 0x8000u) ? (-1.0f - iq_lut::IQ1S_DELTA) : (-1.0f + iq_lut::IQ1S_DELTA);
        const uint32_t idx = (uint32_t) p0_blk[sub8] | (((uint32_t) (qh16 >> (3 * l)) & 7u) << 8);
        unpack_iq1_grid(iq_lut::iq1s_grid_gpu[idx], delta, dl, w0, w1);
    }
};

template <>
struct IqTraits<MOE_Q_IQ1_M> {
    static constexpr int QK    = 256;
    static constexpr int P0B   = 48; // qs[32] + qh[16]
    static constexpr int P1B   = 8;  // scales[8]
    static constexpr bool HAS_D = false;

    __device__ __forceinline__ static void dequant8(
        const uint8_t * p0_blk, const uint8_t * p1_blk, float /*df*/, int sub8,
        float4 & w0, float4 & w1) {
        const int ib = sub8 >> 2;
        const int l  = sub8 & 3;
        const uint16_t * sc = reinterpret_cast<const uint16_t *>(p1_blk);
        const uint16_t d_u16 = (uint16_t) ((sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0u) |
                                           ((sc[2] >> 4) & 0x0f00u) | (sc[3] & 0xf000u));
        const float df = __half2float(__ushort_as_half(d_u16));
        const int ib16 = 2 * ib + (l >> 1);
        const float dl = df * (float) (2 * ((sc[ib16 >> 2] >> (3 * (ib16 & 3))) & 7u) + 1);
        const uint8_t qh_b = p0_blk[32 + 2 * ib + (l >> 1)];
        const int sh = 4 * (l & 1);
        const uint32_t idx = (uint32_t) p0_blk[sub8] | (((uint32_t) (qh_b >> sh) & 7u) << 8);
        const float delta = (qh_b & (0x08u << sh)) ? (-1.0f - iq_lut::IQ1M_DELTA) : (-1.0f + iq_lut::IQ1M_DELTA);
        unpack_iq1_grid(iq_lut::iq1s_grid_gpu[idx], delta, dl, w0, w1);
    }
};

template <>
struct IqTraits<MOE_Q_IQ2_XXS> {
    static constexpr int QK    = 256;
    static constexpr int P0B   = 64; // qs[64]
    static constexpr int P1B   = 0;
    static constexpr bool HAS_D = true;

    __device__ __forceinline__ static void dequant8(
        const uint8_t * p0_blk, const uint8_t * /*p1_blk*/, float df, int sub8,
        float4 & w0, float4 & w1) {
        const int ib32 = sub8 >> 2;
        const int l    = sub8 & 3;
        const uint32_t * a32 = reinterpret_cast<const uint32_t *>(p0_blk) + 2 * ib32;
        const uint32_t aux0  = a32[0];
        const uint32_t aux1  = a32[1];
        const float db       = df * (0.5f + (float) (aux1 >> 28)) * 0.25f;
        const uint32_t gidx  = (aux0 >> (8 * l)) & 0xffu;
        const uint32_t signs = iq_lut::ksigns_iq2xs[(aux1 >> (7 * l)) & 127u];
        unpack_u64_signs(iq_lut::iq2xxs_grid[gidx], signs, db, w0, w1);
    }
};

template <>
struct IqTraits<MOE_Q_IQ2_XS> {
    static constexpr int QK    = 256;
    static constexpr int P0B   = 64; // qs[64] (uint16_t[32])
    static constexpr int P1B   = 8;  // scales[8]
    static constexpr bool HAS_D = true;

    __device__ __forceinline__ static void dequant8(
        const uint8_t * p0_blk, const uint8_t * p1_blk, float df, int sub8,
        float4 & w0, float4 & w1) {
        const int ib32 = sub8 >> 2;
        const int l    = sub8 & 3;
        const uint8_t sc_b   = p1_blk[ib32];
        const uint32_t sc_n  = (l < 2) ? (sc_b & 0xfu) : (sc_b >> 4);
        const float db       = df * (0.5f + (float) sc_n) * 0.25f;
        const uint16_t q16   = reinterpret_cast<const uint16_t *>(p0_blk)[sub8];
        const uint32_t signs = iq_lut::ksigns_iq2xs[q16 >> 9];
        unpack_u64_signs(iq_lut::iq2xs_grid[q16 & 511u], signs, db, w0, w1);
    }
};

template <>
struct IqTraits<MOE_Q_IQ2_S> {
    static constexpr int QK    = 256;
    static constexpr int P0B   = 80; // qs[32] + signs[32] + qh[8] + scales[8]
    static constexpr int P1B   = 0;
    static constexpr bool HAS_D = true;

    __device__ __forceinline__ static void dequant8(
        const uint8_t * p0_blk, const uint8_t * /*p1_blk*/, float df, int sub8,
        float4 & w0, float4 & w1) {
        const int ib32 = sub8 >> 2;
        const int l    = sub8 & 3;
        const uint8_t sc_b  = p0_blk[72 + ib32];
        const uint32_t sc_n = (l < 2) ? (sc_b & 0xfu) : (sc_b >> 4);
        const float dl      = df * (0.5f + (float) sc_n) * 0.25f;
        const uint8_t q_b   = p0_blk[sub8];
        const uint8_t s_b   = p0_blk[32 + sub8];
        const uint8_t qh_b  = p0_blk[64 + ib32];
        const uint32_t gidx = (uint32_t) q_b | (((uint32_t) qh_b << (8 - 2 * l)) & 0x300u);
        unpack_u64_signs(iq_lut::iq2s_grid[gidx], s_b, dl, w0, w1);
    }
};

template <>
struct IqTraits<MOE_Q_IQ3_XXS> {
    static constexpr int QK    = 256;
    static constexpr int P0B   = 96; // qs[64] + scales_and_signs[32]
    static constexpr int P1B   = 0;
    static constexpr bool HAS_D = true;

    __device__ __forceinline__ static void dequant8(
        const uint8_t * p0_blk, const uint8_t * /*p1_blk*/, float df, int sub8,
        float4 & w0, float4 & w1) {
        const int ib32 = sub8 >> 2;
        const int l    = sub8 & 3;
        const uint32_t aux32 = reinterpret_cast<const uint32_t *>(p0_blk + 64)[ib32];
        const float db       = df * (0.5f + (float) (aux32 >> 28)) * 0.5f;
        const uint32_t signs = iq_lut::ksigns_iq2xs[(aux32 >> (7 * l)) & 127u];
        const uint32_t g1    = iq_lut::iq3xxs_grid[p0_blk[2 * sub8 + 0]];
        const uint32_t g2    = iq_lut::iq3xxs_grid[p0_blk[2 * sub8 + 1]];
        unpack_u32x2_signs(g1, g2, signs, db, w0, w1);
    }
};

template <>
struct IqTraits<MOE_Q_IQ3_S> {
    static constexpr int QK    = 256;
    static constexpr int P0B   = 96; // qs[64] + signs[32]
    static constexpr int P1B   = 12; // qh[8] + scales[4]
    static constexpr bool HAS_D = true;

    __device__ __forceinline__ static void dequant8(
        const uint8_t * p0_blk, const uint8_t * p1_blk, float df, int sub8,
        float4 & w0, float4 & w1) {
        const int ib32 = sub8 >> 2;
        const int l    = sub8 & 3;
        const uint8_t sc_b  = p1_blk[8 + (ib32 >> 1)];
        const uint32_t sc_n = (ib32 & 1) ? (sc_b >> 4) : (sc_b & 0xfu);
        const float db      = df * (float) (1 + 2 * (int) sc_n);
        const uint8_t qh_b  = p1_blk[ib32];
        const uint8_t s_b   = p0_blk[64 + sub8];
        const uint32_t g1   = iq_lut::iq3s_grid[(uint32_t) p0_blk[2 * sub8 + 0] | (((uint32_t) qh_b << (8 - 2 * l)) & 256u)];
        const uint32_t g2   = iq_lut::iq3s_grid[(uint32_t) p0_blk[2 * sub8 + 1] | (((uint32_t) qh_b << (7 - 2 * l)) & 256u)];
        unpack_u32x2_signs(g1, g2, s_b, db, w0, w1);
    }
};

// Convenience helper: dequantize 8 contiguous weights at logical K offset k0 (k0 % 8 == 0)
// of a single expert row given base row pointers (p0_row, p1_row, d_row).
template <MoEQuantType Q>
__device__ __forceinline__ void dequant_row_k8(
    const uint8_t * p0_row, const uint8_t * p1_row, const __half * d_row,
    int k0, float4 & w0, float4 & w1) {
    using T = IqTraits<Q>;
    const int blk  = k0 / T::QK;
    const int sub8 = (k0 % T::QK) >> 3;
    const float df = T::HAS_D ? __half2float(d_row[blk]) : 0.0f;
    const uint8_t * p1_blk = (T::P1B > 0) ? (p1_row + (size_t) blk * T::P1B) : nullptr;
    T::dequant8(p0_row + (size_t) blk * T::P0B, p1_blk, df, sub8, w0, w1);
}

} // namespace iq_dev
} // namespace inkling
