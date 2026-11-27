#pragma once

// ============================================================================
// nib_arch_fp8.h — RDNA4-only (gfx1200 / gfx1201) FP8 WMMA primitives
// ============================================================================
// Companion to nib_arch.h for the OCP FP8 E4M3 ("e4m3fn": bias 7, no infinities, max 448,
// 0x7F / 0xFF = NaN) matrix path that only RDNA4 implements:
//   - nib_fp8x8_t: one lane's 16x16x16 FP8 WMMA operand (8 E4M3 bytes in 2 VGPRs)
//   - nib_wmma_f32_16x16x16_fp8: v_wmma_f32_16x16x16_fp8_fp8 with FP32 accumulation
//   - scalar/packed E4M3 <-> FP32 conversions (v_cvt_f32_fp8 / v_cvt_pk_fp8_f32)
//   - NVFP4 (E2M1 nibble x UE4M3 block scale) -> E4M3 group decode through an 8-byte table
//
// Compilation contract
// --------------------
// NIB_FP8_CODEGEN is 1 in the host pass and in gfx12 device passes, and 0 in gfx11 device
// passes.  Everything below (and every FP8 kernel) is wrapped in `#if NIB_FP8_CODEGEN`, so an
// RDNA3 device pass of a multi-arch build compiles none of it: its code object carries zero FP8
// kernels and zero stubs, and RDNA3 code generation is byte-identical to a build without this
// file.  The host pass still sees the kernels so it can emit their launch stubs; the host side
// must therefore only launch them on a gfx12 device (HIP aborts on a launch whose symbol is
// missing from the running GPU's code object).
//
// Deliberately avoided (wrong or unavailable on gfx12):
//   - v_pk_mul_f32 (CDNA-only), v_cvt_pk_fp8_f16 (gfx1250-only)
//   - amd_hip_fp8.h's f32x2 -> fp8x2 helpers (incorrect packing in ROCm 7.2.x)
// v_cvt_pk_fp8_f32 is only ever fed values already bounded by +-448 (see nib_fp8_clamp), so its
// overflow behaviour never matters.
//
// FP8 WMMA operand layout (wave32)
// --------------------------------
//   A (16x16, M x K): lane l holds row (l & 15), B (16x16, K x N): lane l holds column (l & 15);
//   each lane holds 8 of the 16 K bytes; Nib's convention is lanes 0..15 K = 0..7 and lanes
//   16..31 K = 8..15 (byte i = K offset i).
//   C/D (FP32): lane l, element i -> row 8 * (l >> 4) + i, column (l & 15)  (same as FP16 WMMA).
// The hardware pairs A byte (half-wave h, byte i) with B byte (h, i), so only the A/B packing
// *consistency* matters for the dot product (any K permutation shared by A and B gives the same
// sum); that is what lets the MoE kernels interleave K in LDS freely.  Test 10a verifies the
// pairing of all 16 byte positions and the C/D layout on the device.
// ============================================================================

#include "nib_arch.h"

#include <cfloat>
#include <cstdint>

#if !defined(__HIP_DEVICE_COMPILE__) || NIB_ARCH_RDNA4
#  define NIB_FP8_CODEGEN 1
#else
#  define NIB_FP8_CODEGEN 0
#endif

namespace inkling {

// Largest finite OCP E4M3 magnitude.
static constexpr float NIB_FP8_MAX = 448.0f;

// Minimum finite row amax for which NIB_FP8_MAX / amax stays strictly finite in FP32 (~2.63e-36);
// rows below this threshold flush to zero scale rather than overflowing 448 / amax to +Inf (which
// would produce 0.0f * +Inf = NaN on zero elements in the row).
static constexpr float NIB_FP8_AMAX_MIN = 2.0f * (NIB_FP8_MAX / FLT_MAX);

// NVFP4 -> E4M3 fold: a weight is e2m1(nibble) * e4m3(scale) * scale2.  Its first two factors
// reach 6 * 448 = 2688 > 448, so the FP8 byte stores them divided by NIB_NVFP4_FP8_FOLD (a power
// of two, so the division is exact) and the GEMM epilogue multiplies by NIB_NVFP4_FP8_FOLD * scale2.
// 2688 / 8 = 336 <= 448.  Measured on Inkling-2-Mini-NVFP4-All (all 6144 expert matrices): each
// expert's own power-of-two shift would also be 8 (scale2 normalises every expert's largest scale
// to ~448), no folded weight is E4M3-subnormal, and the only loss is the RNE step of the products
// that need 5 significant bits (ties of e2m1 {1.5, 3, 6} x an odd-mantissa scale): 0.045% of the
// weight energy (2.1% RMS relative).
static constexpr float NIB_NVFP4_FP8_FOLD = 8.0f;

#if defined(__HIPCC__) && NIB_FP8_CODEGEN

typedef int nib_fp8x8_t __attribute__((ext_vector_type(2))); // 8 E4M3 bytes = one lane's WMMA operand

static __device__ __forceinline__ nib_float8_t nib_wmma_f32_16x16x16_fp8(
        nib_fp8x8_t a, nib_fp8x8_t b, nib_float8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_fp8_fp8_w32_gfx12(a, b, c);
}

// E4M3 byte BYTE (0..3) of `word` -> FP32 (exact; 0x7F / 0xFF -> NaN).
template <int BYTE>
static __device__ __forceinline__ float nib_fp8_to_f32(uint32_t word) {
    static_assert(BYTE >= 0 && BYTE < 4, "byte select");
    return __builtin_amdgcn_cvt_f32_fp8((int) word, BYTE);
}

// Round two FP32 values (RNE) to E4M3 and insert them into the low (HI = false) or high
// (HI = true) 16 bits of `old`.  Inputs must satisfy |x| <= 448 or be NaN (-> E4M3 NaN).
template <bool HI>
static __device__ __forceinline__ uint32_t nib_f32x2_to_fp8x2(float lo, float hi, uint32_t old) {
    return (uint32_t) __builtin_amdgcn_cvt_pk_fp8_f32(lo, hi, (int) old, HI);
}

// Four FP32 values -> four packed E4M3 bytes (byte i = x_i).
static __device__ __forceinline__ uint32_t nib_f32x4_to_fp8x4(float x0, float x1, float x2, float x3) {
    return nib_f32x2_to_fp8x2<true>(x2, x3, nib_f32x2_to_fp8x2<false>(x0, x1, 0u));
}

// Saturate a finite value into the E4M3 range with a single v_med3_f32.  Callers only pass
// finite values (non-finite inputs are handled before, so v_med3_f32's NaN rules never matter).
static __device__ __forceinline__ float nib_fp8_clamp(float x) {
    return __builtin_amdgcn_fmed3f(x, -NIB_FP8_MAX, NIB_FP8_MAX);
}

// ----------------------------------------------------------------------------
// NVFP4 group (16 weights: 8 nibble bytes + 1 UE4M3 scale) -> 16 E4M3 bytes
// ----------------------------------------------------------------------------
// Table: for UE4M3 scale code `sc`, byte m (0..7) of the returned (lo, hi) pair is
//   RNE_e4m3(e2m1_mag[m] * e4m3(sc) / NIB_NVFP4_FP8_FOLD),  e2m1_mag = {0, .5, 1, 1.5, 2, 3, 4, 6}.
// The sign bit of the scale byte is ignored and code 0x7F (E4M3 NaN) yields an all-zero table,
// matching the FP16 path's UE4M3 decode.  Every product is exact in FP32 (<= 6 significant bits)
// and |value| <= 336, so the only rounding is the single RNE step to E4M3 (bit-exact against a
// CPU RNE reference: Test 10b).  The MoE kernels build this table once per CTA in LDS.
static __device__ __forceinline__ uint2 nib_nvfp4_fp8_table(uint32_t sc) {
    const uint32_t u = sc & 0x7Fu;
    const float s = (u == 0x7Fu) ? 0.0f : nib_fp8_to_f32<0>(u) * (1.0f / NIB_NVFP4_FP8_FOLD);
    const uint32_t lo = nib_f32x4_to_fp8x4(0.0f, s * 0.5f, s, s * 1.5f);
    const uint32_t hi = nib_f32x4_to_fp8x4(s * 2.0f, s * 3.0f, s * 4.0f, s * 6.0f);
    return make_uint2(lo, hi);
}

// Decode one nibble dword (8 FP4 codes: byte j holds element j in its low nibble and element
// j + 8 in its high nibble, per the SoA NVFP4 layout) through a table from nib_nvfp4_fp8_table:
//   lo4 = E4M3 bytes of elements 0..3 (byte j = element j), hi4 = elements 8..11.
// Nibble bit 3 is the E2M1 sign and maps to the E4M3 sign bit (nibble 0x8 -> 0x80 = -0).
static __device__ __forceinline__ void nib_nvfp4_expand(uint32_t q, uint2 t, uint32_t & lo4, uint32_t & hi4) {
    lo4 = __builtin_amdgcn_perm(t.y, t.x, q & 0x07070707u) | ((q & 0x08080808u) << 4);
    hi4 = __builtin_amdgcn_perm(t.y, t.x, (q >> 4) & 0x07070707u) | (q & 0x80808080u);
}

// One 16-weight group: nibble dwords q0 (bytes 0..3) and q1 (bytes 4..7) -> 16 E4M3 bytes in
// natural K order: out.x = K 0..3, out.y = K 4..7, out.z = K 8..11, out.w = K 12..15.
static __device__ __forceinline__ uint4 nib_nvfp4_group_to_fp8(uint32_t q0, uint32_t q1, uint2 t) {
    uint32_t a, b, c, d;
    nib_nvfp4_expand(q0, t, a, c);
    nib_nvfp4_expand(q1, t, b, d);
    return make_uint4(a, b, c, d);
}

#endif // __HIPCC__ && NIB_FP8_CODEGEN

} // namespace inkling
