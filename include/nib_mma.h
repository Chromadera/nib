// ============================================================================
// include/nib_mma.h — 16x16x16 FP16 -> FP32 GEMM tile primitive (arch-dispatch)
//
// gfx1100..gfx1103 (RDNA3, wmma-256b-insts): the native wave32 WMMA builtin, used
// exactly as before (this header changes nothing on gfx11: it only adds names).
// gfx1200 / gfx1201 (RDNA4): the same tile, computed from the same shared-memory
// tiles with an explicit FP32-accumulating FP16 dot product (v_dot2_f32_f16 where
// the target has it). Correctness first: no fragment-layout re-derivation, no
// gfx12 WMMA staging — the tile math is written out.
//
// Native (gfx11) wave32 16x16x16 f16 -> f32 fragment mapping, as documented and
// unit-tested in kernels_wmma_prefill.hip:
//   A (16x16, M x K): lane l holds row (l % 16) as 16 contiguous K halves;
//                     lanes 16..31 replicate lanes 0..15.
//   B (16x16, K x N): lane l holds row (l % 16) of the row-major [n][k] tile as
//                     16 contiguous K halves (B column n of W^T is row n of W).
//   D (16x16, FP32) : lane l, element i -> row (2*i + l/16), column (l % 16).
//
// So one tile needs no cross-lane traffic at all: per lane, 8 A rows
// (m = 2*i + l/16) of 16 contiguous halves, and the single B row (n = l % 16) it
// already holds. nib_mma_* below consume exactly that and accumulate into the
// native D layout, so every existing epilogue (softmax, FWHT-128, stores) keeps
// working unchanged on gfx12.
// ============================================================================
#pragma once

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
#define NIB_MMA_NATIVE 1
#else
#define NIB_MMA_NATIVE 0
#endif

// Targets whose 16x16x16 f16 tile this header can produce (native or emulated).
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__) || \
    defined(__gfx1200__) || defined(__gfx1201__)
#define NIB_MMA_TILE_TARGET 1
#else
#define NIB_MMA_TILE_TARGET 0
#endif

// LDS row stride in halves of the XOR-swizzled 16-half-block layouts (LDK_XOR).
#define NIB_MMA_XOR_ROW 64
// Number of 8-half (16 B) blocks per NIB_MMA_XOR_ROW row.
#define NIB_MMA_XOR_BLKS 8

namespace nib_mma {

typedef _Float16 nib_f16x16 __attribute__((ext_vector_type(16)));
typedef float    nib_f32x8  __attribute__((ext_vector_type(8)));

// 16 contiguous halves (32 B, one ds_load_b128 pair) -> fragment word.
static __device__ __forceinline__ nib_f16x16 load16(const _Float16 * p) {
    const uint4 v0 = *reinterpret_cast<const uint4 *>(p + 0);
    const uint4 v1 = *reinterpret_cast<const uint4 *>(p + 8);
    nib_f16x16 r;
    __builtin_memcpy(&r, &v0, 16);
    __builtin_memcpy(reinterpret_cast<char *>(&r) + 16, &v1, 16);
    return r;
}

// 16 halves of row `row_ptr` for the XOR-swizzled 64-half layout: 16-half chunk
// `kc` lives in 16 B blocks (2*kc+0) ^ (row & 7) and (2*kc+1) ^ (row & 7).
// `row_low3` is (row & 7) of the row being read (the writer swizzles with its own).
static __device__ __forceinline__ nib_f16x16 load16_xor(const _Float16 * row_ptr, int row_low3, int kc) {
    typedef int v4i __attribute__((ext_vector_type(4)));
    const v4i * d4 = reinterpret_cast<const v4i *>(row_ptr);
    const v4i a = d4[(2 * kc + 0) ^ row_low3];
    const v4i b = d4[(2 * kc + 1) ^ row_low3];
    nib_f16x16 r;
    __builtin_memcpy(&r, &a, 16);
    __builtin_memcpy(reinterpret_cast<char *>(&r) + 16, &b, 16);
    return r;
}

// sum(a[i] * b[i]) for i = 0..15, accumulated in FP32 (tensor-core-equivalent
// operand handling: FP16 inputs, FP32 accumulation).
static __device__ __forceinline__ float dot16(nib_f16x16 a, nib_f16x16 b) {
    union U { nib_f16x16 h; __half2 h2[8]; uint32_t u[8]; } ua, ub;
    ua.h = a;
    ub.h = b;
    float s = 0.0f;
#if __has_builtin(__builtin_amdgcn_fdot2)
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        s = __builtin_amdgcn_fdot2(ua.h2[i], ub.h2[i], s, false);
    }
#else
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float2 fa = __half22float2(ua.h2[i]);
        const float2 fb = __half22float2(ub.h2[i]);
        s = fmaf(fa.x, fb.x, fmaf(fa.y, fb.y, s));
    }
#endif
    return s;
}

// ---------------------------------------------------------------------------
// Tile updates: acc[i] += <A[row 2*i + lane/16], B> with B the lane's own 16-half
// B row (n = lane % 16). Plain row-major LDS tile, `astride` halves between rows,
// `a_row0` = row 0 of the 16-row A group at the K chunk being consumed.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void mma_stride(
        const _Float16 * a_row0, int astride, nib_f16x16 b, nib_f32x8 & acc, int lane) {
    const int hw = lane >> 4;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i] += dot16(load16(a_row0 + (2 * i + hw) * astride), b);
    }
}

// XOR-swizzled 64-half rows (see load16_xor); `a_row0` = row 0 of the A group.
static __device__ __forceinline__ void mma_xor64(
        const _Float16 * a_row0, nib_f16x16 b, nib_f32x8 & acc, int lane, int kc) {
    const int hw = lane >> 4;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = 2 * i + hw;
        acc[i] += dot16(load16_xor(a_row0 + m * NIB_MMA_XOR_ROW, m & 7, kc), b);
    }
}

// Row accessor variant for layouts whose per-row addressing is not a constant
// stride (the transposed/swizzled V_T tiles): acc[i] += <row(2*i + lane/16), b>.
// F is a device lambda taking the row index and returning nib_f16x16.
template <typename F>
static __device__ __forceinline__ void mma_rows(F row_of, nib_f16x16 b, nib_f32x8 & acc, int lane) {
    const int hw = lane >> 4;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i] += dot16(row_of(2 * i + hw), b);
    }
}

} // namespace nib_mma
