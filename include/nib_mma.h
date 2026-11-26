// ============================================================================
// include/nib_mma.h — 16x16x16 FP16 -> FP32 GEMM tile primitive (arch-dispatch)
//
// gfx1100..gfx1103 (RDNA3, wmma-256b-insts): the native wave32 WMMA builtin, used
// exactly as before (this header changes nothing on gfx11: it only adds names).
// gfx1200 / gfx1201 (RDNA4): the same tile, computed from the same shared-memory
// tiles. Two implementations, selected at runtime:
//   *_emu  explicit FP32-accumulating FP16 dot product (v_dot2_f32_f16).  DEFAULT.
//   *_hw   the real RDNA4 tensor-core instruction (v_wmma_f32_16x16x16_f16) via
//          __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12.  Opt-in only:
//          NIB_GFX12_WMMA=1 in the environment (see the gate at the bottom).
//   mma_*  dispatcher: *_hw when the gate is on, else *_emu.
// The default stays *_emu until the gfx12 fragment layout has been validated on
// silicon (`nib wmma-selftest`, src/kernels_wmma_prefill.hip).
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
// already holds. nib_mma::*_emu below consume exactly that and accumulate into the
// native D layout, so every existing epilogue (softmax, FWHT-128, stores) keeps
// working unchanged on gfx12.
//
// ----------------------------------------------------------------------------
// RDNA4 (gfx12) wave32 V_WMMA_F32_16x16x16_F16 fragment layout
// ----------------------------------------------------------------------------
// Builtin (ROCm 7.2.0 / clang 22, /opt/rocm/include/ck/utility/amd_wmma.hpp and
// ck_tile/ops/gemm/warp/warp_gemm_attribute_wmma_impl_base_traits.hpp):
//   nib_f16x8 __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(
//       nib_f16x8 a, nib_f16x8 b, nib_f32x8 c);          // -> v_wmma_f32_16x16x16_f16
// A/B are 8 halves per lane (4 VGPRs, no RDNA3-style lane duplication), C/D are 8
// f32 per lane: 32*8 = 256 = 16*16, i.e. the tile is loaded exactly once.
// With laneCol = lane % 16 and laneGrp = lane / 16, the layout is
//   A: a[i] = A[m = laneCol][k = 8*laneGrp + i]
//   B: b[i] = B[k = 8*laneGrp + i][n = laneCol]      (A and B are both K-major:
//                                                     both hold 8 *contiguous* K)
//   D: d[i] = D[m = 8*laneGrp + i][n = laneCol]
// (AMD GPUOpen "Using the Matrix Cores of AMD RDNA 4 architecture GPUs", whose
// sample kernel loads a_frag/b_frag and stores c exactly like that; the D mapping
// is also what CK's wmma_gemm.hpp documents for gfx12 and what the independent
// gfx1201 measurements in the rdna4-wmma-guide reproduce.)
//
// Only D differs from the nib/gfx11 layout, and it differs by a *row permutation*
// inside the tile - never by cross-lane traffic:
//   hw keeps D row 8*laneGrp + i in acc[i], nib wants D row 2*i + laneGrp.
// Feeding the A fragment of hw row r from nib row  perm(r) = 2*(r % 8) + r / 8
// makes hw compute D'[r] = D[perm(r)], hence acc[i] = D'[8*laneGrp + i] =
// D[2*i + laneGrp] = nib's acc[i] with no shuffles at all:
//   perm(0..15) = 0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15
//   (laneGrp 0 -> even D rows, laneGrp 1 -> odd D rows, exactly the nib mapping).
// B needs no permutation: nib's B fragment already is B[0..15][laneCol], so a lane
// group just picks the low or the high 8 halves of the 16 it holds.
// ============================================================================
#pragma once

#include <cstdio>
#include <cstdlib>
#include <hip/hip_runtime.h>   // uint4/__half2 in the fragments, and the NIB_GFX12_WMMA gate
#include <hip/hip_fp16.h>

#include "nib_target.h"

#if NIB_RDNA3
#define NIB_MMA_NATIVE 1
#else
#define NIB_MMA_NATIVE 0
#endif

// Targets whose 16x16x16 f16 tile this header can produce (native or emulated).
#if NIB_RDNA3 || NIB_RDNA4
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
typedef _Float16 nib_f16x8  __attribute__((ext_vector_type(8)));
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

// ===========================================================================
// RDNA4 (gfx12) tensor-core tile
// ===========================================================================
#if NIB_RDNA4
// hw row -> nib row of the 16x16 tile (see the header comment; involution).
static __device__ __forceinline__ int d_row_perm(int r) { return 2 * (r & 7) + (r >> 3); }

// 8 halves = one 16 B LDS/global load worth of fragment (ds_load_b128 /
// global_load_b128).
// memcpy keeps this correct for any pointer alignment; clang narrows it to the
// widest provably aligned load, and every nib call site (16-half A groups on
// >= 8-half strides) gets a single b128 move -- checked in the post-change ISA.
static __device__ __forceinline__ nib_f16x8 load8(const _Float16 * p) {
    nib_f16x8 r;
    __builtin_memcpy(&r, p, 16);
    return r;
}

static __device__ __forceinline__ nib_f16x8 low8(nib_f16x16 v) {
    nib_f16x8 r;
    __builtin_memcpy(&r, &v, 16);
    return r;
}

static __device__ __forceinline__ nib_f16x8 high8(nib_f16x16 v) {
    nib_f16x8 r;
    __builtin_memcpy(&r, reinterpret_cast<const char *>(&v) + 16, 16);
    return r;
}

// A 16-half nib fragment (all 16 K of one row/bias row) -> the lane group's 8 K:
//   B: nib holds B[k = 0..15][n = lane%16], hw wants b[i] = B[k = 8*laneGrp + i][n].
//   A: nib holds A[m][k = 0..15],         hw wants a[i] = A[m][k = 8*laneGrp + i].
// Both are "take the low half for lanes 0..15, the high half for lanes 16..31".
static __device__ __forceinline__ nib_f16x8 pick8(nib_f16x16 v, int lane) {
    return (lane & 16) ? high8(v) : low8(v);
}

// D += A(B fragment) * B(hw fragment), accumulating in the nib D layout.
static __device__ __forceinline__ void wmma_tile(nib_f16x8 a, nib_f16x8 b, nib_f32x8 & acc) {
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, acc);
}

// Hardware gate flag: 0 = v_dot2 emulation (default), 1 = tensor core.
// One private flag per translation unit (nib builds one code object per .hip file and
// links no device code across them, no -fgpu-rdc), written only by the setter kernel
// below. That store is what keeps the flag from being constant-folded to 0 (a
// `static` device variable with no device-side writer is folded at -O3, dropping the
// path entirely); with it the read compiles to one scalar s_load per tile.
static __device__ int nib_mma_gfx12_wmma_flag = 0;

static __global__ void nib_mma_gfx12_wmma_set_kernel(int on) { nib_mma_gfx12_wmma_flag = on; }

static __device__ __forceinline__ bool hw_tile_on() { return nib_mma_gfx12_wmma_flag != 0; }
#endif // NIB_RDNA4

// ---------------------------------------------------------------------------
// Tile updates: acc[i] += <A[row 2*i + lane/16], B> with B the lane's own 16-half
// B row (n = lane % 16). Plain row-major LDS tile, `astride` halves between rows,
// `a_row0` = row 0 of the 16-row A group at the K chunk being consumed.
// ---------------------------------------------------------------------------
// Emulation (always available, the default on gfx12).
static __device__ __forceinline__ void mma_stride_emu(
        const _Float16 * a_row0, int astride, nib_f16x16 b, nib_f32x8 & acc, int lane) {
    const int grp = lane >> 4;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i] += dot16(load16(a_row0 + (2 * i + grp) * astride), b);
    }
}

// Tensor-core tile: one 8-half A read + one v_wmma per tile.
// gfx12: A row fed for hw row r is perm(r) (see d_row_perm), k = 8*(lane/16).
// gfx11: the native builtin wants the lane's own row (lane%16), all 16 K halves.
static __device__ __forceinline__ void mma_stride_hw(
        const _Float16 * a_row0, int astride, nib_f16x16 b, nib_f32x8 & acc, int lane) {
#if NIB_RDNA4
    const nib_f16x8 a8 = load8(a_row0 + d_row_perm(lane & 15) * astride + 8 * (lane >> 4));
    wmma_tile(a8, pick8(b, lane), acc);
#elif NIB_RDNA3
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(load16(a_row0 + (lane & 15) * astride), b, acc);
#else
    mma_stride_emu(a_row0, astride, b, acc, lane);
#endif
}

static __device__ __forceinline__ void mma_stride(
        const _Float16 * a_row0, int astride, nib_f16x16 b, nib_f32x8 & acc, int lane) {
#if NIB_RDNA4
    if (hw_tile_on()) { mma_stride_hw(a_row0, astride, b, acc, lane); return; }
#endif
    mma_stride_emu(a_row0, astride, b, acc, lane);
}

// XOR-swizzled 64-half rows (see load16_xor); `a_row0` = row 0 of the A group.
static __device__ __forceinline__ void mma_xor64_emu(
        const _Float16 * a_row0, nib_f16x16 b, nib_f32x8 & acc, int lane, int kc) {
    const int grp = lane >> 4;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int m = 2 * i + grp;
        acc[i] += dot16(load16_xor(a_row0 + m * NIB_MMA_XOR_ROW, m & 7, kc), b);
    }
}

static __device__ __forceinline__ void mma_xor64_hw(
        const _Float16 * a_row0, nib_f16x16 b, nib_f32x8 & acc, int lane, int kc) {
#if NIB_RDNA4
    typedef int v4i __attribute__((ext_vector_type(4)));
    const int row = d_row_perm(lane & 15);
    const v4i * d4 = reinterpret_cast<const v4i *>(a_row0 + row * NIB_MMA_XOR_ROW);
    // k = 8*(lane/16) inside 16-half chunk kc == block (2*kc + lane/16) ^ (row & 7).
    const v4i a = d4[(2 * kc + (lane >> 4)) ^ (row & 7)];
    nib_f16x8 a8;
    __builtin_memcpy(&a8, &a, 16);
    wmma_tile(a8, pick8(b, lane), acc);
#elif NIB_RDNA3
    const int row = lane & 15;
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(
        load16_xor(a_row0 + row * NIB_MMA_XOR_ROW, row & 7, kc), b, acc);
#else
    mma_xor64_emu(a_row0, b, acc, lane, kc);
#endif
}

static __device__ __forceinline__ void mma_xor64(
        const _Float16 * a_row0, nib_f16x16 b, nib_f32x8 & acc, int lane, int kc) {
#if NIB_RDNA4
    if (hw_tile_on()) { mma_xor64_hw(a_row0, b, acc, lane, kc); return; }
#endif
    mma_xor64_emu(a_row0, b, acc, lane, kc);
}

// Row accessor variant for layouts whose per-row addressing is not a constant
// stride (the transposed/swizzled V_T tiles): acc[i] += <row(2*i + lane/16), b>.
// F is a device lambda taking the row index and returning nib_f16x16.
template <typename F>
static __device__ __forceinline__ void mma_rows_emu(F row_of, nib_f16x16 b, nib_f32x8 & acc, int lane) {
    const int grp = lane >> 4;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        acc[i] += dot16(row_of(2 * i + grp), b);
    }
}

template <typename F>
static __device__ __forceinline__ void mma_rows_hw(F row_of, nib_f16x16 b, nib_f32x8 & acc, int lane) {
#if NIB_RDNA4
    // The accessor returns 16 halves; the hw tile only needs 8 of them per lane.
    wmma_tile(pick8(row_of(d_row_perm(lane & 15)), lane), pick8(b, lane), acc);
#elif NIB_RDNA3
    acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(row_of(lane & 15), b, acc);
#else
    mma_rows_emu(row_of, b, acc, lane);
#endif
}

template <typename F>
static __device__ __forceinline__ void mma_rows(F row_of, nib_f16x16 b, nib_f32x8 & acc, int lane) {
#if NIB_RDNA4
    if (hw_tile_on()) { mma_rows_hw(row_of, b, acc, lane); return; }
#endif
    mma_rows_emu(row_of, b, acc, lane);
}

} // namespace nib_mma

// ===========================================================================
// Runtime gate: NIB_GFX12_WMMA=1 selects the tensor-core tile on gfx12.
// The default (unset / 0) keeps the validated v_dot2 emulation, and when the
// variable is not set this initializer does not touch the GPU at all.
// ===========================================================================
#if NIB_RDNA4 && !defined(__HIP_DEVICE_COMPILE__)
namespace nib_mma {

// Defined in src/engine.cpp (and declared identically in inkling_engine.h): the
// host-side list of per-TU gate appliers, run once the HIP runtime is up.
void register_gate_applier(void (*f)());

static inline bool env_wants_hw_tile() {
    const char * e = std::getenv("NIB_GFX12_WMMA");
    if (e == nullptr || e[0] == '\0') return false;
    return std::atoi(e) != 0;
}

// Set this TU's gate flag from the host. Runs once, after the HIP runtime (and this
// TU's code object) are up - see nib_mma::apply_gate_appliers().
static void apply_env_gate() {
    nib_mma_gfx12_wmma_set_kernel<<<1, 1>>>(1);
    const hipError_t st = hipGetLastError();
    if (st != hipSuccess) {
        std::fprintf(stderr, "[nib_mma] NIB_GFX12_WMMA=1 ignored: setter kernel failed (%s); "
                             "keeping the v_dot2 tile\n", hipGetErrorString(st));
    } else {
        std::fprintf(stderr, "[nib_mma] NIB_GFX12_WMMA=1: gfx12 hardware WMMA tile ENABLED "
                             "(validate with `nib wmma-selftest`)\n");
    }
}

// Registration only, no GPU call: a header-level static initializer cannot safely
// touch the device (the module may not be registered yet), so the engine (or the
// selftest) runs the appliers instead.
struct env_gate_reg {
    env_gate_reg() { if (env_wants_hw_tile()) register_gate_applier(&apply_env_gate); }
};
__attribute__((unused)) static env_gate_reg g_env_gate_reg;

} // namespace nib_mma
#endif

