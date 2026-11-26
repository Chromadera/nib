// ============================================================================
// kernels_moe_v3.hip — Step 7F: 850+ GB/s weight-streaming decode kernels
//                      (gfx1100 / RX 7900 XTX, Wave32, 96 CU, 960 GB/s GDDR6)
//
// Self-contained: depends only on include/inkling_engine.h.  No existing file
// is modified; the four entry points are drop-in A/B replacements (see
// include/kernels_moe_v3.h for the contract list).
//
// ============================================================================
// ROOT CAUSE: what separates lm_head (884 GB/s) from the three slow kernels
// ============================================================================
// Offline ISA extraction of build/kernels_gemv_soa.o
//   (llvm-objcopy --dump-section .hip_fatbin / clang-offload-bundler /
//    llvm-readobj --notes / llvm-objdump -d --mcpu=gfx1100):
//
//   kernel                      VGPR SGPR  LDS(B)  priv  spill | waves/SIMD
//   soa_q8_gemv  <8,8,72,3>       74   19  10,368    0    0/0  |  6   (lm_head)
//   soa_q8_gemv  <8,32,96,1>      51   19  13,824    0    0/0  |  8*  (wo, PD=1)
//   moe_nvfp4_gate_up_situ<4>    131   20   9,792    0    0/0  |  3
//   moe_nvfp4_down_wsum<1,288>   105   34  12,416    0    0/0  |  4
//   moe_q8_gate_up_situ<8,1>      95   18  10,368    0    0/0  |  5 (->4 with 8 warps)
//   moe_q8_down_wsum<2,192>      120   34  12,416    0    0/0  |  4 (->2 with 8 warps)
//
// (waves/SIMD = min(8, floor(512 / roundup8(VGPR))) on gfx11; a wave32 block of
//  4 warps maps 1 warp per SIMD, 8 warps map 2 per SIMD, so
//  blocks/CU = waves_per_SIMD / (warps/block).)
//
// Three measurable consequences, in order of size:
//
//  (a) grid-imposed occupancy, not LDS.  gate_up is launched as 288 blocks x
//      4 warps = 1152 warps because a pair is reduced over only 4 lanes
//      (8 pairs/warp).  1152 warps / (96 CU * 4 SIMD) = exactly 3 warps per
//      SIMD no matter what the register file could hold, and the 131-VGPR
//      allocation happens to allow exactly 3 blocks/CU = 3 waves/SIMD.
//      lm_head has 4 lanes -> LPR 8 (4 rows/warp, 32 rows/block) so it has
//      8x more warps and sits at 6 waves/SIMD.
//  (b) memory-level parallelism per warp.  Every one of these kernels issues
//      its weights as `global_load_b128 slc dlc` (perfect streaming, 50-100%
//      sector efficiency, verified in the disassembly) and then *waits*.  The
//      ratio is what differs: lm_head keeps 3 iterations (2 blocks each) in a
//      rotating VGPR ring and pre-loads the next row's first blocks while the
//      current row's 8 q8_word_dot_mix chains run; wo's ring depth is PD = 1,
//      so its 3 blocks are load -> 72 VALU -> wait -> load -> ... with nothing
//      overlapping and only 6.6 KB of weights in flight per CU.
//  (c) activation staging.  nvfp4 down keeps 48 of its 72 activation floats in
//      VGPRs (rx0[8], rx1[4]) and the other 24 in 12,416 B of LDS - that is
//      what costs it the 105-120 VGPRs and the per-block LDS barrier.  The
//      activation for ONE token is only 8 x 1152 x 4 B = 36.9 KB, i.e. it is
//      L2-resident and mostly L1-resident; the weight stream is non-temporal
//      (`slc dlc`) and therefore never evicts it.
//
// ============================================================================
// v3 MEASURED-OFFLINE PROPERTIES (llvm-readobj --notes on THIS object)
// ============================================================================
//   kernel                     grid x thr  VGPR SGPR LDS(B) priv spill vspill
//   nvfp4_gate_up_v3_l8p2      288 x 256    70   18  10,368  0    0/0    0
//   nvfp4_gate_up_v3_l4p3(ALT) 288 x 128    80   18  10,368  0    0/0    0
//   nvfp4_down_v3              384 x 256    50   34     128  0    0/0    0
//   nvfp4_down_v3 XREG (ALT)   384 x 256    95   34     128  0    0/0    0
//   nvfp4_fused_v3             288 x 256    64   46  10,496  0    0/0    0
//   q8_gate_up_v3_l8p2         288 x 256    61   18  10,368  0    0/0    0
//   q8_gate_up_v3_l4p3(ALT)    288 x 128   100   18  10,368  0    0/0    0
//   q8_down_v3                 384 x 256    56   34     128  0    0/0    0
//   q8_fused_v3                288 x 256    62   34  10,496  0    0/0    0
//   wo_v3 (W1)                 288 x 256    69   25  13,824  0    0/0    0
//   wo_v3 (W2)                 576 x 128    78   26       0  0    0/0    0
//
//   kernel                     waves/SIMD  blocks/CU (grid)  warps/CU
//   nvfp4_gate_up_v3_l8p2          7            3              24   (6 waves)
//   nvfp4_down_v3                  8            4              32   (8 waves)
//   nvfp4_fused_v3                 8            3 (cap 4)      24   (6 waves)
//   q8_gate_up_v3_l8p2             8            3              24   (6 waves)
//   q8_down_v3                     8            4              32   (8 waves)
//   q8_fused_v3                    8            3 (cap 4)      24   (6 waves)
//   wo_v3 (W1)                     7            3              24   (6 waves)
//   wo_v3 (W2)                     6            6              24   (6 waves)
//
// All weight loads are `global_load_b128 slc dlc` (16 B, 2 per 64-weight NVFP4
// block / per 32-weight Q8 block); every 64-weight block of a 32-lane NVFP4
// down wave therefore reads 8 x 64 B fully-contiguous chunks.  Activations use
// ordinary cached loads (no flags) so they stay in L1/L2.  No `flat_`,
// `scratch_` or spill instruction is emitted anywhere (checked with
// llvm-objdump -d).
// ============================================================================

#include "kernels_moe_v3.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace inkling {
namespace {

typedef int          v4i __attribute__((ext_vector_type(4)));
typedef unsigned int v2u __attribute__((ext_vector_type(2)));

// ---------------------------------------------------------------------------
// Device helpers — every one of these is byte-for-byte the same instruction
// sequence as kernels_gemv_soa.hip / kernels_decode_fused.hip so that the
// arithmetic of the v3 kernels is identical (not merely equivalent).
// ---------------------------------------------------------------------------

// 128-bit non-temporal load: emits `global_load_b128 ... slc dlc` (streaming,
// no L1 allocation -> the activation stays resident in L1).
__device__ __forceinline__ v4i ld_nt_b128(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const v4i *>(p));
}

__device__ __forceinline__ uint32_t ld_nt_b32(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const uint32_t *>(p));
}

__device__ __forceinline__ v2u ld_nt_b64(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const v2u *>(p));
}

// Cacheable (L2-allocating) variants: MOE_V3_CACHE tests whether the repeated
// per-token decode weight stream stays in the 96 MB Infinity Cache when the
// `slc` evict-first hint is dropped.  Identical values, only the load policy
// differs, so the arithmetic is bit-identical to the AUTO shape.
template <bool NT>
__device__ __forceinline__ v4i ld_w_b128(const void * p) {
    return NT ? ld_nt_b128(p) : *reinterpret_cast<const v4i *>(p);
}
template <bool NT>
__device__ __forceinline__ uint32_t ld_w_b32(const void * p) {
    return NT ? ld_nt_b32(p) : *reinterpret_cast<const uint32_t *>(p);
}
template <bool NT>
__device__ __forceinline__ v2u ld_w_b64(const void * p) {
    return NT ? ld_nt_b64(p) : *reinterpret_cast<const v2u *>(p);
}

// Exact branchless UE4M3 -> FP32 (returns raw E4M3 * 0.5, kvalues are 2x).
// Written as two explicit value selects instead of one nested `?:` so the
// backend emits v_cndmask_b32 rather than two s_and_saveexec regions per
// scale.  The divergent form costs ~14 instructions per conversion and, much
// worse, forces every live value across the exec-mask regions (that alone was
// worth +60 VGPRs in the down kernel).
__device__ __forceinline__ float ue4m3_to_fp32(uint32_t x) {
    const uint32_t u   = x & 0x7Fu;
    const uint32_t e   = u >> 3;
    const uint32_t m   = u & 7u;
    const float    nrm = __uint_as_float(((e + 119u) << 23) | (m << 20));
    const float    sub = (float) m * (1.0f / 1024.0f);
    const float    r   = (e == 0u) ? sub : nrm;
    return (u == 0x7Fu) ? 0.0f : r;
}

__device__ __forceinline__ float situ_scalar(float gate, float up, float beta, float inv_beta,
                                             float lb, float inv_lb) {
    const float act = beta * tanhf(gate * inv_beta) / (1.0f + expf(-gate));
    const float u   = lb > 0.0f ? lb * tanhf(up * inv_lb) : up;
    return act * u;
}

__device__ __forceinline__ float situ_default(float g, float u) {
    return situ_scalar(g, u, SITU_BETA, 1.0f / SITU_BETA, SITU_LINEAR_BETA, 1.0f / SITU_LINEAR_BETA);
}

// NVFP4 nibbles -> FP16 high bytes via v_perm_b32 on an 8-entry magnitude LUT.
// kvalues {0,1,2,3,4,6,8,12} are exact FP16 with a zero low byte, so the high
// byte alone encodes them and the sign nibble bit is the FP16 sign bit.
__device__ __forceinline__ void nvfp4_hbytes(uint32_t q, uint32_t & lo, uint32_t & hi) {
    constexpr uint32_t T0 = 0x42403C00u; // mags 0,1,2,3
    constexpr uint32_t T1 = 0x4A484644u; // mags 4,6,8,12
    lo = __builtin_amdgcn_perm(T1, T0, q & 0x07070707u)        | ((q & 0x08080808u) << 4);
    hi = __builtin_amdgcn_perm(T1, T0, (q >> 4) & 0x07070707u) | (q & 0x80808080u);
}

// 4 FP16 high bytes x 4 FP32 activations -> v_fma_mix_f32 (1 op per weight,
// exact: the FP16 operand is widened inside the FMA, no v_cvt_f32_f16).
__device__ __forceinline__ float nvfp4_hb_dot_mix(uint32_t hb, const float4 x, float acc) {
    const __half2 h01 = __builtin_bit_cast(__half2, __builtin_amdgcn_perm(hb, 0u, 0x050C040Cu));
    const __half2 h23 = __builtin_bit_cast(__half2, __builtin_amdgcn_perm(hb, 0u, 0x070C060Cu));
    acc = fmaf(__low2float(h01),  x.x, acc);
    acc = fmaf(__high2float(h01), x.y, acc);
    acc = fmaf(__low2float(h23),  x.z, acc);
    acc = fmaf(__high2float(h23), x.w, acc);
    return acc;
}

// Unscaled dot of one 16-element NVFP4 sub-block (2 packed words) with 16 floats.
__device__ __forceinline__ float nvfp4_sub_dot(uint32_t q0, uint32_t q1,
                                               const float4 x0, const float4 x1,
                                               const float4 x2, const float4 x3) {
    uint32_t lo0, hi0, lo1, hi1;
    nvfp4_hbytes(q0, lo0, hi0); // elems 0..3, 8..11
    nvfp4_hbytes(q1, lo1, hi1); // elems 4..7, 12..15
    float acc = 0.0f;
    acc = nvfp4_hb_dot_mix(lo0, x0, acc);
    acc = nvfp4_hb_dot_mix(lo1, x1, acc);
    acc = nvfp4_hb_dot_mix(hi0, x2, acc);
    acc = nvfp4_hb_dot_mix(hi1, x3, acc);
    return acc;
}

// Exact signed int8 x FP32 dot: int8 -> FP16 is exact (0x6400|(q+128) - 1152)
// and fmaf(half, float, float) is v_fma_mix_f32 (bit-identical to
// (float)q * x + acc, 9 VALU ops per 4 weights instead of 12).
__device__ __forceinline__ float q8_word_dot_mix(uint32_t w, const float4 x, float acc) {
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

template <int W>
__device__ __forceinline__ float group_reduce_sum(float v) {
#pragma unroll
    for (int off = W / 2; off > 0; off >>= 1) {
        v += __shfl_xor(v, off, W);
    }
    return v;
}

// Fused 4-tap static short-conv epilogue (identical to sconv_post_res_kernel).
struct SconvEpi {
    const float * conv_w;
    float *       hist;
    const float * res;
};

__device__ __forceinline__ float sconv_post_res_one(const SconvEpi & e, int r, float x3) {
    const float4 w  = reinterpret_cast<const float4 *>(e.conv_w)[r];
    float *      hs = e.hist + (size_t) r * N_CONV_HIST;
    const float x0 = hs[0], x1 = hs[1], x2 = hs[2];
    const float acc = x3 + x0 * w.x + x1 * w.y + x2 * w.z + x3 * w.w;
    const float val = (e.res != nullptr) ? fmaf(acc, 0.4472135954999579f, e.res[r])
                                         : acc * 0.4472135954999579f; // 1/sqrt(5)
    hs[0] = x1;
    hs[1] = x2;
    hs[2] = x3;
    return val;
}

// ---------------------------------------------------------------------------
// gfx1100 memory-ordering + self-resetting barrier helpers (identical to
// kernels_decode_fused.hip, where they are battle-tested by kernels A/B/C/E).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void rdna3_flush_stores() {
    nib_flush_stores();
}

__device__ __forceinline__ void rdna3_inv_l0_l1() {
    nib_inv_l0_l1();
}

__device__ __forceinline__ int scratch_add(float * scratch, int slot) {
    return __hip_atomic_fetch_add(reinterpret_cast<int32_t *>(scratch + slot), 1,
                                  __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ int scratch_load(float * scratch, int slot) {
    return __hip_atomic_load(reinterpret_cast<int32_t *>(scratch + slot),
                             __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void scratch_store(float * scratch, int slot, int v) {
    __hip_atomic_store(reinterpret_cast<int32_t *>(scratch + slot), v,
                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

__device__ __forceinline__ void spin_until(float * scratch, int slot, int target) {
    while (scratch_load(scratch, slot) < target) {
        __builtin_amdgcn_s_sleep(1);
    }
}

// Scratch slots owned by this module (buf_router_scratch_ is zeroed once at
// init; A uses 260..391, B/C use 0..311, E uses 392..395, so 400.. is free).
constexpr int V3_S_BARRIER = 400; // 400..403 shards, 404 ack (all self-resetting)

// Grid barrier: every participating block must be co-resident (the launchers
// only use this when gridDim.x <= resident capacity, see the launch table).
// Producers flush before the release add, consumers invalidate GL0/GL1 after
// the spin (gfx11 L1 is not coherent for global stores).
template <int SHARDS>
__device__ __forceinline__ void grid_barrier(float * scratch, int nblocks) {
    rdna3_flush_stores();
    __syncthreads();
    if (threadIdx.x == 0) {
        rdna3_flush_stores();
        (void) scratch_add(scratch, V3_S_BARRIER + (blockIdx.x % SHARDS));
        const int target = (nblocks + SHARDS - 1) / SHARDS;
#pragma unroll
        for (int i = 0; i < SHARDS; ++i) {
            spin_until(scratch, V3_S_BARRIER + i, target);
        }
        // Last block through: everyone has already passed all spins, so the
        // counters can be zeroed for the next graph replay.
        const int prev = scratch_add(scratch, V3_S_BARRIER + SHARDS);
        if (prev == nblocks - 1) {
#pragma unroll
            for (int i = 0; i < SHARDS; ++i) {
                scratch_store(scratch, V3_S_BARRIER + i, 0);
            }
            scratch_store(scratch, V3_S_BARRIER + SHARDS, 0);
        }
    }
    __syncthreads();
    rdna3_inv_l0_l1();
}

// ---------------------------------------------------------------------------
// Compile-time shapes
// ---------------------------------------------------------------------------
constexpr int NV_K    = N_EMBD;          // 2304   (gate_up contraction length)
constexpr int NV_QSR  = NV_K / 2;        // 1152 B (qs bytes per row)
constexpr int NV_SCR  = NV_K / 16;       // 144 B  (UE4M3 scale bytes per row)

constexpr int DN_K    = N_FF_EXP;        // 1152   (down contraction length)
constexpr int DN_QSR  = DN_K / 2;        // 576 B
constexpr int DN_SCR  = DN_K / 16;       // 72 B

constexpr int Q8_K    = N_EMBD;          // 2304
constexpr int Q8_NB   = Q8_K / 32;       // 72     (32-weight blocks per row)
constexpr int Q8_DS   = Q8_K / 32;       // 72 half scales per row
constexpr int Q8_LO   = 9;               // float4 per 32-weight block, padded

constexpr int WO_K    = N_Q;             // 3072
constexpr int WO_NB   = WO_K / 32;       // 96

// Copy N float4 of the activation into the padded LDS plane with every global
// load issued before the first LDS store (avoids one L2 round trip per slot).
template <int NT, int N, int GRP, int PAD>
__device__ __forceinline__ void lds_fill_pad(float4 * dst, const float4 * __restrict__ src) {
    constexpr int PER = (N + NT - 1) / NT;
    constexpr int NGRP = GRP;
    float4 v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * NT;
        v[i] = src[idx < N ? idx : N - 1];
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * NT;
        if (idx < N) {
            dst[(idx / NGRP) * PAD + (idx % NGRP)] = v[i];
        }
    }
}

// ===========================================================================
// 1. NVFP4 gate_up + SITU
// ---------------------------------------------------------------------------
// Re-tiled around a 32-weight "unit" instead of a 64-weight block:
//   * 72 units per row / 8 lanes = 9 ring iterations with ZERO tail (the
//     64-weight tiling needs 4.5 blocks per lane: 5 iterations at 90% lane
//     utilisation).
//   * the activation footprint per ring step halves from 16 float4 to 8
//     (32 VGPRs), which is what keeps the whole kernel inside one wave32
//     register budget without spilling.
//   * one unit is exactly one `global_load_b128` (16 B = 32 NVFP4 weights) +
//     one uint16 scale (2 UE4M3 sub-block bytes), so the ring costs 6 VGPRs
//     per entry for BOTH rows.
// Lane g owns units {g, g+8, g+16, ..., g+64}; for a fixed iteration the 8
// lanes of a group cover 8 consecutive units = 128 contiguous bytes of the row.
//
//   8 warps x 4 groups = 32 pairs/block -> 9216 pairs / 32 = 288 blocks
//   288 blocks x 8 warps = 2304 warps = 6 warps/SIMD (96 CU x 4 SIMD)
//
// LPR = 4 / variant ALT keeps the shipped 4-lane grouping shape (3 waves/SIMD)
// as a performance reference; its per-lane accumulation order still differs
// from the shipped kernel because it walks 16-weight units.
// ===========================================================================
template <int NWARPS, int LPR, int PD, bool NTC = true>
__device__ __forceinline__ void nvfp4_gate_up_body(
    const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
    const float * __restrict__ scale2, const int32_t * __restrict__ topk_ids,
    float * __restrict__ act, int pair_base, float4 * __restrict__ s_x) {
    constexpr int PPW = 32 / LPR;               // pairs per warp
    constexpr int NU  = NV_K / 32;              // 72 units per row
    constexpr int ITS = NU / LPR;               // 9 (LPR 8) / 18 (LPR 4)
    static_assert(NU % LPR == 0, "LPR must divide 72");

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g    = lane % LPR;
    const int p    = pair_base + warp * PPW + lane / LPR;

    const int k    = p / N_FF_EXP;
    const int r    = p - k * N_FF_EXP;
    const int expr = topk_ids[k];
    const bool eok = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex   = eok ? expr : 0;
    const float s2 = eok ? scale2[ex] : 0.0f;

    const size_t row_g = (size_t) ex * N_GATE_UP_EXP + r;
    const size_t row_u = row_g + N_FF_EXP;
    const uint8_t  * qg = qs + row_g * NV_QSR;
    const uint8_t  * qu = qs + row_u * NV_QSR;
    const uint8_t  * sg = scales + row_g * NV_SCR;
    const uint8_t  * su = scales + row_u * NV_SCR;

    static_assert(PD >= 1 && PD <= 4, "prime() is unguarded: PD*LPR <= 72");

    v4i      rg[PD], ru[PD];
    uint32_t scg[PD], scu[PD];
#pragma unroll
    for (int q = 0; q < PD; ++q) {
        const int b = (g + q * LPR) * 16;                 // byte offset of the unit
        rg[q]  = ld_w_b128<NTC>(qg + b);
        ru[q]  = ld_w_b128<NTC>(qu + b);
        scg[q] = *reinterpret_cast<const uint16_t *>(sg + (b >> 3));
        scu[q] = *reinterpret_cast<const uint16_t *>(su + (b >> 3));
    }

    float accg = 0.0f;
    float accu = 0.0f;
#pragma unroll
    for (int it = 0; it < ITS; ++it) {
        // Compile-time memory barrier + machine-scheduler barrier: without them
        // the fully unrolled loop lets the backend hoist every iteration's
        // activation reads (8 float4 each) and the kernel needs 240 VGPRs.
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        const int slot = it % PD;
        const int u    = g + it * LPR;

        const v4i G = rg[slot], U = ru[slot];
        const uint32_t SG = scg[slot], SU = scu[slot];

        const int un = u + PD * LPR;
        if (it + PD < ITS) {
            const int bn = un * 16;
            rg[slot]  = ld_w_b128<NTC>(qg + bn);
            ru[slot]  = ld_w_b128<NTC>(qu + bn);
            scg[slot] = *reinterpret_cast<const uint16_t *>(sg + (bn >> 3));
            scu[slot] = *reinterpret_cast<const uint16_t *>(su + (bn >> 3));
        }

        const float4 * xb = s_x + u * Q8_LO;               // 32 floats, stride 9
        {
            const float4 x0 = xb[0], x1 = xb[1], x2 = xb[2], x3 = xb[3];
            accg = fmaf(nvfp4_sub_dot((uint32_t) G.x, (uint32_t) G.y, x0, x1, x2, x3),
                        ue4m3_to_fp32(SG & 0xFFu), accg);
            accu = fmaf(nvfp4_sub_dot((uint32_t) U.x, (uint32_t) U.y, x0, x1, x2, x3),
                        ue4m3_to_fp32(SU & 0xFFu), accu);
        }
        __builtin_amdgcn_sched_barrier(0);
        {
            const float4 x0 = xb[4], x1 = xb[5], x2 = xb[6], x3 = xb[7];
            accg = fmaf(nvfp4_sub_dot((uint32_t) G.z, (uint32_t) G.w, x0, x1, x2, x3),
                        ue4m3_to_fp32(SG >> 8), accg);
            accu = fmaf(nvfp4_sub_dot((uint32_t) U.z, (uint32_t) U.w, x0, x1, x2, x3),
                        ue4m3_to_fp32(SU >> 8), accu);
        }
    }

    accg = group_reduce_sum<LPR>(accg);
    accu = group_reduce_sum<LPR>(accu);
    if (g == 0) {
        act[k * N_FF_EXP + r] = eok ? situ_default(accg * s2, accu * s2) : 0.0f;
    }
}

template <int NWARPS, int LPR, int PD, bool NTC = true>
__global__ void __launch_bounds__(NWARPS * 32)
moe_nvfp4_gate_up_v3_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
                            const float * __restrict__ scale2, const float * __restrict__ x_norm,
                            const int32_t * __restrict__ topk_ids, float * __restrict__ act) {
    constexpr int NT   = NWARPS * 32;
    constexpr int N4   = NV_K / 4;
    constexpr int PART = NWARPS * (32 / LPR);

    __shared__ float4 s_x[Q8_NB * Q8_LO];                 // 10,368 B (32-weight stride 9)
    lds_fill_pad<NT, N4, 8, Q8_LO>(s_x, reinterpret_cast<const float4 *>(x_norm));
    __syncthreads();

    nvfp4_gate_up_body<NWARPS, LPR, PD, NTC>(qs, scales, scale2, topk_ids, act,
                                        blockIdx.x * PART, s_x);
}

// ===========================================================================
// 1b. NVFP4 gate_up, SHIPPED 4-lane/64-weight tiling + deep prefetch ring
// ---------------------------------------------------------------------------
// The 8-lane AUTO tiling above wins waves/SIMD but makes the 8 lanes of a pair
// read 8 DIFFERENT activation chunks from LDS (1 KB unique per pair per unit),
// whereas the shipped 4-lane tiling reads ONE 128 B chunk broadcast across the
// group: 8x less LDS traffic for the same weights.  Measurement says LDS
// bandwidth matters more than the extra waves (AUTO 38.5 us vs shipped 31.1
// us per call), so this body keeps the shipped lane/weight mapping - and is
// therefore BIT-IDENTICAL - and only deepens the ring (PD = 1 -> 2/3) plus
// uses the branchless ue4m3 decode.
// ===========================================================================
template <int NWARPS, int PD>
__device__ __forceinline__ void nvfp4_gate_up_bs_body(
    const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
    const float * __restrict__ scale2, const int32_t * __restrict__ topk_ids,
    float * __restrict__ act, int pair_base, const float4 * __restrict__ s_x) {
    constexpr int LPR = 4;
    constexpr int PPW = 8;                          // pairs per warp
    constexpr int NB  = NV_K / 64;                  // 36 x 64-weight blocks
    constexpr int ITS = NB / LPR;                   // 9 blocks per lane
    static_assert(NB % LPR == 0, "36 blocks = 9 per lane");
    static_assert(PD == 1 || PD == 2, "rolled rotation ring: PD in {1,2}");

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g    = lane % LPR;
    const int p    = pair_base + warp * PPW + lane / LPR;

    const int k    = p / N_FF_EXP;
    const int r    = p - k * N_FF_EXP;
    const int expr = topk_ids[k];
    const bool eok = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex   = eok ? expr : 0;
    const float s2 = eok ? scale2[ex] : 0.0f;

    const size_t row_g = (size_t) ex * N_GATE_UP_EXP + r;
    const size_t row_u = row_g + N_FF_EXP;
    const uint8_t * qg = qs + row_g * NV_QSR;
    const uint8_t * qu = qs + row_u * NV_QSR;
    const uint8_t * sg = scales + row_g * NV_SCR;
    const uint8_t * su = scales + row_u * NV_SCR;

    // IMPORTANT: the loop below must stay ROLLED.  A fully unrolled 9-step
    // body is ~4.6k instructions (~37 KB) and thrashes the 32 KB SIMD
    // instruction cache: measured 45.96 us/call vs 31.1 us for the shipped
    // 904-instruction rolled kernel with the same math.  The ring is rotated
    // by register copies instead of an indexed slot array, exactly like the
    // shipped kernel does for PD = 1.
    v4i      cg0, cg1, cu0, cu1, ng0, ng1, nu0, nu1;
    uint32_t csg, csu, nsg, nsu;
    {
        const int b0 = g * 32;
        cg0 = ld_nt_b128(qg + b0);       cg1 = ld_nt_b128(qg + b0 + 16);
        cu0 = ld_nt_b128(qu + b0);       cu1 = ld_nt_b128(qu + b0 + 16);
        csg = *reinterpret_cast<const uint32_t *>(sg + (b0 >> 3));
        csu = *reinterpret_cast<const uint32_t *>(su + (b0 >> 3));
        if (PD == 2) {
            const int b1 = (g + LPR) * 32;
            ng0 = ld_nt_b128(qg + b1);   ng1 = ld_nt_b128(qg + b1 + 16);
            nu0 = ld_nt_b128(qu + b1);   nu1 = ld_nt_b128(qu + b1 + 16);
            nsg = *reinterpret_cast<const uint32_t *>(sg + (b1 >> 3));
            nsu = *reinterpret_cast<const uint32_t *>(su + (b1 >> 3));
        }
    }

    float accg = 0.0f;
    float accu = 0.0f;
#pragma unroll 1
    for (int it = 0; it < ITS; ++it) {
        __builtin_amdgcn_sched_barrier(0);
        const int b = g + it * LPR;

        const uint32_t SG = csg, SU = csu;

        constexpr int LOOK = PD;                     // 1: load next while using current
        v4i tg0, tg1, tu0, tu1;
        uint32_t tsg = 0, tsu = 0;
        if (it + LOOK < ITS) {
            const int bn = (g + (it + LOOK) * LPR) * 32;
            tg0 = ld_nt_b128(qg + bn);        tg1 = ld_nt_b128(qg + bn + 16);
            tu0 = ld_nt_b128(qu + bn);        tu1 = ld_nt_b128(qu + bn + 16);
            tsg = *reinterpret_cast<const uint32_t *>(sg + (bn >> 3));
            tsu = *reinterpret_cast<const uint32_t *>(su + (bn >> 3));
        }

        const float4 * xa = s_x + (2 * b) * Q8_LO;      // 64 floats, two 32-weight units
        const float4 * xc = s_x + (2 * b + 1) * Q8_LO;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const float4 x0 = (s < 2) ? xa[s * 4 + 0] : xc[(s - 2) * 4 + 0];
            const float4 x1 = (s < 2) ? xa[s * 4 + 1] : xc[(s - 2) * 4 + 1];
            const float4 x2 = (s < 2) ? xa[s * 4 + 2] : xc[(s - 2) * 4 + 2];
            const float4 x3 = (s < 2) ? xa[s * 4 + 3] : xc[(s - 2) * 4 + 3];
            const uint32_t gq0 = (s < 2) ? (s == 0 ? (uint32_t) cg0.x : (uint32_t) cg0.z)
                                         : (s == 2 ? (uint32_t) cg1.x : (uint32_t) cg1.z);
            const uint32_t gq1 = (s < 2) ? (s == 0 ? (uint32_t) cg0.y : (uint32_t) cg0.w)
                                         : (s == 2 ? (uint32_t) cg1.y : (uint32_t) cg1.w);
            const uint32_t uq0 = (s < 2) ? (s == 0 ? (uint32_t) cu0.x : (uint32_t) cu0.z)
                                         : (s == 2 ? (uint32_t) cu1.x : (uint32_t) cu1.z);
            const uint32_t uq1 = (s < 2) ? (s == 0 ? (uint32_t) cu0.y : (uint32_t) cu0.w)
                                         : (s == 2 ? (uint32_t) cu1.y : (uint32_t) cu1.w);
            accg = fmaf(nvfp4_sub_dot(gq0, gq1, x0, x1, x2, x3),
                        ue4m3_to_fp32((SG >> (8 * s)) & 0xFFu), accg);
            accu = fmaf(nvfp4_sub_dot(uq0, uq1, x0, x1, x2, x3),
                        ue4m3_to_fp32((SU >> (8 * s)) & 0xFFu), accu);
        }

        // rotate current <- next, next <- prefetched (register moves only)
        if (PD == 1) {
            if (it + 1 < ITS) {
                cg0 = tg0; cg1 = tg1; cu0 = tu0; cu1 = tu1; csg = tsg; csu = tsu;
            }
        } else {
            cg0 = ng0; cg1 = ng1; cu0 = nu0; cu1 = nu1; csg = nsg; csu = nsu;
            if (it + 2 < ITS) {
                ng0 = tg0; ng1 = tg1; nu0 = tu0; nu1 = tu1; nsg = tsg; nsu = tsu;
            }
        }
    }

    accg = group_reduce_sum<LPR>(accg);
    accu = group_reduce_sum<LPR>(accu);
    if (g == 0) {
        act[k * N_FF_EXP + r] = eok ? situ_default(accg * s2, accu * s2) : 0.0f;
    }
}

template <int NWARPS, int PD>
__global__ void __launch_bounds__(NWARPS * 32, 8 / NWARPS)
moe_nvfp4_gate_up_bs_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
                            const float * __restrict__ scale2, const float * __restrict__ x_norm,
                            const int32_t * __restrict__ topk_ids, float * __restrict__ act) {
    constexpr int NT   = NWARPS * 32;
    constexpr int N4   = NV_K / 4;
    constexpr int PART = NWARPS * 8;

    __shared__ float4 s_x[Q8_NB * Q8_LO];
    lds_fill_pad<NT, N4, 8, Q8_LO>(s_x, reinterpret_cast<const float4 *>(x_norm));
    __syncthreads();

    nvfp4_gate_up_bs_body<NWARPS, PD>(qs, scales, scale2, topk_ids, act,
                                      blockIdx.x * PART, s_x);
}

// ===========================================================================
// 2. NVFP4 down + weighted 8-expert sum + sconv_post_res
// ---------------------------------------------------------------------------
// 8 warps per block, warp k = expert slot k, 16 lanes per output row
// (g = lane & 15, rloc = lane >> 4 -> 2 rows per warp per step).
//
// The activation is NOT cached in VGPRs and NOT staged in LDS: lane g re-reads
// its 72 floats (3 chunks of 8/8/2 float4) from L1 every row.  The activation
// for one token is 36.9 KB and every CU walks it in the same order, so it is
// L1/L2 resident; the 11.9 MB weight stream is `slc dlc` and never allocates
// in L1, so it cannot evict it.  Trade: ~18 extra cached loads per lane per row
// (~9% issue slots) for -45 VGPR and -12.4 KB LDS -> 8 warps/SIMD instead of 4.
//
// 384 blocks x 8 warps = 3072 warps = 8 warps/SIMD (the gfx11 wave32 maximum);
// 2304 rows / (384 x 2) = 3 row steps, exactly, so there is no tail wave.
//
// The per-lane weight set, FMA order, 16-lane butterfly and the 8-expert
// summation tree are identical to moe_nvfp4_down_wsum -> bit-exact output.
// ===========================================================================
template <int GRID, bool XREG, bool NTC = true>
__device__ __forceinline__ void nvfp4_down_body(
    const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
    const float * __restrict__ scale2, const float * __restrict__ act,
    const int32_t * __restrict__ topk_ids, const float * __restrict__ topk_w,
    float * __restrict__ out, const SconvEpi & epi, int row_base, float * s_red) {
    const int warp = threadIdx.x >> 5;   // = expert slot
    const int lane = threadIdx.x & 31;
    const int g    = lane & 15;
    const int rloc = lane >> 4;

    const int expr   = topk_ids[warp];
    const bool eok   = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex     = eok ? expr : 0;
    const float coef = eok ? topk_w[warp] * scale2[ex] : 0.0f;

    // act is indexed by expert SLOT (k), not by expert id.
    const float * __restrict__ ak  = act + (size_t) warp * DN_K;
    const float * __restrict__ xa  = ak + g * 32;             // weights   g*32 .. +32
    const float * __restrict__ xb  = ak + 512 + g * 32;       // weights 512+g*32 .. +32
    const float * __restrict__ xc  = ak + 1024 + g * 8;       // weights 1024+8g .. +8

    const uint8_t * __restrict__ qbase = qs     + (size_t) ex * N_EMBD * DN_QSR;
    const uint8_t * __restrict__ sbase = scales + (size_t) ex * N_EMBD * DN_SCR;
    const float4 * XA = reinterpret_cast<const float4 *>(xa);
    const float4 * XB = reinterpret_cast<const float4 *>(xb);
    const float4 * XC = reinterpret_cast<const float4 *>(xc);

    // Activation chunks are consumed one 16-float group at a time so that only
    // 4 float4 (16 VGPRs) are live per sub-dot.  With XREG (variant ALT) the
    // first 16 float4 are hoisted into VGPRs once: same math, 95 VGPRs, 4
    // waves/SIMD, i.e. the shipped register-cached shape for an A/B.
    float4 RX0[8], RX1[8];
    if (XREG) {
#pragma unroll
        for (int c = 0; c < 8; ++c) { RX0[c] = XA[c]; }
#pragma unroll
        for (int c = 0; c < 8; ++c) { RX1[c] = XB[c]; }
    }

    int step = 0;
    for (int r0 = row_base; r0 < N_EMBD; r0 += GRID * 2, ++step) {
        const int r = r0 + rloc;
        const uint8_t  * q  = qbase + (size_t) r * DN_QSR;
        const uint8_t  * sc = sbase + (size_t) r * DN_SCR;

        asm volatile("" ::: "memory");   // keep the activation reads inside the row loop
        const v4i w00 = ld_w_b128<NTC>(q + g * 16);          // 32 weights
        const v4i w10 = ld_w_b128<NTC>(q + 256 + g * 16);    // 32 weights
        const uint32_t w2 = ld_w_b32<NTC>(q + 512 + 4 * g);  //  8 weights
        const uint32_t s0 = *reinterpret_cast<const uint16_t *>(sc + 2 * g);
        const uint32_t s1 = *reinterpret_cast<const uint16_t *>(sc + 32 + 2 * g);
        const uint32_t s2 = sc[64 + (g >> 1)];

        __builtin_amdgcn_sched_barrier(0);
        float4 t0, t1, t2, t3;
        if (XREG) {
            t0 = RX0[0]; t1 = RX0[1]; t2 = RX0[2]; t3 = RX0[3];
        } else {
            t0 = XA[0]; t1 = XA[1]; t2 = XA[2]; t3 = XA[3];
        }
        float acc = nvfp4_sub_dot((uint32_t) w00.x, (uint32_t) w00.y, t0, t1, t2, t3) *
                    ue4m3_to_fp32(s0 & 0xFFu);
        __builtin_amdgcn_sched_barrier(0);
        if (XREG) {
            t0 = RX0[4]; t1 = RX0[5]; t2 = RX0[6]; t3 = RX0[7];
        } else {
            t0 = XA[4]; t1 = XA[5]; t2 = XA[6]; t3 = XA[7];
        }
        acc = fmaf(nvfp4_sub_dot((uint32_t) w00.z, (uint32_t) w00.w, t0, t1, t2, t3),
                   ue4m3_to_fp32(s0 >> 8), acc);
        __builtin_amdgcn_sched_barrier(0);
        if (XREG) {
            t0 = RX1[0]; t1 = RX1[1]; t2 = RX1[2]; t3 = RX1[3];
        } else {
            t0 = XB[0]; t1 = XB[1]; t2 = XB[2]; t3 = XB[3];
        }
        acc = fmaf(nvfp4_sub_dot((uint32_t) w10.x, (uint32_t) w10.y, t0, t1, t2, t3),
                   ue4m3_to_fp32(s1 & 0xFFu), acc);
        __builtin_amdgcn_sched_barrier(0);
        if (XREG) {
            t0 = RX1[4]; t1 = RX1[5]; t2 = RX1[6]; t3 = RX1[7];
        } else {
            t0 = XB[4]; t1 = XB[5]; t2 = XB[6]; t3 = XB[7];
        }
        acc = fmaf(nvfp4_sub_dot((uint32_t) w10.z, (uint32_t) w10.w, t0, t1, t2, t3),
                   ue4m3_to_fp32(s1 >> 8), acc);

        uint32_t lo, hi;
        nvfp4_hbytes(w2, lo, hi);
        __builtin_amdgcn_sched_barrier(0);
        const float4 c0 = XC[0], c1 = XC[1];
        float p3 = 0.0f;
        p3 = nvfp4_hb_dot_mix(lo, c0, p3);
        p3 = nvfp4_hb_dot_mix(hi, c1, p3);
        acc = fmaf(p3, ue4m3_to_fp32(s2), acc);

        const float v = group_reduce_sum<16>(acc) * coef;
        if (g == 0) {
            s_red[(step & 1) * 8 * 2 + warp * 2 + rloc] = v;
        }
        __syncthreads();
        if (warp == 0 && lane < 2) {
            const float * rb = s_red + (step & 1) * 8 * 2;
            const int ro = r0 + lane;
            const float sum = ((rb[0 * 2 + lane] + rb[1 * 2 + lane]) + (rb[2 * 2 + lane] + rb[3 * 2 + lane])) +
                              ((rb[4 * 2 + lane] + rb[5 * 2 + lane]) + (rb[6 * 2 + lane] + rb[7 * 2 + lane]));
            out[ro] = (epi.conv_w != nullptr) ? sconv_post_res_one(epi, ro, sum) : sum;
        }
    }
}

template <int GRID, bool XREG, bool NTC = true>
__global__ void __launch_bounds__(256)
moe_nvfp4_down_v3_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
                         const float * __restrict__ scale2, const float * __restrict__ act,
                         const int32_t * __restrict__ topk_ids, const float * __restrict__ topk_w,
                         float * __restrict__ out, const SconvEpi epi) {
    __shared__ float s_red[2 * 8 * 2];                  // 128 B
    nvfp4_down_body<GRID, XREG, NTC>(qs, scales, scale2, act, topk_ids, topk_w, out, epi,
                                blockIdx.x * 2, s_red);
}

// ===========================================================================
// 3. NVFP4 gate_up + down fused in ONE launch (deadlock-free grid barrier)
// ---------------------------------------------------------------------------
// Same 288 blocks x 256 threads run both phases.  Co-residency (measured:
// 64 VGPRs, 10,496 B LDS, 0 spills): 8 waves/SIMD -> 4 blocks/CU from the
// register file, 6 blocks/CU from LDS (64 KB / 10.4 KB); the grid only asks
// for 3 blocks/CU, so every block is guaranteed resident before any spin
// (33% headroom).  The allocation is pinned with amdgpu_num_vgpr(64) so a
// future edit cannot silently break the barrier's residency precondition.
// Removes one dispatch per MoE layer (-~2 us/layer) and keeps the 36.9 KB
// activation fully in L2 between the phases.
// ===========================================================================
// amdgpu_num_vgpr is the only way to pin the allocation (launch_bounds' second
// argument is not honoured for waves/SIMD on this toolchain); it is marked
// deprecated in favour of amdgpu_waves_per_eu, which the backend rejects with a
// link-time occupancy warning, so the deprecation is silenced locally instead.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
template <int NWARPS, int LPR, int PD, int GRID, int SHARDS, bool NTC = true>
__global__ void __launch_bounds__(NWARPS * 32) __attribute__((amdgpu_num_vgpr(64)))
moe_nvfp4_fused_v3_kernel(const uint8_t * __restrict__ gu_qs, const uint8_t * __restrict__ gu_sc,
                          const float * __restrict__ gu_s2, const int32_t * __restrict__ topk_ids,
                          const float * __restrict__ topk_w, const float * __restrict__ x_norm,
                          const uint8_t * __restrict__ dn_qs, const uint8_t * __restrict__ dn_sc,
                          const float * __restrict__ dn_s2, float * __restrict__ act,
                          float * __restrict__ out, const SconvEpi epi,
                          float * __restrict__ scratch) {
    constexpr int NT   = NWARPS * 32;
    constexpr int N4   = NV_K / 4;
    constexpr int PART = NWARPS * (32 / LPR);
    static_assert(GRID * PART == N_FF_EXP * TOP_K, "phase 1 must cover all pairs exactly");

    __shared__ float4 s_x[Q8_NB * Q8_LO];               // 10,368 B (32-weight stride 9)
    __shared__ float  s_red[2 * 8 * 2];                 //     128 B

    lds_fill_pad<NT, N4, 8, Q8_LO>(s_x, reinterpret_cast<const float4 *>(x_norm));
    __syncthreads();

    nvfp4_gate_up_body<NWARPS, LPR, PD, NTC>(gu_qs, gu_sc, gu_s2, topk_ids, act,
                                        blockIdx.x * PART, s_x);

    grid_barrier<SHARDS>(scratch, GRID);

    nvfp4_down_body<GRID, false, NTC>(dn_qs, dn_sc, dn_s2, act, topk_ids, topk_w, out, epi,
                                 blockIdx.x * 2, s_red);
}
#pragma clang diagnostic pop

// ===========================================================================
// 4. Q8_0 gate_up + SITU  (layers 0/1 and the two loop replays)
// ---------------------------------------------------------------------------
// Same re-tiling as the NVFP4 stage.  The fast path uses a 16-weight unit
// (one `global_load_b128` per row = 4 int8 dwords, 4 float4 of activation):
// 144 units per row / 8 lanes = 18 ring iterations, zero tail, and only 16
// VGPRs of activation live per step.  UW = 32 is the 4-lane/32-weight-unit
// performance reference (variant ALT).  The shipped kernel runs LPR = 32 with
// a 3rd iteration that is only 8/32 lanes active (75% lane utilisation) and
// PD = 1; that shape is not reproduced here because it is strictly worse.
// ===========================================================================
template <int NWARPS, int LPR, int PD, int UW, bool NTC = true>
__device__ __forceinline__ void q8_gate_up_body(
    const int8_t * __restrict__ qs, const __half * __restrict__ d,
    const int32_t * __restrict__ topk_ids, float * __restrict__ act,
    int pair_base, const float4 * __restrict__ s_x) {
    constexpr int PPW  = 32 / LPR;
    constexpr int NU   = Q8_K / UW;                 // units per row
    constexpr int ITS  = NU / LPR;
    constexpr int NV4  = UW / 16;                   // b128 loads per unit (Q8: 1 byte/weight)
    constexpr int NX4  = UW / 4;                    // float4 of activation per unit
    static_assert(NU % LPR == 0, "LPR must divide the unit count");
    static_assert(UW == 16 || UW == 32, "UW in {16, 32}");

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g    = lane % LPR;
    const int p    = pair_base + warp * PPW + lane / LPR;

    const int k    = p / N_FF_EXP;
    const int r    = p - k * N_FF_EXP;
    const int expr = topk_ids[k];
    const bool eok = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex   = eok ? expr : 0;

    const size_t row_g = (size_t) ex * N_GATE_UP_EXP + r;
    const size_t row_u = row_g + N_FF_EXP;
    const int8_t * qg = qs + row_g * Q8_K;
    const int8_t * qu = qs + row_u * Q8_K;
    const __half * dg = d + row_g * Q8_DS;
    const __half * du = d + row_u * Q8_DS;

    static_assert(PD >= 1 && PD <= 4, "prime() is unguarded: PD*LPR <= 72");

    v4i    rg[PD][NV4], ru[PD][NV4];
    __half scg[PD], scu[PD];
#pragma unroll
    for (int q = 0; q < PD; ++q) {
        const int b = (g + q * LPR) * UW;
#pragma unroll
        for (int j = 0; j < NV4; ++j) {
            rg[q][j] = ld_w_b128<NTC>(qg + b + j * 16);
            ru[q][j] = ld_w_b128<NTC>(qu + b + j * 16);
        }
        scg[q] = dg[(g + q * LPR) * UW / 32];
        scu[q] = du[(g + q * LPR) * UW / 32];
    }

    float accg = 0.0f;
    float accu = 0.0f;
#pragma unroll
    for (int it = 0; it < ITS; ++it) {
        asm volatile("" ::: "memory");
        __builtin_amdgcn_sched_barrier(0);
        const int slot = it % PD;
        const int u    = g + it * LPR;

        v4i G[NV4], U[NV4];
#pragma unroll
        for (int j = 0; j < NV4; ++j) {
            G[j] = rg[slot][j];
            U[j] = ru[slot][j];
        }
        const float SG = __half2float(scg[slot]);
        const float SU = __half2float(scu[slot]);

        if (it + PD < ITS) {
            const int b = (u + PD * LPR) * UW;
#pragma unroll
            for (int j = 0; j < NV4; ++j) {
                rg[slot][j] = ld_w_b128<NTC>(qg + b + j * 16);
                ru[slot][j] = ld_w_b128<NTC>(qu + b + j * 16);
            }
            scg[slot] = dg[(u + PD * LPR) * UW / 32];
            scu[slot] = du[(u + PD * LPR) * UW / 32];
        }

        // Padded LDS layout is 9 float4 per 32 weights; a unit starts at
        // (u*UW/32)*9 plus 4 float4 when it is the upper half of a block.
        const int xbase = (u * UW / 32) * Q8_LO + ((u * UW / 16) & 1) * 4;
        const float4 * xb = s_x + xbase;
        float pg = 0.0f, pu = 0.0f;
#pragma unroll
        for (int c = 0; c < NX4; ++c) {
            const float4 xv = xb[c];
            const int j  = c / 4;
            const int sh = (c & 3);
            const uint32_t gw = (sh == 0) ? (uint32_t) G[j].x : (sh == 1) ? (uint32_t) G[j].y
                              : (sh == 2) ? (uint32_t) G[j].z : (uint32_t) G[j].w;
            const uint32_t uw_ = (sh == 0) ? (uint32_t) U[j].x : (sh == 1) ? (uint32_t) U[j].y
                              : (sh == 2) ? (uint32_t) U[j].z : (uint32_t) U[j].w;
            pg = q8_word_dot_mix(gw,  xv, pg);
            pu = q8_word_dot_mix(uw_, xv, pu);
        }
        accg = fmaf(pg, SG, accg);
        accu = fmaf(pu, SU, accu);
    }

    accg = group_reduce_sum<LPR>(accg);
    accu = group_reduce_sum<LPR>(accu);
    if (g == 0) {
        act[k * N_FF_EXP + r] = eok ? situ_default(accg, accu) : 0.0f;
    }
}

template <int NWARPS, int LPR, int PD, int UW, bool NTC = true>
__global__ void __launch_bounds__(NWARPS * 32)
moe_q8_gate_up_v3_kernel(const int8_t * __restrict__ qs, const __half * __restrict__ d,
                         const float * __restrict__ x_norm, const int32_t * __restrict__ topk_ids,
                         float * __restrict__ act) {
    constexpr int NT   = NWARPS * 32;
    constexpr int N4   = Q8_K / 4;
    constexpr int PART = NWARPS * (32 / LPR);

    __shared__ float4 s_x[Q8_NB * Q8_LO];               // 10,368 B
    lds_fill_pad<NT, N4, 8, Q8_LO>(s_x, reinterpret_cast<const float4 *>(x_norm));
    __syncthreads();

    q8_gate_up_body<NWARPS, LPR, PD, UW, NTC>(qs, d, topk_ids, act, blockIdx.x * PART, s_x);
}

// ===========================================================================
// 5. Q8_0 down + weighted 8-expert sum (identical tiling to the NVFP4 stage)
// ---------------------------------------------------------------------------
// Lane g owns blocks g and 16+g (2 x 32 weights = 4 x b128) plus 8 weights at
// 1024+8g (b64, scale d[32 + (g>>2)]) - exactly the shipped per-lane set and
// order, so the output is bit-identical.
// ===========================================================================
template <int GRID, bool XREG, bool NTC = true>
__device__ __forceinline__ void q8_down_body(
    const int8_t * __restrict__ qs, const __half * __restrict__ d,
    const float * __restrict__ act, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ topk_w, float * __restrict__ out,
    const SconvEpi & epi, int row_base, float * s_red) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g    = lane & 15;
    const int rloc = lane >> 4;

    const int expr   = topk_ids[warp];
    const bool eok   = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex     = eok ? expr : 0;
    const float coef = eok ? topk_w[warp] : 0.0f;

    const float * __restrict__ ak = act + (size_t) warp * DN_K;
    const float * __restrict__ xa = ak + g * 32;
    const float * __restrict__ xb = ak + 512 + g * 32;
    const float * __restrict__ xc = ak + 1024 + g * 8;

    const int8_t * __restrict__ qbase = qs + (size_t) ex * N_EMBD * DN_K;
    const __half * __restrict__ dbase = d  + (size_t) ex * N_EMBD * (DN_K / 32);
    const float4 * XA = reinterpret_cast<const float4 *>(xa);
    const float4 * XB = reinterpret_cast<const float4 *>(xb);
    const float4 * XC = reinterpret_cast<const float4 *>(xc);

    float4 RX0[8], RX1[8];
    if (XREG) {
#pragma unroll
        for (int c = 0; c < 8; ++c) { RX0[c] = XA[c]; }
#pragma unroll
        for (int c = 0; c < 8; ++c) { RX1[c] = XB[c]; }
    }

    int step = 0;
    for (int r0 = row_base; r0 < N_EMBD; r0 += GRID * 2, ++step) {
        const int r = r0 + rloc;
        const int8_t * q  = qbase + (size_t) r * DN_K;
        const __half * dd = dbase + (size_t) r * (DN_K / 32);

        asm volatile("" ::: "memory");   // keep the activation reads inside the row loop
        const v4i w00 = ld_w_b128<NTC>(q + g * 32);
        const v4i w01 = ld_w_b128<NTC>(q + g * 32 + 16);
        const v4i w10 = ld_w_b128<NTC>(q + (16 + g) * 32);
        const v4i w11 = ld_w_b128<NTC>(q + (16 + g) * 32 + 16);
        const v2u w2  = ld_w_b64<NTC>(q + 1024 + g * 8);
        const float h0 = __half2float(dd[g]);
        const float h1 = __half2float(dd[16 + g]);
        const float h2 = __half2float(dd[32 + (g >> 2)]);

        __builtin_amdgcn_sched_barrier(0);
        float4 t0, t1, t2, t3;
        if (XREG) { t0 = RX0[0]; t1 = RX0[1]; t2 = RX0[2]; t3 = RX0[3]; }
        else      { t0 = XA[0];   t1 = XA[1];   t2 = XA[2];   t3 = XA[3]; }
        float p0 = 0.0f;
        p0 = q8_word_dot_mix((uint32_t) w00.x, t0, p0);
        p0 = q8_word_dot_mix((uint32_t) w00.y, t1, p0);
        p0 = q8_word_dot_mix((uint32_t) w00.z, t2, p0);
        p0 = q8_word_dot_mix((uint32_t) w00.w, t3, p0);
        __builtin_amdgcn_sched_barrier(0);
        if (XREG) { t0 = RX0[4]; t1 = RX0[5]; t2 = RX0[6]; t3 = RX0[7]; }
        else      { t0 = XA[4];   t1 = XA[5];   t2 = XA[6];   t3 = XA[7]; }
        p0 = q8_word_dot_mix((uint32_t) w01.x, t0, p0);
        p0 = q8_word_dot_mix((uint32_t) w01.y, t1, p0);
        p0 = q8_word_dot_mix((uint32_t) w01.z, t2, p0);
        p0 = q8_word_dot_mix((uint32_t) w01.w, t3, p0);
        float acc = p0 * h0;

        __builtin_amdgcn_sched_barrier(0);
        if (XREG) { t0 = RX1[0]; t1 = RX1[1]; t2 = RX1[2]; t3 = RX1[3]; }
        else      { t0 = XB[0];   t1 = XB[1];   t2 = XB[2];   t3 = XB[3]; }
        float p1 = 0.0f;
        p1 = q8_word_dot_mix((uint32_t) w10.x, t0, p1);
        p1 = q8_word_dot_mix((uint32_t) w10.y, t1, p1);
        p1 = q8_word_dot_mix((uint32_t) w10.z, t2, p1);
        p1 = q8_word_dot_mix((uint32_t) w10.w, t3, p1);
        __builtin_amdgcn_sched_barrier(0);
        if (XREG) { t0 = RX1[4]; t1 = RX1[5]; t2 = RX1[6]; t3 = RX1[7]; }
        else      { t0 = XB[4];   t1 = XB[5];   t2 = XB[6];   t3 = XB[7]; }
        p1 = q8_word_dot_mix((uint32_t) w11.x, t0, p1);
        p1 = q8_word_dot_mix((uint32_t) w11.y, t1, p1);
        p1 = q8_word_dot_mix((uint32_t) w11.z, t2, p1);
        p1 = q8_word_dot_mix((uint32_t) w11.w, t3, p1);
        acc = fmaf(p1, h1, acc);

        __builtin_amdgcn_sched_barrier(0);
        const float4 c0 = XC[0], c1 = XC[1];
        float p2 = 0.0f;
        p2 = q8_word_dot_mix(w2.x, c0, p2);
        p2 = q8_word_dot_mix(w2.y, c1, p2);
        acc = fmaf(p2, h2, acc);

        const float v = group_reduce_sum<16>(acc) * coef;
        if (g == 0) {
            s_red[(step & 1) * 8 * 2 + warp * 2 + rloc] = v;
        }
        __syncthreads();
        if (warp == 0 && lane < 2) {
            const float * rb = s_red + (step & 1) * 8 * 2;
            const int ro = r0 + lane;
            const float sum = ((rb[0 * 2 + lane] + rb[1 * 2 + lane]) + (rb[2 * 2 + lane] + rb[3 * 2 + lane])) +
                              ((rb[4 * 2 + lane] + rb[5 * 2 + lane]) + (rb[6 * 2 + lane] + rb[7 * 2 + lane]));
            out[ro] = (epi.conv_w != nullptr) ? sconv_post_res_one(epi, ro, sum) : sum;
        }
    }
}

template <int GRID, bool XREG, bool NTC = true>
__global__ void __launch_bounds__(256)
moe_q8_down_v3_kernel(const int8_t * __restrict__ qs, const __half * __restrict__ d,
                      const float * __restrict__ act, const int32_t * __restrict__ topk_ids,
                      const float * __restrict__ topk_w, float * __restrict__ out,
                      const SconvEpi epi) {
    __shared__ float s_red[2 * 8 * 2];
    q8_down_body<GRID, XREG, NTC>(qs, d, act, topk_ids, topk_w, out, epi, blockIdx.x * 2, s_red);
}

// ===========================================================================
// 6. Q8_0 gate_up + down fused (same barrier contract as the NVFP4 fusion)
// ===========================================================================
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
template <int NWARPS, int LPR, int PD, int GRID, int SHARDS, bool NTC = true>
__global__ void __launch_bounds__(NWARPS * 32) __attribute__((amdgpu_num_vgpr(64)))
moe_q8_fused_v3_kernel(const int8_t * __restrict__ gu_qs, const __half * __restrict__ gu_d,
                       const int32_t * __restrict__ topk_ids, const float * __restrict__ topk_w,
                       const float * __restrict__ x_norm,
                       const int8_t * __restrict__ dn_qs, const __half * __restrict__ dn_d,
                       float * __restrict__ act, float * __restrict__ out,
                       const SconvEpi epi, float * __restrict__ scratch) {
    constexpr int NT   = NWARPS * 32;
    constexpr int N4   = Q8_K / 4;
    constexpr int PART = NWARPS * (32 / LPR);
    static_assert(GRID * PART == N_FF_EXP * TOP_K, "phase 1 must cover all pairs exactly");

    __shared__ float4 s_x[Q8_NB * Q8_LO];               // 10,368 B
    __shared__ float  s_red[2 * 8 * 2];                 //    128 B

    lds_fill_pad<NT, N4, 8, Q8_LO>(s_x, reinterpret_cast<const float4 *>(x_norm));
    __syncthreads();

    q8_gate_up_body<NWARPS, LPR, PD, 16, NTC>(gu_qs, gu_d, topk_ids, act, blockIdx.x * PART, s_x);

    grid_barrier<SHARDS>(scratch, GRID);

    q8_down_body<GRID, false, NTC>(dn_qs, dn_d, act, topk_ids, topk_w, out, epi, blockIdx.x * 2, s_red);
}
#pragma clang diagnostic pop

// ===========================================================================
// 7. wo (K = 3072, N = 2304, SoA Q8_0) + fused sconv_post_res
// ---------------------------------------------------------------------------
// This is soa_q8_gemv_kernel's K = 3072 path with the ring depth fixed:
// the shipped dispatch hard-codes PD = 1 for K = N_Q, which leaves each warp
// with a strict load -> 72 VALU -> wait -> load loop and no cross-row overlap.
// Here PD = 3 = ITS: the whole 96-block row (3 x 32 weights per lane, 6 x b128)
// is loaded as one burst and the next row's first blocks are prefetched while
// the current row's chains retire.
//
//   W1 (AUTO): 8 warps/block, 288 blocks (2304 / 8), activation staged in
//              13,824 B of padded LDS.  69 VGPRs / 0 spills -> 3 blocks/CU
//              = 24 warps/CU = 6 waves/SIMD (the shipped PD=1 kernel is 4.5).
//   W2 (ALT2): 4 warps/block, 576 blocks, activation straight from L1 (no LDS,
//              no barrier).  78 VGPRs / 0 spills -> 6 blocks/CU = 6 waves/SIMD.
// ===========================================================================
template <int PD, int NB, int LPR, bool NTC = true>
struct WoRing {
    static constexpr int ITS = NB / LPR;                 // 3
    static_assert(ITS == PD, "PD = ITS: full row in flight + cross-row prefetch");

    v4i    r0[PD], r1[PD];
    __half rh[PD];

    __device__ __forceinline__ void load(int slot, const int8_t * wq, const __half * wd, int kb) {
        r0[slot] = ld_w_b128<NTC>(wq + kb * 32);
        r1[slot] = ld_w_b128<NTC>(wq + kb * 32 + 16);
        rh[slot] = wd[kb];
    }

    __device__ __forceinline__ void prime(const int8_t * __restrict__ qs,
                                         const __half * __restrict__ dd, int rowc, int sub) {
        const int8_t * wq = qs + (size_t) rowc * (size_t) (NB * 32);
        const __half * wd = dd + (size_t) rowc * (size_t) NB;
#pragma unroll
        for (int p = 0; p < PD; ++p) {
            load(p, wq, wd, sub + p * LPR);
        }
    }

    template <bool FROM_LDS>
    __device__ __forceinline__ float run(const int8_t * __restrict__ qs, const __half * __restrict__ dd,
                                         int rowc, int nrowc, bool has_next,
                                         const float4 * s_x, const float * __restrict__ x,
                                         int sub) {
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
            const v4i w0 = r0[slot], w1 = r1[slot];
            const float sc = __half2float(rh[slot]);
            if (it + PD < ITS) {
                load(slot, wq, wd, sub + (it + PD) * LPR + xo);
            } else if (has_next) {
                load(slot, nwq, nwd, sub + (it + PD - ITS) * LPR + xo);
            }
            const int kb = sub + it * LPR + xo;
            __builtin_amdgcn_sched_barrier(0);
            float4 xv[8];
            if (FROM_LDS) {
                const float4 * xb = s_x + kb * Q8_LO;
#pragma unroll
                for (int c = 0; c < 8; ++c) { xv[c] = xb[c]; }
            } else {
                const float4 * xb = reinterpret_cast<const float4 *>(x + (size_t) kb * 32);
#pragma unroll
                for (int c = 0; c < 8; ++c) { xv[c] = xb[c]; }
            }
            float p = 0.0f, q = 0.0f;
            p = q8_word_dot_mix((uint32_t) w0.x, xv[0], p);
            q = q8_word_dot_mix((uint32_t) w1.x, xv[4], q);
            p = q8_word_dot_mix((uint32_t) w0.y, xv[1], p);
            q = q8_word_dot_mix((uint32_t) w1.y, xv[5], q);
            p = q8_word_dot_mix((uint32_t) w0.z, xv[2], p);
            q = q8_word_dot_mix((uint32_t) w1.z, xv[6], q);
            p = q8_word_dot_mix((uint32_t) w0.w, xv[3], p);
            q = q8_word_dot_mix((uint32_t) w1.w, xv[7], q);
            acc = fmaf(p + q, sc, acc);
        }
        return acc;
    }
};

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
template <int NWARPS, int PD, bool FROM_LDS, int MINB, bool NTC = true>
__global__ void __launch_bounds__(NWARPS * 32, MINB) NIB_RDNA3_VGPR_CAP(80)
soa_q8_gemv_wo_v3_kernel(const int8_t * __restrict__ qs, const __half * __restrict__ d,
                         const float * __restrict__ x, float * __restrict__ y,
                         int rows, const SconvEpi epi) {
    constexpr int NT  = NWARPS * 32;
    constexpr int LPR = 32;
    constexpr int RPB = NWARPS;                          // 1 row per warp
    constexpr int NX4 = WO_K / 4;

    __shared__ float4 s_x[FROM_LDS ? WO_NB * Q8_LO : 1];

    if (FROM_LDS) {
        lds_fill_pad<NT, NX4, 8, Q8_LO>(s_x, reinterpret_cast<const float4 *>(x));
        __syncthreads();
    }

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sub  = lane;                                // LPR = 32
    const int nblk = gridDim.x;
    const int step = nblk * RPB;

    int row0 = blockIdx.x * RPB;
    WoRing<PD, WO_NB, LPR, NTC> ring;
    {
        const int row = row0 + warp;
        ring.prime(qs, d, row < rows ? row : rows - 1, sub);
    }

    for (; row0 < rows; row0 += step) {
        const int  row      = row0 + warp;
        const bool valid    = row < rows;
        const bool has_next = row0 + step < rows;
        const int  nrow     = row + step;
        float acc = ring.template run<FROM_LDS>(qs, d, valid ? row : rows - 1,
                                       nrow < rows ? nrow : rows - 1,
                                       has_next, s_x, x, sub);
        acc = group_reduce_sum<LPR>(acc);
        if (lane == 0 && valid) {
            y[row] = sconv_post_res_one(epi, row, acc);
        }
    }
}
#pragma clang diagnostic pop

// ===========================================================================
// Host-side helpers
// ===========================================================================
[[noreturn]] void moe_v3_fail(const char * what) {
    std::fprintf(stderr, "[inkling kernels_moe_v3] contract violation: %s\n", what);
    std::abort();
}

SconvEpi make_epi(const float * conv_w, float * hist, const float * res) {
    SconvEpi e;
    e.conv_w = conv_w;
    e.hist   = hist;
    e.res    = res;
    return e;
}

// Launch table (all shapes divide their work exactly; no tail waves):
//   gate_up : 9216 pairs / (8 warps x 4 groups) = 288 blocks
//   down    : 2304 rows / (384 blocks x 2)      = 3 steps
//   fused   : phase 1 needs 288 blocks, phase 2 runs 4 steps
constexpr int V3_GU_BLOCKS    = (N_FF_EXP * TOP_K) / (8 * (32 / 8)); // 288
constexpr int V3_GU_WARPS     = 8;
constexpr int V3_DN_BLOCKS    = 384;
constexpr int V3_FUSED_BLOCKS = V3_GU_BLOCKS;                        // 288
constexpr int V3_BAR_SHARDS   = 4;

void check_nvfp4_gu(const SoaNVFP4Experts & W) {
    if (W.rows != N_GATE_UP_EXP || W.K != N_EMBD) {
        moe_v3_fail("NVFP4 gate_up must be [*, 2304, 2304]");
    }
}

void check_nvfp4_dn(const SoaNVFP4Experts & W) {
    if (W.rows != N_EMBD || W.K != N_FF_EXP) {
        moe_v3_fail("NVFP4 down must be [*, 2304, 1152]");
    }
}

void check_q8_gu(const SoaQ8Experts & W) {
    if (W.rows != N_GATE_UP_EXP || W.K != N_EMBD) {
        moe_v3_fail("Q8 gate_up must be [*, 2304, 2304]");
    }
}

void check_q8_dn(const SoaQ8Experts & W) {
    if (W.rows != N_EMBD || W.K != N_FF_EXP) {
        moe_v3_fail("Q8 down must be [*, 2304, 1152]");
    }
}

// ---------------------------------------------------------------------------
// Deadlock guard for the fused launches.
// The in-kernel grid barrier is only safe if all GRID blocks are resident at
// once.  The static analysis says 4 blocks/CU at 64 VGPRs / 10.5 KB LDS, but
// the register allocation is a compiler decision, so the occupancy is queried
// once at runtime (pure arithmetic, no sync/allocation -> HIP-graph safe) and
// the fused path is abandoned in favour of the two-launch path if the answer
// ever drops below what the grid needs.
// ---------------------------------------------------------------------------
int device_cu_count();

template <class Fn>
bool fused_fits(Fn kernel, int block_size, int grid) {
    static int capacity = -1;              // per instantiation; queried once
    if (capacity < 0) {
        int per_cu = 0;
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&per_cu, kernel, block_size, 0) != hipSuccess) {
            per_cu = 0;
        }
        capacity = per_cu * device_cu_count();
    }
    return capacity >= grid;
}

int device_cu_count() {
    static int cache[64] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) {
        return 96;
    }
    if (cache[dev] == 0) {
        int n = 0;
        if (hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || n <= 0) {
            n = 96;
        }
        cache[dev] = n;
    }
    return cache[dev];
}

} // namespace

// ============================================================================
// Public API
// ============================================================================

// Query + cache the resident-block capacity of both fused kernels before graph
// capture (see the header).  Instantiates the exact same template arguments as
// the launch paths below so the cached values are the ones they will read.
void moe_v3_prime_occupancy() {
    (void)fused_fits(moe_nvfp4_fused_v3_kernel<V3_GU_WARPS, 8, 2, V3_FUSED_BLOCKS, V3_BAR_SHARDS>,
                     V3_GU_WARPS * 32, V3_FUSED_BLOCKS);
    (void)fused_fits(moe_q8_fused_v3_kernel<V3_GU_WARPS, 8, 2, V3_FUSED_BLOCKS, V3_BAR_SHARDS>,
                     V3_GU_WARPS * 32, V3_FUSED_BLOCKS);
}
void launch_moe_nvfp4_gate_up_v3(const SoaNVFP4Experts & gate_up, const float * x_norm,
                                 const int32_t * topk_ids, float * act,
                                 hipStream_t stream, int variant) {
    check_nvfp4_gu(gate_up);
    if (variant == MOE_V3_ALT) {
        // Bit-identical to the shipped kernel (1 pair per 4 lanes, PD = 3),
        // 288 blocks x 4 warps = 3 waves/SIMD.  A/B reference point only.
        moe_nvfp4_gate_up_v3_kernel<4, 4, 3><<<V3_GU_BLOCKS, 128, 0, stream>>>(
            gate_up.qs, gate_up.scales, gate_up.scale2, x_norm, topk_ids, act);
        return;
    }
    if (variant == MOE_V3_SHIP) {
        moe_nvfp4_gate_up_bs_kernel<4, 1><<<V3_GU_BLOCKS, 128, 0, stream>>>(
            gate_up.qs, gate_up.scales, gate_up.scale2, x_norm, topk_ids, act);
        return;
    }
    if (variant == MOE_V3_SHIP3) {
        moe_nvfp4_gate_up_bs_kernel<4, 2><<<V3_GU_BLOCKS, 128, 0, stream>>>(
            gate_up.qs, gate_up.scales, gate_up.scale2, x_norm, topk_ids, act);
        return;
    }
    if (variant == MOE_V3_CACHE) {
        moe_nvfp4_gate_up_v3_kernel<V3_GU_WARPS, 8, 2, false><<<V3_GU_BLOCKS, V3_GU_WARPS * 32, 0, stream>>>(
            gate_up.qs, gate_up.scales, gate_up.scale2, x_norm, topk_ids, act);
        return;
    }
    moe_nvfp4_gate_up_v3_kernel<V3_GU_WARPS, 8, 2><<<V3_GU_BLOCKS, V3_GU_WARPS * 32, 0, stream>>>(
        gate_up.qs, gate_up.scales, gate_up.scale2, x_norm, topk_ids, act);
}

void launch_moe_nvfp4_down_v3(const SoaNVFP4Experts & down, const float * act,
                              const int32_t * topk_ids, const float * topk_w,
                              float * out, hipStream_t stream,
                              const float * post_conv_w, float * post_sconv_h,
                              const float * post_res_add, int variant) {
    check_nvfp4_dn(down);
    const SconvEpi epi = make_epi(post_conv_w, post_sconv_h, post_res_add);
    if (variant == MOE_V3_ALT) {
        moe_nvfp4_down_v3_kernel<V3_DN_BLOCKS, true><<<V3_DN_BLOCKS, 256, 0, stream>>>(
            down.qs, down.scales, down.scale2, act, topk_ids, topk_w, out, epi);
        return;
    }
    moe_nvfp4_down_v3_kernel<V3_DN_BLOCKS, false><<<V3_DN_BLOCKS, 256, 0, stream>>>(
        down.qs, down.scales, down.scale2, act, topk_ids, topk_w, out, epi);
}

void launch_moe_nvfp4_decode_v3(const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
                                const float * x_norm, const int32_t * topk_ids, const float * topk_w,
                                float * tmp_act, float * out, hipStream_t stream,
                                const float * post_conv_w, float * post_sconv_h,
                                const float * post_res_add, int variant) {
    launch_moe_nvfp4_gate_up_v3(gate_up, x_norm, topk_ids, tmp_act, stream, variant);
    launch_moe_nvfp4_down_v3(down, tmp_act, topk_ids, topk_w, out, stream,
                             post_conv_w, post_sconv_h, post_res_add, variant);
}

void launch_moe_nvfp4_fused_v3(const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
                               const float * x_norm, const int32_t * topk_ids, const float * topk_w,
                               float * tmp_act, float * out, hipStream_t stream,
                               const float * post_conv_w, float * post_sconv_h,
                               const float * post_res_add, float * scratch, int variant) {
    if (scratch == nullptr || variant == MOE_V3_ALT) {
        launch_moe_nvfp4_decode_v3(gate_up, down, x_norm, topk_ids, topk_w, tmp_act, out, stream,
                                   post_conv_w, post_sconv_h, post_res_add, variant);
        return;
    }
    check_nvfp4_gu(gate_up);
    check_nvfp4_dn(down);
    if (!fused_fits(moe_nvfp4_fused_v3_kernel<V3_GU_WARPS, 8, 2, V3_FUSED_BLOCKS, V3_BAR_SHARDS>,
                    V3_GU_WARPS * 32, V3_FUSED_BLOCKS)) {
        launch_moe_nvfp4_decode_v3(gate_up, down, x_norm, topk_ids, topk_w, tmp_act, out, stream,
                                   post_conv_w, post_sconv_h, post_res_add, variant);
        return;
    }
    if (variant == MOE_V3_CACHE) {
        moe_nvfp4_fused_v3_kernel<V3_GU_WARPS, 8, 2, V3_FUSED_BLOCKS, V3_BAR_SHARDS, false>
            <<<V3_FUSED_BLOCKS, V3_GU_WARPS * 32, 0, stream>>>(
                gate_up.qs, gate_up.scales, gate_up.scale2, topk_ids, topk_w, x_norm,
                down.qs, down.scales, down.scale2, tmp_act, out,
                make_epi(post_conv_w, post_sconv_h, post_res_add), scratch);
        return;
    }
    moe_nvfp4_fused_v3_kernel<V3_GU_WARPS, 8, 2, V3_FUSED_BLOCKS, V3_BAR_SHARDS>
        <<<V3_FUSED_BLOCKS, V3_GU_WARPS * 32, 0, stream>>>(
            gate_up.qs, gate_up.scales, gate_up.scale2, topk_ids, topk_w, x_norm,
            down.qs, down.scales, down.scale2, tmp_act, out,
            make_epi(post_conv_w, post_sconv_h, post_res_add), scratch);
}

void launch_moe_q8_gate_up_v3(const SoaQ8Experts & gate_up, const float * x_norm,
                              const int32_t * topk_ids, float * act,
                              hipStream_t stream, int variant) {
    check_q8_gu(gate_up);
    if (variant == MOE_V3_ALT) {
        moe_q8_gate_up_v3_kernel<4, 4, 3, 32><<<V3_GU_BLOCKS, 128, 0, stream>>>(
            gate_up.qs, gate_up.d, x_norm, topk_ids, act);
        return;
    }
    if (variant == MOE_V3_CACHE) {
        moe_q8_gate_up_v3_kernel<V3_GU_WARPS, 8, 2, 16, false><<<V3_GU_BLOCKS, V3_GU_WARPS * 32, 0, stream>>>(
            gate_up.qs, gate_up.d, x_norm, topk_ids, act);
        return;
    }
    moe_q8_gate_up_v3_kernel<V3_GU_WARPS, 8, 2, 16><<<V3_GU_BLOCKS, V3_GU_WARPS * 32, 0, stream>>>(
        gate_up.qs, gate_up.d, x_norm, topk_ids, act);
}

void launch_moe_q8_down_v3(const SoaQ8Experts & down, const float * act,
                           const int32_t * topk_ids, const float * topk_w,
                           float * out, hipStream_t stream,
                           const float * post_conv_w, float * post_sconv_h,
                           const float * post_res_add, int variant) {
    check_q8_dn(down);
    const SconvEpi epi = make_epi(post_conv_w, post_sconv_h, post_res_add);
    if (variant == MOE_V3_ALT) {
        moe_q8_down_v3_kernel<V3_DN_BLOCKS, true><<<V3_DN_BLOCKS, 256, 0, stream>>>(
            down.qs, down.d, act, topk_ids, topk_w, out, epi);
        return;
    }
    if (variant == MOE_V3_CACHE) {
        moe_q8_down_v3_kernel<V3_DN_BLOCKS, false, false><<<V3_DN_BLOCKS, 256, 0, stream>>>(
            down.qs, down.d, act, topk_ids, topk_w, out, epi);
        return;
    }
    moe_q8_down_v3_kernel<V3_DN_BLOCKS, false><<<V3_DN_BLOCKS, 256, 0, stream>>>(
        down.qs, down.d, act, topk_ids, topk_w, out, epi);
}

void launch_moe_q8_decode_v3(const SoaQ8Experts & gate_up, const SoaQ8Experts & down,
                             const float * x_norm, const int32_t * topk_ids, const float * topk_w,
                             float * tmp_act, float * out, hipStream_t stream,
                             const float * post_conv_w, float * post_sconv_h,
                             const float * post_res_add, int variant) {
    launch_moe_q8_gate_up_v3(gate_up, x_norm, topk_ids, tmp_act, stream, variant);
    launch_moe_q8_down_v3(down, tmp_act, topk_ids, topk_w, out, stream,
                          post_conv_w, post_sconv_h, post_res_add, variant);
}

void launch_moe_q8_fused_v3(const SoaQ8Experts & gate_up, const SoaQ8Experts & down,
                            const float * x_norm, const int32_t * topk_ids, const float * topk_w,
                            float * tmp_act, float * out, hipStream_t stream,
                            const float * post_conv_w, float * post_sconv_h,
                            const float * post_res_add, float * scratch, int variant) {
    if (scratch == nullptr || variant == MOE_V3_ALT) {
        launch_moe_q8_decode_v3(gate_up, down, x_norm, topk_ids, topk_w, tmp_act, out, stream,
                                post_conv_w, post_sconv_h, post_res_add, variant);
        return;
    }
    check_q8_gu(gate_up);
    check_q8_dn(down);
    if (!fused_fits(moe_q8_fused_v3_kernel<V3_GU_WARPS, 8, 2, V3_FUSED_BLOCKS, V3_BAR_SHARDS>,
                    V3_GU_WARPS * 32, V3_FUSED_BLOCKS)) {
        launch_moe_q8_decode_v3(gate_up, down, x_norm, topk_ids, topk_w, tmp_act, out, stream,
                                post_conv_w, post_sconv_h, post_res_add, variant);
        return;
    }
    if (variant == MOE_V3_CACHE) {
        moe_q8_fused_v3_kernel<V3_GU_WARPS, 8, 2, V3_FUSED_BLOCKS, V3_BAR_SHARDS, false>
            <<<V3_FUSED_BLOCKS, V3_GU_WARPS * 32, 0, stream>>>(
                gate_up.qs, gate_up.d, topk_ids, topk_w, x_norm,
                down.qs, down.d, tmp_act, out,
                make_epi(post_conv_w, post_sconv_h, post_res_add), scratch);
        return;
    }
    moe_q8_fused_v3_kernel<V3_GU_WARPS, 8, 2, V3_FUSED_BLOCKS, V3_BAR_SHARDS>
        <<<V3_FUSED_BLOCKS, V3_GU_WARPS * 32, 0, stream>>>(
            gate_up.qs, gate_up.d, topk_ids, topk_w, x_norm,
            down.qs, down.d, tmp_act, out,
            make_epi(post_conv_w, post_sconv_h, post_res_add), scratch);
}

void launch_soa_q8_gemv_wo_v3(const SoaQ8Matrix & W, const float * x,
                              const float * conv_w, float * sconv_hist,
                              const float * res, float * out,
                              hipStream_t stream, int variant) {
    if (W.rows <= 0) {
        return;
    }
    if (W.K != WO_K || conv_w == nullptr || sconv_hist == nullptr) {
        // Not the decode wo shape: the caller must use the generic GEMV.
        moe_v3_fail("wo_v3 needs K == 3072 and a sconv epilogue");
    }
    const SconvEpi epi = make_epi(conv_w, sconv_hist, res);
    if (variant == MOE_V3_ALT2) {
        constexpr int NW = 4;
        const int grid = (W.rows + NW - 1) / NW;
        soa_q8_gemv_wo_v3_kernel<NW, 3, false, 6><<<grid, NW * 32, 0, stream>>>(
            W.qs, W.d, x, out, W.rows, epi);
        return;
    }
    constexpr int NW = 8;
    const int grid = (W.rows + NW - 1) / NW;
    if (variant == MOE_V3_CACHE) {
        soa_q8_gemv_wo_v3_kernel<NW, 3, true, 3, false><<<grid, NW * 32, 0, stream>>>(
            W.qs, W.d, x, out, W.rows, epi);
        return;
    }
    soa_q8_gemv_wo_v3_kernel<NW, 3, true, 3><<<grid, NW * 32, 0, stream>>>(
        W.qs, W.d, x, out, W.rows, epi);
}

} // namespace inkling
