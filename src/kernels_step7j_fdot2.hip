// ============================================================================
// kernels_step7j_fdot2.hip — Step 7J for RDNA3 gfx1100 (RX 7900 XTX, 96 CU).
//
// Every decode weight stream is DRAM-bound (585-849 GB/s measured against a
// 875 GB/s roof), yet each was paying one MAC per instruction and caching its
// activation vector as FP32 in LDS.  This file rebuilds them with
//   * `__builtin_amdgcn_fdot2` (RDNA3 `v_dot2_f32_f16`, 2 MACs / inst, FP32 acc)
//   * `__half2` activation staging in LDS (half the LDS, half the traffic)
//   * a `union` for the fused MoE's two phases (LDS 20.6 KB -> 6.3 KB)
// while keeping the exact weight bytes, the exact per-lane 32/64-weight block
// order and the exact warp/group reduction trees of the shipped kernels.
//
// Precision: only activations round to FP16 (~5e-4 relative, 240x more precise
// than the 4-bit weights and 8x more precise than the 8-bit weights); every dot
// accumulates in FP32 and every epilogue (sconv/residual/SITU/RMSNorm/softmax)
// stays FP32.  This matches the activation precision the FP16 WMMA prefill path
// has always used.
//
// Numeric parity of the weight <-> activation pairing was checked against the
// shipped kernels block by block.  Note the NVFP4 *tail* (weights 1024..1151 of
// `down`) is a split-nibble layout: a 4-byte word holds elements j..j+3 in its
// low nibbles and j+8..j+11 in its high nibbles, so the shipped
// `ak[256 + 4*(g>>1) + (g&1)]` / `+2` staging is CORRECT (not a bug) and is
// reproduced here exactly.
// ============================================================================
#include "kernels_step7j_fdot2.h"

#include <cstdio>
#include <cstdlib>
#include <cstdlib>

namespace inkling {
namespace {

typedef int          v4i __attribute__((ext_vector_type(4)));
typedef unsigned int v2u __attribute__((ext_vector_type(2)));

// ---------------------------------------------------------------------------
// Device helpers (same values as the shipped kernels)
// ---------------------------------------------------------------------------
__device__ __forceinline__ v4i ld_nt_b128(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const v4i *>(p));
}

__device__ __forceinline__ v2u ld_nt_b64(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const v2u *>(p));
}

__device__ __forceinline__ uint32_t ld_nt_b32(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const uint32_t *>(p));
}

// Branchless UE4M3 -> FP32 (same values as the shipped ternary form).
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

__device__ __forceinline__ void nvfp4_hbytes(uint32_t q, uint32_t & lo, uint32_t & hi) {
    constexpr uint32_t T0 = 0x42403C00u;
    constexpr uint32_t T1 = 0x4A484644u;
    lo = __builtin_amdgcn_perm(T1, T0, q & 0x07070707u)        | ((q & 0x08080808u) << 4);
    hi = __builtin_amdgcn_perm(T1, T0, (q >> 4) & 0x07070707u) | (q & 0x80808080u);
}

// ---- fdot2 primitives -----------------------------------------------------
__device__ __forceinline__ float fdot2(uint32_t a, uint32_t x, float acc) {
    return __builtin_amdgcn_fdot2(__builtin_bit_cast(__half2, a),
                                  __builtin_bit_cast(__half2, x), acc, false);
}

__device__ __forceinline__ uint32_t h2u(float a, float b) {
    return __builtin_bit_cast(uint32_t, __floats2half2_rn(a, b));
}

// hb = 4 FP16 high bytes (elements j..j+3); (xa, xb) = those 4 weights as 2 half2.
__device__ __forceinline__ float hb_dot_fdot2(uint32_t hb, uint32_t xa, uint32_t xb, float acc) {
    acc = fdot2(__builtin_amdgcn_perm(hb, 0u, 0x050C040Cu), xa, acc);
    acc = fdot2(__builtin_amdgcn_perm(hb, 0u, 0x070C060Cu), xb, acc);
    return acc;
}

// One 16-weight NVFP4 sub-block (2 packed words, x = 8 half2 = 2 uint4).
__device__ __forceinline__ float nvfp4_sub_dot_h(uint32_t q0, uint32_t q1,
                                                 uint4 xa, uint4 xb, float acc) {
    uint32_t lo0, hi0, lo1, hi1;
    nvfp4_hbytes(q0, lo0, hi0);
    nvfp4_hbytes(q1, lo1, hi1);
    acc = hb_dot_fdot2(lo0, xa.x, xa.y, acc);
    acc = hb_dot_fdot2(lo1, xa.z, xa.w, acc);
    acc = hb_dot_fdot2(hi0, xb.x, xb.y, acc);
    acc = hb_dot_fdot2(hi1, xb.z, xb.w, acc);
    return acc;
}

// The 8-weight NVFP4 tail word (low nibbles -> elements 0..3, high -> 8..11).
__device__ __forceinline__ float nvfp4_tail_dot_h(uint32_t q, uint4 x, float acc) {
    uint32_t lo, hi;
    nvfp4_hbytes(q, lo, hi);
    acc = hb_dot_fdot2(lo, x.x, x.y, acc);
    acc = hb_dot_fdot2(hi, x.z, x.w, acc);
    return acc;
}

// Exact unbiased int8 -> half2 dot (identical values to q8_word_dot_mix).
__device__ __forceinline__ float q8_word_dot_fdot2(uint32_t w, uint32_t xa, uint32_t xb, float acc) {
    const uint32_t u   = w ^ 0x80808080u;
    const __half2  b   = __builtin_bit_cast(__half2, 0x64806480u);
    const __half2  h01 = __hsub2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u)), b);
    const __half2  h23 = __hsub2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u)), b);
    acc = __builtin_amdgcn_fdot2(h01, __builtin_bit_cast(__half2, xa), acc, false);
    acc = __builtin_amdgcn_fdot2(h23, __builtin_bit_cast(__half2, xb), acc, false);
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

__device__ __forceinline__ float warp_reduce_sum32(float v) {
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_xor(v, off, 32);
    }
    return v;
}

// ---- half2 LDS layouts ----------------------------------------------------
// Q8_0: one 32-weight block = 16 half2 (4 uint4); padded stride 20 half2 (80 B)
//       so the 8 lanes of an LPR=8 group (and lanes 0..7 of an LPR=32 group)
//       hit 8 distinct 4-word bank groups -> conflict free.
constexpr int Q8H_STRIDE = 20;
// NVFP4: one 64-weight block = 32 half2 (8 uint4); padded stride 36 half2
//       (144 B) so the 4 sub-lanes of a 4-lane broadcast group (stride 36 mod
//       32 = 4 words) are conflict free.
constexpr int NVH_STRIDE = 36;

__device__ __forceinline__ int lay_q8h(int i) { return (i >> 3) * Q8H_STRIDE + (i & 7) * 2; }
__device__ __forceinline__ int lay_nvh(int i) { return (i >> 4) * NVH_STRIDE + (i & 15) * 2; }

// Copy K/4 float4 from global into the padded half2 layout (all global loads
// issued before the first LDS store, same trick as the shipped lds_fill).
template <int NT, int N4, bool NV>
__device__ __forceinline__ void lds_fill_h2(__half2 * dst, const float4 * __restrict__ src) {
    constexpr int PER = (N4 + NT - 1) / NT;
    float4 v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * NT;
        v[i] = src[idx < N4 ? idx : N4 - 1];
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * NT;
        if (idx < N4) {
            const int base = NV ? lay_nvh(idx) : lay_q8h(idx);
            dst[base]     = __floats2half2_rn(v[i].x, v[i].y);
            dst[base + 1] = __floats2half2_rn(v[i].z, v[i].w);
        }
    }
}

// ---- gfx1100 memory ordering + self-resetting cooperative barrier ----------
__device__ __forceinline__ void rdna3_flush_stores() {
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    // gfx12: no equivalent instruction - on gfx1201 __scoped_atomic_thread_fence(RELEASE, AGENT)
    // above already lowers to s_waitcnt + `global_wb scope:SCOPE_DEV` (ISA-checked with
    // llvm-mc/hipcc -S), so gfx1200/gfx1201 need nothing extra here.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#endif
}

__device__ __forceinline__ void rdna3_inv_l0_l1() {
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
    // gfx12: no equivalent instruction - scope-qualified acquire atomics on gfx1201 emit
    // `global_inv scope:SCOPE_DEV` themselves (ISA-checked), which replaces the RDNA3
    // L0/L1 invalidation below and needs no hand-written cache asm.
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("buffer_gl1_inv\n\tbuffer_gl0_inv" ::: "memory");
#endif
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

constexpr int J_S_BARRIER = 400; // 400..403 shards, 404 ack (same as 7F/7H)

template <int SHARDS>
__device__ __forceinline__ void grid_barrier(float * scratch, int nblocks) {
    rdna3_flush_stores();
    __syncthreads();
    if (threadIdx.x == 0) {
        rdna3_flush_stores();
        (void) scratch_add(scratch, J_S_BARRIER + (blockIdx.x % SHARDS));
        const int target = (nblocks + SHARDS - 1) / SHARDS;
#pragma unroll
        for (int i = 0; i < SHARDS; ++i) {
            spin_until(scratch, J_S_BARRIER + i, target);
        }
        const int prev = scratch_add(scratch, J_S_BARRIER + SHARDS);
        if (prev == nblocks - 1) {
#pragma unroll
            for (int i = 0; i < SHARDS; ++i) {
                scratch_store(scratch, J_S_BARRIER + i, 0);
            }
            scratch_store(scratch, J_S_BARRIER + SHARDS, 0);
        }
    }
    __syncthreads();
    rdna3_inv_l0_l1();
}

// ---- sconv_post_res epilogue (bit-identical to the shipped one) -----------
struct SconvEpi {
    const float * conv_w;
    float *       hist;
    const float * res;
    float *       raw_out; // Step 8A: optional dump of the raw pre-conv x3
};

__device__ __forceinline__ float sconv_post_res_one(const SconvEpi & e, int r, float x3) {
    const float4 w  = reinterpret_cast<const float4 *>(e.conv_w)[r];
    float *      hs = e.hist + (size_t) r * N_CONV_HIST;
    const float x0 = hs[0], x1 = hs[1], x2 = hs[2];
    if (e.raw_out != nullptr) {
        e.raw_out[r] = x3;
    }
    const float acc = x3 + x0 * w.x + x1 * w.y + x2 * w.z + x3 * w.w;
    const float val = (e.res != nullptr) ? fmaf(acc, 0.4472135954999579f, e.res[r])
                                         : acc * 0.4472135954999579f;
    hs[0] = x1;
    hs[1] = x2;
    hs[2] = x3;
    return val;
}

// ===========================================================================
// Part 1. NVFP4 MoE: gate_up + SITU (4-lane broadcast, exact shipped order)
// ===========================================================================
constexpr int NV_K     = N_EMBD;        // 2304
constexpr int NV_QSR   = NV_K / 2;      // 1152 B/row
constexpr int NV_SCR   = NV_K / 16;     // 144 B/row
constexpr int NV_NB    = NV_K / 64;     // 36 x 64-weight blocks
constexpr int NV_GULPR = 4;             // 4-lane pair mapping
constexpr int NV_GUPPW = 32 / NV_GULPR; // 8 pairs/warp

template <int NWARPS_ACTIVE>
__device__ __forceinline__ void nvfp4_gu_fdot2_phase(
    const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
    const float * __restrict__ scale2, const int32_t * __restrict__ topk_ids,
    float * __restrict__ act, int pair_base, const __half2 * __restrict__ s_x) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g    = lane % NV_GULPR;
    if (warp >= NWARPS_ACTIVE) {
        return;
    }
    const int p    = pair_base + warp * NV_GUPPW + lane / NV_GULPR;
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
    const uint32_t * sg = reinterpret_cast<const uint32_t *>(scales + row_g * NV_SCR);
    const uint32_t * su = reinterpret_cast<const uint32_t *>(scales + row_u * NV_SCR);

    constexpr int ITS = NV_NB / NV_GULPR;   // 9
    int b = g;
    v4i g0 = ld_nt_b128(qg + b * 32), g1 = ld_nt_b128(qg + b * 32 + 16);
    v4i u0 = ld_nt_b128(qu + b * 32), u1 = ld_nt_b128(qu + b * 32 + 16);
    uint32_t sgw = sg[b], suw = su[b];

    float accg = 0.0f;
    float accu = 0.0f;
#pragma unroll 1
    for (int it = 0; it < ITS; ++it) {
        const int bn = b + NV_GULPR;
        v4i ng0 = g0, ng1 = g1, nu0 = u0, nu1 = u1;
        uint32_t nsg = sgw, nsu = suw;
        if (it + 1 < ITS) {
            ng0 = ld_nt_b128(qg + bn * 32); ng1 = ld_nt_b128(qg + bn * 32 + 16);
            nu0 = ld_nt_b128(qu + bn * 32); nu1 = ld_nt_b128(qu + bn * 32 + 16);
            nsg = sg[bn]; nsu = su[bn];
        }
        const uint4 * xb = reinterpret_cast<const uint4 *>(s_x + b * NVH_STRIDE);
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const uint4 xa = xb[2 * s + 0];
            const uint4 xc = xb[2 * s + 1];
            const uint32_t gq0 = (uint32_t) (s < 2 ? (s == 0 ? g0.x : g0.z) : (s == 2 ? g1.x : g1.z));
            const uint32_t gq1 = (uint32_t) (s < 2 ? (s == 0 ? g0.y : g0.w) : (s == 2 ? g1.y : g1.w));
            const uint32_t uq0 = (uint32_t) (s < 2 ? (s == 0 ? u0.x : u0.z) : (s == 2 ? u1.x : u1.z));
            const uint32_t uq1 = (uint32_t) (s < 2 ? (s == 0 ? u0.y : u0.w) : (s == 2 ? u1.y : u1.w));
            accg = fmaf(nvfp4_sub_dot_h(gq0, gq1, xa, xc, 0.0f), ue4m3_to_fp32((sgw >> (8 * s)) & 0xFFu), accg);
            accu = fmaf(nvfp4_sub_dot_h(uq0, uq1, xa, xc, 0.0f), ue4m3_to_fp32((suw >> (8 * s)) & 0xFFu), accu);
        }
        g0 = ng0; g1 = ng1; u0 = nu0; u1 = nu1; sgw = nsg; suw = nsu; b = bn;
    }
    accg = group_reduce_sum<NV_GULPR>(accg);
    accu = group_reduce_sum<NV_GULPR>(accu);
    if (g == 0) {
        act[k * N_FF_EXP + r] = situ_default(accg * s2, accu * s2);
    }
}

// ===========================================================================
// Part 2. NVFP4 MoE: down + weighted sum + sconv epilogue
//   x layout (per lane g = lane&15): 32 floats = 4 uint4 in VGPRs,
//   16 floats = 2 uint4 in VGPRs, 16 floats = 2 uint4 in LDS, tail 8 floats =
//   1 uint4 in LDS.  LDS: 8 experts x 16 lanes x 3 uint4 + 128 B = 6272 B.
// ===========================================================================
constexpr int DN_K   = N_FF_EXP;    // 1152
constexpr int DN_QSR = DN_K / 2;    // 576
constexpr int DN_SCR = DN_K / 16;   // 72

template <int GRID, int R>
__device__ __forceinline__ void nvfp4_down_fdot2_phase(
    const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
    const float * __restrict__ scale2, const float * __restrict__ act,
    const int32_t * __restrict__ topk_ids, const float * __restrict__ topk_w,
    float * __restrict__ out, const SconvEpi & epi,
    uint4 (*s_a1)[16][2], uint4 (*s_a2)[16], float (*s_red)[8][4]) {
    constexpr int RPB = 2 * R;
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k    = warp;
    const int g    = lane & 15;
    const int rloc = lane >> 4;

    const int expr   = topk_ids[k];
    const bool eok   = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex     = eok ? expr : 0;
    const float coef = eok ? topk_w[k] * scale2[ex] : 0.0f;

    const float4 * __restrict__ ak = reinterpret_cast<const float4 *>(act + (size_t) k * DN_K);

    // 32 floats [32g, 32g+32) -> 16 half2 in 4 uint4 (VGPRs)
    uint4 x0[4];
    {
        const float4 a0 = ak[g * 8 + 0], a1 = ak[g * 8 + 1], a2 = ak[g * 8 + 2], a3 = ak[g * 8 + 3];
        const float4 a4 = ak[g * 8 + 4], a5 = ak[g * 8 + 5], a6 = ak[g * 8 + 6], a7 = ak[g * 8 + 7];
        x0[0] = make_uint4(h2u(a0.x, a0.y), h2u(a0.z, a0.w), h2u(a1.x, a1.y), h2u(a1.z, a1.w));
        x0[1] = make_uint4(h2u(a2.x, a2.y), h2u(a2.z, a2.w), h2u(a3.x, a3.y), h2u(a3.z, a3.w));
        x0[2] = make_uint4(h2u(a4.x, a4.y), h2u(a4.z, a4.w), h2u(a5.x, a5.y), h2u(a5.z, a5.w));
        x0[3] = make_uint4(h2u(a6.x, a6.y), h2u(a6.z, a6.w), h2u(a7.x, a7.y), h2u(a7.z, a7.w));
    }
    // 16 floats [512+32g, +16) -> 8 half2 in 2 uint4 (VGPRs)
    uint4 x1[2];
    {
        const float4 b0 = ak[(16 + g) * 8 + 0], b1 = ak[(16 + g) * 8 + 1];
        const float4 b2 = ak[(16 + g) * 8 + 2], b3 = ak[(16 + g) * 8 + 3];
        x1[0] = make_uint4(h2u(b0.x, b0.y), h2u(b0.z, b0.w), h2u(b1.x, b1.y), h2u(b1.z, b1.w));
        x1[1] = make_uint4(h2u(b2.x, b2.y), h2u(b2.z, b2.w), h2u(b3.x, b3.y), h2u(b3.z, b3.w));
    }
    if (rloc == 0) {
        const float4 c0 = ak[(16 + g) * 8 + 4], c1 = ak[(16 + g) * 8 + 5];
        const float4 c2 = ak[(16 + g) * 8 + 6], c3 = ak[(16 + g) * 8 + 7];
        s_a1[k][g][0] = make_uint4(h2u(c0.x, c0.y), h2u(c0.z, c0.w), h2u(c1.x, c1.y), h2u(c1.z, c1.w));
        s_a1[k][g][1] = make_uint4(h2u(c2.x, c2.y), h2u(c2.z, c2.w), h2u(c3.x, c3.y), h2u(c3.z, c3.w));
    } else {
        const int sb = g >> 1;
        const int wi = g & 1;
        const float4 d0 = ak[256 + sb * 4 + wi];
        const float4 d1 = ak[256 + sb * 4 + 2 + wi];
        s_a2[k][g] = make_uint4(h2u(d0.x, d0.y), h2u(d0.z, d0.w), h2u(d1.x, d1.y), h2u(d1.z, d1.w));
    }
    __syncthreads();

    int iter = 0;
    for (int r0 = blockIdx.x * RPB; r0 < N_EMBD; r0 += GRID * RPB, ++iter) {
        // R rows in flight per half-warp: every weight load of the iteration is
        // issued before the first dot product, so the DRAM latency of row rr+1
        // is covered by the arithmetic of row rr (the shipped R=1 shape has no
        // cross-row prefetch and only reaches ~640 GB/s here).
        v4i      w00[R], w10[R];
        uint32_t w2[R], s0[R], s1[R], s2[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            const int r = r0 + rloc * R + rr;
            const size_t row = (size_t) ex * N_EMBD + r;
            const uint8_t * q = qs + row * DN_QSR;
            w00[rr] = ld_nt_b128(q + g * 16);
            w10[rr] = ld_nt_b128(q + 256 + g * 16);
            w2[rr]  = ld_nt_b32(q + 512 + 4 * g);
            s0[rr]  = reinterpret_cast<const uint16_t *>(scales + row * DN_SCR)[g];
            s1[rr]  = reinterpret_cast<const uint16_t *>(scales + row * DN_SCR)[16 + g];
            s2[rr]  = (scales + row * DN_SCR)[64 + (g >> 1)];
        }

        const uint4 y1a = s_a1[k][g][0];
        const uint4 y1b = s_a1[k][g][1];
        const uint4 y2  = s_a2[k][g];
        float acc[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            __builtin_amdgcn_sched_barrier(0);
            float a = nvfp4_sub_dot_h((uint32_t) w00[rr].x, (uint32_t) w00[rr].y, x0[0], x0[1], 0.0f) *
                      ue4m3_to_fp32(s0[rr] & 0xFFu);
            a = fmaf(nvfp4_sub_dot_h((uint32_t) w00[rr].z, (uint32_t) w00[rr].w, x0[2], x0[3], 0.0f),
                     ue4m3_to_fp32(s0[rr] >> 8), a);
            __builtin_amdgcn_sched_barrier(0);
            a = fmaf(nvfp4_sub_dot_h((uint32_t) w10[rr].x, (uint32_t) w10[rr].y, x1[0], x1[1], 0.0f),
                     ue4m3_to_fp32(s1[rr] & 0xFFu), a);
            a = fmaf(nvfp4_sub_dot_h((uint32_t) w10[rr].z, (uint32_t) w10[rr].w, y1a, y1b, 0.0f),
                     ue4m3_to_fp32(s1[rr] >> 8), a);
            __builtin_amdgcn_sched_barrier(0);
            a = fmaf(nvfp4_tail_dot_h(w2[rr], y2, 0.0f), ue4m3_to_fp32(s2[rr]), a);
            acc[rr] = a;
        }

#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            const float v = group_reduce_sum<16>(acc[rr]) * coef;
            if (g == 0) {
                s_red[iter & 1][k][rloc * R + rr] = v;
            }
        }
        __syncthreads();
        if (warp == 0 && lane < RPB) {
            const float (*rb)[4] = s_red[iter & 1];
            const float sum = ((rb[0][lane] + rb[1][lane]) + (rb[2][lane] + rb[3][lane])) +
                              ((rb[4][lane] + rb[5][lane]) + (rb[6][lane] + rb[7][lane]));
            const int ro = r0 + lane;
            out[ro] = (epi.conv_w != nullptr) ? sconv_post_res_one(epi, ro, sum) : sum;
        }
    }
}

union MoeFdotSharedU {
    __half2 gu_x[NV_NB * NVH_STRIDE];   // 1296 half2 = 5184 B
    struct {
        uint4 a1[8][16][2];             // 4096 B
        uint4 a2[8][16];                // 2048 B
        float red[2][8][4];             //  256 B (R = 1 or 2)
    } dn;
};

template <int GRID, int GU_BLOCKS, int NWARPS_ACTIVE, int DN_R>
__device__ __forceinline__ void moe_nvfp4_fdot2_body(
    const uint8_t * __restrict__ gu_qs, const uint8_t * __restrict__ gu_sc,
    const float * __restrict__ gu_s2, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ topk_w, const float * __restrict__ x_norm,
    const uint8_t * __restrict__ dn_qs, const uint8_t * __restrict__ dn_sc,
    const float * __restrict__ dn_s2, float * __restrict__ act,
    float * __restrict__ out, const SconvEpi epi, float * __restrict__ scratch,
    MoeFdotSharedU & sh) {
    constexpr int NT = 256;
    constexpr int PAIRS_PER_BLOCK = NWARPS_ACTIVE * NV_GUPPW;
    static_assert(GU_BLOCKS * PAIRS_PER_BLOCK == N_FF_EXP * TOP_K, "phase 1 coverage");

    if ((int) blockIdx.x < GU_BLOCKS) {
        lds_fill_h2<NT, N_EMBD / 4, true>(sh.gu_x, reinterpret_cast<const float4 *>(x_norm));
    }
    __syncthreads();
    if ((int) blockIdx.x < GU_BLOCKS) {
        nvfp4_gu_fdot2_phase<NWARPS_ACTIVE>(gu_qs, gu_sc, gu_s2, topk_ids, act,
                                            blockIdx.x * PAIRS_PER_BLOCK, sh.gu_x);
    }

    grid_barrier<4>(scratch, GRID);

    nvfp4_down_fdot2_phase<GRID, DN_R>(dn_qs, dn_sc, dn_s2, act, topk_ids, topk_w, out, epi,
                                       sh.dn.a1, sh.dn.a2, sh.dn.red);
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
// GRID = 288: 3 blocks/CU pinned to 72 VGPR (6 waves/SIMD), LDS 6.3 KB.
__global__ void __launch_bounds__(256) __attribute__((amdgpu_num_vgpr(72)))
moe_nvfp4_fdot2_fused288_kernel(
    const uint8_t * __restrict__ gu_qs, const uint8_t * __restrict__ gu_sc,
    const float * __restrict__ gu_s2, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ topk_w, const float * __restrict__ x_norm,
    const uint8_t * __restrict__ dn_qs, const uint8_t * __restrict__ dn_sc,
    const float * __restrict__ dn_s2, float * __restrict__ act,
    float * __restrict__ out, const SconvEpi epi, float * __restrict__ scratch) {
    __shared__ MoeFdotSharedU sh;
    // R = 1 here: the 72-VGPR pin that 3 blocks/CU needs cannot hold the R = 2
    // ring without spilling, and this grid was slower than `_192` anyway.
    moe_nvfp4_fdot2_body<288, 288, 4, 1>(gu_qs, gu_sc, gu_s2, topk_ids, topk_w, x_norm,
                                      dn_qs, dn_sc, dn_s2, act, out, epi, scratch, sh);
}

// GRID = 192: 2 blocks/CU, no VGPR pin (A/B).
__global__ void __launch_bounds__(256)
moe_nvfp4_fdot2_fused192_kernel(
    const uint8_t * __restrict__ gu_qs, const uint8_t * __restrict__ gu_sc,
    const float * __restrict__ gu_s2, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ topk_w, const float * __restrict__ x_norm,
    const uint8_t * __restrict__ dn_qs, const uint8_t * __restrict__ dn_sc,
    const float * __restrict__ dn_s2, float * __restrict__ act,
    float * __restrict__ out, const SconvEpi epi, float * __restrict__ scratch) {
    __shared__ MoeFdotSharedU sh;
    moe_nvfp4_fdot2_body<192, 192, 6, 2>(gu_qs, gu_sc, gu_s2, topk_ids, topk_w, x_norm,
                                      dn_qs, dn_sc, dn_s2, act, out, epi, scratch, sh);
}


// Split A/B: phase-1-only and phase-2-only launches (no grid barrier).
__global__ void __launch_bounds__(128)
moe_nvfp4_fdot2_gu_kernel(
    const uint8_t * __restrict__ gu_qs, const uint8_t * __restrict__ gu_sc,
    const float * __restrict__ gu_s2, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ x_norm, float * __restrict__ act) {
    __shared__ __half2 s_x[NV_NB * NVH_STRIDE];
    lds_fill_h2<128, N_EMBD / 4, true>(s_x, reinterpret_cast<const float4 *>(x_norm));
    __syncthreads();
    nvfp4_gu_fdot2_phase<4>(gu_qs, gu_sc, gu_s2, topk_ids, act, blockIdx.x * 32, s_x);
}

__global__ void __launch_bounds__(256)
moe_nvfp4_fdot2_dn_kernel(
    const uint8_t * __restrict__ dn_qs, const uint8_t * __restrict__ dn_sc,
    const float * __restrict__ dn_s2, const float * __restrict__ act,
    const int32_t * __restrict__ topk_ids, const float * __restrict__ topk_w,
    float * __restrict__ out, const SconvEpi epi) {
    __shared__ uint4 s_a1[8][16][2];
    __shared__ uint4 s_a2[8][16];
    __shared__ float s_red[2][8][4];
    nvfp4_down_fdot2_phase<288, 2>(dn_qs, dn_sc, dn_s2, act, topk_ids, topk_w, out, epi,
                                   s_a1, s_a2, s_red);
}

// ===========================================================================
// Part 3. Q8_0 MoE (4 layers): gate_up + SITU, down + wsum, fused.
// ===========================================================================
constexpr int Q8_K    = N_EMBD;
constexpr int Q8_NB   = Q8_K / 32;   // 72
constexpr int Q8_LPR  = 32;          // shipped: one pair per warp

__device__ __forceinline__ void q8_gu_fdot2_phase(
    const int8_t * __restrict__ qs, const __half * __restrict__ d,
    const int32_t * __restrict__ topk_ids, float * __restrict__ act,
    const __half2 * __restrict__ s_x) {
    constexpr int FULL_ITS  = Q8_NB / Q8_LPR;   // 2
    constexpr int TOTAL_ITS = FULL_ITS + 1;     // 3 (128-weight tail)
    constexpr int NGROUP    = (N_FF_EXP * TOP_K) / (192 * 8);

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g    = lane % Q8_LPR;

#pragma unroll 1
    for (int grp = 0; grp < NGROUP; ++grp) {
        const int p    = (blockIdx.x * 8 + warp) * NGROUP + grp;
        const int k    = p / N_FF_EXP;
        const int r    = p - k * N_FF_EXP;
        const int expr = topk_ids[k];
        const bool eok = (unsigned) expr < (unsigned) N_EXPERT;
        const int ex   = eok ? expr : 0;

        const size_t row_g = (size_t) ex * N_GATE_UP_EXP + r;
        const size_t row_u = row_g + N_FF_EXP;
        const int8_t * qg = qs + row_g * Q8_K;
        const int8_t * qu = qs + row_u * Q8_K;
        const __half * dg = d + row_g * Q8_NB;
        const __half * du = d + row_u * Q8_NB;

        v4i rg0, rg1, ru0, ru1;
        __half rhg, rhu;
        {
            const int bb = g;
            rg0 = ld_nt_b128(qg + bb * 32); rg1 = ld_nt_b128(qg + bb * 32 + 16);
            ru0 = ld_nt_b128(qu + bb * 32); ru1 = ld_nt_b128(qu + bb * 32 + 16);
            rhg = dg[bb]; rhu = du[bb];
        }
        float accg = 0.0f;
        float accu = 0.0f;
#pragma unroll
        for (int it = 0; it < TOTAL_ITS; ++it) {
            __builtin_amdgcn_sched_barrier(0);
            const int b = g + it * Q8_LPR;
            const bool active = (it < FULL_ITS) || (b < Q8_NB);
            const v4i g0 = rg0, g1 = rg1, u0 = ru0, u1 = ru1;
            const float sgv = __half2float(rhg);
            const float suv = __half2float(rhu);
            if (it + 1 < TOTAL_ITS) {
                const int bn = b + Q8_LPR;
                if (bn < Q8_NB) {
                    rg0 = ld_nt_b128(qg + bn * 32); rg1 = ld_nt_b128(qg + bn * 32 + 16);
                    ru0 = ld_nt_b128(qu + bn * 32); ru1 = ld_nt_b128(qu + bn * 32 + 16);
                    rhg = dg[bn]; rhu = du[bn];
                }
            }
            if (active) {
                const uint4 * xb = reinterpret_cast<const uint4 *>(s_x + b * Q8H_STRIDE);
                float pg = 0.0f, pu = 0.0f;
                pg = q8_word_dot_fdot2((uint32_t) g0.x, xb[0].x, xb[0].y, pg);
                pu = q8_word_dot_fdot2((uint32_t) u0.x, xb[0].x, xb[0].y, pu);
                pg = q8_word_dot_fdot2((uint32_t) g0.y, xb[0].z, xb[0].w, pg);
                pu = q8_word_dot_fdot2((uint32_t) u0.y, xb[0].z, xb[0].w, pu);
                pg = q8_word_dot_fdot2((uint32_t) g0.z, xb[1].x, xb[1].y, pg);
                pu = q8_word_dot_fdot2((uint32_t) u0.z, xb[1].x, xb[1].y, pu);
                pg = q8_word_dot_fdot2((uint32_t) g0.w, xb[1].z, xb[1].w, pg);
                pu = q8_word_dot_fdot2((uint32_t) u0.w, xb[1].z, xb[1].w, pu);
                pg = q8_word_dot_fdot2((uint32_t) g1.x, xb[2].x, xb[2].y, pg);
                pu = q8_word_dot_fdot2((uint32_t) u1.x, xb[2].x, xb[2].y, pu);
                pg = q8_word_dot_fdot2((uint32_t) g1.y, xb[2].z, xb[2].w, pg);
                pu = q8_word_dot_fdot2((uint32_t) u1.y, xb[2].z, xb[2].w, pu);
                pg = q8_word_dot_fdot2((uint32_t) g1.z, xb[3].x, xb[3].y, pg);
                pu = q8_word_dot_fdot2((uint32_t) u1.z, xb[3].x, xb[3].y, pu);
                pg = q8_word_dot_fdot2((uint32_t) g1.w, xb[3].z, xb[3].w, pg);
                pu = q8_word_dot_fdot2((uint32_t) u1.w, xb[3].z, xb[3].w, pu);
                accg = fmaf(pg, sgv, accg);
                accu = fmaf(pu, suv, accu);
            }
        }
        accg = group_reduce_sum<Q8_LPR>(accg);
        accu = group_reduce_sum<Q8_LPR>(accu);
        if (g == 0) {
            act[k * N_FF_EXP + r] = eok ? situ_default(accg, accu) : 0.0f;
        }
    }
}

template <int R, int GRID>
__device__ __forceinline__ void q8_down_fdot2_phase(
    const int8_t * __restrict__ qs, const __half * __restrict__ d,
    const float * __restrict__ act, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ topk_w, float * __restrict__ out, const SconvEpi & epi,
    uint4 (*s_a1)[16][2], uint4 (*s_a2)[16], float (*s_red)[2 * R]) {
    constexpr int K   = DN_K;
    constexpr int NB  = K / 32;   // 36
    constexpr int RPB = 2 * R;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k    = warp;
    const int g    = lane & 15;
    const int rloc = lane >> 4;

    const int expr   = topk_ids[k];
    const bool eok   = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex     = eok ? expr : 0;
    const float coef = eok ? topk_w[k] : 0.0f;

    const float4 * ak = reinterpret_cast<const float4 *>(act) + k * (K / 4);
    uint4 x0[4];
    {
        const float4 a0 = ak[g * 8 + 0], a1 = ak[g * 8 + 1], a2 = ak[g * 8 + 2], a3 = ak[g * 8 + 3];
        const float4 a4 = ak[g * 8 + 4], a5 = ak[g * 8 + 5], a6 = ak[g * 8 + 6], a7 = ak[g * 8 + 7];
        x0[0] = make_uint4(h2u(a0.x, a0.y), h2u(a0.z, a0.w), h2u(a1.x, a1.y), h2u(a1.z, a1.w));
        x0[1] = make_uint4(h2u(a2.x, a2.y), h2u(a2.z, a2.w), h2u(a3.x, a3.y), h2u(a3.z, a3.w));
        x0[2] = make_uint4(h2u(a4.x, a4.y), h2u(a4.z, a4.w), h2u(a5.x, a5.y), h2u(a5.z, a5.w));
        x0[3] = make_uint4(h2u(a6.x, a6.y), h2u(a6.z, a6.w), h2u(a7.x, a7.y), h2u(a7.z, a7.w));
    }
    uint4 x1[2];
    {
        const float4 b0 = ak[(16 + g) * 8 + 0], b1 = ak[(16 + g) * 8 + 1];
        const float4 b2 = ak[(16 + g) * 8 + 2], b3 = ak[(16 + g) * 8 + 3];
        x1[0] = make_uint4(h2u(b0.x, b0.y), h2u(b0.z, b0.w), h2u(b1.x, b1.y), h2u(b1.z, b1.w));
        x1[1] = make_uint4(h2u(b2.x, b2.y), h2u(b2.z, b2.w), h2u(b3.x, b3.y), h2u(b3.z, b3.w));
    }
    if (rloc == 0) {
        const float4 c0 = ak[(16 + g) * 8 + 4], c1 = ak[(16 + g) * 8 + 5];
        const float4 c2 = ak[(16 + g) * 8 + 6], c3 = ak[(16 + g) * 8 + 7];
        s_a1[k][g][0] = make_uint4(h2u(c0.x, c0.y), h2u(c0.z, c0.w), h2u(c1.x, c1.y), h2u(c1.z, c1.w));
        s_a1[k][g][1] = make_uint4(h2u(c2.x, c2.y), h2u(c2.z, c2.w), h2u(c3.x, c3.y), h2u(c3.z, c3.w));
    } else {
        const float4 d0 = ak[256 + g * 2 + 0];
        const float4 d1 = ak[256 + g * 2 + 1];
        s_a2[k][g] = make_uint4(h2u(d0.x, d0.y), h2u(d0.z, d0.w), h2u(d1.x, d1.y), h2u(d1.z, d1.w));
    }
    __syncthreads();

    for (int r0 = blockIdx.x * RPB; r0 < N_EMBD; r0 += GRID * RPB) {
        v4i    w0_0[R], w0_1[R], w1_0[R], w1_1[R];
        v2u    w2[R];
        __half h0[R], h1[R], h2[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            const int r = r0 + rloc * R + rr;
            const size_t row = (size_t) ex * N_EMBD + r;
            const int8_t * q  = qs + row * K;
            const __half * dd = d + row * NB;
            w0_0[rr] = ld_nt_b128(q + g * 32);
            w0_1[rr] = ld_nt_b128(q + g * 32 + 16);
            h0[rr]   = dd[g];
            w1_0[rr] = ld_nt_b128(q + (16 + g) * 32);
            w1_1[rr] = ld_nt_b128(q + (16 + g) * 32 + 16);
            h1[rr]   = dd[16 + g];
            w2[rr]   = ld_nt_b64(q + 1024 + g * 8);
            h2[rr]   = dd[32 + (g >> 2)];
        }

        float acc[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            float p0 = 0.0f;
            p0 = q8_word_dot_fdot2((uint32_t) w0_0[rr].x, x0[0].x, x0[0].y, p0);
            p0 = q8_word_dot_fdot2((uint32_t) w0_0[rr].y, x0[0].z, x0[0].w, p0);
            p0 = q8_word_dot_fdot2((uint32_t) w0_0[rr].z, x0[1].x, x0[1].y, p0);
            p0 = q8_word_dot_fdot2((uint32_t) w0_0[rr].w, x0[1].z, x0[1].w, p0);
            p0 = q8_word_dot_fdot2((uint32_t) w0_1[rr].x, x0[2].x, x0[2].y, p0);
            p0 = q8_word_dot_fdot2((uint32_t) w0_1[rr].y, x0[2].z, x0[2].w, p0);
            p0 = q8_word_dot_fdot2((uint32_t) w0_1[rr].z, x0[3].x, x0[3].y, p0);
            p0 = q8_word_dot_fdot2((uint32_t) w0_1[rr].w, x0[3].z, x0[3].w, p0);
            acc[rr] = p0 * __half2float(h0[rr]);
        }

        const uint4 y1a = s_a1[k][g][0];
        const uint4 y1b = s_a1[k][g][1];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            float p1 = 0.0f;
            p1 = q8_word_dot_fdot2((uint32_t) w1_0[rr].x, x1[0].x, x1[0].y, p1);
            p1 = q8_word_dot_fdot2((uint32_t) w1_0[rr].y, x1[0].z, x1[0].w, p1);
            p1 = q8_word_dot_fdot2((uint32_t) w1_0[rr].z, x1[1].x, x1[1].y, p1);
            p1 = q8_word_dot_fdot2((uint32_t) w1_0[rr].w, x1[1].z, x1[1].w, p1);
            p1 = q8_word_dot_fdot2((uint32_t) w1_1[rr].x, y1a.x, y1a.y, p1);
            p1 = q8_word_dot_fdot2((uint32_t) w1_1[rr].y, y1a.z, y1a.w, p1);
            p1 = q8_word_dot_fdot2((uint32_t) w1_1[rr].z, y1b.x, y1b.y, p1);
            p1 = q8_word_dot_fdot2((uint32_t) w1_1[rr].w, y1b.z, y1b.w, p1);
            acc[rr] = fmaf(p1, __half2float(h1[rr]), acc[rr]);
        }

        const uint4 y2 = s_a2[k][g];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            float p2 = 0.0f;
            p2 = q8_word_dot_fdot2(w2[rr].x, y2.x, y2.y, p2);
            p2 = q8_word_dot_fdot2(w2[rr].y, y2.z, y2.w, p2);
            acc[rr] = fmaf(p2, __half2float(h2[rr]), acc[rr]);
        }
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            const float v = group_reduce_sum<16>(acc[rr]) * coef;
            if (g == 0) {
                s_red[k][rloc * R + rr] = v;
            }
        }
        __syncthreads();
        if (threadIdx.x < RPB) {
            const int rr = threadIdx.x;
            const float sum = ((s_red[0][rr] + s_red[1][rr]) + (s_red[2][rr] + s_red[3][rr])) +
                              ((s_red[4][rr] + s_red[5][rr]) + (s_red[6][rr] + s_red[7][rr]));
            const int r = r0 + rr;
            out[r] = (epi.conv_w != nullptr) ? sconv_post_res_one(epi, r, sum) : sum;
        }
        __syncthreads();
    }
}

union MoeQ8FdotSharedU {
    __half2 gu_x[Q8_NB * Q8H_STRIDE];   // 1440 half2 = 5760 B
    struct {
        uint4 a1[8][16][2];             // 4096 B
        uint4 a2[8][16];                // 2048 B
        float red[8][4];                //  128 B
    } dn;
};

__global__ void __launch_bounds__(256)
moe_q8_fdot2_fused_kernel(
    const int8_t * __restrict__ gu_qs, const __half * __restrict__ gu_d,
    const float * __restrict__ x_norm, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ topk_w,
    const int8_t * __restrict__ dn_qs, const __half * __restrict__ dn_d,
    float * __restrict__ act, float * __restrict__ out, const SconvEpi epi,
    float * __restrict__ scratch) {
    __shared__ MoeQ8FdotSharedU sh;
    lds_fill_h2<256, N_EMBD / 4, false>(sh.gu_x, reinterpret_cast<const float4 *>(x_norm));
    __syncthreads();
    q8_gu_fdot2_phase(gu_qs, gu_d, topk_ids, act, sh.gu_x);

    grid_barrier<4>(scratch, 192);

    q8_down_fdot2_phase<2, 192>(dn_qs, dn_d, act, topk_ids, topk_w, out, epi,
                                sh.dn.a1, sh.dn.a2, sh.dn.red);
}

// ===========================================================================
// Part 4. SoA Q8_0 GEMV: half2 LDS ring + fdot2 (qkv prep, wo, lm_head)
// ===========================================================================
struct QkvSeg {
    const int8_t * qs;
    const __half * d;
    const float *  x;
    float *        y;
    int            rows;
    int            blk_end;
};

template <int LPR, int NB, int PD>
struct GemvRingH {
    static constexpr int ITS = NB / LPR;
    static_assert(ITS % PD == 0, "ring depth must divide the blocks per lane");
    v4i    r0[PD], r1[PD];
    __half rh[PD];

    __device__ __forceinline__ void load(int slot, const int8_t * wq, const __half * wd, int kb) {
        r0[slot] = ld_nt_b128(wq + kb * 32);
        r1[slot] = ld_nt_b128(wq + kb * 32 + 16);
        rh[slot] = wd[kb];
    }

    __device__ __forceinline__ void prime(const int8_t * __restrict__ qs,
                                          const __half * __restrict__ dd,
                                          int rowc, int sub) {
        const int8_t * wq = qs + (size_t) rowc * (size_t) (NB * 32);
        const __half * wd = dd + (size_t) rowc * (size_t) NB;
#pragma unroll
        for (int p = 0; p < PD; ++p) {
            if (p < ITS) {
                load(p, wq, wd, sub + p * LPR);
            }
        }
    }

    __device__ __forceinline__ float run(const int8_t * __restrict__ qs,
                                         const __half * __restrict__ dd,
                                         int rowc, int nrowc, bool has_next,
                                         const __half2 * s_x, int sub) {
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
            const uint4 * xb = reinterpret_cast<const uint4 *>(s_x + (sub + it * LPR + xo) * Q8H_STRIDE);
            float p = 0.0f, q = 0.0f;
            p = q8_word_dot_fdot2((uint32_t) w0.x, xb[0].x, xb[0].y, p);
            q = q8_word_dot_fdot2((uint32_t) w1.x, xb[2].x, xb[2].y, q);
            p = q8_word_dot_fdot2((uint32_t) w0.y, xb[0].z, xb[0].w, p);
            q = q8_word_dot_fdot2((uint32_t) w1.y, xb[2].z, xb[2].w, q);
            p = q8_word_dot_fdot2((uint32_t) w0.z, xb[1].x, xb[1].y, p);
            q = q8_word_dot_fdot2((uint32_t) w1.z, xb[3].x, xb[3].y, q);
            p = q8_word_dot_fdot2((uint32_t) w0.w, xb[1].z, xb[1].w, p);
            q = q8_word_dot_fdot2((uint32_t) w1.w, xb[3].z, xb[3].w, q);
            acc = fmaf(p + q, sc, acc);
        }
        return acc;
    }
};

template <int NWARPS, int LPR, int NB, int PD, bool EPI>
__device__ __forceinline__ void q8_gemv_fdot2_body(
        QkvSeg s0, QkvSeg s1, QkvSeg s2, __half2 * s_x, const SconvEpi & epi) {
    constexpr int NT  = NWARPS * 32;
    constexpr int K   = NB * 32;
    constexpr int RPW = 32 / LPR;
    constexpr int RPB = NWARPS * RPW;
    constexpr int NX4 = K / 4;
    constexpr int PER = (NX4 + NT - 1) / NT;

    const int b  = blockIdx.x;
    const int si = (b >= s0.blk_end) + (b >= s1.blk_end);
    const QkvSeg & sg = si == 0 ? s0 : (si == 1 ? s1 : s2);
    const int8_t * __restrict__ qs = sg.qs;
    const __half * __restrict__ dd = sg.d;
    float * __restrict__ y = sg.y;
    const int rows = sg.rows;
    const int b0   = si == 0 ? 0 : (si == 1 ? s0.blk_end : s1.blk_end);
    const int lb   = b - b0;
    const int nblk = sg.blk_end - b0;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sub  = lane % LPR;
    const int rloc = lane / LPR;
    const int rofs = warp * RPW + rloc;
    const int step = nblk * RPB;

    const float4 * __restrict__ x4 = reinterpret_cast<const float4 *>(sg.x);
    float4 xv[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * NT;
        xv[i] = x4[idx < NX4 ? idx : NX4 - 1];
    }
    int row0 = lb * RPB;
    GemvRingH<LPR, NB, PD> ring;
    {
        const int row = row0 + rofs;
        ring.prime(qs, dd, row < rows ? row : rows - 1, sub);
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * NT;
        if (idx < NX4) {
            const int base = lay_q8h(idx);
            s_x[base]     = __floats2half2_rn(xv[i].x, xv[i].y);
            s_x[base + 1] = __floats2half2_rn(xv[i].z, xv[i].w);
        }
    }
    __syncthreads();

    for (; row0 < rows; row0 += step) {
        const int  row      = row0 + rofs;
        const bool valid    = row < rows;
        const bool has_next = row0 + step < rows;
        const int  nrow     = row + step;
        float acc = ring.run(qs, dd, valid ? row : rows - 1, nrow < rows ? nrow : rows - 1,
                             has_next, s_x, sub);
        acc = group_reduce_sum<LPR>(acc);
        if (sub == 0 && valid) {
            if (EPI && si == 0) {
                y[row] = sconv_post_res_one(epi, row, acc);
            } else {
                y[row] = acc;
            }
        }
    }
}

// ---- QKV GEMV + qkvr_prep ------------------------------------------------
constexpr int WARP_SIZE = 32;

__device__ __forceinline__ void wave32_fwht_128_float4(float4 & v, const int lane) {
    const float a0 = v.x + v.y, a1 = v.x - v.y;
    const float a2 = v.z + v.w, a3 = v.z - v.w;
    float r[4] = { a0 + a2, a1 + a3, a0 - a2, a1 - a3 };
#pragma unroll
    for (int m = 1; m <= 16; m <<= 1) {
        const bool hi = (lane & m) != 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float cur   = r[j];
            const float other = __shfl_xor(cur, m, WARP_SIZE);
            r[j] = hi ? (other - cur) : (cur + other);
        }
    }
    v = make_float4(r[0], r[1], r[2], r[3]);
}

__device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

__device__ __forceinline__ uint32_t pack_u4x4_lo(int q0, int q1, int q2, int q3) {
    return ((uint32_t)(q0 & 0xF)) |
           ((uint32_t)(q1 & 0xF) << 8) |
           ((uint32_t)(q2 & 0xF) << 16) |
           ((uint32_t)(q3 & 0xF) << 24);
}

__device__ __forceinline__ void store_kv_quantized(
        float4 v, const KVCacheSlot & kv_slot, bool is_v, int h, int idx, int lane) {
    int8_t * dst_qs = is_v ? kv_slot.v_qs : kv_slot.k_qs;
    __half * dst_d  = is_v ? kv_slot.v_d  : kv_slot.k_d;
    const size_t head_slot = (size_t)h * (size_t)kv_slot.cap + (size_t)idx;
    const int grp = lane >> 3;
    const int sub = lane & 7;

    if (kv_slot.kv_type == KV_Q8_0) {
        float amax = fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)));
        amax = fmaxf(amax, __shfl_xor(amax, 4, 8));
        amax = fmaxf(amax, __shfl_xor(amax, 2, 8));
        amax = fmaxf(amax, __shfl_xor(amax, 1, 8));

        const float  d_q  = amax * (1.0f / 127.0f);
        const float  id_q = (amax > 0.0f) ? (127.0f / amax) : 0.0f;
        const __half d_h  = __float2half(d_q);

        const int q0 = __float2int_rn(v.x * id_q);
        const int q1 = __float2int_rn(v.y * id_q);
        const int q2 = __float2int_rn(v.z * id_q);
        const int q3 = __float2int_rn(v.w * id_q);

        ((int32_t *)(dst_qs + head_slot * HEAD_DIM))[lane] = pack_i8x4(q0, q1, q2, q3);
        if (sub == 0) {
            dst_d[head_slot * 4 + grp] = d_h;
        }
    } else if (kv_slot.kv_type == KV_Q4_1) {
        float vmin = fminf(fminf(v.x, v.y), fminf(v.z, v.w));
        float vmax = fmaxf(fmaxf(v.x, v.y), fmaxf(v.z, v.w));
        vmin = fminf(vmin, __shfl_xor(vmin, 4, 8));
        vmin = fminf(vmin, __shfl_xor(vmin, 2, 8));
        vmin = fminf(vmin, __shfl_xor(vmin, 1, 8));
        vmax = fmaxf(vmax, __shfl_xor(vmax, 4, 8));
        vmax = fmaxf(vmax, __shfl_xor(vmax, 2, 8));
        vmax = fmaxf(vmax, __shfl_xor(vmax, 1, 8));

        const float span = vmax - vmin;
        const float d_q  = span * (1.0f / 15.0f);
        const float id_q = (span > 0.0f) ? (15.0f / span) : 0.0f;

        const int q0 = min(15, max(0, __float2int_rn((v.x - vmin) * id_q)));
        const int q1 = min(15, max(0, __float2int_rn((v.y - vmin) * id_q)));
        const int q2 = min(15, max(0, __float2int_rn((v.z - vmin) * id_q)));
        const int q3 = min(15, max(0, __float2int_rn((v.w - vmin) * id_q)));

        const uint32_t lo4 = pack_u4x4_lo(q0, q1, q2, q3);
        const uint32_t hi4 = __shfl_xor(lo4, 1, 32);
        if ((lane & 1) == 0) {
            ((uint32_t *)(dst_qs + head_slot * (HEAD_DIM / 2)))[lane >> 1] = lo4 | (hi4 << 4);
        }
        if (sub == 0) {
            ((__half2 *)dst_d)[head_slot * 4 + grp] = __floats2half2_rn(d_q, vmin);
        }
    } else {
        float max_v = v.x;
        float amax  = fabsf(v.x);
        if (fabsf(v.y) > amax) { amax = fabsf(v.y); max_v = v.y; }
        if (fabsf(v.z) > amax) { amax = fabsf(v.z); max_v = v.z; }
        if (fabsf(v.w) > amax) { amax = fabsf(v.w); max_v = v.w; }

#pragma unroll
        for (int mask = 4; mask >= 1; mask >>= 1) {
            const float o_amax = __shfl_xor(amax,  mask, 8);
            const float o_maxv = __shfl_xor(max_v, mask, 8);
            if (o_amax > amax || (o_amax == amax && o_maxv < max_v)) {
                amax  = o_amax;
                max_v = o_maxv;
            }
        }

        const float  d_q  = max_v * (-1.0f / 8.0f);
        const float  id_q = (d_q != 0.0f) ? (1.0f / d_q) : 0.0f;
        const __half d_h  = __float2half(d_q);

        const int q0 = min(15, max(0, __float2int_rn(v.x * id_q) + 8));
        const int q1 = min(15, max(0, __float2int_rn(v.y * id_q) + 8));
        const int q2 = min(15, max(0, __float2int_rn(v.z * id_q) + 8));
        const int q3 = min(15, max(0, __float2int_rn(v.w * id_q) + 8));

        const uint32_t lo4 = pack_u4x4_lo(q0, q1, q2, q3);
        const uint32_t hi4 = __shfl_xor(lo4, 1, 32);
        if ((lane & 1) == 0) {
            ((uint32_t *)(dst_qs + head_slot * (HEAD_DIM / 2)))[lane >> 1] = lo4 | (hi4 << 4);
        }
        if (sub == 0) {
            dst_d[head_slot * 4 + grp] = d_h;
        }
    }
}

// LDS: the GEMV x-buffer (5760 B) is unioned with the rel-table staging
// (24 x 16 floats + 8 x 17 floats = 2080 B) -- the sharded grid barrier
// separates them, exactly like the v3 kernel's union.
constexpr int QKV_LDS_H2 = Q8_NB * Q8H_STRIDE;      // 1440 half2 = 5760 B
constexpr int QKV_LDS_B  = QKV_LDS_H2 * 4;

template <int NWARPS, int GRID>
__global__ void __launch_bounds__(NWARPS * 32)
qkv_qkvr_prep_fdot2_kernel(
        QkvSeg s0, QkvSeg s1, QkvSeg s2,
        const float * __restrict__ q_in, const float * __restrict__ q_clip_w,
        const float * __restrict__ r_in, const float * __restrict__ r_clip_w,
        const float * __restrict__ rel_bias_w,
        const float * __restrict__ k_raw, const float * __restrict__ k_conv_w,
        const float * __restrict__ k_dyn,
        const float * __restrict__ v_raw, const float * __restrict__ v_conv_w,
        const float * __restrict__ v_dyn,
        float * __restrict__ sconv_hist_kv,
        KVCacheSlot kv_slot,
        float * __restrict__ q_out,
        float * __restrict__ rel_table_out,
        bool apply_tau,
        const StepMeta * __restrict__ d_meta,
        float * __restrict__ scratch) {
    constexpr int N_RT_BLOCKS = GRID - 40;
    const int bid  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;

    __shared__ __align__(16) unsigned char s_lds[QKV_LDS_B];
    __half2 * s_x = reinterpret_cast<__half2 *>(s_lds);
    float (*s_r)[N_REL] = reinterpret_cast<float (*)[N_REL]>(s_lds);
    float (*s_b)[N_REL + 1] = reinterpret_cast<float (*)[N_REL + 1]>(s_lds + (size_t) N_HEAD * N_REL * 4);

    const SconvEpi none = {nullptr, nullptr, nullptr, nullptr};
    q8_gemv_fdot2_body<NWARPS, 8, Q8_NB, 3, false>(s0, s1, s2, s_x, none);

    // ------------------------- sharded grid barrier -------------------------
    rdna3_flush_stores();
    __syncthreads();
    if (tid == 0) {
        rdna3_flush_stores();
        (void) scratch_add(scratch, 392 + (bid % 3));
    }
    __syncthreads();
    if (tid == 0) {
        spin_until(scratch, 392, GRID / 3);
        spin_until(scratch, 393, GRID / 3);
        spin_until(scratch, 394, GRID / 3);
    }
    __syncthreads();
    rdna3_inv_l0_l1();

    // ------------------------------ qkvr prep ------------------------------
    if (bid < 24) {
        if (tid < 32) {
            const int h = bid;
            float4 v = ((const float4 *)(q_in + (size_t)h * HEAD_DIM))[lane];
            if (q_clip_w) {
                const float4 w = ((const float4 *)(q_clip_w + (size_t)h * HEAD_DIM))[lane];
                v.x *= w.x; v.y *= w.y; v.z *= w.z; v.w *= w.w;
            }
            float sumsq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            sumsq = warp_reduce_sum32(sumsq);
            const float tv = (apply_tau && d_meta != nullptr) ? d_meta->tau : 1.0f;
            const float factor = fminf(1.0f, QK_CLIP / (sqrtf(sumsq) + RMS_EPS)) * tv * KQ_SCALE;
            v.x *= factor; v.y *= factor; v.z *= factor; v.w *= factor;
            wave32_fwht_128_float4(v, lane);
            ((float4 *)(q_out + (size_t)h * HEAD_DIM))[lane] = v;
        }
    } else if (bid < 40) {
        if (tid < 32) {
            const bool is_v = (bid >= 32);
            const int  h    = is_v ? (bid - 32) : (bid - 24);
            const int  c_base = h * HEAD_DIM + lane * 4;

            const float * raw_ptr  = is_v ? v_raw : k_raw;
            const float * cw_ptr   = is_v ? v_conv_w : k_conv_w;
            const float * dyn_ptr  = is_v ? v_dyn : k_dyn;
            float       * hist_ptr = is_v ? (sconv_hist_kv + (size_t)N_KV * N_CONV_HIST) : sconv_hist_kv;
            const bool    has_dyn  = (dyn_ptr != nullptr);

            float4 w[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                w[j] = ((const float4 *)cw_ptr)[c_base + j];
            }

            const float * rt = raw_ptr + c_base;
            float x3[4] = { rt[0], rt[1], rt[2], rt[3] };

            float4 d_dyn = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (has_dyn) {
                d_dyn = ((const float4 *)dyn_ptr)[h];
            }
            const float conv_scale = has_dyn ? (1.0f / 3.0f) : 0.4472135954999579f;

            float y[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float * hs = hist_ptr + (size_t)(c_base + j) * N_CONV_HIST;
                const float x0 = hs[0];
                const float x1 = hs[1];
                const float x2 = hs[2];
                float acc = x3[j] + x0 * w[j].x + x1 * w[j].y + x2 * w[j].z + x3[j] * w[j].w;
                if (has_dyn) {
                    acc += x3[j] * d_dyn.x + x2 * d_dyn.y + x1 * d_dyn.z + x0 * d_dyn.w;
                }
                y[j] = acc * conv_scale;
                hs[0] = x1;
                hs[1] = x2;
                hs[2] = x3[j];
            }

            float4 v = make_float4(y[0], y[1], y[2], y[3]);
            float sumsq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            sumsq = warp_reduce_sum32(sumsq);

            const float factor = is_v
                ? (rsqrtf(sumsq * (1.0f / (float)HEAD_DIM) + RMS_EPS) * KQ_SCALE)
                : (fminf(1.0f, QK_CLIP / (sqrtf(sumsq) + RMS_EPS)) * KQ_SCALE);
            v.x *= factor; v.y *= factor; v.z *= factor; v.w *= factor;

            wave32_fwht_128_float4(v, lane);

            const int32_t pos = (d_meta != nullptr) ? d_meta->pos : 0;
            const int32_t idx = (d_meta != nullptr)
                ? (kv_slot.is_swa ? d_meta->kv_write_swa : d_meta->kv_write_global)
                : 0;

            store_kv_quantized(v, kv_slot, is_v, h, idx, lane);
            if (!is_v && h == 0 && lane == 0) {
                kv_slot.k_pos[idx] = pos;
            }
        }
    } else {
        const int rid   = bid - 40;
        const int d_lo  = (rid * N_REL_EXTENT) / N_RT_BLOCKS;
        const int d_hi  = ((rid + 1) * N_REL_EXTENT) / N_RT_BLOCKS;
        const int d_cnt = d_hi - d_lo;

        const float tv = (apply_tau && d_meta != nullptr) ? d_meta->tau : 1.0f;

        // The (tid >> 4) mapping defines exactly 8 x 16 = 128 slots.
        if (tid < 128) {
#pragma unroll
            for (int step = 0; step < 3; ++step) {
                const int h = step * 8 + (tid >> 4);
                const int k = tid & 15;
                float rv = r_in[(size_t)h * N_REL + k];
                if (r_clip_w) {
                    rv *= r_clip_w[(size_t)h * N_REL + k];
                }
                const float sumsq = group_reduce_sum<N_REL>(rv * rv);
                const float factor = fminf(1.0f, REL_CLIP / (sqrtf(sumsq) + RMS_EPS)) * tv;
                s_r[h][k] = rv * factor;
            }
        }
        if (tid < d_cnt * N_REL) {
            const int dl = tid / N_REL;
            const int k  = tid % N_REL;
            s_b[dl][k] = rel_bias_w[(size_t)(d_lo + dl) * N_REL + k];
        }
        __syncthreads();

        if (tid < d_cnt * N_HEAD) {
            const int dl = tid / N_HEAD;
            const int h  = tid % N_HEAD;
            float dot = 0.0f;
#pragma unroll
            for (int k = 0; k < N_REL; ++k) {
                dot = fmaf(s_b[dl][k], s_r[h][k], dot);
            }
            rel_table_out[(size_t)h * N_REL_EXTENT + d_lo + dl] = dot;
        }
    }

    // ------------------------- reset the barrier -------------------------
    rdna3_flush_stores();
    __syncthreads();
    if (tid == 0) {
        const int prev = scratch_add(scratch, 395);
        if (prev == GRID - 1) {
            scratch_store(scratch, 392, 0);
            scratch_store(scratch, 393, 0);
            scratch_store(scratch, 394, 0);
            scratch_store(scratch, 395, 0);
        }
    }
}

// ---- wo (2304 x 3072) standalone + sconv epilogue -------------------------
template <int NWARPS, int LPR, int PD>
__global__ void __launch_bounds__(NWARPS * 32)
wo_fdot2_kernel(QkvSeg s0, SconvEpi epi) {
    __shared__ __half2 s_x[(N_Q / 32) * Q8H_STRIDE];
    QkvSeg none;
    none.qs = nullptr; none.d = nullptr; none.x = nullptr; none.y = nullptr;
    none.rows = 0; none.blk_end = 0x7FFFFFFF;
    q8_gemv_fdot2_body<NWARPS, LPR, N_Q / 32, PD, true>(s0, none, none, s_x, epi);
}

// ---- lm_head (rows up to 201024, K = 2304) -------------------------------
template <int NWARPS, int LPR, int PD>
__global__ void __launch_bounds__(NWARPS * 32)
gemv_fdot2_kernel(QkvSeg s0) {
    __shared__ __half2 s_x[Q8_NB * Q8H_STRIDE];
    QkvSeg none;
    none.qs = nullptr; none.d = nullptr; none.x = nullptr; none.y = nullptr;
    none.rows = 0; none.blk_end = 0x7FFFFFFF;
    const SconvEpi noepi = {nullptr, nullptr, nullptr, nullptr};
    q8_gemv_fdot2_body<NWARPS, LPR, Q8_NB, PD, false>(s0, none, none, s_x, noepi);
}

// ---------------------------------------------------------------------------
// Step 8A: m-row lm_head GEMV (m = 1..8).  One weight stream for all m rows.
// Per row r the accumulation order is identical to gemv_fdot2_kernel
// <NWARPS, LPR, PD>: acc_r = fmaf(p + q, scale, acc_r) over it = 0..ITS-1 in
// ascending order, then the same LPR-way shfl_xor reduction, so the result is
// bit-identical to m single-row launches.  The x staging uses a 16-half2 block
// stride (the shipped kernel pads to 20 for bank conflicts) shared by all rows:
// M * 96 * 16 * 4 B = 49 KiB for M = 8.
// ---------------------------------------------------------------------------
template <int NWARPS, int LPR, int PD, int M>
__global__ void __launch_bounds__(NWARPS * 32)
gemv_fdot2_mrows_kernel(QkvSeg s0, int m, int ystride) {
    constexpr int NB  = Q8_NB;
    constexpr int K   = NB * 32;
    constexpr int RPW = 32 / LPR;
    constexpr int RPB = NWARPS * RPW;
    constexpr int NX4 = K / 4;
    constexpr int PER = (NX4 + NWARPS * 32 - 1) / (NWARPS * 32);
    constexpr int XROW = NB * 16; // half2 per row of the staged activations

    __shared__ __half2 s_x[M * XROW];

    const int b  = blockIdx.x;
    const int8_t * __restrict__ qs = s0.qs;
    const __half * __restrict__ dd = s0.d;
    float * __restrict__ y = s0.y;
    const int rows = s0.rows;
    const int lb   = b;
    const int nblk = s0.blk_end;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sub  = lane % LPR;
    const int rloc = lane / LPR;
    const int rofs = warp * RPW + rloc;
    const int step = nblk * RPB;

    // Stage m activation rows.
    #pragma unroll
    for (int r = 0; r < M; ++r) {
        if (r >= m) break;
        const float4 * __restrict__ x4 = reinterpret_cast<const float4 *>(s0.x + (size_t)r * K);
        float4 xv[PER];
        #pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int idx = threadIdx.x + i * (NWARPS * 32);
            xv[i] = x4[idx < NX4 ? idx : NX4 - 1];
        }
        #pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int idx = threadIdx.x + i * (NWARPS * 32);
            if (idx < NX4) {
                const int base = r * XROW + (idx >> 3) * 16 + (idx & 7) * 2;
                s_x[base]     = __floats2half2_rn(xv[i].x, xv[i].y);
                s_x[base + 1] = __floats2half2_rn(xv[i].z, xv[i].w);
            }
        }
    }
    __syncthreads();

    int row0 = lb * RPB;
    for (; row0 < rows; row0 += step) {
        const int  row      = row0 + rofs;
        const bool valid    = row < rows;
        const bool has_next = row0 + step < rows;
        const int  nrow     = row + step;
        const int  rowc     = valid ? row : rows - 1;
        const int  nrowc    = nrow < rows ? nrow : rows - 1;

        const int8_t * wq  = qs + (size_t) rowc * (size_t) (NB * 32);
        const __half * wd  = dd + (size_t) rowc * (size_t) NB;
        const int8_t * nwq = qs + (size_t) nrowc * (size_t) (NB * 32);
        const __half * nwd = dd + (size_t) nrowc * (size_t) NB;

        // Ring prefetch (PD blocks in flight) of the row's weight blocks.
        v4i r0[PD], r1[PD];
        __half rh[PD];
        #pragma unroll
        for (int p = 0; p < PD; ++p) {
            if (p < NB / LPR) {
                r0[p] = ld_nt_b128(wq + (sub + p * LPR) * 32);
                r1[p] = ld_nt_b128(wq + (sub + p * LPR) * 32 + 16);
                rh[p] = wd[sub + p * LPR];
            }
        }

        float acc[M];
        #pragma unroll
        for (int r = 0; r < M; ++r) acc[r] = 0.0f;

        #pragma unroll
        for (int it = 0; it < NB / LPR; ++it) {
            const int slot = it % PD;
            const v4i w0 = r0[slot], w1 = r1[slot];
            const float sc = __half2float(rh[slot]);
            if (it + PD < NB / LPR) {
                r0[slot] = ld_nt_b128(wq + (sub + (it + PD) * LPR) * 32);
                r1[slot] = ld_nt_b128(wq + (sub + (it + PD) * LPR) * 32 + 16);
                rh[slot] = wd[sub + (it + PD) * LPR];
            } else if (has_next) {
                r0[slot] = ld_nt_b128(nwq + (sub + (it + PD - NB / LPR) * LPR) * 32);
                r1[slot] = ld_nt_b128(nwq + (sub + (it + PD - NB / LPR) * LPR) * 32 + 16);
                rh[slot] = nwd[sub + (it + PD - NB / LPR) * LPR];
            }
            const int blk = sub + it * LPR;
            #pragma unroll
            for (int r = 0; r < M; ++r) {
                if (r >= m) break;
                const uint4 * xb = reinterpret_cast<const uint4 *>(s_x + r * XROW + blk * 16);
                float p = 0.0f, q = 0.0f;
                p = q8_word_dot_fdot2((uint32_t) w0.x, xb[0].x, xb[0].y, p);
                q = q8_word_dot_fdot2((uint32_t) w1.x, xb[2].x, xb[2].y, q);
                p = q8_word_dot_fdot2((uint32_t) w0.y, xb[0].z, xb[0].w, p);
                q = q8_word_dot_fdot2((uint32_t) w1.y, xb[2].z, xb[2].w, q);
                p = q8_word_dot_fdot2((uint32_t) w0.z, xb[1].x, xb[1].y, p);
                q = q8_word_dot_fdot2((uint32_t) w1.z, xb[3].x, xb[3].y, q);
                p = q8_word_dot_fdot2((uint32_t) w0.w, xb[1].z, xb[1].w, p);
                q = q8_word_dot_fdot2((uint32_t) w1.w, xb[3].z, xb[3].w, q);
                acc[r] = fmaf(p + q, sc, acc[r]);
            }
        }
        #pragma unroll
        for (int r = 0; r < M; ++r) {
            acc[r] = group_reduce_sum<LPR>(acc[r]);
            if (sub == 0 && valid && r < m) {
                y[(size_t)r * ystride + row] = acc[r];
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
int g_occ_qkv4 = 1;   // blocks/CU of the 384-block (4 blocks/CU) QKV variant

int device_cu_count_j() {
    static int cache[64] = {0};
    int dev = -1;
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

[[noreturn]] void fdot2_contract_fail(const char * what) {
    std::fprintf(stderr, "[inkling kernels_step7j] contract violation: %s\n", what);
    std::abort();
}

SconvEpi make_epi_j(const float * conv_w, float * hist, const float * res, float * raw_out = nullptr) {
    if (conv_w != nullptr && hist == nullptr) {
        fdot2_contract_fail("sconv epilogue: post_conv_w given without post_sconv_h");
    }
    SconvEpi e;
    e.conv_w = conv_w; e.hist = hist; e.res = res; e.raw_out = raw_out;
    return e;
}

QkvSeg make_seg_j(const SoaQ8Matrix & W, const float * x, float * y) {
    QkvSeg s;
    s.qs = W.qs; s.d = W.d; s.x = x; s.y = y; s.rows = W.rows; s.blk_end = 0;
    return s;
}

template <int RPB>
int partition_segments_j(QkvSeg * s, int nseg, int cap, int grid) {
    long long groups[3] = {0, 0, 0}, total = 0;
    for (int i = 0; i < nseg; ++i) {
        groups[i] = (s[i].rows + RPB - 1) / RPB;
        total += groups[i];
    }
    int end = 0;
    for (int i = 0; i < 3; ++i) {
        if (i < nseg) {
            long long nb = groups[i];
            if (cap > 0 && total > cap) {
                const long long share = (long long) cap * groups[i] / total;
                nb = share < 1 ? 1 : (share < groups[i] ? share : groups[i]);
            }
            end += (int) nb;
        }
        s[i].blk_end = (i < nseg) ? end : 0x7FFFFFFF;
    }
    // Tail blocks (grid > row groups) must stay in the LAST REAL segment: their
    // row is clamped to rows-1 (a valid read) instead of selecting a spare
    // segment whose rows == 0 would make `rows - 1` a negative row index.
    if (nseg > 0 && end < grid) {
        s[nseg - 1].blk_end = grid;
        end = grid;
    }
    return end;
}

void check_moe_nvfp4_j(const SoaNVFP4Experts & gu, const SoaNVFP4Experts & dn) {
    if (gu.rows != N_GATE_UP_EXP || gu.K != N_EMBD) {
        fdot2_contract_fail("NVFP4 gate_up must be [*, 2304, 2304]");
    }
    if (dn.rows != N_EMBD || dn.K != N_FF_EXP) {
        fdot2_contract_fail("NVFP4 down must be [*, 2304, 1152]");
    }
}

void check_moe_q8_j(const SoaQ8Experts & gu, const SoaQ8Experts & dn) {
    if (gu.rows != N_GATE_UP_EXP || gu.K != N_EMBD) {
        fdot2_contract_fail("Q8 gate_up must be [*, 2304, 2304]");
    }
    if (dn.rows != N_EMBD || dn.K != N_FF_EXP) {
        fdot2_contract_fail("Q8 down must be [*, 2304, 1152]");
    }
}

} // namespace

// ============================================================================
// Public API
// ============================================================================
void launch_moe_nvfp4_fdot2_fused(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * scratch) {
    check_moe_nvfp4_j(gate_up, down);
    moe_nvfp4_fdot2_fused192_kernel<<<192, 256, 0, stream>>>(
        gate_up.qs, gate_up.scales, gate_up.scale2, topk_ids, topk_w, x_norm,
        down.qs, down.scales, down.scale2, tmp_act, out,
        make_epi_j(post_conv_w, post_sconv_h, post_res_add), scratch);
}

void launch_moe_nvfp4_fdot2_fused288(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * scratch) {
    check_moe_nvfp4_j(gate_up, down);
    moe_nvfp4_fdot2_fused288_kernel<<<288, 256, 0, stream>>>(
        gate_up.qs, gate_up.scales, gate_up.scale2, topk_ids, topk_w, x_norm,
        down.qs, down.scales, down.scale2, tmp_act, out,
        make_epi_j(post_conv_w, post_sconv_h, post_res_add), scratch);
}

void launch_moe_nvfp4_fdot2_split(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add) {
    check_moe_nvfp4_j(gate_up, down);
    moe_nvfp4_fdot2_gu_kernel<<<288, 128, 0, stream>>>(
        gate_up.qs, gate_up.scales, gate_up.scale2, topk_ids, x_norm, tmp_act);
    moe_nvfp4_fdot2_dn_kernel<<<288, 256, 0, stream>>>(
        down.qs, down.scales, down.scale2, tmp_act, topk_ids, topk_w, out,
        make_epi_j(post_conv_w, post_sconv_h, post_res_add));
}

void launch_moe_q8_fdot2_fused(
    const SoaQ8Experts & gate_up, const SoaQ8Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * scratch) {
    check_moe_q8_j(gate_up, down);
    moe_q8_fdot2_fused_kernel<<<192, 256, 0, stream>>>(
        gate_up.qs, gate_up.d, x_norm, topk_ids, topk_w,
        down.qs, down.d, tmp_act, out,
        make_epi_j(post_conv_w, post_sconv_h, post_res_add), scratch);
}

void launch_qkv_qkvr_prep_fdot2(
    const SoaQ8Matrix & W0, const float * x0, float * y0,
    const SoaQ8Matrix & W1, const float * x1, float * y1,
    const SoaQ8Matrix & W2, const float * x2, float * y2,
    const float * q_in, const float * q_clip_w,
    const float * r_in, const float * r_clip_w, const float * rel_bias_w,
    const float * k_raw, const float * k_conv_w, const float * k_dyn,
    const float * v_raw, const float * v_conv_w, const float * v_dyn,
    float * sconv_hist_kv,
    KVCacheSlot kv_slot,
    float * q_out,
    float * rel_table_out,
    bool apply_tau,
    const StepMeta * d_meta,
    float * scratch,
    hipStream_t stream,
    int variant) {
    if (W0.K != N_EMBD || (W1.rows > 0 && W1.K != N_EMBD) || (W2.rows > 0 && W2.K != N_EMBD) ||
        W0.rows > 12288 || W1.rows > 12288 || W2.rows > 12288 || scratch == nullptr) {
        fdot2_contract_fail("qkv fdot2 needs K == 2304, rows <= 12288 and a scratch buffer");
    }
    QkvSeg s[3];
    int nseg = 0;
    if (W0.rows > 0) s[nseg++] = make_seg_j(W0, x0, y0);
    if (W1.rows > 0) s[nseg++] = make_seg_j(W1, x1, y1);
    if (W2.rows > 0) s[nseg++] = make_seg_j(W2, x2, y2);
    if (nseg == 0) {
        return;
    }
    // Unused segments must never be selected: alias them to segment 0 with zero
    // rows and an unreachable blk_end (the v3 HIP-700 bug).
    for (int i = nseg; i < 3; ++i) {
        s[i] = s[0];
        s[i].rows = 0;
        s[i].blk_end = 0x7FFFFFFF;
    }
    if (variant == 1) {
        (void) partition_segments_j<32>(s, nseg, 288, 288);
        qkv_qkvr_prep_fdot2_kernel<8, 288><<<288, 256, 0, stream>>>(
            s[0], s[1], s[2], q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
            k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
            sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta, scratch);
    } else if (variant == 2 && g_occ_qkv4 >= 4) {
        (void) partition_segments_j<32>(s, nseg, 384, 384);
        qkv_qkvr_prep_fdot2_kernel<8, 384><<<384, 256, 0, stream>>>(
            s[0], s[1], s[2], q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
            k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
            sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta, scratch);
    } else {
        (void) partition_segments_j<16>(s, nseg, 288, 288);
        qkv_qkvr_prep_fdot2_kernel<4, 288><<<288, 128, 0, stream>>>(
            s[0], s[1], s[2], q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
            k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
            sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta, scratch);
    }
}

void launch_soa_q8_gemv_wo_fdot2(
    const SoaQ8Matrix & W, const float * x,
    const float * conv_w, float * sconv_hist, const float * res, float * out,
    hipStream_t stream, int variant) {
    if (W.rows <= 0) {
        return;
    }
    if (W.K != N_Q) {
        fdot2_contract_fail("wo_fdot2 needs K == 3072");
    }
    const SconvEpi epi = make_epi_j(conv_w, sconv_hist, res);
    QkvSeg s0 = make_seg_j(W, x, out);
    if (variant == 1) {
        constexpr int NW = 8, LPR = 32;
        const int grid = partition_segments_j<NW * (32 / LPR)>(&s0, 1, 288, 288);
        wo_fdot2_kernel<NW, LPR, 3><<<grid, NW * 32, 0, stream>>>(s0, epi);
    } else if (variant == 2) {
        constexpr int NW = 4, LPR = 16;
        const int grid = partition_segments_j<NW * (32 / LPR)>(&s0, 1, 288, 288);
        wo_fdot2_kernel<NW, LPR, 3><<<grid, NW * 32, 0, stream>>>(s0, epi);
    } else if (variant == 3) {
        constexpr int NW = 4, LPR = 32;
        const int grid = partition_segments_j<NW * (32 / LPR)>(&s0, 1, 576, 576);
        wo_fdot2_kernel<NW, LPR, 3><<<grid, NW * 32, 0, stream>>>(s0, epi);
    } else {
        constexpr int NW = 6, LPR = 32;
        const int grid = partition_segments_j<NW * (32 / LPR)>(&s0, 1, 384, 384);
        wo_fdot2_kernel<NW, LPR, 3><<<grid, NW * 32, 0, stream>>>(s0, epi);
    }
}

void launch_soa_q8_gemv_fdot2(
    const SoaQ8Matrix & W, const float * x, float * y,
    hipStream_t stream, int variant) {
    if (W.rows <= 0) {
        return;
    }
    if (W.K != N_EMBD) {
        fdot2_contract_fail("lm_head_fdot2 needs K == 2304");
    }
    QkvSeg s0 = make_seg_j(W, x, y);
    if (variant == 1) {
        // 4 warps x LPR 8 (RPB 16) -> 2x the row groups, same 16 blocks/CU cap.
        constexpr int NW = 4, LPR = 8;
        const int cap = device_cu_count_j() * 16;
        const int grid = partition_segments_j<NW * (32 / LPR)>(&s0, 1, cap, cap);
        gemv_fdot2_kernel<NW, LPR, 3><<<grid, NW * 32, 0, stream>>>(s0);
    } else {
        constexpr int NW = 8, LPR = 8;
        const int cap = device_cu_count_j() * 16;
        const int grid = partition_segments_j<NW * (32 / LPR)>(&s0, 1, cap, cap);
        gemv_fdot2_kernel<NW, LPR, 3><<<grid, NW * 32, 0, stream>>>(s0);
    }
}

void launch_soa_q8_gemv_fdot2_mrows(
    const SoaQ8Matrix & W, const float * x_rows, float * y_rows,
    int m, hipStream_t stream, int variant) {
    if (W.rows <= 0 || m <= 0) {
        return;
    }
    if (m > 8 || W.K != N_EMBD) {
        fdot2_contract_fail("launch_soa_q8_gemv_fdot2_mrows needs 1 <= m <= 8 and K == 2304");
    }
    QkvSeg s0 = make_seg_j(W, x_rows, y_rows);
    constexpr int NW = 8, LPR = 8;
    const int cap = device_cu_count_j() * 16;
    const int grid = partition_segments_j<NW * (32 / LPR)>(&s0, 1, cap, cap);
    (void)variant;
    switch (m) {
        case 1: gemv_fdot2_mrows_kernel<NW, LPR, 3, 1><<<grid, NW * 32, 0, stream>>>(s0, 1, W.rows); break;
        case 2: gemv_fdot2_mrows_kernel<NW, LPR, 3, 2><<<grid, NW * 32, 0, stream>>>(s0, 2, W.rows); break;
        case 3: gemv_fdot2_mrows_kernel<NW, LPR, 3, 3><<<grid, NW * 32, 0, stream>>>(s0, 3, W.rows); break;
        case 4: gemv_fdot2_mrows_kernel<NW, LPR, 3, 4><<<grid, NW * 32, 0, stream>>>(s0, 4, W.rows); break;
        case 5: gemv_fdot2_mrows_kernel<NW, LPR, 3, 5><<<grid, NW * 32, 0, stream>>>(s0, 5, W.rows); break;
        case 6: gemv_fdot2_mrows_kernel<NW, LPR, 3, 6><<<grid, NW * 32, 0, stream>>>(s0, 6, W.rows); break;
        case 7: gemv_fdot2_mrows_kernel<NW, LPR, 3, 7><<<grid, NW * 32, 0, stream>>>(s0, 7, W.rows); break;
        default: gemv_fdot2_mrows_kernel<NW, LPR, 3, 8><<<grid, NW * 32, 0, stream>>>(s0, 8, W.rows); break;
    }
}

bool soa_q8_gemv_fdot2_mrows_supported(const SoaQ8Matrix & W) {
    return W.K == N_EMBD && W.rows > 0 && W.rows <= VOCAB_UNPADDED;
}

void step7j_prime_occupancy() {
    (void) device_cu_count_j();
    // The 384-block QKV variant and the 288/384-block MoE variants sit behind an
    // in-kernel grid barrier, so their whole grid must be co-resident.  Query the
    // runtime's own blocks/CU estimate once (outside capture) and let the
    // launchers fall back instead of deadlocking.
    int n = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(
            &n, (const void *) qkv_qkvr_prep_fdot2_kernel<8, 384>, 256, 0) == hipSuccess) {
        g_occ_qkv4 = n;
    }
    std::fprintf(stderr, "[step7j] occupancy blocks/CU: qkv_384=%d\n", g_occ_qkv4);
}

} // namespace inkling
