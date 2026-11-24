// ============================================================================
// kernels_gemv_soa.hip — Peak-bandwidth SoA decode GEMV + fused Top-8 MoE decode
// Target: AMD RDNA3 gfx1100 (RX 7900 XTX), Wave32, 96 CUs, 960 GB/s GDDR6.
//
// Kernels (every activation stays FP32: zero activation quantisation anywhere in this file):
//   1. soa_q8_gemv_kernel          : y = W x, W SoA Q8_0, x cached FP32 in bank-conflict-free
//                                    padded LDS, int8 -> FP16 (exact) -> v_fma_mix_f32 against
//                                    FP32 x, per-lane PD-deep register ring of 32-weight blocks
//                                    (2 x b128 + fp16 scale) that runs across row groups, Wave32
//                                    shuffle reduce. Up to 3 GEMVs with the same K share one
//                                    launch (qkv3), optional fused sconv_post_res epilogue.
//   2. moe_nvfp4_gate_up_situ      : 8 experts x 1152 (gate,up) row pairs in ONE launch,
//                                    exact FP32 FMA against x_norm cached in LDS, SITU fused.
//   3. moe_nvfp4_down_wsum         : 2304 outputs, hybrid 12.1 KB LDS + 48-VGPR register-cached
//                                    warp-per-expert layout (<= 16 KB LDS, <= 128 VGPRs, 2+ blocks/CU),
//                                    weighted 8-expert sum in LDS/registers (no atomics), optional
//                                    fused sconv_post_res epilogue.
//   4. moe_q8_gate_up_situ / moe_q8_down_wsum : the same for SoA Q8_0 experts (layers 20..23).
//
// Fused sconv_post_res epilogue: the lane that stores output channel r also applies the 4-tap
// short conv + 1/sqrt(5) + residual and shifts sconv_hist[r] in place (one writer per channel:
// race-free, HIP-Graph safe). Bit-identical to sconv_post_res_kernel (kernels_core.hip).
// ============================================================================
#include "inkling_engine.h"

#include <cstdio>
#include <cstdlib>

namespace inkling {
namespace {

typedef int          v4i __attribute__((ext_vector_type(4)));
typedef unsigned int v2u __attribute__((ext_vector_type(2)));

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
// 128-bit streaming load of weights (read exactly once per decode step -> non-temporal).
__device__ __forceinline__ v4i ld_nt_b128(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const v4i *>(p));
}

// 64-bit streaming load of weights.
__device__ __forceinline__ v2u ld_nt_b64(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const v2u *>(p));
}

// Exact branchless UE4M3 -> FP32, returns raw E4M3 value * 0.5 (kvalues_mxfp4 are stored at 2x).
// Identical to ggml_cuda_ue4m3_to_fp32 (non-FP8 path). 0x7F / 0xFF (NaN) -> 0.
__device__ __forceinline__ float ue4m3_to_fp32(uint32_t x) {
    const uint32_t u   = (x == 0x7Fu || x == 0xFFu) ? 0u : (x & 0x7Fu);
    const uint32_t e   = u >> 3;
    const uint32_t m   = u & 7u;
    const uint32_t nrm = ((e + 119u) << 23) | (m << 20);
    return (e == 0u) ? ((float) m * (1.0f / 1024.0f)) : __uint_as_float(nrm);
}

// Identical to situ_scalar in ggml-cuda/inkling-rdna3.cu
__device__ __forceinline__ float situ_scalar(float gate, float up, float beta, float inv_beta, float lb, float inv_lb) {
    const float act = beta * tanhf(gate * inv_beta) / (1.0f + expf(-gate));
    const float u   = lb > 0.0f ? lb * tanhf(up * inv_lb) : up;
    return act * u;
}

__device__ __forceinline__ float situ_default(float g, float u) {
    return situ_scalar(g, u, SITU_BETA, 1.0f / SITU_BETA, SITU_LINEAR_BETA, 1.0f / SITU_LINEAR_BETA);
}

// NVFP4 nibble decode via v_perm_b32 LUT straight to FP16 bit patterns.
// kvalues_mxfp4 magnitudes {0,1,2,3,4,6,8,12} as FP16 have a zero low byte, so the high byte alone
// encodes them: {0x00,0x3C,0x40,0x42,0x44,0x46,0x48,0x4A}; the sign nibble bit maps to FP16 bit 15.
// Input q = 4 packed bytes j..j+3 of a 16-element sub-block.
//   lo -> high bytes of elements j..j+3      (low nibbles)
//   hi -> high bytes of elements j+8..j+11   (high nibbles)
__device__ __forceinline__ void nvfp4_hbytes(uint32_t q, uint32_t & lo, uint32_t & hi) {
    constexpr uint32_t T0 = 0x42403C00u; // mags 0,1,2,3
    constexpr uint32_t T1 = 0x4A484644u; // mags 4,6,8,12
    lo = __builtin_amdgcn_perm(T1, T0, q & 0x07070707u)        | ((q & 0x08080808u) << 4);
    hi = __builtin_amdgcn_perm(T1, T0, (q >> 4) & 0x07070707u) | (q & 0x80808080u);
}

// 4 FP16 high bytes -> exact FP32 FMA with x (4 floats) via v_fma_mix_f32 (bit-identical to
// v_cvt_f32_f16 + v_fma_f32, saving 4 VALU convert ops and 3 temporary VGPRs per 4 weights).
__device__ __forceinline__ float nvfp4_hb_dot_mix(uint32_t hb, const float4 x, float acc) {
    const __half2 h01 = __builtin_bit_cast(__half2, __builtin_amdgcn_perm(hb, 0u, 0x050C040Cu));
    const __half2 h23 = __builtin_bit_cast(__half2, __builtin_amdgcn_perm(hb, 0u, 0x070C060Cu));
    acc = fmaf(__low2float(h01),  x.x, acc);
    acc = fmaf(__high2float(h01), x.y, acc);
    acc = fmaf(__low2float(h23),  x.z, acc);
    acc = fmaf(__high2float(h23), x.w, acc);
    return acc;
}

// Unscaled dot of one 16-element NVFP4 sub-block (2 packed words) with x[0..15].
__device__ __forceinline__ float nvfp4_sub_dot(uint32_t q0, uint32_t q1,
                                               const float4 x0, const float4 x1, const float4 x2, const float4 x3) {
    uint32_t lo0, hi0, lo1, hi1;
    nvfp4_hbytes(q0, lo0, hi0); // elems 0..3 , 8..11
    nvfp4_hbytes(q1, lo1, hi1); // elems 4..7 , 12..15
    float acc = 0.0f;
    acc = nvfp4_hb_dot_mix(lo0, x0, acc);
    acc = nvfp4_hb_dot_mix(lo1, x1, acc);
    acc = nvfp4_hb_dot_mix(hi0, x2, acc);
    acc = nvfp4_hb_dot_mix(hi1, x3, acc);
    return acc;
}

// 4 signed int8 packed in w -> exact FP32 dot with x (4 floats). int8 -> FP16 is exact
// (0x6400 | (q + 128) = 1152 + q, minus 1152 in FP16 is exact) and fmaf(float(half), x, acc)
// compiles to v_fma_mix_f32, which widens the FP16 operand exactly inside the FMA: bit-identical
// to fmaf((float) q, x, acc). Per 4 weights: 1 xor + 2 perm + 2 v_pk_add_f16 + 4 v_fma_mix_f32
// (9 VALU ops) instead of 4 bfe/ashr + 4 cvt + 4 fma (12) with a 3x shorter s_delay_alu tally.
__device__ __forceinline__ float q8_word_dot_mix(uint32_t w, const float4 x, float acc) {
    const uint32_t u   = w ^ 0x80808080u;                                   // q + 128 in 0..255
    const __half2  b   = __builtin_bit_cast(__half2, 0x64806480u);          // (1152, 1152)
    const __half2  h01 = __hsub2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u)), b);
    const __half2  h23 = __hsub2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u)), b);
    acc = fmaf(__low2float(h01),  x.x, acc);
    acc = fmaf(__high2float(h01), x.y, acc);
    acc = fmaf(__low2float(h23),  x.z, acc);
    acc = fmaf(__high2float(h23), x.w, acc);
    return acc;
}

// Bank-conflict-free LDS layouts for activation vectors.
template <int SEG, int SEGP, int GRP, int GRPP>
struct LdsLayout {
    static constexpr int SEG_  = SEG;
    static constexpr int SEGP_ = SEGP;
    static constexpr int GRPP_ = GRPP;
    __device__ static constexpr int map(int i) {
        return (i / GRP) * GRPP + ((i % GRP) / SEG) * SEGP + (i % SEG);
    }
};
typedef LdsLayout<16, 17, N_EMBD / 4, (N_EMBD / 64) * 17> LayNvX;
typedef LdsLayout<8, 9, N_EMBD / 4, (N_EMBD / 32) * 9>   LayQ8X;

// Copy N float4 from global into a padded LDS layout with every global load issued before the
// first LDS store (a plain strided loop serialises one L2 round trip per iteration).
template <int NT, int N, class LAY>
__device__ __forceinline__ void lds_fill(float4 * dst, const float4 * __restrict__ src) {
    constexpr int PER = (N + NT - 1) / NT;
    float4 v[PER];
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        // unconditional (clamped, in-bounds) load keeps v[] in registers; a conditional element
        // makes AMDGPUPromoteAlloca move the array into LDS (+16 KB, measured)
        const int idx = threadIdx.x + i * NT;
        v[i] = src[idx < N ? idx : N - 1];
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * NT;
        if (idx < N) {
            dst[LAY::map(idx)] = v[i];
        }
    }
}

template <int W>
__device__ __forceinline__ float group_reduce_sum(float v) {
#pragma unroll
    for (int off = W / 2; off > 0; off >>= 1) {
        v += __shfl_xor(v, off, W);
    }
    return v;
}

// ---------------------------------------------------------------------------
// Fused 4-tap static short-conv epilogue (exactly sconv_post_res_kernel of kernels_core.hip for
// one token). Called by the single lane that owns output channel r, so the in-place history shift
// is race-free and HIP-Graph safe.
// ---------------------------------------------------------------------------
struct SconvEpi {
    const float * conv_w; // [rows, 4]   (nullptr -> plain store of x3 into out)
    float *       hist;   // [rows, 3]
    const float * res;    // [rows] or nullptr
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
    // kernels_core compiles `acc * scale + res[r]` to one FMA (HAS_RES is constexpr there);
    // written explicitly so both paths round identically (bit-exact, verified by the unit test)
    const float val = (e.res != nullptr) ? fmaf(acc, 0.4472135954999579f, e.res[r])
                                         : acc * 0.4472135954999579f; // 1 / sqrt(5)
    hs[0] = x1;
    hs[1] = x2;
    hs[2] = x3;
    return val;
}

// ---------------------------------------------------------------------------
// 1. SoA Q8_0 GEMV (decode, M = 1), exact FP32 activations
// ---------------------------------------------------------------------------
struct GemvSeg {
    const int8_t * qs;
    const __half * d;
    const float *  x;
    float *        y;
    int            rows;
    int            blk_end; // exclusive end of this segment's block range
};
struct GemvArgs {
    GemvSeg  s[3];
    SconvEpi epi;
};

__device__ __forceinline__ int lay_x(int i) { return (i >> 3) * 9 + (i & 7); } // float4 index -> padded

// One lane's stream of 32-weight blocks through a PD-deep register ring.
template <int LPR, int NB, int PD>
struct GemvRing {
    static constexpr int ITS = NB / LPR;
    static_assert(ITS % PD == 0, "ring depth must divide the blocks per lane (static slots)");
    v4i    r0[PD], r1[PD];
    __half rh[PD];

    __device__ __forceinline__ void load(int slot, const int8_t * wq, const __half * wd, int kb) {
        r0[slot] = ld_nt_b128(wq + kb * 32);
        r1[slot] = ld_nt_b128(wq + kb * 32 + 16);
        rh[slot] = wd[kb];
    }

    // Prime the ring with the first PD blocks of row `rowc`.
    __device__ __forceinline__ void prime(const int8_t * __restrict__ qs, const __half * __restrict__ dd, int rowc, int sub) {
        const int8_t * wq = qs + (size_t) rowc * (size_t) (NB * 32);
        const __half * wd = dd + (size_t) rowc * (size_t) NB;
#pragma unroll
        for (int p = 0; p < PD; ++p) {
            if (p < ITS) {
                load(p, wq, wd, sub + p * LPR);
            }
        }
    }

    // Dot of row `rowc` (its first PD blocks already in the ring); refills the ring with the first
    // PD blocks of the next row only when has_next is true.
    __device__ __forceinline__ float run(const int8_t * __restrict__ qs, const __half * __restrict__ dd,
                                         int rowc, int nrowc, bool has_next,
                                         const float4 * s_x, int sub) {
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
            const float4 * xb = s_x + (sub + it * LPR + xo) * 9;
            float p = 0.0f, q = 0.0f;
            p = q8_word_dot_mix((uint32_t) w0.x, xb[0], p);
            q = q8_word_dot_mix((uint32_t) w1.x, xb[4], q);
            p = q8_word_dot_mix((uint32_t) w0.y, xb[1], p);
            q = q8_word_dot_mix((uint32_t) w1.y, xb[5], q);
            p = q8_word_dot_mix((uint32_t) w0.z, xb[2], p);
            q = q8_word_dot_mix((uint32_t) w1.z, xb[6], q);
            p = q8_word_dot_mix((uint32_t) w0.w, xb[3], p);
            q = q8_word_dot_mix((uint32_t) w1.w, xb[7], q);
            acc = fmaf(p + q, sc, acc);
        }
        return acc;
    }
};

template <int NWARPS, int LPR, int NB, int PD, bool EPI>
__global__ void __launch_bounds__(NWARPS * 32)
soa_q8_gemv_kernel(const GemvArgs a) {
    static_assert(NB > 0 && NB % LPR == 0, "compile-time K, whole blocks per lane");
    constexpr int NT  = NWARPS * 32;
    constexpr int K   = NB * 32;
    constexpr int RPW = 32 / LPR;
    constexpr int RPB = NWARPS * RPW;
    constexpr int NX4 = K / 4;                  // float4 of x
    constexpr int PER = (NX4 + NT - 1) / NT;

    __shared__ float4 s_x[NB * 9];

    // segment of this block (uniform)
    const int b  = blockIdx.x;
    const int si = (b >= a.s[0].blk_end) + (b >= a.s[1].blk_end);
    const GemvSeg & sg = si == 0 ? a.s[0] : (si == 1 ? a.s[1] : a.s[2]);
    const int8_t * __restrict__ qs = sg.qs;
    const __half * __restrict__ dd = sg.d;
    float * __restrict__ y = sg.y;
    const int rows = sg.rows;
    const int b0   = si == 0 ? 0 : (si == 1 ? a.s[0].blk_end : a.s[1].blk_end);
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
    GemvRing<LPR, NB, PD> ring;
    {
        const int row = row0 + rofs;
        ring.prime(qs, dd, row < rows ? row : rows - 1, sub);
    }
#pragma unroll
    for (int i = 0; i < PER; ++i) {
        const int idx = threadIdx.x + i * NT;
        if (idx < NX4) {
            s_x[lay_x(idx)] = xv[i];
        }
    }
    __syncthreads();

    for (; row0 < rows; row0 += step) {
        const int  row      = row0 + rofs;
        const bool valid    = row < rows;
        const bool has_next = row0 + step < rows;
        const int  nrow     = row + step;
        float acc = ring.run(qs, dd, valid ? row : rows - 1, nrow < rows ? nrow : rows - 1, has_next, s_x, sub);
        acc = group_reduce_sum<LPR>(acc);
        if (sub == 0 && valid) {
            if (EPI && si == 0) {
                y[row] = sconv_post_res_one(a.epi, row, acc);
            } else {
                y[row] = acc;
            }
        }
    }
}

// Runtime-K fallback (K % 32 == 0, any K up to the LDS limit): same math, dynamic LDS, rolled loop.
template <int NWARPS, int LPR, bool EPI>
__global__ void __launch_bounds__(NWARPS * 32)
soa_q8_gemv_generic_kernel(const GemvArgs a, int K) {
    constexpr int NT  = NWARPS * 32;
    constexpr int RPW = 32 / LPR;
    constexpr int RPB = NWARPS * RPW;
    extern __shared__ __align__(16) unsigned char smem_gemv[];
    float4 * s_x = reinterpret_cast<float4 *>(smem_gemv);

    const int b  = blockIdx.x;
    const int si = (b >= a.s[0].blk_end) + (b >= a.s[1].blk_end);
    const GemvSeg & sg = si == 0 ? a.s[0] : (si == 1 ? a.s[1] : a.s[2]);
    const int b0   = si == 0 ? 0 : (si == 1 ? a.s[0].blk_end : a.s[1].blk_end);
    const int lb   = b - b0;
    const int nblk = sg.blk_end - b0;
    const int nb   = K >> 5;

    const float4 * x4 = reinterpret_cast<const float4 *>(sg.x);
    for (int i = threadIdx.x; i < K / 4; i += NT) {
        s_x[lay_x(i)] = x4[i];
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sub  = lane % LPR;
    const int rloc = lane / LPR;
    for (int row0 = lb * RPB; row0 < sg.rows; row0 += nblk * RPB) {
        const int  row   = row0 + warp * RPW + rloc;
        const bool valid = row < sg.rows;
        const int  rowc  = valid ? row : sg.rows - 1;
        const int8_t * wq = sg.qs + (size_t) rowc * (size_t) K;
        const __half * wd = sg.d + (size_t) rowc * (size_t) nb;
        float acc = 0.0f;
#pragma unroll 2
        for (int kb = sub; kb < nb; kb += LPR) {
            const v4i w0 = ld_nt_b128(wq + kb * 32);
            const v4i w1 = ld_nt_b128(wq + kb * 32 + 16);
            const float sw = __half2float(wd[kb]);
            const float4 * xb = s_x + kb * 9;
            float p = 0.0f;
            p = q8_word_dot_mix((uint32_t) w0.x, xb[0], p);
            p = q8_word_dot_mix((uint32_t) w0.y, xb[1], p);
            p = q8_word_dot_mix((uint32_t) w0.z, xb[2], p);
            p = q8_word_dot_mix((uint32_t) w0.w, xb[3], p);
            p = q8_word_dot_mix((uint32_t) w1.x, xb[4], p);
            p = q8_word_dot_mix((uint32_t) w1.y, xb[5], p);
            p = q8_word_dot_mix((uint32_t) w1.z, xb[6], p);
            p = q8_word_dot_mix((uint32_t) w1.w, xb[7], p);
            acc = fmaf(p, sw, acc);
        }
        acc = group_reduce_sum<LPR>(acc);
        if (sub == 0 && valid) {
            sg.y[row] = (EPI && si == 0) ? sconv_post_res_one(a.epi, row, acc) : acc;
        }
    }
}

// ---------------------------------------------------------------------------
// 2. NVFP4 MoE decode, Stage 1: gate_up (both rows of a pair in the same lane group) + SITU
// ---------------------------------------------------------------------------
constexpr int NVFP4_GU_LPR = 4;

template <int NWARPS>
__global__ void __launch_bounds__(NWARPS * 32, 8 / NWARPS)
moe_nvfp4_gate_up_situ_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
                              const float * __restrict__ scale2, const float * __restrict__ x_norm,
                              const int32_t * __restrict__ topk_ids, float * __restrict__ act) {
    constexpr int NT  = NWARPS * 32;
    constexpr int K   = N_EMBD;              // 2304
    constexpr int NB  = K / 64;              // 36
    constexpr int QSR = K / 2;               // 1152 bytes per row
    constexpr int SCR = K / 16;              // 144 scale bytes per row
    constexpr int PPW = 32 / NVFP4_GU_LPR;   // 8 pairs per warp

    __shared__ float4 s_x[NB * LayNvX::SEGP_];
    lds_fill<NT, K / 4, LayNvX>(s_x, reinterpret_cast<const float4 *>(x_norm));
    __syncthreads();

    const int k     = blockIdx.y;
    const int expr  = topk_ids[k];
    const bool eok  = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex    = eok ? expr : 0;
    const float s2  = eok ? scale2[ex] : 0.0f;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g    = lane % NVFP4_GU_LPR;
    const int r    = blockIdx.x * (NWARPS * PPW) + warp * PPW + lane / NVFP4_GU_LPR; // 0..1151

    const size_t row_g = (size_t) ex * N_GATE_UP_EXP + r;
    const size_t row_u = row_g + N_FF_EXP;
    const uint8_t  * qg = qs + row_g * QSR;
    const uint8_t  * qu = qs + row_u * QSR;
    const uint32_t * sg = reinterpret_cast<const uint32_t *>(scales + row_g * SCR);
    const uint32_t * su = reinterpret_cast<const uint32_t *>(scales + row_u * SCR);

    static_assert(NB % NVFP4_GU_LPR == 0, "36 blocks = 9 per lane");
    constexpr int ITS = NB / NVFP4_GU_LPR;

    int b = g;
    v4i g0 = ld_nt_b128(qg + b * 32), g1 = ld_nt_b128(qg + b * 32 + 16);
    v4i u0 = ld_nt_b128(qu + b * 32), u1 = ld_nt_b128(qu + b * 32 + 16);
    uint32_t sgw = sg[b], suw = su[b];

    float accg = 0.0f;
    float accu = 0.0f;
#pragma unroll 1
    for (int it = 0; it < ITS; ++it) {
        const int bn = b + NVFP4_GU_LPR;
        v4i ng0 = g0, ng1 = g1, nu0 = u0, nu1 = u1;
        uint32_t nsg = sgw, nsu = suw;
        if (it + 1 < ITS) {
            ng0 = ld_nt_b128(qg + bn * 32); ng1 = ld_nt_b128(qg + bn * 32 + 16);
            nu0 = ld_nt_b128(qu + bn * 32); nu1 = ld_nt_b128(qu + bn * 32 + 16);
            nsg = sg[bn]; nsu = su[bn];
        }
        const float4 * xb = s_x + b * LayNvX::SEGP_;
#pragma unroll
        for (int s = 0; s < 4; ++s) {
            const float4 x0 = xb[s * 4 + 0];
            const float4 x1 = xb[s * 4 + 1];
            const float4 x2 = xb[s * 4 + 2];
            const float4 x3 = xb[s * 4 + 3];
            const uint32_t gq0 = (uint32_t) (s < 2 ? (s == 0 ? g0.x : g0.z) : (s == 2 ? g1.x : g1.z));
            const uint32_t gq1 = (uint32_t) (s < 2 ? (s == 0 ? g0.y : g0.w) : (s == 2 ? g1.y : g1.w));
            const uint32_t uq0 = (uint32_t) (s < 2 ? (s == 0 ? u0.x : u0.z) : (s == 2 ? u1.x : u1.z));
            const uint32_t uq1 = (uint32_t) (s < 2 ? (s == 0 ? u0.y : u0.w) : (s == 2 ? u1.y : u1.w));
            accg = fmaf(nvfp4_sub_dot(gq0, gq1, x0, x1, x2, x3), ue4m3_to_fp32((sgw >> (8 * s)) & 0xFFu), accg);
            accu = fmaf(nvfp4_sub_dot(uq0, uq1, x0, x1, x2, x3), ue4m3_to_fp32((suw >> (8 * s)) & 0xFFu), accu);
        }
        g0 = ng0; g1 = ng1; u0 = nu0; u1 = nu1; sgw = nsg; suw = nsu; b = bn;
    }
    accg = group_reduce_sum<NVFP4_GU_LPR>(accg);
    accu = group_reduce_sum<NVFP4_GU_LPR>(accu);
    if (g == 0) {
        act[k * N_FF_EXP + r] = situ_default(accg * s2, accu * s2);
    }
}

// ---------------------------------------------------------------------------
// 3. NVFP4 MoE decode, Stage 2: down + weighted 8-expert sum
// ---------------------------------------------------------------------------
// Hybrid 12.1 KB LDS + 48-VGPR register-cached Warp-Per-Expert layout:
//   - 8 warps (256 threads): warp k (0..7) owns expert k (0..7).
//   - Within warp k: g = lane & 15 (16 lanes collaborate on 1 output row of expert k),
//     rloc = lane >> 4 (0..1, 2 row streams per warp, R rows per stream -> RPB = 2 * R rows/step).
//   - Because (k, g) is invariant across all output rows:
//     * First 12 float4s (48 floats) per lane are cached directly in VGPRs (rx0[8], rx1[4]) once!
//     * Remaining 6 float4s (24 floats) per lane are cached in 12.0 KB LDS (s_act1 + s_act2) once!
//   - Cuts LDS from 39,424 B (38.5 KB) to 12,416 B (12.1 KB <= 16 KB) and VGPRs to 118 (12 waves/SIMD).
template <int R, int GRID_BLOCKS>
__global__ void __launch_bounds__(256)
moe_nvfp4_down_wsum_kernel(const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
                           const float * __restrict__ scale2, const float * __restrict__ act,
                           const int32_t * __restrict__ topk_ids, const float * __restrict__ topk_w,
                           float * __restrict__ out, const SconvEpi epi) {
    constexpr int K   = N_FF_EXP;   // 1152
    constexpr int QSR = K / 2;      // 576
    constexpr int SCR = K / 16;     // 72
    constexpr int RPB = 2 * R;

    __shared__ float4 s_act1[8][4][16]; // 8 KB (second half of 32-weight block 16+g)
    __shared__ float4 s_act2[8][2][16]; // 4 KB (tail 128 weights: 8 weights = 2 float4s per lane g)
    __shared__ float  s_red[2][8][RPB]; // 128 B double-buffered -> eliminates 2nd barrier!

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int k    = warp;
    const int g    = lane & 15;
    const int rloc = lane >> 4;

    const int expr   = topk_ids[k];
    const bool eok   = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex     = eok ? expr : 0;
    const float coef = eok ? topk_w[k] * scale2[ex] : 0.0f;

    const float4 * __restrict__ ak = reinterpret_cast<const float4 *>(act + (size_t) k * K);
    float4 rx0[8];
    float4 rx1[4];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        rx0[c] = ak[g * 8 + c];
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        rx1[c] = ak[(16 + g) * 8 + c];
    }
    if (rloc == 0) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            s_act1[k][c][g] = ak[(16 + g) * 8 + 4 + c];
        }
    } else {
        const int sb = g >> 1;
        const int wi = g & 1;
        s_act2[k][0][g] = ak[256 + sb * 4 + wi];
        s_act2[k][1][g] = ak[256 + sb * 4 + 2 + wi];
    }
    __syncthreads();

    int iter = 0;
    for (int r0 = blockIdx.x * RPB; r0 < N_EMBD; r0 += GRID_BLOCKS * RPB, ++iter) {
        v4i      w0[R], w1[R];
        uint16_t sc0[R], sc1[R];
        uint32_t w2[R];
        uint8_t  sc2[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            const int r = r0 + rloc + 2 * rr;
            const size_t row = (size_t) ex * N_EMBD + r;
            const uint8_t  * q  = qs + row * QSR;
            const uint16_t * sc = reinterpret_cast<const uint16_t *>(scales + row * SCR);
            w0[rr]  = ld_nt_b128(q + g * 16);
            w1[rr]  = ld_nt_b128(q + (16 + g) * 16);
            w2[rr]  = __builtin_nontemporal_load(reinterpret_cast<const uint32_t *>(q + 512) + g);
            sc0[rr] = sc[g];
            sc1[rr] = sc[16 + g];
            sc2[rr] = (scales + row * SCR)[64 + (g >> 1)];
        }

        float acc[R];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            acc[rr] = nvfp4_sub_dot((uint32_t) w0[rr].x, (uint32_t) w0[rr].y, rx0[0], rx0[1], rx0[2], rx0[3]) *
                      ue4m3_to_fp32(sc0[rr] & 0xFFu);
            acc[rr] = fmaf(nvfp4_sub_dot((uint32_t) w0[rr].z, (uint32_t) w0[rr].w, rx0[4], rx0[5], rx0[6], rx0[7]),
                           ue4m3_to_fp32(sc0[rr] >> 8), acc[rr]);
        }

        __builtin_amdgcn_sched_barrier(0);
        const float4 x1_4 = s_act1[k][0][g];
        const float4 x1_5 = s_act1[k][1][g];
        const float4 x1_6 = s_act1[k][2][g];
        const float4 x1_7 = s_act1[k][3][g];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            acc[rr] = fmaf(nvfp4_sub_dot((uint32_t) w1[rr].x, (uint32_t) w1[rr].y, rx1[0], rx1[1], rx1[2], rx1[3]),
                           ue4m3_to_fp32(sc1[rr] & 0xFFu), acc[rr]);
            acc[rr] = fmaf(nvfp4_sub_dot((uint32_t) w1[rr].z, (uint32_t) w1[rr].w, x1_4, x1_5, x1_6, x1_7),
                           ue4m3_to_fp32(sc1[rr] >> 8), acc[rr]);
        }

        __builtin_amdgcn_sched_barrier(0);
        const float4 x2_0 = s_act2[k][0][g];
        const float4 x2_1 = s_act2[k][1][g];
        const int buf = iter & 1;
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            uint32_t lo, hi;
            nvfp4_hbytes(w2[rr], lo, hi);
            float p3 = 0.0f;
            p3 = nvfp4_hb_dot_mix(lo, x2_0, p3);
            p3 = nvfp4_hb_dot_mix(hi, x2_1, p3);
            acc[rr] = fmaf(p3, ue4m3_to_fp32(sc2[rr]), acc[rr]);
            const float v = group_reduce_sum<16>(acc[rr]) * coef;
            if (g == 0) {
                s_red[buf][k][rloc + 2 * rr] = v;
            }
        }
        __syncthreads();
        if (warp == 0 && lane < RPB) {
            const float sum = ((s_red[buf][0][lane] + s_red[buf][1][lane]) + (s_red[buf][2][lane] + s_red[buf][3][lane])) +
                              ((s_red[buf][4][lane] + s_red[buf][5][lane]) + (s_red[buf][6][lane] + s_red[buf][7][lane]));
            const int ro = r0 + lane;
            out[ro] = (epi.conv_w != nullptr) ? sconv_post_res_one(epi, ro, sum) : sum;
        }
    }
}

// ---------------------------------------------------------------------------
// 4. Q8_0 MoE decode, Stage 1: gate_up + SITU (exact FP32 FMA against x_norm in LDS)
// ---------------------------------------------------------------------------
constexpr int Q8_GU_LPR = 32;

template <int NWARPS, int PD>
__global__ void __launch_bounds__(NWARPS * 32)
moe_q8_gate_up_situ_kernel(const int8_t * __restrict__ qs, const __half * __restrict__ d,
                           const float * __restrict__ x_norm, const int32_t * __restrict__ topk_ids,
                           float * __restrict__ act) {
    constexpr int NT        = NWARPS * 32;
    constexpr int K         = N_EMBD;
    constexpr int NB        = K / 32;             // 72
    constexpr int PPW       = 32 / Q8_GU_LPR;     // 1 pair per warp
    constexpr int FULL_ITS  = NB / Q8_GU_LPR;
    constexpr bool HAS_TAIL = (NB % Q8_GU_LPR) != 0;
    constexpr int TOTAL_ITS = FULL_ITS + (HAS_TAIL ? 1 : 0);

    __shared__ float4 s_x[NB * LayQ8X::SEGP_];
    lds_fill<NT, K / 4, LayQ8X>(s_x, reinterpret_cast<const float4 *>(x_norm));
    __syncthreads();

    const int k    = blockIdx.y;
    const int expr = topk_ids[k];
    const bool eok = (unsigned) expr < (unsigned) N_EXPERT;
    const int ex   = eok ? expr : 0;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g    = lane % Q8_GU_LPR;
    const int r    = blockIdx.x * (NWARPS * PPW) + warp * PPW + lane / Q8_GU_LPR;

    const size_t row_g = (size_t) ex * N_GATE_UP_EXP + r;
    const size_t row_u = row_g + N_FF_EXP;
    const int8_t * qg = qs + row_g * K;
    const int8_t * qu = qs + row_u * K;
    const __half * dg = d + row_g * NB;
    const __half * du = d + row_u * NB;

    v4i rg0[PD], rg1[PD], ru0[PD], ru1[PD];
    __half rhg[PD], rhu[PD];
#pragma unroll
    for (int p = 0; p < PD; ++p) {
        const int bb = g + p * Q8_GU_LPR;
        if (p < TOTAL_ITS && (!HAS_TAIL || bb < NB)) {
            rg0[p] = ld_nt_b128(qg + bb * 32); rg1[p] = ld_nt_b128(qg + bb * 32 + 16);
            ru0[p] = ld_nt_b128(qu + bb * 32); ru1[p] = ld_nt_b128(qu + bb * 32 + 16);
            rhg[p] = dg[bb]; rhu[p] = du[bb];
        }
    }

    float accg = 0.0f;
    float accu = 0.0f;
#pragma unroll
    for (int it = 0; it < TOTAL_ITS; ++it) {
        __builtin_amdgcn_sched_barrier(0);
        const int slot = it % PD;
        const int b = g + it * Q8_GU_LPR;
        const bool active = (!HAS_TAIL) || (it < FULL_ITS) || (b < NB);
        const v4i g0 = rg0[slot], g1 = rg1[slot], u0 = ru0[slot], u1 = ru1[slot];
        const float sgv = __half2float(rhg[slot]);
        const float suv = __half2float(rhu[slot]);
        if (it + PD < TOTAL_ITS) {
            const int bn = b + PD * Q8_GU_LPR;
            if (!HAS_TAIL || bn < NB) {
                rg0[slot] = ld_nt_b128(qg + bn * 32); rg1[slot] = ld_nt_b128(qg + bn * 32 + 16);
                ru0[slot] = ld_nt_b128(qu + bn * 32); ru1[slot] = ld_nt_b128(qu + bn * 32 + 16);
                rhg[slot] = dg[bn]; rhu[slot] = du[bn];
            }
        }
        if (active) {
            const float4 * xb = s_x + b * LayQ8X::SEGP_;
            float pg = 0.0f, pu = 0.0f;
            float4 xv;
            xv = xb[0]; pg = q8_word_dot_mix((uint32_t) g0.x, xv, pg); pu = q8_word_dot_mix((uint32_t) u0.x, xv, pu);
            xv = xb[1]; pg = q8_word_dot_mix((uint32_t) g0.y, xv, pg); pu = q8_word_dot_mix((uint32_t) u0.y, xv, pu);
            xv = xb[2]; pg = q8_word_dot_mix((uint32_t) g0.z, xv, pg); pu = q8_word_dot_mix((uint32_t) u0.z, xv, pu);
            xv = xb[3]; pg = q8_word_dot_mix((uint32_t) g0.w, xv, pg); pu = q8_word_dot_mix((uint32_t) u0.w, xv, pu);
            xv = xb[4]; pg = q8_word_dot_mix((uint32_t) g1.x, xv, pg); pu = q8_word_dot_mix((uint32_t) u1.x, xv, pu);
            xv = xb[5]; pg = q8_word_dot_mix((uint32_t) g1.y, xv, pg); pu = q8_word_dot_mix((uint32_t) u1.y, xv, pu);
            xv = xb[6]; pg = q8_word_dot_mix((uint32_t) g1.z, xv, pg); pu = q8_word_dot_mix((uint32_t) u1.z, xv, pu);
            xv = xb[7]; pg = q8_word_dot_mix((uint32_t) g1.w, xv, pg); pu = q8_word_dot_mix((uint32_t) u1.w, xv, pu);
            accg = fmaf(pg, sgv, accg);
            accu = fmaf(pu, suv, accu);
        }
    }
    accg = group_reduce_sum<Q8_GU_LPR>(accg);
    accu = group_reduce_sum<Q8_GU_LPR>(accu);
    if (g == 0) {
        act[k * N_FF_EXP + r] = eok ? situ_default(accg, accu) : 0.0f;
    }
}

// ---------------------------------------------------------------------------
// 5. Q8_0 MoE decode, Stage 2: down + weighted 8-expert sum
// ---------------------------------------------------------------------------
// Hybrid 12.1 KB LDS + 48-VGPR register-cached Warp-Per-Expert layout:
//   - K = 1152 = 36 blocks of 32 weights.
//   - Warp k (0..7) owns expert k (0..7); g = lane & 15 (0..15), rloc = lane >> 4 (0..1).
//   - Blocks 0..15 (weights 0..511): lane g owns block g (8 float4s in rx0[8] VGPRs).
//   - Blocks 16..31 (weights 512..1023): lane g owns block 16+g (4 float4s in rx1[4] VGPRs +
//     4 float4s in s_act1[8][4][16] 8.0 KB LDS).
//   - Blocks 32..35 (weights 1024..1151 = 128 weights): lane g owns 8 contiguous weights
//     (2 float4s in s_act2[8][2][16] 4.0 KB LDS, scale d[32 + (g >> 2)]).
//   - Cuts LDS from 41,472 B (40.5 KB) to 12,416 B (12.1 KB <= 16 KB) and VGPRs to 120 (12 waves/SIMD).
template <int R, int GRID_BLOCKS>
__global__ void __launch_bounds__(256)
moe_q8_down_wsum_kernel(const int8_t * __restrict__ qs, const __half * __restrict__ d,
                        const float * __restrict__ act, const int32_t * __restrict__ topk_ids,
                        const float * __restrict__ topk_w, float * __restrict__ out, const SconvEpi epi) {
    constexpr int K   = N_FF_EXP;   // 1152
    constexpr int NB  = K / 32;     // 36
    constexpr int RPB = 2 * R;

    __shared__ float4 s_act1[8][4][16]; // 8 KB
    __shared__ float4 s_act2[8][2][16]; // 4 KB
    __shared__ float  s_red[8][RPB];

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
    float4 rx0[8];
    float4 rx1[4];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
        rx0[c] = ak[g * 8 + c];
    }
#pragma unroll
    for (int c = 0; c < 4; ++c) {
        rx1[c] = ak[(16 + g) * 8 + c];
    }
    if (rloc == 0) {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
            s_act1[k][c][g] = ak[(16 + g) * 8 + 4 + c];
        }
    } else {
#pragma unroll
        for (int c = 0; c < 2; ++c) {
            s_act2[k][c][g] = ak[256 + g * 2 + c];
        }
    }
    __syncthreads();

    for (int r0 = blockIdx.x * RPB; r0 < N_EMBD; r0 += GRID_BLOCKS * RPB) {
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
            p0 = q8_word_dot_mix((uint32_t) w0_0[rr].x, rx0[0], p0);
            p0 = q8_word_dot_mix((uint32_t) w0_0[rr].y, rx0[1], p0);
            p0 = q8_word_dot_mix((uint32_t) w0_0[rr].z, rx0[2], p0);
            p0 = q8_word_dot_mix((uint32_t) w0_0[rr].w, rx0[3], p0);
            p0 = q8_word_dot_mix((uint32_t) w0_1[rr].x, rx0[4], p0);
            p0 = q8_word_dot_mix((uint32_t) w0_1[rr].y, rx0[5], p0);
            p0 = q8_word_dot_mix((uint32_t) w0_1[rr].z, rx0[6], p0);
            p0 = q8_word_dot_mix((uint32_t) w0_1[rr].w, rx0[7], p0);
            acc[rr] = p0 * __half2float(h0[rr]);
        }

        const float4 x1_4 = s_act1[k][0][g];
        const float4 x1_5 = s_act1[k][1][g];
        const float4 x1_6 = s_act1[k][2][g];
        const float4 x1_7 = s_act1[k][3][g];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            float p1 = 0.0f;
            p1 = q8_word_dot_mix((uint32_t) w1_0[rr].x, rx1[0], p1);
            p1 = q8_word_dot_mix((uint32_t) w1_0[rr].y, rx1[1], p1);
            p1 = q8_word_dot_mix((uint32_t) w1_0[rr].z, rx1[2], p1);
            p1 = q8_word_dot_mix((uint32_t) w1_0[rr].w, rx1[3], p1);
            p1 = q8_word_dot_mix((uint32_t) w1_1[rr].x, x1_4,   p1);
            p1 = q8_word_dot_mix((uint32_t) w1_1[rr].y, x1_5,   p1);
            p1 = q8_word_dot_mix((uint32_t) w1_1[rr].z, x1_6,   p1);
            p1 = q8_word_dot_mix((uint32_t) w1_1[rr].w, x1_7,   p1);
            acc[rr] = fmaf(p1, __half2float(h1[rr]), acc[rr]);
        }

        const float4 x2_0 = s_act2[k][0][g];
        const float4 x2_1 = s_act2[k][1][g];
#pragma unroll
        for (int rr = 0; rr < R; ++rr) {
            float p2 = 0.0f;
            p2 = q8_word_dot_mix(w2[rr].x, x2_0, p2);
            p2 = q8_word_dot_mix(w2[rr].y, x2_1, p2);
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

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------
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

[[noreturn]] void gemv_contract_fail(const char * what) {
    std::fprintf(stderr, "[inkling kernels_gemv_soa] contract violation: %s\n", what);
    std::abort();
}

// Split the grid into per-segment block ranges. Each segment gets one block per row group, capped
// (persistent grid-stride beyond that) at its share of `cap` total blocks.
template <int RPB>
dim3 gemv_partition(GemvArgs & a, int nseg, int cap) {
    long long groups[3] = {0, 0, 0}, total = 0;
    for (int i = 0; i < nseg; ++i) {
        groups[i] = (a.s[i].rows + RPB - 1) / RPB;
        total += groups[i];
    }
    int end = 0;
    for (int i = 0; i < 3; ++i) {
        if (i < nseg) {
            long long nb = groups[i];
            if (total > cap) {
                const long long share = (long long) cap * groups[i] / total;
                nb = share < 1 ? 1 : (share < groups[i] ? share : groups[i]);
            }
            end += (int) nb;
        }
        a.s[i].blk_end = (i < nseg) ? end : 0x7FFFFFFF;
    }
    return dim3(end);
}

template <int NWARPS, int LPR, int NB, int PD, bool EPI>
void launch_gemv_t(GemvArgs a, int nseg, int blocks_per_cu, hipStream_t stream) {
    constexpr int RPB = NWARPS * (32 / LPR);
    const dim3 grid = gemv_partition<RPB>(a, nseg, device_cu_count() * blocks_per_cu);
    soa_q8_gemv_kernel<NWARPS, LPR, NB, PD, EPI><<<grid, NWARPS * 32, 0, stream>>>(a);
}

template <bool EPI>
void launch_gemv_generic(GemvArgs a, int nseg, int K, hipStream_t stream) {
    constexpr int NWARPS = 8, LPR = 8, RPB = NWARPS * (32 / LPR);
    const dim3 grid  = gemv_partition<RPB>(a, nseg, device_cu_count() * 16);
    const size_t lds = (size_t) (K / 32) * 9 * 16;
    soa_q8_gemv_generic_kernel<NWARPS, LPR, EPI><<<grid, NWARPS * 32, lds, stream>>>(a, K);
}

// Config per K, tuned on RX 7900 XTX (96 CUs / 48 WGPs):
//   K = 2304 (N_EMBD):
//            - Medium single/qkv3 projections (3072 < total_rows <= 12288, e.g. 8600..8664 rows):
//              4 warps, LPR 8, ring 3 (RPB = 16 rows/block -> 538..542 groups) with 288 blocks
//              (6 blocks/WGP = exact 3 x 96 CUs, 12 warps/CU, 31 KB LDS): 250/288 blocks (87%)
//              execute 2 persistent passes with ring-prefetch overlap across rows!
//            - Large lm_head (200058 rows): 8 warps, LPR 8, ring 3 (RPB = 32), 768 blocks (16/WGP).
//   K = 3072 (N_Q, wo 2304x3072): 8 warps, LPR 32, PD = 1 (RPB = 8 rows/block -> 288 row groups)
//            with 96 blocks (2 blocks/WGP = exact 1 x 96 CUs, 3 persistent rows/block).
template <bool EPI>
void launch_gemv_dispatch(const GemvArgs & a, int nseg, int K, hipStream_t stream) {
    if (K == N_EMBD) {
        long long total_rows = 0;
        for (int i = 0; i < nseg; ++i) {
            total_rows += a.s[i].rows;
        }
        if (total_rows > 3072 && total_rows <= 12288) {
            launch_gemv_t<4, 8, N_EMBD / 32, 3, EPI>(a, nseg, 6, stream);
        } else {
            launch_gemv_t<8, 8, N_EMBD / 32, 3, EPI>(a, nseg, 16, stream);
        }
    } else if (K == N_Q) {
        launch_gemv_t<8, 32, N_Q / 32, 1, EPI>(a, nseg, 2, stream);
    } else {
        launch_gemv_generic<EPI>(a, nseg, K, stream);
    }
}

void check_gemv_matrix(const SoaQ8Matrix & W) {
    if (W.K <= 0 || (W.K % 32) != 0 || (size_t) (W.K / 32) * 9 * 16 > 65536) {
        gemv_contract_fail("SoaQ8Matrix.K must be a positive multiple of 32 with K * 4.5 B <= 64 KiB");
    }
}

SconvEpi make_epi(const float * conv_w, float * hist, const float * res, float * raw_out = nullptr) {
    if (conv_w != nullptr && hist == nullptr) {
        gemv_contract_fail("fused sconv_post_res epilogue: post_conv_w given without post_sconv_h");
    }
    SconvEpi e;
    e.conv_w = conv_w; e.hist = hist; e.res = res; e.raw_out = raw_out;
    return e;
}

GemvSeg make_seg(const SoaQ8Matrix & W, const float * x, float * y) {
    GemvSeg s;
    s.qs = W.qs; s.d = W.d; s.x = x; s.y = y; s.rows = W.rows; s.blk_end = 0;
    return s;
}

} // namespace

// ============================================================================
// Public API
// ============================================================================
void launch_soa_q8_gemv(const SoaQ8Matrix & W, const float * x, float * y, hipStream_t stream) {
    if (W.rows <= 0) {
        return;
    }
    check_gemv_matrix(W);
    GemvArgs a = {};
    a.s[0] = make_seg(W, x, y);
    launch_gemv_dispatch<false>(a, 1, W.K, stream);
}

void launch_soa_q8_gemv_qkv3(const SoaQ8Matrix & W0, const float * x0, float * y0,
                             const SoaQ8Matrix & W1, const float * x1, float * y1,
                             const SoaQ8Matrix & W2, const float * x2, float * y2,
                             hipStream_t stream) {
    if (W0.K != W1.K || W0.K != W2.K) {
        gemv_contract_fail("launch_soa_q8_gemv_qkv3: all three matrices must share K");
    }
    GemvArgs a = {};
    int n = 0;
    const SoaQ8Matrix * Ws[3] = {&W0, &W1, &W2};
    const float * xs[3] = {x0, x1, x2};
    float * ys[3] = {y0, y1, y2};
    for (int i = 0; i < 3; ++i) {
        if (Ws[i]->rows > 0) {
            check_gemv_matrix(*Ws[i]);
            a.s[n++] = make_seg(*Ws[i], xs[i], ys[i]);
        }
    }
    if (n == 0) {
        return;
    }
    launch_gemv_dispatch<false>(a, n, W0.K, stream);
}

void launch_soa_q8_gemv_sconv_post_res(const SoaQ8Matrix & W, const float * x, const float * conv_w,
                                       float * sconv_hist, const float * res, float * out, hipStream_t stream) {
    if (W.rows <= 0) {
        return;
    }
    check_gemv_matrix(W);
    if (conv_w == nullptr || sconv_hist == nullptr) {
        gemv_contract_fail("launch_soa_q8_gemv_sconv_post_res: conv_w and sconv_hist are required");
    }
    GemvArgs a = {};
    a.s[0] = make_seg(W, x, out);
    a.epi.conv_w = conv_w;
    a.epi.hist   = sconv_hist;
    a.epi.res    = res;
    launch_gemv_dispatch<true>(a, 1, W.K, stream);
}

// Step 7F: stage-1 / stage-2 split so the engine can A/B each stage independently
// (v3 gate_up + shipped down, shipped gate_up + v3 down, ...).  The combined
// launch_moe_*_decode entry points below are byte-for-byte the old two launches.
void launch_moe_nvfp4_gate_up_ship(const SoaNVFP4Experts & gate_up,
                                   const float * x_norm, const int32_t * topk_ids, float * tmp_act,
                                   hipStream_t stream) {
    if (gate_up.rows != N_GATE_UP_EXP || gate_up.K != N_EMBD) {
        gemv_contract_fail("NVFP4 gate_up must be [*, 2304, 2304]");
    }
    // Stage 1: 4-warp blocks (32 pairs/block -> 36 x 8 = 288 blocks = exact 3 x 96 CUs).
    // 50% more blocks than GU_WARPS=6 -> 50% more independent DRAM streams per CU,
    // better saturating L2 MSHRs and 960 GB/s memory bandwidth.
    constexpr int GU_WARPS = 4;
    constexpr int GU_PAIRS = GU_WARPS * (32 / NVFP4_GU_LPR);       // 32 pairs per block
    static_assert(N_FF_EXP % GU_PAIRS == 0, "pair tiling");
    moe_nvfp4_gate_up_situ_kernel<GU_WARPS><<<dim3(N_FF_EXP / GU_PAIRS, TOP_K), GU_WARPS * 32, 0, stream>>>(
        gate_up.qs, gate_up.scales, gate_up.scale2, x_norm, topk_ids, tmp_act);
}

void launch_moe_nvfp4_down_ship(const SoaNVFP4Experts & down,
                                const float * tmp_act, const int32_t * topk_ids, const float * topk_w,
                                float * out, hipStream_t stream,
                                const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
                                float * raw_out) {
    if (down.rows != N_EMBD || down.K != N_FF_EXP) {
        gemv_contract_fail("NVFP4 down must be [*, 2304, 1152]");
    }
    // Stage 2: Hybrid 12.1 KB LDS + 48-VGPR register-cached kernel, 288 blocks (3 x 96 CUs), R=1.
    moe_nvfp4_down_wsum_kernel<1, 288><<<288, 256, 0, stream>>>(
        down.qs, down.scales, down.scale2, tmp_act, topk_ids, topk_w, out, make_epi(post_conv_w, post_sconv_h, post_res_add, raw_out));
}

void launch_moe_nvfp4_decode(const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
                             const float * x_norm, const int32_t * topk_ids, const float * topk_w,
                             float * tmp_act, float * out, hipStream_t stream,
                             const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
                             float * raw_out) {
    launch_moe_nvfp4_gate_up_ship(gate_up, x_norm, topk_ids, tmp_act, stream);
    launch_moe_nvfp4_down_ship(down, tmp_act, topk_ids, topk_w, out, stream,
                               post_conv_w, post_sconv_h, post_res_add, raw_out);
}

void launch_moe_q8_gate_up_ship(const SoaQ8Experts & gate_up,
                                const float * x_norm, const int32_t * topk_ids, float * tmp_act,
                                hipStream_t stream) {
    if (gate_up.rows != N_GATE_UP_EXP || gate_up.K != N_EMBD) {
        gemv_contract_fail("Q8 gate_up must be [*, 2304, 2304]");
    }
    // Stage 1: 8-warp blocks, LPR=32 (8 pairs/block -> 144 x 8 = 1152 blocks = 12 x 96 CUs), PD = 1.
    constexpr int GU_WARPS = 8;
    constexpr int GU_PAIRS = GU_WARPS * (32 / Q8_GU_LPR);           // 8 pairs per block
    static_assert(N_FF_EXP % GU_PAIRS == 0, "pair tiling");
    moe_q8_gate_up_situ_kernel<GU_WARPS, 1><<<dim3(N_FF_EXP / GU_PAIRS, TOP_K), GU_WARPS * 32, 0, stream>>>(
        gate_up.qs, gate_up.d, x_norm, topk_ids, tmp_act);
}

void launch_moe_q8_down_ship(const SoaQ8Experts & down,
                             const float * tmp_act, const int32_t * topk_ids, const float * topk_w,
                             float * out, hipStream_t stream,
                             const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
                             float * raw_out) {
    if (down.rows != N_EMBD || down.K != N_FF_EXP) {
        gemv_contract_fail("Q8 down must be [*, 2304, 1152]");
    }
    // Stage 2: Hybrid 12.1 KB LDS + 48-VGPR register-cached kernel, 192 blocks (2 blocks/CU).
    moe_q8_down_wsum_kernel<2, 192><<<192, 256, 0, stream>>>(
        down.qs, down.d, tmp_act, topk_ids, topk_w, out, make_epi(post_conv_w, post_sconv_h, post_res_add, raw_out));
}

void launch_moe_q8_decode(const SoaQ8Experts & gate_up, const SoaQ8Experts & down,
                          const float * x_norm, const int32_t * topk_ids, const float * topk_w,
                          float * tmp_act, float * out, hipStream_t stream,
                          const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
                          float * raw_out) {
    launch_moe_q8_gate_up_ship(gate_up, x_norm, topk_ids, tmp_act, stream);
    launch_moe_q8_down_ship(down, tmp_act, topk_ids, topk_w, out, stream,
                            post_conv_w, post_sconv_h, post_res_add, raw_out);
}

} // namespace inkling
