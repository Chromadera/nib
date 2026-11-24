// Step 8B: target-side batched (m-token) speculative-verify kernels, RDNA3
// gfx1100 Wave32.  Zero spills; never launched unless --mtp is active.
//
// The batched verify reads each weight matrix / expert ONCE for all m rows
// instead of once per row, which is the whole point: the shipped per-row
// decode path re-streams 3.9 GB of weights per token.

#include "kernels_verify_batch.h"
#include "sparse_global.h"

#include <cstdint>

namespace inkling {
namespace {

typedef int          v4i __attribute__((ext_vector_type(4)));
typedef unsigned int v2u __attribute__((ext_vector_type(2)));

__device__ __forceinline__ v4i ld_nt_b128(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const v4i *>(p));
}
__device__ __forceinline__ v2u ld_nt_b64(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const v2u *>(p));
}

// Exact branchless UE4M3 -> FP32 (identical to kernels_gemv_soa.hip).
__device__ __forceinline__ float ue4m3_to_fp32(uint32_t x) {
    const uint32_t u   = (x == 0x7Fu || x == 0xFFu) ? 0u : (x & 0x7Fu);
    const uint32_t e   = u >> 3;
    const uint32_t m   = u & 7u;
    const uint32_t nrm = ((e + 119u) << 23) | (m << 20);
    return (e == 0u) ? ((float) m * (1.0f / 1024.0f)) : __uint_as_float(nrm);
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

__device__ __forceinline__ float nvfp4_hb_dot_mix(uint32_t hb, const float4 x, float acc) {
    const __half2 h01 = __builtin_bit_cast(__half2, __builtin_amdgcn_perm(hb, 0u, 0x050C040Cu));
    const __half2 h23 = __builtin_bit_cast(__half2, __builtin_amdgcn_perm(hb, 0u, 0x070C060Cu));
    acc = fmaf(__low2float(h01),  x.x, acc);
    acc = fmaf(__high2float(h01), x.y, acc);
    acc = fmaf(__low2float(h23),  x.z, acc);
    acc = fmaf(__high2float(h23), x.w, acc);
    return acc;
}

__device__ __forceinline__ float nvfp4_sub_dot(uint32_t q0, uint32_t q1,
                                               const float4 x0, const float4 x1,
                                               const float4 x2, const float4 x3) {
    uint32_t lo0, hi0, lo1, hi1;
    nvfp4_hbytes(q0, lo0, hi0);
    nvfp4_hbytes(q1, lo1, hi1);
    float acc = 0.0f;
    acc = nvfp4_hb_dot_mix(lo0, x0, acc);
    acc = nvfp4_hb_dot_mix(lo1, x1, acc);
    acc = nvfp4_hb_dot_mix(hi0, x2, acc);
    acc = nvfp4_hb_dot_mix(hi1, x3, acc);
    return acc;
}

// int8 -> FP16 is exact; fmaf(float(half), x, acc) is v_fma_mix_f32 (exact widen),
// i.e. bit-identical to fmaf((float) q, x, acc).
__device__ __forceinline__ float q8_word_dot(uint32_t w, const float4 x, float acc) {
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

// ---------------------------------------------------------------------------
// 1. Exact m-row SoA Q8 GEMV (1..3 segments)
// ---------------------------------------------------------------------------
// 32 output rows per CTA (8 warps x 4 lane groups of 8), one x row per
// accumulator, K tiled in LDS so the per-row fmaf order is the engine GEMV's.
static constexpr int VG_TILE = 256;                       // floats staged per tile
static constexpr int VG_GT4  = VG_TILE / 4;               // float4 per row per tile
static constexpr int VG_GTP  = (VG_GT4 / 8) * 9;          // intra-row padded layout
static constexpr int VG_ROW  = VG_GTP + 1;                // +1 float4 row pad (row-to-row bank offset)

// Fused 4-tap shortconv + residual epilogue (bit-identical to the decode wo
// kernel's sconv_post_res_one: one fmaf, history shifted in the same lane).
struct VerifySconvEpi {
    const float * conv_w;   // [rows, 4]
    float * hist;           // [rows, 3]
    const float * res;      // [m, N_EMBD] or nullptr
    float * raw_out;        // [m, raw_stride] or nullptr
    int raw_stride;
};

template <bool EPI>
__global__ __launch_bounds__(256) void verify_gemv_mrows_kernel(
        const VerifyGemvSeg s0, const VerifyGemvSeg s1, const VerifyGemvSeg s2,
        const int n_seg, const int m, const int xstride, const VerifySconvEpi epi) {
    constexpr int NW = 8, LPR = 8, RPW = 4, RPL = 2;   // RPL output rows per lane
    constexpr int RPB = NW * RPW * RPL;                 // 64 output rows per CTA
    __shared__ float4 s_x[8][VG_ROW];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int sub  = lane % LPR;
    const int rofs = (warp * RPW + lane / LPR) * RPL;

    const int b  = blockIdx.x;
    const int si = (b >= s0.blk_end) + (b >= s1.blk_end);
    if (si >= n_seg) return;
    const VerifyGemvSeg sg = (si == 0) ? s0 : (si == 1 ? s1 : s2);
    const int b0   = (si == 0) ? 0 : (si == 1 ? s0.blk_end : s1.blk_end);
    const int lb   = b - b0;
    const int nblk = sg.blk_end - b0;
    const int rows = sg.W.rows;
    const int K    = sg.W.K;
    const int nb   = K / 32;
    const int step = nblk * RPB;

    for (int row0 = lb * RPB; row0 < rows; row0 += step) {
        const int row_a = row0 + rofs;
        const int row_b = row_a + 1;
        const bool val_a = row_a < rows;
        const bool val_b = row_b < rows;
        const int rc_a = val_a ? row_a : rows - 1;
        const int rc_b = val_b ? row_b : rows - 1;
        float acc_a[8], acc_b[8];
#pragma unroll
        for (int r = 0; r < 8; ++r) { acc_a[r] = 0.0f; acc_b[r] = 0.0f; }
        const int8_t * wqa = sg.W.qs + (size_t)rc_a * K;
        const int8_t * wqb = sg.W.qs + (size_t)rc_b * K;
        const __half * wda = sg.W.d + (size_t)rc_a * nb;
        const __half * wdb = sg.W.d + (size_t)rc_b * nb;
        for (int kt = 0; kt < K; kt += VG_TILE) {
            __syncthreads();
#pragma unroll 1
            for (int idx = (int)threadIdx.x; idx < m * VG_GT4; idx += 256) {
                const int r = idx / VG_GT4;
                const int i = idx - r * VG_GT4;
                const float4 v = *reinterpret_cast<const float4 *>(
                    sg.x + (size_t)r * xstride + kt + i * 4);
                s_x[r][(i >> 3) * 9 + (i & 7)] = v;
            }
            __syncthreads();
            const int kb = (kt >> 5) + sub;
            const v4i a0 = ld_nt_b128(wqa + (size_t)kb * 32);
            const v4i a1 = ld_nt_b128(wqa + (size_t)kb * 32 + 16);
            const v4i b0 = ld_nt_b128(wqb + (size_t)kb * 32);
            const v4i b1 = ld_nt_b128(wqb + (size_t)kb * 32 + 16);
            const float sca = __half2float(wda[kb]);
            const float scb = __half2float(wdb[kb]);
            const int32_t * wa0 = reinterpret_cast<const int32_t *>(&a0);
            const int32_t * wa1 = reinterpret_cast<const int32_t *>(&a1);
            const int32_t * wb0 = reinterpret_cast<const int32_t *>(&b0);
            const int32_t * wb1 = reinterpret_cast<const int32_t *>(&b1);
#pragma unroll 2
            for (int r = 0; r < 8; ++r) {
                if (r >= m) break;
                const float4 * xb = s_x[r] + sub * 9;
                float pa = 0.0f, qa = 0.0f, pb = 0.0f, qb = 0.0f;
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    pa = q8_word_dot((uint32_t)wa0[j], xb[j], pa);
                    pb = q8_word_dot((uint32_t)wb0[j], xb[j], pb);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    qa = q8_word_dot((uint32_t)wa1[j], xb[4 + j], qa);
                    qb = q8_word_dot((uint32_t)wb1[j], xb[4 + j], qb);
                }
                acc_a[r] = fmaf(pa + qa, sca, acc_a[r]);
                acc_b[r] = fmaf(pb + qb, scb, acc_b[r]);
            }
        }
#pragma unroll
        for (int r = 0; r < 8; ++r) {
            acc_a[r] = group_reduce_sum<LPR>(acc_a[r]);
            acc_b[r] = group_reduce_sum<LPR>(acc_b[r]);
        }
        if (sub == 0) {
            if (EPI) {
                // (Only used by the dead generic path; the wo projection has its
                //  own LPR=32 kernel below.)  Per-row independent epilogues.
#pragma unroll
                for (int g = 0; g < RPL; ++g) {
                    const int row = (g == 0) ? row_a : row_b;
                    const bool valid = (g == 0) ? val_a : val_b;
                    if (!valid) continue;
                    const float * accr = (g == 0) ? acc_a : acc_b;
                    const float4 w = reinterpret_cast<const float4 *>(epi.conv_w)[row];
                    float * hs = epi.hist + (size_t)row * N_CONV_HIST;
                    float h0 = hs[0], h1 = hs[1], h2 = hs[2];
#pragma unroll
                    for (int r = 0; r < 8; ++r) {
                        if (r >= m) break;
                        const float x3 = accr[r];
                        if (epi.raw_out != nullptr) {
                            epi.raw_out[(size_t)r * epi.raw_stride + row] = x3;
                        }
                        const float conv = x3 + h0 * w.x + h1 * w.y + h2 * w.z + x3 * w.w;
                        float val = __fmul_rn(conv, 0.4472135954999579f);
                        if (epi.res != nullptr) {
                            val = __fadd_rn(val, epi.res[(size_t)r * N_EMBD + row]);
                        }
                        h0 = h1; h1 = h2; h2 = x3;
                        sg.y[(size_t)r * rows + row] = val;
                    }
                    hs[0] = h0; hs[1] = h1; hs[2] = h2;
                }
            } else {
#pragma unroll
                for (int r = 0; r < 8; ++r) {
                    if (r >= m) break;
                    if (val_a) sg.y[(size_t)r * rows + row_a] = acc_a[r];
                    if (val_b) sg.y[(size_t)r * rows + row_b] = acc_b[r];
                }
            }
        }
    }
}

} // namespace

void launch_verify_gemv_mrows(const VerifyGemvSeg seg[3], int n_seg, int m,
                              int xstride, hipStream_t stream) {
    if (n_seg <= 0 || m <= 0) return;
    if (n_seg > 3) n_seg = 3;
    // One CTA computes 64 output rows, so a segment needs ceil(rows / 64) blocks
    // (NOT K / 32: that launched 72 CTAs for the 8600-row QKV and left 2/3 of
    // the 96 CUs idle).  Segments above the cap keep the persistent row loop.
    constexpr int MAX_BLOCKS = 96 * 16;
    const int cap = MAX_BLOCKS / n_seg;
    int total_blocks = 0;
    VerifyGemvSeg s[3];
    for (int i = 0; i < 3; ++i) {
        if (i < n_seg) {
            s[i] = seg[i];
            const int want = (s[i].W.rows + 63) / 64;
            const int blks = (want < cap) ? want : cap;
            total_blocks += (blks > 0) ? blks : 0;
            s[i].blk_end = total_blocks;
        } else {
            s[i] = VerifyGemvSeg{};
            s[i].blk_end = 0x7fffffff;   // unused segment: never selected
        }
    }
    if (total_blocks <= 0) return;
    const VerifySconvEpi none = {nullptr, nullptr, nullptr, nullptr, 0};
    verify_gemv_mrows_kernel<false><<<total_blocks, 256, 0, stream>>>(s[0], s[1], s[2], n_seg, m, xstride, none);
}

// ---------------------------------------------------------------------------
// 1b. Exact m-row Wo (K = N_Q = 3072): ONE weight stream for all m rows
// ---------------------------------------------------------------------------
// The decode wo GEMV (launch_soa_q8_gemv_wo_v2: NWARPS=6, LPR=32, NB=96, PD=3)
// gives each lane 3 of the 96 32-weight blocks of one output row and reduces
// over the 32 lanes of the warp.  Here warp w owns output row
// blockIdx.x * 6 + w and reuses its 3 blocks for every one of the m rows, then
// lane 0 runs the same causal 4-tap sconv_post_res epilogue across the rows.
namespace {
__global__ __launch_bounds__(192, 2) void verify_wo_mrows_kernel(
        const SoaQ8Matrix W, const float * __restrict__ x, int xstride,
        const float * __restrict__ conv_w, float * __restrict__ hist,
        const float * __restrict__ res, float * __restrict__ y, int m,
        float * __restrict__ raw_out, int raw_stride) {
    constexpr int NB = N_Q / 32;   // 96 = 3 blocks per lane
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * 6 + warp;
    if (row >= W.rows) return;

    const int8_t * wq = W.qs + (size_t)row * N_Q;
    const __half * wd = W.d + (size_t)row * NB;
    const int kb0 = lane, kb1 = lane + 32, kb2 = lane + 64;
    const v4i w0a = ld_nt_b128(wq + kb0 * 32), w0b = ld_nt_b128(wq + kb0 * 32 + 16);
    const v4i w1a = ld_nt_b128(wq + kb1 * 32), w1b = ld_nt_b128(wq + kb1 * 32 + 16);
    const v4i w2a = ld_nt_b128(wq + kb2 * 32), w2b = ld_nt_b128(wq + kb2 * 32 + 16);
    const float s0 = __half2float(wd[kb0]);
    const float s1 = __half2float(wd[kb1]);
    const float s2 = __half2float(wd[kb2]);

    // Lane 0's causal 4-tap sconv state (identical to sconv_post_res_one()).
    const float4 cw = reinterpret_cast<const float4 *>(conv_w)[row];
    float * hs = hist + (size_t)row * N_CONV_HIST;
    float h0 = hs[0], h1 = hs[1], h2 = hs[2];

#pragma unroll 1
    for (int t = 0; t < m; ++t) {
        const float4 * xr = reinterpret_cast<const float4 *>(x + (size_t)t * xstride);
        float a = 0.0f;
        {
            const float4 * xb = xr + kb0 * 8;
            float p = 0.0f, q = 0.0f;
            p = q8_word_dot((uint32_t)w0a.x, xb[0], p);
            q = q8_word_dot((uint32_t)w0b.x, xb[4], q);
            p = q8_word_dot((uint32_t)w0a.y, xb[1], p);
            q = q8_word_dot((uint32_t)w0b.y, xb[5], q);
            p = q8_word_dot((uint32_t)w0a.z, xb[2], p);
            q = q8_word_dot((uint32_t)w0b.z, xb[6], q);
            p = q8_word_dot((uint32_t)w0a.w, xb[3], p);
            q = q8_word_dot((uint32_t)w0b.w, xb[7], q);
            a = fmaf(p + q, s0, a);
        }
        {
            const float4 * yb = xr + kb1 * 8;
            float p = 0.0f, q = 0.0f;
            p = q8_word_dot((uint32_t)w1a.x, yb[0], p);
            q = q8_word_dot((uint32_t)w1b.x, yb[4], q);
            p = q8_word_dot((uint32_t)w1a.y, yb[1], p);
            q = q8_word_dot((uint32_t)w1b.y, yb[5], q);
            p = q8_word_dot((uint32_t)w1a.z, yb[2], p);
            q = q8_word_dot((uint32_t)w1b.z, yb[6], q);
            p = q8_word_dot((uint32_t)w1a.w, yb[3], p);
            q = q8_word_dot((uint32_t)w1b.w, yb[7], q);
            a = fmaf(p + q, s1, a);
        }
        {
            const float4 * zb = xr + kb2 * 8;
            float p = 0.0f, q = 0.0f;
            p = q8_word_dot((uint32_t)w2a.x, zb[0], p);
            q = q8_word_dot((uint32_t)w2b.x, zb[4], q);
            p = q8_word_dot((uint32_t)w2a.y, zb[1], p);
            q = q8_word_dot((uint32_t)w2b.y, zb[5], q);
            p = q8_word_dot((uint32_t)w2a.z, zb[2], p);
            q = q8_word_dot((uint32_t)w2b.z, zb[6], q);
            p = q8_word_dot((uint32_t)w2a.w, zb[3], p);
            q = q8_word_dot((uint32_t)w2b.w, zb[7], q);
            a = fmaf(p + q, s2, a);
        }
        a = group_reduce_sum<32>(a);
        if (lane == 0) {
            const float x3 = a;
            if (raw_out != nullptr) {
                raw_out[(size_t)t * raw_stride + row] = x3;
            }
            const float ca = x3 + h0 * cw.x + h1 * cw.y + h2 * cw.z + x3 * cw.w;
            const float val = (res != nullptr) ? fmaf(ca, 0.4472135954999579f, res[(size_t)t * N_EMBD + row])
                                               : ca * 0.4472135954999579f;
            y[(size_t)t * W.rows + row] = val;
            h0 = h1; h1 = h2; h2 = x3;
        }
    }
    if (lane == 0) {
        hs[0] = h0; hs[1] = h1; hs[2] = h2;
    }
}
} // namespace

void launch_verify_wo_mrows(const SoaQ8Matrix & W, const float * x, int xstride,
                            const float * conv_w, float * hist, const float * res,
                            float * y, int m, float * raw_out, int raw_stride,
                            hipStream_t stream) {
    if (W.rows <= 0 || m <= 0) return;
    if (W.K != N_Q) {
        // Unexpected shape: fall back to m exact single-row launches.
        for (int t = 0; t < m; ++t) {
            launch_soa_q8_gemv_sconv_post_res(
                W, x + (size_t)t * xstride, conv_w, hist,
                (res != nullptr) ? (res + (size_t)t * N_EMBD) : nullptr,
                y + (size_t)t * N_EMBD, stream);
        }
        return;
    }
    const int grid = (W.rows + 5) / 6;
    verify_wo_mrows_kernel<<<grid, 192, 0, stream>>>(W, x, xstride, conv_w, hist, res,
                                                     y, m, raw_out, raw_stride);
}


// ---------------------------------------------------------------------------
// 2. Expert dedup (single thread; 64 slots max)
// ---------------------------------------------------------------------------
static constexpr int VG_MOE_MAXU = 64;   // min(N_EXPERT, MAX_VERIFY_TOKENS * TOP_K)
namespace {
__global__ void verify_moe_dedup_kernel(const int32_t * __restrict__ topk_ids,
                                        const float * __restrict__ topk_w, int m,
                                        int32_t * __restrict__ unique_exp,
                                        uint32_t * __restrict__ exp_tok_mask,
                                        int32_t * __restrict__ exp_tok_slot,
                                        float * __restrict__ exp_tok_w,
                                        int32_t * __restrict__ exp_tok_cnt,
                                        int32_t * __restrict__ exp_tok_list,
                                        int32_t * __restrict__ d_num_unique) {
    if (blockIdx.x != 0) return;
    const int tid = threadIdx.x;
    // Everything is built in LDS: the shipped single-thread version did a
    // dependent global read+write per (token, slot) entry, which cost ~16 us at
    // m = 8 even though the table has only 64 entries.
    __shared__ int32_t s_ids[8 * TOP_K];
    __shared__ float   s_w[8 * TOP_K];
    __shared__ int32_t s_first[N_EXPERT];
    __shared__ int32_t s_unique[VG_MOE_MAXU];
    __shared__ uint32_t s_mask[VG_MOE_MAXU];
    __shared__ int32_t s_cnt[VG_MOE_MAXU];
    __shared__ int32_t s_slot[VG_MOE_MAXU * 8];
    __shared__ float   s_tw[VG_MOE_MAXU * 8];
    __shared__ int32_t s_list[VG_MOE_MAXU * 8];
    __shared__ int32_t s_nu;

    const int nent = (m < 8 ? m : 8) * TOP_K;
    for (int i = tid; i < nent; i += 32) {
        s_ids[i] = topk_ids[i];
        s_w[i] = topk_w[i];
    }
    __syncwarp();
    if (tid == 0) {
        #pragma unroll
        for (int e = 0; e < N_EXPERT; ++e) s_first[e] = -1;
        int nu = 0;
        for (int r = 0; r < m; ++r) {
            for (int k = 0; k < TOP_K; ++k) {
                const int raw = s_ids[r * TOP_K + k];
                const bool ok = (unsigned)raw < (unsigned)N_EXPERT;
                const int e = ok ? raw : 0;          // invalid -> expert 0, weight 0
                int u = s_first[e];
                if (u < 0) {
                    u = nu++;
                    s_first[e] = u;
                    s_unique[u] = e;
                    s_mask[u] = 0u;
                    s_cnt[u] = 0;
                    #pragma unroll
                    for (int q = 0; q < 8; ++q) {
                        s_slot[u * 8 + q] = 0;
                        s_tw[u * 8 + q] = 0.0f;
                    }
                }
                s_mask[u] |= (1u << r);
                s_slot[u * 8 + r] = k;
                s_tw[u * 8 + r] = ok ? s_w[r * TOP_K + k] : 0.0f;
                s_list[u * 8 + s_cnt[u]] = r;
                ++s_cnt[u];
            }
        }
        s_nu = nu;
    }
    __syncwarp();
    const int nu = s_nu;
    for (int u = tid; u < nu; u += 32) {
        unique_exp[u]   = s_unique[u];
        exp_tok_mask[u] = s_mask[u];
        exp_tok_cnt[u]  = s_cnt[u];
    }
    for (int i = tid; i < nu * TOP_K; i += 32) {
        exp_tok_slot[i] = s_slot[i];
        exp_tok_w[i]    = s_tw[i];
        exp_tok_list[i] = s_list[i];
    }
    __syncwarp();
    if (tid == 0) *d_num_unique = nu;
}

} // namespace

void launch_verify_moe_dedup(const int32_t * topk_ids, const float * topk_w, int m,
                             int32_t * unique_exp, uint32_t * exp_tok_mask,
                             int32_t * exp_tok_slot, float * exp_tok_w,
                             int32_t * exp_tok_cnt, int32_t * exp_tok_list,
                             int32_t * d_num_unique, hipStream_t stream) {
    verify_moe_dedup_kernel<<<1, 32, 0, stream>>>(topk_ids, topk_w, m, unique_exp,
                                                  exp_tok_mask, exp_tok_slot, exp_tok_w,
                                                  exp_tok_cnt, exp_tok_list, d_num_unique);
}

// ---------------------------------------------------------------------------
// 3. Unique-expert gate_up + SITU (NVFP4 / Q8_0)
// ---------------------------------------------------------------------------
// Grid: (max_unique, N_FF_EXP/64, 1) -- static for HIP graph capture;
// blocks with u >= *d_num_unique or with no active token exit, and each
// expert's gate/up weight block is read ONCE for all of its active tokens.
// 4 lanes per output row (LPR=4), 8 rows per warp, 64 rows per CTA.
namespace {

template <bool NVFP4>
__global__ __launch_bounds__(256) void verify_moe_gu_kernel(
        const uint8_t * __restrict__ nv_qs, const uint8_t * __restrict__ nv_sc,
        const float * __restrict__ nv_s2,
        const int8_t * __restrict__ q8_qs, const __half * __restrict__ q8_d,
        const float * __restrict__ x_norm, int xstride,
        const int32_t * __restrict__ unique_exp,
        const uint32_t * __restrict__ exp_tok_mask,
        const int32_t * __restrict__ exp_tok_slot,
        const float * __restrict__ exp_tok_w,
        const int32_t * __restrict__ exp_tok_cnt,
        const int32_t * __restrict__ exp_tok_list,
        const int32_t * __restrict__ d_num_unique, int m,
        float * __restrict__ act) {
    const int u = blockIdx.x;
    if (u >= *d_num_unique) return;
    const int cnt = exp_tok_cnt[u];
    if (cnt <= 0) return;
    const int ex = unique_exp[u];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;

    constexpr int K = N_EMBD;
    float accg[8], accu[8];
    #pragma unroll
    for (int t = 0; t < 8; ++t) { accg[t] = 0.0f; accu[t] = 0.0f; }

    if constexpr (NVFP4) {
        // 4 lanes per row (the shipped NVFP4 gate_up partition), 8 rows/warp,
        // 64 rows per CTA, 36 x 64-weight blocks -> 9 blocks per lane.
        const int g = lane & 3;
        const int r = blockIdx.y * 64 + warp * 8 + lane / 4;
        if (r >= N_FF_EXP) return;
        constexpr int NB = K / 64;
        const float s2 = nv_s2[ex];
        const size_t row_g = (size_t)ex * N_GATE_UP_EXP + r;
        const size_t row_u = row_g + N_FF_EXP;
        const uint8_t * qg = nv_qs + row_g * (K / 2);
        const uint8_t * qu = nv_qs + row_u * (K / 2);
        const uint32_t * sg = reinterpret_cast<const uint32_t *>(nv_sc + row_g * (K / 16));
        const uint32_t * su = reinterpret_cast<const uint32_t *>(nv_sc + row_u * (K / 16));
        for (int it = 0; it < NB / 4; ++it) {
            const int b = g + it * 4;
            const v4i g0 = ld_nt_b128(qg + b * 32), g1 = ld_nt_b128(qg + b * 32 + 16);
            const v4i u0 = ld_nt_b128(qu + b * 32), u1 = ld_nt_b128(qu + b * 32 + 16);
            const uint32_t sgw = sg[b], suw = su[b];
            const uint32_t gw[8] = {(uint32_t)g0.x, (uint32_t)g0.y, (uint32_t)g0.z, (uint32_t)g0.w,
                                    (uint32_t)g1.x, (uint32_t)g1.y, (uint32_t)g1.z, (uint32_t)g1.w};
            const uint32_t uw[8] = {(uint32_t)u0.x, (uint32_t)u0.y, (uint32_t)u0.z, (uint32_t)u0.w,
                                    (uint32_t)u1.x, (uint32_t)u1.y, (uint32_t)u1.z, (uint32_t)u1.w};
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (t >= cnt) break;
                const float * xr = x_norm + (size_t)exp_tok_list[u * 8 + t] * xstride + b * 64;
                const float4 xs[16] = {
                    *reinterpret_cast<const float4 *>(xr + 0),  *reinterpret_cast<const float4 *>(xr + 4),
                    *reinterpret_cast<const float4 *>(xr + 8),  *reinterpret_cast<const float4 *>(xr + 12),
                    *reinterpret_cast<const float4 *>(xr + 16), *reinterpret_cast<const float4 *>(xr + 20),
                    *reinterpret_cast<const float4 *>(xr + 24), *reinterpret_cast<const float4 *>(xr + 28),
                    *reinterpret_cast<const float4 *>(xr + 32), *reinterpret_cast<const float4 *>(xr + 36),
                    *reinterpret_cast<const float4 *>(xr + 40), *reinterpret_cast<const float4 *>(xr + 44),
                    *reinterpret_cast<const float4 *>(xr + 48), *reinterpret_cast<const float4 *>(xr + 52),
                    *reinterpret_cast<const float4 *>(xr + 56), *reinterpret_cast<const float4 *>(xr + 60)};
                #pragma unroll
                for (int s = 0; s < 4; ++s) {
                    accg[t] = fmaf(nvfp4_sub_dot(gw[2 * s], gw[2 * s + 1], xs[4 * s], xs[4 * s + 1],
                                                 xs[4 * s + 2], xs[4 * s + 3]),
                                   ue4m3_to_fp32((sgw >> (8 * s)) & 0xFFu), accg[t]);
                    accu[t] = fmaf(nvfp4_sub_dot(uw[2 * s], uw[2 * s + 1], xs[4 * s], xs[4 * s + 1],
                                                 xs[4 * s + 2], xs[4 * s + 3]),
                                   ue4m3_to_fp32((suw >> (8 * s)) & 0xFFu), accu[t]);
                }
            }
        }
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= cnt) break;
            accg[t] = group_reduce_sum<4>(accg[t]);
            accu[t] = group_reduce_sum<4>(accu[t]);
            if (g == 0) {
                const int tok = exp_tok_list[u * 8 + t];
                const int slot = exp_tok_slot[u * 8 + tok];
                act[(size_t)(tok * TOP_K + slot) * N_FF_EXP + r] =
                    situ_default(accg[t] * s2, accu[t] * s2);
            }
        }
    } else {
        // 32 lanes per row (the shipped Q8 gate_up partition), 1 row per warp,
        // 8 rows per CTA, 72 x 32-weight blocks -> 2 full + 1 tail per lane.
        const int g = lane;
        const int r = blockIdx.y * 8 + warp;
        if (r >= N_FF_EXP) return;
        constexpr int NB = K / 32;
        constexpr int FULL_ITS = NB / 32;
        const size_t row_g = (size_t)ex * N_GATE_UP_EXP + r;
        const size_t row_u = row_g + N_FF_EXP;
        const int8_t * qg = q8_qs + row_g * K;
        const int8_t * qu = q8_qs + row_u * K;
        const __half * dg = q8_d + row_g * NB;
        const __half * du = q8_d + row_u * NB;
        for (int it = 0; it <= FULL_ITS; ++it) {
            const int b = g + it * 32;
            if (b >= NB) break;
            const v4i g0 = ld_nt_b128(qg + b * 32), g1 = ld_nt_b128(qg + b * 32 + 16);
            const v4i u0 = ld_nt_b128(qu + b * 32), u1 = ld_nt_b128(qu + b * 32 + 16);
            const float sgv = __half2float(dg[b]);
            const float suv = __half2float(du[b]);
            #pragma unroll
            for (int t = 0; t < 8; ++t) {
                if (t >= cnt) break;
                const float * xr = x_norm + (size_t)exp_tok_list[u * 8 + t] * xstride + b * 32;
                const float4 x0 = *reinterpret_cast<const float4 *>(xr + 0);
                const float4 x1 = *reinterpret_cast<const float4 *>(xr + 4);
                const float4 x2 = *reinterpret_cast<const float4 *>(xr + 8);
                const float4 x3 = *reinterpret_cast<const float4 *>(xr + 12);
                const float4 x4 = *reinterpret_cast<const float4 *>(xr + 16);
                const float4 x5 = *reinterpret_cast<const float4 *>(xr + 20);
                const float4 x6 = *reinterpret_cast<const float4 *>(xr + 24);
                const float4 x7 = *reinterpret_cast<const float4 *>(xr + 28);
                float pg = 0.0f, pu = 0.0f;
                pg = q8_word_dot((uint32_t)g0.x, x0, pg); pu = q8_word_dot((uint32_t)u0.x, x0, pu);
                pg = q8_word_dot((uint32_t)g0.y, x1, pg); pu = q8_word_dot((uint32_t)u0.y, x1, pu);
                pg = q8_word_dot((uint32_t)g0.z, x2, pg); pu = q8_word_dot((uint32_t)u0.z, x2, pu);
                pg = q8_word_dot((uint32_t)g0.w, x3, pg); pu = q8_word_dot((uint32_t)u0.w, x3, pu);
                pg = q8_word_dot((uint32_t)g1.x, x4, pg); pu = q8_word_dot((uint32_t)u1.x, x4, pu);
                pg = q8_word_dot((uint32_t)g1.y, x5, pg); pu = q8_word_dot((uint32_t)u1.y, x5, pu);
                pg = q8_word_dot((uint32_t)g1.z, x6, pg); pu = q8_word_dot((uint32_t)u1.z, x6, pu);
                pg = q8_word_dot((uint32_t)g1.w, x7, pg); pu = q8_word_dot((uint32_t)u1.w, x7, pu);
                accg[t] = fmaf(pg, sgv, accg[t]);
                accu[t] = fmaf(pu, suv, accu[t]);
            }
        }
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= cnt) break;
            accg[t] = group_reduce_sum<32>(accg[t]);
            accu[t] = group_reduce_sum<32>(accu[t]);
            if (g == 0) {
                const int tok = exp_tok_list[u * 8 + t];
                const int slot = exp_tok_slot[u * 8 + tok];
                act[(size_t)(tok * TOP_K + slot) * N_FF_EXP + r] =
                    situ_default(accg[t], accu[t]);
            }
        }
    }
}

} // namespace
void launch_verify_moe_gu_nvfp4(const SoaNVFP4Experts & gate_up,
                                const float * x_norm, int xstride,
                                const int32_t * unique_exp, const uint32_t * exp_tok_mask,
                                const int32_t * exp_tok_slot, const float * exp_tok_w,
                                const int32_t * exp_tok_cnt, const int32_t * exp_tok_list,
                                const int32_t * d_num_unique, int m,
                                float * act, hipStream_t stream) {
    if (m <= 0) return;
    const dim3 grid(VG_MOE_MAXU, (N_FF_EXP + 63) / 64, 1);
    verify_moe_gu_kernel<true><<<grid, 256, 0, stream>>>(
        gate_up.qs, gate_up.scales, gate_up.scale2, nullptr, nullptr,
        x_norm, xstride, unique_exp, exp_tok_mask, exp_tok_slot, exp_tok_w,
        exp_tok_cnt, exp_tok_list, d_num_unique, m, act);
}

void launch_verify_moe_gu_q8(const SoaQ8Experts & gate_up,
                             const float * x_norm, int xstride,
                             const int32_t * unique_exp, const uint32_t * exp_tok_mask,
                             const int32_t * exp_tok_slot, const float * exp_tok_w,
                             const int32_t * exp_tok_cnt, const int32_t * exp_tok_list,
                             const int32_t * d_num_unique, int m,
                             float * act, hipStream_t stream) {
    if (m <= 0) return;
    const dim3 grid(VG_MOE_MAXU, N_FF_EXP / 8, 1);
    verify_moe_gu_kernel<false><<<grid, 256, 0, stream>>>(
        nullptr, nullptr, nullptr, gate_up.qs, gate_up.d,
        x_norm, xstride, unique_exp, exp_tok_mask, exp_tok_slot, exp_tok_w,
        exp_tok_cnt, exp_tok_list, d_num_unique, m, act);
}

// ---------------------------------------------------------------------------
// 4. Unique-expert down -> per-(row, slot) partial sums
// ---------------------------------------------------------------------------
namespace {
// Grid: (max_unique, N_EMBD/16, 1).  Each CTA computes 256 output rows of ONE
// expert for ALL of its active tokens: the full K = 1152 weight row (w0, w1,
// w2 + scales) is loaded into registers ONCE and then dotted against each
// token's SITU activations, writing buf_slot in the loop (0 extra VGPRs).
// Per-row math is the shipped down_wsum kernel's: lane g in 0..15 owns
// 32-weight chunk g, chunk 16+g and the 8-weight tail chunk;
// group_reduce_sum<16> over the lane group.
template <bool NVFP4>
__global__ __launch_bounds__(256) void verify_moe_dn_kernel(
        const uint8_t * __restrict__ nv_qs, const uint8_t * __restrict__ nv_sc,
        const float * __restrict__ nv_s2,
        const int8_t * __restrict__ q8_qs, const __half * __restrict__ q8_d,
        const float * __restrict__ act,
        const int32_t * __restrict__ unique_exp,
        const uint32_t * __restrict__ exp_tok_mask,
        const int32_t * __restrict__ exp_tok_slot,
        const float * __restrict__ exp_tok_w,
        const int32_t * __restrict__ exp_tok_cnt,
        const int32_t * __restrict__ exp_tok_list,
        const int32_t * __restrict__ d_num_unique, int m,
        float * __restrict__ buf_slot) {
    const int u = blockIdx.x;
    if (u >= *d_num_unique) return;
    const int cnt = exp_tok_cnt[u];
    if (cnt <= 0) return;
    const int ex = unique_exp[u];

    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane & 15;          // 16 lanes per row, each owns 3 weight chunks
    const int rloc = lane >> 4;       // 0..1 -> 2 rows per warp
    const int row = blockIdx.y * 16 + warp * 2 + rloc;
    if (row >= N_EMBD) return;

    constexpr int K  = N_FF_EXP;      // 1152
    constexpr int NB = K / 32;        // 36 x 32-weight chunks
    constexpr int QSR = K / 2;        // 576 B / row (NVFP4) or 1152 B (Q8)
    constexpr int SCR = K / 16;       // 72 scale bytes (NVFP4)

    if constexpr (NVFP4) {
        const float s2 = nv_s2[ex];
        const size_t rowb = (size_t)ex * N_EMBD + row;
        const uint8_t * q = nv_qs + rowb * QSR;
        const uint16_t * sc = reinterpret_cast<const uint16_t *>(nv_sc + rowb * SCR);
        const v4i w0 = ld_nt_b128(q + g * 16);
        const v4i w1 = ld_nt_b128(q + (16 + g) * 16);
        const uint32_t w2 = __builtin_nontemporal_load(reinterpret_cast<const uint32_t *>(q + 512) + g);
        const uint16_t sc0 = sc[g];
        const uint16_t sc1 = sc[16 + g];
        const uint8_t sc2 = (nv_sc + rowb * SCR)[64 + (g >> 1)];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= cnt) break;
            const int tok = exp_tok_list[u * 8 + t];
            const int slot = exp_tok_slot[u * 8 + tok];
            const float coef = exp_tok_w[u * 8 + tok];
            const float * ar = act + (size_t)(tok * TOP_K + slot) * N_FF_EXP;
            const float4 * ak = reinterpret_cast<const float4 *>(ar);
            const float4 a0 = ak[g * 8 + 0], a1 = ak[g * 8 + 1], a2 = ak[g * 8 + 2], a3 = ak[g * 8 + 3];
            const float4 a4 = ak[g * 8 + 4], a5 = ak[g * 8 + 5], a6 = ak[g * 8 + 6], a7 = ak[g * 8 + 7];
            float p = nvfp4_sub_dot((uint32_t)w0.x, (uint32_t)w0.y, a0, a1, a2, a3) *
                      ue4m3_to_fp32(sc0 & 0xFFu);
            p = fmaf(nvfp4_sub_dot((uint32_t)w0.z, (uint32_t)w0.w, a4, a5, a6, a7),
                     ue4m3_to_fp32(sc0 >> 8), p);
            const float4 b0 = ak[(16 + g) * 8 + 0], b1 = ak[(16 + g) * 8 + 1];
            const float4 b2 = ak[(16 + g) * 8 + 2], b3 = ak[(16 + g) * 8 + 3];
            const float4 b4 = ak[(16 + g) * 8 + 4], b5 = ak[(16 + g) * 8 + 5];
            const float4 b6 = ak[(16 + g) * 8 + 6], b7 = ak[(16 + g) * 8 + 7];
            p = fmaf(nvfp4_sub_dot((uint32_t)w1.x, (uint32_t)w1.y, b0, b1, b2, b3),
                     ue4m3_to_fp32(sc1 & 0xFFu), p);
            p = fmaf(nvfp4_sub_dot((uint32_t)w1.z, (uint32_t)w1.w, b4, b5, b6, b7),
                     ue4m3_to_fp32(sc1 >> 8), p);
            const int sb = g >> 1, wi = g & 1;
            const float4 c0 = ak[256 + sb * 4 + wi];
            const float4 c1 = ak[256 + sb * 4 + 2 + wi];
            uint32_t lo, hi;
            nvfp4_hbytes(w2, lo, hi);
            float p3 = 0.0f;
            p3 = nvfp4_hb_dot_mix(lo, c0, p3);
            p3 = nvfp4_hb_dot_mix(hi, c1, p3);
            p = fmaf(p3, ue4m3_to_fp32(sc2), p);
            buf_slot[(size_t)(tok * TOP_K + slot) * N_EMBD + row] =
                group_reduce_sum<16>(p) * (coef * s2);
        }
    } else {
        const size_t rowb = (size_t)ex * N_EMBD + row;
        const int8_t * q = q8_qs + rowb * K;
        const __half * dd = q8_d + rowb * NB;
        const v4i w0_0 = ld_nt_b128(q + g * 32);
        const v4i w0_1 = ld_nt_b128(q + g * 32 + 16);
        const __half h0 = dd[g];
        const v4i w1_0 = ld_nt_b128(q + (16 + g) * 32);
        const v4i w1_1 = ld_nt_b128(q + (16 + g) * 32 + 16);
        const __half h1 = dd[16 + g];
        const v2u w2 = ld_nt_b64(q + 1024 + g * 8);
        const __half h2 = dd[32 + (g >> 2)];
        #pragma unroll
        for (int t = 0; t < 8; ++t) {
            if (t >= cnt) break;
            const int tok = exp_tok_list[u * 8 + t];
            const int slot = exp_tok_slot[u * 8 + tok];
            const float coef = exp_tok_w[u * 8 + tok];
            const float * ar = act + (size_t)(tok * TOP_K + slot) * N_FF_EXP;
            const float4 * ak = reinterpret_cast<const float4 *>(ar);
            const float4 a0 = ak[g * 8 + 0], a1 = ak[g * 8 + 1], a2 = ak[g * 8 + 2], a3 = ak[g * 8 + 3];
            const float4 a4 = ak[g * 8 + 4], a5 = ak[g * 8 + 5], a6 = ak[g * 8 + 6], a7 = ak[g * 8 + 7];
            float p = 0.0f;
            p = q8_word_dot((uint32_t)w0_0.x, a0, p);
            p = q8_word_dot((uint32_t)w0_0.y, a1, p);
            p = q8_word_dot((uint32_t)w0_0.z, a2, p);
            p = q8_word_dot((uint32_t)w0_0.w, a3, p);
            p = q8_word_dot((uint32_t)w0_1.x, a4, p);
            p = q8_word_dot((uint32_t)w0_1.y, a5, p);
            p = q8_word_dot((uint32_t)w0_1.z, a6, p);
            p = q8_word_dot((uint32_t)w0_1.w, a7, p);
            float acc_t = p * __half2float(h0);
            float p1 = 0.0f;
            const float4 b0 = ak[(16 + g) * 8 + 0], b1 = ak[(16 + g) * 8 + 1];
            const float4 b2 = ak[(16 + g) * 8 + 2], b3 = ak[(16 + g) * 8 + 3];
            const float4 b4 = ak[(16 + g) * 8 + 4], b5 = ak[(16 + g) * 8 + 5];
            const float4 b6 = ak[(16 + g) * 8 + 6], b7 = ak[(16 + g) * 8 + 7];
            p1 = q8_word_dot((uint32_t)w1_0.x, b0, p1);
            p1 = q8_word_dot((uint32_t)w1_0.y, b1, p1);
            p1 = q8_word_dot((uint32_t)w1_0.z, b2, p1);
            p1 = q8_word_dot((uint32_t)w1_0.w, b3, p1);
            p1 = q8_word_dot((uint32_t)w1_1.x, b4, p1);
            p1 = q8_word_dot((uint32_t)w1_1.y, b5, p1);
            p1 = q8_word_dot((uint32_t)w1_1.z, b6, p1);
            p1 = q8_word_dot((uint32_t)w1_1.w, b7, p1);
            acc_t = fmaf(p1, __half2float(h1), acc_t);
            const float4 c0 = ak[256 + g * 2];
            const float4 c1 = ak[256 + g * 2 + 1];
            uint32_t w2x = w2[0], w2y = w2[1];
            float p2 = 0.0f;
            p2 = q8_word_dot(w2x, c0, p2);
            p2 = q8_word_dot(w2y, c1, p2);
            acc_t = fmaf(p2, __half2float(h2), acc_t);
            buf_slot[(size_t)(tok * TOP_K + slot) * N_EMBD + row] =
                group_reduce_sum<16>(acc_t) * coef;
        }
    }
}

} // namespace

void launch_verify_moe_dn_nvfp4(const SoaNVFP4Experts & down, const float * act,
                                const int32_t * unique_exp, const uint32_t * exp_tok_mask,
                                const int32_t * exp_tok_slot, const float * exp_tok_w,
                                const int32_t * exp_tok_cnt, const int32_t * exp_tok_list,
                                const int32_t * d_num_unique, int m,
                                float * buf_slot, hipStream_t stream) {
    if (m <= 0) return;
    const dim3 grid(VG_MOE_MAXU, N_EMBD / 16, 1);
    verify_moe_dn_kernel<true><<<grid, 256, 0, stream>>>(
        down.qs, down.scales, down.scale2, nullptr, nullptr, act,
        unique_exp, exp_tok_mask, exp_tok_slot, exp_tok_w, exp_tok_cnt, exp_tok_list,
        d_num_unique, m, buf_slot);
}

void launch_verify_moe_dn_q8(const SoaQ8Experts & down, const float * act,
                             const int32_t * unique_exp, const uint32_t * exp_tok_mask,
                             const int32_t * exp_tok_slot, const float * exp_tok_w,
                             const int32_t * exp_tok_cnt, const int32_t * exp_tok_list,
                             const int32_t * d_num_unique, int m,
                             float * buf_slot, hipStream_t stream) {
    if (m <= 0) return;
    const dim3 grid(VG_MOE_MAXU, N_EMBD / 16, 1);
    verify_moe_dn_kernel<false><<<grid, 256, 0, stream>>>(
        nullptr, nullptr, nullptr, down.qs, down.d, act,
        unique_exp, exp_tok_mask, exp_tok_slot, exp_tok_w, exp_tok_cnt, exp_tok_list,
        d_num_unique, m, buf_slot);
}


// Router-only (x_norm already computed with the exact decode mix_rms_norm
// reduction): identical body to ffn_norm_router_topk_decode_kernel's router half.
__device__ __forceinline__ float vg_dot_h8_f8(uint4 w, float4 x0, float4 x1, float acc) {
    const __half2 h0 = __builtin_bit_cast(__half2, w.x);
    const __half2 h1 = __builtin_bit_cast(__half2, w.y);
    const __half2 h2 = __builtin_bit_cast(__half2, w.z);
    const __half2 h3 = __builtin_bit_cast(__half2, w.w);
    acc = fmaf(__low2float(h0),  x0.x, acc);
    acc = fmaf(__high2float(h0), x0.y, acc);
    acc = fmaf(__low2float(h1),  x0.z, acc);
    acc = fmaf(__high2float(h1), x0.w, acc);
    acc = fmaf(__low2float(h2),  x1.x, acc);
    acc = fmaf(__high2float(h2), x1.y, acc);
    acc = fmaf(__low2float(h3),  x1.z, acc);
    acc = fmaf(__high2float(h3), x1.w, acc);
    return acc;
}

__device__ __forceinline__ void vg_wave32_topk8_128(const float * s_sel, const float * s_p,
                                                   int32_t * topk_ids_out, float * topk_w_out,
                                                   const int lane) {
    float my_sel[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) my_sel[j] = s_sel[lane + j * 32];
    float   top_p[TOP_K];
    int32_t top_i[TOP_K];
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        float   best_v = -INFINITY;
        int32_t best_i = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + j * 32;
            const float v = my_sel[j];
            if (v > best_v || (v == best_v && idx < best_i)) { best_v = v; best_i = idx; }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float   ov = __shfl_xor(best_v, off, 32);
            const int32_t oi = __shfl_xor(best_i, off, 32);
            if (ov > best_v || (ov == best_v && oi < best_i)) { best_v = ov; best_i = oi; }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (lane + j * 32 == best_i) my_sel[j] = -INFINITY;
        }
        if (lane == 0) { top_i[k] = best_i; top_p[k] = s_p[best_i]; }
    }
    if (lane == 0) {
        float sum_p = 0.0f;
#pragma unroll
        for (int k = 0; k < TOP_K; ++k) sum_p += top_p[k];
        const float scale = EXPERT_W_SCALE / (sum_p + EXPERT_NORM_EPS);
#pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            topk_ids_out[k] = top_i[k];
            topk_w_out[k]   = top_p[k] * scale;
        }
    }
}

// One CTA (512 threads, 16 warps x 8 experts) per token; reads the already
// normalised x_norm so the router probabilities match the fused decode kernel.
__global__ __launch_bounds__(512) void verify_router_topk_kernel(
        const float * __restrict__ x_norm, const __half * __restrict__ router_w,
        const float * __restrict__ exp_probs_b,
        int32_t * __restrict__ topk_ids_out, float * __restrict__ topk_w_out, int m) {
    constexpr int N8 = N_EMBD / 8;   // 288 uint4s per expert row
    const int t = blockIdx.x;
    if (t >= m) return;
    const int tid = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane = tid & 31;

    __shared__ float4 s_xnorm4[N_EMBD / 4];
    __shared__ float  s_sel[N_EXPERT];
    __shared__ float  s_p[N_EXPERT];
    const float4 * xt = reinterpret_cast<const float4 *>(x_norm + (size_t)t * N_EMBD);
#pragma unroll
    for (int p = 0; p < 2; ++p) {
        const int i = tid + p * 512;
        if (i < N_EMBD / 4) s_xnorm4[i] = xt[i];
    }
    __syncthreads();

    const int e_base = warp_id * 8;
    float acc[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const uint4 * rw8 = reinterpret_cast<const uint4 *>(router_w);
#pragma unroll 1
    for (int step = 0; step < 9; ++step) {
        const int i8 = lane + step * 32;
        const float4 x0 = s_xnorm4[i8 * 2 + 0];
        const float4 x1 = s_xnorm4[i8 * 2 + 1];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            const uint4 wv = rw8[(size_t)(e_base + k) * N8 + i8];
            acc[k] = vg_dot_h8_f8(wv, x0, x1, acc[k]);
        }
        __builtin_amdgcn_sched_barrier(0);
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float sum = group_reduce_sum<32>(acc[k]);
        if (lane == 0) {
            const int e = e_base + k;
            const float p = 1.0f / (1.0f + expf(-sum));
            s_p[e]   = p;
            s_sel[e] = exp_probs_b ? (p + exp_probs_b[e]) : p;
        }
    }
    __syncthreads();
    if (warp_id == 0) {
        vg_wave32_topk8_128(s_sel, s_p, topk_ids_out + (size_t)t * TOP_K,
                            topk_w_out + (size_t)t * TOP_K, lane);
    }
}

// ---------------------------------------------------------------------------
// 5. ffn_post: fixed-order 8-slot tree sum + causal 4-tap sconv + residual
// ---------------------------------------------------------------------------
namespace {
__global__ __launch_bounds__(32) void verify_ffn_post_kernel(
        const float * __restrict__ buf_slot, int m,
        const float * __restrict__ conv_w, float * __restrict__ hist,
        const float * __restrict__ res, float * __restrict__ out,
        float * __restrict__ raw_out, int raw_stride) {
    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;   // 0..575
    if (i4 >= N_EMBD / 4 || m <= 0) return;
    const int c_base = i4 * 4;
    float4 w[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) w[j] = reinterpret_cast<const float4 *>(conv_w)[c_base + j];
    float x0[4], x1[4], x2[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float * hs = hist + (size_t)(c_base + j) * N_CONV_HIST;
        x0[j] = hs[0];
        x1[j] = hs[1];
        x2[j] = hs[2];
    }
    constexpr float scale = 0.4472135954999579f;   // 1 / sqrt(5)
    for (int t = 0; t < m; ++t) {
        float x3[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const size_t off = (size_t)((t * TOP_K + 0) * N_EMBD) + c_base + j;
            const float s0 = buf_slot[off + (size_t)0 * N_EMBD];
            const float s1 = buf_slot[off + (size_t)1 * N_EMBD];
            const float s2 = buf_slot[off + (size_t)2 * N_EMBD];
            const float s3 = buf_slot[off + (size_t)3 * N_EMBD];
            const float s4 = buf_slot[off + (size_t)4 * N_EMBD];
            const float s5 = buf_slot[off + (size_t)5 * N_EMBD];
            const float s6 = buf_slot[off + (size_t)6 * N_EMBD];
            const float s7 = buf_slot[off + (size_t)7 * N_EMBD];
            x3[j] = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
        }
        if (raw_out != nullptr) {
#pragma unroll
            for (int j = 0; j < 4; ++j) raw_out[(size_t)t * raw_stride + c_base + j] = x3[j];
        }
        float y[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float acc = x3[j] + x0[j] * w[j].x + x1[j] * w[j].y + x2[j] * w[j].z + x3[j] * w[j].w;
            const float val = (res != nullptr) ? fmaf(acc, scale, res[(size_t)t * N_EMBD + c_base + j])
                                               : acc * scale;
            y[j] = val;
            x0[j] = x1[j];
            x1[j] = x2[j];
            x2[j] = x3[j];
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) out[(size_t)t * N_EMBD + c_base + j] = y[j];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        float * hs = hist + (size_t)(c_base + j) * N_CONV_HIST;
        hs[0] = x0[j];
        hs[1] = x1[j];
        hs[2] = x2[j];
    }
}

__global__ void verify_save_raw_kernel(const float * __restrict__ src, int m, int width,
                                       float * __restrict__ raw, int raw_stride) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int t = blockIdx.y;
    if (i >= width || t >= m) return;
    raw[(size_t)t * raw_stride + i] = src[(size_t)t * N_EMBD + i];
}
} // namespace

void launch_verify_router_topk(const float * x_norm, const __half * router_w,
                               const float * exp_probs_b, int32_t * topk_ids,
                               float * topk_w, int m, hipStream_t stream) {
    if (m <= 0) return;
    verify_router_topk_kernel<<<m, 512, 0, stream>>>(x_norm, router_w, exp_probs_b,
                                                     topk_ids, topk_w, m);
}

void launch_verify_ffn_post(const float * buf_slot, int m, const float * conv_w,
                            float * hist, const float * res, float * out,
                            float * raw_out, int raw_stride, hipStream_t stream) {
    if (m <= 0) return;
    verify_ffn_post_kernel<<<(N_EMBD / 4) / 32, 32, 0, stream>>>(
        buf_slot, m, conv_w, hist, res, out, raw_out, raw_stride);
}

void launch_verify_save_raw(const float * src, int m, int width, float * raw,
                            int raw_stride, hipStream_t stream) {
    if (m <= 0 || width <= 0) return;
    verify_save_raw_kernel<<<dim3((width + 255) / 256, m), 256, 0, stream>>>(
        src, m, width, raw, raw_stride);
}

namespace {
__global__ void verify_gather_tokens_kernel(const StepMeta * __restrict__ d_meta,
                                            int32_t * __restrict__ d_token_ids, int m) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= m) return;
    d_token_ids[r] = d_meta[r].token_id;
}
} // namespace

void launch_verify_gather_tokens(const StepMeta * d_meta, int32_t * d_token_ids, int m,
                                 hipStream_t stream) {
    if (m <= 0) return;
    verify_gather_tokens_kernel<<<(m + 31) / 32, 32, 0, stream>>>(d_meta, d_token_ids, m);
}

// ---------------------------------------------------------------------------
// 6. Multi-stream independent-slot Wo & FFN post-conv epilogues
// ---------------------------------------------------------------------------
namespace {
__global__ __launch_bounds__(192, 2) void batch_wo_mrows_kernel(
        const SoaQ8Matrix W, const float * __restrict__ x, int xstride,
        const float * __restrict__ conv_w, BatchHistPtrs hists,
        const float * __restrict__ res, float * __restrict__ y, int m) {
    constexpr int NB = N_Q / 32;   // 96 = 3 blocks per lane
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int row  = blockIdx.x * 6 + warp;
    if (row >= W.rows) return;

    const int8_t * wq = W.qs + (size_t)row * N_Q;
    const __half * wd = W.d + (size_t)row * NB;
    const int kb0 = lane, kb1 = lane + 32, kb2 = lane + 64;
    const v4i w0a = ld_nt_b128(wq + kb0 * 32), w0b = ld_nt_b128(wq + kb0 * 32 + 16);
    const v4i w1a = ld_nt_b128(wq + kb1 * 32), w1b = ld_nt_b128(wq + kb1 * 32 + 16);
    const v4i w2a = ld_nt_b128(wq + kb2 * 32), w2b = ld_nt_b128(wq + kb2 * 32 + 16);
    const float s0 = __half2float(wd[kb0]);
    const float s1 = __half2float(wd[kb1]);
    const float s2 = __half2float(wd[kb2]);

    const float4 cw = reinterpret_cast<const float4 *>(conv_w)[row];

#pragma unroll 1
    for (int t = 0; t < m; ++t) {
        const float4 * xr = reinterpret_cast<const float4 *>(x + (size_t)t * xstride);
        float a = 0.0f;
        {
            const float4 * xb = xr + kb0 * 8;
            float p = 0.0f, q = 0.0f;
            p = q8_word_dot((uint32_t)w0a.x, xb[0], p);
            q = q8_word_dot((uint32_t)w0b.x, xb[4], q);
            p = q8_word_dot((uint32_t)w0a.y, xb[1], p);
            q = q8_word_dot((uint32_t)w0b.y, xb[5], q);
            p = q8_word_dot((uint32_t)w0a.z, xb[2], p);
            q = q8_word_dot((uint32_t)w0b.z, xb[6], q);
            p = q8_word_dot((uint32_t)w0a.w, xb[3], p);
            q = q8_word_dot((uint32_t)w0b.w, xb[7], q);
            a = fmaf(p + q, s0, a);
        }
        {
            const float4 * yb = xr + kb1 * 8;
            float p = 0.0f, q = 0.0f;
            p = q8_word_dot((uint32_t)w1a.x, yb[0], p);
            q = q8_word_dot((uint32_t)w1b.x, yb[4], q);
            p = q8_word_dot((uint32_t)w1a.y, yb[1], p);
            q = q8_word_dot((uint32_t)w1b.y, yb[5], q);
            p = q8_word_dot((uint32_t)w1a.z, yb[2], p);
            q = q8_word_dot((uint32_t)w1b.z, yb[6], q);
            p = q8_word_dot((uint32_t)w1a.w, yb[3], p);
            q = q8_word_dot((uint32_t)w1b.w, yb[7], q);
            a = fmaf(p + q, s1, a);
        }
        {
            const float4 * zb = xr + kb2 * 8;
            float p = 0.0f, q = 0.0f;
            p = q8_word_dot((uint32_t)w2a.x, zb[0], p);
            q = q8_word_dot((uint32_t)w2b.x, zb[4], q);
            p = q8_word_dot((uint32_t)w2a.y, zb[1], p);
            q = q8_word_dot((uint32_t)w2b.y, zb[5], q);
            p = q8_word_dot((uint32_t)w2a.z, zb[2], p);
            q = q8_word_dot((uint32_t)w2b.z, zb[6], q);
            p = q8_word_dot((uint32_t)w2a.w, zb[3], p);
            q = q8_word_dot((uint32_t)w2b.w, zb[7], q);
            a = fmaf(p + q, s2, a);
        }
        a = group_reduce_sum<32>(a);
        if (lane == 0) {
            float * hs = hists.p[t] + (size_t)row * N_CONV_HIST;
            const float h0 = hs[0], h1 = hs[1], h2 = hs[2];
            const float x3 = a;
            const float ca = x3 + h0 * cw.x + h1 * cw.y + h2 * cw.z + x3 * cw.w;
            const float val = (res != nullptr) ? fmaf(ca, 0.4472135954999579f, res[(size_t)t * N_EMBD + row])
                                               : ca * 0.4472135954999579f;
            y[(size_t)t * W.rows + row] = val;
            hs[0] = h1; hs[1] = h2; hs[2] = x3;
        }
    }
}

__global__ __launch_bounds__(32) void batch_ffn_post_kernel(
        const float * __restrict__ buf_slot, int m,
        const float * __restrict__ conv_w, BatchHistPtrs hists,
        const float * __restrict__ res, float * __restrict__ out) {
    const int i4 = blockIdx.x * blockDim.x + threadIdx.x;   // 0..575
    const int t  = blockIdx.y;                              // 0..m-1
    if (i4 >= N_EMBD / 4 || t >= m) return;
    const int c_base = i4 * 4;
    float * hist = hists.p[t];
    constexpr float scale = 0.4472135954999579f;   // 1 / sqrt(5)
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float4 w = reinterpret_cast<const float4 *>(conv_w)[c_base + j];
        const size_t off = (size_t)((t * TOP_K + 0) * N_EMBD) + c_base + j;
        const float s0 = buf_slot[off + (size_t)0 * N_EMBD];
        const float s1 = buf_slot[off + (size_t)1 * N_EMBD];
        const float s2 = buf_slot[off + (size_t)2 * N_EMBD];
        const float s3 = buf_slot[off + (size_t)3 * N_EMBD];
        const float s4 = buf_slot[off + (size_t)4 * N_EMBD];
        const float s5 = buf_slot[off + (size_t)5 * N_EMBD];
        const float s6 = buf_slot[off + (size_t)6 * N_EMBD];
        const float s7 = buf_slot[off + (size_t)7 * N_EMBD];
        const float x3 = ((s0 + s1) + (s2 + s3)) + ((s4 + s5) + (s6 + s7));
        float * hs = hist + (size_t)(c_base + j) * N_CONV_HIST;
        const float h0 = hs[0], h1 = hs[1], h2 = hs[2];
        const float acc = x3 + h0 * w.x + h1 * w.y + h2 * w.z + x3 * w.w;
        const float val = (res != nullptr) ? fmaf(acc, scale, res[(size_t)t * N_EMBD + c_base + j])
                                           : acc * scale;
        out[(size_t)t * N_EMBD + c_base + j] = val;
        hs[0] = h1;
        hs[1] = h2;
        hs[2] = x3;
    }
}
} // namespace

void launch_batch_wo_mrows(const SoaQ8Matrix & W, const float * x, int xstride,
                           const float * conv_w, BatchHistPtrs hists, const float * res,
                           float * y, int m, hipStream_t stream) {
    if (W.rows <= 0 || m <= 0) return;
    if (W.K != N_Q) {
        for (int t = 0; t < m; ++t) {
            launch_soa_q8_gemv_sconv_post_res(
                W, x + (size_t)t * xstride, conv_w, hists.p[t],
                (res != nullptr) ? (res + (size_t)t * N_EMBD) : nullptr,
                y + (size_t)t * N_EMBD, stream);
        }
        return;
    }
    const int grid = (W.rows + 5) / 6;
    batch_wo_mrows_kernel<<<grid, 192, 0, stream>>>(W, x, xstride, conv_w, hists, res, y, m);
}

void launch_batch_ffn_post(const float * buf_slot, int m, const float * conv_w,
                           BatchHistPtrs hists, const float * res, float * out,
                           hipStream_t stream) {
    if (m <= 0) return;
    const dim3 grid((N_EMBD / 4) / 32, (unsigned)m);
    batch_ffn_post_kernel<<<grid, 32, 0, stream>>>(buf_slot, m, conv_w, hists, res, out);
}

} // namespace inkling
