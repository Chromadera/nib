// ============================================================================
// kernels_wmma_fp8_iq_rdna4.hip — RDNA4 (gfx1200 / gfx1201) FP8 WMMA grouped
// MoE prefill for all 9 GSQ-RCO IQ / Q2_0 formats
// ============================================================================
// Zero modifications to kernels_wmma_fp8_rdna4.hip or any existing .hip file.
// Uses the exact same MoePrefillScratch layout, K-interleaving (fp8_moe_kpos),
// X_norm -> E4M3 row-scaling (launch_x_to_fp8_rowscale), SITU -> E4M3 static
// scale (FP8_MOE_ACT_SCALE = 4.0f), and v_wmma_f32_16x16x16_fp8_fp8 mainloop
// as the NVFP4 RDNA4 FP8 path, decoding IQ / Q2_0 weights on-the-fly into
// per-expert E4M3 (scaled by 448.0f / exp_amax[e]).
// ============================================================================

#include "inkling_engine.h"
#include "kernels_iq.h"
#include "kernels_wmma_fp8.h"
#include "nib_arch_fp8.h"
#include "iq_dequant_device.h"

#include <cfloat>
#include <cstdio>
#include <cstdlib>

// ============================================================================
// Arch-neutral kernels (every target) so gfx11 code objects stay non-empty
// for scripts/spill_audit.sh when building multi-arch binaries.
// ============================================================================
namespace inkling {
namespace iq_fp8_moe {

constexpr int ROUTE_THREADS = 1024;
constexpr int MASK_WORDS    = FP8_MAX_UBATCH / 32; // 32 words (1024 tokens)

__global__ void __launch_bounds__(ROUTE_THREADS)
route_kernel(const int32_t * __restrict__ topk_ids, const float * __restrict__ topk_w, int M, int TM,
             int32_t * __restrict__ exp_counts, int32_t * __restrict__ exp_offsets,
             int32_t * __restrict__ exp_cursors, int32_t * __restrict__ perm_tok,
             float * __restrict__ perm_w, int32_t * __restrict__ tile_exp,
             int32_t * __restrict__ tile_m0, int32_t * __restrict__ d_num_tiles,
             int32_t * __restrict__ tok_slots) {
    __shared__ uint32_t s_mask[N_EXPERT][MASK_WORDS];
    __shared__ int32_t  s_pref[N_EXPERT][MASK_WORDS];
    __shared__ int32_t  s_cnt[N_EXPERT];
    __shared__ int32_t  s_off[N_EXPERT + 1];
    __shared__ int32_t  s_toff[N_EXPERT + 1];

    const int tid   = threadIdx.x;
    const int npair = M * TOP_K;
    const int nw    = (M + 31) >> 5;

    for (int i = tid; i < N_EXPERT * MASK_WORDS; i += ROUTE_THREADS) {
        (&s_mask[0][0])[i] = 0u;
    }
    __syncthreads();

    for (int p = tid; p < npair; p += ROUTE_THREADS) {
        const int e = topk_ids[p];
        if ((unsigned) e < (unsigned) N_EXPERT) {
            const int tok = p / TOP_K;
            atomicOr(&s_mask[e][tok >> 5], 1u << (tok & 31));
        }
    }
    __syncthreads();

    if (tid < N_EXPERT) {
        int run = 0;
        for (int w = 0; w < nw; ++w) {
            s_pref[tid][w] = run;
            run += __popc(s_mask[tid][w]);
        }
        s_cnt[tid] = run;
    }
    __syncthreads();

    if (tid == 0) {
        int off = 0, toff = 0;
        for (int e = 0; e < N_EXPERT; ++e) {
            s_off[e]  = off;
            s_toff[e] = toff;
            off  += s_cnt[e];
            toff += (s_cnt[e] + TM - 1) / TM;
        }
        s_off[N_EXPERT]  = off;
        s_toff[N_EXPERT] = toff;
        *d_num_tiles = toff;
    }
    __syncthreads();

    if (tid <= N_EXPERT) {
        exp_offsets[tid] = s_off[tid];
    }
    if (tid < N_EXPERT) {
        exp_counts[tid]  = s_cnt[tid];
        exp_cursors[tid] = s_cnt[tid];
        const int nt = (s_cnt[tid] + TM - 1) / TM;
        for (int j = 0; j < nt; ++j) {
            tile_exp[s_toff[tid] + j] = tid;
            tile_m0[s_toff[tid] + j]  = j * TM;
        }
    }

    for (int p = tid; p < npair; p += ROUTE_THREADS) {
        const int e = topk_ids[p];
        if ((unsigned) e >= (unsigned) N_EXPERT) {
            tok_slots[p] = -1;
            continue;
        }
        const int tok = p / TOP_K;
        const int k   = p - tok * TOP_K;
        bool first = true;
        float w = 0.0f;
        for (int kk = 0; kk < TOP_K; ++kk) {
            if (topk_ids[tok * TOP_K + kk] == e) {
                if (kk < k) {
                    first = false;
                }
                w += topk_w[tok * TOP_K + kk];
            }
        }
        if (!first) {
            tok_slots[p] = -1;
            continue;
        }
        const int wi   = tok >> 5;
        const int rank = s_pref[e][wi] + __popc(s_mask[e][wi] & ((1u << (tok & 31)) - 1u));
        const int slot = s_off[e] + rank;
        perm_tok[slot] = tok;
        perm_w[slot]   = w;
        tok_slots[p]   = slot;
    }
}

constexpr int COMBINE_THREADS = 192;

__global__ void __launch_bounds__(COMBINE_THREADS)
combine_kernel(const int32_t * __restrict__ tok_slots, const float * __restrict__ perm_down,
               float * __restrict__ Y) {
    const int tok = blockIdx.x;
    const int32_t * ts = tok_slots + tok * TOP_K;
    int s[TOP_K];
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        s[k] = ts[k];
    }
    float4 * y4 = reinterpret_cast<float4 *>(Y + (size_t) tok * N_EMBD);
    for (int c = threadIdx.x; c < N_EMBD / 4; c += COMBINE_THREADS) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            if (s[k] >= 0) {
                const float4 v = reinterpret_cast<const float4 *>(perm_down + (size_t) s[k] * N_EMBD)[c];
                acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
            }
        }
        y4[c] = acc;
    }
}

constexpr int COMBINE_SCONV_THREADS = 64;
constexpr int COMBINE_SCONV_TCHUNK  = 8;

__device__ __forceinline__ float4 load_tok_sum_c4(const int32_t * __restrict__ tok_slots,
                                                  const float * __restrict__ perm_down, int tok, int c4) {
    const int4 ts0 = reinterpret_cast<const int4 *>(tok_slots + tok * TOP_K)[0];
    const int4 ts1 = reinterpret_cast<const int4 *>(tok_slots + tok * TOP_K)[1];
    const int s[TOP_K] = {ts0.x, ts0.y, ts0.z, ts0.w, ts1.x, ts1.y, ts1.z, ts1.w};
    const float4 * p4 = reinterpret_cast<const float4 *>(perm_down);
    float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        if (s[k] >= 0) {
            const float4 v = p4[(size_t) s[k] * (N_EMBD / 4) + c4];
            x.x += v.x; x.y += v.y; x.z += v.z; x.w += v.w;
        }
    }
    return x;
}

__device__ __forceinline__ float sconv_step_one(float x, float & h0, float & h1, float & h2,
                                                float4 w, bool has_res, float res_val) {
    constexpr float kScale = 0.44721359549995793928f; // 1 / sqrt(5)
    const float acc = x + h0 * w.x + h1 * w.y + h2 * w.z + x * w.w;
    h0 = h1;
    h1 = h2;
    h2 = x;
    return has_res ? fmaf(acc, kScale, res_val) : (acc * kScale);
}

__global__ void __launch_bounds__(COMBINE_SCONV_THREADS)
combine_sconv_kernel(const int32_t * __restrict__ tok_slots, const float * __restrict__ perm_down,
                     const float * __restrict__ conv_w, float * __restrict__ sconv_hist,
                     const float * __restrict__ res_add, float * __restrict__ Y, int M) {
    const int c4    = blockIdx.x * COMBINE_SCONV_THREADS + threadIdx.x;
    const int c0    = c4 * 4;
    const int t_blk = blockIdx.y;
    const int t0    = t_blk * COMBINE_SCONV_TCHUNK;
    if (t0 >= M) {
        return;
    }
    const int t1 = (t0 + COMBINE_SCONV_TCHUNK < M) ? (t0 + COMBINE_SCONV_TCHUNK) : M;

    const float4 * cw4 = reinterpret_cast<const float4 *>(conv_w);
    const float4 w0 = cw4[c0 + 0];
    const float4 w1 = cw4[c0 + 1];
    const float4 w2 = cw4[c0 + 2];
    const float4 w3 = cw4[c0 + 3];

    float * h_base = sconv_hist + c0 * 3;
    float4 h0, h1, h2;
    if (t_blk == 0) {
        h0 = make_float4(h_base[0], h_base[3], h_base[6], h_base[9]);
        h1 = make_float4(h_base[1], h_base[4], h_base[7], h_base[10]);
        h2 = make_float4(h_base[2], h_base[5], h_base[8], h_base[11]);
    } else {
        h0 = load_tok_sum_c4(tok_slots, perm_down, t0 - 3, c4);
        h1 = load_tok_sum_c4(tok_slots, perm_down, t0 - 2, c4);
        h2 = load_tok_sum_c4(tok_slots, perm_down, t0 - 1, c4);
    }

    const bool has_res = (res_add != nullptr);
    const float4 * r4_ptr = reinterpret_cast<const float4 *>(res_add);
    float4 * y4_ptr       = reinterpret_cast<float4 *>(Y);

    for (int t = t0; t < t1; ++t) {
        const float4 x = load_tok_sum_c4(tok_slots, perm_down, t, c4);
        const float4 r = has_res ? r4_ptr[(size_t) t * (N_EMBD / 4) + c4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 out;
        out.x = sconv_step_one(x.x, h0.x, h1.x, h2.x, w0, has_res, r.x);
        out.y = sconv_step_one(x.y, h0.y, h1.y, h2.y, w1, has_res, r.y);
        out.z = sconv_step_one(x.z, h0.z, h1.z, h2.z, w2, has_res, r.z);
        out.w = sconv_step_one(x.w, h0.w, h1.w, h2.w, w3, has_res, r.w);
        y4_ptr[(size_t) t * (N_EMBD / 4) + c4] = out;
    }

    if (t_blk == 0) {
        if (t1 < M) {
            h0 = load_tok_sum_c4(tok_slots, perm_down, M - 3, c4);
            h1 = load_tok_sum_c4(tok_slots, perm_down, M - 2, c4);
            h2 = load_tok_sum_c4(tok_slots, perm_down, M - 1, c4);
        }
        h_base[0]  = h0.x; h_base[1]  = h1.x; h_base[2]  = h2.x;
        h_base[3]  = h0.y; h_base[4]  = h1.y; h_base[5]  = h2.y;
        h_base[6]  = h0.z; h_base[7]  = h1.z; h_base[8]  = h2.z;
        h_base[9]  = h0.w; h_base[10] = h1.w; h_base[11] = h2.w;
    }
}

} // namespace iq_fp8_moe
} // namespace inkling

// ============================================================================
// FP8 kernels and host launcher (host pass + gfx12 device passes only)
// ============================================================================
#if NIB_FP8_CODEGEN

namespace inkling {
namespace {

using namespace iq_fp8_moe;

typedef int v4i __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float situ_scalar(float gate, float up) {
    const float act = SITU_BETA * tanhf(gate * (1.0f / SITU_BETA)) / (1.0f + expf(-gate));
    const float u   = SITU_LINEAR_BETA * tanhf(up * (1.0f / SITU_LINEAR_BETA));
    return act * u;
}

__device__ __forceinline__ nib_fp8x8_t lo8(const v4i & v) { return nib_fp8x8_t{v.x, v.y}; }
__device__ __forceinline__ nib_fp8x8_t hi8(const v4i & v) { return nib_fp8x8_t{v.z, v.w}; }

__device__ __forceinline__ uint2 f4x2_to_fp8x8(float4 w0, float4 w1, float inv_wscale) {
    const uint32_t lo = nib_f32x4_to_fp8x4(
        nib_fp8_clamp(w0.x * inv_wscale),
        nib_fp8_clamp(w0.y * inv_wscale),
        nib_fp8_clamp(w0.z * inv_wscale),
        nib_fp8_clamp(w0.w * inv_wscale));
    const uint32_t hi = nib_f32x4_to_fp8x4(
        nib_fp8_clamp(w1.x * inv_wscale),
        nib_fp8_clamp(w1.y * inv_wscale),
        nib_fp8_clamp(w1.z * inv_wscale),
        nib_fp8_clamp(w1.w * inv_wscale));
    return make_uint2(lo, hi);
}

// Grouped MoE FP8 WMMA tile mainloop for IQ / Q2_0 weights.
// BK8 = 64, BN = 128, NT = 256 (WARPS_M = 4, WARPS_N = 2):
//   - 2 adjacent threads per B row (brow = tid >> 1, bchk = tid & 1).
//   - Each B thread dequantizes 32 weights (4 x 8) at k_base = k0 + bchk * 32,
//     converts to E4M3 with per-expert inv_wscale, and writes two K-interleaved
//     16-byte units into sm.b matching fp8_moe_kpos.
template <MoEQuantType Q, int BM, int BN, int WARPS_M, int WARPS_N, bool PAIRED>
struct TileGemmMoeIqF8 {
    static constexpr int BK8   = 64;
    static constexpr int NT    = WARPS_M * WARPS_N * 32;
    static constexpr int WTM   = BM / WARPS_M;
    static constexpr int FM    = WTM / 16;
    static constexpr int OUTN  = PAIRED ? BN / 2 : BN;
    static constexpr int WTN   = OUTN / WARPS_N;
    static constexpr int FNO   = WTN / 16;
    static constexpr int FN    = PAIRED ? 2 * FNO : FNO;
    static constexpr int UPR   = BK8 / 16;             // 4
    static constexpr int PAIRS = BK8 / 32;             // 2
    static constexpr int RPL   = 8 / UPR;              // 2
    static constexpr int A_N   = BM * UPR;
    static constexpr int A_IT  = (A_N + NT - 1) / NT;
    static constexpr bool A_EXACT = (A_N % NT) == 0;
    static_assert(NT == 2 * BN, "two lanes per weight row");
    static_assert(BM % (16 * WARPS_M) == 0 && OUTN % (16 * WARPS_N) == 0, "warp tiling");

    struct Smem {
        v4i a[BM * UPR];
        v4i b[BN * UPR];
    };

    const uint8_t * a_ptr[A_IT];
    const uint8_t * s_p0 = nullptr;
    const uint8_t * s_p1 = nullptr;
    const __half  * s_d  = nullptr;
    v4i             ar[A_IT];

    __device__ __forceinline__ static int swz(int row) { return (row / RPL) & (UPR - 1); }
    __device__ __forceinline__ static int unit_idx(int row, int u) { return row * UPR + (u ^ swz(row)); }

    __device__ __forceinline__ static bool a_live(int p) { return A_EXACT || threadIdx.x + p * NT < A_N; }
    __device__ __forceinline__ static int  a_row(int p)  { return a_live(p) ? (threadIdx.x + p * NT) / UPR : 0; }
    __device__ __forceinline__ static int  a_unit(int p) { return (threadIdx.x + p * NT) % UPR; }

    __device__ __forceinline__ void loadA(int k0, int K) {
#pragma unroll
        for (int p = 0; p < A_IT; ++p) {
            const int kk = k0 + a_unit(p) * 16;
            ar[p] = (a_ptr[p] != nullptr && kk < K) ? *reinterpret_cast<const v4i *>(a_ptr[p] + kk)
                                                    : v4i{0, 0, 0, 0};
        }
    }

    __device__ __forceinline__ void storeA(Smem & sm) {
#pragma unroll
        for (int p = 0; p < A_IT; ++p) {
            if (a_live(p)) {
                sm.a[unit_idx(a_row(p), a_unit(p))] = ar[p];
            }
        }
    }

    // Decode 32 weights [k_base .. k_base + 31] into two K-interleaved E4M3 LDS units:
    //   unit 2*bchk + 0 = [K 0..7  | K 16..23]
    //   unit 2*bchk + 1 = [K 8..15 | K 24..31]
    __device__ __forceinline__ void decodeStoreB(Smem & sm, int brow, int bchk, int k0, float inv_wscale) {
        const int kb = k0 + bchk * 32;
        float4 w0, w1;
        iq_dev::dequant_row_k8<Q>(s_p0, s_p1, s_d, kb + 0, w0, w1);
        const uint2 e0 = f4x2_to_fp8x8(w0, w1, inv_wscale);
        iq_dev::dequant_row_k8<Q>(s_p0, s_p1, s_d, kb + 8, w0, w1);
        const uint2 e1 = f4x2_to_fp8x8(w0, w1, inv_wscale);
        iq_dev::dequant_row_k8<Q>(s_p0, s_p1, s_d, kb + 16, w0, w1);
        const uint2 e2 = f4x2_to_fp8x8(w0, w1, inv_wscale);
        iq_dev::dequant_row_k8<Q>(s_p0, s_p1, s_d, kb + 24, w0, w1);
        const uint2 e3 = f4x2_to_fp8x8(w0, w1, inv_wscale);
        sm.b[unit_idx(brow, 2 * bchk + 0)] = v4i{(int) e0.x, (int) e0.y, (int) e2.x, (int) e2.y};
        sm.b[unit_idx(brow, 2 * bchk + 1)] = v4i{(int) e1.x, (int) e1.y, (int) e3.x, (int) e3.y};
    }

    __device__ __forceinline__ static int b_frag_row(int wn, int f) {
        if (PAIRED) {
            return (f < FNO) ? (wn * WTN + f * 16) : (OUTN + wn * WTN + (f - FNO) * 16);
        }
        return wn * WTN + f * 16;
    }

    __device__ __forceinline__ void compute(const Smem & sm, nib_float8_t (&acc)[FM][FN], int wm, int wn,
                                            int l16, int half_w, int m_valid) {
        if (wm * WTM >= m_valid) {
            return;
        }
#pragma unroll
        for (int p = 0; p < PAIRS; ++p) {
            __builtin_amdgcn_sched_barrier(0);
            const int u = 2 * p + half_w;
            v4i af[FM], bf[FN];
#pragma unroll
            for (int i = 0; i < FM; ++i) {
                af[i] = sm.a[unit_idx(wm * WTM + i * 16 + l16, u)];
            }
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                bf[j] = sm.b[unit_idx(b_frag_row(wn, j) + l16, u)];
            }
#pragma unroll
            for (int j = 0; j < FN; ++j) {
#pragma unroll
                for (int i = 0; i < FM; ++i) {
                    acc[i][j] = nib_wmma_f32_16x16x16_fp8(lo8(af[i]), lo8(bf[j]), acc[i][j]);
                }
            }
#pragma unroll
            for (int j = 0; j < FN; ++j) {
#pragma unroll
                for (int i = 0; i < FM; ++i) {
                    acc[i][j] = nib_wmma_f32_16x16x16_fp8(hi8(af[i]), hi8(bf[j]), acc[i][j]);
                }
            }
        }
    }

    __device__ __forceinline__ void run(Smem & sm, int K, float inv_wscale,
                                        nib_float8_t (&acc)[FM][FN], int m_valid) {
        const int warp   = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
        const int lane   = threadIdx.x & 31;
        const int wm     = warp / WARPS_N;
        const int wn     = warp % WARPS_N;
        const int l16    = lane & 15;
        const int half_w = lane >> 4;
        const int brow   = threadIdx.x >> 1;
        const int bchk   = threadIdx.x & 1;
#pragma unroll
        for (int i = 0; i < FM; ++i) {
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                acc[i][j] = nib_float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
        const int nk = K / BK8;
        loadA(0, K);
#pragma unroll 1
        for (int kt = 0; kt < nk; ++kt) {
            const int k0 = kt * BK8;
            __builtin_amdgcn_sched_barrier(0);
            storeA(sm);
            decodeStoreB(sm, brow, bchk, k0, inv_wscale);
            __syncthreads();
            if (kt + 1 < nk) {
                loadA(k0 + BK8, K);
            }
            compute(sm, acc, wm, wn, l16, half_w, m_valid);
            __syncthreads();
        }
    }

    __device__ __forceinline__ static int out_row(int i, int v) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        return (warp / WARPS_N) * WTM + i * 16 + nib_wmma_c_row(lane >> 4, v);
    }
    __device__ __forceinline__ static int out_col(int j) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        return (warp % WARPS_N) * WTN + j * 16 + (lane & 15);
    }
};

constexpr int TM_SMALL = 64;
constexpr int TM_LARGE = 128;
constexpr int GU_OUT   = 64;
constexpr int DN_BN    = 128;
constexpr int MOE_WM   = 4;
constexpr int MOE_WN   = 2;
constexpr int MOE_NT   = MOE_WM * MOE_WN * 32;

struct IqArgs {
    const uint8_t * p0;
    const uint8_t * p1;
    const __half  * d;
    const float   * exp_amax;
};

template <MoEQuantType Q, int TM_VAL>
__global__ void __launch_bounds__(MOE_NT, 2)
moe_iq_gate_up_fp8_wmma_kernel(const IqArgs W, const uint8_t * __restrict__ X8, const float * __restrict__ xs,
                               const int32_t * __restrict__ exp_offsets, const int32_t * __restrict__ perm_tok,
                               const int32_t * __restrict__ tile_exp, const int32_t * __restrict__ tile_m0,
                               const int32_t * __restrict__ d_num_tiles, uint8_t * __restrict__ act8) {
    using G = TileGemmMoeIqF8<Q, TM_VAL, 2 * GU_OUT, MOE_WM, MOE_WN, true>;
    using T = iq_dev::IqTraits<Q>;
    __shared__ __align__(16) typename G::Smem sm;
    __shared__ float s_xs[TM_VAL];

    const int tile = blockIdx.y;
    if (tile >= *d_num_tiles) {
        return;
    }
    constexpr int K      = N_EMBD;
    constexpr int NB_ROW = K / T::QK;
    static_assert(K % T::QK == 0, "gate_up K must be divisible by QK");

    const int e     = tile_exp[tile];
    const int slot0 = exp_offsets[e] + tile_m0[tile];
    const int mrem  = exp_offsets[e + 1] - slot0;
    const int mcnt  = mrem < TM_VAL ? mrem : TM_VAL;
    const int n0    = blockIdx.x * GU_OUT;

    const float amax       = W.exp_amax[e];
    const float inv_wscale = (amax > NIB_FP8_AMAX_MIN) ? (NIB_FP8_MAX / amax) : 0.0f;
    const float wscale     = (amax > NIB_FP8_AMAX_MIN) ? (amax * (1.0f / NIB_FP8_MAX)) : 0.0f;

    const int tid = threadIdx.x;
    if (tid < TM_VAL) {
        s_xs[tid] = (tid < mcnt) ? xs[perm_tok[slot0 + tid]] : 0.0f;
    }

    G g;
#pragma unroll
    for (int p = 0; p < G::A_IT; ++p) {
        const int i = G::a_row(p);
        g.a_ptr[p] = (G::a_live(p) && i < mcnt) ? X8 + (size_t) perm_tok[slot0 + i] * K : nullptr;
    }
    const int brow = threadIdx.x >> 1;
    const int row  = (brow < GU_OUT) ? (n0 + brow) : (N_FF_EXP + n0 + (brow - GU_OUT));
    const size_t blk_row = ((size_t) e * N_GATE_UP_EXP + row) * NB_ROW;
    g.s_p0 = W.p0 + blk_row * T::P0B;
    g.s_p1 = (T::P1B > 0) ? (W.p1 + blk_row * T::P1B) : nullptr;
    g.s_d  = T::HAS_D ? (W.d + blk_row) : nullptr;

    nib_float8_t acc[G::FM][G::FN];
    g.run(sm, K, inv_wscale, acc, mcnt);

#pragma unroll
    for (int i = 0; i < G::FM; ++i) {
#pragma unroll
        for (int j = 0; j < G::FNO; ++j) {
            const int col = fp8_moe_kpos(n0 + G::out_col(j));
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int m = G::out_row(i, v);
                if (m < mcnt) {
                    const float sx = s_xs[m] * wscale;
                    const float a  = situ_scalar(acc[i][j][v] * sx, acc[i][j + G::FNO][v] * sx);
                    const uint32_t b = nib_f32x2_to_fp8x2<false>(a * FP8_MOE_ACT_SCALE, 0.0f, 0u);
                    act8[(size_t) (slot0 + m) * N_FF_EXP + col] = (uint8_t) (b & 0xFFu);
                }
            }
        }
    }
}

template <MoEQuantType Q, int TM_VAL>
__global__ void __launch_bounds__(MOE_NT, 2)
moe_iq_down_fp8_wmma_kernel(const IqArgs W, const uint8_t * __restrict__ act8,
                            const int32_t * __restrict__ exp_offsets, const float * __restrict__ perm_w,
                            const int32_t * __restrict__ tile_exp, const int32_t * __restrict__ tile_m0,
                            const int32_t * __restrict__ d_num_tiles, float * __restrict__ perm_down) {
    using G = TileGemmMoeIqF8<Q, TM_VAL, DN_BN, MOE_WM, MOE_WN, false>;
    using T = iq_dev::IqTraits<Q>;
    __shared__ __align__(16) typename G::Smem sm;

    const int tile = blockIdx.y;
    if (tile >= *d_num_tiles) {
        return;
    }
    constexpr int K      = N_FF_EXP;
    constexpr int NB_ROW = K / T::QK;
    static_assert(K % T::QK == 0, "down K must be divisible by QK");

    const int e     = tile_exp[tile];
    const int slot0 = exp_offsets[e] + tile_m0[tile];
    const int mrem  = exp_offsets[e + 1] - slot0;
    const int mcnt  = mrem < TM_VAL ? mrem : TM_VAL;
    const int n0    = blockIdx.x * DN_BN;

    const float amax       = W.exp_amax[e];
    const float inv_wscale = (amax > NIB_FP8_AMAX_MIN) ? (NIB_FP8_MAX / amax) : 0.0f;
    const float wscale     = (amax > NIB_FP8_AMAX_MIN) ? (amax * (1.0f / NIB_FP8_MAX)) : 0.0f;

    G g;
#pragma unroll
    for (int p = 0; p < G::A_IT; ++p) {
        const int i = G::a_row(p);
        g.a_ptr[p] = (G::a_live(p) && i < mcnt) ? act8 + (size_t) (slot0 + i) * K : nullptr;
    }
    const int brow = threadIdx.x >> 1;
    const size_t blk_row = ((size_t) e * N_EMBD + n0 + brow) * NB_ROW;
    g.s_p0 = W.p0 + blk_row * T::P0B;
    g.s_p1 = (T::P1B > 0) ? (W.p1 + blk_row * T::P1B) : nullptr;
    g.s_d  = T::HAS_D ? (W.d + blk_row) : nullptr;

    nib_float8_t acc[G::FM][G::FN];
    g.run(sm, K, inv_wscale, acc, mcnt);

    const float s2 = wscale * (1.0f / FP8_MOE_ACT_SCALE);
#pragma unroll
    for (int i = 0; i < G::FM; ++i) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int m = G::out_row(i, v);
            if (m < mcnt) {
                const float wsc = s2 * perm_w[slot0 + m];
                float * dst = perm_down + (size_t) (slot0 + m) * N_EMBD + n0;
#pragma unroll
                for (int j = 0; j < G::FN; ++j) {
                    dst[G::out_col(j)] = acc[i][j][v] * wsc;
                }
            }
        }
    }
}

[[noreturn]] void fp8_iq_fail(const char * what) {
    std::fprintf(stderr, "[inkling kernels_wmma_fp8_iq_rdna4] contract violation: %s\n", what);
    std::abort();
}

void require_rdna4_iq(const char * who) {
    if (!rdna4_fp8_prefill_supported()) {
        std::fprintf(stderr, "[inkling kernels_wmma_fp8_iq_rdna4] %s requires an RDNA4 (gfx12) GPU, got %s\n",
                     who, nib_device_info().gcn_arch);
        std::abort();
    }
}

constexpr int COMBINE_SCONV_BLOCKS = (N_EMBD / 4) / COMBINE_SCONV_THREADS;

inline int max_tiles(int M, int tm) {
    const int P = M * TOP_K;
    return P / tm + (P < N_EXPERT ? P : N_EXPERT);
}

struct Fp8Scratch {
    uint8_t * x8;
    float   * xs;
    int32_t * tok_slots;
    uint8_t * act8;
};

Fp8Scratch carve(const MoePrefillScratch & s, int cap) {
    uint8_t * base = reinterpret_cast<uint8_t *>(s.perm_x);
    const size_t x8_bytes = (size_t) cap * N_EMBD;
    const size_t xs_bytes = ((size_t) cap * sizeof(float) + 15u) & ~size_t(15);
    Fp8Scratch f;
    f.x8        = base;
    f.xs        = reinterpret_cast<float *>(base + x8_bytes);
    f.tok_slots = reinterpret_cast<int32_t *>(base + x8_bytes + xs_bytes);
    f.act8      = reinterpret_cast<uint8_t *>(s.perm_act);
    return f;
}

template <int TM_VAL>
void dispatch_gu_fp8(MoEQuantType qt, const IqArgs & gu, const Fp8Scratch & f,
                     const MoePrefillScratch & s, int tiles, hipStream_t stream) {
#define LAUNCH_GU_F8(Q) \
    moe_iq_gate_up_fp8_wmma_kernel<Q, TM_VAL><<<dim3(N_FF_EXP / GU_OUT, tiles), MOE_NT, 0, stream>>>( \
        gu, f.x8, f.xs, s.exp_offsets, s.perm_tok, s.tile_exp, s.tile_m0, s.d_num_tiles, f.act8)
    switch (qt) {
        case MOE_Q_IQ1_S:   LAUNCH_GU_F8(MOE_Q_IQ1_S);   break;
        case MOE_Q_IQ1_M:   LAUNCH_GU_F8(MOE_Q_IQ1_M);   break;
        case MOE_Q_IQ2_XXS: LAUNCH_GU_F8(MOE_Q_IQ2_XXS); break;
        case MOE_Q_IQ2_XS:  LAUNCH_GU_F8(MOE_Q_IQ2_XS);  break;
        case MOE_Q_IQ2_S:   LAUNCH_GU_F8(MOE_Q_IQ2_S);   break;
        case MOE_Q_IQ3_XXS: LAUNCH_GU_F8(MOE_Q_IQ3_XXS); break;
        case MOE_Q_IQ3_S:   LAUNCH_GU_F8(MOE_Q_IQ3_S);   break;
        case MOE_Q_Q2_0:    LAUNCH_GU_F8(MOE_Q_Q2_0);    break;
        case MOE_Q_IQ4_NL:  LAUNCH_GU_F8(MOE_Q_IQ4_NL);  break;
        default: fp8_iq_fail("unsupported gate_up IQ MoEQuantType");
    }
#undef LAUNCH_GU_F8
}

template <int TM_VAL>
void dispatch_dn_fp8(MoEQuantType qt, const IqArgs & dn, const Fp8Scratch & f,
                     const MoePrefillScratch & s, int tiles, hipStream_t stream) {
#define LAUNCH_DN_F8(Q) \
    moe_iq_down_fp8_wmma_kernel<Q, TM_VAL><<<dim3(N_EMBD / DN_BN, tiles), MOE_NT, 0, stream>>>( \
        dn, f.act8, s.exp_offsets, s.perm_w, s.tile_exp, s.tile_m0, s.d_num_tiles, s.perm_down)
    switch (qt) {
        case MOE_Q_Q2_0:    LAUNCH_DN_F8(MOE_Q_Q2_0);    break;
        case MOE_Q_IQ4_NL:  LAUNCH_DN_F8(MOE_Q_IQ4_NL);  break;
        default: fp8_iq_fail("unsupported down IQ MoEQuantType (must have QK | 1152)");
    }
#undef LAUNCH_DN_F8
}

void launch_moe_iq_fp8_chunk(MoEQuantType gu_t, const IqArgs & gu,
                             MoEQuantType dn_t, const IqArgs & dn,
                             const float * X_norm, const int32_t * topk_ids, const float * topk_w,
                             MoePrefillScratch & s, float * Y_out, int M, hipStream_t stream,
                             const float * post_conv_w, float * post_sconv_h, const float * post_res_add) {
    const Fp8Scratch f = carve(s, M);
    const bool use_large = (M > MAX_UBATCH);
    const int tm = use_large ? TM_LARGE : TM_SMALL;
    route_kernel<<<1, ROUTE_THREADS, 0, stream>>>(
        topk_ids, topk_w, M, tm, s.exp_counts, s.exp_offsets, s.exp_cursors,
        s.perm_tok, s.perm_w, s.tile_exp, s.tile_m0, s.d_num_tiles, f.tok_slots);
    launch_x_to_fp8_rowscale(X_norm, f.x8, f.xs, M, stream);
    const int tiles = max_tiles(M, tm);
    if (use_large) {
        dispatch_gu_fp8<TM_LARGE>(gu_t, gu, f, s, tiles, stream);
        dispatch_dn_fp8<TM_LARGE>(dn_t, dn, f, s, tiles, stream);
    } else {
        dispatch_gu_fp8<TM_SMALL>(gu_t, gu, f, s, tiles, stream);
        dispatch_dn_fp8<TM_SMALL>(dn_t, dn, f, s, tiles, stream);
    }
    if (post_conv_w != nullptr) {
        if (post_sconv_h == nullptr) {
            fp8_iq_fail("launch_moe_iq_prefill_fp8_wmma: post_sconv_h must be non-null with post_conv_w");
        }
        const int n_tchunks = (M + COMBINE_SCONV_TCHUNK - 1) / COMBINE_SCONV_TCHUNK;
        combine_sconv_kernel<<<dim3(COMBINE_SCONV_BLOCKS, n_tchunks), COMBINE_SCONV_THREADS, 0, stream>>>(
            f.tok_slots, s.perm_down, post_conv_w, post_sconv_h, post_res_add, Y_out, M);
    } else {
        combine_kernel<<<M, COMBINE_THREADS, 0, stream>>>(f.tok_slots, s.perm_down, Y_out);
    }
}

} // namespace

void launch_moe_iq_prefill_fp8_wmma(const SoaIQExperts & gate_up, const SoaIQExperts & down,
                                    const float * X_norm, const int32_t * topk_ids, const float * topk_w,
                                    MoePrefillScratch & scratch, float * Y_out, int M, hipStream_t stream,
                                    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
                                    int scratch_cap) {
    if (M <= 0) {
        return;
    }
    if (gate_up.rows != N_GATE_UP_EXP || gate_up.K != N_EMBD || down.rows != N_EMBD || down.K != N_FF_EXP) {
        fp8_iq_fail("IQ experts must be gate_up [*, 2304, 2304] and down [*, 2304, 1152]");
    }
    if (gate_up.exp_amax == nullptr || down.exp_amax == nullptr) {
        fp8_iq_fail("IQ experts need their per-expert exp_amax for FP8 prefill");
    }
    if (scratch_cap <= 0) {
        fp8_iq_fail("scratch_cap must be positive");
    }
    require_rdna4_iq("launch_moe_iq_prefill_fp8_wmma");
    const IqArgs gu{gate_up.p0, gate_up.p1, gate_up.d, gate_up.exp_amax};
    const IqArgs dn{down.p0, down.p1, down.d, down.exp_amax};
    const int chunk_cap = (scratch_cap < FP8_MAX_UBATCH) ? scratch_cap : FP8_MAX_UBATCH;
    for (int t0 = 0; t0 < M; t0 += chunk_cap) {
        const int mc = (M - t0) < chunk_cap ? (M - t0) : chunk_cap;
        const float * chunk_res = (post_res_add != nullptr) ? (post_res_add + (size_t) t0 * N_EMBD) : nullptr;
        launch_moe_iq_fp8_chunk(gate_up.type, gu, down.type, dn,
                                X_norm + (size_t) t0 * N_EMBD, topk_ids + (size_t) t0 * TOP_K,
                                topk_w + (size_t) t0 * TOP_K, scratch, Y_out + (size_t) t0 * N_EMBD,
                                mc, stream, post_conv_w, post_sconv_h, chunk_res);
    }
}

} // namespace inkling

#endif // NIB_FP8_CODEGEN
