// ============================================================================
// kernels_wmma_fp8_rdna4.hip — RDNA4 (gfx1200 / gfx1201) FP8 WMMA grouped NVFP4 MoE prefill
// ============================================================================
// Same pipeline and scratch as the FP16 grouped MoE in kernels_wmma_prefill.hip (route ->
// convert -> gate_up + SITU -> down -> combine), with both GEMMs on v_wmma_f32_16x16x16_fp8_fp8
// (2x the FP16 WMMA rate on RDNA4, half the LDS bytes per MAC):
//
//   route   : deterministic per-expert slot order + tile worklist (identical algorithm and
//             outputs to the FP16 path's moe_route_kernel; Test 11 checks combine parity).
//   convert : X_norm row -> E4M3 with a per-row scale 448 / amax (one global read, in-block
//             amax reduction) into scratch.perm_x, K-interleaved (fp8_moe_kpos).
//   stage 1 : grouped FP8 WMMA gate+up; NVFP4 weights are decoded to E4M3 through a 1 KB per-CTA
//             LDS table (nib_nvfp4_fp8_table, fold / 8); epilogue rescales by
//             xs[tok] * 8 * scale2, applies SITU and rounds x FP8_MOE_ACT_SCALE to E4M3
//             (bounded: |x| < 400) into scratch.perm_act, K-interleaved.
//   stage 2 : grouped FP8 WMMA down * (8 / FP8_MOE_ACT_SCALE) * scale2 * router weight -> perm_down.
//   combine : per token, sum of its <= 8 slots in k order (+ optional fused 4-tap sconv + residual).
//
// LDS operand layout: one 16-byte unit per (row, unit u); the units of a K step are XOR-swizzled
// by row (by row pair when a row is 64 bytes) so the 8 lanes of every ds_load_b128 /
// ds_store_b128 phase hit distinct banks.  A
// lane's unit holds its 8-byte K slice of two consecutive 16-deep K chunks (the K-interleave),
// so one ds_load_b128 feeds two WMMAs.  Weights are decoded into exactly the same layout.
//
// Compiled for every target of the build.  The FP8 kernels and the host launchers exist only in
// the host pass and gfx12 device passes (NIB_FP8_CODEGEN); a gfx11 code object of this file holds
// just the arch-neutral route / combine kernels, which RDNA3 never launches.
// ============================================================================
#include "inkling_engine.h"
#include "kernels_wmma_fp8.h"
#include "nib_arch_fp8.h"

#include <cfloat>
#include <cstdio>
#include <cstdlib>

// ============================================================================
// Arch-neutral kernels (every target).  Routing and combine use no FP8 instructions, so they are
// compiled for all targets of the build with external linkage; this also keeps every code object
// of this file non-empty, which the spill / stub audit (scripts/spill_audit.sh) requires.
// ============================================================================
namespace inkling {
namespace fp8_moe {

// ---------------------------------------------------------------------------
// MoE routing (single block) — same algorithm and outputs as kernels_wmma_prefill.hip:
// per-expert counts / offsets, deterministic slots (tokens ascending inside each expert),
// tile worklist of TM-token tiles, perm_tok, perm_w and tok_slots.
// ---------------------------------------------------------------------------
constexpr int ROUTE_THREADS = 1024;
constexpr int MASK_WORDS    = FP8_MAX_UBATCH / 32; // 32 words (1024 tokens): 34,312 B <= 64 KB LDS

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

    // Expert membership bitmasks (out-of-range expert ids contribute nothing).
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

    // One slot per distinct (token, expert); a repeated expert sums its router weights into the
    // first occurrence (mathematically identical) and later occurrences get slot -1.
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

// ---------------------------------------------------------------------------
// Deterministic combine (+ optional fused causal 4-tap sconv + residual) — same algorithm and
// rounding as kernels_wmma_prefill.hip (per token, its slots summed in k order).
// ---------------------------------------------------------------------------
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

constexpr int COMBINE_SCONV_THREADS = 64;                 // 4 channels per thread
constexpr int COMBINE_SCONV_TCHUNK  = 8;                  // tokens per time chunk

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

// blockIdx.x = channel block (4 channels per thread), blockIdx.y = 8-token time chunk.  Only
// t_blk 0 reads and finally rewrites sconv_hist (with the last 3 tokens of the batch), so the
// history update is race-free; later chunks rebuild their 3-token history from perm_down.
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
        // t0 >= 8 > 3, so t0 - 3 .. t0 - 1 are valid tokens of this batch.
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

} // namespace fp8_moe
} // namespace inkling

// ============================================================================
// FP8 kernels and host launchers (host pass + gfx12 device passes only)
// ============================================================================
#if NIB_FP8_CODEGEN

namespace inkling {
namespace {

using namespace fp8_moe;

typedef int v4i __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------------------
// Activation / layout helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float situ_scalar(float gate, float up) {
    // Same expression as the FP16 path: SITU(beta = 4) * soft-capped (25) linear unit.
    const float act = SITU_BETA * tanhf(gate * (1.0f / SITU_BETA)) / (1.0f + expf(-gate));
    const float u   = SITU_LINEAR_BETA * tanhf(up * (1.0f / SITU_LINEAR_BETA));
    return act * u;
}

__device__ __forceinline__ nib_fp8x8_t lo8(const v4i & v) { return nib_fp8x8_t{v.x, v.y}; }
__device__ __forceinline__ nib_fp8x8_t hi8(const v4i & v) { return nib_fp8x8_t{v.z, v.w}; }

// 128-entry NVFP4 -> E4M3 group table (one 8-byte entry per UE4M3 scale code), built per CTA.
constexpr int LUT_ENTRIES = 128;

__device__ __forceinline__ void build_lut(uint2 * s_lut) {
    if (threadIdx.x < LUT_ENTRIES) {
        s_lut[threadIdx.x] = nib_nvfp4_fp8_table(threadIdx.x);
    }
}

// ---------------------------------------------------------------------------
// Grouped MoE FP8 WMMA tile mainloop
// ---------------------------------------------------------------------------
// BM x BN output tile (PAIRED: BN/2 gate rows + the matching BN/2 up rows), WARPS_M x WARPS_N
// warps, BK8 K bytes per K step, SU K steps of weights prefetched in registers per super-step.
// A operand: BM rows gathered by the caller (a_ptr), already E4M3 and K-interleaved -> copied.
// B operand: 2 adjacent lanes per weight row (bchk = 0/1 owns the first/second half of the
// step), each streaming BK8/4 nibble bytes + BK8/32 UE4M3 scales per K step and decoding them
// through the CTA's LUT straight into the LDS operand layout.
template <int BM, int BN, int WARPS_M, int WARPS_N, bool PAIRED, int BK8, int SU>
struct TileGemmMoeF8 {
    static constexpr int NT    = WARPS_M * WARPS_N * 32;
    static constexpr int WTM   = BM / WARPS_M;
    static constexpr int FM    = WTM / 16;
    static constexpr int OUTN  = PAIRED ? BN / 2 : BN;
    static constexpr int WTN   = OUTN / WARPS_N;
    static constexpr int FNO   = WTN / 16;
    static constexpr int FN    = PAIRED ? 2 * FNO : FNO;
    static constexpr int UPR   = BK8 / 16;                     // 16-byte LDS units per row per step
    static constexpr int PAIRS = BK8 / 32;                     // WMMA K-chunk pairs per step
    static constexpr int RPL   = (UPR >= 8) ? 1 : 8 / UPR;     // rows sharing one 128-byte bank line
    static constexpr int A_N   = BM * UPR;
    static constexpr int A_IT  = (A_N + NT - 1) / NT;
    static constexpr bool A_EXACT = (A_N % NT) == 0;
    static constexpr int BQ    = BK8 / 64;                     // nibble v4i per lane per step
    static_assert(BK8 == 64 || BK8 == 128, "BK8 is 64 or 128 K bytes per step");
    static_assert(BM % (16 * WARPS_M) == 0 && OUTN % (16 * WARPS_N) == 0, "warp tiling");
    static_assert(NT >= 2 * BN, "two lanes per weight row");

    struct Smem {
        v4i a[BM * UPR];
        v4i b[BN * UPR];
    };

    // Per-lane weight registers for SU K steps: BK8/4 nibble bytes and BK8/32 scale bytes each.
    struct SReg {
        v4i      q[SU][BQ];
        uint32_t sc[SU];
    };

    const uint8_t * a_ptr[A_IT];
    const uint8_t * s_rowq = nullptr;   // this lane's weight row: nibbles
    const uint8_t * s_rows = nullptr;   //                         UE4M3 scales
    SReg            sr;
    v4i             ar[A_IT];

    // Lanes [0, 2 * BN) stream weights (brow = threadIdx.x >> 1); with NT == 2 * BN that is all.
    __device__ __forceinline__ static bool is_b_lane(int brow) { return NT == 2 * BN || brow < BN; }

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

    // Weights of K steps [ks0, ks0 + SU * BK8) for this lane's half of each step.
    __device__ __forceinline__ void sload(int bchk, int ks0, int K) {
#pragma unroll
        for (int t = 0; t < SU; ++t) {
            const int kk = ks0 + t * BK8 + bchk * (BK8 / 2);
            const bool in = kk < K;
            const v4i * q = reinterpret_cast<const v4i *>(s_rowq + (kk >> 1));
#pragma unroll
            for (int c = 0; c < BQ; ++c) {
                sr.q[t][c] = in ? q[c] : v4i{0, 0, 0, 0};
            }
            if (BK8 == 128) {
                sr.sc[t] = in ? *reinterpret_cast<const uint32_t *>(s_rows + (kk >> 4)) : 0u;
            } else {
                sr.sc[t] = in ? (uint32_t) *reinterpret_cast<const uint16_t *>(s_rows + (kk >> 4)) : 0u;
            }
        }
    }

    // Decode K step t of the lane's registers into its B row: every nibble v4i is one pair of
    // 16-weight groups (g0 = dwords x,y; g1 = dwords z,w) and becomes two LDS units:
    //   unit 2P     = [g0 K 0..7  | g1 K 0..7 ]   (read by lanes 0..15)
    //   unit 2P + 1 = [g0 K 8..15 | g1 K 8..15]   (read by lanes 16..31)
    __device__ __forceinline__ void storeB(Smem & sm, const uint2 * s_lut, int brow, int bchk, int t) {
#pragma unroll
        for (int c = 0; c < BQ; ++c) {
            const v4i q = sr.q[t][c];
            const uint2 t0 = s_lut[(sr.sc[t] >> (16 * c)) & 0x7Fu];
            const uint2 t1 = s_lut[(sr.sc[t] >> (16 * c + 8)) & 0x7Fu];
            const uint4 g0 = nib_nvfp4_group_to_fp8((uint32_t) q.x, (uint32_t) q.y, t0);
            const uint4 g1 = nib_nvfp4_group_to_fp8((uint32_t) q.z, (uint32_t) q.w, t1);
            const int P = bchk * BQ + c;
            sm.b[unit_idx(brow, 2 * P + 0)] = v4i{(int) g0.x, (int) g0.y, (int) g1.x, (int) g1.y};
            sm.b[unit_idx(brow, 2 * P + 1)] = v4i{(int) g0.z, (int) g0.w, (int) g1.z, (int) g1.w};
        }
    }

    __device__ __forceinline__ static int b_frag_row(int wn, int f) {
        if (PAIRED) {
            return (f < FNO) ? (wn * WTN + f * 16) : (OUTN + wn * WTN + (f - FNO) * 16);
        }
        return wn * WTN + f * 16;
    }

    __device__ __forceinline__ void compute(const Smem & sm, nib_float8_t (&acc)[FM][FN], int wm, int wn,
                                            int l16, int half_w, int m_valid) {
        // Skip a warp none of whose rows hold tokens (wave-uniform).  A partially live warp also
        // runs its dead 16-row sub-tiles (their A rows are zero): predicating single WMMAs would
        // make the compiler copy whole accumulators around every conditional (measured: 256
        // VGPRs + spills for FM = 2 instead of ~110).
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
            // K chunk 2p for every (i, j), then chunk 2p + 1: FM * FN independent chains.
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

    // Full K loop.  Prologue: the caller has filled s_lut (no barrier yet); loads are issued
    // before the barrier that publishes the LUT so they overlap it.
    __device__ __forceinline__ void run(Smem & sm, const uint2 * s_lut, int K, nib_float8_t (&acc)[FM][FN],
                                        int m_valid) {
        // Wave-uniform by construction; readfirstlane lets the compiler see it, so the warp skip
        // in compute() is a scalar branch.
        const int warp   = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
        const int lane   = threadIdx.x & 31;
        const int wm     = warp / WARPS_N;
        const int wn     = warp % WARPS_N;
        const int l16    = lane & 15;
        const int half_w = lane >> 4;
        const int brow   = threadIdx.x >> 1;
        const int bchk   = threadIdx.x & 1;
        const bool b_lane = is_b_lane(brow);
#pragma unroll
        for (int i = 0; i < FM; ++i) {
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                acc[i][j] = nib_float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
        const int nk   = (K + BK8 - 1) / BK8;
        const int nsup = (nk + SU - 1) / SU;
        loadA(0, K);
        if (b_lane) {
            sload(bchk, 0, K);
        }
        __syncthreads(); // s_lut complete
#pragma unroll 1
        for (int sp = 0; sp < nsup; ++sp) {
#pragma unroll
            for (int t = 0; t < SU; ++t) {
                const int kt = sp * SU + t;
                __builtin_amdgcn_sched_barrier(0);
                storeA(sm);
                if (b_lane) {
                    storeB(sm, s_lut, brow, bchk, t);
                }
                __syncthreads();
                if (kt + 1 < nk) {
                    loadA((kt + 1) * BK8, K);
                }
                if (t + 1 == SU && sp + 1 < nsup && b_lane) {
                    sload(bchk, (sp + 1) * SU * BK8, K);
                }
                compute(sm, acc, wm, wn, l16, half_w, m_valid);
                __syncthreads();
            }
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

// ---------------------------------------------------------------------------
// Launch configuration (shared tile height: one tile worklist for both stages)
// ---------------------------------------------------------------------------
// TM = 64 for M <= 512 (mean ~32 tok/exp -> single tile per expert); TM = 128 for M > 512
// (mean ~64 tok/exp at M=1024 -> ~95%+ experts fit in one tile, avoiding 2x weight reloads).
// Both use 8 warps (4 x 2, 256 threads), BN = 128, BK8 = 64 (LDS: 13,568 B at TM=64; 17,920 B at TM=128).
constexpr int TM_SMALL = 64;
constexpr int TM_LARGE = 128;
constexpr int GU_OUT   = 64;                 // activation columns per gate_up tile
constexpr int DN_BN    = 128;                // output columns per down tile
constexpr int MOE_BK8  = 64;
constexpr int MOE_WM   = 4;
constexpr int MOE_WN   = 2;
constexpr int MOE_NT   = MOE_WM * MOE_WN * 32;
constexpr int SU_GU    = 4;                  // gate_up: K = 2304 = 36 steps -> 9 super-steps
constexpr int SU_DN    = 3;                  // down:    K = 1152 = 18 steps -> 6 super-steps
static_assert((N_EMBD / MOE_BK8) % SU_GU == 0 && (N_FF_EXP / MOE_BK8) % SU_DN == 0, "no K tail");
static_assert(N_FF_EXP % GU_OUT == 0 && N_EMBD % DN_BN == 0, "N tiling");

template <int TM_VAL>
using GateUpTile = TileGemmMoeF8<TM_VAL, 2 * GU_OUT, MOE_WM, MOE_WN, true,  MOE_BK8, SU_GU>;
template <int TM_VAL>
using DownTile   = TileGemmMoeF8<TM_VAL, DN_BN,      MOE_WM, MOE_WN, false, MOE_BK8, SU_DN>;

struct NvFp4Args {
    const uint8_t * qs;
    const uint8_t * sc;
    const float   * scale2;
};

// ---------------------------------------------------------------------------
// X_norm row -> E4M3 with a per-row scale (one block per row, 8 elements per thread)
// ---------------------------------------------------------------------------
constexpr int CVT_THREADS = N_EMBD / 8;   // 288 threads = 9 warps
constexpr int CVT_WARPS   = CVT_THREADS / 32;
static_assert(CVT_THREADS % 32 == 0, "whole warps");

__global__ void __launch_bounds__(CVT_THREADS)
x_to_fp8_rowscale_kernel(const float * __restrict__ X, uint8_t * __restrict__ X8, float * __restrict__ xs) {
    __shared__ float s_wmax[CVT_WARPS];
    const int row = blockIdx.x;
    const int t   = threadIdx.x;
    const float4 * src = reinterpret_cast<const float4 *>(X + (size_t) row * N_EMBD) + 2 * t;
    const float4 a = src[0];
    const float4 b = src[1];
    const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};

    // Row amax; any Inf/NaN maps to +Inf (NaN fails the <= test), flagging the whole row.
    float m = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float x = fabsf(v[i]);
        m = fmaxf(m, (x <= FLT_MAX) ? x : INFINITY);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor(m, off, 32));
    }
    if ((t & 31) == 0) {
        s_wmax[t >> 5] = m;
    }
    __syncthreads();
    float amax = s_wmax[0];
#pragma unroll
    for (int w = 1; w < CVT_WARPS; ++w) {
        amax = fmaxf(amax, s_wmax[w]);
    }

    float sa;        // quantisation scale
    float inv;       // dequantisation multiplier written to xs
    if (amax == INFINITY) {
        sa  = 0.0f;
        inv = __builtin_nanf("");
    } else if (amax < NIB_FP8_AMAX_MIN) {
        sa  = 0.0f;
        inv = 0.0f;
    } else {
        sa  = NIB_FP8_MAX / amax;
        inv = amax * (1.0f / NIB_FP8_MAX);
    }
    // |v * sa| <= 448 up to one FP32 rounding; the clamp makes the E4M3 input range exact.
    float q[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        q[i] = (sa == 0.0f) ? 0.0f : nib_fp8_clamp(v[i] * sa);
    }
    const uint2 out = make_uint2(nib_f32x4_to_fp8x4(q[0], q[1], q[2], q[3]),
                                 nib_f32x4_to_fp8x4(q[4], q[5], q[6], q[7]));
    // Thread t holds logical K 8t..8t+7: one 8-byte sub-block, stored at its interleaved slot.
    *reinterpret_cast<uint2 *>(X8 + (size_t) row * N_EMBD + fp8_moe_kpos(8 * t)) = out;
    if (t == 0) {
        xs[row] = inv;
    }
}

// ---------------------------------------------------------------------------
// Grouped stage 1: FP8 WMMA gate_up + dequant + SITU -> E4M3 activations (perm_act)
// ---------------------------------------------------------------------------
// Expert-contiguous grid: blockIdx.x = N tile (N_FF_EXP / GU_OUT), blockIdx.y = M tile.
template <int TM_VAL>
__global__ void __launch_bounds__(MOE_NT, 2)
moe_gate_up_fp8_wmma_kernel(const NvFp4Args W, const uint8_t * __restrict__ X8, const float * __restrict__ xs,
                            const int32_t * __restrict__ exp_offsets, const int32_t * __restrict__ perm_tok,
                            const int32_t * __restrict__ tile_exp, const int32_t * __restrict__ tile_m0,
                            const int32_t * __restrict__ d_num_tiles, uint8_t * __restrict__ act8) {
    typedef GateUpTile<TM_VAL> G;
    __shared__ __align__(16) typename G::Smem sm;
    __shared__ uint2 s_lut[LUT_ENTRIES];
    __shared__ float s_xs[TM_VAL];

    const int tile = blockIdx.y;
    if (tile >= *d_num_tiles) {
        return; // uniform for the whole block
    }
    constexpr int K = N_EMBD;
    const int e     = tile_exp[tile];
    const int slot0 = exp_offsets[e] + tile_m0[tile];
    const int mrem  = exp_offsets[e + 1] - slot0;
    const int mcnt  = mrem < TM_VAL ? mrem : TM_VAL;
    const int n0    = blockIdx.x * GU_OUT;

    build_lut(s_lut);
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
    if (G::is_b_lane(brow)) {
        // B tile row brow: [0, GU_OUT) = gate rows n0.., [GU_OUT, 2 GU_OUT) = the matching up rows.
        const int row = (brow < GU_OUT) ? (n0 + brow) : (N_FF_EXP + n0 + (brow - GU_OUT));
        const size_t grow = (size_t) e * N_GATE_UP_EXP + row;
        g.s_rowq = W.qs + grow * (K / 2);
        g.s_rows = W.sc + grow * (K / 16);
    }

    nib_float8_t acc[G::FM][G::FN];
    g.run(sm, s_lut, K, acc, mcnt);

    const float wsc = W.scale2[e] * NIB_NVFP4_FP8_FOLD;
#pragma unroll
    for (int i = 0; i < G::FM; ++i) {
#pragma unroll
        for (int j = 0; j < G::FNO; ++j) {
            const int col = fp8_moe_kpos(n0 + G::out_col(j));
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int m = G::out_row(i, v);
                if (m < mcnt) {
                    const float sx = s_xs[m] * wsc;
                    const float a  = situ_scalar(acc[i][j][v] * sx, acc[i][j + G::FNO][v] * sx);
                    // |a| < 100 -> |a * 4| < 400: inside E4M3 range without a clamp (NaN propagates).
                    const uint32_t b = nib_f32x2_to_fp8x2<false>(a * FP8_MOE_ACT_SCALE, 0.0f, 0u);
                    act8[(size_t) (slot0 + m) * N_FF_EXP + col] = (uint8_t) (b & 0xFFu);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Grouped stage 2: FP8 WMMA down * scale2 * router weight -> perm_down (FP32)
// ---------------------------------------------------------------------------
// Expert-contiguous grid: blockIdx.x = N tile (N_EMBD / DN_BN), blockIdx.y = M tile.
template <int TM_VAL>
__global__ void __launch_bounds__(MOE_NT, 2)
moe_down_fp8_wmma_kernel(const NvFp4Args W, const uint8_t * __restrict__ act8,
                         const int32_t * __restrict__ exp_offsets, const float * __restrict__ perm_w,
                         const int32_t * __restrict__ tile_exp, const int32_t * __restrict__ tile_m0,
                         const int32_t * __restrict__ d_num_tiles, float * __restrict__ perm_down) {
    typedef DownTile<TM_VAL> G;
    __shared__ __align__(16) typename G::Smem sm;
    __shared__ uint2 s_lut[LUT_ENTRIES];

    const int tile = blockIdx.y;
    if (tile >= *d_num_tiles) {
        return;
    }
    constexpr int K = N_FF_EXP;
    const int e     = tile_exp[tile];
    const int slot0 = exp_offsets[e] + tile_m0[tile];
    const int mrem  = exp_offsets[e + 1] - slot0;
    const int mcnt  = mrem < TM_VAL ? mrem : TM_VAL;
    const int n0    = blockIdx.x * DN_BN;

    build_lut(s_lut);

    G g;
#pragma unroll
    for (int p = 0; p < G::A_IT; ++p) {
        const int i = G::a_row(p);
        g.a_ptr[p] = (G::a_live(p) && i < mcnt) ? act8 + (size_t) (slot0 + i) * K : nullptr;
    }
    const int brow = threadIdx.x >> 1;
    if (G::is_b_lane(brow)) {
        const size_t grow = (size_t) e * N_EMBD + n0 + brow;
        g.s_rowq = W.qs + grow * (K / 2);
        g.s_rows = W.sc + grow * (K / 16);
    }

    nib_float8_t acc[G::FM][G::FN];
    g.run(sm, s_lut, K, acc, mcnt);

    const float s2 = W.scale2[e] * (NIB_NVFP4_FP8_FOLD / FP8_MOE_ACT_SCALE);
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

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
[[noreturn]] void fp8_contract_fail(const char * what) {
    std::fprintf(stderr, "[inkling kernels_wmma_fp8_rdna4] contract violation: %s\n", what);
    std::abort();
}

void require_rdna4(const char * who) {
    if (!rdna4_fp8_prefill_supported()) {
        std::fprintf(stderr, "[inkling kernels_wmma_fp8_rdna4] %s requires an RDNA4 (gfx12) GPU, got %s\n",
                     who, nib_device_info().gcn_arch);
        std::abort();
    }
}

constexpr int COMBINE_SCONV_BLOCKS = (N_EMBD / 4) / COMBINE_SCONV_THREADS; // 9 channel blocks

// Worst-case number of M tiles: sum_e ceil(c_e / tm) <= floor(P / tm) + min(P, N_EXPERT).
inline int max_tiles(int M, int tm) {
    const int P = M * TOP_K;
    return P / tm + (P < N_EXPERT ? P : N_EXPERT);
}

// Scratch carve-up (fits inside the reclaimed MoePrefillScratch):
//   perm_x   (>= cap * N_EMBD + align16(cap * 4) + cap * TOP_K * 4 bytes; ~2.29 MiB at cap=1024):
//            X8 [cap, N_EMBD] bytes, then xs [cap] floats, then 16B-aligned tok_slots [cap * TOP_K] int32.
//   perm_act (>= cap * TOP_K * N_FF_EXP bytes; 9.0 MiB at cap=1024): E4M3 activations [slots, N_FF_EXP] bytes.
struct Fp8Scratch {
    uint8_t * x8;
    float   * xs;
    int32_t * tok_slots;
    uint8_t * act8;
};

Fp8Scratch carve(const MoePrefillScratch & s, int cap) {
    uint8_t * base = reinterpret_cast<uint8_t *>(s.perm_x);
    const size_t x8_bytes = (size_t) cap * N_EMBD;                          // N_EMBD % 16 == 0
    const size_t xs_bytes = ((size_t) cap * sizeof(float) + 15u) & ~size_t(15);
    Fp8Scratch f;
    f.x8        = base;
    f.xs        = reinterpret_cast<float *>(base + x8_bytes);
    f.tok_slots = reinterpret_cast<int32_t *>(base + x8_bytes + xs_bytes);  // 16-byte aligned for int4 loads
    f.act8      = reinterpret_cast<uint8_t *>(s.perm_act);
    return f;
}
static_assert((N_EMBD % 16) == 0, "xs stays 16-byte aligned for any cap");

void check_scratch(const MoePrefillScratch & s) {
    if (!s.exp_counts || !s.exp_offsets || !s.exp_cursors || !s.perm_tok || !s.perm_w || !s.perm_x || !s.perm_act ||
        !s.perm_down || !s.tile_exp || !s.tile_m0 || !s.d_num_tiles) {
        fp8_contract_fail("MoePrefillScratch has a null buffer");
    }
}

void launch_moe_fp8_chunk(const NvFp4Args & gu, const NvFp4Args & dn, const float * X_norm,
                          const int32_t * topk_ids, const float * topk_w, MoePrefillScratch & s,
                          float * Y_out, int M, hipStream_t stream,
                          const float * post_conv_w, float * post_sconv_h, const float * post_res_add) {
    const Fp8Scratch f = carve(s, M);
    const bool use_large = (M > MAX_UBATCH);
    const int tm = use_large ? TM_LARGE : TM_SMALL;
    route_kernel<<<1, ROUTE_THREADS, 0, stream>>>(topk_ids, topk_w, M, tm, s.exp_counts, s.exp_offsets,
                                                  s.exp_cursors, s.perm_tok, s.perm_w, s.tile_exp,
                                                  s.tile_m0, s.d_num_tiles, f.tok_slots);
    x_to_fp8_rowscale_kernel<<<M, CVT_THREADS, 0, stream>>>(X_norm, f.x8, f.xs);
    const int tiles = max_tiles(M, tm);
    if (use_large) {
        moe_gate_up_fp8_wmma_kernel<TM_LARGE><<<dim3(N_FF_EXP / GU_OUT, tiles), MOE_NT, 0, stream>>>(
            gu, f.x8, f.xs, s.exp_offsets, s.perm_tok, s.tile_exp, s.tile_m0, s.d_num_tiles, f.act8);
        moe_down_fp8_wmma_kernel<TM_LARGE><<<dim3(N_EMBD / DN_BN, tiles), MOE_NT, 0, stream>>>(
            dn, f.act8, s.exp_offsets, s.perm_w, s.tile_exp, s.tile_m0, s.d_num_tiles, s.perm_down);
    } else {
        moe_gate_up_fp8_wmma_kernel<TM_SMALL><<<dim3(N_FF_EXP / GU_OUT, tiles), MOE_NT, 0, stream>>>(
            gu, f.x8, f.xs, s.exp_offsets, s.perm_tok, s.tile_exp, s.tile_m0, s.d_num_tiles, f.act8);
        moe_down_fp8_wmma_kernel<TM_SMALL><<<dim3(N_EMBD / DN_BN, tiles), MOE_NT, 0, stream>>>(
            dn, f.act8, s.exp_offsets, s.perm_w, s.tile_exp, s.tile_m0, s.d_num_tiles, s.perm_down);
    }
    if (post_conv_w != nullptr) {
        if (post_sconv_h == nullptr) {
            fp8_contract_fail("launch_moe_nvfp4_prefill_fp8_wmma: post_sconv_h must be non-null with post_conv_w");
        }
        const int n_tchunks = (M + COMBINE_SCONV_TCHUNK - 1) / COMBINE_SCONV_TCHUNK;
        combine_sconv_kernel<<<dim3(COMBINE_SCONV_BLOCKS, n_tchunks), COMBINE_SCONV_THREADS, 0, stream>>>(
            f.tok_slots, s.perm_down, post_conv_w, post_sconv_h, post_res_add, Y_out, M);
    } else {
        combine_kernel<<<M, COMBINE_THREADS, 0, stream>>>(f.tok_slots, s.perm_down, Y_out);
    }
}

} // namespace

// ============================================================================
// Public API
// ============================================================================
bool rdna4_fp8_prefill_built() {
    return true;
}

bool rdna4_fp8_prefill_supported() {
    return nib_device_info().is_rdna4;
}

void launch_x_to_fp8_rowscale(const float * X, uint8_t * X8, float * xs, int M, hipStream_t stream) {
    if (M <= 0) {
        return;
    }
    require_rdna4("launch_x_to_fp8_rowscale");
    x_to_fp8_rowscale_kernel<<<M, CVT_THREADS, 0, stream>>>(X, X8, xs);
}

void launch_moe_nvfp4_prefill_fp8_wmma(const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
                                       const float * X_norm, const int32_t * topk_ids, const float * topk_w,
                                       MoePrefillScratch & scratch, float * Y_out, int M, hipStream_t stream,
                                       const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
                                       int scratch_cap) {
    if (M <= 0) {
        return;
    }
    if (gate_up.rows != N_GATE_UP_EXP || gate_up.K != N_EMBD || down.rows != N_EMBD || down.K != N_FF_EXP) {
        fp8_contract_fail("NVFP4 experts must be gate_up [*, 2304, 2304] and down [*, 2304, 1152]");
    }
    if (gate_up.scale2 == nullptr || down.scale2 == nullptr) {
        fp8_contract_fail("NVFP4 experts need their per-expert scale2");
    }
    if (scratch_cap <= 0) {
        fp8_contract_fail("scratch_cap must be positive");
    }
    check_scratch(scratch);
    require_rdna4("launch_moe_nvfp4_prefill_fp8_wmma");
    const NvFp4Args gu{gate_up.qs, gate_up.scales, gate_up.scale2};
    const NvFp4Args dn{down.qs, down.scales, down.scale2};
    const int chunk_cap = (scratch_cap < FP8_MAX_UBATCH) ? scratch_cap : FP8_MAX_UBATCH;
    // The scratch holds chunk_cap tokens (<= FP8_MAX_UBATCH=1024): longer batches chunk stream-ordered.
    for (int t0 = 0; t0 < M; t0 += chunk_cap) {
        const int mc = (M - t0) < chunk_cap ? (M - t0) : chunk_cap;
        const float * chunk_res = (post_res_add != nullptr) ? (post_res_add + (size_t) t0 * N_EMBD) : nullptr;
        launch_moe_fp8_chunk(gu, dn, X_norm + (size_t) t0 * N_EMBD, topk_ids + (size_t) t0 * TOP_K,
                             topk_w + (size_t) t0 * TOP_K, scratch, Y_out + (size_t) t0 * N_EMBD, mc, stream,
                             post_conv_w, post_sconv_h, chunk_res);
    }
}

} // namespace inkling

#endif // NIB_FP8_CODEGEN
