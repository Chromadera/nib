// ============================================================================
// kernels_step7h_fused.hip — Step 7H for RDNA3 gfx1100 (RX 7900 XTX, 96 CU).
//
// Part 1: shipped-shape fused MoE (NVFP4 44L, Q8 4L).
//   Step 7F proved that re-tiling gate_up (8 lanes/pair) LOSES 7 us/call because
//   the shipped 4-lane tiling broadcasts ONE 128 B activation chunk across the
//   lane group (8x less LDS traffic).  Its fused kernel was correct and saved
//   -1.91 us/layer busy + -44 graph nodes of dispatch, but paid the 8-lane
//   gate_up.  Step 7H keeps the fused single launch and restores the EXACT
//   shipped 4-lane/64-weight gate_up and the EXACT shipped down_wsum shape, so
//   the arithmetic is bit-identical to the shipped two-launch pipeline:
//     phase 1  blocks [0, 144) x 8 warps, 8 pair-groups/warp = 64 pairs/block,
//              pair p = bid*64 + warp*8 + lane/4  (1152/64 = 18 blocks/expert,
//              no pair crosses an expert, same per-lane weights and FMA order)
//     phase 2  all GRID blocks, 8 warps, the shipped warp-per-expert down with
//              blockIdx*2 row start and GRID*2 stride, exact 8-expert sum tree
//   Both phases sit inside one launch behind the Step 7F 4-shard self-resetting
//   grid barrier.  Co-residency: GRID=288 x 256 threads = 3 blocks/CU.  The
//   body is pinned to 80 VGPRs (6 waves/SIMD = 3 blocks/CU) and 12.4 KB LDS
//   (3 blocks = 37 KB < 64 KB), so every block is resident before any spin.
//   Variant 1 runs GRID=192 (2 blocks/CU, no VGPR pin) as a spill-free A/B.
//
// Part 2: A/B/C_v3 (96-CU parallel small/mid fused kernels).
//   B_v2 waits on ONE mix block (6.9 us of pure latency) before 32 router blocks
//   can start.  B_v3 runs the identical 192-thread mix+norm in EVERY one of 128
//   blocks (no grid barrier at all; the 5 states are 46 KB and L2-resident) and
//   dots exactly one expert row per block, so the mix latency is paid in
//   parallel 128x.  The single last-arriving block runs the branchless Top-8.
//   C_v2 runs 32 blocks (67% of the GPU idle).  C_v3 runs 96 blocks x 288
//   threads: same per-rank / per-channel dot math, 3x the DRAM streams.
//
// Part 3: wo re-tile A/B (the K-order and reduction tree are unchanged).
// Part 4: Pass-0 mix+RMSNorm folded into the QKV prep launch (-24 nodes).
// ============================================================================

#include "kernels_step7h_fused.h"
#include "nib_target.h"

#include <cstdio>
#include <cstdlib>

namespace inkling {
namespace {

typedef int          v4i __attribute__((ext_vector_type(4)));
typedef unsigned int v2u __attribute__((ext_vector_type(2)));

// ---------------------------------------------------------------------------
// Exact copies of the shipped device helpers (same instruction sequences, so
// every kernel below is numerically identical to its shipped counterpart).
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

// Branchless UE4M3 -> FP32 (same values as the shipped ternary form, but only
// value selects - the nested exec-mask form cost 7 s_and_saveexec regions).
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

static __device__ __forceinline__ __half2 h2_of(uint32_t u) {
    return __builtin_bit_cast(__half2, u);
}

static __device__ __forceinline__ float dot_h8_f8(uint4 w, float4 x0, float4 x1, float acc) {
    const __half2 h0 = h2_of(w.x);
    const __half2 h1 = h2_of(w.y);
    const __half2 h2 = h2_of(w.z);
    const __half2 h3 = h2_of(w.w);
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

static __device__ __forceinline__ float dot_h8_f8v(v4i w, float4 x0, float4 x1, float acc) {
    const __half2 h0 = h2_of((uint32_t) w.x);
    const __half2 h1 = h2_of((uint32_t) w.y);
    const __half2 h2 = h2_of((uint32_t) w.z);
    const __half2 h3 = h2_of((uint32_t) w.w);
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

template <int W>
__device__ __forceinline__ float group_reduce_sum(float v) {
#pragma unroll
    for (int off = W / 2; off > 0; off >>= 1) {
        v += __shfl_xor(v, off, W);
    }
    return v;
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, 32);
    }
    return x;
}

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
typedef LdsLayout<8, 9, N_EMBD / 4, (N_EMBD / 32) * 9>    LayQ8X;

template <int NT, int N, class LAY>
__device__ __forceinline__ void lds_fill(float4 * dst, const float4 * __restrict__ src) {
    constexpr int PER = (N + NT - 1) / NT;
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
            dst[LAY::map(idx)] = v[i];
        }
    }
}

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

// ---------------------------------------------------------------------------
// gfx1100 memory ordering + self-resetting cooperative counters
// ---------------------------------------------------------------------------
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

constexpr int H_S_BARRIER = 400; // 400..403 shards, 404 ack (same as Step 7F)

template <int SHARDS>
__device__ __forceinline__ void grid_barrier(float * scratch, int nblocks) {
    rdna3_flush_stores();
    __syncthreads();
    if (threadIdx.x == 0) {
        rdna3_flush_stores();
        (void) scratch_add(scratch, H_S_BARRIER + (blockIdx.x % SHARDS));
        const int target = (nblocks + SHARDS - 1) / SHARDS;
#pragma unroll
        for (int i = 0; i < SHARDS; ++i) {
            spin_until(scratch, H_S_BARRIER + i, target);
        }
        const int prev = scratch_add(scratch, H_S_BARRIER + SHARDS);
        if (prev == nblocks - 1) {
#pragma unroll
            for (int i = 0; i < SHARDS; ++i) {
                scratch_store(scratch, H_S_BARRIER + i, 0);
            }
            scratch_store(scratch, H_S_BARRIER + SHARDS, 0);
        }
    }
    __syncthreads();
    rdna3_inv_l0_l1();
}

// Deterministic Top-8 over 128 expert logits by one Wave32 (identical to the
// shipped/Step 7D kernel).
__device__ __forceinline__ void wave32_topk8_128(
        const float * s_sel, const float * s_p,
        int32_t * topk_ids_out, float * topk_w_out, const int lane) {
    float my_sel[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        my_sel[j] = s_sel[lane + j * 32];
    }
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
            if (v > best_v || (v == best_v && idx < best_i)) {
                best_v = v;
                best_i = idx;
            }
        }
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float   ov = __shfl_xor(best_v, off, 32);
            const int32_t oi = __shfl_xor(best_i, off, 32);
            if (ov > best_v || (ov == best_v && oi < best_i)) {
                best_v = ov;
                best_i = oi;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            if (lane + j * 32 == best_i) {
                my_sel[j] = -INFINITY;
            }
        }
        if (lane == 0) {
            top_i[k] = best_i;
            top_p[k] = s_p[best_i];
        }
    }
    if (lane == 0) {
        float sum_p = 0.0f;
#pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            sum_p += top_p[k];
        }
        const float scale = EXPERT_W_SCALE / (sum_p + EXPERT_NORM_EPS);
#pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            topk_ids_out[k] = top_i[k];
            topk_w_out[k]   = top_p[k] * scale;
        }
    }
}

// ---------------------------------------------------------------------------
// MoE decode shapes
// ---------------------------------------------------------------------------
constexpr int NV_K    = N_EMBD;   // 2304  (gate_up contraction)
constexpr int NV_QSR  = NV_K / 2; // 1152 B/row
constexpr int NV_SCR  = NV_K / 16;// 144 B/row
constexpr int NV_NB   = NV_K / 64;// 36 x 64-weight blocks/row
constexpr int NV_GULPR = 4;       // shipped 4-lane pair mapping
constexpr int NV_GUPPW = 32 / NV_GULPR; // 8 pairs/warp

constexpr int DN_K    = N_FF_EXP; // 1152
constexpr int DN_QSR  = DN_K / 2; // 576
constexpr int DN_SCR  = DN_K / 16;// 72

constexpr int Q8_K    = N_EMBD;
constexpr int Q8_NB   = Q8_K / 32; // 72
constexpr int Q8_LPR  = 32;        // shipped q8 gate_up: one pair per warp

// ===========================================================================
// Part 1a. NVFP4 gate_up phase (EXACT shipped moe_nvfp4_gate_up_situ_kernel<4>)
// ===========================================================================
template <int NWARPS_ACTIVE>
__device__ __forceinline__ void nvfp4_gu_ship_phase(
    const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
    const float * __restrict__ scale2, const int32_t * __restrict__ topk_ids,
    float * __restrict__ act, int pair_base, const float4 * __restrict__ s_x) {
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int g    = lane % NV_GULPR;
    if (warp < NWARPS_ACTIVE) {
        // EXACT shipped pair mapping: one pair per 4-lane group, 8 pairs/warp,
        // 4 active warps per block => 32 pairs/block, 288 blocks = 9216 pairs.
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

        static_assert(NV_NB % NV_GULPR == 0, "36 blocks = 9 per lane");
        constexpr int ITS = NV_NB / NV_GULPR;

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
        accg = group_reduce_sum<NV_GULPR>(accg);
        accu = group_reduce_sum<NV_GULPR>(accu);
        if (g == 0) {
            act[k * N_FF_EXP + r] = situ_default(accg * s2, accu * s2);
        }
    }
}

// ===========================================================================
// Part 1b. NVFP4 down phase (EXACT shipped moe_nvfp4_down_wsum_kernel<1, GRID>)
// ===========================================================================
template <int GRID, bool LEAN>
__device__ __forceinline__ void nvfp4_down_ship_phase(
    const uint8_t * __restrict__ qs, const uint8_t * __restrict__ scales,
    const float * __restrict__ scale2, const float * __restrict__ act,
    const int32_t * __restrict__ topk_ids, const float * __restrict__ topk_w,
    float * __restrict__ out, const SconvEpi & epi,
    float4 (*s_act1)[8][16], float4 (*s_act2)[2][16], float (*s_red)[8][2]) {
    constexpr int K   = DN_K;
    constexpr int RPB = 2;

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
    if (LEAN) {
        // rx0 stays in VGPRs; the whole 16+g block (8 float4) is staged in LDS so
        // the down phase fits in 80 VGPRs without spilling.  Values are identical
        // to the register copy -> bit-exact.
#pragma unroll
        for (int c = 0; c < 8; ++c) {
            rx0[c] = ak[g * 8 + c];
        }
        if (rloc == 0) {
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                s_act1[k][c][g] = ak[(16 + g) * 8 + c];
            }
        }
    } else {
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
        }
    }
    if (rloc == 0) {
        // handled above for both variants
    } else {
        const int sb = g >> 1;
        const int wi = g & 1;
        s_act2[k][0][g] = ak[256 + sb * 4 + wi];
        s_act2[k][1][g] = ak[256 + sb * 4 + 2 + wi];
    }
    __syncthreads();

    int iter = 0;
    for (int r0 = blockIdx.x * RPB; r0 < N_EMBD; r0 += GRID * RPB, ++iter) {
        const int r = r0 + rloc;
        const size_t row = (size_t) ex * N_EMBD + r;
        const uint8_t  * q  = qs + row * DN_QSR;
        const uint16_t * sc = reinterpret_cast<const uint16_t *>(scales + row * DN_SCR);
        const v4i w00 = ld_nt_b128(q + g * 16);
        v4i w10;
        if (!LEAN) {
            w10 = ld_nt_b128(q + 256 + g * 16);
        }
        const uint32_t w2 = ld_nt_b32(q + 512 + 4 * g);
        const uint32_t s0 = sc[g];
        const uint32_t s1 = sc[16 + g];
        const uint32_t s2 = (scales + row * DN_SCR)[64 + (g >> 1)];

        __builtin_amdgcn_sched_barrier(0);
        float acc = nvfp4_sub_dot((uint32_t) w00.x, (uint32_t) w00.y, rx0[0], rx0[1], rx0[2], rx0[3]) *
                    ue4m3_to_fp32(s0 & 0xFFu);
        acc = fmaf(nvfp4_sub_dot((uint32_t) w00.z, (uint32_t) w00.w, rx0[4], rx0[5], rx0[6], rx0[7]),
                   ue4m3_to_fp32(s0 >> 8), acc);

        __builtin_amdgcn_sched_barrier(0);
        float4 x1_0, x1_1, x1_2, x1_3, x1_4, x1_5, x1_6, x1_7;
        if (LEAN) {
            w10 = ld_nt_b128(q + 256 + g * 16);
            x1_0 = s_act1[k][0][g];
            x1_1 = s_act1[k][1][g];
            x1_2 = s_act1[k][2][g];
            x1_3 = s_act1[k][3][g];
            x1_4 = s_act1[k][4][g];
            x1_5 = s_act1[k][5][g];
            x1_6 = s_act1[k][6][g];
            x1_7 = s_act1[k][7][g];
        } else {
            x1_0 = rx1[0]; x1_1 = rx1[1]; x1_2 = rx1[2]; x1_3 = rx1[3];
            x1_4 = s_act1[k][0][g];
            x1_5 = s_act1[k][1][g];
            x1_6 = s_act1[k][2][g];
            x1_7 = s_act1[k][3][g];
        }
        acc = fmaf(nvfp4_sub_dot((uint32_t) w10.x, (uint32_t) w10.y, x1_0, x1_1, x1_2, x1_3),
                   ue4m3_to_fp32(s1 & 0xFFu), acc);
        acc = fmaf(nvfp4_sub_dot((uint32_t) w10.z, (uint32_t) w10.w, x1_4, x1_5, x1_6, x1_7),
                   ue4m3_to_fp32(s1 >> 8), acc);

        __builtin_amdgcn_sched_barrier(0);
        const float4 x2_0 = s_act2[k][0][g];
        const float4 x2_1 = s_act2[k][1][g];
        uint32_t lo, hi;
        nvfp4_hbytes(w2, lo, hi);
        float p3 = 0.0f;
        p3 = nvfp4_hb_dot_mix(lo, x2_0, p3);
        p3 = nvfp4_hb_dot_mix(hi, x2_1, p3);
        acc = fmaf(p3, ue4m3_to_fp32(s2), acc);

        const float v = group_reduce_sum<16>(acc) * coef;
        if (g == 0) {
            s_red[iter & 1][k][rloc] = v;
        }
        __syncthreads();
        if (warp == 0 && lane < RPB) {
            const float (*rb)[2] = s_red[iter & 1];
            const float sum = ((rb[0][lane] + rb[1][lane]) + (rb[2][lane] + rb[3][lane])) +
                              ((rb[4][lane] + rb[5][lane]) + (rb[6][lane] + rb[7][lane]));
            const int ro = r0 + lane;
            out[ro] = (epi.conv_w != nullptr) ? sconv_post_res_one(epi, ro, sum) : sum;
        }
    }
}

// ===========================================================================
// Part 1c. NVFP4 fused kernel (one launch, shipped shapes, 4-shard barrier)
// ===========================================================================
union MoeSharedU {
    float4 gu_x[NV_NB * LayNvX::SEGP_];      // 612 float4 = 9792 B
    struct {
        float4 a1[8][8][16];                  // 16384 B (LEAN: all 8 float4 of the 16+g block)
        float4 a2[8][2][16];                  //  4096 B
        float  red[2][8][2];                  //   128 B (shipped RPB=2 slots)
    } dn;
};

template <int GRID, int GU_BLOCKS, int NWARPS, int SHARDS, bool LEAN>
__device__ __forceinline__ void moe_nvfp4_fused_body(
    const uint8_t * __restrict__ gu_qs, const uint8_t * __restrict__ gu_sc,
    const float * __restrict__ gu_s2, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ topk_w, const float * __restrict__ x_norm,
    const uint8_t * __restrict__ dn_qs, const uint8_t * __restrict__ dn_sc,
    const float * __restrict__ dn_s2, float * __restrict__ act,
    float * __restrict__ out, const SconvEpi epi, float * __restrict__ scratch,
    MoeSharedU & sh) {
    constexpr int NT = NWARPS * 32;
    static_assert(N_FF_EXP % (NWARPS * NV_GUPPW) == 0, "pair tiling");
    // Shipped 4-lane pair tiling: 8 pairs/warp.  288 blocks -> 4 active warps
    // (32 pairs/block, exactly the shipped grid); 192 blocks -> 6 active warps
    // (48 pairs/block).  Both cover the 9216 pairs exactly.
    constexpr int GU_WARPS_ACTIVE = (GRID == 288) ? 4 : 6;
    constexpr int PAIRS_PER_BLOCK = GU_WARPS_ACTIVE * NV_GUPPW;
    static_assert(GU_BLOCKS * PAIRS_PER_BLOCK == N_FF_EXP * TOP_K, "phase 1 coverage");

    // ---- phase 1: gate_up + SITU (blocks [0, GU_BLOCKS), exact shipped math)
    if ((int) blockIdx.x < GU_BLOCKS) {
        lds_fill<NT, N_EMBD / 4, LayNvX>(sh.gu_x, reinterpret_cast<const float4 *>(x_norm));
    }
    __syncthreads();
    if ((int) blockIdx.x < GU_BLOCKS) {
        nvfp4_gu_ship_phase<GU_WARPS_ACTIVE>(gu_qs, gu_sc, gu_s2, topk_ids, act,
                                             blockIdx.x * PAIRS_PER_BLOCK, sh.gu_x);
    }

    grid_barrier<SHARDS>(scratch, GRID);

    // ---- phase 2: down + weighted sum + sconv epilogue (all GRID blocks)
    nvfp4_down_ship_phase<GRID, LEAN>(dn_qs, dn_sc, dn_s2, act, topk_ids, topk_w, out, epi,
                                      sh.dn.a1, sh.dn.a2, sh.dn.red);
}

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
// GRID = 288: 3 blocks/CU must be co-resident, so the allocation is pinned to
// 80 VGPRs (6 waves/SIMD) and 12.4 KB LDS (3 x 12.4 = 37 KB < 64 KB).
__global__ void __launch_bounds__(256) __attribute__((amdgpu_num_vgpr(72)))
moe_nvfp4_ship_fused288_kernel(
    const uint8_t * __restrict__ gu_qs, const uint8_t * __restrict__ gu_sc,
    const float * __restrict__ gu_s2, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ topk_w, const float * __restrict__ x_norm,
    const uint8_t * __restrict__ dn_qs, const uint8_t * __restrict__ dn_sc,
    const float * __restrict__ dn_s2, float * __restrict__ act,
    float * __restrict__ out, const SconvEpi epi, float * __restrict__ scratch) {
    __shared__ MoeSharedU sh;
    moe_nvfp4_fused_body<288, 288, 8, 4, true>(gu_qs, gu_sc, gu_s2, topk_ids, topk_w, x_norm,
                                               dn_qs, dn_sc, dn_s2, act, out, epi, scratch, sh);
}

// GRID = 192: 2 blocks/CU co-resident with no VGPR pin (spill-free A/B).
__global__ void __launch_bounds__(256)
moe_nvfp4_ship_fused192_kernel(
    const uint8_t * __restrict__ gu_qs, const uint8_t * __restrict__ gu_sc,
    const float * __restrict__ gu_s2, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ topk_w, const float * __restrict__ x_norm,
    const uint8_t * __restrict__ dn_qs, const uint8_t * __restrict__ dn_sc,
    const float * __restrict__ dn_s2, float * __restrict__ act,
    float * __restrict__ out, const SconvEpi epi, float * __restrict__ scratch) {
    __shared__ MoeSharedU sh;
    moe_nvfp4_fused_body<192, 192, 8, 4, false>(gu_qs, gu_sc, gu_s2, topk_ids, topk_w, x_norm,
                                                dn_qs, dn_sc, dn_s2, act, out, epi, scratch, sh);
}

// ===========================================================================
// Part 1d. Q8 MoE fused (192 blocks x 256 threads)
// ---------------------------------------------------------------------------
// gate_up: the shipped per-pair math (LPR = 32, one warp per pair, 72 blocks of
// 32 weights over the warp, 1-deep ring) but each warp walks 6 pairs of its
// expert.  Every pair is an independent accumulation with the identical
// per-lane block order (sub + i*32) and the identical 32-lane butterfly, so the
// result is bit-identical; only the block packing changes (1152 -> 192 blocks).
// down: exact shipped moe_q8_down_wsum_kernel<2, 192> (192 blocks, R = 2).
// ===========================================================================
__device__ __forceinline__ void q8_gu_ship_phase(
    const int8_t * __restrict__ qs, const __half * __restrict__ d,
    const float * __restrict__ x_norm, const int32_t * __restrict__ topk_ids,
    float * __restrict__ act, const float4 * __restrict__ s_x) {
    constexpr int FULL_ITS  = Q8_NB / Q8_LPR;   // 2
    constexpr int TOTAL_ITS = FULL_ITS + 1;     // 3 (128-weight tail)
    constexpr int PPW       = 1;                // pairs per warp per group
    constexpr int NGROUP    = (N_FF_EXP * TOP_K) / (192 * 8 * PPW); // 6

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
        accg = group_reduce_sum<Q8_LPR>(accg);
        accu = group_reduce_sum<Q8_LPR>(accu);
        if (g == 0) {
            act[k * N_FF_EXP + r] = eok ? situ_default(accg, accu) : 0.0f;
        }
    }
}

// Exact shipped moe_q8_down_wsum_kernel<2, GRID_BLOCKS> body.
template <int R, int GRID>
__device__ __forceinline__ void q8_down_ship_phase(
    const int8_t * __restrict__ qs, const __half * __restrict__ d,
    const float * __restrict__ act, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ topk_w, float * __restrict__ out, const SconvEpi & epi,
    float4 (*s_act1)[8][16], float4 (*s_act2)[2][16], float (*s_red)[2 * R]) {
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

union MoeQ8SharedU {
    float4 gu_x[NV_NB * LayQ8X::SEGP_];  // 5184 B
    struct {
        float4 a1[8][8][16];
        float4 a2[8][2][16];
        float  red[8][4];                // R = 2 -> RPB = 4
    } dn;
};

template <int GRID>
__device__ __forceinline__ void moe_q8_fused_body(
    const int8_t * __restrict__ gu_qs, const __half * __restrict__ gu_d,
    const float * __restrict__ x_norm, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ topk_w,
    const int8_t * __restrict__ dn_qs, const __half * __restrict__ dn_d,
    float * __restrict__ act, float * __restrict__ out, const SconvEpi epi,
    float * __restrict__ scratch, MoeQ8SharedU & sh) {
    constexpr int NT = 256;
    static_assert(GRID == 192, "q8 down shape is <2, 192>");
    lds_fill<NT, N_EMBD / 4, LayQ8X>(sh.gu_x, reinterpret_cast<const float4 *>(x_norm));
    __syncthreads();
    q8_gu_ship_phase(gu_qs, gu_d, x_norm, topk_ids, act, sh.gu_x);

    grid_barrier<4>(scratch, GRID);

    q8_down_ship_phase<2, GRID>(dn_qs, dn_d, act, topk_ids, topk_w, out, epi,
                                sh.dn.a1, sh.dn.a2, sh.dn.red);
}

__global__ void __launch_bounds__(256)
moe_q8_ship_fused_kernel(
    const int8_t * __restrict__ gu_qs, const __half * __restrict__ gu_d,
    const float * __restrict__ x_norm, const int32_t * __restrict__ topk_ids,
    const float * __restrict__ topk_w,
    const int8_t * __restrict__ dn_qs, const __half * __restrict__ dn_d,
    float * __restrict__ act, float * __restrict__ out, const SconvEpi epi,
    float * __restrict__ scratch) {
    __shared__ MoeQ8SharedU sh;
    moe_q8_fused_body<192>(gu_qs, gu_d, x_norm, topk_ids, topk_w,
                           dn_qs, dn_d, act, out, epi, scratch, sh);
}


// ===========================================================================
// Part 0. n_states-specialised mix + RMSNorm (bit-exact, same tree)
// ---------------------------------------------------------------------------
// The shipped inkling_mix_rms_norm_2304_kernel<5, Q> runs a runtime
// `if (m < n_states)` around every state load, which keeps the 15 activation
// loads per thread from being batched: the single block pays ~7 us of mostly
// serialised DRAM latency for 46 KB.  Specialising MAX_M == n_states makes every
// load unconditional (same addresses, same per-thread 3-float4 mapping, same
// warp tree, same 6-warp sum, same effw algebra), so the result is bit-identical
// and the block can overlap its loads.
// ===========================================================================
typedef struct { const float * p[MAX_RES_BLOCKS]; } HStatePtrs;
typedef struct { const float * p[4]; } HNormPtrs;

template <int M, int Q>
__launch_bounds__(192, 4)
__global__ void mix_rms_norm_2304_fast_kernel(
        const float * __restrict__ query,
        HStatePtrs states,
        HNormPtrs  w_norms,
        float * __restrict__ dst,
        float eps) {
    constexpr int N_WARPS = 6;
    constexpr int NT      = 192;

    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane_id = tid & 31;

    __shared__ float smem_sumsq[M][N_WARPS];
    __shared__ float smem_dot[Q * M][N_WARPS];
    __shared__ float smem_out_ssq[Q][N_WARPS];

    float4 s_vec[M][3];
#pragma unroll
    for (int m = 0; m < M; ++m) {
        const float4 * sp = (const float4 *)(states.p[m]);
        const float4 v0 = sp[tid + 0 * NT];
        const float4 v1 = sp[tid + 1 * NT];
        const float4 v2 = sp[tid + 2 * NT];
        s_vec[m][0] = v0;
        s_vec[m][1] = v1;
        s_vec[m][2] = v2;
        float ssq = (v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w)
                  + (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w)
                  + (v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w);
        ssq = warp_reduce_sum(ssq);
        if (lane_id == 0) {
            smem_sumsq[m][warp_id] = ssq;
        }
    }

#pragma unroll
    for (int q = 0; q < Q; ++q) {
        const float4 * qp = (const float4 *)(query + (size_t)q * N_EMBD);
        const float4 q0 = qp[tid + 0 * NT];
        const float4 q1 = qp[tid + 1 * NT];
        const float4 q2 = qp[tid + 2 * NT];
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float4 v0 = s_vec[m][0];
            const float4 v1 = s_vec[m][1];
            const float4 v2 = s_vec[m][2];
            float d = (v0.x * q0.x + v0.y * q0.y + v0.z * q0.z + v0.w * q0.w)
                    + (v1.x * q1.x + v1.y * q1.y + v1.z * q1.z + v1.w * q1.w)
                    + (v2.x * q2.x + v2.y * q2.y + v2.z * q2.z + v2.w * q2.w);
            d = warp_reduce_sum(d);
            if (lane_id == 0) {
                smem_dot[q * M + m][warp_id] = d;
            }
        }
    }

    __syncthreads();

    constexpr float inv_n    = 1.0f / (float)N_EMBD;
    const float     inv_sq_n = rsqrtf((float)N_EMBD);

    float my_inv_rms = 0.0f;
    if (lane_id < M) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            tot += smem_sumsq[lane_id][w];
        }
        my_inv_rms = rsqrtf(tot * inv_n + RMS_EPS);
    }

    float4 ov[Q][3];
#pragma unroll
    for (int q = 0; q < Q; ++q) {
        float sc = -INFINITY;
        if (lane_id < M) {
            float tot_d = 0.0f;
#pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                tot_d += smem_dot[q * M + lane_id][w];
            }
            sc = tot_d * my_inv_rms * inv_sq_n;
        }
        float max_s = sc;
        max_s = fmaxf(max_s, __shfl_xor(max_s, 1, 8));
        max_s = fmaxf(max_s, __shfl_xor(max_s, 2, 8));
        max_s = fmaxf(max_s, __shfl_xor(max_s, 4, 8));
        const float ex = (lane_id < M) ? expf(sc - max_s) : 0.0f;
        float sum_e = ex;
        sum_e += __shfl_xor(sum_e, 1, 8);
        sum_e += __shfl_xor(sum_e, 2, 8);
        sum_e += __shfl_xor(sum_e, 4, 8);
        const float my_effw = (lane_id < M) ? (ex * (1.0f / sum_e) * my_inv_rms) : 0.0f;

        float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 o1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 o2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int m = 0; m < M; ++m) {
            const float w = __shfl(my_effw, m, 32);
            o0.x = fmaf(w, s_vec[m][0].x, o0.x); o0.y = fmaf(w, s_vec[m][0].y, o0.y);
            o0.z = fmaf(w, s_vec[m][0].z, o0.z); o0.w = fmaf(w, s_vec[m][0].w, o0.w);
            o1.x = fmaf(w, s_vec[m][1].x, o1.x); o1.y = fmaf(w, s_vec[m][1].y, o1.y);
            o1.z = fmaf(w, s_vec[m][1].z, o1.z); o1.w = fmaf(w, s_vec[m][1].w, o1.w);
            o2.x = fmaf(w, s_vec[m][2].x, o2.x); o2.y = fmaf(w, s_vec[m][2].y, o2.y);
            o2.z = fmaf(w, s_vec[m][2].z, o2.z); o2.w = fmaf(w, s_vec[m][2].w, o2.w);
        }
        ov[q][0] = o0; ov[q][1] = o1; ov[q][2] = o2;
        float ssq_o = (o0.x * o0.x + o0.y * o0.y + o0.z * o0.z + o0.w * o0.w)
                    + (o1.x * o1.x + o1.y * o1.y + o1.z * o1.z + o1.w * o1.w)
                    + (o2.x * o2.x + o2.y * o2.y + o2.z * o2.z + o2.w * o2.w);
        ssq_o = warp_reduce_sum(ssq_o);
        if (lane_id == 0) {
            smem_out_ssq[q][warp_id] = ssq_o;
        }
    }

    __syncthreads();

#pragma unroll
    for (int q = 0; q < Q; ++q) {
        float tot = 0.0f;
#pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            tot += smem_out_ssq[q][w];
        }
        const float inv = rsqrtf(tot * inv_n + RMS_EPS);
        const float * wn = w_norms.p[q];
        const float4 w0 = wn ? ((const float4 *)wn)[tid + 0 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        const float4 w1 = wn ? ((const float4 *)wn)[tid + 1 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        const float4 w2 = wn ? ((const float4 *)wn)[tid + 2 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
        float4 o0 = ov[q][0], o1 = ov[q][1], o2 = ov[q][2];
        o0.x *= inv * w0.x; o0.y *= inv * w0.y; o0.z *= inv * w0.z; o0.w *= inv * w0.w;
        o1.x *= inv * w1.x; o1.y *= inv * w1.y; o1.z *= inv * w1.z; o1.w *= inv * w1.w;
        o2.x *= inv * w2.x; o2.y *= inv * w2.y; o2.z *= inv * w2.z; o2.w *= inv * w2.w;
        float4 * out4 = (float4 *)(dst + (size_t)q * N_EMBD);
        out4[tid + 0 * NT] = o0;
        out4[tid + 1 * NT] = o1;
        out4[tid + 2 * NT] = o2;
    }
}

// ===========================================================================
// Part 2. B_v3: Pass-0 pre-MoE mix + FFN RMSNorm + router + Top-8
// ---------------------------------------------------------------------------
// B_v2 serialises: ONE block runs the 6.9 us mix_rms_norm while 32 router blocks
// spin, then 32 blocks dot 4 experts each.  B_v3 runs the *identical* 192-thread
// mix+norm in every one of 128 blocks (the 5 states are 46 KB and L2/L1 hot, so
// the redundancy is cheap and there is no grid barrier at all), then each block
// dots exactly one expert row with the identical 1-warp / 9 x uint4 pattern, so
// x_norm and every logit are bit-identical to B_v2.
// ===========================================================================
__launch_bounds__(192, 4)
__global__ void mix_ffn_norm_router_topk_v3_kernel(
        const float * __restrict__ res_query,
        HStatePtrs states, int n_states,
        const float * __restrict__ ffn_norm_w,
        const __half * __restrict__ router_w,
        const float * __restrict__ exp_probs_b,
        float * __restrict__ x_norm_out,
        int32_t * __restrict__ topk_ids_out,
        float * __restrict__ topk_w_out,
        float * __restrict__ scratch) {
    constexpr int N_WARPS = 6;
    constexpr int NT      = 192;

    const int bid     = blockIdx.x;   // 0..127 = expert id
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    __shared__ float4 s_xn[N_EMBD / 4];
    __shared__ float  lds_p[N_EXPERT];
    __shared__ float  lds_sel[N_EXPERT];
    __shared__ bool   s_last;

    // ---------------- exact inkling_mix_rms_norm_2304_kernel<5,1> ----------
    {
        constexpr int MAX_M = 5;
        __shared__ float smem_sumsq[MAX_M][N_WARPS];
        __shared__ float smem_dot[MAX_M][N_WARPS];
        __shared__ float smem_out_ssq[N_WARPS];

        float4 s_vec[MAX_M][3];
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_states) {
                const float4 * sp = (const float4 *)(states.p[m]);
                const float4 v0 = sp[tid + 0 * NT];
                const float4 v1 = sp[tid + 1 * NT];
                const float4 v2 = sp[tid + 2 * NT];
                s_vec[m][0] = v0;
                s_vec[m][1] = v1;
                s_vec[m][2] = v2;
                float ssq = (v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w)
                          + (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w)
                          + (v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w);
                ssq = warp_reduce_sum(ssq);
                if (lane == 0) {
                    smem_sumsq[m][warp_id] = ssq;
                }
            }
        }
        {
            const float4 * qp = (const float4 *)(res_query);
            const float4 q0 = qp[tid + 0 * NT];
            const float4 q1 = qp[tid + 1 * NT];
            const float4 q2 = qp[tid + 2 * NT];
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_states) {
                    const float4 v0 = s_vec[m][0];
                    const float4 v1 = s_vec[m][1];
                    const float4 v2 = s_vec[m][2];
                    float d = (v0.x * q0.x + v0.y * q0.y + v0.z * q0.z + v0.w * q0.w)
                            + (v1.x * q1.x + v1.y * q1.y + v1.z * q1.z + v1.w * q1.w)
                            + (v2.x * q2.x + v2.y * q2.y + v2.z * q2.z + v2.w * q2.w);
                    d = warp_reduce_sum(d);
                    if (lane == 0) {
                        smem_dot[m][warp_id] = d;
                    }
                }
            }
        }
        __syncthreads();

        constexpr float inv_n    = 1.0f / (float)N_EMBD;
        const float     inv_sq_n = rsqrtf((float)N_EMBD);

        float my_inv_rms = 0.0f;
        if (lane < n_states) {
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                tot += smem_sumsq[lane][w];
            }
            my_inv_rms = rsqrtf(tot * inv_n + RMS_EPS);
        }

        float4 ov[3];
        {
            float sc = -INFINITY;
            if (lane < n_states) {
                float tot_d = 0.0f;
#pragma unroll
                for (int w = 0; w < N_WARPS; ++w) {
                    tot_d += smem_dot[lane][w];
                }
                sc = tot_d * my_inv_rms * inv_sq_n;
            }
            float max_s = sc;
            max_s = fmaxf(max_s, __shfl_xor(max_s, 1, 8));
            max_s = fmaxf(max_s, __shfl_xor(max_s, 2, 8));
            max_s = fmaxf(max_s, __shfl_xor(max_s, 4, 8));
            const float ex = (lane < n_states) ? expf(sc - max_s) : 0.0f;
            float sum_e = ex;
            sum_e += __shfl_xor(sum_e, 1, 8);
            sum_e += __shfl_xor(sum_e, 2, 8);
            sum_e += __shfl_xor(sum_e, 4, 8);
            const float my_effw = (lane < n_states) ? (ex * (1.0f / sum_e) * my_inv_rms) : 0.0f;

            float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 o1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 o2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_states) {
                    const float w = __shfl(my_effw, m, 32);
                    o0.x = fmaf(w, s_vec[m][0].x, o0.x); o0.y = fmaf(w, s_vec[m][0].y, o0.y);
                    o0.z = fmaf(w, s_vec[m][0].z, o0.z); o0.w = fmaf(w, s_vec[m][0].w, o0.w);
                    o1.x = fmaf(w, s_vec[m][1].x, o1.x); o1.y = fmaf(w, s_vec[m][1].y, o1.y);
                    o1.z = fmaf(w, s_vec[m][1].z, o1.z); o1.w = fmaf(w, s_vec[m][1].w, o1.w);
                    o2.x = fmaf(w, s_vec[m][2].x, o2.x); o2.y = fmaf(w, s_vec[m][2].y, o2.y);
                    o2.z = fmaf(w, s_vec[m][2].z, o2.z); o2.w = fmaf(w, s_vec[m][2].w, o2.w);
                }
            }
            ov[0] = o0; ov[1] = o1; ov[2] = o2;
            float ssq_o = (o0.x * o0.x + o0.y * o0.y + o0.z * o0.z + o0.w * o0.w)
                        + (o1.x * o1.x + o1.y * o1.y + o1.z * o1.z + o1.w * o1.w)
                        + (o2.x * o2.x + o2.y * o2.y + o2.z * o2.z + o2.w * o2.w);
            ssq_o = warp_reduce_sum(ssq_o);
            if (lane == 0) {
                smem_out_ssq[warp_id] = ssq_o;
            }
        }

        __syncthreads();

        {
            float tot = 0.0f;
#pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                tot += smem_out_ssq[w];
            }
            const float inv = rsqrtf(tot * inv_n + RMS_EPS);

            const float4 w0 = ffn_norm_w ? ((const float4 *)ffn_norm_w)[tid + 0 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float4 w1 = ffn_norm_w ? ((const float4 *)ffn_norm_w)[tid + 1 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float4 w2 = ffn_norm_w ? ((const float4 *)ffn_norm_w)[tid + 2 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);

            float4 o0 = ov[0], o1 = ov[1], o2 = ov[2];
            o0.x *= inv * w0.x; o0.y *= inv * w0.y; o0.z *= inv * w0.z; o0.w *= inv * w0.w;
            o1.x *= inv * w1.x; o1.y *= inv * w1.y; o1.z *= inv * w1.z; o1.w *= inv * w1.w;
            o2.x *= inv * w2.x; o2.y *= inv * w2.y; o2.z *= inv * w2.z; o2.w *= inv * w2.w;
            s_xn[tid + 0 * NT] = o0;
            s_xn[tid + 1 * NT] = o1;
            s_xn[tid + 2 * NT] = o2;
            if (bid == 0) {
                float4 * out4 = (float4 *)x_norm_out;
                out4[tid + 0 * NT] = o0;
                out4[tid + 1 * NT] = o1;
                out4[tid + 2 * NT] = o2;
            }
        }
    }
    __syncthreads();

    // ------------------------- router: expert = bid ------------------------
    if (warp_id < 4) {
        // warps 1..3 idle (the dot is one warp, exactly like B_v2)
        if (warp_id == 0) {
            const __half * rbase = router_w + (size_t) bid * N_EMBD;
            v4i rw[9];
#pragma unroll
            for (int s = 0; s < 9; ++s) {
                rw[s] = *reinterpret_cast<const v4i *>(rbase + (size_t)(lane + s * 32) * 8);
            }
            __builtin_amdgcn_sched_barrier(0);
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 9; ++s) {
                const int i8 = lane + s * 32;
                acc = dot_h8_f8v(rw[s], s_xn[i8 * 2 + 0], s_xn[i8 * 2 + 1], acc);
            }
            const float sum = warp_reduce_sum(acc);
            if (lane == 0) {
                const float p   = 1.0f / (1.0f + expf(-sum));
                const float sel = exp_probs_b ? (p + exp_probs_b[bid]) : p;
                __hip_atomic_store(&scratch[bid],            p,   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_store(&scratch[N_EXPERT + bid], sel, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
    }
    rdna3_flush_stores();
    __syncthreads();

    if (tid == 0) {
        rdna3_flush_stores();
        const int prev = scratch_add(scratch, 256);
        s_last = (prev == 127);
        if (s_last) {
            scratch_store(scratch, 256, 0);
        }
    }
    __syncthreads();

    if (s_last && warp_id == 0) {
        rdna3_inv_l0_l1();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + j * 32;
            lds_p[idx]   = __hip_atomic_load(&scratch[idx],            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            lds_sel[idx] = __hip_atomic_load(&scratch[N_EXPERT + idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        wave32_topk8_128(lds_sel, lds_p, topk_ids_out, topk_w_out, lane);
    }
}

// ===========================================================================
// Part 3. C_v3: Pass-1/2 pre-MoE mix + ctrl_down + ctrl_apply + FFN RMSNorm +
//              router + Top-8 -> one 96-block x 288-thread launch.
// ---------------------------------------------------------------------------
// C_v2 runs 32 blocks (67% of the 96 CUs idle -> 95 GB/s).  C_v3 keeps every
// per-rank / per-channel dot exactly as C_v2 computes it (same uint4 layout, same
// 8-lane and 9-warp butterflies) and only re-partitions:
//   * 64 of the 96 blocks each own ONE ctrl rank r = bid (C_v2: 32 x 2 ranks)
//   * every block owns 24 ctrl_apply channels (C_v2: 32 x 72)
//   * the ssq total is the deterministic sum of 96 block partials
// so x_norm differs by at most 1 ulp (the same class as the A/B/C v2 fusion).
// ===========================================================================
constexpr int CV3_GRID = 96;

__launch_bounds__(288, 1)
__global__ void mix_ctrl_ffn_norm_router_topk_v3_kernel(
        const float * __restrict__ res_query,
        HStatePtrs states, int n_states,
        const float * __restrict__ msg,
        const __half * __restrict__ w_down,
        const __half * __restrict__ w_gain,
        const __half * __restrict__ w_forget,
        const float * __restrict__ ffn_norm_w,
        const __half * __restrict__ router_w,
        const float * __restrict__ exp_probs_b,
        float * __restrict__ out_mixed,
        float * __restrict__ out_gate,
        float * __restrict__ x_norm_out,
        int32_t * __restrict__ topk_ids_out,
        float * __restrict__ topk_w_out,
        float * __restrict__ scratch) {
    constexpr int N_WARPS = 9;
    constexpr int NT      = 288;
    constexpr int MAX_M   = 5;

    const int bid     = blockIdx.x;   // 0..95
    const int tid     = threadIdx.x;  // 0..287
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int grp     = tid >> 3;     // 0..35
    const int k8      = tid & 7;

    __shared__ float4 s_xn[N_EMBD / 4];
    __shared__ float  lds_p[N_EXPERT];
    __shared__ float  lds_sel[N_EXPERT];
    __shared__ bool   s_last;
    __shared__ alignas(16) float s_gate[LOOP_GATE_RANK];

    // ---------------- exact mix_ctrl_down_decode_kernel<5> mix -------------
    const float4 rq0 = ((const float4 *)res_query)[tid * 2 + 0];
    const float4 rq1 = ((const float4 *)res_query)[tid * 2 + 1];

    __shared__ float smem_ssq[MAX_M][N_WARPS];
    __shared__ float smem_dot[MAX_M][N_WARPS];
    __shared__ float s_effw[MAX_M];
    __shared__ float smem_g[6][N_WARPS];

    float4 c0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 c1 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 m0 = {0.0f, 0.0f, 0.0f, 0.0f};
    float4 m1 = {0.0f, 0.0f, 0.0f, 0.0f};

    if (bid < 64) {
        // Only the 64 rank blocks need the depth mix; the remaining 32 blocks
        // read out_mixed / msg from L2 in the apply phase.
        float4 s_vec0[MAX_M];
        float4 s_vec1[MAX_M];
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_states) {
                const float4 v0 = ((const float4 *)states.p[m])[tid * 2 + 0];
                const float4 v1 = ((const float4 *)states.p[m])[tid * 2 + 1];
                s_vec0[m] = v0;
                s_vec1[m] = v1;
                const float ssq = v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w +
                                  v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w;
                const float dot = v0.x * rq0.x + v0.y * rq0.y + v0.z * rq0.z + v0.w * rq0.w +
                                  v1.x * rq1.x + v1.y * rq1.y + v1.z * rq1.z + v1.w * rq1.w;
                const float ws = warp_reduce_sum(ssq);
                const float wd = warp_reduce_sum(dot);
                if (lane == 0) {
                    smem_ssq[m][warp_id] = ws;
                    smem_dot[m][warp_id] = wd;
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            constexpr float inv_n    = 1.0f / (float)N_EMBD;
            const float     inv_sq_n = rsqrtf((float)N_EMBD);
            float inv_rms[MAX_M];
            float sc[MAX_M];
            float max_s = -INFINITY;
            for (int m = 0; m < n_states; ++m) {
                float ts = 0.0f, td = 0.0f;
#pragma unroll
                for (int w = 0; w < N_WARPS; ++w) {
                    ts += smem_ssq[m][w];
                    td += smem_dot[m][w];
                }
                inv_rms[m] = rsqrtf(ts * inv_n + RMS_EPS);
                sc[m] = td * inv_rms[m] * inv_sq_n;
                max_s = fmaxf(max_s, sc[m]);
            }
            float sum_e = 0.0f;
            for (int m = 0; m < n_states; ++m) {
                sc[m] = expf(sc[m] - max_s);
                sum_e += sc[m];
            }
            const float inv_e = 1.0f / sum_e;
            for (int m = 0; m < n_states; ++m) {
                s_effw[m] = sc[m] * inv_e * inv_rms[m];
            }
        }
        __syncthreads();

        float4 p0 = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 p1 = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_states) {
                const float w = s_effw[m];
                p0.x += w * s_vec0[m].x; p0.y += w * s_vec0[m].y; p0.z += w * s_vec0[m].z; p0.w += w * s_vec0[m].w;
                p1.x += w * s_vec1[m].x; p1.y += w * s_vec1[m].y; p1.z += w * s_vec1[m].z; p1.w += w * s_vec1[m].w;
            }
        }
        c0 = p0;
        c1 = p1;
        if (bid == 0) {
            ((float4 *)out_mixed)[tid * 2 + 0] = c0;
            ((float4 *)out_mixed)[tid * 2 + 1] = c1;
        }
        m0 = ((const float4 *)msg)[tid * 2 + 0];
        m1 = ((const float4 *)msg)[tid * 2 + 1];

        float ssq_c = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z + c0.w * c0.w +
                      c1.x * c1.x + c1.y * c1.y + c1.z * c1.z + c1.w * c1.w;
        float ssq_m = m0.x * m0.x + m0.y * m0.y + m0.z * m0.z + m0.w * m0.w +
                      m1.x * m1.x + m1.y * m1.y + m1.z * m1.z + m1.w * m1.w;

        const int r = bid;
        const __half * wr0 = w_down + (size_t)r * (2 * N_EMBD);
        const __half * wr1 = wr0 + N_EMBD;
        const uint4 w0 = ((const uint4 *)wr0)[tid];
        const uint4 w1 = ((const uint4 *)wr1)[tid];
        const float dot0 = dot_h8_f8(w0, c0, c1, 0.0f);
        const float dot1 = dot_h8_f8(w1, m0, m1, 0.0f);

        ssq_c = warp_reduce_sum(ssq_c);
        ssq_m = warp_reduce_sum(ssq_m);
        const float rd0 = warp_reduce_sum(dot0);
        const float rd1 = warp_reduce_sum(dot1);
        if (lane == 0) {
            smem_g[0][warp_id] = ssq_c;
            smem_g[1][warp_id] = ssq_m;
            smem_g[2][warp_id] = rd0;
            smem_g[3][warp_id] = rd1;
        }
        __syncthreads();
        if (tid == 0) {
            float tc = 0.0f, tm = 0.0f, td0 = 0.0f, td1 = 0.0f;
#pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                tc  += smem_g[0][w];
                tm  += smem_g[1][w];
                td0 += smem_g[2][w];
                td1 += smem_g[3][w];
            }
            const float ic = rsqrtf(tc * (1.0f / (float)N_EMBD) + RMS_EPS);
            const float im = rsqrtf(tm * (1.0f / (float)N_EMBD) + RMS_EPS);
            out_gate[r] = td0 * ic + td1 * im;
        }
        rdna3_flush_stores();
        __syncthreads();
    }

    if (tid == 0) {
        (void) scratch_add(scratch, 384);   // C_S_GATE_DONE
    }

    // ---- prefetch this block's apply weights (24 channels x 64 ranks) -----
    v4i wg, wf;
    const int c_base = bid * 24 + grp;   // grp < 24 -> tid < 192
    const bool apply_active = (grp < 24);
    if (apply_active) {
        wg = ld_nt_b128(w_gain + (size_t)c_base * LOOP_GATE_RANK + (size_t)k8 * 8);
        wf = ld_nt_b128(w_forget + (size_t)c_base * LOOP_GATE_RANK + (size_t)k8 * 8);
    }

    // ---- gate barrier (96 blocks) ----------------------------------------
    if (tid == 0) {
        spin_until(scratch, 384, CV3_GRID);
    }
    __syncthreads();
    rdna3_inv_l0_l1();
    if (tid == 0) {
        const int prev = scratch_add(scratch, 385);
        if (prev == CV3_GRID - 1) {
            scratch_store(scratch, 384, 0);
            scratch_store(scratch, 385, 0);
        }
    }
    if (tid < LOOP_GATE_RANK) {
        s_gate[tid] = out_gate[tid];
    }
    __syncthreads();

    // ---- ctrl_apply + per-block ssq (24 channels) ------------------------
    __shared__ float s_block_ssq[N_WARPS];
    float val = 0.0f;
    {
        float ssq = 0.0f;
        if (apply_active) {
            const float4 g0 = ((const float4 *)s_gate)[k8 * 2 + 0];
            const float4 g1 = ((const float4 *)s_gate)[k8 * 2 + 1];
            float dg = dot_h8_f8v(wg, g0, g1, 0.0f);
            float df = dot_h8_f8v(wf, g0, g1, 0.0f);
            dg = group_reduce_sum<8>(dg);
            df = group_reduce_sum<8>(df);
            if (k8 == 0) {
                const float gain   = tanhf(dg);
                const float forget = tanhf(df);
                const float cv     = out_mixed[c_base];
                const float mv     = msg[c_base];
                val = cv - forget * cv + gain * mv;
                ssq = val * val;
            }
        }
        ssq = warp_reduce_sum(ssq);
        if (lane == 0) {
            s_block_ssq[warp_id] = ssq;
        }
    }
    rdna3_flush_stores();
    __syncthreads();

    if (tid == 0) {
        float blk = 0.0f;
#pragma unroll
        for (int w = 0; w < N_WARPS; ++w) {
            blk += s_block_ssq[w];
        }
        scratch[280 + bid] = blk;
        rdna3_flush_stores();
        (void) scratch_add(scratch, 386);      // C_S_SSQ_DONE
        spin_until(scratch, 386, CV3_GRID);
    }
    __syncthreads();
    rdna3_inv_l0_l1();
    if (tid == 0) {
        const int prev = scratch_add(scratch, 387);
        if (prev == CV3_GRID - 1) {
            scratch_store(scratch, 386, 0);
            scratch_store(scratch, 387, 0);
        }
    }

    float tot_ssq;
    {
        // deterministic: lane sums slots lane, lane+32, lane+64 then a 32-lane tree
        const int idx = lane;
        float partial = __hip_atomic_load(&scratch[280 + idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        partial += __hip_atomic_load(&scratch[280 + idx + 32], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        partial += __hip_atomic_load(&scratch[280 + idx + 64], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        tot_ssq = warp_reduce_sum(partial);
    }
    const float inv = rsqrtf(tot_ssq * (1.0f / (float)N_EMBD) + RMS_EPS);

    if (apply_active && k8 == 0) {
        const float w0v = ffn_norm_w ? ffn_norm_w[c_base] : 1.0f;
        x_norm_out[c_base] = val * inv * w0v;
    }
    rdna3_flush_stores();
    __syncthreads();
    if (tid == 0) {
        (void) scratch_add(scratch, 388);      // C_S_XN_DONE
        spin_until(scratch, 388, CV3_GRID);
    }
    __syncthreads();
    rdna3_inv_l0_l1();
    if (tid == 0) {
        const int prev = scratch_add(scratch, 389);
        if (prev == CV3_GRID - 1) {
            scratch_store(scratch, 388, 0);
            scratch_store(scratch, 389, 0);
        }
    }

#pragma unroll
    for (int k = 0; k < 2; ++k) {
        s_xn[tid + k * NT] = ((const float4 *)x_norm_out)[tid + k * NT];
    }
    __syncthreads();

    // ------------------------- router (128 experts / 96 blocks) -----------
    // warp 0 -> expert bid; blocks 0..31 warp 1 -> expert 96 + bid
    if (warp_id < 2) {
        const int e = (warp_id == 0) ? bid : (96 + bid);
        const bool has = (warp_id == 0) || (bid < 32);
        if (has) {
            const __half * rbase = router_w + (size_t) e * N_EMBD;
            v4i rw[9];
#pragma unroll
            for (int s = 0; s < 9; ++s) {
                rw[s] = *reinterpret_cast<const v4i *>(rbase + (size_t)(lane + s * 32) * 8);
            }
            __builtin_amdgcn_sched_barrier(0);
            float acc = 0.0f;
#pragma unroll
            for (int s = 0; s < 9; ++s) {
                const int i8 = lane + s * 32;
                acc = dot_h8_f8v(rw[s], s_xn[i8 * 2 + 0], s_xn[i8 * 2 + 1], acc);
            }
            const float sum = warp_reduce_sum(acc);
            if (lane == 0) {
                const float p   = 1.0f / (1.0f + expf(-sum));
                const float sel = exp_probs_b ? (p + exp_probs_b[e]) : p;
                __hip_atomic_store(&scratch[e],            p,   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_store(&scratch[N_EXPERT + e], sel, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
    }
    rdna3_flush_stores();
    __syncthreads();

    if (tid == 0) {
        rdna3_flush_stores();
        // 96 arrivals: dedicated slot 257 (B_v3 owns 256).
        const int prev = scratch_add(scratch, 257);
        s_last = (prev == CV3_GRID - 1);
        if (s_last) {
            scratch_store(scratch, 257, 0);
        }
    }
    __syncthreads();

    if (s_last && warp_id == 0) {
        rdna3_inv_l0_l1();
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int idx = lane + j * 32;
            lds_p[idx]   = __hip_atomic_load(&scratch[idx],            __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            lds_sel[idx] = __hip_atomic_load(&scratch[N_EXPERT + idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
        }
        __atomic_signal_fence(__ATOMIC_SEQ_CST);
        wave32_topk8_128(lds_sel, lds_p, topk_ids_out, topk_w_out, lane);
    }
}

} // namespace

// ============================================================================
// Public API
// ============================================================================

// Co-residency gate for the grid-barrier MoE kernels below (see nib_target.h).
// hipOccupancyMaxActiveBlocksPerMultiprocessor() reports blocks/CU; multiplying by
// the device's MultiprocessorCount attribute (CUs, or WGPs on drivers that report
// half) is a lower bound either way, so a pass can never over-commit the GPU.
// gfx11 keeps its shipped wgp-count thresholds untouched.
namespace {
int g_occ_ship288 = -1, g_occ_ship192 = -1, g_occ_q8ship192 = -1;
int g_wgp_7h = 48;
inline bool h7h_coresident(int occ, int grid) {
#if NIB_RDNA3
    (void)occ; (void)grid;
    return true;
#else
    if (occ <= 0) return false;
    return (long long)occ * (long long)g_wgp_7h >= (long long)grid;
#endif
}
} // namespace

void launch_moe_nvfp4_ship_fused192(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * scratch, float * raw_out);

void launch_moe_nvfp4_ship_fused(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * scratch, float * raw_out) {
    if (gate_up.rows != N_GATE_UP_EXP || gate_up.K != N_EMBD ||
        down.rows != N_EMBD || down.K != N_FF_EXP) {
        std::fprintf(stderr, "[kernels_step7h] nvfp4 MoE shape mismatch\n");
        std::abort();
    }
    if (g_wgp_7h < 48 || !h7h_coresident(g_occ_ship288, 288)) {
        launch_moe_nvfp4_ship_fused192(gate_up, down, x_norm, topk_ids, topk_w,
                                       tmp_act, out, stream, post_conv_w, post_sconv_h,
                                       post_res_add, scratch, raw_out);
        return;
    }
    if (scratch == nullptr || post_conv_w == nullptr || post_sconv_h == nullptr) {
        extern void launch_moe_nvfp4_decode(const SoaNVFP4Experts &, const SoaNVFP4Experts &,
                                            const float *, const int32_t *, const float *,
                                            float *, float *, hipStream_t,
                                            const float *, float *, const float *, float *);
        launch_moe_nvfp4_decode(gate_up, down, x_norm, topk_ids, topk_w,
                                tmp_act, out, stream, post_conv_w, post_sconv_h, post_res_add, raw_out);
        return;
    }
    SconvEpi epi;
    epi.conv_w  = post_conv_w;
    epi.hist    = post_sconv_h;
    epi.res     = post_res_add;
    epi.raw_out = raw_out;
    moe_nvfp4_ship_fused288_kernel<<<288, 256, 0, stream>>>(
        gate_up.qs, gate_up.scales, gate_up.scale2, topk_ids, topk_w, x_norm,
        down.qs, down.scales, down.scale2, tmp_act, out, epi, scratch);
}

void launch_moe_nvfp4_ship_fused192(
    const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * scratch, float * raw_out) {
    // 192-block 256-thread MoE kernel (121 VGPRs, 20.6 KB LDS) fits <=5 blocks/WGP:
    // requires >=40 WGPs (7900 GRE/XT/XTX/W7900); W7800 (35 WGPs) falls back cleanly.
    if (scratch == nullptr || post_conv_w == nullptr || post_sconv_h == nullptr || g_wgp_7h < 40 ||
        !h7h_coresident(g_occ_ship192, 192)) {
        extern void launch_moe_nvfp4_decode(const SoaNVFP4Experts &, const SoaNVFP4Experts &,
                                            const float *, const int32_t *, const float *,
                                            float *, float *, hipStream_t,
                                            const float *, float *, const float *, float *);
        launch_moe_nvfp4_decode(gate_up, down, x_norm, topk_ids, topk_w,
                                tmp_act, out, stream, post_conv_w, post_sconv_h, post_res_add, raw_out);
        return;
    }
    SconvEpi epi;
    epi.conv_w  = post_conv_w;
    epi.hist    = post_sconv_h;
    epi.res     = post_res_add;
    epi.raw_out = raw_out;
    moe_nvfp4_ship_fused192_kernel<<<192, 256, 0, stream>>>(
        gate_up.qs, gate_up.scales, gate_up.scale2, topk_ids, topk_w, x_norm,
        down.qs, down.scales, down.scale2, tmp_act, out, epi, scratch);
}

void launch_moe_q8_ship_fused(
    const SoaQ8Experts & gate_up, const SoaQ8Experts & down,
    const float * x_norm, const int32_t * topk_ids, const float * topk_w,
    float * tmp_act, float * out, hipStream_t stream,
    const float * post_conv_w, float * post_sconv_h, const float * post_res_add,
    float * scratch) {
    if (gate_up.rows != N_GATE_UP_EXP || gate_up.K != N_EMBD ||
        down.rows != N_EMBD || down.K != N_FF_EXP) {
        std::fprintf(stderr, "[kernels_step7h] q8 MoE shape mismatch\n");
        std::abort();
    }
    if (scratch == nullptr || post_conv_w == nullptr || post_sconv_h == nullptr || g_wgp_7h < 40 ||
        !h7h_coresident(g_occ_q8ship192, 192)) {
        launch_moe_q8_decode(gate_up, down, x_norm, topk_ids, topk_w,
                             tmp_act, out, stream, post_conv_w, post_sconv_h, post_res_add);
        return;
    }
    SconvEpi epi;
    epi.conv_w = post_conv_w;
    epi.hist   = post_sconv_h;
    epi.res    = post_res_add;
    moe_q8_ship_fused_kernel<<<192, 256, 0, stream>>>(
        gate_up.qs, gate_up.d, x_norm, topk_ids, topk_w,
        down.qs, down.d, tmp_act, out, epi, scratch);
}


void launch_mix_ffn_norm_router_topk_decode_v3(
    const float * res_query,
    const float * const * states, int n_states,
    const float * ffn_norm_w,
    const __half * router_w,
    const float * exp_probs_b,
    float * x_norm_out,
    int32_t * topk_ids_out,
    float * topk_w_out,
    float * router_scratch,
    hipStream_t stream) {
    if (n_states <= 0) {
        return;
    }
    if (n_states > 5 || router_scratch == nullptr) {
        launch_mix_ffn_norm_router_topk_decode(res_query, states, n_states, ffn_norm_w,
                                               router_w, exp_probs_b, x_norm_out,
                                               topk_ids_out, topk_w_out, router_scratch, stream);
        return;
    }
    HStatePtrs s = {};
    for (int i = 0; i < n_states && i < MAX_RES_BLOCKS; ++i) {
        s.p[i] = states[i];
    }
    mix_ffn_norm_router_topk_v3_kernel<<<128, 192, 0, stream>>>(
        res_query, s, n_states, ffn_norm_w, router_w, exp_probs_b,
        x_norm_out, topk_ids_out, topk_w_out, router_scratch);
}

void launch_mix_ctrl_ffn_norm_router_topk_decode_v3(
    const float * res_query,
    const float * const * states, int n_states,
    const float * msg,
    const __half * w_down,
    const __half * w_gain,
    const __half * w_forget,
    const float * ffn_norm_w,
    const __half * router_w,
    const float * exp_probs_b,
    float * out_mixed,
    float * out_gate,
    float * x_norm_out,
    int32_t * topk_ids_out,
    float * topk_w_out,
    float * router_scratch,
    hipStream_t stream) {
    if (n_states <= 0) {
        return;
    }
    if (n_states > 5 || router_scratch == nullptr) {
        launch_mix_ctrl_down_decode(res_query, states, n_states, msg, w_down,
                                    out_mixed, out_gate, stream);
        launch_ctrl_apply_ffn_norm_router_topk_decode(out_mixed, msg, out_gate,
                                                      w_gain, w_forget, ffn_norm_w,
                                                      router_w, exp_probs_b,
                                                      x_norm_out, topk_ids_out, topk_w_out,
                                                      router_scratch, stream);
        return;
    }
    HStatePtrs s = {};
    for (int i = 0; i < n_states && i < MAX_RES_BLOCKS; ++i) {
        s.p[i] = states[i];
    }
    mix_ctrl_ffn_norm_router_topk_v3_kernel<<<CV3_GRID, 288, 0, stream>>>(
        res_query, s, n_states, msg, w_down, w_gain, w_forget,
        ffn_norm_w, router_w, exp_probs_b, out_mixed, out_gate,
        x_norm_out, topk_ids_out, topk_w_out, router_scratch);
}


void launch_mix_rms_norm_2304_fast(
    const float * query, int n_q,
    const float * const * states, int n_states,
    const float * const * w_norms,
    float * out, float eps, hipStream_t stream) {
    if (n_states <= 0 || n_q <= 0) {
        return;
    }
    HStatePtrs s = {};
    for (int i = 0; i < n_states && i < MAX_RES_BLOCKS; ++i) {
        s.p[i] = states[i];
    }
    HNormPtrs n = {};
    for (int q = 0; q < n_q && q < 4; ++q) {
        n.p[q] = w_norms ? w_norms[q] : nullptr;
    }
#define LRN(M, Q) mix_rms_norm_2304_fast_kernel<M, Q><<<1, 192, 0, stream>>>(query, s, n, out, eps)
    if (n_q == 1) {
        switch (n_states) {
            case 2: LRN(2, 1); return;
            case 3: LRN(3, 1); return;
            case 4: LRN(4, 1); return;
            case 5: LRN(5, 1); return;
            default: break;
        }
    } else if (n_q == 3) {
        switch (n_states) {
            case 2: LRN(2, 3); return;
            case 3: LRN(3, 3); return;
            case 4: LRN(4, 3); return;
            case 5: LRN(5, 3); return;
            default: break;
        }
    }
#undef LRN
    launch_inkling_mix_rms_norm(query, n_q, states, n_states, w_norms, out, 1, eps, stream);
}

void step7h_prime_occupancy() {
    // Query device WGP count once outside capture so 288/192-block grid-barrier
    // MoE kernels automatically downgrade on smaller gfx1100 GPUs (7900 XT/GRE, W7800).
    int dev = 0, wgp = 48;
    if (hipGetDevice(&dev) == hipSuccess &&
        hipDeviceGetAttribute(&wgp, hipDeviceAttributeMultiprocessorCount, dev) == hipSuccess &&
        wgp > 0) {
        g_wgp_7h = wgp;
    } else {
        g_wgp_7h = 48;
    }
    // RDNA4: the wgp thresholds above assume an RDNA3 part; also require the grid to
    // actually be co-resident (blocks/CU x device attribute >= grid).
    int n = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *)moe_nvfp4_ship_fused288_kernel, 256, 0) == hipSuccess) g_occ_ship288 = n;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *)moe_nvfp4_ship_fused192_kernel, 256, 0) == hipSuccess) g_occ_ship192 = n;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *)moe_q8_ship_fused_kernel, 256, 0) == hipSuccess) g_occ_q8ship192 = n;
    if (std::getenv("NIB_DUMP_OCC") != nullptr) {
        std::fprintf(stderr, "[step7h] occ/blk: ship288=%d ship192=%d q8ship192=%d (attr=%d)\n",
                     g_occ_ship288, g_occ_ship192, g_occ_q8ship192, g_wgp_7h);
    }
}

} // namespace inkling
