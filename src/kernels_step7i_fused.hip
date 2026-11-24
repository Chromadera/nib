// ============================================================================
// kernels_step7i_fused.hip — Step 7I-1 super-fusions (gfx1100, Wave32).
//
// Self-contained: include/inkling_engine.h + include/sparse_global.h only.
// Every helper that must produce bit-identical results to the shipped engine
// kernels is re-implemented here with the same source expression sequence.
// See include/kernels_step7i_fused.h for the inventory and scratch slot map.
//
// RDNA3 co-residency rules obeyed by every cooperative barrier in this file:
//   blocks <= 96 * blocks_per_cu, all blocks resident, every counter resets
//   itself (producer flushes stores, consumer invalidates L0/L1 after the spin).
// ============================================================================
#include "kernels_step7i_fused.h"
#include "kernels_decode_fused.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

namespace inkling {
namespace {

constexpr int WARP_SIZE = 32;
constexpr int MAX_Q_MAX_S7I = 4;

// ---------------------------------------------------------------------------
// Harness scratch map (buf_mix_scratch_, 16384 floats, zeroed once).
// F1/F3 and F4 never run in the same launch, so their partial regions alias.
// ---------------------------------------------------------------------------
constexpr int S7I_DOT    = 0;      // [4][8][192] = 6144
constexpr int S7I_SSQ    = 6144;   // [8][192]    = 1536
constexpr int S7I_SSQO   = 7680;   // [4][192]    =  768
constexpr int S7I_AUX    = 8448;   // effw / inv / flags / counters
constexpr int S7I_BAR    = 8768;   // 5 barrier groups x 8 floats
constexpr int S7I_P      = 9216;   // router p   [128]
constexpr int S7I_SEL    = 9344;   // router sel [128]
constexpr int S7I_TOP    = 9472;   // topk arrival counter (+3 spare)
constexpr int S7I_CSSQ   = 9480;   // per-block ssq slots [96]

// A_v3 partials (aliases S7I_DOT/SSQ/SSQO/AUX ranges; different launches).
constexpr int A_SSQ0 = S7I_DOT + 0;     // [5][192] = 960
constexpr int A_SSQ1 = S7I_DOT + 960;   // [5][192] = 960
constexpr int A_DOT0 = S7I_DOT + 1920;  // [4][5][192] = 3840
constexpr int A_DOT1 = S7I_DOT + 5760;  // [3][5][192] = 2880
// (A_v3 scratch offsets 8640..8719 are unused by the shipped A path)

constexpr int S7I_C_GATE_DONE = S7I_AUX + 64;
constexpr int S7I_C_GATE_ACK  = S7I_AUX + 65;
constexpr int S7I_C_SSQ_DONE  = S7I_AUX + 66;
constexpr int S7I_C_SSQ_ACK   = S7I_AUX + 67;
constexpr int S7I_C_XN_DONE   = S7I_AUX + 68;
constexpr int S7I_C_XN_ACK    = S7I_AUX + 69;
constexpr int S7I_A_DOWN_DONE = S7I_AUX + 74;  // +74..76
constexpr int S7I_A_DOWN_ACK  = S7I_AUX + 78;  // +78..80
constexpr int S7I_APPLY_DONE  = S7I_AUX + 96;  // +96..98 (A apply ssq arrival)
constexpr int S7I_B_TOP       = S7I_TOP;

constexpr int S7I_BAR_WOB  = S7I_BAR + 0;
constexpr int S7I_BAR_MQKV = S7I_BAR + 8;
constexpr int S7I_BAR_A    = S7I_BAR + 16;  // 96-block A mix barrier
constexpr int S7I_BAR_AG   = S7I_BAR + 24;  // A -> GEMV (192 blocks)
constexpr int S7I_BAR_QKV  = S7I_BAR + 32;  // GEMV -> prep (192 blocks)
constexpr int S7I_BAR_A2   = S7I_BAR + 40;  // A ctrl-phase barrier (96 blocks)
constexpr int S7I_BAR_WOB2 = S7I_BAR + 48;  // distributed mix partials
constexpr int S7I_BAR_WOB3 = S7I_BAR + 56;  // distributed mix ssq-out
constexpr int S7I_BAR_WOB4 = S7I_BAR + 64;  // x_norm published
constexpr int S7I_BAR_MQKV2 = S7I_BAR + 72; // fused norm+qkv mix partials
constexpr int S7I_BAR_MQKV3 = S7I_BAR + 80; // fused norm+qkv mix ssq-out
constexpr int S7I_BAR_MQKV4 = S7I_BAR + 88; // fused norm+qkv x_norm published

// ---------------------------------------------------------------------------
// ABI-identical pointer bundles (same layout as the engine's StatePtrs etc.)
// ---------------------------------------------------------------------------
struct S7IStatePtrs  { const float * p[MAX_RES_BLOCKS]; };
struct S7IPtr3       { const float * p[3]; };
struct S7IHalfPtr3   { const __half * p[3]; };
struct S7IMutPtr3    { float * p[3]; };
struct S7INormPtr3   { const float * p[3]; };

typedef int v4i __attribute__((ext_vector_type(4)));

// ---------------------------------------------------------------------------
// gfx1100 memory ordering + self-resetting cooperative counters
// ---------------------------------------------------------------------------
static __device__ __forceinline__ void rdna3_flush_stores() {
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#endif
}

static __device__ __forceinline__ void rdna3_inv_l0_l1() {
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
    asm volatile("buffer_gl1_inv\n\tbuffer_gl0_inv" ::: "memory");
#endif
}

static __device__ __forceinline__ int scratch_add(float * s, int slot) {
    return __hip_atomic_fetch_add(reinterpret_cast<int32_t *>(s + slot), 1,
                                  __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
}
static __device__ __forceinline__ int scratch_load(float * s, int slot) {
    return __hip_atomic_load(reinterpret_cast<int32_t *>(s + slot),
                             __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
}
static __device__ __forceinline__ void scratch_store(float * s, int slot, int v) {
    __hip_atomic_store(reinterpret_cast<int32_t *>(s + slot), v,
                       __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
static __device__ __forceinline__ void spin_until(float * s, int slot, int target) {
    while (scratch_load(s, slot) < target) {
        __builtin_amdgcn_s_sleep(1);
    }
}

template <int SHARDS>
static __device__ __forceinline__ void s7i_grid_barrier(float * scratch, int base, int nblocks) {
    rdna3_flush_stores();
    __syncthreads();
    if (threadIdx.x == 0) {
        rdna3_flush_stores();
        (void)scratch_add(scratch, base + (blockIdx.x % SHARDS));
        const int target = (nblocks + SHARDS - 1) / SHARDS;
        #pragma unroll
        for (int i = 0; i < SHARDS; ++i) {
            spin_until(scratch, base + i, target);
        }
        const int prev = scratch_add(scratch, base + SHARDS);
        if (prev == nblocks - 1) {
            #pragma unroll
            for (int i = 0; i < SHARDS; ++i) {
                scratch_store(scratch, base + i, 0);
            }
            scratch_store(scratch, base + SHARDS, 0);
        }
    }
    __syncthreads();
    rdna3_inv_l0_l1();
}

// ---------------------------------------------------------------------------
// Exact helpers (identical source to kernels_core.hip / kernels_gemv_soa.hip /
// kernels_decode_fused.hip / kernels_sparse_global.hip).
// ---------------------------------------------------------------------------
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
    const __half2 h0 = h2_of((uint32_t)w.x);
    const __half2 h1 = h2_of((uint32_t)w.y);
    const __half2 h2 = h2_of((uint32_t)w.z);
    const __half2 h3 = h2_of((uint32_t)w.w);
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

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
    #pragma unroll
    for (int offset = width >> 1; offset > 0; offset >>= 1) {
        x += __shfl_xor(x, offset, width);
    }
    return x;
}

static __device__ __forceinline__ float q8_word_dot_mix(uint32_t w, const float4 x, float acc) {
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

static __device__ __forceinline__ v4i ld_nt_b128(const void * p) {
    return __builtin_nontemporal_load(reinterpret_cast<const v4i *>(p));
}

static __device__ __forceinline__ void wave32_fwht_128_float4(float4 & v, const int lane) {
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

static __device__ __forceinline__ int pack_i8x4(int a, int b, int c, int d) {
    return (a & 0xFF) | ((b & 0xFF) << 8) | ((c & 0xFF) << 16) | ((d & 0xFF) << 24);
}

static __device__ __forceinline__ uint32_t pack_u4x4_lo(int q0, int q1, int q2, int q3) {
    return ((uint32_t)(q0 & 0xF)) |
           ((uint32_t)(q1 & 0xF) << 8) |
           ((uint32_t)(q2 & 0xF) << 16) |
           ((uint32_t)(q3 & 0xF) << 24);
}

static __device__ __forceinline__ void store_kv_quantized(
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

static __device__ __forceinline__ void wave32_topk8_128(
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
            const float   ov = __shfl_xor(best_v, off, WARP_SIZE);
            const int32_t oi = __shfl_xor(best_i, off, WARP_SIZE);
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

struct SconvEpi {
    const float * conv_w;
    float *       hist;
    const float * res;
    float *       raw_out; // Step 8A: optional dump of the raw pre-conv x3
};

static __device__ __forceinline__ float sconv_post_res_one(const SconvEpi & e, int r, float x3) {
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
// Page min/max metadata (identical to kernels_sparse_global.hip)
// ---------------------------------------------------------------------------
static __device__ __forceinline__ float warp_max32(float x) {
    x = fmaxf(x, __shfl_xor(x, 16, 32));
    x = fmaxf(x, __shfl_xor(x,  8, 32));
    x = fmaxf(x, __shfl_xor(x,  4, 32));
    x = fmaxf(x, __shfl_xor(x,  2, 32));
    x = fmaxf(x, __shfl_xor(x,  1, 32));
    return x;
}

static __device__ __forceinline__ void load_k_row4(
        const KVCacheSlot & kv_slot, size_t head_slot, int lane, float4 & v) {
    if (kv_slot.kv_type == KV_Q8_0) {
        const float d = __half2float(__ldg(&kv_slot.k_d[head_slot * 4 + (size_t)(lane >> 3)]));
        const uint32_t w = *(const uint32_t *)(kv_slot.k_qs + head_slot * HEAD_DIM + (size_t)lane * 4);
        v.x = (float)(int8_t)( w        & 0xFF) * d;
        v.y = (float)(int8_t)((w >>  8) & 0xFF) * d;
        v.z = (float)(int8_t)((w >> 16) & 0xFF) * d;
        v.w = (float)(int8_t)((w >> 24) & 0xFF) * d;
    } else {
        const uint32_t w = *((const uint32_t *)(kv_slot.k_qs + head_slot * (HEAD_DIM / 2))
                             + (size_t)(lane >> 1));
        const uint32_t nibs = ((lane & 1) ? (w >> 4) : w) & 0x0F0F0F0Fu;
        const float q0 = (float)( nibs        & 0xF);
        const float q1 = (float)((nibs >>  8) & 0xF);
        const float q2 = (float)((nibs >> 16) & 0xF);
        const float q3 = (float)((nibs >> 24) & 0xF);
        if (kv_slot.kv_type == KV_Q4_1) {
            const __half2 dm = __ldg(&((const __half2 *)kv_slot.k_d)[head_slot * 4 + (size_t)(lane >> 3)]);
            const float d = __low2float(dm);
            const float m = __high2float(dm);
            v.x = fmaf(q0, d, m);
            v.y = fmaf(q1, d, m);
            v.z = fmaf(q2, d, m);
            v.w = fmaf(q3, d, m);
        } else {
            const float d = __half2float(__ldg(&kv_slot.k_d[head_slot * 4 + (size_t)(lane >> 3)]));
            v.x = (q0 - 8.0f) * d;
            v.y = (q1 - 8.0f) * d;
            v.z = (q2 - 8.0f) * d;
            v.w = (q3 - 8.0f) * d;
        }
    }
}

static __device__ __forceinline__ void quantize_page_minmax_store(
        const float lo[4], const float hi[4], int lane,
        int8_t * min_row, int8_t * max_row, __half * d_slot) {
    float amax = 0.0f;
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        amax = fmaxf(amax, fmaxf(fabsf(lo[i]), fabsf(hi[i])));
    }
    amax = warp_max32(amax);

    const float scale = fmaxf(amax * (1.0f / 127.0f),
                              5.9604644775390625e-8f);   // 2^-24
    const __half dh = __float2half_ru(scale);
    const float de = __half2float(dh);
    const float inv = 1.0f / de;

    int qmn[4], qmx[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        qmn[i] = (int)fminf(127.0f, fmaxf(-127.0f, rintf(lo[i] * inv)));
        qmx[i] = (int)fminf(127.0f, fmaxf(-127.0f, rintf(hi[i] * inv)));
    }
    const int32_t wmin = (int32_t)((qmn[0] & 0xFF) | ((qmn[1] & 0xFF) << 8) |
                                   ((qmn[2] & 0xFF) << 16) | ((qmn[3] & 0xFF) << 24));
    const int32_t wmax = (int32_t)((qmx[0] & 0xFF) | ((qmx[1] & 0xFF) << 8) |
                                   ((qmx[2] & 0xFF) << 16) | ((qmx[3] & 0xFF) << 24));
    ((int32_t *)min_row)[lane] = wmin;
    ((int32_t *)max_row)[lane] = wmax;
    if (lane == 0) {
        *d_slot = dh;
    }
}

static __device__ __forceinline__ void page_minmax_decode_one_head(
        const KVCacheSlot & kv_slot, const SparseGlobalPageMeta & meta,
        int h, int pos, int lane) {
    const int page       = pos / SPARSE_PAGE_SIZE;
    const int page_start = page * SPARSE_PAGE_SIZE;
    const int tok_end    = pos + 1;
    if (page >= meta.max_pages) {
        return;
    }
    const size_t head_base = (size_t)h * (size_t)kv_slot.cap;
    float mn[4]  = {  INFINITY,  INFINITY,  INFINITY,  INFINITY };
    float mx[4]  = { -INFINITY, -INFINITY, -INFINITY, -INFINITY };
    float sum[4] = {  0.0f,      0.0f,      0.0f,      0.0f     };
    for (int t = page_start; t < tok_end; ++t) {
        float4 v;
        load_k_row4(kv_slot, head_base + (size_t)t, lane, v);
        wave32_fwht_128_float4(v, lane);
        v.x *= KQ_SCALE; v.y *= KQ_SCALE; v.z *= KQ_SCALE; v.w *= KQ_SCALE;
        mn[0] = fminf(mn[0], v.x); mx[0] = fmaxf(mx[0], v.x); sum[0] += v.x;
        mn[1] = fminf(mn[1], v.y); mx[1] = fmaxf(mx[1], v.y); sum[1] += v.y;
        mn[2] = fminf(mn[2], v.z); mx[2] = fmaxf(mx[2], v.z); sum[2] += v.z;
        mn[3] = fminf(mn[3], v.w); mx[3] = fmaxf(mx[3], v.w); sum[3] += v.w;
    }
    const float inv_n = 1.0f / (float)(tok_end - page_start);
    float lo[4], hi[4];
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float mean = sum[i] * inv_n;
        const float rad  = 0.5f * (mx[i] - mn[i]);
        lo[i] = mean - rad;
        hi[i] = mean + rad;
    }
    const size_t page_off = ((size_t)h * (size_t)meta.max_pages + (size_t)page) * HEAD_DIM;
    quantize_page_minmax_store(lo, hi, lane,
        meta.k_min_q + page_off,
        meta.k_max_q + page_off,
        meta.k_mm_d + (size_t)h * (size_t)meta.max_pages + (size_t)page);
}

// ---------------------------------------------------------------------------
// SoA Q8_0 GEMV ring (exact copy of kernels_decode_fused.hip)
// ---------------------------------------------------------------------------
struct QkvSeg {
    const int8_t * qs;
    const __half * d;
    const float *  x;
    float *        y;
    int            rows;
    int            blk_end;
};

static __device__ __forceinline__ int lay_x(int i) { return (i >> 3) * 9 + (i & 7); }

template <int LPR, int NB, int PD>
struct GemvRing {
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

// ---------------------------------------------------------------------------
// Step 7J: `v_dot2_f32_f16` + `__half2` LDS twin of GemvRing (identical weight
// bytes, identical per-lane block order, identical reduction tree; only x is
// rounded to FP16, which is 240x more precise than the 4-bit weights).
// ---------------------------------------------------------------------------
constexpr int S7I_Q8H_STRIDE = 20;   // half2 per 32-weight block (16 + 4 pad)

static __device__ __forceinline__ int lay_xh(int i) {
    return (i >> 3) * S7I_Q8H_STRIDE + (i & 7) * 2;
}

static __device__ __forceinline__ float q8_word_dot_fdot2(uint32_t w, uint32_t xa, uint32_t xb, float acc) {
    const uint32_t u   = w ^ 0x80808080u;
    const __half2  b   = __builtin_bit_cast(__half2, 0x64806480u);
    const __half2  h01 = __hsub2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u)), b);
    const __half2  h23 = __hsub2(__builtin_bit_cast(__half2, __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u)), b);
    acc = __builtin_amdgcn_fdot2(h01, __builtin_bit_cast(__half2, xa), acc, false);
    acc = __builtin_amdgcn_fdot2(h23, __builtin_bit_cast(__half2, xb), acc, false);
    return acc;
}

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
            const uint4 * xb = reinterpret_cast<const uint4 *>(s_x + (sub + it * LPR + xo) * S7I_Q8H_STRIDE);
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

// Exact q8 GEMV body.  STAGE=true performs the original x staging from seg.x;
// STAGE=false assumes s_x was already filled in the lay_x() layout.  `active`
// lets a wider block (288 threads) run the body with its first 192 threads.
template <int NWARPS, int LPR, int NB, int PD, bool STAGE>
static __device__ __forceinline__ void q8_gemv_compute(
        QkvSeg s0, QkvSeg s1, QkvSeg s2, float4 * s_x, bool active) {
    constexpr int NT  = NWARPS * 32;
    constexpr int NX4 = (NB * 32) / 4;
    constexpr int PER = (NX4 + NT - 1) / NT;
    constexpr int RPB = NWARPS * (32 / LPR);

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

    const int wx   = threadIdx.x;
    const int warp = wx >> 5;
    const int lane = wx & 31;
    const int sub  = lane % LPR;
    const int rloc = lane / LPR;
    const int rofs = warp * (32 / LPR) + rloc;
    const int step = nblk * RPB;

    if (STAGE && active) {
        const float4 * __restrict__ x4 = reinterpret_cast<const float4 *>(sg.x);
        float4 xv[PER];
        #pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int idx = wx + i * NT;
            xv[i] = x4[idx < NX4 ? idx : NX4 - 1];
        }
        #pragma unroll
        for (int i = 0; i < PER; ++i) {
            const int idx = wx + i * NT;
            if (idx < NX4) {
                s_x[lay_x(idx)] = xv[i];
            }
        }
    }
    __syncthreads();
    if (!active) {
        return;
    }

    int row0 = lb * RPB;
    GemvRing<LPR, NB, PD> ring;
    {
        const int row = row0 + rofs;
        ring.prime(qs, dd, row < rows ? row : rows - 1, sub);
    }
    for (; row0 < rows; row0 += step) {
        const int  row      = row0 + rofs;
        const bool valid    = row < rows;
        const bool has_next = row0 + step < rows;
        const int  nrow     = row + step;
        float acc = ring.run(qs, dd, valid ? row : rows - 1, nrow < rows ? nrow : rows - 1,
                             has_next, s_x, sub);
        acc = warp_reduce_sum<LPR>(acc);
        if (sub == 0 && valid) {
            y[row] = acc;
        }
    }
}

// ---------------------------------------------------------------------------
// Exact qkvr prep body (identical to qkv_qkvr_prep_fused_decode_kernel),
// with the optional page-minmax refresh folded into the K-store epilogue.
// Runs after the GEMV grid barrier; resets the barrier at the end.
// ---------------------------------------------------------------------------
template <int GRID>
static __device__ __forceinline__ void qkvr_prep_body(
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
        bool do_minmax,
        SparseGlobalPageMeta sparse_meta,
        float * __restrict__ scratch,
        int bar_base) {
    constexpr int N_RT_BLOCKS = GRID - 40;
    constexpr int NT_PREP = 128;   // shipped prep thread count
    const int bid  = blockIdx.x;
    const int tid  = threadIdx.x;
    const int lane = tid & 31;

    __shared__ alignas(16) float s_r[N_HEAD][N_REL];
    __shared__ alignas(16) float s_b[8][N_REL + 1];

    if (bid < 24) {
        if (tid < 32) {
            const int h = bid;
            float4 v = ((const float4 *)(q_in + (size_t)h * HEAD_DIM))[lane];
            if (q_clip_w) {
                const float4 w = ((const float4 *)(q_clip_w + (size_t)h * HEAD_DIM))[lane];
                v.x *= w.x; v.y *= w.y; v.z *= w.z; v.w *= w.w;
            }
            float sumsq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
            sumsq = warp_reduce_sum<WARP_SIZE>(sumsq);
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
            sumsq = warp_reduce_sum<WARP_SIZE>(sumsq);

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
            if (!is_v && do_minmax && !kv_slot.is_swa && d_meta != nullptr &&
                sparse_meta.k_min_q != nullptr) {
                rdna3_flush_stores();
                rdna3_inv_l0_l1();
                page_minmax_decode_one_head(kv_slot, sparse_meta, h, (int)pos, lane);
            }
        }
    } else if (tid < NT_PREP) {
        const int rid   = bid - 40;
        const int d_lo  = (rid * N_REL_EXTENT) / N_RT_BLOCKS;
        const int d_hi  = ((rid + 1) * N_REL_EXTENT) / N_RT_BLOCKS;
        const int d_cnt = d_hi - d_lo;

        const float tv = (apply_tau && d_meta != nullptr) ? d_meta->tau : 1.0f;

        #pragma unroll
        for (int step = 0; step < 3; ++step) {
            const int h = step * 8 + (tid >> 4);
            const int k = tid & 15;
            float rv = r_in[(size_t)h * N_REL + k];
            if (r_clip_w) {
                rv *= r_clip_w[(size_t)h * N_REL + k];
            }
            const float sumsq = warp_reduce_sum<N_REL>(rv * rv);
            const float factor = fminf(1.0f, REL_CLIP / (sqrtf(sumsq) + RMS_EPS)) * tv;
            s_r[h][k] = rv * factor;
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

    // reset the barrier
    rdna3_flush_stores();
    __syncthreads();
    if (tid == 0) {
        const int prev = scratch_add(scratch, bar_base + 3);
        if (prev == GRID - 1) {
            #pragma unroll
            for (int i = 0; i < 3; ++i) {
                scratch_store(scratch, bar_base + i, 0);
            }
            scratch_store(scratch, bar_base + 3, 0);
        }
    }
}

// ============================================================================
// Kernel 1: wo + B (Pass-0 pre-MoE), 192 x 192
// ============================================================================
__launch_bounds__(288, 2)
static __global__ void wo_b_fused_kernel(
        const int8_t * __restrict__ w_qs, const __half * __restrict__ w_d,
        const float * __restrict__ w_x, float * __restrict__ w_y,
        SconvEpi epi,
        const float * __restrict__ res_query,
        S7IStatePtrs states, int n_states,
        const float * __restrict__ ffn_norm_w,
        const __half * __restrict__ router_w,
        const float * __restrict__ exp_probs_b,
        float * __restrict__ x_norm_out,
        int32_t * __restrict__ topk_ids_out,
        float * __restrict__ topk_w_out,
        float * __restrict__ scratch) {
    constexpr int N_WARPS = 6;
    constexpr int NT      = 192;
    constexpr int MAX_M   = 5;

    const int bid     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    __shared__ float4 s_x[(N_Q / 32) * 9];
    __shared__ float4 s_xn[N_EMBD / 4];
    __shared__ float  s_row[12];
    __shared__ float  smem_sumsq[MAX_M][N_WARPS];
    __shared__ float  smem_dot[MAX_M][N_WARPS];
    __shared__ float  smem_effw[MAX_M];
    __shared__ float  smem_out_ssq[N_WARPS];
    __shared__ float  s_inv;
    __shared__ float  lds_p[N_EXPERT];
    __shared__ float  lds_sel[N_EXPERT];
    __shared__ bool   s_last;

    // ------------------------------ wo ----------------------------------
    {
        constexpr int NB = N_Q / 32;
        constexpr int NTG = 288;
        constexpr int NX4 = N_Q / 4;
        constexpr int PER = (NX4 + NTG - 1) / NTG;
        {
            const float4 * __restrict__ x4 = reinterpret_cast<const float4 *>(w_x);
            float4 xv[PER];
            #pragma unroll
            for (int i = 0; i < PER; ++i) {
                const int idx = tid + i * NTG;
                xv[i] = x4[idx < NX4 ? idx : NX4 - 1];
            }
            #pragma unroll
            for (int i = 0; i < PER; ++i) {
                const int idx = tid + i * NTG;
                if (idx < NX4) {
                    s_x[lay_x(idx)] = xv[i];
                }
            }
            __syncthreads();
        }
        const int rows[12] = {
            4 * bid + 0, 4 * bid + 1, 4 * bid + 2, 4 * bid + 3,
            4 * bid + 768, 4 * bid + 769, 4 * bid + 770, 4 * bid + 771,
            4 * bid + 1536, 4 * bid + 1537, 4 * bid + 1538, 4 * bid + 1539
        };
        const int nrows_w = (warp_id < 3) ? 2 : 1;
        const int base    = (warp_id < 3) ? (2 * warp_id) : (6 + (warp_id - 3));
        for (int rr = 0; rr < nrows_w; ++rr) {
            const int ri  = base + rr;
            const int row = rows[ri];
            GemvRing<32, NB, 3> ring;
            ring.prime(w_qs, w_d, row, lane);
            float acc = ring.run(w_qs, w_d, row, row, false, s_x, lane);
            acc = warp_reduce_sum<32>(acc);
            if (lane == 0) {
                const float v = sconv_post_res_one(epi, row, acc);
                w_y[row] = v;
                s_row[ri] = v;
            }
        }
        __syncthreads();
    }

    s7i_grid_barrier<3>(scratch, S7I_BAR_WOB, 192);

    // ---- exact inkling_mix_rms_norm_2304_kernel<5,1>, DISTRIBUTED --------
    // Block b owns the 12 channels of virtual mix thread b, i.e. exactly the
    // channels its wo rows just produced.  Partials -> grid barrier -> the
    // identical 6-warp tree (per-row lanes) in every block -> effw -> mixed ->
    // grid barrier -> inv -> buf_x_norm_ -> grid barrier -> router.
    __shared__ float4 s_mv[3];
    {
        const float4 * qp = (const float4 *)(res_query);
        const float4 q0 = qp[bid + 0 * NT];
        const float4 q1 = qp[bid + 1 * NT];
        const float4 q2 = qp[bid + 2 * NT];
        // one lane per state: all state partials issue in parallel
        if (warp_id == 0 && lane < n_states) {
            const int m = lane;
            const float4 * sp = (const float4 *)(states.p[m]);
            const float4 v0 = sp[bid + 0 * NT];
            const float4 v1 = sp[bid + 1 * NT];
            const float4 v2 = sp[bid + 2 * NT];
            const float ssq = (v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w)
                            + (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w)
                            + (v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w);
            const float dot = (v0.x * q0.x + v0.y * q0.y + v0.z * q0.z + v0.w * q0.w)
                            + (v1.x * q1.x + v1.y * q1.y + v1.z * q1.z + v1.w * q1.w)
                            + (v2.x * q2.x + v2.y * q2.y + v2.z * q2.z + v2.w * q2.w);
            scratch[S7I_SSQ + m * 192 + bid] = ssq;
            scratch[S7I_DOT + m * 192 + bid] = dot;
        }
    }

    s7i_grid_barrier<3>(scratch, S7I_BAR_WOB2, 192);

    // ---- exact 6-warp tree over the 192 per-thread partials --------------
    {
        if (warp_id < 6) {
            float pv[2 * MAX_M];
            #pragma unroll
            for (int r = 0; r < MAX_M; ++r) {
                pv[r]         = (r < n_states) ? scratch[S7I_SSQ + r * 192 + warp_id * 32 + lane] : 0.0f;
                pv[MAX_M + r] = (r < n_states) ? scratch[S7I_DOT + r * 192 + warp_id * 32 + lane] : 0.0f;
            }
            #pragma unroll
            for (int r = 0; r < MAX_M; ++r) {
                if (r < n_states) {
                    const float p0 = warp_reduce_sum<WARP_SIZE>(pv[r]);
                    const float p1 = warp_reduce_sum<WARP_SIZE>(pv[MAX_M + r]);
                    if (lane == 0) {
                        smem_sumsq[r][warp_id] = p0;
                        smem_dot[r][warp_id]   = p1;
                    }
                }
            }
        }
        __syncthreads();
        if (tid == 0) {
            constexpr float inv_n    = 1.0f / (float)N_EMBD;
            const float     inv_sq_n = rsqrtf((float)N_EMBD);
            float inv_rms[MAX_M];
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_states) {
                    float total = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < N_WARPS; ++w) {
                        total += smem_sumsq[m][w];
                    }
                    inv_rms[m] = rsqrtf(total * inv_n + RMS_EPS);
                }
            }
            float sc[MAX_M];
            float max_s = -INFINITY;
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_states) {
                    float total_d = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < N_WARPS; ++w) {
                        total_d += smem_dot[m][w];
                    }
                    sc[m] = total_d * inv_rms[m] * inv_sq_n;
                    max_s = fmaxf(max_s, sc[m]);
                }
            }
            // b_v2 reference: 8-lane butterfly tree over exp(sc - max)
            float e8[8];
            for (int m = 0; m < 8; ++m) {
                e8[m] = (m < n_states) ? expf(sc[m] - max_s) : 0.0f;
            }
            const float sum_e = ((e8[0] + e8[1]) + (e8[2] + e8[3])) +
                                ((e8[4] + e8[5]) + (e8[6] + e8[7]));
            const float inv_e = 1.0f / sum_e;
            for (int m = 0; m < n_states; ++m) {
                smem_effw[m] = e8[m] * inv_e * inv_rms[m];
            }
        }
        __syncthreads();
    }

    if (tid == 0) {
        float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 o1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 o2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_states) {
                const float4 * sp = (const float4 *)(states.p[m]);
                const float4 v0 = sp[bid + 0 * NT];
                const float4 v1 = sp[bid + 1 * NT];
                const float4 v2 = sp[bid + 2 * NT];
                const float w = smem_effw[m];
                o0.x += w * v0.x; o0.y += w * v0.y; o0.z += w * v0.z; o0.w += w * v0.w;
                o1.x += w * v1.x; o1.y += w * v1.y; o1.z += w * v1.z; o1.w += w * v1.w;
                o2.x += w * v2.x; o2.y += w * v2.y; o2.z += w * v2.z; o2.w += w * v2.w;
            }
        }
        const float ssq_o = (o0.x * o0.x + o0.y * o0.y + o0.z * o0.z + o0.w * o0.w)
                          + (o1.x * o1.x + o1.y * o1.y + o1.z * o1.z + o1.w * o1.w)
                          + (o2.x * o2.x + o2.y * o2.y + o2.z * o2.z + o2.w * o2.w);
        scratch[S7I_SSQO + bid] = ssq_o;
        s_mv[0] = o0; s_mv[1] = o1; s_mv[2] = o2;
    }
    __syncthreads();

    s7i_grid_barrier<3>(scratch, S7I_BAR_WOB3, 192);

    {
        float p = 0.0f;
        if (warp_id < 6) {
            p = scratch[S7I_SSQO + warp_id * 32 + lane];
            p = warp_reduce_sum<WARP_SIZE>(p);
            if (lane == 0) smem_out_ssq[warp_id] = p;
        }
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            #pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                tot += smem_out_ssq[w];
            }
            s_inv = rsqrtf(tot * (1.0f / (float)N_EMBD) + RMS_EPS);
        }
        __syncthreads();
        if (tid == 0) {
            const float inv = s_inv;
            const float4 w0 = ffn_norm_w ? ((const float4 *)ffn_norm_w)[bid + 0 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float4 w1 = ffn_norm_w ? ((const float4 *)ffn_norm_w)[bid + 1 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            const float4 w2 = ffn_norm_w ? ((const float4 *)ffn_norm_w)[bid + 2 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
            float4 o0 = s_mv[0], o1 = s_mv[1], o2 = s_mv[2];
            o0.x *= inv * w0.x; o0.y *= inv * w0.y; o0.z *= inv * w0.z; o0.w *= inv * w0.w;
            o1.x *= inv * w1.x; o1.y *= inv * w1.y; o1.z *= inv * w1.z; o1.w *= inv * w1.w;
            o2.x *= inv * w2.x; o2.y *= inv * w2.y; o2.z *= inv * w2.z; o2.w *= inv * w2.w;
            float4 * out4 = (float4 *)x_norm_out;
            out4[bid + 0 * NT] = o0;
            out4[bid + 1 * NT] = o1;
            out4[bid + 2 * NT] = o2;
        }
    }

    // x_norm is complete -> stage it for the router.
    s7i_grid_barrier<3>(scratch, S7I_BAR_WOB4, 192);
    if (bid < 32 && tid < NT) {
        #pragma unroll
        for (int k = 0; k < 3; ++k) {
            s_xn[tid + k * NT] = ((const float4 *)x_norm_out)[tid + k * NT];
        }
    }
    __syncthreads();

    // ------------------------- router (32 blocks) ------------------------
    if (bid < 32) {
        const int e0 = bid * 4;
        v4i rw[9];
        if (warp_id < 4) {
            const __half * rbase = router_w + (size_t)(e0 + warp_id) * N_EMBD;
            #pragma unroll
            for (int s = 0; s < 9; ++s) {
                rw[s] = ld_nt_b128(rbase + (size_t)(lane + s * WARP_SIZE) * 8);
            }
        }
        __builtin_amdgcn_sched_barrier(0);

        if (warp_id < 4) {
            float acc = 0.0f;
            #pragma unroll
            for (int s = 0; s < 9; ++s) {
                const int i8 = lane + s * WARP_SIZE;
                acc = dot_h8_f8v(rw[s], s_xn[i8 * 2 + 0], s_xn[i8 * 2 + 1], acc);
            }
            const float sum = warp_reduce_sum<WARP_SIZE>(acc);
            if (lane == 0) {
                const int e = e0 + warp_id;
                const float p   = 1.0f / (1.0f + expf(-sum));
                const float sel = exp_probs_b ? (p + exp_probs_b[e]) : p;
                __hip_atomic_store(&scratch[S7I_P + e],   p,   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_store(&scratch[S7I_SEL + e], sel, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
        rdna3_flush_stores();
        __syncthreads();

        if (tid == 0) {
            rdna3_flush_stores();
            const int prev = scratch_add(scratch, S7I_B_TOP);
            s_last = (prev == 31);
            if (s_last) {
                scratch_store(scratch, S7I_B_TOP, 0);
            }
        }
        __syncthreads();

        if (s_last && warp_id == 0) {
            rdna3_inv_l0_l1();
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int idx = lane + j * 32;
                lds_p[idx]   = __hip_atomic_load(&scratch[S7I_P + idx],   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                lds_sel[idx] = __hip_atomic_load(&scratch[S7I_SEL + idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
            __atomic_signal_fence(__ATOMIC_SEQ_CST);
            wave32_topk8_128(lds_sel, lds_p, topk_ids_out, topk_w_out, lane);
        }
    }
}

// ============================================================================
// Kernel 2: wo + C (Pass-1/2 pre-MoE), 192 x 288.  Blocks 0..31 run the exact
// C-v2 body after the wo; the others exit.
// ============================================================================
template <bool FDOT2>
__launch_bounds__(288, 2)
static __global__ void wo_c_fused_kernel(
        const int8_t * __restrict__ w_qs, const __half * __restrict__ w_d,
        const float * __restrict__ w_x, float * __restrict__ w_y,
        SconvEpi epi,
        const float * __restrict__ res_query,
        S7IStatePtrs states, int n_states,
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
    constexpr int MAX_M   = 5;

    const int bid     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int grp     = tid >> 3;
    const int k8      = tid & 7;

    constexpr int SX_BYTES = FDOT2 ? (N_Q / 32) * S7I_Q8H_STRIDE * 4 : (N_Q / 32) * 9 * 16;
    __shared__ __align__(16) unsigned char s_x_raw[SX_BYTES];
    __half2 * s_xh = reinterpret_cast<__half2 *>(s_x_raw);
    float4  * s_xf = reinterpret_cast<float4 *>(s_x_raw);
    __shared__ float4 s_xn[N_EMBD / 4];
    __shared__ float  lds_p[N_EXPERT];
    __shared__ float  lds_sel[N_EXPERT];
    __shared__ bool   s_last;

    // ------------------------------ wo ----------------------------------
    {
        constexpr int NB = N_Q / 32;
        constexpr int NT = N_WARPS * 32;
        constexpr int NX4 = N_Q / 4;
        constexpr int PER = (NX4 + NT - 1) / NT;
        {
            const float4 * __restrict__ x4 = reinterpret_cast<const float4 *>(w_x);
            float4 xv[PER];
            #pragma unroll
            for (int i = 0; i < PER; ++i) {
                const int idx = tid + i * NT;
                xv[i] = x4[idx < NX4 ? idx : NX4 - 1];
            }
            #pragma unroll
            for (int i = 0; i < PER; ++i) {
                const int idx = tid + i * NT;
                if (idx < NX4) {
                    if constexpr (FDOT2) {
                        const int base = lay_xh(idx);
                        s_xh[base]     = __floats2half2_rn(xv[i].x, xv[i].y);
                        s_xh[base + 1] = __floats2half2_rn(xv[i].z, xv[i].w);
                    } else {
                        s_xf[lay_x(idx)] = xv[i];
                    }
                }
            }
            __syncthreads();
        }
        const int rows[12] = {
            4 * bid + 0, 4 * bid + 1, 4 * bid + 2, 4 * bid + 3,
            4 * bid + 768, 4 * bid + 769, 4 * bid + 770, 4 * bid + 771,
            4 * bid + 1536, 4 * bid + 1537, 4 * bid + 1538, 4 * bid + 1539
        };
        const int nrows_w = (warp_id < 3) ? 2 : 1;
        const int base    = (warp_id < 3) ? (2 * warp_id) : (6 + (warp_id - 3));
        for (int rr = 0; rr < nrows_w; ++rr) {
            const int row = rows[base + rr];
            float acc;
            if constexpr (FDOT2) {
                GemvRingH<32, NB, 3> ring;
                ring.prime(w_qs, w_d, row, lane);
                acc = ring.run(w_qs, w_d, row, row, false, s_xh, lane);
            } else {
                GemvRing<32, NB, 3> ring;
                ring.prime(w_qs, w_d, row, lane);
                acc = ring.run(w_qs, w_d, row, row, false, s_xf, lane);
            }
            acc = warp_reduce_sum<32>(acc);
            if (lane == 0) {
                w_y[row] = sconv_post_res_one(epi, row, acc);
            }
        }
    }

    s7i_grid_barrier<3>(scratch, S7I_BAR_WOB, 192);
    if (bid >= 32) {
        return;
    }

    // ================= exact C-v2 body (mix_ctrl_ffn_norm_router_topk) ====
    {
        const float4 rq0 = ((const float4 *)res_query)[tid * 2 + 0];
        const float4 rq1 = ((const float4 *)res_query)[tid * 2 + 1];

        float4 s_vec0[MAX_M];
        float4 s_vec1[MAX_M];

        __shared__ float smem_ssq[MAX_M][N_WARPS];
        __shared__ float smem_dot[MAX_M][N_WARPS];
        __shared__ float s_effw[MAX_M];

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
                const float ws = warp_reduce_sum<WARP_SIZE>(ssq);
                const float wd = warp_reduce_sum<WARP_SIZE>(dot);
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

        float4 c0 = {0.0f, 0.0f, 0.0f, 0.0f};
        float4 c1 = {0.0f, 0.0f, 0.0f, 0.0f};
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < n_states) {
                const float w = s_effw[m];
                c0.x += w * s_vec0[m].x; c0.y += w * s_vec0[m].y; c0.z += w * s_vec0[m].z; c0.w += w * s_vec0[m].w;
                c1.x += w * s_vec1[m].x; c1.y += w * s_vec1[m].y; c1.z += w * s_vec1[m].z; c1.w += w * s_vec1[m].w;
            }
        }
        if (bid == 0) {
            ((float4 *)out_mixed)[tid * 2 + 0] = c0;
            ((float4 *)out_mixed)[tid * 2 + 1] = c1;
        }

        const float4 m0 = ((const float4 *)msg)[tid * 2 + 0];
        const float4 m1 = ((const float4 *)msg)[tid * 2 + 1];

        float ssq_c = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z + c0.w * c0.w +
                      c1.x * c1.x + c1.y * c1.y + c1.z * c1.z + c1.w * c1.w;
        float ssq_m = m0.x * m0.x + m0.y * m0.y + m0.z * m0.z + m0.w * m0.w +
                      m1.x * m1.x + m1.y * m1.y + m1.z * m1.z + m1.w * m1.w;

        float dot0[2], dot1[2];
        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            const int r = bid * 2 + k;
            const __half * wr0 = w_down + (size_t)r * (2 * N_EMBD);
            const __half * wr1 = wr0 + N_EMBD;
            const uint4 w0 = ((const uint4 *)wr0)[tid];
            const uint4 w1 = ((const uint4 *)wr1)[tid];
            dot0[k] = dot_h8_f8(w0, c0, c1, 0.0f);
            dot1[k] = dot_h8_f8(w1, m0, m1, 0.0f);
        }

        ssq_c = warp_reduce_sum<WARP_SIZE>(ssq_c);
        ssq_m = warp_reduce_sum<WARP_SIZE>(ssq_m);
        dot0[0] = warp_reduce_sum<WARP_SIZE>(dot0[0]);
        dot0[1] = warp_reduce_sum<WARP_SIZE>(dot0[1]);
        dot1[0] = warp_reduce_sum<WARP_SIZE>(dot1[0]);
        dot1[1] = warp_reduce_sum<WARP_SIZE>(dot1[1]);

        __shared__ float smem_g[6][N_WARPS];
        if (lane == 0) {
            smem_g[0][warp_id] = ssq_c;
            smem_g[1][warp_id] = ssq_m;
            smem_g[2][warp_id] = dot0[0];
            smem_g[3][warp_id] = dot0[1];
            smem_g[4][warp_id] = dot1[0];
            smem_g[5][warp_id] = dot1[1];
        }
        __syncthreads();

        if (tid == 0) {
            float tc = 0.0f, tm = 0.0f, td00 = 0.0f, td01 = 0.0f, td10 = 0.0f, td11 = 0.0f;
            #pragma unroll
            for (int w = 0; w < N_WARPS; ++w) {
                tc   += smem_g[0][w];
                tm   += smem_g[1][w];
                td00 += smem_g[2][w];
                td01 += smem_g[3][w];
                td10 += smem_g[4][w];
                td11 += smem_g[5][w];
            }
            const float ic = rsqrtf(tc * (1.0f / (float)N_EMBD) + RMS_EPS);
            const float im = rsqrtf(tm * (1.0f / (float)N_EMBD) + RMS_EPS);
            out_gate[bid * 2 + 0] = td00 * ic + td10 * im;
            out_gate[bid * 2 + 1] = td01 * ic + td11 * im;
        }
        rdna3_flush_stores();
        __syncthreads();
        if (tid == 0) {
            (void)scratch_add(scratch, S7I_C_GATE_DONE);
        }

        v4i wg[2], wf[2];
        {
            const int cb = bid * 72 + grp;
            #pragma unroll
            for (int s = 0; s < 2; ++s) {
                const int c = cb + s * 36;
                wg[s] = ld_nt_b128(w_gain + (size_t)c * LOOP_GATE_RANK + (size_t)k8 * 8);
                wf[s] = ld_nt_b128(w_forget + (size_t)c * LOOP_GATE_RANK + (size_t)k8 * 8);
            }
        }
        const int e0 = bid * 4;
        v4i rw[9];
        if (warp_id < 4) {
            const __half * rbase = router_w + (size_t)(e0 + warp_id) * N_EMBD;
            #pragma unroll
            for (int s = 0; s < 9; ++s) {
                rw[s] = ld_nt_b128(rbase + (size_t)(lane + s * WARP_SIZE) * 8);
            }
        }

        if (tid == 0) {
            spin_until(scratch, S7I_C_GATE_DONE, 32);
        }
        __syncthreads();
        rdna3_inv_l0_l1();
        if (tid == 0) {
            const int prev = scratch_add(scratch, S7I_C_GATE_ACK);
            if (prev == 31) {
                scratch_store(scratch, S7I_C_GATE_DONE, 0);
                scratch_store(scratch, S7I_C_GATE_ACK, 0);
            }
        }

        __shared__ alignas(16) float s_gate[LOOP_GATE_RANK];
        if (tid < LOOP_GATE_RANK) {
            s_gate[tid] = out_gate[tid];
        }
        __syncthreads();

        const float4 g0 = ((const float4 *)s_gate)[k8 * 2 + 0];
        const float4 g1 = ((const float4 *)s_gate)[k8 * 2 + 1];

        __shared__ float s_block_ssq[N_WARPS];
        float val0 = 0.0f, val1 = 0.0f;
        {
            float ssq = 0.0f;
            const int c_base = bid * 72 + grp;
            #pragma unroll
            for (int step = 0; step < 2; ++step) {
                const int c = c_base + step * 36;
                float dg = dot_h8_f8v(wg[step], g0, g1, 0.0f);
                float df = dot_h8_f8v(wf[step], g0, g1, 0.0f);
                dg = warp_reduce_sum<8>(dg);
                df = warp_reduce_sum<8>(df);
                if (k8 == 0) {
                    const float gain   = tanhf(dg);
                    const float forget = tanhf(df);
                    const float cv     = out_mixed[c];
                    const float mv     = msg[c];
                    const float val    = cv - forget * cv + gain * mv;
                    if (step == 0) val0 = val; else val1 = val;
                    ssq += val * val;
                }
            }
            ssq = warp_reduce_sum<WARP_SIZE>(ssq);
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
            scratch[S7I_CSSQ + bid] = blk;
            rdna3_flush_stores();
            (void)scratch_add(scratch, S7I_C_SSQ_DONE);
            spin_until(scratch, S7I_C_SSQ_DONE, 32);
        }
        __syncthreads();
        rdna3_inv_l0_l1();
        if (tid == 0) {
            const int prev = scratch_add(scratch, S7I_C_SSQ_ACK);
            if (prev == 31) {
                scratch_store(scratch, S7I_C_SSQ_DONE, 0);
                scratch_store(scratch, S7I_C_SSQ_ACK, 0);
            }
        }

        float tot_ssq;
        {
            const int idx = tid & 31;
            const float partial = __hip_atomic_load(&scratch[S7I_CSSQ + idx],
                                                    __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            tot_ssq = warp_reduce_sum<WARP_SIZE>(partial);
        }
        const float inv = rsqrtf(tot_ssq * (1.0f / (float)N_EMBD) + RMS_EPS);

        if (k8 == 0) {
            const int c_base = bid * 72 + grp;
            const float w0v = ffn_norm_w ? ffn_norm_w[c_base] : 1.0f;
            x_norm_out[c_base] = val0 * inv * w0v;
            const int c1v = c_base + 36;
            const float w1v = ffn_norm_w ? ffn_norm_w[c1v] : 1.0f;
            x_norm_out[c1v] = val1 * inv * w1v;
        }
        rdna3_flush_stores();
        __syncthreads();

        if (tid == 0) {
            (void)scratch_add(scratch, S7I_C_XN_DONE);
            spin_until(scratch, S7I_C_XN_DONE, 32);
        }
        __syncthreads();
        rdna3_inv_l0_l1();
        if (tid == 0) {
            const int prev = scratch_add(scratch, S7I_C_XN_ACK);
            if (prev == 31) {
                scratch_store(scratch, S7I_C_XN_DONE, 0);
                scratch_store(scratch, S7I_C_XN_ACK, 0);
            }
        }

        #pragma unroll
        for (int k = 0; k < 2; ++k) {
            s_xn[tid + k * 288] = ((const float4 *)x_norm_out)[tid + k * 288];
        }
        __syncthreads();
        __builtin_amdgcn_sched_barrier(0);

        if (warp_id < 4) {
            float acc = 0.0f;
            #pragma unroll
            for (int s = 0; s < 9; ++s) {
                const int i8 = lane + s * WARP_SIZE;
                acc = dot_h8_f8v(rw[s], s_xn[i8 * 2 + 0], s_xn[i8 * 2 + 1], acc);
            }
            const float sum = warp_reduce_sum<WARP_SIZE>(acc);
            if (lane == 0) {
                const int e = e0 + warp_id;
                const float p   = 1.0f / (1.0f + expf(-sum));
                const float sel = exp_probs_b ? (p + exp_probs_b[e]) : p;
                __hip_atomic_store(&scratch[S7I_P + e],   p,   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_store(&scratch[S7I_SEL + e], sel, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
        }
        rdna3_flush_stores();
        __syncthreads();

        if (tid == 0) {
            rdna3_flush_stores();
            const int prev = scratch_add(scratch, S7I_B_TOP);
            s_last = (prev == 31);
            if (s_last) {
                scratch_store(scratch, S7I_B_TOP, 0);
            }
        }
        __syncthreads();

        if (s_last && warp_id == 0) {
            rdna3_inv_l0_l1();
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int idx = lane + j * 32;
                lds_p[idx]   = __hip_atomic_load(&scratch[S7I_P + idx],   __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                lds_sel[idx] = __hip_atomic_load(&scratch[S7I_SEL + idx], __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            }
            __atomic_signal_fence(__ATOMIC_SEQ_CST);
            wave32_topk8_128(lds_sel, lds_p, topk_ids_out, topk_w_out, lane);
        }
    }
}

// ============================================================================
// Kernel 3: Pass-0 attention input norm + qkv_qkvr_prep (+ minmax), 192 x 192
// ============================================================================
template <int GRID>
__launch_bounds__(192, 2)
static __global__ void mix_rms_qkv_fused_kernel(
        // mix (n_states == 0 -> plain RMSNorm of mix_query)
        const float * __restrict__ mix_query, int n_q,
        S7IStatePtrs mix_states, int n_states,
        S7INormPtr3 mix_w_norms,
        float * __restrict__ mix_out,
        // projection segments
        QkvSeg s0, QkvSeg s1, QkvSeg s2,
        // prep
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
        bool do_minmax,
        SparseGlobalPageMeta sparse_meta,
        float * __restrict__ scratch) {
    constexpr int N_WARPS = 6;
    constexpr int NT      = 192;
    constexpr int MAX_M   = 5;
    constexpr int NB      = N_EMBD / 32;

    const int bid     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;

    __shared__ float4 s_x[NB * 9];
    __shared__ float  smem_sumsq[MAX_M][N_WARPS];
    __shared__ float  smem_dot[MAX_Q_MAX_S7I * MAX_M][N_WARPS];
    __shared__ float  smem_out_ssq[MAX_Q_MAX_S7I][N_WARPS];
    __shared__ float  s_inv[MAX_Q_MAX_S7I];
    __shared__ float  s_ssq18[18];

    // ---------------- distributed 192-CTA depth mix / RMSNorm ------------
    // Block b owns the 12 channels of virtual mix thread b (float4 b, b+192,
    // b+384).  It publishes the per-thread ssq/dot partials for EVERY query of
    // this layer (n_q <= 3), then every block runs the identical butterfly /
    // 6-warp tree, computes the full effw table, its own 12 mixed values for
    // all n_q queries, and the n_q ssq partials.  After the third barrier the
    // GEMV stages the finished buf_mix_out vectors from L2.
    __shared__ float4 s_mv[MAX_Q_MAX_S7I][3];
    __shared__ float  s_effw[MAX_Q_MAX_S7I][MAX_M];
    if (n_states > 0) {
        if (tid == 0) {
            for (int m = 0; m < n_states; ++m) {
                const float4 * sp = (const float4 *)(mix_states.p[m]);
                const float4 v0 = sp[bid + 0 * NT];
                const float4 v1 = sp[bid + 1 * NT];
                const float4 v2 = sp[bid + 2 * NT];
                const float ssq = (v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w)
                                + (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w)
                                + (v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w);
                scratch[S7I_SSQ + m * 192 + bid] = ssq;
                for (int qi = 0; qi < n_q; ++qi) {
                    const float4 * qp = (const float4 *)(mix_query + (size_t)qi * N_EMBD);
                    const float4 q0 = qp[bid + 0 * NT], q1 = qp[bid + 1 * NT], q2 = qp[bid + 2 * NT];
                    const float dot = (v0.x * q0.x + v0.y * q0.y + v0.z * q0.z + v0.w * q0.w)
                                    + (v1.x * q1.x + v1.y * q1.y + v1.z * q1.z + v1.w * q1.w)
                                    + (v2.x * q2.x + v2.y * q2.y + v2.z * q2.z + v2.w * q2.w);
                    scratch[S7I_DOT + ((qi * MAX_M + m) * 192) + bid] = dot;
                }
            }
        }

        s7i_grid_barrier<3>(scratch, S7I_BAR_MQKV2, GRID);

        // exact 6-warp tree over the 192 per-thread partials
        {
            const int n_red = n_states + n_q * n_states;
            for (int r = 0; r < n_red; ++r) {
                int base;
                int flat = 0;
                if (r < n_states) {
                    base = S7I_SSQ + r * 192;
                } else {
                    const int c  = r - n_states;
                    const int qi = c / n_states;
                    const int mm = c - qi * n_states;
                    flat = qi * MAX_M + mm;
                    base = S7I_DOT + flat * 192;
                }
                float p = scratch[base + warp_id * 32 + lane];
                p = warp_reduce_sum<WARP_SIZE>(p);
                if (lane == 0) {
                    if (r < n_states) smem_sumsq[r][warp_id] = p;
                    else              smem_dot[flat][warp_id] = p;
                }
            }
            __syncthreads();
        }
        if (tid == 0) {
            constexpr float inv_n    = 1.0f / (float)N_EMBD;
            const float     inv_sq_n = rsqrtf((float)N_EMBD);
            float inv_rms[MAX_M];
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < n_states) {
                    float total = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < N_WARPS; ++w) {
                        total += smem_sumsq[m][w];
                    }
                    inv_rms[m] = rsqrtf(total * inv_n + RMS_EPS);
                }
            }
            for (int qi = 0; qi < n_q; ++qi) {
                float sc[MAX_M];
                float max_s = -INFINITY;
                for (int m = 0; m < n_states; ++m) {
                    float total_d = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < N_WARPS; ++w) {
                        total_d += smem_dot[qi * MAX_M + m][w];
                    }
                    sc[m] = total_d * inv_rms[m] * inv_sq_n;
                    max_s = fmaxf(max_s, sc[m]);
                }
                // reference: 8-lane butterfly tree over exp(sc - max)
                float e8[8];
                for (int m = 0; m < 8; ++m) {
                    e8[m] = (m < n_states) ? expf(sc[m] - max_s) : 0.0f;
                }
                const float sum_e = ((e8[0] + e8[1]) + (e8[2] + e8[3])) +
                                    ((e8[4] + e8[5]) + (e8[6] + e8[7]));
                const float inv_e = 1.0f / sum_e;
                for (int m = 0; m < n_states; ++m) {
                    s_effw[qi][m] = e8[m] * inv_e * inv_rms[m];
                }
            }
        }
        __syncthreads();

        if (tid == 0) {
            for (int qi = 0; qi < n_q; ++qi) {
                float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 o1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                float4 o2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                for (int m = 0; m < n_states; ++m) {
                    const float4 * sp = (const float4 *)(mix_states.p[m]);
                    const float4 v0 = sp[bid + 0 * NT];
                    const float4 v1 = sp[bid + 1 * NT];
                    const float4 v2 = sp[bid + 2 * NT];
                    const float w = s_effw[qi][m];
                    o0.x = fmaf(w, v0.x, o0.x); o0.y = fmaf(w, v0.y, o0.y);
                    o0.z = fmaf(w, v0.z, o0.z); o0.w = fmaf(w, v0.w, o0.w);
                    o1.x = fmaf(w, v1.x, o1.x); o1.y = fmaf(w, v1.y, o1.y);
                    o1.z = fmaf(w, v1.z, o1.z); o1.w = fmaf(w, v1.w, o1.w);
                    o2.x = fmaf(w, v2.x, o2.x); o2.y = fmaf(w, v2.y, o2.y);
                    o2.z = fmaf(w, v2.z, o2.z); o2.w = fmaf(w, v2.w, o2.w);
                }
                const float ssq_o = (o0.x * o0.x + o0.y * o0.y + o0.z * o0.z + o0.w * o0.w)
                                  + (o1.x * o1.x + o1.y * o1.y + o1.z * o1.z + o1.w * o1.w)
                                  + (o2.x * o2.x + o2.y * o2.y + o2.z * o2.z + o2.w * o2.w);
                scratch[S7I_SSQO + qi * 192 + bid] = ssq_o;
                s_mv[qi][0] = o0; s_mv[qi][1] = o1; s_mv[qi][2] = o2;
            }
        }
        __syncthreads();

        s7i_grid_barrier<3>(scratch, S7I_BAR_MQKV3, GRID);

        {
            const int n_red = n_q;
            for (int qi = 0; qi < n_red; ++qi) {
                float p = scratch[S7I_SSQO + qi * 192 + warp_id * 32 + lane];
                p = warp_reduce_sum<WARP_SIZE>(p);
                if (lane == 0) smem_out_ssq[qi][warp_id] = p;
            }
            __syncthreads();
            if (tid == 0) {
                for (int qi = 0; qi < n_q; ++qi) {
                    float tot = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < N_WARPS; ++w) {
                        tot += smem_out_ssq[qi][w];
                    }
                    s_inv[qi] = rsqrtf(tot * (1.0f / (float)N_EMBD) + RMS_EPS);
                }
            }
            __syncthreads();
            if (tid == 0) {
                for (int qi = 0; qi < n_q; ++qi) {
                    const float inv = s_inv[qi];
                    const float * wn_q = mix_w_norms.p[qi];
                    const float4 w0 = wn_q ? ((const float4 *)wn_q)[bid + 0 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                    const float4 w1 = wn_q ? ((const float4 *)wn_q)[bid + 1 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                    const float4 w2 = wn_q ? ((const float4 *)wn_q)[bid + 2 * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                    float4 o0 = s_mv[qi][0], o1 = s_mv[qi][1], o2 = s_mv[qi][2];
                    o0.x *= inv * w0.x; o0.y *= inv * w0.y; o0.z *= inv * w0.z; o0.w *= inv * w0.w;
                    o1.x *= inv * w1.x; o1.y *= inv * w1.y; o1.z *= inv * w1.z; o1.w *= inv * w1.w;
                    o2.x *= inv * w2.x; o2.y *= inv * w2.y; o2.z *= inv * w2.z; o2.w *= inv * w2.w;
                    if (mix_out != nullptr) {
                        float4 * mo = (float4 *)(mix_out + (size_t)qi * N_EMBD);
                        mo[bid + 0 * NT] = o0;
                        mo[bid + 1 * NT] = o1;
                        mo[bid + 2 * NT] = o2;
                    }
                }
            }
        }
    } else {
        // Layer 0: distributed plain RMSNorm with the exact 18-warp tree.
        const float4 * x4 = (const float4 *)(mix_query);
        if (tid == 0) {
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                const float4 v = x4[bid + k * NT];
                const float ssq = v.x * v.x + v.y * v.y + v.z * v.z + v.w * v.w;
                scratch[S7I_SSQ + k * 192 + bid] = ssq;
            }
        }
        s7i_grid_barrier<3>(scratch, S7I_BAR_MQKV2, GRID);

        if (warp_id < 6) {
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                const int refw = k * 6 + warp_id;
                float p = scratch[S7I_SSQ + k * 192 + warp_id * 32 + lane];
                p = warp_reduce_sum<WARP_SIZE>(p);
                if (lane == 0) s_ssq18[refw] = p;
            }
        }
        __syncthreads();
        if (tid == 0) {
            float tot = 0.0f;
            #pragma unroll
            for (int i = 0; i < 18; ++i) {
                tot += s_ssq18[i];
            }
            s_inv[0] = rsqrtf(tot * (1.0f / (float)N_EMBD) + RMS_EPS);
        }
        __syncthreads();
        if (tid == 0) {
            const float inv = s_inv[0];
            const float * wn = mix_w_norms.p[0];
            #pragma unroll
            for (int k = 0; k < 3; ++k) {
                const float4 v = x4[bid + k * NT];
                const float4 w = wn ? ((const float4 *)wn)[bid + k * NT] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                float4 o;
                o.x = v.x * inv * w.x; o.y = v.y * inv * w.y;
                o.z = v.z * inv * w.z; o.w = v.w * inv * w.w;
                if (mix_out != nullptr) ((float4 *)mix_out)[bid + k * NT] = o;
            }
        }
    }

    // buf_norm_ is complete only once every block has published its channels.
    s7i_grid_barrier<3>(scratch, S7I_BAR_MQKV4, GRID);

    // ---------- exact shipped GEMV (stages buf_norm_ from L2) ------------
    q8_gemv_compute<N_WARPS, 8, NB, 3, true>(s0, s1, s2, s_x, true);

    // -------------------------- grid barrier ----------------------------
    rdna3_flush_stores();
    __syncthreads();
    if (tid == 0) {
        rdna3_flush_stores();
        (void)scratch_add(scratch, S7I_BAR_MQKV + (bid % 3));
    }
    __syncthreads();
    if (tid == 0) {
        const int target = GRID / 3;
        spin_until(scratch, S7I_BAR_MQKV + 0, target);
        spin_until(scratch, S7I_BAR_MQKV + 1, target);
        spin_until(scratch, S7I_BAR_MQKV + 2, target);
    }
    __syncthreads();
    rdna3_inv_l0_l1();

    qkvr_prep_body<GRID>(q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
                         k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
                         sconv_hist_kv, kv_slot, q_out, rel_table_out,
                         apply_tau, d_meta, do_minmax, sparse_meta,
                         scratch, S7I_BAR_MQKV);
}

// ============================================================================
// Kernel 4: A_v3 distributed mix_dual + ctrl_qkv3 + norm (96 x 288).
// `active` lets the A+QKV fused kernel run the same body on blocks 0..95.
// ============================================================================
template <int GRID>   // 96
static __device__ __forceinline__ void a_v3_body(
        const float * __restrict__ query0, int n_q0,
        const S7IStatePtrs & states0, int n_states0, float * __restrict__ out0,
        const float * __restrict__ query1, int n_q1,
        const S7IStatePtrs & states1, int n_states1, float * __restrict__ out1,
        const float * const cur_p[3], const float * __restrict__ msg_base,
        const S7IHalfPtr3 & w_down, const S7IHalfPtr3 & w_gain,
        const S7IHalfPtr3 & w_forget, const S7INormPtr3 & w_norm,
        float * __restrict__ gate, float * const out_norm[3],
        float eps, float * __restrict__ scratch) {
    constexpr int MAX_M   = 5;
    constexpr int N_WARPS = 9;
    const int bid     = blockIdx.x;
    const int tid     = threadIdx.x;
    const int warp_id = tid >> 5;
    const int lane    = tid & 31;
    const int grp     = tid >> 3;
    const int k8      = tid & 7;

    __shared__ float sm_dn[10][N_WARPS];
    __shared__ alignas(16) float s_gate[LOOP_GATE_RANK];
    __shared__ float smem_ssq[6];
    __shared__ bool  s_is_last;
    __shared__ float s_inv;
    __shared__ float s_red[56][6];
    __shared__ float s_effw0[4][MAX_M];
    __shared__ float s_effw1[4][MAX_M];

    // ---- M0: this block's two virtual mix threads (bid, bid+96), both branches
    // Lane 0 of warps 0..3: (branch, vt).
    if (lane == 0 && warp_id < 4) {
        const int br = warp_id >> 1;            // 0 = msg, 1 = attn
        const int vt = bid + (warp_id & 1) * GRID;
        const S7IStatePtrs & st = br ? states1 : states0;
        const int nst = br ? n_states1 : n_states0;
        const int nq  = br ? n_q1 : n_q0;
        const float * qsrc = br ? query1 : query0;
        #pragma unroll
        for (int m = 0; m < MAX_M; ++m) {
            if (m < nst) {
                const float4 * sp = (const float4 *)(st.p[m]);
                const float4 v0 = sp[vt + 0 * 192];
                const float4 v1 = sp[vt + 1 * 192];
                const float4 v2 = sp[vt + 2 * 192];
                float ssq = (v0.x * v0.x + v0.y * v0.y + v0.z * v0.z + v0.w * v0.w)
                          + (v1.x * v1.x + v1.y * v1.y + v1.z * v1.z + v1.w * v1.w)
                          + (v2.x * v2.x + v2.y * v2.y + v2.z * v2.z + v2.w * v2.w);
                if (br == 0) scratch[A_SSQ0 + m * 192 + vt] = ssq;
                else         scratch[A_SSQ1 + m * 192 + vt] = ssq;
                for (int qi = 0; qi < nq; ++qi) {
                    const float4 * qp = (const float4 *)(qsrc + (size_t)qi * N_EMBD);
                    const float4 q0 = qp[vt + 0 * 192];
                    const float4 q1 = qp[vt + 1 * 192];
                    const float4 q2 = qp[vt + 2 * 192];
                    float d = (v0.x * q0.x + v0.y * q0.y + v0.z * q0.z + v0.w * q0.w)
                            + (v1.x * q1.x + v1.y * q1.y + v1.z * q1.z + v1.w * q1.w)
                            + (v2.x * q2.x + v2.y * q2.y + v2.z * q2.z + v2.w * q2.w);
                    if (br == 0) scratch[A_DOT0 + ((qi * MAX_M + m) * 192) + vt] = d;
                    else         scratch[A_DOT1 + ((qi * MAX_M + m) * 192) + vt] = d;
                }
            }
        }
    }
    __syncthreads();

    // ---- M1: 96-block barrier so every partial is published -------------
    s7i_grid_barrier<3>(scratch, S7I_BAR_A, GRID);

    // ---- M2: exact 6-warp tree over the 192 partials, per reduction -----
    const int N_RED = n_states0 + n_states1 + n_q0 * n_states0 + n_q1 * n_states1;
    for (int r = 0; r < N_RED; ++r) {
        int base;
        if (r < n_states0)                       base = A_SSQ0 + r * 192;
        else if (r < n_states0 + n_states1)      base = A_SSQ1 + (r - n_states0) * 192;
        else if (r < n_states0 + n_states1 + n_q0 * n_states0)
                                                 base = A_DOT0 + (r - n_states0 - n_states1) * 192;
        else                                     base = A_DOT1 + (r - n_states0 - n_states1 - n_q0 * n_states0) * 192;
        float part = 0.0f;
        if (warp_id < 6) {
            part = scratch[base + warp_id * 32 + lane];
            part = warp_reduce_sum<WARP_SIZE>(part);
            if (lane == 0) s_red[r][warp_id] = part;
        }
    }
    __syncthreads();

    if (tid == 0 && n_states0 > 0) {
        constexpr float inv_n    = 1.0f / (float)N_EMBD;
        const float     inv_sq_n = rsqrtf((float)N_EMBD);
        // msg branch
        {
            float inv_rms[MAX_M];
            for (int m = 0; m < n_states0; ++m) {
                float tot = 0.0f;
                #pragma unroll
                for (int w = 0; w < 6; ++w) tot += s_red[m][w];
                inv_rms[m] = rsqrtf(tot * inv_n + RMS_EPS);
            }
            for (int qi = 0; qi < n_q0; ++qi) {
                float sc[MAX_M];
                float max_s = -INFINITY;
                for (int m = 0; m < n_states0; ++m) {
                    const int r = n_states0 + n_states1 + qi * n_states0 + m;
                    float td = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < 6; ++w) td += s_red[r][w];
                    sc[m] = td * inv_rms[m] * inv_sq_n;
                    max_s = fmaxf(max_s, sc[m]);
                }
                float ex[MAX_M];
                for (int m = 0; m < n_states0; ++m) ex[m] = expf(sc[m] - max_s);
                float e8[8];
                for (int m = 0; m < 8; ++m) e8[m] = (m < n_states0) ? ex[m] : 0.0f;
                const float sum_e = ((e8[0] + e8[1]) + (e8[2] + e8[3])) +
                                    ((e8[4] + e8[5]) + (e8[6] + e8[7]));
                const float inv_e = 1.0f / sum_e;
                for (int m = 0; m < n_states0; ++m) {
                    s_effw0[qi][m] = ex[m] * inv_e * inv_rms[m];
                }
            }
        }
        // attn branch
        {
            float inv_rms[MAX_M];
            for (int m = 0; m < n_states1; ++m) {
                float tot = 0.0f;
                const int r = n_states0 + m;
                #pragma unroll
                for (int w = 0; w < 6; ++w) tot += s_red[r][w];
                inv_rms[m] = rsqrtf(tot * inv_n + RMS_EPS);
            }
            for (int qi = 0; qi < n_q1; ++qi) {
                float sc[MAX_M];
                float max_s = -INFINITY;
                for (int m = 0; m < n_states1; ++m) {
                    const int r = n_states0 + n_states1 + n_q0 * n_states0 + qi * n_states1 + m;
                    float td = 0.0f;
                    #pragma unroll
                    for (int w = 0; w < 6; ++w) td += s_red[r][w];
                    sc[m] = td * inv_rms[m] * inv_sq_n;
                    max_s = fmaxf(max_s, sc[m]);
                }
                float ex[MAX_M];
                for (int m = 0; m < n_states1; ++m) ex[m] = expf(sc[m] - max_s);
                float e8[8];
                for (int m = 0; m < 8; ++m) e8[m] = (m < n_states1) ? ex[m] : 0.0f;
                const float sum_e = ((e8[0] + e8[1]) + (e8[2] + e8[3])) +
                                    ((e8[4] + e8[5]) + (e8[6] + e8[7]));
                const float inv_e = 1.0f / sum_e;
                for (int m = 0; m < n_states1; ++m) {
                    s_effw1[qi][m] = ex[m] * inv_e * inv_rms[m];
                }
            }
        }
    }
    __syncthreads();

    // ---- M3: mixed channels for this block's two virtual threads --------
    if (lane == 0 && warp_id < 4) {
        const int br = warp_id >> 1;
        const int vt = bid + (warp_id & 1) * GRID;
        const S7IStatePtrs & st = br ? states1 : states0;
        const int nst = br ? n_states1 : n_states0;
        const int nq  = br ? n_q1 : n_q0;
        float * dst = br ? out1 : out0;
        for (int qi = 0; qi < nq; ++qi) {
            float4 o0 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 o1 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 o2 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            #pragma unroll
            for (int m = 0; m < MAX_M; ++m) {
                if (m < nst) {
                    const float4 * sp = (const float4 *)(st.p[m]);
                    const float4 v0 = sp[vt + 0 * 192];
                    const float4 v1 = sp[vt + 1 * 192];
                    const float4 v2 = sp[vt + 2 * 192];
                    const float w = br ? s_effw1[qi][m] : s_effw0[qi][m];
                    o0.x = fmaf(w, v0.x, o0.x); o0.y = fmaf(w, v0.y, o0.y);
                    o0.z = fmaf(w, v0.z, o0.z); o0.w = fmaf(w, v0.w, o0.w);
                    o1.x = fmaf(w, v1.x, o1.x); o1.y = fmaf(w, v1.y, o1.y);
                    o1.z = fmaf(w, v1.z, o1.z); o1.w = fmaf(w, v1.w, o1.w);
                    o2.x = fmaf(w, v2.x, o2.x); o2.y = fmaf(w, v2.y, o2.y);
                    o2.z = fmaf(w, v2.z, o2.z); o2.w = fmaf(w, v2.w, o2.w);
                }
            }
            float4 * out4 = (float4 *)(dst + (size_t)qi * N_EMBD);
            out4[vt + 0 * 192] = o0;
            out4[vt + 1 * 192] = o1;
            out4[vt + 2 * 192] = o2;
        }
    }

    // ---- M4: barrier so the ctrl phase sees the full mixes --------------
    s7i_grid_barrier<3>(scratch, S7I_BAR_A2, GRID);

    // ======================= ctrl_down / apply ==========================
    const int ja      = bid / 32;
    const int tile    = bid % 32;
    const bool do_down = (tile < 16);
    const int jd      = ja;
    const int rd0     = tile * 4;

    v4i wg[3], wf[3];
    {
        const __half * wgp = w_gain.p[ja];
        const __half * wfp = w_forget.p[ja];
        const int cb = tile * 72 + grp;
        #pragma unroll
        for (int s = 0; s < 3; ++s) {
            const int c = cb + s * 24;
            wg[s] = ld_nt_b128(wgp + (size_t)c * LOOP_GATE_RANK + (size_t)k8 * 8);
            wf[s] = ld_nt_b128(wfp + (size_t)c * LOOP_GATE_RANK + (size_t)k8 * 8);
        }
    }

    // ---- ctrl_down (exact 16 tiles x 4 ranks) --------------------------
    if (do_down) {
        const float  * ct = cur_p[jd];
        const float  * mt = msg_base + (size_t)jd * N_EMBD;
        const float4 c0 = ((const float4 *)ct)[tid * 2 + 0];
        const float4 c1 = ((const float4 *)ct)[tid * 2 + 1];
        const float4 m0 = ((const float4 *)mt)[tid * 2 + 0];
        const float4 m1 = ((const float4 *)mt)[tid * 2 + 1];

        float ssq_c = c0.x * c0.x + c0.y * c0.y + c0.z * c0.z + c0.w * c0.w +
                      c1.x * c1.x + c1.y * c1.y + c1.z * c1.z + c1.w * c1.w;
        float ssq_m = m0.x * m0.x + m0.y * m0.y + m0.z * m0.z + m0.w * m0.w +
                      m1.x * m1.x + m1.y * m1.y + m1.z * m1.z + m1.w * m1.w;

        float dot0[4], dot1[4];
        const __half * wd = w_down.p[jd];
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            const size_t base = (size_t)(rd0 + k) * (2 * N_EMBD) + (size_t)tid * 8;
            const v4i p0 = ld_nt_b128(wd + base);
            const v4i p1 = ld_nt_b128(wd + base + N_EMBD);
            dot0[k] = dot_h8_f8v(p0, c0, c1, 0.0f);
            dot1[k] = dot_h8_f8v(p1, m0, m1, 0.0f);
        }
        ssq_c = warp_reduce_sum<WARP_SIZE>(ssq_c);
        ssq_m = warp_reduce_sum<WARP_SIZE>(ssq_m);
        #pragma unroll
        for (int k = 0; k < 4; ++k) {
            dot0[k] = warp_reduce_sum<WARP_SIZE>(dot0[k]);
            dot1[k] = warp_reduce_sum<WARP_SIZE>(dot1[k]);
        }
        if (lane == 0) {
            sm_dn[0][warp_id] = ssq_c;
            sm_dn[1][warp_id] = ssq_m;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                sm_dn[2 + k][warp_id] = dot0[k];
                sm_dn[6 + k][warp_id] = dot1[k];
            }
        }
        __syncthreads();

        if (tid == 0) {
            float tc = 0.0f, tm = 0.0f;
            float td0[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            float td1[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int w = 0; w < 9; ++w) {
                tc += sm_dn[0][w];
                tm += sm_dn[1][w];
                #pragma unroll
                for (int k = 0; k < 4; ++k) {
                    td0[k] += sm_dn[2 + k][w];
                    td1[k] += sm_dn[6 + k][w];
                }
            }
            const float ic = rsqrtf(tc * (1.0f / (float)N_EMBD) + RMS_EPS);
            const float im = rsqrtf(tm * (1.0f / (float)N_EMBD) + RMS_EPS);
            float * g_out = gate + (size_t)jd * LOOP_GATE_RANK + rd0;
            #pragma unroll
            for (int k = 0; k < 4; ++k) {
                g_out[k] = td0[k] * ic + td1[k] * im;
            }
            rdna3_flush_stores();
            (void)scratch_add(scratch, S7I_A_DOWN_DONE + jd);
        }
    }

    // ---- ctrl_apply + RMSNorm (exact 32 tiles) -------------------------
    if (tid == 0) {
        spin_until(scratch, S7I_A_DOWN_DONE + ja, 16);
    }
    __syncthreads();
    rdna3_inv_l0_l1();
    if (tid == 0) {
        const int prev = scratch_add(scratch, S7I_A_DOWN_ACK + ja);
        if (prev == 31) {
            scratch_store(scratch, S7I_A_DOWN_DONE + ja, 0);
            scratch_store(scratch, S7I_A_DOWN_ACK + ja, 0);
        }
    }
    if (tid < LOOP_GATE_RANK) {
        s_gate[tid] = gate[(size_t)ja * LOOP_GATE_RANK + tid];
    }
    __syncthreads();

    const bool apply_active = (tid < 192);
    if (apply_active) {
        const float4 g0 = ((const float4 *)s_gate)[k8 * 2 + 0];
        const float4 g1 = ((const float4 *)s_gate)[k8 * 2 + 1];
        const float  * cur_j = cur_p[ja];
        const float  * msg_j = msg_base + (size_t)ja * N_EMBD;
        float        * out_j = out_norm[ja];

        float ssq = 0.0f;
        const int c_base = tile * 72 + grp;
        #pragma unroll
        for (int step = 0; step < 3; ++step) {
            const int c = c_base + step * 24;
            float dg = dot_h8_f8v(wg[step], g0, g1, 0.0f);
            float df = dot_h8_f8v(wf[step], g0, g1, 0.0f);
            dg = warp_reduce_sum<8>(dg);
            df = warp_reduce_sum<8>(df);
            if (k8 == 0) {
                const float gain   = tanhf(dg);
                const float forget = tanhf(df);
                const float cv     = cur_j[c];
                const float mv     = msg_j[c];
                const float val    = cv - forget * cv + gain * mv;
                out_j[c] = val;
                ssq += val * val;
            }
        }
        ssq = warp_reduce_sum<WARP_SIZE>(ssq);
        if (lane == 0) {
            smem_ssq[warp_id] = ssq;
        }
    }
    rdna3_flush_stores();
    __syncthreads();

    if (tid == 0) {
        float blk_ssq = 0.0f;
        #pragma unroll
        for (int w = 0; w < 6; ++w) {
            blk_ssq += smem_ssq[w];
        }
        scratch[S7I_CSSQ + ja * 32 + tile] = blk_ssq;
        rdna3_flush_stores();
        int32_t * done_ptr = reinterpret_cast<int32_t *>(scratch + S7I_APPLY_DONE + ja);
        const int prev = __hip_atomic_fetch_add(done_ptr, 1, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
        s_is_last = (prev == 31);
    }
    __syncthreads();

    if (s_is_last) {
        rdna3_inv_l0_l1();
        if (warp_id == 0) {
            const float my_ssq = scratch[S7I_CSSQ + ja * 32 + lane];
            const float tot_ssq = warp_reduce_sum<WARP_SIZE>(my_ssq);
            if (lane == 0) {
                int32_t * done_ptr = reinterpret_cast<int32_t *>(scratch + S7I_APPLY_DONE + ja);
                __hip_atomic_store(done_ptr, 0, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
                s_inv = rsqrtf(tot_ssq * (1.0f / (float)N_EMBD) + eps);
            }
        }
        __syncthreads();

        if (apply_active) {
            const float inv = s_inv;
            const float * wn_j = w_norm.p[ja];
            float * out_j = out_norm[ja];
            #pragma unroll
            for (int p = 0; p < 3; ++p) {
                const int idx4 = tid + p * 192;
                const float4 xv = ((const float4 *)out_j)[idx4];
                const float4 wv = wn_j ? ((const float4 *)wn_j)[idx4] : make_float4(1.0f, 1.0f, 1.0f, 1.0f);
                float4 ov;
                ov.x = xv.x * inv * wv.x;
                ov.y = xv.y * inv * wv.y;
                ov.z = xv.z * inv * wv.z;
                ov.w = xv.w * inv * wv.w;
                ((float4 *)out_j)[idx4] = ov;
            }
        }
    }
}

__launch_bounds__(288, 1)
static __global__ void a_v3_kernel(
        const float * __restrict__ query0, int n_q0,
        S7IStatePtrs states0, int n_states0, float * __restrict__ out0,
        const float * __restrict__ query1, int n_q1,
        S7IStatePtrs states1, int n_states1, float * __restrict__ out1,
        S7IPtr3 cur, const float * __restrict__ msg_base,
        S7IHalfPtr3 w_down, S7IHalfPtr3 w_gain, S7IHalfPtr3 w_forget, S7INormPtr3 w_norm,
        float * __restrict__ gate, S7IMutPtr3 out_norm,
        float eps, float * __restrict__ scratch) {
    const float * cur_p[3] = { cur.p[0], cur.p[1], cur.p[2] };
    float * outn[3] = { out_norm.p[0], out_norm.p[1], out_norm.p[2] };
    a_v3_body<96>(query0, n_q0, states0, n_states0, out0,
                  query1, n_q1, states1, n_states1, out1,
                  cur_p, msg_base, w_down, w_gain, w_forget, w_norm,
                  gate, outn, eps, scratch);
}

// ============================================================================
// Kernel 4b: A_v3 + qkv_qkvr_prep (+ minmax) in ONE 192 x 288 launch.
// ============================================================================
__launch_bounds__(288, 2)
static __global__ void a_qkv_fused_kernel(
        // A
        const float * __restrict__ query0, int n_q0,
        S7IStatePtrs states0, int n_states0, float * __restrict__ out0,
        const float * __restrict__ query1, int n_q1,
        S7IStatePtrs states1, int n_states1, float * __restrict__ out1,
        S7IPtr3 cur, const float * __restrict__ msg_base,
        S7IHalfPtr3 w_down, S7IHalfPtr3 w_gain, S7IHalfPtr3 w_forget, S7INormPtr3 w_norm,
        float * __restrict__ gate, S7IMutPtr3 out_norm,
        float eps,
        // projection segments
        QkvSeg s0, QkvSeg s1, QkvSeg s2,
        // prep
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
        bool do_minmax,
        SparseGlobalPageMeta sparse_meta,
        float * __restrict__ scratch) {
    constexpr int GRID = 192;
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;

    if (bid < 96) {
        const float * cur_p[3] = { cur.p[0], cur.p[1], cur.p[2] };
        float * outn[3] = { out_norm.p[0], out_norm.p[1], out_norm.p[2] };
        a_v3_body<96>(query0, n_q0, states0, n_states0, out0,
                      query1, n_q1, states1, n_states1, out1,
                      cur_p, msg_base, w_down, w_gain, w_forget, w_norm,
                      gate, outn, eps, scratch);
    }

    // All 192 blocks: the GEMV may now read buf_norm_.
    s7i_grid_barrier<3>(scratch, S7I_BAR_AG, GRID);

    __shared__ float4 s_x[(N_EMBD / 32) * 9];
    q8_gemv_compute<6, 8, N_EMBD / 32, 3, true>(s0, s1, s2, s_x, tid < 192);

    rdna3_flush_stores();
    __syncthreads();
    if (tid == 0) {
        rdna3_flush_stores();
        (void)scratch_add(scratch, S7I_BAR_QKV + (bid % 3));
    }
    __syncthreads();
    if (tid == 0) {
        const int target = GRID / 3;
        spin_until(scratch, S7I_BAR_QKV + 0, target);
        spin_until(scratch, S7I_BAR_QKV + 1, target);
        spin_until(scratch, S7I_BAR_QKV + 2, target);
    }
    __syncthreads();
    rdna3_inv_l0_l1();

    qkvr_prep_body<GRID>(q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
                         k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
                         sconv_hist_kv, kv_slot, q_out, rel_table_out,
                         apply_tau, d_meta, do_minmax, sparse_meta,
                         scratch, S7I_BAR_QKV);
}

// ---------------------------------------------------------------------------
// Host-side helpers
// ---------------------------------------------------------------------------
static QkvSeg s7i_make_seg(const SoaQ8Matrix & W, const float * x, float * y) {
    QkvSeg s;
    s.qs = W.qs;
    s.d = W.d;
    s.x = x;
    s.y = y;
    s.rows = W.rows;
    s.blk_end = 0;
    return s;
}

template <int RPB>
static int s7i_partition(QkvSeg * s, int nseg, int cap) {
    long long groups[3] = {0, 0, 0};
    long long total = 0;
    for (int i = 0; i < nseg; ++i) {
        groups[i] = ((long long) s[i].rows + RPB - 1) / RPB;
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
            s[i].blk_end = end;
        } else {
            s[i].blk_end = 0x7FFFFFFF;
        }
    }
    return end;
}

static void s7i_states(const float * const * src, int n, S7IStatePtrs & dst) {
    dst = {};
    for (int i = 0; i < n && i < MAX_RES_BLOCKS; ++i) {
        dst.p[i] = src[i];
    }
}

int g_occ_wo_b = -1;
int g_occ_wo_c = -1;
int g_occ_wo_c_h2 = -1;
int g_occ_mixqkv = -1;
int g_occ_aqkv = -1;
int g_wgp_7i = 48;

// RDNA3 288-thread (9-wave) blocks can achieve at most 4 blocks/WGP in hardware
// (9 waves split 3+2+2+2 across 4 SIMDs), even when hipOccupancy reports 5-6.
// Require actual WGP capacity >= GRID so grid-wide barriers never deadlock on
// smaller gfx1100 GPUs (RX 7900 XT = 42 WGPs, 7900 GRE = 40, W7800 = 35).
inline bool s7i_coresident(int occ, int hw_cap_per_wgp, int grid) {
    if (occ < 2 || g_wgp_7i <= 0) return false;
    const int eff = (occ < hw_cap_per_wgp) ? occ : hw_cap_per_wgp;
    return eff * g_wgp_7i >= grid;
}

} // namespace

// ============================================================================
// Public API
// ============================================================================
void launch_wo_b_fused(
        const SoaQ8Matrix & W, const float * x,
        const float * conv_w, float * sconv_hist, const float * res, float * out,
        const float * res_query, const float * const * states, int n_states,
        const float * ffn_norm_w,
        const __half * router_w, const float * exp_probs_b,
        float * x_norm_out, int32_t * topk_ids_out, float * topk_w_out,
        float * mix_scratch, hipStream_t stream) {
    if (W.K != N_Q || conv_w == nullptr || sconv_hist == nullptr || res_query == nullptr) {
        launch_soa_q8_gemv_wo_v2(W, x, conv_w, sconv_hist, res, out, stream);
        launch_mix_ffn_norm_router_topk_decode_v2(res_query, states, n_states, ffn_norm_w,
                                                  router_w, exp_probs_b, x_norm_out,
                                                  topk_ids_out, topk_w_out, mix_scratch, stream);
        return;
    }
    if (n_states <= 0 || n_states > 5 || mix_scratch == nullptr || !s7i_coresident(g_occ_wo_b, 4, 192)) {
        launch_soa_q8_gemv_wo_v2(W, x, conv_w, sconv_hist, res, out, stream);
        launch_mix_ffn_norm_router_topk_decode_v2(res_query, states, n_states, ffn_norm_w,
                                                  router_w, exp_probs_b, x_norm_out,
                                                  topk_ids_out, topk_w_out, mix_scratch, stream);
        return;
    }
    S7IStatePtrs st;
    s7i_states(states, n_states, st);
    SconvEpi epi = {conv_w, sconv_hist, res, nullptr};
    wo_b_fused_kernel<<<192, 288, 0, stream>>>(
        W.qs, W.d, x, out, epi,
        res_query, st, n_states, ffn_norm_w, router_w, exp_probs_b,
        x_norm_out, topk_ids_out, topk_w_out, mix_scratch);
}

void launch_wo_c_fused_fdot2(
        const SoaQ8Matrix & W, const float * x,
        const float * conv_w, float * sconv_hist, const float * res, float * out,
        const float * res_query, const float * const * states, int n_states,
        const float * msg,
        const __half * w_down, const __half * w_gain, const __half * w_forget,
        const float * ffn_norm_w,
        const __half * router_w, const float * exp_probs_b,
        float * out_mixed, float * out_gate,
        float * x_norm_out, int32_t * topk_ids_out, float * topk_w_out,
        float * mix_scratch, hipStream_t stream) {
    if (W.K != N_Q || conv_w == nullptr || sconv_hist == nullptr || res_query == nullptr ||
        n_states <= 0 || n_states > 5 || mix_scratch == nullptr || !s7i_coresident(g_occ_wo_c_h2, 4, 192)) {
        launch_soa_q8_gemv_wo_v2(W, x, conv_w, sconv_hist, res, out, stream);
        launch_mix_ctrl_ffn_norm_router_topk_decode_v2(
            res_query, states, n_states, msg, w_down, w_gain, w_forget,
            ffn_norm_w, router_w, exp_probs_b, out_mixed, out_gate,
            x_norm_out, topk_ids_out, topk_w_out, mix_scratch, stream);
        return;
    }
    S7IStatePtrs st;
    s7i_states(states, n_states, st);
    SconvEpi epi = {conv_w, sconv_hist, res, nullptr};
    wo_c_fused_kernel<true><<<192, 288, 0, stream>>>(
        W.qs, W.d, x, out, epi,
        res_query, st, n_states, msg, w_down, w_gain, w_forget,
        ffn_norm_w, router_w, exp_probs_b, out_mixed, out_gate,
        x_norm_out, topk_ids_out, topk_w_out, mix_scratch);
}

void launch_wo_c_fused(
        const SoaQ8Matrix & W, const float * x,
        const float * conv_w, float * sconv_hist, const float * res, float * out,
        const float * res_query, const float * const * states, int n_states,
        const float * msg,
        const __half * w_down, const __half * w_gain, const __half * w_forget,
        const float * ffn_norm_w,
        const __half * router_w, const float * exp_probs_b,
        float * out_mixed, float * out_gate,
        float * x_norm_out, int32_t * topk_ids_out, float * topk_w_out,
        float * mix_scratch, hipStream_t stream, float * raw_out) {
    if (W.K != N_Q || conv_w == nullptr || sconv_hist == nullptr || res_query == nullptr ||
        n_states <= 0 || n_states > 5 || mix_scratch == nullptr || !s7i_coresident(g_occ_wo_c, 4, 192)) {
        launch_soa_q8_gemv_wo_v2(W, x, conv_w, sconv_hist, res, out, stream, raw_out);
        launch_mix_ctrl_ffn_norm_router_topk_decode_v2(
            res_query, states, n_states, msg, w_down, w_gain, w_forget,
            ffn_norm_w, router_w, exp_probs_b, out_mixed, out_gate,
            x_norm_out, topk_ids_out, topk_w_out, mix_scratch, stream);
        return;
    }
    S7IStatePtrs st;
    s7i_states(states, n_states, st);
    SconvEpi epi = {conv_w, sconv_hist, res, raw_out};
    wo_c_fused_kernel<false><<<192, 288, 0, stream>>>(
        W.qs, W.d, x, out, epi,
        res_query, st, n_states, msg, w_down, w_gain, w_forget,
        ffn_norm_w, router_w, exp_probs_b, out_mixed, out_gate,
        x_norm_out, topk_ids_out, topk_w_out, mix_scratch);
}

void launch_mix_rms_qkv_qkvr_prep_fused(
        const float * mix_query, int n_q,
        const float * const * mix_states, int n_states,
        const float * const * mix_w_norms,
        float * mix_out,
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
        bool fuse_minmax,
        SparseGlobalPageMeta sparse_meta,
        float * mix_scratch,
        hipStream_t stream) {
    QkvSeg s[3];
    int nseg = 0;
    if (W0.rows > 0) s[nseg++] = s7i_make_seg(W0, x0, y0);
    if (W1.rows > 0) s[nseg++] = s7i_make_seg(W1, x1, y1);
    if (W2.rows > 0) s[nseg++] = s7i_make_seg(W2, x2, y2);
    const bool ok = (nseg > 0) && W0.K == N_EMBD &&
                    (W1.rows <= 0 || W1.K == N_EMBD) && (W2.rows <= 0 || W2.K == N_EMBD) &&
                    W0.rows <= 12288 && W1.rows <= 12288 && W2.rows <= 12288 &&
                    mix_scratch != nullptr && n_states <= 5 && n_q <= 3 &&
                    s7i_coresident(g_occ_mixqkv, 6, 192);
    if (!ok) {
        if (n_states > 0) {
            launch_inkling_mix_rms_norm(mix_query, n_q, mix_states, n_states, mix_w_norms,
                                        mix_out, 1, RMS_EPS, stream);
        } else {
            launch_rms_norm(mix_query, mix_w_norms ? mix_w_norms[0] : nullptr,
                            mix_out, 1, N_EMBD, RMS_EPS, stream);
        }
        launch_qkv_qkvr_prep_fused_decode_g(
            W0, x0, y0, W1, x1, y1, W2, x2, y2,
            q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
            k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
            sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta,
            mix_scratch, stream, 288);
        return;
    }
    constexpr int GRID = 192;
    constexpr int RPB = 6 * (32 / 8);   // 24 rows/block
    (void) s7i_partition<RPB>(s, nseg, GRID);
    S7IStatePtrs st;
    s7i_states(mix_states, n_states > 0 ? n_states : 0, st);
    S7INormPtr3 wn;
    for (int i = 0; i < 3; ++i) wn.p[i] = mix_w_norms ? mix_w_norms[i] : nullptr;
    mix_rms_qkv_fused_kernel<GRID><<<GRID, 192, 0, stream>>>(
        mix_query, n_q, st, n_states, wn, mix_out,
        s[0], s[1], s[2],
        q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
        k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
        sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta,
        fuse_minmax, sparse_meta, mix_scratch);
}

void launch_a_v3_decode(
        const float * query0, int n_q0, const float * const * states0, int n_states0, float * out0,
        const float * query1, int n_q1, const float * const * states1, int n_states1, float * out1,
        const __half * const w_down[3],
        const __half * const w_gain[3],
        const __half * const w_forget[3],
        const float * const w_norm[3],
        float * tmp_gate,
        float * const out_norm[3],
        float eps, hipStream_t stream, float * mix_scratch) {
    const float * cur[3] = {
        out1,
        out1 + (size_t)(n_q1 >= 3 ? 1 : 0) * N_EMBD,
        out1 + (size_t)(n_q1 >= 3 ? 2 : 0) * N_EMBD
    };
    if (n_states0 <= 0 || n_states0 > 5 || n_states1 > 5 || n_states0 > 5 || n_q0 > 4 ||
        n_q1 > 4 || mix_scratch == nullptr || query0 == nullptr || query1 == nullptr) {
        launch_mix_dual_ctrl_qkv3_norm_decode(
            query0, n_q0, states0, n_states0, out0,
            query1, n_q1, states1, n_states1, out1,
            w_down, w_gain, w_forget, w_norm, tmp_gate, out_norm, 1, eps, stream, mix_scratch);
        return;
    }
    S7IStatePtrs s0, s1;
    s7i_states(states0, n_states0, s0);
    s7i_states(states1, n_states1, s1);
    S7IPtr3 cur3;
    cur3.p[0] = cur[0]; cur3.p[1] = cur[1]; cur3.p[2] = cur[2];
    S7IHalfPtr3 wd3, wg3, wf3;
    S7INormPtr3 wn3;
    S7IMutPtr3 on3;
    for (int i = 0; i < 3; ++i) {
        wd3.p[i] = w_down[i];
        wg3.p[i] = w_gain[i];
        wf3.p[i] = w_forget[i];
        wn3.p[i] = w_norm ? w_norm[i] : nullptr;
        on3.p[i] = out_norm[i];
    }
    a_v3_kernel<<<96, 288, 0, stream>>>(
        query0, n_q0, s0, n_states0, out0,
        query1, n_q1, s1, n_states1, out1,
        cur3, out0, wd3, wg3, wf3, wn3, tmp_gate, on3, eps, mix_scratch);
}

void launch_a_qkv_qkvr_prep_fused(
        const float * query0, int n_q0, const float * const * states0, int n_states0, float * out0,
        const float * query1, int n_q1, const float * const * states1, int n_states1, float * out1,
        const __half * const w_down[3],
        const __half * const w_gain[3],
        const __half * const w_forget[3],
        const float * const w_norm[3],
        float * tmp_gate,
        float * const out_norm[3],
        float eps,
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
        bool fuse_minmax,
        SparseGlobalPageMeta sparse_meta,
        float * mix_scratch,
        hipStream_t stream) {
    const float * cur[3] = {
        out1,
        out1 + (size_t)(n_q1 >= 3 ? 1 : 0) * N_EMBD,
        out1 + (size_t)(n_q1 >= 3 ? 2 : 0) * N_EMBD
    };
    QkvSeg s[3];
    int nseg = 0;
    if (W0.rows > 0) s[nseg++] = s7i_make_seg(W0, x0, y0);
    if (W1.rows > 0) s[nseg++] = s7i_make_seg(W1, x1, y1);
    if (W2.rows > 0) s[nseg++] = s7i_make_seg(W2, x2, y2);
    const bool ok = n_states1 > 0 && n_states1 <= 5 && n_q0 <= 4 && n_q1 <= 4 &&
                    nseg > 0 && W0.K == N_EMBD &&
                    (W1.rows <= 0 || W1.K == N_EMBD) && (W2.rows <= 0 || W2.K == N_EMBD) &&
                    W0.rows <= 12288 && W1.rows <= 12288 && W2.rows <= 12288 &&
                    mix_scratch != nullptr && s7i_coresident(g_occ_aqkv, 4, 192);
    if (!ok) {
        launch_a_v3_decode(query0, n_q0, states0, n_states0, out0,
                           query1, n_q1, states1, n_states1, out1,
                           w_down, w_gain, w_forget, w_norm, tmp_gate, out_norm,
                           eps, stream, mix_scratch);
        launch_qkv_qkvr_prep_fused_decode_g(
            W0, x0, y0, W1, x1, y1, W2, x2, y2,
            q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
            k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
            sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta,
            mix_scratch, stream, 288);
        return;
    }
    constexpr int GRID = 192;
    constexpr int RPB = 6 * (32 / 8);
    (void) s7i_partition<RPB>(s, nseg, GRID);
    S7IStatePtrs s0, s1;
    s7i_states(states0, n_states0, s0);
    s7i_states(states1, n_states1, s1);
    S7IPtr3 cur3;
    cur3.p[0] = cur[0]; cur3.p[1] = cur[1]; cur3.p[2] = cur[2];
    S7IHalfPtr3 wd3, wg3, wf3;
    S7INormPtr3 wn3;
    S7IMutPtr3 on3;
    for (int i = 0; i < 3; ++i) {
        wd3.p[i] = w_down[i];
        wg3.p[i] = w_gain[i];
        wf3.p[i] = w_forget[i];
        wn3.p[i] = w_norm ? w_norm[i] : nullptr;
        on3.p[i] = out_norm[i];
    }
    a_qkv_fused_kernel<<<GRID, 288, 0, stream>>>(
        query0, n_q0, s0, n_states0, out0,
        query1, n_q1, s1, n_states1, out1,
        cur3, out0, wd3, wg3, wf3, wn3, tmp_gate, on3, eps,
        s[0], s[1], s[2],
        q_in, q_clip_w, r_in, r_clip_w, rel_bias_w,
        k_raw, k_conv_w, k_dyn, v_raw, v_conv_w, v_dyn,
        sconv_hist_kv, kv_slot, q_out, rel_table_out, apply_tau, d_meta,
        fuse_minmax, sparse_meta, mix_scratch);
}

void step7i_prime_occupancy() {
    // Every fused grid must be fully co-resident (in-kernel grid barriers).
    // Query both blocks/WGP and device WGP count once, outside capture, and
    // let the launchers fall back when a grid cannot be co-resident.
    int dev = 0, wgp = 48;
    if (hipGetDevice(&dev) == hipSuccess &&
        hipDeviceGetAttribute(&wgp, hipDeviceAttributeMultiprocessorCount, dev) == hipSuccess &&
        wgp > 0) {
        g_wgp_7i = wgp;
    } else {
        g_wgp_7i = 48;
    }
    g_occ_wo_b   = 1;
    g_occ_wo_c   = 1;
    g_occ_wo_c_h2 = 1;
    g_occ_mixqkv = 1;
    g_occ_aqkv   = 1;
    int n = 0;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *)wo_b_fused_kernel, 288, 0) == hipSuccess) g_occ_wo_b = n;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *)wo_c_fused_kernel<false>, 288, 0) == hipSuccess) g_occ_wo_c = n;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *)wo_c_fused_kernel<true>, 288, 0) == hipSuccess) g_occ_wo_c_h2 = n;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *)mix_rms_qkv_fused_kernel<192>, 192, 0) == hipSuccess) g_occ_mixqkv = n;
    if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *)a_qkv_fused_kernel, 288, 0) == hipSuccess) g_occ_aqkv = n;
    std::fprintf(stderr, "[step7i] occupancy blocks/WGP (wgp=%d): wo_b=%d wo_c=%d wo_c_h2=%d mix_rms_qkv=%d a_qkv=%d\n",
                 g_wgp_7i, g_occ_wo_b, g_occ_wo_c, g_occ_wo_c_h2, g_occ_mixqkv, g_occ_aqkv);
}

} // namespace inkling
