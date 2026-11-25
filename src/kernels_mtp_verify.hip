// Step 8A: target-side DSpark MTP support kernels (tap capture, raw pre-conv
// input recording, short-conv history save / exact rollback).  Wave32 HIP,
// gfx1100; zero spills; never launched unless MTP I/O is enabled.

#include "kernels_mtp_verify.h"

namespace inkling {

// float -> bfloat16, round-to-nearest-even (same value as __float2bfloat16_rn
// for the finite activations here, without pulling in the OCML-backed header).
static __device__ __forceinline__ uint16_t f32_to_bf16_rn_bits(float x) {
    uint32_t u = __float_as_uint(x);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t)(u >> 16);
}

// ---------------------------------------------------------------------------
// Tap capture: bf16 round of one [n_rows, N_EMBD] f32 slab into
// d_taps[(r * 4 + tap) * N_EMBD + c].
// ---------------------------------------------------------------------------
static __global__ void mtp_capture_tap_kernel(
        const float * __restrict__ src,
        uint16_t    * __restrict__ taps,
        int                        n_rows,
        int                        tap) {
    const int r = blockIdx.x;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (r >= n_rows || c >= N_EMBD) {
        return;
    }
    const float v = src[(size_t)r * N_EMBD + c];
    taps[(size_t)(r * 4 + tap) * N_EMBD + c] = f32_to_bf16_rn_bits(v);
}

void launch_mtp_capture_tap(const float * src, uint16_t * taps, int n_rows, int tap, hipStream_t stream) {
    if (n_rows <= 0 || src == nullptr || taps == nullptr || tap < 0 || tap > 3) {
        return;
    }
    const dim3 grid((unsigned)n_rows, (unsigned)((N_EMBD + 575) / 576));
    mtp_capture_tap_kernel<<<grid, 576, 0, stream>>>(src, taps, n_rows, tap);
}

// ---------------------------------------------------------------------------
// Raw pre-conv K/V pair of one row: dst[0..N_KV) = k_raw row, dst[N_KV..2N_KV) = v_raw row.
// ---------------------------------------------------------------------------
static __global__ void mtp_copy_raw_pair_kernel(
        const float * __restrict__ k_raw,
        const float * __restrict__ v_raw,
        float       * __restrict__ dst) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x; // 0..2*N_KV
    if (c >= 2 * N_KV) {
        return;
    }
    dst[c] = (c < N_KV) ? k_raw[c] : v_raw[c - N_KV];
}

// Step 8C: all m verify rows in ONE launch (grid.y = row).
static __global__ void mtp_copy_raw_pair_mrows_kernel(
        const float * __restrict__ k_raw, int k_stride,
        const float * __restrict__ v_raw, int v_stride,
        float       * __restrict__ dst, int dst_stride) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;   // 0..2*N_KV
    const int r = blockIdx.y;
    if (c >= 2 * N_KV) {
        return;
    }
    const float * k = k_raw + (size_t)r * k_stride;
    const float * v = v_raw + (size_t)r * v_stride;
    float * d = dst + (size_t)r * dst_stride;
    d[c] = (c < N_KV) ? k[c] : v[c - N_KV];
}

void launch_mtp_copy_raw_pair_mrows(const float * k_raw, int k_stride, const float * v_raw,
                                    int v_stride, float * dst, int dst_stride, int nrows,
                                    hipStream_t stream) {
    if (k_raw == nullptr || v_raw == nullptr || dst == nullptr || nrows <= 0) {
        return;
    }
    const dim3 grid((2 * N_KV + 255) / 256, (unsigned)nrows);
    mtp_copy_raw_pair_mrows_kernel<<<grid, 256, 0, stream>>>(k_raw, k_stride, v_raw, v_stride,
                                                            dst, dst_stride);
}

void launch_mtp_copy_raw_pair(const float * k_raw, int k_stride, const float * v_raw, int v_stride,
                              float * dst, hipStream_t stream) {
    if (k_raw == nullptr || v_raw == nullptr || dst == nullptr) {
        return;
    }
    (void)k_stride;  // verify rows are always n_tokens == 1 (contiguous row)
    (void)v_stride;
    mtp_copy_raw_pair_kernel<<<(2 * N_KV + 255) / 256, 256, 0, stream>>>(k_raw, v_raw, dst);
}

// ---------------------------------------------------------------------------
// Short-conv history: [N_LAYER_SLOTS][N_SCONV_CHANNELS][N_CONV_HIST] f32, where
// hs[0] is the OLDEST and hs[N_CONV_HIST-1] the NEWEST raw pre-conv input (see
// sconv_kv_store_q8_kernel / sconv_post_res_kernel in kernels_core.hip).
// ---------------------------------------------------------------------------
static __global__ void mtp_save_hist_kernel(
        float * const * __restrict__ hist_ptrs,
        float         * __restrict__ bak) {
    const int s = blockIdx.x;
    const float * hs = hist_ptrs[s];
    float * dst = bak + (size_t)s * N_SCONV_CHANNELS * N_CONV_HIST;
    const int n = N_SCONV_CHANNELS * N_CONV_HIST;
    for (int i = blockIdx.y * blockDim.x + threadIdx.x; i < n; i += gridDim.y * blockDim.x) {
        dst[i] = hs[i];
    }
}

void launch_mtp_save_hist(float * const * hist_ptrs, float * bak, hipStream_t stream) {
    if (hist_ptrs == nullptr || bak == nullptr) {
        return;
    }
    mtp_save_hist_kernel<<<dim3(N_LAYER_SLOTS, 16), 256, 0, stream>>>(hist_ptrs, bak);
}

// After `commit` verify rows the history of every channel c must hold the raw
// inputs of positions pos_start + commit - 3, -2, -1.  Inputs of positions
// >= pos_start are in `raw[slot][row][channel]`; the (up to two) older ones
// come from the pre-verify save.  A per-channel 3-element rewrite — no
// dependency on the rejected rows at all, hence bit-exact.
static __global__ void mtp_rebuild_hist_kernel(
        float * const * __restrict__ hist_ptrs,
        const float   * __restrict__ bak,
        const float   * __restrict__ raw,
        int                          commit) {
    const int s = blockIdx.x;
    float * hs = hist_ptrs[s];
    const float * saved = bak + (size_t)s * N_SCONV_CHANNELS * N_CONV_HIST;
    const float * rw = raw + (size_t)s * 8 * N_SCONV_CHANNELS; // MAX_VERIFY_TOKENS = 8
    for (int c = blockIdx.y * blockDim.x + threadIdx.x; c < N_SCONV_CHANNELS; c += gridDim.y * blockDim.x) {
        #pragma unroll
        for (int i = 0; i < N_CONV_HIST; ++i) {
            const int j = commit - N_CONV_HIST + i; // position offset from pos_start
            const float v = (j < 0) ? saved[(size_t)c * N_CONV_HIST + (j + N_CONV_HIST)]
                                    : rw[(size_t)j * N_SCONV_CHANNELS + c];
            hs[(size_t)c * N_CONV_HIST + i] = v;
        }
    }
}

void launch_mtp_rebuild_hist(float * const * hist_ptrs, const float * bak, const float * raw,
                             int commit, hipStream_t stream) {
    if (hist_ptrs == nullptr || bak == nullptr || raw == nullptr) {
        return;
    }
    if (commit < 1 || commit > 8) {
        return;
    }
    mtp_rebuild_hist_kernel<<<dim3(N_LAYER_SLOTS, 16), 256, 0, stream>>>(hist_ptrs, bak, raw, commit);
}

// ---------------------------------------------------------------------------
// Exact GPU top-256 + full-vocab (max_l, sum_exp) for lossless temperature>0
// speculative rejection sampling across all m verify rows in parallel.
// ---------------------------------------------------------------------------
static __device__ __forceinline__ uint32_t mtp_float_key(float v) {
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__launch_bounds__(256, 4)
static __global__ void mtp_verify_topk_kernel(
        const float * __restrict__ logits,      // [m, vocab_size] (already softcapped)
        int                        vocab_size,
        float                      inv_t,
        float       * __restrict__ out_vals,    // [m, MTP_VERIFY_TOPK]
        int32_t     * __restrict__ out_ids,     // [m, MTP_VERIFY_TOPK]
        float       * __restrict__ out_max_l,   // [m]
        float       * __restrict__ out_sum_exp) { // [m]
    const int r    = (int)blockIdx.x;
    const int tid  = (int)threadIdx.x;          // 0..255
    const int warp = tid >> 5;                  // 0..7
    const int lane = tid & 31;                  // 0..31

    const float * row = logits + (size_t)r * (size_t)vocab_size;

    __shared__ uint32_t s_whist[8][256];
    __shared__ uint32_t s_hist[256];
    __shared__ float    s_wred[8];
    __shared__ uint32_t s_prefix;
    __shared__ int      s_need;
    __shared__ int      s_gt_cnt;
    __shared__ int      s_eq_cnt;
    __shared__ float    s_row_max;

    // ---- Pass 0: row_max + top byte (bits 24..31) with per-warp histograms --
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        s_whist[w][tid] = 0u;
    }
    __syncthreads();

    float lmax = -INFINITY;
    for (int i = tid; i < vocab_size; i += 256) {
        const float v = row[i];
        lmax = fmaxf(lmax, v);
        const uint32_t k = mtp_float_key(v);
        atomicAdd(&s_whist[warp][k >> 24], 1u);
    }

    lmax = fmaxf(lmax, __shfl_xor(lmax, 16, 32));
    lmax = fmaxf(lmax, __shfl_xor(lmax,  8, 32));
    lmax = fmaxf(lmax, __shfl_xor(lmax,  4, 32));
    lmax = fmaxf(lmax, __shfl_xor(lmax,  2, 32));
    lmax = fmaxf(lmax, __shfl_xor(lmax,  1, 32));
    if (lane == 0) {
        s_wred[warp] = lmax;
    }
    __syncthreads();

    uint32_t hsum = 0u;
    #pragma unroll
    for (int w = 0; w < 8; ++w) {
        hsum += s_whist[w][tid];
    }
    s_hist[tid] = hsum;
    __syncthreads();

    if (tid == 0) {
        float rmax = s_wred[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) {
            rmax = fmaxf(rmax, s_wred[w]);
        }
        s_row_max = rmax;
        out_max_l[r] = rmax;

        int acc = 0;
        uint32_t chosen = 0u;
        for (int b = 255; b >= 0; --b) {
            const int c = (int)s_hist[b];
            if (acc + c >= MTP_VERIFY_TOPK) {
                chosen = (uint32_t)b;
                break;
            }
            acc += c;
        }
        s_prefix = chosen << 24;
        s_need   = MTP_VERIFY_TOPK - acc;
    }
    __syncthreads();

    // ---- Passes 1..3: remaining 3 radix bytes + full-vocab sum_exp on pass 1 -
    const float rmax = s_row_max;
    for (int pass = 1; pass < 4; ++pass) {
        const int      shift = 24 - pass * 8;
        const uint32_t mask  = 0xFFFFFFFFu << (shift + 8);
        const uint32_t pref  = s_prefix;

        s_hist[tid] = 0u;
        __syncthreads();

        float lsum = 0.0f;
        for (int i = tid; i < vocab_size; i += 256) {
            const float v = row[i];
            if (pass == 1) {
                lsum += expf((v - rmax) * inv_t);
            }
            const uint32_t k = mtp_float_key(v);
            if ((k & mask) == pref) {
                atomicAdd(&s_hist[(k >> shift) & 0xFFu], 1u);
            }
        }

        if (pass == 1) {
            lsum += __shfl_xor(lsum, 16, 32);
            lsum += __shfl_xor(lsum,  8, 32);
            lsum += __shfl_xor(lsum,  4, 32);
            lsum += __shfl_xor(lsum,  2, 32);
            lsum += __shfl_xor(lsum,  1, 32);
            if (lane == 0) {
                s_wred[warp] = lsum;
            }
        }
        __syncthreads();

        if (tid == 0) {
            if (pass == 1) {
                float rsum = 0.0f;
                #pragma unroll
                for (int w = 0; w < 8; ++w) {
                    rsum += s_wred[w];
                }
                out_sum_exp[r] = rsum;
            }
            const int need = s_need;
            int acc = 0;
            uint32_t chosen = 0u;
            for (int b = 255; b >= 0; --b) {
                const int c = (int)s_hist[b];
                if (acc + c >= need) {
                    chosen = (uint32_t)b;
                    break;
                }
                acc += c;
            }
            s_prefix = pref | (chosen << shift);
            s_need   = need - acc;
        }
        __syncthreads();
    }

    // ---- Collect exact top-MTP_VERIFY_TOPK (vals, ids) ----------------------
    if (tid == 0) {
        s_gt_cnt = 0;
        s_eq_cnt = 0;
    }
    __syncthreads();

    const uint32_t pivot   = s_prefix;
    const int      need_eq = s_need;
    const int      n_gt    = MTP_VERIFY_TOPK - need_eq;
    float   * dst_v = out_vals + (size_t)r * MTP_VERIFY_TOPK;
    int32_t * dst_i = out_ids  + (size_t)r * MTP_VERIFY_TOPK;

    for (int i = tid; i < vocab_size; i += 256) {
        const float v = row[i];
        const uint32_t k = mtp_float_key(v);
        if (k > pivot) {
            const int pos = atomicAdd(&s_gt_cnt, 1);
            if (pos < n_gt) {
                dst_v[pos] = v;
                dst_i[pos] = i;
            }
        } else if (k == pivot) {
            const int eq = atomicAdd(&s_eq_cnt, 1);
            if (eq < need_eq) {
                dst_v[n_gt + eq] = v;
                dst_i[n_gt + eq] = i;
            }
        }
    }
}

void launch_mtp_verify_topk(const float * logits, int m, int vocab_size, float inv_t,
                            float * out_vals, int32_t * out_ids,
                            float * out_max_l, float * out_sum_exp, hipStream_t stream) {
    if (logits == nullptr || out_vals == nullptr || out_ids == nullptr ||
        out_max_l == nullptr || out_sum_exp == nullptr ||
        m < 1 || m > 8 || vocab_size < MTP_VERIFY_TOPK) {
        return;
    }
    mtp_verify_topk_kernel<<<dim3((unsigned)m, 1, 1), dim3(256, 1, 1), 0, stream>>>(
        logits, vocab_size, inv_t, out_vals, out_ids, out_max_l, out_sum_exp);
}

// ===========================================================================
// Exact full-vocab GPU sampler (issue #4).  See kernels_mtp_verify.h.
// ===========================================================================

// Philox4x32-10 (Salmon et al., "Parallel random numbers: as easy as 1, 2, 3").
static __device__ __forceinline__ uint4 mtp_philox4x32_10(uint4 c, uint32_t k0, uint32_t k1) {
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        const uint32_t hi0 = __umulhi(0xD2511F53u, c.x);
        const uint32_t lo0 = 0xD2511F53u * c.x;
        const uint32_t hi1 = __umulhi(0xCD9E8D57u, c.z);
        const uint32_t lo1 = 0xCD9E8D57u * c.z;
        c = make_uint4(hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0);
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }
    return c;
}

// log of an Exp(1) variate for vocab entry x of row r.  w is uniform on
// (0, 1) with 2^-32 granularity near 0 (exact small values: the race is decided
// by the SMALL exponentials) and is clamped below 1 so E is never 0 or inf.
static __device__ __forceinline__ float mtp_log_exp1(uint32_t x, uint32_t r, uint32_t k0, uint32_t k1) {
    const uint4 h = mtp_philox4x32_10(make_uint4(x, r, 0u, 0u), k0, k1);
    float w = (float)(h.x | 1u) * 2.3283064365386963e-10f;   // (0, 1]
    w = fminf(w, 0.99999994f);                                // largest float < 1
    const float e = -log1pf(-w);                              // Exp(1), in [2.3e-10, 16.6]
    return logf(e);
}

// Larger key wins; equal keys -> smaller index (deterministic).
static __device__ __forceinline__ void mtp_best_merge(float & bk, int & bi, float ok, int oi) {
    if (ok > bk || (ok == bk && (unsigned)oi < (unsigned)bi)) {
        bk = ok;
        bi = oi;
    }
}

// K1: per-(row, chunk) max.
__launch_bounds__(256)
static __global__ void mtp_sample_max_kernel(const float * __restrict__ logits,
                                             int vocab_size, int row_stride,
                                             MtpSampleBatchParams params,
                                             float * __restrict__ part_max) {
    const int b = (int)blockIdx.x;
    const int r = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int chunk = (vocab_size + MTP_SAMPLE_BLOCKS - 1) / MTP_SAMPLE_BLOCKS;
    const int x0 = b * chunk;
    const int x1 = min(vocab_size, x0 + chunk);
    const int src_row = params.row_idx[r];
    const float * row = logits + (size_t)src_row * (size_t)row_stride;

    float mx = -INFINITY;
    for (int x = x0 + tid; x < x1; x += 256) {
        mx = fmaxf(mx, row[x]);
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        mx = fmaxf(mx, __shfl_xor(mx, off, 32));
    }
    __shared__ float s_m[8];
    if ((tid & 31) == 0) {
        s_m[tid >> 5] = mx;
    }
    __syncthreads();
    if (tid == 0) {
        float m = s_m[0];
        #pragma unroll
        for (int w = 1; w < 8; ++w) {
            m = fmaxf(m, s_m[w]);
        }
        part_max[r * MTP_SAMPLE_BLOCKS + b] = m;
    }
}

// K2: per-(row, chunk) Z partial + race winner over keep \ {excl}.
__launch_bounds__(256)
static __global__ void mtp_sample_race_kernel(const float * __restrict__ logits,
                                              int vocab_size, int row_stride,
                                              MtpSampleBatchParams params,
                                              const float * __restrict__ part_max,
                                              float * __restrict__ part_z,
                                              float * __restrict__ part_key,
                                              int32_t * __restrict__ part_idx) {
    const int b = (int)blockIdx.x;
    const int r = (int)blockIdx.y;
    const int tid = (int)threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int chunk = (vocab_size + MTP_SAMPLE_BLOCKS - 1) / MTP_SAMPLE_BLOCKS;
    const int x0 = b * chunk;
    const int x1 = min(vocab_size, x0 + chunk);
    const int src_row = params.row_idx[r];
    const float * row = logits + (size_t)src_row * (size_t)row_stride;

    __shared__ float s_gmax;
    if (tid < 32) {
        float m = part_max[r * MTP_SAMPLE_BLOCKS + lane];   // MTP_SAMPLE_BLOCKS == 32
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            m = fmaxf(m, __shfl_xor(m, off, 32));
        }
        if (tid == 0) {
            s_gmax = m;
        }
    }
    __syncthreads();
    const float    gmax  = s_gmax;
    const int      ex    = params.excl[r];
    const float    inv_t = params.inv_t[r];
    const float    min_p = params.min_p[r];
    const uint32_t k0    = params.k0[r];
    const uint32_t k1    = params.k1[r];
    const uint32_t salt  = params.salt[r];

    float z  = 0.0f;
    float bk = -INFINITY;
    int   bi = -1;
    for (int x = x0 + tid; x < x1; x += 256) {
        const float s = (row[x] - gmax) * inv_t;
        const float e = expf(s);
        if (e >= min_p) {                       // min_p <= 0: keep all
            z += e;
            if (x != ex) {
                mtp_best_merge(bk, bi, s - mtp_log_exp1((uint32_t)x, salt, k0, k1), x);
            }
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        z += __shfl_xor(z, off, 32);
        mtp_best_merge(bk, bi, __shfl_xor(bk, off, 32), __shfl_xor(bi, off, 32));
    }
    __shared__ float s_z[8];
    __shared__ float s_k[8];
    __shared__ int   s_i[8];
    if (lane == 0) {
        s_z[warp] = z;
        s_k[warp] = bk;
        s_i[warp] = bi;
    }
    __syncthreads();
    if (tid == 0) {
        float tz = 0.0f;
        float tk = -INFINITY;
        int   ti = -1;
        #pragma unroll
        for (int w = 0; w < 8; ++w) {
            tz += s_z[w];
            mtp_best_merge(tk, ti, s_k[w], s_i[w]);
        }
        part_z[r * MTP_SAMPLE_BLOCKS + b]   = tz;
        part_key[r * MTP_SAMPLE_BLOCKS + b] = tk;
        part_idx[r * MTP_SAMPLE_BLOCKS + b] = ti;
    }
}

// K3: one wave per row combines the 32 chunk partials (fixed order).
__launch_bounds__(32)
static __global__ void mtp_sample_combine_kernel(const float * __restrict__ logits, int row_stride,
                                                 MtpSampleBatchParams params,
                                                 const float * __restrict__ part_max,
                                                 const float * __restrict__ part_z,
                                                 const float * __restrict__ part_key,
                                                 const int32_t * __restrict__ part_idx,
                                                 int32_t * __restrict__ out_tok,
                                                 float * __restrict__ out_stats) {
    const int r = (int)blockIdx.x;
    const int lane = (int)threadIdx.x;
    float m  = part_max[r * MTP_SAMPLE_BLOCKS + lane];
    float z  = part_z[r * MTP_SAMPLE_BLOCKS + lane];
    float bk = part_key[r * MTP_SAMPLE_BLOCKS + lane];
    int   bi = part_idx[r * MTP_SAMPLE_BLOCKS + lane];
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        m = fmaxf(m, __shfl_xor(m, off, 32));
        z += __shfl_xor(z, off, 32);
        mtp_best_merge(bk, bi, __shfl_xor(bk, off, 32), __shfl_xor(bi, off, 32));
    }
    if (lane == 0) {
        const int   src_row = params.row_idx[r];
        const int   ex      = params.excl[r];
        const float inv_t   = params.inv_t[r];
        const float min_p   = params.min_p[r];
        float e_ex = 0.0f;
        float k_ex = 0.0f;
        if (ex >= 0) {
            e_ex = expf((logits[(size_t)src_row * (size_t)row_stride + (size_t)ex] - m) * inv_t);
            k_ex = (e_ex >= min_p) ? 1.0f : 0.0f;
        }
        out_tok[r] = (bk > -INFINITY) ? bi : -1;
        out_stats[4 * r + 0] = m;
        out_stats[4 * r + 1] = z;
        out_stats[4 * r + 2] = e_ex;
        out_stats[4 * r + 3] = k_ex;
    }
}

void launch_mtp_sample_full_batch(const float * logits, int m, int vocab_size, int row_stride,
                                  const MtpSampleBatchParams & params, float * scratch,
                                  int32_t * out_tok, float * out_stats, hipStream_t stream) {
    static_assert(MTP_SAMPLE_BLOCKS == 32, "K2/K3 reduce the chunk partials with one wave");
    if (logits == nullptr || scratch == nullptr || out_tok == nullptr || out_stats == nullptr ||
        m < 1 || m > MTP_SAMPLE_MAX_ROWS || vocab_size < MTP_SAMPLE_BLOCKS || row_stride < vocab_size) {
        return;
    }
    const int np = MTP_SAMPLE_MAX_ROWS * MTP_SAMPLE_BLOCKS;
    float   * part_max = scratch;
    float   * part_z   = scratch + np;
    float   * part_key = scratch + 2 * np;
    int32_t * part_idx = reinterpret_cast<int32_t *>(scratch + 3 * np);
    const dim3 grid(MTP_SAMPLE_BLOCKS, (unsigned)m);
    mtp_sample_max_kernel<<<grid, 256, 0, stream>>>(logits, vocab_size, row_stride, params, part_max);
    mtp_sample_race_kernel<<<grid, 256, 0, stream>>>(logits, vocab_size, row_stride, params,
                                                     part_max, part_z, part_key, part_idx);
    mtp_sample_combine_kernel<<<dim3((unsigned)m), 32, 0, stream>>>(logits, row_stride, params,
                                                                    part_max, part_z, part_key, part_idx,
                                                                    out_tok, out_stats);
}

void launch_mtp_sample_full(const float * logits, int m, int vocab_size, int row_stride,
                            float inv_t, float min_p, const MtpSampleExcl & excl,
                            uint64_t seed, float * scratch,
                            int32_t * out_tok, float * out_stats, hipStream_t stream) {
    if (m < 1 || m > MTP_SAMPLE_MAX_ROWS) return;
    MtpSampleBatchParams params{};
    const uint32_t k0 = (uint32_t)seed;
    const uint32_t k1 = (uint32_t)(seed >> 32);
    const float mp = (min_p > 0.0f) ? min_p : 0.0f;
    for (int r = 0; r < m; ++r) {
        params.row_idx[r] = r;
        params.excl[r]    = excl.tok[r];
        params.inv_t[r]   = inv_t;
        params.min_p[r]   = mp;
        params.k0[r]      = k0;
        params.k1[r]      = k1;
        params.salt[r]    = (uint32_t)r;
    }
    launch_mtp_sample_full_batch(logits, m, vocab_size, row_stride, params, scratch,
                                 out_tok, out_stats, stream);
}

} // namespace inkling
