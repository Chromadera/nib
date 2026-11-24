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

} // namespace inkling
