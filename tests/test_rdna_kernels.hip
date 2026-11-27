// ============================================================================
// test_rdna_kernels.hip — Standalone RDNA3 / RDNA4 GPU Kernel Verification Suite
// ============================================================================
// Self-contained, weight-free, low-VRAM (<200 MiB peak) correctness and
// coherence verification suite covering every RDNA3/RDNA4 architecture-specific
// code path in Nib:
//   1. Raw Wave32 FP16 WMMA 16x16x16 + nib_pack_p_frag chained GEMM
//   2. Dense SoA Q8_0 WMMA GEMM & QKV3 (tail + full tiles)
//   3. Grouped MoE NVFP4 & Q8_0 WMMA Prefill (SITU + weighted combine + sconv)
//   4. Loop Controller (down/apply) & FFN Norm + Router Top-8 WMMA Prefill
//   5. GQA FlashAttention Prefill WMMA (SWA + Global, Q8_0 / Q4_1 / Q4_0,
//      RelBias + sinks + qgate + FWHT-128 + split-KV)
//   6. Sparse Global Quest Page Scoring (Q8 & Q4 meta) + Sparse FA Prefill WMMA
//   7. Vision Patch Embed Conv2D WMMA + Vision FA d=64 WMMA
//   8. Cross-Workgroup Cache Coherence Stress Test (nib_flush_stores + nib_inv_l0_l1)
//   9. Cooperative Spin-Barrier Co-Residency Safety Test (MTP mrows + decode)
//  10. RDNA4 only: raw FP8 WMMA, NVFP4 -> E4M3 decode, FP8 row quantiser (test_rdna4_fp8.hip)
//  11. RDNA4 only: grouped NVFP4 MoE prefill on FP8 WMMA (test_rdna4_fp8.hip)
// ============================================================================

#include "inkling_engine.h"
#include "kernels_decode_fused.h"
#include "kernels_step7h_fused.h"
#include "kernels_step7j_fdot2.h"
#include "kernels_wmma_fp8.h"
#include "nib_arch.h"
#include "sparse_global.h"
#include "vision_encoder.hpp"

#include <algorithm>
#include <cfloat>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>

using namespace inkling;

namespace {

#define HIP_CHECK(call)                                                                          \
    do {                                                                                         \
        hipError_t _e = (call);                                                                  \
        if (_e != hipSuccess) {                                                                  \
            std::fprintf(stderr, "HIP error %d (%s) at %s:%d: %s\n", (int)_e,                    \
                         hipGetErrorString(_e), __FILE__, __LINE__, #call);                      \
            std::exit(2);                                                                        \
        }                                                                                        \
    } while (0)

struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed) {}
    uint64_t next_u64() {
        uint64_t z = (s += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    float uniform(float lo, float hi) {
        const double u = (double)(next_u64() >> 11) * (1.0 / 9007199254740992.0);
        return (float)(lo + u * (hi - lo));
    }
    int randint(int lo, int hi) {
        return lo + (int)(next_u64() % (uint64_t)(hi - lo + 1));
    }
};

struct DiffStats {
    double max_abs = 0.0;
    double max_rel = 0.0;
    double cos_sim = 1.0;
    int    nan_cnt = 0;
};

DiffStats compare_vecs(const float * got, const double * ref, size_t n) {
    DiffStats st;
    double dot = 0.0, ng = 0.0, nr = 0.0;
    for (size_t i = 0; i < n; ++i) {
        const double g = (double)got[i];
        const double r = ref[i];
        if (!std::isfinite(g)) {
            ++st.nan_cnt;
            continue;
        }
        const double ad = std::fabs(g - r);
        if (ad > st.max_abs) st.max_abs = ad;
        const double denom = std::max(std::fabs(r), 1e-3);
        const double rd = ad / denom;
        if (rd > st.max_rel) st.max_rel = rd;
        dot += g * r;
        ng += g * g;
        nr += r * r;
    }
    if (ng > 0.0 && nr > 0.0) {
        st.cos_sim = dot / (std::sqrt(ng) * std::sqrt(nr));
    }
    return st;
}

DiffStats compare_vecs_f32(const float * got, const float * ref, size_t n) {
    std::vector<double> r(n);
    for (size_t i = 0; i < n; ++i) r[i] = (double)ref[i];
    return compare_vecs(got, r.data(), n);
}

void fwht128_cpu(double * v) {
    for (int len = 1; len < 128; len <<= 1) {
        for (int i = 0; i < 128; i += 2 * len) {
            for (int j = 0; j < len; ++j) {
                const double a = v[i + j];
                const double b = v[i + j + len];
                v[i + j]       = a + b;
                v[i + j + len] = a - b;
            }
        }
    }
    const double inv_sqrt128 = 1.0 / std::sqrt(128.0);
    for (int i = 0; i < 128; ++i) v[i] *= inv_sqrt128;
}

// ============================================================================
// Test 1: Raw Wave32 FP16 WMMA 16x16x16 + nib_pack_p_frag chained GEMM
// ============================================================================
__global__ void __launch_bounds__(32)
test_raw_wmma_kernel(const __half * __restrict__ A,   // [16, 16] row-major
                     const __half * __restrict__ B,   // [16, 16] col-major (i.e. B^T[16,16] row-major)
                     const __half * __restrict__ V,   // [16, 16] row-major (V^T for chained O = V @ P)
                     float * __restrict__ C_out,      // [16, 16] row-major: A @ B^T
                     float * __restrict__ O_out) {    // [16, 16] row-major: V @ fp16(C)
    __shared__ _Float16 sA[16][16];
    __shared__ _Float16 sB[16][16];
    __shared__ _Float16 sV[16][16];

    const int lane   = threadIdx.x;
    const int l16    = lane & 15;
    const int half_w = lane >> 4;

    for (int idx = lane; idx < 256; idx += 32) {
        const int r = idx >> 4;
        const int c = idx & 15;
        sA[r][c] = (_Float16)__half2float(A[idx]);
        sB[r][c] = (_Float16)__half2float(B[idx]);
        sV[r][c] = (_Float16)__half2float(V[idx]);
    }
    __syncthreads();

    const nib_wmma_hfrag_t a_f = nib_lds_wmma_frag(&sA[l16][0], half_w);
    const nib_wmma_hfrag_t b_f = nib_lds_wmma_frag(&sB[l16][0], half_w);
    nib_float8_t c_acc = {0, 0, 0, 0, 0, 0, 0, 0};
    c_acc = nib_wmma_f32_16x16x16_f16(a_f, b_f, c_acc);

    // Store C_out[row, col] = (A @ B^T)[row, col]
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = nib_wmma_c_row(half_w, i);
        C_out[row * 16 + l16] = c_acc[i];
    }

    // Chain: P = fp16(C) packed directly via nib_pack_p_frag -> O = V @ P
    const nib_wmma_hfrag_t v_f = nib_lds_wmma_frag(&sV[l16][0], half_w);
    const nib_wmma_hfrag_t p_f = nib_pack_p_frag(c_acc, half_w);
    nib_float8_t o_acc = {0, 0, 0, 0, 0, 0, 0, 0};
    o_acc = nib_wmma_f32_16x16x16_f16(v_f, p_f, o_acc);

#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = nib_wmma_c_row(half_w, i);
        O_out[row * 16 + l16] = o_acc[i];
    }
}

bool run_test_raw_wmma(hipStream_t stream) {
    Rng rng(1001);
    std::vector<__half> hA(256), hB(256), hV(256);
    for (int i = 0; i < 256; ++i) {
        hA[i] = __float2half(rng.uniform(-1.0f, 1.0f));
        hB[i] = __float2half(rng.uniform(-1.0f, 1.0f));
        hV[i] = __float2half(rng.uniform(-1.0f, 1.0f));
    }
    std::vector<double> refC(256, 0.0), refO(256, 0.0);
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            double s = 0.0;
            for (int k = 0; k < 16; ++k) {
                s += (double)__half2float(hA[r * 16 + k]) * (double)__half2float(hB[c * 16 + k]);
            }
            refC[r * 16 + c] = s;
        }
    }
    for (int r = 0; r < 16; ++r) {
        for (int c = 0; c < 16; ++c) {
            double s = 0.0;
            for (int k = 0; k < 16; ++k) {
                const double p_hk = (double)__half2float(__float2half((float)refC[k * 16 + c]));
                s += (double)__half2float(hV[r * 16 + k]) * p_hk;
            }
            refO[r * 16 + c] = s;
        }
    }

    __half *dA, *dB, *dV;
    float *dC, *dO;
    HIP_CHECK(hipMalloc(&dA, 256 * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dB, 256 * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dV, 256 * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dC, 256 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dO, 256 * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), 256 * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), 256 * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dV, hV.data(), 256 * sizeof(__half), hipMemcpyHostToDevice));

    test_raw_wmma_kernel<<<1, 32, 0, stream>>>(dA, dB, dV, dC, dO);
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> hC(256), hO(256);
    HIP_CHECK(hipMemcpy(hC.data(), dC, 256 * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hO.data(), dO, 256 * sizeof(float), hipMemcpyDeviceToHost));

    HIP_CHECK(hipFree(dA)); HIP_CHECK(hipFree(dB)); HIP_CHECK(hipFree(dV));
    HIP_CHECK(hipFree(dC)); HIP_CHECK(hipFree(dO));

    const DiffStats sC = compare_vecs(hC.data(), refC.data(), 256);
    const DiffStats sO = compare_vecs(hO.data(), refO.data(), 256);
    const bool ok = (sC.nan_cnt == 0 && sO.nan_cnt == 0 && sC.max_abs < 1e-3 && sO.max_abs < 5e-3);
    std::printf("[%s] 1. Raw Wave32 WMMA + pack_p_frag: C max_abs=%.2e cos=%.6f | O max_abs=%.2e cos=%.6f\n",
                ok ? "PASS" : "FAIL", sC.max_abs, sC.cos_sim, sO.max_abs, sO.cos_sim);
    return ok;
}

// ============================================================================
// Test 2: Dense SoA Q8_0 WMMA GEMM & QKV3 (M=17 tail + M=73 multi-tile)
// ============================================================================
void fill_soa_q8(Rng & rng, int rows, int K,
                 std::vector<int8_t> & h_qs, std::vector<__half> & h_d) {
    const int nb = K / 32;
    h_qs.resize((size_t)rows * K);
    h_d.resize((size_t)rows * nb);
    for (int r = 0; r < rows; ++r) {
        for (int b = 0; b < nb; ++b) {
            h_d[(size_t)r * nb + b] = __float2half(rng.uniform(0.002f, 0.015f));
            for (int j = 0; j < 32; ++j) {
                h_qs[(size_t)r * K + b * 32 + j] = (int8_t)rng.randint(-127, 127);
            }
        }
    }
}

bool run_test_dense_wmma(hipStream_t stream) {
    Rng rng(2002);
    const int K = N_EMBD;
    const int N0 = 256, N1 = 128, N2 = 128;
    const int M = 73; // exercises M>64 tiling + non-multiple-of-16 tail

    std::vector<int8_t> hq0, hq1, hq2;
    std::vector<__half> hd0, hd1, hd2;
    fill_soa_q8(rng, N0, K, hq0, hd0);
    fill_soa_q8(rng, N1, K, hq1, hd1);
    fill_soa_q8(rng, N2, K, hq2, hd2);

    std::vector<float> hX0((size_t)M * K), hX1((size_t)M * K), hX2((size_t)M * K);
    for (size_t i = 0; i < hX0.size(); ++i) {
        hX0[i] = rng.uniform(-0.5f, 0.5f);
        hX1[i] = rng.uniform(-0.5f, 0.5f);
        hX2[i] = rng.uniform(-0.5f, 0.5f);
    }

    auto alloc_mat = [&](int rows, const std::vector<int8_t> & hq, const std::vector<__half> & hd) {
        SoaQ8Matrix W;
        W.rows = rows;
        W.K = K;
        HIP_CHECK(hipMalloc(&W.qs, hq.size()));
        HIP_CHECK(hipMalloc(&W.d, hd.size() * sizeof(__half)));
        HIP_CHECK(hipMemcpy(W.qs, hq.data(), hq.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(W.d, hd.data(), hd.size() * sizeof(__half), hipMemcpyHostToDevice));
        return W;
    };
    SoaQ8Matrix W0 = alloc_mat(N0, hq0, hd0);
    SoaQ8Matrix W1 = alloc_mat(N1, hq1, hd1);
    SoaQ8Matrix W2 = alloc_mat(N2, hq2, hd2);

    float *dX0, *dX1, *dX2, *dY0, *dY1, *dY2, *dY0_single;
    HIP_CHECK(hipMalloc(&dX0, hX0.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dX1, hX1.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dX2, hX2.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dY0, (size_t)M * N0 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dY1, (size_t)M * N1 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dY2, (size_t)M * N2 * sizeof(float)));
    HIP_CHECK(hipMalloc(&dY0_single, (size_t)M * N0 * sizeof(float)));
    HIP_CHECK(hipMemcpy(dX0, hX0.data(), hX0.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dX1, hX1.data(), hX1.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dX2, hX2.data(), hX2.size() * sizeof(float), hipMemcpyHostToDevice));

    launch_soa_q8_gemm_wmma(W0, dX0, dY0_single, M, stream);
    launch_soa_q8_gemm_wmma_qkv3(W0, dX0, dY0, W1, dX1, dY1, W2, dX2, dY2, M, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> hY0s((size_t)M * N0), hY0((size_t)M * N0), hY1((size_t)M * N1), hY2((size_t)M * N2);
    HIP_CHECK(hipMemcpy(hY0s.data(), dY0_single, hY0s.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hY0.data(), dY0, hY0.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hY1.data(), dY1, hY1.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hY2.data(), dY2, hY2.size() * sizeof(float), hipMemcpyDeviceToHost));

    // CPU FP64 reference for W0 (rows 0, 17, 72) and W1/W2 (row 72)
    auto cpu_gemm = [&](const std::vector<int8_t> & hq, const std::vector<__half> & hd,
                        const std::vector<float> & hX, int rows, int m_row) {
        std::vector<double> out(rows, 0.0);
        const int nb = K / 32;
        for (int n = 0; n < rows; ++n) {
            double s = 0.0;
            for (int b = 0; b < nb; ++b) {
                const double scale = (double)__half2float(hd[(size_t)n * nb + b]);
                double bs = 0.0;
                for (int j = 0; j < 32; ++j) {
                    bs += (double)hX[(size_t)m_row * K + b * 32 + j] * (double)hq[(size_t)n * K + b * 32 + j];
                }
                s += bs * scale;
            }
            out[n] = s;
        }
        return out;
    };

    double worst_cos = 1.0, worst_abs = 0.0;
    int nans = 0;
    for (int mr : {0, 17, 72}) {
        auto r0 = cpu_gemm(hq0, hd0, hX0, N0, mr);
        auto st = compare_vecs(hY0s.data() + (size_t)mr * N0, r0.data(), N0);
        worst_cos = std::min(worst_cos, st.cos_sim);
        worst_abs = std::max(worst_abs, st.max_abs);
        nans += st.nan_cnt;
    }
    auto r1 = cpu_gemm(hq1, hd1, hX1, N1, 72);
    auto r2 = cpu_gemm(hq2, hd2, hX2, N2, 72);
    auto st1 = compare_vecs(hY1.data() + (size_t)72 * N1, r1.data(), N1);
    auto st2 = compare_vecs(hY2.data() + (size_t)72 * N2, r2.data(), N2);
    auto st_qkv = compare_vecs_f32(hY0.data(), hY0s.data(), hY0.size());
    worst_cos = std::min({worst_cos, st1.cos_sim, st2.cos_sim});
    worst_abs = std::max({worst_abs, st1.max_abs, st2.max_abs});
    nans += st1.nan_cnt + st2.nan_cnt + st_qkv.nan_cnt;

    HIP_CHECK(hipFree(W0.qs)); HIP_CHECK(hipFree(W0.d));
    HIP_CHECK(hipFree(W1.qs)); HIP_CHECK(hipFree(W1.d));
    HIP_CHECK(hipFree(W2.qs)); HIP_CHECK(hipFree(W2.d));
    HIP_CHECK(hipFree(dX0)); HIP_CHECK(hipFree(dX1)); HIP_CHECK(hipFree(dX2));
    HIP_CHECK(hipFree(dY0)); HIP_CHECK(hipFree(dY1)); HIP_CHECK(hipFree(dY2));
    HIP_CHECK(hipFree(dY0_single));

    const bool ok = (nans == 0 && worst_cos > 0.99999 && worst_abs < 0.05 && st_qkv.max_abs < 1e-4);
    std::printf("[%s] 2. Dense SoA-Q8 WMMA & QKV3 (M=%d): max_abs=%.4f cos=%.7f qkv3_diff=%.2e\n",
                ok ? "PASS" : "FAIL", M, worst_abs, worst_cos, st_qkv.max_abs);
    return ok;
}

// ============================================================================
// Test 3: Grouped MoE NVFP4 & Q8_0 WMMA Prefill vs Decode Reference
// ============================================================================
MoePrefillScratch alloc_moe_scratch() {
    MoePrefillScratch s;
    HIP_CHECK(hipMalloc(&s.exp_counts,  N_EXPERT * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&s.exp_offsets, (N_EXPERT + 1) * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&s.exp_cursors, N_EXPERT * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&s.perm_tok,    MAX_UBATCH * TOP_K * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&s.perm_w,      MAX_UBATCH * TOP_K * sizeof(float)));
    HIP_CHECK(hipMalloc(&s.perm_x,      (size_t)MAX_UBATCH * TOP_K * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&s.perm_act,    (size_t)MAX_UBATCH * TOP_K * N_FF_EXP * sizeof(float)));
    HIP_CHECK(hipMalloc(&s.perm_down,   (size_t)MAX_UBATCH * TOP_K * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&s.tile_exp,    (MAX_UBATCH * TOP_K + N_EXPERT) * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&s.tile_m0,     (MAX_UBATCH * TOP_K + N_EXPERT) * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&s.d_num_tiles, sizeof(int32_t)));
    return s;
}

void free_moe_scratch(MoePrefillScratch & s) {
    HIP_CHECK(hipFree(s.exp_counts));
    HIP_CHECK(hipFree(s.exp_offsets));
    HIP_CHECK(hipFree(s.exp_cursors));
    HIP_CHECK(hipFree(s.perm_tok));
    HIP_CHECK(hipFree(s.perm_w));
    HIP_CHECK(hipFree(s.perm_x));
    HIP_CHECK(hipFree(s.perm_act));
    HIP_CHECK(hipFree(s.perm_down));
    HIP_CHECK(hipFree(s.tile_exp));
    HIP_CHECK(hipFree(s.tile_m0));
    HIP_CHECK(hipFree(s.d_num_tiles));
}

bool run_test_moe_wmma(hipStream_t stream) {
    Rng rng(3003);
    const int M = 19;      // non-multiple-of-16 tail + multi-token per expert
    const int N_EXP = 16;  // allocate only experts 0..15 to stay <130 MiB peak VRAM

    std::vector<float> hX((size_t)M * N_EMBD);
    for (float & v : hX) v = rng.uniform(-0.4f, 0.4f);

    std::vector<int32_t> h_ids((size_t)M * TOP_K);
    std::vector<float>   h_w((size_t)M * TOP_K);
    for (int t = 0; t < M; ++t) {
        // Pick 8 distinct experts in 0..N_EXP-1
        std::vector<int> pool(N_EXP);
        for (int e = 0; e < N_EXP; ++e) pool[e] = e;
        float wsum = 0.0f;
        for (int k = 0; k < TOP_K; ++k) {
            const int idx = rng.randint(k, N_EXP - 1);
            std::swap(pool[k], pool[idx]);
            h_ids[(size_t)t * TOP_K + k] = pool[k];
            h_w[(size_t)t * TOP_K + k] = rng.uniform(0.5f, 1.5f);
            wsum += h_w[(size_t)t * TOP_K + k];
        }
        for (int k = 0; k < TOP_K; ++k) h_w[(size_t)t * TOP_K + k] *= (EXPERT_W_SCALE / wsum);
    }

    float *dX, *dW, *dY_prefill, *dY_dec, *dTmpAct;
    int32_t *dIds;
    HIP_CHECK(hipMalloc(&dX, hX.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dIds, h_ids.size() * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&dW, h_w.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dY_prefill, (size_t)M * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&dY_dec, (size_t)M * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&dTmpAct, (size_t)TOP_K * N_FF_EXP * sizeof(float)));
    HIP_CHECK(hipMemcpy(dX, hX.data(), hX.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dIds, h_ids.data(), h_ids.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dW, h_w.data(), h_w.size() * sizeof(float), hipMemcpyHostToDevice));

    MoePrefillScratch scratch = alloc_moe_scratch();

    // ---- 3a. NVFP4 MoE WMMA Prefill vs exact FP32-activation NVFP4 Decode ----
    SoaNVFP4Experts gu_fp4, dn_fp4;
    gu_fp4.n_exp = N_EXP; gu_fp4.rows = N_GATE_UP_EXP; gu_fp4.K = N_EMBD;
    dn_fp4.n_exp = N_EXP; dn_fp4.rows = N_EMBD;        dn_fp4.K = N_FF_EXP;
    const size_t gu_qs_b = (size_t)N_EXP * gu_fp4.rows * (gu_fp4.K / 2);
    const size_t gu_sc_b = (size_t)N_EXP * gu_fp4.rows * (gu_fp4.K / 16);
    const size_t dn_qs_b = (size_t)N_EXP * dn_fp4.rows * (dn_fp4.K / 2);
    const size_t dn_sc_b = (size_t)N_EXP * dn_fp4.rows * (dn_fp4.K / 16);

    std::vector<uint8_t> h_gu_qs(gu_qs_b), h_gu_sc(gu_sc_b), h_dn_qs(dn_qs_b), h_dn_sc(dn_sc_b);
    std::vector<float> h_gu_s2(N_EXP), h_dn_s2(N_EXP);
    for (uint8_t & b : h_gu_qs) b = (uint8_t)(rng.next_u64() & 0xFF);
    for (uint8_t & b : h_dn_qs) b = (uint8_t)(rng.next_u64() & 0xFF);
    for (uint8_t & b : h_gu_sc) b = (uint8_t)rng.randint(0x20, 0x38); // well-conditioned UE4M3
    for (uint8_t & b : h_dn_sc) b = (uint8_t)rng.randint(0x20, 0x38);
    for (int e = 0; e < N_EXP; ++e) {
        h_gu_s2[e] = rng.uniform(0.02f, 0.06f);
        h_dn_s2[e] = rng.uniform(0.02f, 0.06f);
    }

    HIP_CHECK(hipMalloc(&gu_fp4.qs, gu_qs_b));
    HIP_CHECK(hipMalloc(&gu_fp4.scales, gu_sc_b));
    HIP_CHECK(hipMalloc(&gu_fp4.scale2, N_EXP * sizeof(float)));
    HIP_CHECK(hipMalloc(&dn_fp4.qs, dn_qs_b));
    HIP_CHECK(hipMalloc(&dn_fp4.scales, dn_sc_b));
    HIP_CHECK(hipMalloc(&dn_fp4.scale2, N_EXP * sizeof(float)));
    HIP_CHECK(hipMemcpy(gu_fp4.qs, h_gu_qs.data(), gu_qs_b, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gu_fp4.scales, h_gu_sc.data(), gu_sc_b, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gu_fp4.scale2, h_gu_s2.data(), N_EXP * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dn_fp4.qs, h_dn_qs.data(), dn_qs_b, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dn_fp4.scales, h_dn_sc.data(), dn_sc_b, hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dn_fp4.scale2, h_dn_s2.data(), N_EXP * sizeof(float), hipMemcpyHostToDevice));

    launch_moe_nvfp4_prefill_wmma(gu_fp4, dn_fp4, dX, dIds, dW, scratch, dY_prefill, M, stream);
    for (int t = 0; t < M; ++t) {
        launch_moe_nvfp4_decode(gu_fp4, dn_fp4, dX + (size_t)t * N_EMBD,
                                dIds + (size_t)t * TOP_K, dW + (size_t)t * TOP_K,
                                dTmpAct, dY_dec + (size_t)t * N_EMBD, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> hYp((size_t)M * N_EMBD), hYd((size_t)M * N_EMBD);
    HIP_CHECK(hipMemcpy(hYp.data(), dY_prefill, hYp.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hYd.data(), dY_dec,     hYd.size() * sizeof(float), hipMemcpyDeviceToHost));
    const DiffStats st_fp4 = compare_vecs_f32(hYp.data(), hYd.data(), hYp.size());

    HIP_CHECK(hipFree(gu_fp4.qs)); HIP_CHECK(hipFree(gu_fp4.scales)); HIP_CHECK(hipFree(gu_fp4.scale2));
    HIP_CHECK(hipFree(dn_fp4.qs)); HIP_CHECK(hipFree(dn_fp4.scales)); HIP_CHECK(hipFree(dn_fp4.scale2));

    // ---- 3b. Q8_0 MoE WMMA Prefill vs exact FP32-activation Q8_0 Decode ----
    SoaQ8Experts gu_q8, dn_q8;
    gu_q8.n_exp = N_EXP; gu_q8.rows = N_GATE_UP_EXP; gu_q8.K = N_EMBD;
    dn_q8.n_exp = N_EXP; dn_q8.rows = N_EMBD;        dn_q8.K = N_FF_EXP;
    std::vector<int8_t> h_gu_q8, h_dn_q8;
    std::vector<__half> h_gu_d8, h_dn_d8;
    fill_soa_q8(rng, N_EXP * gu_q8.rows, gu_q8.K, h_gu_q8, h_gu_d8);
    fill_soa_q8(rng, N_EXP * dn_q8.rows, dn_q8.K, h_dn_q8, h_dn_d8);

    HIP_CHECK(hipMalloc(&gu_q8.qs, h_gu_q8.size()));
    HIP_CHECK(hipMalloc(&gu_q8.d,  h_gu_d8.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&dn_q8.qs, h_dn_q8.size()));
    HIP_CHECK(hipMalloc(&dn_q8.d,  h_dn_d8.size() * sizeof(__half)));
    HIP_CHECK(hipMemcpy(gu_q8.qs, h_gu_q8.data(), h_gu_q8.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(gu_q8.d,  h_gu_d8.data(), h_gu_d8.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dn_q8.qs, h_dn_q8.data(), h_dn_q8.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dn_q8.d,  h_dn_d8.data(), h_dn_d8.size() * sizeof(__half), hipMemcpyHostToDevice));

    launch_moe_q8_prefill_wmma(gu_q8, dn_q8, dX, dIds, dW, scratch, dY_prefill, M, stream);
    for (int t = 0; t < M; ++t) {
        launch_moe_q8_decode(gu_q8, dn_q8, dX + (size_t)t * N_EMBD,
                             dIds + (size_t)t * TOP_K, dW + (size_t)t * TOP_K,
                             dTmpAct, dY_dec + (size_t)t * N_EMBD, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    HIP_CHECK(hipMemcpy(hYp.data(), dY_prefill, hYp.size() * sizeof(float), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(hYd.data(), dY_dec,     hYd.size() * sizeof(float), hipMemcpyDeviceToHost));
    const DiffStats st_q8 = compare_vecs_f32(hYp.data(), hYd.data(), hYp.size());

    HIP_CHECK(hipFree(gu_q8.qs)); HIP_CHECK(hipFree(gu_q8.d));
    HIP_CHECK(hipFree(dn_q8.qs)); HIP_CHECK(hipFree(dn_q8.d));
    free_moe_scratch(scratch);
    HIP_CHECK(hipFree(dX)); HIP_CHECK(hipFree(dIds)); HIP_CHECK(hipFree(dW));
    HIP_CHECK(hipFree(dY_prefill)); HIP_CHECK(hipFree(dY_dec)); HIP_CHECK(hipFree(dTmpAct));

    const bool ok = (st_fp4.nan_cnt == 0 && st_q8.nan_cnt == 0 &&
                     st_fp4.cos_sim > 0.9999 && st_q8.cos_sim > 0.9999);
    std::printf("[%s] 3. Grouped MoE WMMA Prefill (M=%d): NVFP4 cos=%.7f max_abs=%.4f | Q8 cos=%.7f max_abs=%.4f\n",
                ok ? "PASS" : "FAIL", M, st_fp4.cos_sim, st_fp4.max_abs, st_q8.cos_sim, st_q8.max_abs);
    return ok;
}

// ============================================================================
// Test 4: Loop Controller (down/apply) & FFN Norm + Router Top-8 WMMA Prefill
// ============================================================================
bool run_test_core_wmma(hipStream_t stream) {
    Rng rng(4004);
    const int M = 37; // >16 -> triggers WMMA paths in kernels_core.hip with non-16 tail

    std::vector<float> h_cur((size_t)M * N_EMBD), h_msg((size_t)M * N_EMBD);
    std::vector<float> h_norm_w(N_EMBD), h_bias(N_EXPERT);
    std::vector<__half> h_wd((size_t)LOOP_GATE_RANK * 2 * N_EMBD);
    std::vector<__half> h_wg((size_t)N_EMBD * LOOP_GATE_RANK);
    std::vector<__half> h_wf((size_t)N_EMBD * LOOP_GATE_RANK);
    std::vector<__half> h_rw((size_t)N_EXPERT * N_EMBD);

    for (float & v : h_cur)    v = rng.uniform(-0.8f, 0.8f);
    for (float & v : h_msg)    v = rng.uniform(-0.8f, 0.8f);
    for (float & v : h_norm_w) v = rng.uniform(0.5f, 1.5f);
    for (float & v : h_bias)   v = rng.uniform(-0.05f, 0.05f);
    for (__half & h : h_wd)    h = __float2half(rng.uniform(-0.05f, 0.05f));
    for (__half & h : h_wg)    h = __float2half(rng.uniform(-0.15f, 0.15f));
    for (__half & h : h_wf)    h = __float2half(rng.uniform(-0.15f, 0.15f));
    for (__half & h : h_rw)    h = __float2half(rng.uniform(-0.08f, 0.08f));

    float *d_cur, *d_msg, *d_gate, *d_out_pf, *d_out_dec, *d_norm_w, *d_bias;
    float *d_xn_pf, *d_xn_dec, *d_tw_pf, *d_tw_dec, *d_rscr;
    int32_t *d_tid_pf, *d_tid_dec;
    __half *d_wd, *d_wg, *d_wf, *d_rw;

    HIP_CHECK(hipMalloc(&d_cur,     h_cur.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_msg,     h_msg.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gate,    (size_t)M * LOOP_GATE_RANK * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_pf,  (size_t)M * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out_dec, (size_t)M * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_norm_w,  N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_bias,    N_EXPERT * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_xn_pf,   (size_t)M * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_xn_dec,  (size_t)M * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tw_pf,   (size_t)M * TOP_K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tw_dec,  (size_t)M * TOP_K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tid_pf,  (size_t)M * TOP_K * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_tid_dec, (size_t)M * TOP_K * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_rscr,    512 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_wd,      h_wd.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_wg,      h_wg.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_wf,      h_wf.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_rw,      h_rw.size() * sizeof(__half)));

    HIP_CHECK(hipMemset(d_rscr, 0, 512 * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_cur,    h_cur.data(),    h_cur.size() * sizeof(float),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_msg,    h_msg.data(),    h_msg.size() * sizeof(float),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_norm_w, h_norm_w.data(), N_EMBD * sizeof(float),          hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias,   h_bias.data(),   N_EXPERT * sizeof(float),        hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wd,     h_wd.data(),     h_wd.size() * sizeof(__half),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wg,     h_wg.data(),     h_wg.size() * sizeof(__half),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wf,     h_wf.data(),     h_wf.size() * sizeof(__half),    hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rw,     h_rw.data(),     h_rw.size() * sizeof(__half),    hipMemcpyHostToDevice));

    // Prefill WMMA vs single-token decode reference
    launch_inkling_ctrl(d_cur, d_msg, d_wd, d_wg, d_wf, d_gate, d_out_pf, M, stream);
    for (int t = 0; t < M; ++t) {
        launch_inkling_ctrl(d_cur + (size_t)t * N_EMBD, d_msg + (size_t)t * N_EMBD,
                            d_wd, d_wg, d_wf, d_gate, d_out_dec + (size_t)t * N_EMBD, 1, stream);
    }

    launch_ffn_norm_router_topk(d_cur, d_norm_w, d_rw, d_bias, d_xn_pf, d_tid_pf, d_tw_pf, M, stream, d_rscr);
    for (int t = 0; t < M; ++t) {
        launch_ffn_norm_router_topk(d_cur + (size_t)t * N_EMBD, d_norm_w, d_rw, d_bias,
                                    d_xn_dec + (size_t)t * N_EMBD,
                                    d_tid_dec + (size_t)t * TOP_K,
                                    d_tw_dec + (size_t)t * TOP_K,
                                    1, stream, d_rscr);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> h_op(h_cur.size()), h_od(h_cur.size());
    std::vector<float> h_twp((size_t)M * TOP_K), h_twd((size_t)M * TOP_K);
    std::vector<int32_t> h_idp((size_t)M * TOP_K), h_idd((size_t)M * TOP_K);
    HIP_CHECK(hipMemcpy(h_op.data(),  d_out_pf,  h_op.size() * sizeof(float),    hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_od.data(),  d_out_dec, h_od.size() * sizeof(float),    hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_twp.data(), d_tw_pf,   h_twp.size() * sizeof(float),   hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_twd.data(), d_tw_dec,  h_twd.size() * sizeof(float),   hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_idp.data(), d_tid_pf,  h_idp.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipMemcpy(h_idd.data(), d_tid_dec, h_idd.size() * sizeof(int32_t), hipMemcpyDeviceToHost));

    const DiffStats st_ctrl = compare_vecs_f32(h_op.data(), h_od.data(), h_op.size());
    const DiffStats st_rtw  = compare_vecs_f32(h_twp.data(), h_twd.data(), h_twp.size());
    int id_matches = 0;
    for (size_t i = 0; i < h_idp.size(); ++i) {
        if (h_idp[i] == h_idd[i]) ++id_matches;
    }

    HIP_CHECK(hipFree(d_cur)); HIP_CHECK(hipFree(d_msg)); HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_out_pf)); HIP_CHECK(hipFree(d_out_dec));
    HIP_CHECK(hipFree(d_norm_w)); HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_xn_pf)); HIP_CHECK(hipFree(d_xn_dec));
    HIP_CHECK(hipFree(d_tw_pf)); HIP_CHECK(hipFree(d_tw_dec));
    HIP_CHECK(hipFree(d_tid_pf)); HIP_CHECK(hipFree(d_tid_dec));
    HIP_CHECK(hipFree(d_rscr));
    HIP_CHECK(hipFree(d_wd)); HIP_CHECK(hipFree(d_wg)); HIP_CHECK(hipFree(d_wf)); HIP_CHECK(hipFree(d_rw));

    const double id_acc = (double)id_matches / (double)h_idp.size();
    const bool ok = (st_ctrl.nan_cnt == 0 && st_rtw.nan_cnt == 0 &&
                     st_ctrl.cos_sim > 0.99999 && st_ctrl.max_abs < 5e-3 &&
                     id_acc >= 0.98 && st_rtw.max_abs < 2e-2);
    std::printf("[%s] 4. Core WMMA (ctrl + router, M=%d): ctrl cos=%.7f max_abs=%.4f | router top8=%.1f%% w_abs=%.4f\n",
                ok ? "PASS" : "FAIL", M, st_ctrl.cos_sim, st_ctrl.max_abs, id_acc * 100.0, st_rtw.max_abs);
    return ok;
}

// ============================================================================
// Helper: Build quantized KV cache on host + upload to device + CPU dequant
// ============================================================================
struct KvTestBuf {
    KVCacheSlot slot;
    std::vector<double> k_deq; // [N_HEAD_KV, cap, HEAD_DIM]
    std::vector<double> v_deq; // [N_HEAD_KV, cap, HEAD_DIM]
    std::vector<int32_t> k_pos;

    void init(Rng & rng, int cap, int kv_type, bool is_swa, int n_fill, int start_abs_pos) {
        slot.cap = cap;
        slot.kv_type = kv_type;
        slot.is_swa = is_swa;
        const int qs_per_tok = (kv_type == KV_Q8_0) ? HEAD_DIM : (HEAD_DIM / 2);
        const int d_per_tok  = (kv_type == KV_Q4_1) ? 8 : 4;

        std::vector<int8_t> h_kqs((size_t)N_HEAD_KV * cap * qs_per_tok, 0);
        std::vector<int8_t> h_vqs((size_t)N_HEAD_KV * cap * qs_per_tok, 0);
        std::vector<__half> h_kd ((size_t)N_HEAD_KV * cap * d_per_tok,  __float2half(0.0f));
        std::vector<__half> h_vd ((size_t)N_HEAD_KV * cap * d_per_tok,  __float2half(0.0f));
        k_pos.assign(cap, -1);
        k_deq.assign((size_t)N_HEAD_KV * cap * HEAD_DIM, 0.0);
        v_deq.assign((size_t)N_HEAD_KV * cap * HEAD_DIM, 0.0);

        for (int i = 0; i < n_fill; ++i) {
            const int abs_p = start_abs_pos + i;
            const int idx   = is_swa ? (abs_p % cap) : abs_p;
            k_pos[idx] = abs_p;
            for (int h = 0; h < N_HEAD_KV; ++h) {
                for (int kv = 0; kv < 2; ++kv) {
                    auto & hqs = (kv == 0) ? h_kqs : h_vqs;
                    auto & hd  = (kv == 0) ? h_kd  : h_vd;
                    auto & deq = (kv == 0) ? k_deq : v_deq;
                    const size_t base_q = ((size_t)h * cap + idx) * qs_per_tok;
                    const size_t base_d = ((size_t)h * cap + idx) * d_per_tok;
                    const size_t base_f = ((size_t)h * cap + idx) * HEAD_DIM;

                    for (int blk = 0; blk < 4; ++blk) {
                        if (kv_type == KV_Q8_0) {
                            const float sc = rng.uniform(0.004f, 0.015f);
                            const __half dh = __float2half(sc);
                            hd[base_d + blk] = dh;
                            const double df = (double)__half2float(dh);
                            for (int j = 0; j < 32; ++j) {
                                const int8_t q = (int8_t)rng.randint(-120, 120);
                                hqs[base_q + blk * 32 + j] = q;
                                deq[base_f + blk * 32 + j] = df * (double)q;
                            }
                        } else if (kv_type == KV_Q4_1) {
                            const __half dh = __float2half(rng.uniform(0.03f, 0.08f));
                            const __half mh = __float2half(rng.uniform(-0.6f, -0.2f));
                            hd[base_d + blk * 2 + 0] = dh;
                            hd[base_d + blk * 2 + 1] = mh;
                            const double df = (double)__half2float(dh);
                            const double mf = (double)__half2float(mh);
                            for (int w = 0; w < 4; ++w) {
                                for (int j = 0; j < 4; ++j) {
                                    const int lo = rng.randint(0, 15);
                                    const int hi = rng.randint(0, 15);
                                    hqs[base_q + blk * 16 + w * 4 + j] = (int8_t)(lo | (hi << 4));
                                    deq[base_f + blk * 32 + w * 8 + j]     = df * (double)lo + mf;
                                    deq[base_f + blk * 32 + w * 8 + j + 4] = df * (double)hi + mf;
                                }
                            }
                        } else { // KV_Q4_0
                            const __half dh = __float2half(rng.uniform(0.03f, 0.08f));
                            hd[base_d + blk] = dh;
                            const double df = (double)__half2float(dh);
                            for (int w = 0; w < 4; ++w) {
                                for (int j = 0; j < 4; ++j) {
                                    const int lo = rng.randint(0, 15);
                                    const int hi = rng.randint(0, 15);
                                    hqs[base_q + blk * 16 + w * 4 + j] = (int8_t)(lo | (hi << 4));
                                    deq[base_f + blk * 32 + w * 8 + j]     = df * (double)(lo - 8);
                                    deq[base_f + blk * 32 + w * 8 + j + 4] = df * (double)(hi - 8);
                                }
                            }
                        }
                    }
                }
            }
        }

        HIP_CHECK(hipMalloc(&slot.k_qs,  h_kqs.size()));
        HIP_CHECK(hipMalloc(&slot.v_qs,  h_vqs.size()));
        HIP_CHECK(hipMalloc(&slot.k_d,   h_kd.size() * sizeof(__half)));
        HIP_CHECK(hipMalloc(&slot.v_d,   h_vd.size() * sizeof(__half)));
        HIP_CHECK(hipMalloc(&slot.k_pos, (size_t)cap * sizeof(int32_t)));
        HIP_CHECK(hipMemcpy(slot.k_qs,  h_kqs.data(), h_kqs.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(slot.v_qs,  h_vqs.data(), h_vqs.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(slot.k_d,   h_kd.data(),  h_kd.size() * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(slot.v_d,   h_vd.data(),  h_vd.size() * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(slot.k_pos, k_pos.data(), (size_t)cap * sizeof(int32_t), hipMemcpyHostToDevice));
    }

    void destroy() {
        HIP_CHECK(hipFree(slot.k_qs));
        HIP_CHECK(hipFree(slot.v_qs));
        HIP_CHECK(hipFree(slot.k_d));
        HIP_CHECK(hipFree(slot.v_d));
        HIP_CHECK(hipFree(slot.k_pos));
    }
};

// CPU FP64 FlashAttention reference (dense or selected-page list)
std::vector<double> cpu_fa_ref(
        const std::vector<float> & h_q,
        const KvTestBuf & kv,
        const std::vector<float> & h_sinks,
        const std::vector<float> & h_rel,
        const std::vector<float> & h_qgate,
        int n_tokens, int pos_start,
        const std::vector<int32_t> * sel_pages = nullptr,
        const std::vector<int32_t> * n_sel     = nullptr) {
    std::vector<double> out((size_t)n_tokens * N_HEAD * HEAD_DIM, 0.0);
    for (int t = 0; t < n_tokens; ++t) {
        const int q_pos = pos_start + t;
        const int qb = t / SPARSE_PAGE_SIZE;
        for (int h = 0; h < N_HEAD; ++h) {
            const int h_kv = h / GQA_RATIO;
            const float * qv = h_q.data() + ((size_t)t * N_HEAD + h) * HEAD_DIM;
            const float * rt = h_rel.data() + ((size_t)t * N_HEAD + h) * N_REL_EXTENT;
            const float * qg = h_qgate.data() + ((size_t)t * N_HEAD + h) * HEAD_DIM;

            std::vector<int> kv_idxs;
            if (sel_pages != nullptr && n_sel != nullptr) {
                const int ns = (*n_sel)[qb * N_HEAD_KV + h_kv];
                const int32_t * sp = sel_pages->data() + (size_t)(qb * N_HEAD_KV + h_kv) * SPARSE_MAX_SELECTED;
                for (int pi = 0; pi < ns; ++pi) {
                    const int p0 = sp[pi] * SPARSE_PAGE_SIZE;
                    for (int j = p0; j < p0 + SPARSE_PAGE_SIZE && j < kv.slot.cap; ++j) {
                        const int kp = kv.k_pos[j];
                        if (kp >= 0 && kp <= q_pos) kv_idxs.push_back(j);
                    }
                }
            } else {
                for (int j = 0; j < kv.slot.cap; ++j) {
                    const int kp = kv.k_pos[j];
                    if (kp < 0 || kp > q_pos) continue;
                    if (kv.slot.is_swa && (q_pos - kp) >= SWA_WINDOW) continue;
                    kv_idxs.push_back(j);
                }
            }

            double max_s = (double)h_sinks[(size_t)t * N_HEAD + h];
            std::vector<double> scores(kv_idxs.size());
            for (size_t ii = 0; ii < kv_idxs.size(); ++ii) {
                const int j = kv_idxs[ii];
                const int dist = q_pos - kv.k_pos[j];
                const double * kv_k = kv.k_deq.data() + ((size_t)h_kv * kv.slot.cap + j) * HEAD_DIM;
                double dot = 0.0;
                for (int d = 0; d < HEAD_DIM; ++d) dot += (double)qv[d] * kv_k[d];
                double sc = dot * (double)KQ_SCALE;
                if (dist >= 0 && dist < N_REL_EXTENT) sc += (double)rt[dist];
                scores[ii] = sc;
                if (sc > max_s) max_s = sc;
            }

            double denom = std::exp((double)h_sinks[(size_t)t * N_HEAD + h] - max_s);
            double acc[HEAD_DIM] = {};
            for (size_t ii = 0; ii < kv_idxs.size(); ++ii) {
                const double w = std::exp(scores[ii] - max_s);
                denom += w;
                const int j = kv_idxs[ii];
                const double * kv_v = kv.v_deq.data() + ((size_t)h_kv * kv.slot.cap + j) * HEAD_DIM;
                for (int d = 0; d < HEAD_DIM; ++d) acc[d] += w * kv_v[d];
            }
            const double inv_d = 1.0 / denom;
            for (int d = 0; d < HEAD_DIM; ++d) acc[d] *= inv_d;
            fwht128_cpu(acc);
            double * dst = out.data() + ((size_t)t * N_HEAD + h) * HEAD_DIM;
            for (int d = 0; d < HEAD_DIM; ++d) {
                const double sig = 1.0 / (1.0 + std::exp(-(double)qg[d]));
                dst[d] = acc[d] * sig;
            }
        }
    }
    return out;
}

// ============================================================================
// Test 5: GQA FlashAttention Prefill WMMA (SWA + Global, Q8_0 / Q4_1 / Q4_0)
// ============================================================================
bool run_test_fa_prefill_wmma(hipStream_t stream) {
    Rng rng(5005);
    bool all_ok = true;
    const char * kv_names[3] = {"Q8_0", "Q4_1", "Q4_0"};

    // Scratch buffers for split-K
    float *d_tmp_o;
    float2 *d_tmp_meta;
    HIP_CHECK(hipMalloc(&d_tmp_o,    (size_t)8 * MAX_UBATCH * N_HEAD * HEAD_DIM * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tmp_meta, (size_t)8 * MAX_UBATCH * N_HEAD * sizeof(float2)));

    for (int kvt = 0; kvt < 3; ++kvt) {
        for (int is_swa = 0; is_swa <= 1; ++is_swa) {
            const int n_tokens = 73; // >64 -> 2 Q-blocks with tail
            const int pos_start = is_swa ? 1600 : 120; // SWA tests ring wrap-around past cap=1536!
            const int cap = is_swa ? 1536 : 512;
            const int fill_start = is_swa ? (pos_start + n_tokens - 1400) : 0;
            const int fill_cnt   = is_swa ? 1400 : (pos_start + n_tokens);

            KvTestBuf kv;
            kv.init(rng, cap, kvt, is_swa != 0, fill_cnt, fill_start);

            std::vector<float> h_q((size_t)n_tokens * N_HEAD * HEAD_DIM);
            std::vector<float> h_sinks((size_t)n_tokens * N_HEAD);
            std::vector<float> h_rel((size_t)n_tokens * N_HEAD * N_REL_EXTENT);
            std::vector<float> h_qg((size_t)n_tokens * N_HEAD * HEAD_DIM);
            for (float & v : h_q)     v = rng.uniform(-0.8f, 0.8f);
            for (float & v : h_sinks) v = rng.uniform(-1.0f, 1.0f);
            for (float & v : h_rel)   v = rng.uniform(-0.3f, 0.3f);
            for (float & v : h_qg)    v = rng.uniform(-1.5f, 1.5f);

            float *d_q, *d_sinks, *d_rel, *d_qg, *d_out;
            HIP_CHECK(hipMalloc(&d_q,     h_q.size() * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_sinks, h_sinks.size() * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_rel,   h_rel.size() * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_qg,    h_qg.size() * sizeof(float)));
            HIP_CHECK(hipMalloc(&d_out,   h_q.size() * sizeof(float)));
            HIP_CHECK(hipMemcpy(d_q,     h_q.data(),     h_q.size() * sizeof(float),     hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_sinks, h_sinks.data(), h_sinks.size() * sizeof(float), hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_rel,   h_rel.data(),   h_rel.size() * sizeof(float),   hipMemcpyHostToDevice));
            HIP_CHECK(hipMemcpy(d_qg,    h_qg.data(),    h_qg.size() * sizeof(float),    hipMemcpyHostToDevice));

            launch_fa_relbias(d_q, kv.slot, d_sinks, N_HEAD, d_rel, d_qg, N_Q,
                              d_out, d_tmp_o, d_tmp_meta, n_tokens, pos_start, nullptr, stream);
            HIP_CHECK(hipStreamSynchronize(stream));

            std::vector<float> h_out(h_q.size());
            HIP_CHECK(hipMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), hipMemcpyDeviceToHost));

            auto ref = cpu_fa_ref(h_q, kv, h_sinks, h_rel, h_qg, n_tokens, pos_start);
            const DiffStats st = compare_vecs(h_out.data(), ref.data(), h_out.size());
            const bool ok = (st.nan_cnt == 0 && st.cos_sim > 0.9999 && st.max_abs < 8e-3);
            all_ok = all_ok && ok;
            std::printf("[%s] 5. FA Prefill WMMA (%s %s, M=%d): max_abs=%.4f cos=%.7f\n",
                        ok ? "PASS" : "FAIL", is_swa ? "SWA   " : "Global", kv_names[kvt],
                        n_tokens, st.max_abs, st.cos_sim);

            HIP_CHECK(hipFree(d_q)); HIP_CHECK(hipFree(d_sinks)); HIP_CHECK(hipFree(d_rel));
            HIP_CHECK(hipFree(d_qg)); HIP_CHECK(hipFree(d_out));
            kv.destroy();
        }
    }

    HIP_CHECK(hipFree(d_tmp_o));
    HIP_CHECK(hipFree(d_tmp_meta));
    return all_ok;
}

// ============================================================================
// Test 6: Sparse Global Quest Page Scoring (Q8/Q4) + Sparse FA Prefill WMMA
// ============================================================================
bool run_test_sparse_global_wmma(hipStream_t stream) {
    Rng rng(6006);
    bool all_ok = true;
    const char * fmt_names[2] = {"Q8_MM", "Q4_MM"};

    const int n_tokens  = 73;
    const int pos_start = 4096; // 64 historical pages -> exercises WMMA Quest page scoring!
    const int n_kv      = pos_start + n_tokens;
    const int max_pages = (n_kv + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
    const int sparse_pages = 36; // 2 sinks + 10 top-K + 24 recent

    KvTestBuf kv;
    kv.init(rng, n_kv, KV_Q8_0, false, n_kv, 0);

    std::vector<float> h_q((size_t)n_tokens * N_HEAD * HEAD_DIM);
    std::vector<float> h_sinks((size_t)n_tokens * N_HEAD);
    std::vector<float> h_rel((size_t)n_tokens * N_HEAD * N_REL_EXTENT);
    std::vector<float> h_qg((size_t)n_tokens * N_HEAD * HEAD_DIM);
    for (float & v : h_q)     v = rng.uniform(-0.8f, 0.8f);
    for (float & v : h_sinks) v = rng.uniform(-1.0f, 1.0f);
    for (float & v : h_rel)   v = rng.uniform(-0.3f, 0.3f);
    for (float & v : h_qg)    v = rng.uniform(-1.5f, 1.5f);

    float *d_q, *d_sinks, *d_rel, *d_qg, *d_out;
    HIP_CHECK(hipMalloc(&d_q,     h_q.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_sinks, h_sinks.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rel,   h_rel.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_qg,    h_qg.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out,   h_q.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_q,     h_q.data(),     h_q.size() * sizeof(float),     hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_sinks, h_sinks.data(), h_sinks.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rel,   h_rel.data(),   h_rel.size() * sizeof(float),   hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_qg,    h_qg.data(),    h_qg.size() * sizeof(float),    hipMemcpyHostToDevice));

    SparseGlobalScratch sscr;
    if (!sparse_global_scratch_alloc(sscr, stream)) {
        std::fprintf(stderr, "sparse_global_scratch_alloc failed\n");
        std::exit(2);
    }

    for (int fmt = 0; fmt <= 1; ++fmt) {
        SparseGlobalPageMeta meta;
        if (!sparse_global_page_meta_alloc(meta, max_pages, fmt, stream)) {
            std::fprintf(stderr, "sparse_global_page_meta_alloc failed\n");
            std::exit(2);
        }
        launch_kv_update_page_minmax_prefill(kv.slot, meta, 0, n_kv, stream);
        launch_sparse_global_select_pages_prefill(d_q, meta, sscr, n_tokens, pos_start, sparse_pages, stream);
        launch_fa_sparse_global_prefill(d_q, kv.slot, d_sinks, N_HEAD, d_rel, nullptr, nullptr,
                                        d_qg, N_Q, d_out, sscr.selected_pages, sscr.n_selected,
                                        n_tokens, pos_start, n_kv, stream);
        HIP_CHECK(hipStreamSynchronize(stream));

        const int n_qb = (n_tokens + SPARSE_PAGE_SIZE - 1) / SPARSE_PAGE_SIZE;
        std::vector<int32_t> h_sel((size_t)n_qb * N_HEAD_KV * SPARSE_MAX_SELECTED);
        std::vector<int32_t> h_nsel((size_t)n_qb * N_HEAD_KV);
        std::vector<float>   h_out(h_q.size());
        HIP_CHECK(hipMemcpy(h_sel.data(),  sscr.selected_pages, h_sel.size() * sizeof(int32_t),  hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_nsel.data(), sscr.n_selected,     h_nsel.size() * sizeof(int32_t), hipMemcpyDeviceToHost));
        HIP_CHECK(hipMemcpy(h_out.data(),  d_out,               h_out.size() * sizeof(float),    hipMemcpyDeviceToHost));

        // Verify sinks [0,1] present and selected pages within bounds
        bool sel_valid = true;
        for (int qb = 0; qb < n_qb; ++qb) {
            for (int h = 0; h < N_HEAD_KV; ++h) {
                const int ns = h_nsel[qb * N_HEAD_KV + h];
                if (ns < SPARSE_MIN_PAGES || ns > SPARSE_MAX_SELECTED) sel_valid = false;
                const int32_t * sp = h_sel.data() + (size_t)(qb * N_HEAD_KV + h) * SPARSE_MAX_SELECTED;
                if (sp[0] != 0 || sp[1] != 1) sel_valid = false;
                for (int i = 0; i < ns; ++i) {
                    if (sp[i] < 0 || sp[i] >= max_pages) sel_valid = false;
                }
            }
        }

        auto ref = cpu_fa_ref(h_q, kv, h_sinks, h_rel, h_qg, n_tokens, pos_start, &h_sel, &h_nsel);
        const DiffStats st = compare_vecs(h_out.data(), ref.data(), h_out.size());
        const bool ok = (sel_valid && st.nan_cnt == 0 && st.cos_sim > 0.9999 && st.max_abs < 8e-3);
        all_ok = all_ok && ok;
        std::printf("[%s] 6. Sparse Global Quest+FA WMMA (%s, n_kv=%d): sel=%s max_abs=%.4f cos=%.7f\n",
                    ok ? "PASS" : "FAIL", fmt_names[fmt], n_kv,
                    sel_valid ? "ok" : "BAD", st.max_abs, st.cos_sim);

        sparse_global_page_meta_free(meta);
    }

    sparse_global_scratch_free(sscr);
    HIP_CHECK(hipFree(d_q)); HIP_CHECK(hipFree(d_sinks)); HIP_CHECK(hipFree(d_rel));
    HIP_CHECK(hipFree(d_qg)); HIP_CHECK(hipFree(d_out));
    kv.destroy();
    return all_ok;
}

// ============================================================================
// Test 7: Vision Patch Embed Conv2D WMMA + Vision FA d=64 WMMA
// ============================================================================
bool run_test_vision_wmma(hipStream_t stream) {
    Rng rng(7007);
    const int grid_w = 8, grid_h = 6, patch_size = 20, pos_grid = 16;
    const int n_patches = grid_w * grid_h; // 48
    const int D = 1024;
    const int H_pad = grid_h * patch_size;
    const int W_pad = grid_w * patch_size;
    const int p2 = patch_size * patch_size;
    const int kdim = 3 * p2; // 1200

    std::vector<float> h_img((size_t)H_pad * W_pad * 3);
    std::vector<float> h_pw((size_t)D * kdim);
    std::vector<float> h_pos((size_t)pos_grid * pos_grid * D);
    std::vector<float> h_cls(D);
    for (float & v : h_img) v = rng.uniform(-1.0f, 1.0f);
    for (float & v : h_pw)  v = rng.uniform(-0.08f, 0.08f);
    for (float & v : h_pos) v = rng.uniform(-0.2f, 0.2f);
    for (float & v : h_cls) v = rng.uniform(-0.5f, 0.5f);

    float *d_img, *d_pw, *d_pos, *d_cls, *d_out;
    HIP_CHECK(hipMalloc(&d_img, h_img.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_pw,  h_pw.size()  * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_pos, h_pos.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_cls, h_cls.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, (size_t)(n_patches + 1) * D * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_img, h_img.data(), h_img.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_pw,  h_pw.data(),  h_pw.size()  * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_pos, h_pos.data(), h_pos.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_cls, h_cls.data(), h_cls.size() * sizeof(float), hipMemcpyHostToDevice));

    launch_vision_patch_embed_pos_cls(d_img, d_pw, d_pos, d_cls, d_out, grid_w, grid_h, patch_size, pos_grid, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> h_out((size_t)(n_patches + 1) * D);
    HIP_CHECK(hipMemcpy(h_out.data(), d_out, h_out.size() * sizeof(float), hipMemcpyDeviceToHost));

    // CPU FP64 reference for patch embed + bilinear pos embed + CLS
    std::vector<double> ref_pe((size_t)(n_patches + 1) * D, 0.0);
    for (int oc = 0; oc < D; ++oc) ref_pe[oc] = (double)h_cls[oc];
    for (int p = 0; p < n_patches; ++p) {
        const int py = p / grid_w, px = p % grid_w;
        const float sf_x = (float)grid_w / (float)pos_grid;
        const float sf_y = (float)grid_h / (float)pos_grid;
        const float x_f  = ((float)px + 0.5f) / sf_x - 0.5f;
        const float y_f  = ((float)py + 0.5f) / sf_y - 0.5f;
        int x0 = std::max(0, std::min((int)std::floor(x_f), pos_grid - 1));
        int x1 = std::max(0, std::min(x0 + 1, pos_grid - 1));
        float dx = std::min(1.0f, std::max(0.0f, x_f - (float)x0));
        int y0 = std::max(0, std::min((int)std::floor(y_f), pos_grid - 1));
        int y1 = std::max(0, std::min(y0 + 1, pos_grid - 1));
        float dy = std::min(1.0f, std::max(0.0f, y_f - (float)y0));
        const double wa = (1.0 - dx) * (1.0 - dy);
        const double wb = dx * (1.0 - dy);
        const double wc = (1.0 - dx) * dy;
        const double wd = dx * dy;

        std::vector<double> patch_vec(kdim);
        for (int k = 0; k < kdim; ++k) {
            const int c = k / p2, rem = k - c * p2;
            const int ky = rem / patch_size, kx = rem - ky * patch_size;
            patch_vec[k] = (double)h_img[((size_t)(py * patch_size + ky) * W_pad + (px * patch_size + kx)) * 3 + c];
        }
        for (int oc = 0; oc < D; ++oc) {
            double acc = 0.0;
            const float * wr = h_pw.data() + (size_t)oc * kdim;
            for (int k = 0; k < kdim; ++k) acc += (double)wr[k] * patch_vec[k];
            const double pos_v = wa * h_pos[((size_t)y0 * pos_grid + x0) * D + oc]
                               + wb * h_pos[((size_t)y0 * pos_grid + x1) * D + oc]
                               + wc * h_pos[((size_t)y1 * pos_grid + x0) * D + oc]
                               + wd * h_pos[((size_t)y1 * pos_grid + x1) * D + oc];
            ref_pe[(size_t)(p + 1) * D + oc] = acc + pos_v;
        }
    }
    const DiffStats st_pe = compare_vecs(h_out.data(), ref_pe.data(), h_out.size());
    HIP_CHECK(hipFree(d_img)); HIP_CHECK(hipFree(d_pw)); HIP_CHECK(hipFree(d_pos));
    HIP_CHECK(hipFree(d_cls)); HIP_CHECK(hipFree(d_out));

    // ---- 7b. Vision FA d=64 WMMA (non-causal, n_pos=137 non-multiple of 64/16) ----
    const int n_pos = 137, n_head = 16, d_head = 64;
    const float scale = 1.0f / std::sqrt((float)d_head);
    std::vector<__half> h_qh((size_t)n_head * n_pos * d_head);
    std::vector<__half> h_kh((size_t)n_head * n_pos * d_head);
    std::vector<__half> h_vh((size_t)n_head * n_pos * d_head);
    for (size_t i = 0; i < h_qh.size(); ++i) {
        h_qh[i] = __float2half(rng.uniform(-1.0f, 1.0f));
        h_kh[i] = __float2half(rng.uniform(-1.0f, 1.0f));
        h_vh[i] = __float2half(rng.uniform(-1.0f, 1.0f));
    }

    __half *d_qh, *d_kh, *d_vh;
    float *d_fa_out;
    HIP_CHECK(hipMalloc(&d_qh, h_qh.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_kh, h_kh.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_vh, h_vh.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_fa_out, (size_t)n_pos * n_head * d_head * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_qh, h_qh.data(), h_qh.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_kh, h_kh.data(), h_kh.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_vh, h_vh.data(), h_vh.size() * sizeof(__half), hipMemcpyHostToDevice));

    launch_vision_fa_wmma_d64(d_qh, d_kh, d_vh, d_fa_out, n_pos, n_head, scale, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> h_fa_out((size_t)n_pos * n_head * d_head);
    HIP_CHECK(hipMemcpy(h_fa_out.data(), d_fa_out, h_fa_out.size() * sizeof(float), hipMemcpyDeviceToHost));

    std::vector<double> ref_fa((size_t)n_pos * n_head * d_head, 0.0);
    for (int h = 0; h < n_head; ++h) {
        for (int q = 0; q < n_pos; ++q) {
            std::vector<double> sc(n_pos);
            double mx = -1e30;
            for (int k = 0; k < n_pos; ++k) {
                double dot = 0.0;
                for (int d = 0; d < d_head; ++d) {
                    dot += (double)__half2float(h_qh[((size_t)h * n_pos + q) * d_head + d])
                         * (double)__half2float(h_kh[((size_t)h * n_pos + k) * d_head + d]);
                }
                sc[k] = dot * (double)scale;
                if (sc[k] > mx) mx = sc[k];
            }
            double sum = 0.0;
            for (int k = 0; k < n_pos; ++k) {
                sc[k] = std::exp(sc[k] - mx);
                sum += sc[k];
            }
            const double inv = 1.0 / sum;
            for (int d = 0; d < d_head; ++d) {
                double acc = 0.0;
                for (int k = 0; k < n_pos; ++k) {
                    acc += sc[k] * inv * (double)__half2float(h_vh[((size_t)h * n_pos + k) * d_head + d]);
                }
                ref_fa[((size_t)q * n_head + h) * d_head + d] = acc;
            }
        }
    }
    const DiffStats st_fa = compare_vecs(h_fa_out.data(), ref_fa.data(), h_fa_out.size());
    HIP_CHECK(hipFree(d_qh)); HIP_CHECK(hipFree(d_kh)); HIP_CHECK(hipFree(d_vh)); HIP_CHECK(hipFree(d_fa_out));

    const bool ok = (st_pe.nan_cnt == 0 && st_fa.nan_cnt == 0 &&
                     st_pe.cos_sim > 0.99999 && st_pe.max_abs < 2e-2 &&
                     st_fa.cos_sim > 0.99999 && st_fa.max_abs < 5e-3);
    std::printf("[%s] 7. Vision WMMA (patch_embed + FA d=64): pe cos=%.7f max_abs=%.4f | fa cos=%.7f max_abs=%.4f\n",
                ok ? "PASS" : "FAIL", st_pe.cos_sim, st_pe.max_abs, st_fa.cos_sim, st_fa.max_abs);
    return ok;
}

// ============================================================================
// Test 8: Cross-Workgroup Cache Coherence Stress Test (nib_flush_stores + nib_inv_l0_l1)
// ============================================================================
__global__ void __launch_bounds__(64)
test_coherence_stress_kernel(uint32_t * __restrict__ payload,  // [N_WG, 32]
                             int32_t  * __restrict__ flags,    // [N_WG]
                             uint32_t * __restrict__ errors,   // [1]
                             int n_wg, int rounds) {
    const int bid = blockIdx.x;
    const int tid = threadIdx.x;
    const int prev = (bid == 0) ? (n_wg - 1) : (bid - 1);

    for (int r = 1; r <= rounds; ++r) {
        const uint32_t expect_self = (uint32_t)(r * 1315423911u) ^ (uint32_t)(bid * 2654435761u);
        const uint32_t expect_prev = (uint32_t)(r * 1315423911u) ^ (uint32_t)(prev * 2654435761u);

        // Producer writes 32 distinct words to its slot
        if (tid < 32) {
            payload[bid * 32 + tid] = expect_self ^ (uint32_t)tid;
        }
        nib_flush_stores();
        __syncthreads();
        if (tid == 0) {
            __atomic_store_n(&flags[bid], r, __ATOMIC_RELEASE);
        }

        // Consumer spins until neighbor reaches round r, invalidates L0/L1, verifies payload
        if (tid == 0) {
            while (__atomic_load_n(&flags[prev], __ATOMIC_ACQUIRE) < r) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __syncthreads();
        nib_inv_l0_l1();
        if (tid < 32) {
            const uint32_t got = payload[prev * 32 + tid];
            if (got != (expect_prev ^ (uint32_t)tid)) {
                atomicAdd(errors, 1u);
            }
        }
        __syncthreads();

        // Second barrier so prev doesn't overwrite before we finish reading
        if (tid == 0) {
            __atomic_store_n(&flags[n_wg + bid], r, __ATOMIC_RELEASE);
            const int next = (bid + 1 == n_wg) ? 0 : (bid + 1);
            while (__atomic_load_n(&flags[n_wg + next], __ATOMIC_ACQUIRE) < r) {
                __builtin_amdgcn_s_sleep(1);
            }
        }
        __syncthreads();
    }
}

bool run_test_coherence(hipStream_t stream) {
    const int mp = nib_device_mp_count();
    const int n_wg = std::min(32, std::max(8, mp)); // guaranteed co-resident on >=8 WGP GPUs
    const int rounds = 200;

    uint32_t *d_payload, *d_errors;
    int32_t  *d_flags;
    HIP_CHECK(hipMalloc(&d_payload, (size_t)n_wg * 32 * sizeof(uint32_t)));
    HIP_CHECK(hipMalloc(&d_flags,   (size_t)2 * n_wg * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_errors,  sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_payload, 0, (size_t)n_wg * 32 * sizeof(uint32_t)));
    HIP_CHECK(hipMemset(d_flags,   0, (size_t)2 * n_wg * sizeof(int32_t)));
    HIP_CHECK(hipMemset(d_errors,  0, sizeof(uint32_t)));

    test_coherence_stress_kernel<<<n_wg, 64, 0, stream>>>(d_payload, d_flags, d_errors, n_wg, rounds);
    HIP_CHECK(hipStreamSynchronize(stream));

    uint32_t h_err = 999;
    HIP_CHECK(hipMemcpy(&h_err, d_errors, sizeof(uint32_t), hipMemcpyDeviceToHost));
    HIP_CHECK(hipFree(d_payload)); HIP_CHECK(hipFree(d_flags)); HIP_CHECK(hipFree(d_errors));

    const bool ok = (h_err == 0);
    std::printf("[%s] 8. Cross-WG Coherence Stress (%d WGs x %d rounds): errors=%u\n",
                ok ? "PASS" : "FAIL", n_wg, rounds, h_err);
    return ok;
}

// ============================================================================
// Test 9: Cooperative Spin-Barrier Co-Residency Safety (MTP mrows on any WGP count)
// ============================================================================
bool run_test_coresident_safety(hipStream_t stream) {
    Rng rng(9009);
    const int nrows = 4; // 96*4 = 384 blocks (exceeds 32-WGP / 16-WGP -> must safely fall back!)

    std::vector<float> h_q((size_t)4 * N_EMBD), h_st((size_t)nrows * N_EMBD);
    std::vector<float> h_nw(N_EMBD), h_bias(N_EXPERT);
    std::vector<__half> h_wd((size_t)LOOP_GATE_RANK * 2 * N_EMBD);
    std::vector<__half> h_wg((size_t)N_EMBD * LOOP_GATE_RANK);
    std::vector<__half> h_wf((size_t)N_EMBD * LOOP_GATE_RANK);
    std::vector<__half> h_rw((size_t)N_EXPERT * N_EMBD);
    for (float & v : h_q)    v = rng.uniform(-0.3f, 0.3f);
    for (float & v : h_st)   v = rng.uniform(-0.5f, 0.5f);
    for (float & v : h_nw)   v = rng.uniform(0.8f, 1.2f);
    for (float & v : h_bias) v = rng.uniform(-0.05f, 0.05f);
    for (__half & h : h_wd)  h = __float2half(rng.uniform(-0.04f, 0.04f));
    for (__half & h : h_wg)  h = __float2half(rng.uniform(-0.08f, 0.08f));
    for (__half & h : h_wf)  h = __float2half(rng.uniform(-0.08f, 0.08f));
    for (__half & h : h_rw)  h = __float2half(rng.uniform(-0.05f, 0.05f));

    float *d_q, *d_st0, *d_st1, *d_o0, *d_o1, *d_gate, *d_nw, *d_bias;
    float *d_on0, *d_on1, *d_on2, *d_scr, *d_xn, *d_tw, *d_mixed;
    int32_t *d_tid;
    __half *d_wd, *d_wg, *d_wf, *d_rw;

    HIP_CHECK(hipMalloc(&d_q,     h_q.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_st0,   h_st.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_st1,   h_st.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_o0,    (size_t)nrows * 4 * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_o1,    (size_t)nrows * 3 * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_gate,  (size_t)nrows * 3 * LOOP_GATE_RANK * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_nw,    N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_bias,  N_EXPERT * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_on0,   (size_t)nrows * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_on1,   (size_t)nrows * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_on2,   (size_t)nrows * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_scr,   (size_t)nrows * 512 * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_xn,    (size_t)nrows * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_mixed, (size_t)nrows * N_EMBD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tw,    (size_t)nrows * TOP_K * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_tid,   (size_t)nrows * TOP_K * sizeof(int32_t)));
    HIP_CHECK(hipMalloc(&d_wd,    h_wd.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_wg,    h_wg.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_wf,    h_wf.size() * sizeof(__half)));
    HIP_CHECK(hipMalloc(&d_rw,    h_rw.size() * sizeof(__half)));

    HIP_CHECK(hipMemset(d_scr, 0, (size_t)nrows * 512 * sizeof(float)));
    HIP_CHECK(hipMemcpy(d_q,    h_q.data(),    h_q.size() * sizeof(float),   hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_st0,  h_st.data(),   h_st.size() * sizeof(float),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_st1,  h_st.data(),   h_st.size() * sizeof(float),  hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_nw,   h_nw.data(),   N_EMBD * sizeof(float),       hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_bias, h_bias.data(), N_EXPERT * sizeof(float),     hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wd,   h_wd.data(),   h_wd.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wg,   h_wg.data(),   h_wg.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_wf,   h_wf.data(),   h_wf.size() * sizeof(__half), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d_rw,   h_rw.data(),   h_rw.size() * sizeof(__half), hipMemcpyHostToDevice));

    const float * st_arr[2] = {d_st0, d_st1};
    const __half * wd3[3]   = {d_wd, d_wd, d_wd};
    const __half * wg3[3]   = {d_wg, d_wg, d_wg};
    const __half * wf3[3]   = {d_wf, d_wf, d_wf};
    const float  * wn3[3]   = {d_nw, d_nw, d_nw};
    float        * on3[3]   = {d_on0, d_on1, d_on2};

    // Exercises co-residency guards across decode_fused and step7h_fused (must complete with 0 hangs & finite outputs!)
    launch_mix_dual_ctrl_qkv3_norm_decode_mrows(
        d_q, 4, st_arr, 2, d_o0,
        d_q, 3, st_arr, 2, d_o1,
        wd3, wg3, wf3, wn3, d_gate, on3, RMS_EPS, stream, d_scr, nrows);
    launch_mix_ffn_norm_router_topk_decode_v2_mrows(
        d_q, st_arr, 2, d_nw, d_rw, d_bias, d_xn, d_tid, d_tw, d_scr, nrows, stream);
    launch_mix_ctrl_ffn_norm_router_topk_decode_v2_mrows(
        d_q, st_arr, 2, d_o0, d_wd, d_wg, d_wf, d_nw, d_rw, d_bias,
        d_mixed, d_gate, d_xn, d_tid, d_tw, d_scr, nrows, stream);
    launch_mix_ffn_norm_router_topk_decode_v3(
        d_q, st_arr, 2, d_nw, d_rw, d_bias, d_xn, d_tid, d_tw, d_scr, stream);
    launch_mix_ctrl_ffn_norm_router_topk_decode_v3(
        d_q, st_arr, 2, d_o0, d_wd, d_wg, d_wf, d_nw, d_rw, d_bias,
        d_mixed, d_gate, d_xn, d_tid, d_tw, d_scr, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> h_chk((size_t)nrows * N_EMBD);
    HIP_CHECK(hipMemcpy(h_chk.data(), d_xn, h_chk.size() * sizeof(float), hipMemcpyDeviceToHost));
    int nans = 0;
    for (float v : h_chk) if (!std::isfinite(v)) ++nans;

    HIP_CHECK(hipFree(d_q)); HIP_CHECK(hipFree(d_st0)); HIP_CHECK(hipFree(d_st1));
    HIP_CHECK(hipFree(d_o0)); HIP_CHECK(hipFree(d_o1)); HIP_CHECK(hipFree(d_gate));
    HIP_CHECK(hipFree(d_nw)); HIP_CHECK(hipFree(d_bias));
    HIP_CHECK(hipFree(d_on0)); HIP_CHECK(hipFree(d_on1)); HIP_CHECK(hipFree(d_on2));
    HIP_CHECK(hipFree(d_scr)); HIP_CHECK(hipFree(d_xn)); HIP_CHECK(hipFree(d_mixed));
    HIP_CHECK(hipFree(d_tw)); HIP_CHECK(hipFree(d_tid));
    HIP_CHECK(hipFree(d_wd)); HIP_CHECK(hipFree(d_wg)); HIP_CHECK(hipFree(d_wf)); HIP_CHECK(hipFree(d_rw));

    const bool ok = (nans == 0);
    std::printf("[%s] 9. Cooperative Co-Residency Safety (nrows=%d, %d MPs): nans=%d (no deadlock)\n",
                ok ? "PASS" : "FAIL", nrows, nib_device_mp_count(), nans);
    return ok;
}

// Suites 10-11 (RDNA4 FP8 WMMA); absent from gfx11 device passes.
#include "test_rdna4_fp8.hip"

} // namespace

int main() {
    const NibDeviceInfo & info = nib_device_info(0);
    std::printf("=== Nib RDNA3/RDNA4 Kernel Verification Suite ===\n");
    std::printf("Device: %s | arch=%s | family=%s | MPs=%d | wave=%d | VRAM=%.2f GiB\n",
                info.name, info.gcn_arch, info.arch_family, info.mp_count, info.warp_size,
                (double)info.total_vram_bytes / (1024.0 * 1024.0 * 1024.0));
    if (info.warp_size != 32) {
        std::fprintf(stderr, "ERROR: Requires Wave32 AMD RDNA3/RDNA4 GPU.\n");
        return 1;
    }

    hipStream_t stream = nullptr;
    HIP_CHECK(hipStreamCreate(&stream));

    int fails = 0;
    if (!run_test_raw_wmma(stream))           ++fails;
    if (!run_test_dense_wmma(stream))         ++fails;
    if (!run_test_moe_wmma(stream))           ++fails;
    if (!run_test_core_wmma(stream))          ++fails;
    if (!run_test_fa_prefill_wmma(stream))    ++fails;
    if (!run_test_sparse_global_wmma(stream)) ++fails;
    if (!run_test_vision_wmma(stream))        ++fails;
    if (!run_test_coherence(stream))          ++fails;
    if (!run_test_coresident_safety(stream))  ++fails;
    int n_suites = 9;
#if NIB_FP8_CODEGEN   // host pass (main() is host code; gfx11 device passes never see the suites)
    if (rdna4_fp8_prefill_supported()) {
        fails += run_rdna4_fp8_suites(stream, &n_suites);
    } else {
        std::printf("[SKIP] 10-11. RDNA4 FP8 WMMA suites (%s)\n",
                    rdna4_fp8_prefill_built() ? "device is not RDNA4" : "built without gfx1200/gfx1201");
    }
#endif

    HIP_CHECK(hipStreamDestroy(stream));
    std::printf("=================================================\n");
    if (fails == 0) {
        std::printf("ALL %d RDNA KERNEL SUITES PASSED (%s / %s)\n", n_suites, info.gcn_arch, info.arch_family);
        return 0;
    }
    std::printf("FAILED: %d suite(s) failed\n", fails);
    return 1;
}
