// ============================================================================
// test_rdna4_fp8.hip — RDNA4 (gfx12) FP8 WMMA verification suites 10a-10c and 11
// ============================================================================
// Included by test_rdna_kernels.hip inside its anonymous namespace (not a standalone translation
// unit): it reuses that file's Rng / DiffStats / HIP_CHECK / MoE-scratch helpers, and
// nib_kernel_test's gfx11 device passes compile none of it (NIB_FP8_CODEGEN), so its RDNA3 code
// objects are unchanged.  main() runs these suites only when rdna4_fp8_prefill_supported().
//
//  10a. Raw v_wmma_f32_16x16x16_fp8_fp8: A/B byte pairing of all 16 (half-wave, byte) operand
//       positions, C/D layout, and a 4-chunk chained FP8 GEMM vs an FP64 reference.
//  10b. Exhaustive NVFP4 -> E4M3 group decode (256 scale bytes x 16 nibbles) vs a CPU RNE E4M3
//       reference, bit-exact (incl. scale 0x00 / 0x7E / 0x7F NaN guard and the -0 nibble 0x8).
//  10c. x_to_fp8_rowscale: bit-exact bytes and scales vs CPU, round-trip bound, zero / tiny /
//       wide-range rows, Inf / NaN rows poisoned through xs, K-interleave involution.
//  11.  Grouped NVFP4 MoE on FP8 WMMA at M = 17, 73, 512 vs FP64 references (FP8 operands and
//       FP32 operands) and vs the FP16 WMMA MoE kernel, plus fused combine + sconv parity.
// ============================================================================

#if NIB_FP8_CODEGEN

// ---------------------------------------------------------------------------
// CPU OCP E4M3 reference
// ---------------------------------------------------------------------------
double e4m3_value(uint8_t b) {
    const int e = (b >> 3) & 15, m = b & 7;
    if (e == 15 && m == 7) {
        return std::nan("");
    }
    const double v = (e == 0) ? std::ldexp((double) m, -9) : std::ldexp(1.0 + m / 8.0, e - 7);
    return (b & 0x80) ? -v : v;
}

// Round-to-nearest-even to E4M3 for finite |v| <= 448 (callers guarantee the range).
uint8_t e4m3_rne(double v) {
    const uint8_t sign = std::signbit(v) ? 0x80 : 0x00;
    const double a = std::fabs(v);
    if (a == 0.0) {
        return sign;
    }
    int e2 = 0;
    std::frexp(a, &e2);
    const int E = e2 - 1;                                      // a = 1.f * 2^E
    if (E < -6) {                                              // subnormal grid 2^-9 (8 -> 2^-6)
        return sign | (uint8_t) std::nearbyint(std::ldexp(a, 9));
    }
    const int m = (int) std::nearbyint(std::ldexp(a, 3 - E)); // 8..16
    return sign | (uint8_t) (((E + 7) << 3) + (m - 8));        // 16 carries into the exponent
}

constexpr double kE2M1[8] = {0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0};

double ue4m3_scale(uint8_t sc) {
    const uint8_t u = sc & 0x7F;
    return (u == 0x7F) ? 0.0 : e4m3_value(u);
}

// The folded E4M3 byte of NVFP4 nibble `nib` under scale byte `sc` (see nib_nvfp4_fp8_table).
uint8_t nvfp4_fold_byte(uint8_t sc, int nib) {
    const double v = kE2M1[nib & 7] * ue4m3_scale(sc) / (double) NIB_NVFP4_FP8_FOLD;
    return (uint8_t) (e4m3_rne(v) | ((nib & 8) ? 0x80 : 0x00));
}

// ---------------------------------------------------------------------------
// 10a. Raw FP8 WMMA
// ---------------------------------------------------------------------------
// One lane's operand for 16-deep K chunk c of a row-major [16, K] byte matrix: lanes 0..15 take
// K 16c + 0..7, lanes 16..31 take 16c + 8..15 (byte i of the 8 = K offset i).
__device__ nib_fp8x8_t fp8_operand(const uint8_t * row, int c, int half_w) {
    uint32_t d[2] = {0u, 0u};
    for (int i = 0; i < 8; ++i) {
        d[i >> 2] |= (uint32_t) row[16 * c + 8 * half_w + i] << (8 * (i & 3));
    }
    return nib_fp8x8_t{(int) d[0], (int) d[1]};
}

// D[16, 16] = A[16, K] . B[16, K]^T, K = 16 * KC, accumulated over KC chained WMMAs.
__global__ void __launch_bounds__(32)
test_fp8_wmma_kernel(const uint8_t * __restrict__ A, const uint8_t * __restrict__ B, int KC,
                     float * __restrict__ D) {
    const int lane = threadIdx.x, l16 = lane & 15, half_w = lane >> 4;
    nib_float8_t acc = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int c = 0; c < KC; ++c) {
        acc = nib_wmma_f32_16x16x16_fp8(fp8_operand(A + l16 * 16 * KC, c, half_w),
                                        fp8_operand(B + l16 * 16 * KC, c, half_w), acc);
    }
    for (int i = 0; i < 8; ++i) {
        D[nib_wmma_c_row(half_w, i) * 16 + l16] = acc[i];
    }
}

bool run_test_fp8_wmma_raw(hipStream_t stream) {
    uint8_t * dA;
    uint8_t * dB;
    float * dD;
    HIP_CHECK(hipMalloc(&dA, 16 * 64));
    HIP_CHECK(hipMalloc(&dB, 16 * 64));
    HIP_CHECK(hipMalloc(&dD, 256 * sizeof(float)));

    // (i) Pairing: A row 0 = one-hot (1.0) at K position P; every B row holds the value P' + 1 at
    //     K position P'.  D[0][n] = (pair of P) + 1 must be P + 1 for all 16 positions and all n,
    //     and every other row 0.  Values 1..16 are exact in E4M3.
    int pair_bad = 0;
    std::vector<uint8_t> hA(16 * 16), hB(16 * 16);
    for (int n = 0; n < 16; ++n) {
        for (int k = 0; k < 16; ++k) {
            hB[n * 16 + k] = e4m3_rne((double) (k + 1));
        }
    }
    HIP_CHECK(hipMemcpy(dB, hB.data(), 256, hipMemcpyHostToDevice));
    for (int P = 0; P < 16; ++P) {
        std::fill(hA.begin(), hA.end(), 0);
        hA[P] = e4m3_rne(1.0);
        HIP_CHECK(hipMemcpy(dA, hA.data(), 256, hipMemcpyHostToDevice));
        test_fp8_wmma_kernel<<<1, 32, 0, stream>>>(dA, dB, 1, dD);
        float hD[256];
        HIP_CHECK(hipMemcpyAsync(hD, dD, sizeof(hD), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        for (int r = 0; r < 16; ++r) {
            for (int n = 0; n < 16; ++n) {
                const float want = (r == 0) ? (float) (P + 1) : 0.0f;
                pair_bad += (hD[r * 16 + n] != want);
            }
        }
    }

    // (ii) Chained K = 64 GEMM on random E4M3 operands (|x| in [2^-6, 448], random sign).
    Rng rng(10101);
    constexpr int KC = 4, K = 16 * KC;
    std::vector<uint8_t> gA(16 * K), gB(16 * K);
    auto rnd_code = [&]() {
        const int c = rng.randint(0x08, 0x7E);                 // normal, finite
        return (uint8_t) (c | (rng.randint(0, 1) << 7));
    };
    for (auto & b : gA) b = rnd_code();
    for (auto & b : gB) b = rnd_code();
    std::vector<double> ref(256);
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            double s = 0.0;
            for (int k = 0; k < K; ++k) {
                s += e4m3_value(gA[m * K + k]) * e4m3_value(gB[n * K + k]);
            }
            ref[m * 16 + n] = s;
        }
    }
    HIP_CHECK(hipMemcpy(dA, gA.data(), gA.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, gB.data(), gB.size(), hipMemcpyHostToDevice));
    test_fp8_wmma_kernel<<<1, 32, 0, stream>>>(dA, dB, KC, dD);
    std::vector<float> hD(256);
    HIP_CHECK(hipMemcpyAsync(hD.data(), dD, 256 * sizeof(float), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    const DiffStats st = compare_vecs(hD.data(), ref.data(), 256);
    double max_rel_scale = 0.0;   // error relative to the sum of |products| (FP32 accumulation bound)
    for (int m = 0; m < 16; ++m) {
        for (int n = 0; n < 16; ++n) {
            double mag = 0.0;
            for (int k = 0; k < K; ++k) {
                mag += std::fabs(e4m3_value(gA[m * K + k]) * e4m3_value(gB[n * K + k]));
            }
            max_rel_scale = std::max(max_rel_scale, std::fabs(hD[m * 16 + n] - ref[m * 16 + n]) / mag);
        }
    }
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dD));

    // FP32 summation bound: |error| <= K * 2^-24 * sum |products|.
    const bool ok = pair_bad == 0 && st.nan_cnt == 0 && st.cos_sim > 0.999999 &&
                    max_rel_scale <= K * std::ldexp(1.0, -24);
    std::printf("[%s] 10a. Raw FP8 WMMA: pairing mismatches=%d/4096 | K=64 GEMM cos=%.9f max_abs=%.3e "
                "max_err/sum|p|=%.2e\n",
                ok ? "PASS" : "FAIL", pair_bad, st.cos_sim, st.max_abs, max_rel_scale);
    return ok;
}

// ---------------------------------------------------------------------------
// 10b. Exhaustive NVFP4 -> E4M3 decode
// ---------------------------------------------------------------------------
// Thread t = scale byte t (all 256, so the ignored sign bit is covered too) decodes one group
// whose nibble k is k: bytes j = 0..7 hold (low nibble j, high nibble j + 8).
__global__ void __launch_bounds__(256)
test_nvfp4_decode_kernel(uint4 * __restrict__ out) {
    __shared__ uint2 s_lut[256];
    s_lut[threadIdx.x] = nib_nvfp4_fp8_table(threadIdx.x);    // through LDS, as the MoE kernels do
    __syncthreads();
    out[threadIdx.x] = nib_nvfp4_group_to_fp8(0xB3A29180u, 0xF7E6D5C4u, s_lut[threadIdx.x]);
}

bool run_test_nvfp4_fp8_decode(hipStream_t stream) {
    uint4 * dOut;
    HIP_CHECK(hipMalloc(&dOut, 256 * sizeof(uint4)));
    test_nvfp4_decode_kernel<<<1, 256, 0, stream>>>(dOut);
    std::vector<uint4> h(256);
    HIP_CHECK(hipMemcpyAsync(h.data(), dOut, 256 * sizeof(uint4), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(dOut));

    int bad = 0, nan_bytes = 0;
    bool special_ok = true;
    for (int sc = 0; sc < 256; ++sc) {
        const uint32_t w[4] = {h[sc].x, h[sc].y, h[sc].z, h[sc].w};
        for (int k = 0; k < 16; ++k) {
            const uint8_t got  = (uint8_t) (w[k >> 2] >> (8 * (k & 3)));
            const uint8_t want = nvfp4_fold_byte((uint8_t) sc, k);
            nan_bytes += ((got & 0x7F) == 0x7F);
            if (got != want) {
                if (bad < 4) {
                    std::printf("       sc=0x%02X nibble=0x%X: got 0x%02X want 0x%02X\n", sc, k, got, want);
                }
                ++bad;
                if ((sc & 0x7F) == 0x00 || (sc & 0x7F) == 0x7E || (sc & 0x7F) == 0x7F) {
                    special_ok = false;
                }
            }
        }
    }
    // Spot facts: the largest entry, 6 * 448 / 8 = 336, is a tie between 320 and 352 and rounds to
    // even: 320 = 0x7A (element 7 = nibble 7 of scale 0x7E); nibble 0x8 decodes to -0 (0x80).
    const uint8_t max_byte = (uint8_t) (h[0x7E].y >> 24);
    const uint8_t neg_zero = (uint8_t) (h[0x40].z & 0xFF);
    const bool ok = bad == 0 && nan_bytes == 0 && special_ok && max_byte == 0x7A && neg_zero == 0x80;
    std::printf("[%s] 10b. NVFP4->E4M3 decode (256 scales x 16 nibbles): mismatches=%d NaN bytes=%d | "
                "sc 0x00/0x7E/0x7F %s | max 0x%02X (320) | nibble 0x8 -> 0x%02X\n",
                ok ? "PASS" : "FAIL", bad, nan_bytes, special_ok ? "ok" : "BAD", max_byte, neg_zero);
    return ok;
}

// ---------------------------------------------------------------------------
// 10c. x_to_fp8_rowscale
// ---------------------------------------------------------------------------
// CPU model of the quantiser: identical FP32 operations, so bytes and scales must match exactly.
void cpu_x_to_fp8_row(const float * x, uint8_t * x8, float * xs) {
    float amax = 0.0f;
    bool finite = true;
    for (int k = 0; k < N_EMBD; ++k) {
        finite &= std::isfinite(x[k]);
        amax = std::max(amax, std::fabs(x[k]));
    }
    float sa = 0.0f;
    if (!finite) {
        *xs = std::nanf("");
    } else if (amax < NIB_FP8_AMAX_MIN) {
        *xs = 0.0f;
    } else {
        sa  = NIB_FP8_MAX / amax;
        *xs = amax * (1.0f / NIB_FP8_MAX);
    }
    for (int k = 0; k < N_EMBD; ++k) {
        const float q = (sa == 0.0f) ? 0.0f : std::min(std::max(x[k] * sa, -NIB_FP8_MAX), NIB_FP8_MAX);
        x8[fp8_moe_kpos(k)] = e4m3_rne((double) q);
    }
}

bool run_test_x_to_fp8(hipStream_t stream) {
    Rng rng(10303);
    constexpr int M = 40;
    std::vector<float> hX((size_t) M * N_EMBD);
    for (int r = 0; r < M; ++r) {
        float * x = hX.data() + (size_t) r * N_EMBD;
        for (int k = 0; k < N_EMBD; ++k) {
            x[k] = rng.uniform(-1.0f, 1.0f) * rng.uniform(0.0f, 1.0f);
        }
        if (r % 5 == 0) {
            x[rng.randint(0, N_EMBD - 1)] = 60.0f;             // RMSNorm-style outlier channel
        }
    }
    float * zero_row  = hX.data() + (size_t) 32 * N_EMBD;
    float * tiny_row  = hX.data() + (size_t) 33 * N_EMBD;
    float * nan_row   = hX.data() + (size_t) 34 * N_EMBD;
    float * inf_row   = hX.data() + (size_t) 35 * N_EMBD;
    float * wide_row  = hX.data() + (size_t) 36 * N_EMBD;
    float * flush_row = hX.data() + (size_t) 37 * N_EMBD;      // amax > FLT_MIN but 448/amax would overflow FP32
    for (int k = 0; k < N_EMBD; ++k) {
        zero_row[k] = 0.0f;
        tiny_row[k] *= 1e-30f;
        wide_row[k] = std::ldexp(rng.uniform(0.5f, 1.0f), rng.randint(-30, 10)) * (k & 1 ? -1.0f : 1.0f);
        flush_row[k] = (k == 17) ? 1.0e-37f : 0.0f;
    }
    nan_row[777] = std::nanf("");
    inf_row[5]   = -INFINITY;

    float * dX;
    uint8_t * dX8;
    float * dXs;
    HIP_CHECK(hipMalloc(&dX, hX.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dX8, (size_t) M * N_EMBD));
    HIP_CHECK(hipMalloc(&dXs, M * sizeof(float)));
    HIP_CHECK(hipMemcpy(dX, hX.data(), hX.size() * sizeof(float), hipMemcpyHostToDevice));
    launch_x_to_fp8_rowscale(dX, dX8, dXs, M, stream);
    std::vector<uint8_t> g8((size_t) M * N_EMBD);
    std::vector<float> gs(M);
    HIP_CHECK(hipMemcpyAsync(g8.data(), dX8, g8.size(), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipMemcpyAsync(gs.data(), dXs, M * sizeof(float), hipMemcpyDeviceToHost, stream));
    HIP_CHECK(hipStreamSynchronize(stream));
    HIP_CHECK(hipFree(dX));
    HIP_CHECK(hipFree(dX8));
    HIP_CHECK(hipFree(dXs));

    int byte_bad = 0, scale_bad = 0, nan_bytes = 0, bound_bad = 0;
    std::vector<uint8_t> c8(N_EMBD);
    for (int r = 0; r < M; ++r) {
        const float * x = hX.data() + (size_t) r * N_EMBD;
        const uint8_t * g = g8.data() + (size_t) r * N_EMBD;
        float cs = 0.0f;
        cpu_x_to_fp8_row(x, c8.data(), &cs);
        const bool poisoned = std::isnan(cs);
        scale_bad += poisoned ? !std::isnan(gs[r]) : (std::memcmp(&cs, &gs[r], sizeof(float)) != 0);
        for (int k = 0; k < N_EMBD; ++k) {
            byte_bad  += (g[k] != c8[k]);
            nan_bytes += ((g[k] & 0x7F) == 0x7F);
            if (!poisoned && gs[r] > 0.0f) {
                // |dequant - x| <= half an E4M3 ulp: 2^-4 relative (normal) or 2^-10 * xs (subnormal).
                const double xk  = x[k];
                const double deq = e4m3_value(g[fp8_moe_kpos(k)]) * (double) gs[r];
                const double tol = std::max(std::ldexp(std::fabs(xk), -4), std::ldexp((double) gs[r], -10)) * (1 + 1e-6);
                bound_bad += !(std::fabs(deq - xk) <= tol);
            }
        }
    }
    int kpos_bad = 0;
    for (int k = 0; k < N_EMBD; ++k) {
        kpos_bad += (fp8_moe_kpos(fp8_moe_kpos(k)) != k) || ((fp8_moe_kpos(k) & ~31) != (k & ~31));
    }
    const bool special = gs[32] == 0.0f && gs[33] > 0.0f && std::isnan(gs[34]) && std::isnan(gs[35]) && gs[37] == 0.0f;
    const bool ok = byte_bad == 0 && scale_bad == 0 && nan_bytes == 0 && bound_bad == 0 && kpos_bad == 0 && special;
    std::printf("[%s] 10c. x_to_fp8_rowscale (M=%d): byte mismatches=%d scale mismatches=%d NaN bytes=%d "
                "round-trip violations=%d | zero/tiny/flush/NaN/Inf rows %s | kpos involution %s\n",
                ok ? "PASS" : "FAIL", M, byte_bad, scale_bad, nan_bytes, bound_bad,
                special ? "ok" : "BAD", kpos_bad == 0 ? "ok" : "BAD");
    return ok;
}

// ---------------------------------------------------------------------------
// 11. Grouped NVFP4 MoE on FP8 WMMA
// ---------------------------------------------------------------------------
double situ_ref(double g, double u) {
    return SITU_BETA * std::tanh(g / SITU_BETA) / (1.0 + std::exp(-g)) * SITU_LINEAR_BETA * std::tanh(u / SITU_LINEAR_BETA);
}

struct Fp4ExpertsHost {
    int rows = 0, K = 0;
    std::vector<uint8_t> qs, sc;
    std::vector<float>   s2;
    // Expert e as dense [rows, K]: exact NVFP4 values / scale2 and their E4M3 folds (x FOLD).
    void decode(int e, std::vector<float> & exact, std::vector<float> & fold) const {
        float ex_tab[128][16], fd_tab[128][16];
        for (int sc7 = 0; sc7 < 128; ++sc7) {
            for (int nib = 0; nib < 16; ++nib) {
                ex_tab[sc7][nib] = (float) (kE2M1[nib & 7] * ue4m3_scale((uint8_t) sc7) * ((nib & 8) ? -1.0 : 1.0));
                fd_tab[sc7][nib] = (float) (e4m3_value(nvfp4_fold_byte((uint8_t) sc7, nib)) * NIB_NVFP4_FP8_FOLD);
            }
        }
        exact.resize((size_t) rows * K);
        fold.resize((size_t) rows * K);
        for (int r = 0; r < rows; ++r) {
            const size_t row = (size_t) e * rows + r;
            const uint8_t * q = &qs[row * (K / 2)];
            const uint8_t * s = &sc[row * (K / 16)];
            float * ex = &exact[(size_t) r * K];
            float * fd = &fold[(size_t) r * K];
            for (int g = 0; g < K / 16; ++g) {               // 16-weight group g: bytes 8g..8g+7
                const int sc7 = s[g] & 0x7F;
                for (int j = 0; j < 8; ++j) {
                    const uint8_t b = q[8 * g + j];
                    ex[16 * g + j]     = ex_tab[sc7][b & 0xF];
                    fd[16 * g + j]     = fd_tab[sc7][b & 0xF];
                    ex[16 * g + j + 8] = ex_tab[sc7][b >> 4];
                    fd[16 * g + j + 8] = fd_tab[sc7][b >> 4];
                }
            }
        }
    }
};

Fp4ExpertsHost make_fp4_experts(Rng & rng, int n_exp, int rows, int K) {
    Fp4ExpertsHost h;
    h.rows = rows;
    h.K = K;
    h.qs.resize((size_t) n_exp * rows * K / 2);
    h.sc.resize((size_t) n_exp * rows * K / 16);
    h.s2.resize(n_exp);
    for (auto & b : h.qs) b = (uint8_t) (rng.next_u64() & 0xFF);
    for (auto & b : h.sc) {
        const int u = rng.randint(0, 999);
        // Mostly the band real NVFP4 experts use (2 .. 448, incl. 0x7E), plus rare 0x00 / 0x7F.
        b = (uint8_t) (u == 0 ? 0x00 : u == 1 ? 0x7F : u == 2 ? 0x7E : rng.randint(0x40, 0x7D));
    }
    for (auto & v : h.s2) v = rng.uniform(1.0e-4f, 3.0e-4f);
    return h;
}

SoaNVFP4Experts upload_fp4_experts(const Fp4ExpertsHost & h, int n_exp) {
    SoaNVFP4Experts d;
    d.n_exp = n_exp;
    d.rows = h.rows;
    d.K = h.K;
    HIP_CHECK(hipMalloc(&d.qs, h.qs.size()));
    HIP_CHECK(hipMalloc(&d.scales, h.sc.size()));
    HIP_CHECK(hipMalloc(&d.scale2, h.s2.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(d.qs, h.qs.data(), h.qs.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d.scales, h.sc.data(), h.sc.size(), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(d.scale2, h.s2.data(), h.s2.size() * sizeof(float), hipMemcpyHostToDevice));
    return d;
}

void free_fp4_experts(SoaNVFP4Experts & d) {
    HIP_CHECK(hipFree(d.qs));
    HIP_CHECK(hipFree(d.scales));
    HIP_CHECK(hipFree(d.scale2));
}

// FP64 references for the tokens with check[t]: ref8 uses exactly the FP8 kernel's operands (E4M3
// activations, folded weights, E4M3 SITU at FP8_MOE_ACT_SCALE); ref32 uses FP32 activations and
// exact NVFP4 weights.  Experts are decoded one at a time.
void moe_fp8_reference(const Fp4ExpertsHost & gu, const Fp4ExpertsHost & dn, int n_exp, int M,
                       const std::vector<float> & X, const std::vector<int32_t> & ids, const std::vector<float> & w,
                       const std::vector<char> & check, std::vector<double> & ref8, std::vector<double> & ref32) {
    std::vector<uint8_t> x8((size_t) M * N_EMBD);
    std::vector<float> xs(M);
    for (int t = 0; t < M; ++t) {
        cpu_x_to_fp8_row(&X[(size_t) t * N_EMBD], &x8[(size_t) t * N_EMBD], &xs[t]);
    }
    ref8.assign((size_t) M * N_EMBD, 0.0);
    ref32.assign((size_t) M * N_EMBD, 0.0);
    std::vector<float> gu_ex, gu_fd, dn_ex, dn_fd;
    std::vector<double> xq(N_EMBD), act8(N_FF_EXP), act32(N_FF_EXP);
    for (int e = 0; e < n_exp; ++e) {
        bool used = false;
        for (int t = 0; t < M && !used; ++t) {
            for (int k = 0; k < TOP_K; ++k) {
                used |= check[t] && ids[(size_t) t * TOP_K + k] == e;
            }
        }
        if (!used) {
            continue;
        }
        gu.decode(e, gu_ex, gu_fd);
        dn.decode(e, dn_ex, dn_fd);
        for (int t = 0; t < M; ++t) {
            if (!check[t]) {
                continue;
            }
            double we = 0.0;                                   // repeated experts sum their weights
            for (int k = 0; k < TOP_K; ++k) {
                if (ids[(size_t) t * TOP_K + k] == e) {
                    we += w[(size_t) t * TOP_K + k];
                }
            }
            if (we == 0.0) {
                continue;
            }
            const float * x = &X[(size_t) t * N_EMBD];
            for (int k = 0; k < N_EMBD; ++k) {
                xq[k] = e4m3_value(x8[(size_t) t * N_EMBD + fp8_moe_kpos(k)]) * xs[t];
            }
            for (int n = 0; n < N_FF_EXP; ++n) {
                const float * g_ex = &gu_ex[(size_t) n * N_EMBD];
                const float * u_ex = &gu_ex[(size_t) (N_FF_EXP + n) * N_EMBD];
                const float * g_fd = &gu_fd[(size_t) n * N_EMBD];
                const float * u_fd = &gu_fd[(size_t) (N_FF_EXP + n) * N_EMBD];
                double g8 = 0, u8 = 0, g32 = 0, u32 = 0;
                for (int k = 0; k < N_EMBD; ++k) {
                    g8  += xq[k] * g_fd[k];
                    u8  += xq[k] * u_fd[k];
                    g32 += (double) x[k] * g_ex[k];
                    u32 += (double) x[k] * u_ex[k];
                }
                const double s2 = gu.s2[e];
                act8[n]  = e4m3_value(e4m3_rne(situ_ref(g8 * s2, u8 * s2) * FP8_MOE_ACT_SCALE)) / FP8_MOE_ACT_SCALE;
                act32[n] = situ_ref(g32 * s2, u32 * s2);
            }
            for (int n = 0; n < N_EMBD; ++n) {
                const float * d_ex = &dn_ex[(size_t) n * N_FF_EXP];
                const float * d_fd = &dn_fd[(size_t) n * N_FF_EXP];
                double y8 = 0, y32 = 0;
                for (int k = 0; k < N_FF_EXP; ++k) {
                    y8  += act8[k] * d_fd[k];
                    y32 += act32[k] * d_ex[k];
                }
                ref8[(size_t) t * N_EMBD + n]  += y8 * dn.s2[e] * we;
                ref32[(size_t) t * N_EMBD + n] += y32 * dn.s2[e] * we;
            }
        }
    }
}

DiffStats compare_checked(const std::vector<float> & got, const std::vector<double> & ref,
                          const std::vector<char> & check) {
    std::vector<float> g;
    std::vector<double> r;
    for (size_t t = 0; t < check.size(); ++t) {
        if (check[t]) {
            g.insert(g.end(), got.begin() + t * N_EMBD, got.begin() + (t + 1) * N_EMBD);
            r.insert(r.end(), ref.begin() + t * N_EMBD, ref.begin() + (t + 1) * N_EMBD);
        }
    }
    return compare_vecs(g.data(), r.data(), g.size());
}

bool run_test_moe_fp8(hipStream_t stream) {
    Rng rng(11011);
    constexpr int N_EXP = 16;                 // experts 0..15 only (low VRAM, like Test 3)
    const Fp4ExpertsHost h_gu = make_fp4_experts(rng, N_EXP, N_GATE_UP_EXP, N_EMBD);
    const Fp4ExpertsHost h_dn = make_fp4_experts(rng, N_EXP, N_EMBD, N_FF_EXP);
    SoaNVFP4Experts gu = upload_fp4_experts(h_gu, N_EXP);
    SoaNVFP4Experts dn = upload_fp4_experts(h_dn, N_EXP);
    // Size scratch to FP8_MAX_UBATCH=1024 with the reclaimed perm_x/perm_act footprint used by
    // InklingEngine (verifies both M=1024 single-chunk routing and the tightened scratch contract).
    MoePrefillScratch scratch{};
    {
        constexpr size_t cap       = FP8_MAX_UBATCH;
        constexpr size_t max_slots = cap * TOP_K;
        constexpr size_t perm_x_bytes = std::max(
            (size_t) MAX_UBATCH * N_EMBD * sizeof(_Float16) + (size_t) MAX_UBATCH * TOP_K * sizeof(int32_t),
            cap * N_EMBD + ((cap * sizeof(float) + 15u) & ~size_t(15)) + cap * TOP_K * sizeof(int32_t));
        constexpr size_t perm_act_bytes = std::max(
            (size_t) MAX_UBATCH * TOP_K * N_FF_EXP * sizeof(_Float16),
            max_slots * N_FF_EXP * sizeof(uint8_t));
        HIP_CHECK(hipMalloc(&scratch.exp_counts,  N_EXPERT * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&scratch.exp_offsets, (N_EXPERT + 1) * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&scratch.exp_cursors, N_EXPERT * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&scratch.perm_tok,    max_slots * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&scratch.perm_w,      max_slots * sizeof(float)));
        HIP_CHECK(hipMalloc(&scratch.perm_x,      perm_x_bytes));
        HIP_CHECK(hipMalloc(&scratch.perm_act,    perm_act_bytes));
        HIP_CHECK(hipMalloc(&scratch.perm_down,   max_slots * N_EMBD * sizeof(float)));
        HIP_CHECK(hipMalloc(&scratch.tile_exp,    (max_slots + N_EXPERT) * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&scratch.tile_m0,     (max_slots + N_EXPERT) * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&scratch.d_num_tiles, sizeof(int32_t)));
    }

    bool all_ok = true;
    for (const int M : {17, 73, 512, 1024}) {
        std::vector<float> hX((size_t) M * N_EMBD);
        for (float & v : hX) v = rng.uniform(-0.4f, 0.4f);
        for (int t = 0; t < M; ++t) {
            hX[(size_t) t * N_EMBD + rng.randint(0, N_EMBD - 1)] *= 25.0f;   // one outlier channel
        }
        std::vector<int32_t> ids((size_t) M * TOP_K);
        std::vector<float> w((size_t) M * TOP_K);
        for (int t = 0; t < M; ++t) {
            int pool[N_EXP];
            for (int e = 0; e < N_EXP; ++e) pool[e] = e;
            float wsum = 0.0f;
            for (int k = 0; k < TOP_K; ++k) {
                std::swap(pool[k], pool[rng.randint(k, N_EXP - 1)]);
                ids[(size_t) t * TOP_K + k] = pool[k];
                w[(size_t) t * TOP_K + k] = rng.uniform(0.5f, 1.5f);
                wsum += w[(size_t) t * TOP_K + k];
            }
            for (int k = 0; k < TOP_K; ++k) w[(size_t) t * TOP_K + k] *= EXPERT_W_SCALE / wsum;
        }
        if (M == 73) {
            ids[5 * TOP_K + 3] = ids[5 * TOP_K + 1];          // a repeated expert (summed weights)
            ids[9 * TOP_K + 7] = -1;                          // an out-of-range id (ignored)
        }
        // FP64 references on every token for M <= 73 and on every 8th/16th token at M = 512/1024
        // (64 checked tokens still touch every expert and every M tile).
        std::vector<char> check(M);
        const int stride = (M <= 73) ? 1 : (M / 64);
        for (int t = 0; t < M; ++t) check[t] = (M <= 73) || (t % stride == 3);

        float *dX, *dW, *dY8, *dY16, *dYsc, *dConv, *dHist, *dRes;
        int32_t * dIds;
        const size_t ny = (size_t) M * N_EMBD;
        HIP_CHECK(hipMalloc(&dX, ny * sizeof(float)));
        HIP_CHECK(hipMalloc(&dIds, ids.size() * sizeof(int32_t)));
        HIP_CHECK(hipMalloc(&dW, w.size() * sizeof(float)));
        HIP_CHECK(hipMalloc(&dY8, ny * sizeof(float)));
        HIP_CHECK(hipMalloc(&dY16, ny * sizeof(float)));
        HIP_CHECK(hipMalloc(&dYsc, ny * sizeof(float)));
        HIP_CHECK(hipMalloc(&dConv, (size_t) N_EMBD * 4 * sizeof(float)));
        HIP_CHECK(hipMalloc(&dHist, (size_t) N_EMBD * 3 * sizeof(float)));
        HIP_CHECK(hipMalloc(&dRes, ny * sizeof(float)));
        HIP_CHECK(hipMemcpy(dX, hX.data(), ny * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dIds, ids.data(), ids.size() * sizeof(int32_t), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dW, w.data(), w.size() * sizeof(float), hipMemcpyHostToDevice));

        std::vector<float> hConv((size_t) N_EMBD * 4), hHist((size_t) N_EMBD * 3), hRes(ny);
        for (float & v : hConv) v = rng.uniform(-0.5f, 0.5f);
        for (float & v : hHist) v = rng.uniform(-1.0f, 1.0f);
        for (float & v : hRes) v = rng.uniform(-1.0f, 1.0f);
        HIP_CHECK(hipMemcpy(dConv, hConv.data(), hConv.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dHist, hHist.data(), hHist.size() * sizeof(float), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(dRes, hRes.data(), ny * sizeof(float), hipMemcpyHostToDevice));

        launch_moe_nvfp4_prefill_fp8_wmma(gu, dn, dX, dIds, dW, scratch, dY8, M, stream,
                                          nullptr, nullptr, nullptr, FP8_MAX_UBATCH);
        launch_moe_nvfp4_prefill_wmma(gu, dn, dX, dIds, dW, scratch, dY16, M, stream);
        launch_moe_nvfp4_prefill_fp8_wmma(gu, dn, dX, dIds, dW, scratch, dYsc, M, stream,
                                          dConv, dHist, dRes, FP8_MAX_UBATCH);
        std::vector<float> y8(ny), y16(ny), ysc(ny), hist_out((size_t) N_EMBD * 3);
        HIP_CHECK(hipMemcpyAsync(y8.data(), dY8, ny * sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(y16.data(), dY16, ny * sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(ysc.data(), dYsc, ny * sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipMemcpyAsync(hist_out.data(), dHist, hist_out.size() * sizeof(float), hipMemcpyDeviceToHost, stream));
        HIP_CHECK(hipStreamSynchronize(stream));
        for (float * p : {dX, dW, dY8, dY16, dYsc, dConv, dHist, dRes}) HIP_CHECK(hipFree(p));
        HIP_CHECK(hipFree(dIds));

        std::vector<double> ref8, ref32;
        moe_fp8_reference(h_gu, h_dn, N_EXP, M, hX, ids, w, check, ref8, ref32);
        const DiffStats s8  = compare_checked(y8, ref8, check);
        const DiffStats s32 = compare_checked(y8, ref32, check);
        const DiffStats s16 = compare_vecs_f32(y8.data(), y16.data(), ny);

        // Fused combine + sconv + residual vs the same applied on the CPU to the plain combine
        // (identical FP32 operation order), and the history = the batch's last 3 MoE outputs.
        double sc_max = 0.0, hist_max = 0.0;
        constexpr float kScale = 0.44721359549995793928f;
        for (int c = 0; c < N_EMBD; ++c) {
            float h0 = hHist[c * 3 + 0], h1 = hHist[c * 3 + 1], h2 = hHist[c * 3 + 2];
            const float * cw = &hConv[(size_t) c * 4];
            for (int t = 0; t < M; ++t) {
                const float x = y8[(size_t) t * N_EMBD + c];
                const float acc = x + h0 * cw[0] + h1 * cw[1] + h2 * cw[2] + x * cw[3];
                const float want = std::fma(acc, kScale, hRes[(size_t) t * N_EMBD + c]);
                sc_max = std::max(sc_max, (double) std::fabs(ysc[(size_t) t * N_EMBD + c] - want));
                h0 = h1; h1 = h2; h2 = x;
            }
            hist_max = std::max({hist_max, (double) std::fabs(hist_out[c * 3 + 0] - h0),
                                 (double) std::fabs(hist_out[c * 3 + 1] - h1),
                                 (double) std::fabs(hist_out[c * 3 + 2] - h2)});
        }
        // Exactness gate: identical FP8 operands -> cos > 0.99999 (measured 1.0000000).
        // Quantisation budget (FP32 operands / FP16 kernel): E4M3 keeps 3 mantissa bits, i.e.
        // ~2.4% RMS relative noise on each quantised X / SITU element, and this test's uniform
        // nibbles and odd-mantissa scales make ~20% of the folded weights RNE ties (+-3..6%).
        // Together that predicts cos ~0.998 (measured 0.9984); 0.997 bounds it while any layout or
        // scale defect would fall far below.  (The FP8 path is opt-in for this reason.)
        const bool ok = s8.nan_cnt == 0 && s32.nan_cnt == 0 && s16.nan_cnt == 0 &&
                        s8.cos_sim > 0.99999 && s32.cos_sim > 0.997 && s16.cos_sim > 0.997 &&
                        sc_max < 1e-4 && hist_max == 0.0;
        all_ok &= ok;
        std::printf("[%s] 11. Grouped MoE FP8 WMMA (M=%d): vs FP64[FP8 ops] cos=%.7f | vs FP64[FP32 ops] cos=%.6f | "
                    "vs FP16 WMMA cos=%.6f max_abs=%.4f | sconv max_abs=%.1e hist=%.1e | NaNs=%d\n",
                    ok ? "PASS" : "FAIL", M, s8.cos_sim, s32.cos_sim, s16.cos_sim, s16.max_abs, sc_max, hist_max,
                    s8.nan_cnt + s16.nan_cnt);
    }
    free_moe_scratch(scratch);
    free_fp4_experts(gu);
    free_fp4_experts(dn);
    return all_ok;
}

// Runs suites 10a-10c and 11; returns the number of failed suites and adds the number run.
int run_rdna4_fp8_suites(hipStream_t stream, int * n_suites) {
    int fails = 0;
    fails += !run_test_fp8_wmma_raw(stream);
    fails += !run_test_nvfp4_fp8_decode(stream);
    fails += !run_test_x_to_fp8(stream);
    fails += !run_test_moe_fp8(stream);
    *n_suites += 4;
    return fails;
}

#endif // NIB_FP8_CODEGEN
