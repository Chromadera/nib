// Micro-benchmark + bit-exactness check for launch_fa_sparse_global_prefill
// (inkling_fa_prefill_gqa_wmma_sparse_global_kernel, gfx1100).
// Links against a prebuilt kernels_sparse_global.o so the same source can be
// A/B tested across compiler/flag variants.
// Build (run from the repo root; build/ must already contain the object under test):
//   /opt/rocm/bin/hipcc --offload-arch=gfx1100 -O3 --std=c++17 -DGPU_TARGET=gfx1100 \
//       -DWITH_ROCM -DWMMA_M=64 -I include -I src -c <this file> -o bench.o
//   /opt/rocm/bin/hipcc bench.o build/kernels_sparse_global.o -o bench -lamdhip64
// Args: <n_kv> <n_tokens> <sparse_pages> <iters> <rel_mode> <kv_type> <meta_fmt> [out_dump]
// KNOWN HARNESS BUG: kv_type != 0 (Q4_1/Q4_0 KV) dies with an illegal memory access at
//   n_kv >= 16384, and the CPU reference disagrees for those KV types at any depth. The
//   engine serves q4_1 at 262,848 context fine, so this harness (not the kernel) needs the
//   Q4 KV buffer sizing fixed. Use kv_type=0.
#include "inkling_engine.h"
#include "sparse_global.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <vector>

#define HIP_CHECK(x) do { hipError_t e_ = (x); if (e_ != hipSuccess) { \
    std::fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e_), __FILE__, __LINE__); \
    return 1; } } while (0)

static uint32_t rng_state = 12345u;
static inline uint32_t xrand() {
    rng_state = rng_state * 1664525u + 1013904223u;
    return rng_state;
}
static inline float runiform(float lo, float hi) {
    return lo + (hi - lo) * ((float)(xrand() >> 8) / 16777216.0f);
}

int main(int argc, char ** argv) {
    const int n_kv      = argc > 1 ? std::atoi(argv[1]) : 32768;
    const int n_tokens  = argc > 2 ? std::atoi(argv[2]) : 512;
    const int sparse_pages = argc > 3 ? std::atoi(argv[3]) : 224;
    const int iters     = argc > 4 ? std::atoi(argv[4]) : 12;
    const int rel_mode  = argc > 5 ? std::atoi(argv[5]) : 0;   // 0 = rel_table, 1 = r_prep
    const int kv_type   = argc > 6 ? std::atoi(argv[6]) : 0;   // 0 Q8_0, 1 Q4_1, 2 Q4_0
    const int meta_fmt  = argc > 7 ? std::atoi(argv[7]) : 0;   // 0 Q8, 1 Q4

    hipDeviceProp_t prop{};
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    std::printf("device=%s arch=%s MPs=%d | n_kv=%d n_tokens=%d pages=%d rel=%d kv=%d meta=%d\n",
                prop.name, prop.gcnArchName, prop.multiProcessorCount,
                n_kv, n_tokens, sparse_pages, rel_mode, kv_type, meta_fmt);

    hipStream_t stream;
    HIP_CHECK(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));

    const int cap = n_kv;
    const int qs  = (kv_type == inkling::KV_Q8_0) ? inkling::HEAD_DIM : inkling::HEAD_DIM / 2;

    inkling::KVCacheSlot kv;
    kv.cap = cap;
    kv.kv_type = kv_type;
    kv.is_swa = false;
    HIP_CHECK(hipMalloc(&kv.k_qs, (size_t)inkling::N_HEAD_KV * cap * qs));
    HIP_CHECK(hipMalloc(&kv.v_qs, (size_t)inkling::N_HEAD_KV * cap * qs));
    HIP_CHECK(hipMalloc(&kv.k_d,  (size_t)inkling::N_HEAD_KV * cap * 4 * sizeof(__half)));
    HIP_CHECK(hipMalloc(&kv.v_d,  (size_t)inkling::N_HEAD_KV * cap * 4 * sizeof(__half)));
    HIP_CHECK(hipMalloc(&kv.k_pos, (size_t)cap * sizeof(int32_t)));
    {
        std::vector<int8_t> hq((size_t)inkling::N_HEAD_KV * cap * qs);
        for (auto & b : hq) b = (int8_t)(int32_t)(xrand() % 201) - 100;
        HIP_CHECK(hipMemcpy(kv.k_qs, hq.data(), hq.size(), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(kv.v_qs, hq.data(), hq.size(), hipMemcpyHostToDevice));
        std::vector<__half> hd((size_t)inkling::N_HEAD_KV * cap * 4, __float2half(0.01f));
        HIP_CHECK(hipMemcpy(kv.k_d, hd.data(), hd.size() * sizeof(__half), hipMemcpyHostToDevice));
        HIP_CHECK(hipMemcpy(kv.v_d, hd.data(), hd.size() * sizeof(__half), hipMemcpyHostToDevice));
        std::vector<int32_t> hp(cap);
        for (int i = 0; i < cap; ++i) hp[i] = i;
        HIP_CHECK(hipMemcpy(kv.k_pos, hp.data(), hp.size() * sizeof(int32_t), hipMemcpyHostToDevice));
    }

    const int max_pages = (n_kv + inkling::SPARSE_PAGE_SIZE - 1) / inkling::SPARSE_PAGE_SIZE;
    inkling::SparseGlobalPageMeta meta;
    if (!inkling::sparse_global_page_meta_alloc(meta, max_pages, meta_fmt, stream)) {
        std::fprintf(stderr, "page_meta_alloc failed\n"); return 2;
    }
    inkling::SparseGlobalScratch sscr;
    if (!inkling::sparse_global_scratch_alloc(sscr, stream)) {
        std::fprintf(stderr, "scratch_alloc failed\n"); return 2;
    }

    const size_t nq = (size_t)n_tokens * inkling::N_HEAD * inkling::HEAD_DIM;
    float *d_q, *d_sinks, *d_rel, *d_rprep, *d_bias, *d_qg, *d_out;
    HIP_CHECK(hipMalloc(&d_q, nq * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_sinks, (size_t)n_tokens * inkling::N_HEAD * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_out, nq * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_qg, nq * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rel, (size_t)n_tokens * inkling::N_HEAD * inkling::N_REL_EXTENT * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_rprep, (size_t)n_tokens * inkling::N_HEAD * inkling::N_REL * sizeof(float)));
    HIP_CHECK(hipMalloc(&d_bias, (size_t)inkling::N_REL_EXTENT * inkling::N_REL * sizeof(float)));
    {
        std::vector<float> tmp(nq);
        for (auto & v : tmp) v = runiform(-0.8f, 0.8f);
        HIP_CHECK(hipMemcpy(d_q, tmp.data(), nq * sizeof(float), hipMemcpyHostToDevice));
        for (auto & v : tmp) v = runiform(-1.5f, 1.5f);
        HIP_CHECK(hipMemcpy(d_qg, tmp.data(), nq * sizeof(float), hipMemcpyHostToDevice));
    }
    {
        std::vector<float> tmp((size_t)n_tokens * inkling::N_HEAD);
        for (auto & v : tmp) v = runiform(-1.0f, 1.0f);
        HIP_CHECK(hipMemcpy(d_sinks, tmp.data(), tmp.size() * sizeof(float), hipMemcpyHostToDevice));
        tmp.assign((size_t)n_tokens * inkling::N_HEAD * inkling::N_REL_EXTENT, 0.f);
        for (auto & v : tmp) v = runiform(-0.3f, 0.3f);
        HIP_CHECK(hipMemcpy(d_rel, tmp.data(), tmp.size() * sizeof(float), hipMemcpyHostToDevice));
        tmp.assign((size_t)n_tokens * inkling::N_HEAD * inkling::N_REL, 0.f);
        for (auto & v : tmp) v = runiform(-0.3f, 0.3f);
        HIP_CHECK(hipMemcpy(d_rprep, tmp.data(), tmp.size() * sizeof(float), hipMemcpyHostToDevice));
        tmp.assign((size_t)inkling::N_REL_EXTENT * inkling::N_REL, 0.f);
        for (auto & v : tmp) v = runiform(-0.3f, 0.3f);
        HIP_CHECK(hipMemcpy(d_bias, tmp.data(), tmp.size() * sizeof(float), hipMemcpyHostToDevice));
    }

    const int pos_start = n_kv - n_tokens;

    inkling::launch_kv_update_page_minmax_prefill(kv, meta, 0, n_kv, stream);
    inkling::launch_sparse_global_select_pages_prefill(d_q, meta, sscr, n_tokens, pos_start,
                                                       sparse_pages, stream);
    HIP_CHECK(hipStreamSynchronize(stream));

    const float * rel_table = (rel_mode == 0) ? d_rel : nullptr;
    const float * r_prep    = (rel_mode == 0) ? nullptr : d_rprep;

    // warmup
    for (int i = 0; i < 3; ++i) {
        inkling::launch_fa_sparse_global_prefill(d_q, kv, d_sinks, inkling::N_HEAD,
                                                 rel_table, r_prep, d_bias,
                                                 d_qg, inkling::N_Q, d_out,
                                                 sscr.selected_pages, sscr.n_selected,
                                                 n_tokens, pos_start, n_kv, stream);
    }
    HIP_CHECK(hipStreamSynchronize(stream));

    std::vector<float> ref(nq);
    HIP_CHECK(hipMemcpy(ref.data(), d_out, nq * sizeof(float), hipMemcpyDeviceToHost));

    hipEvent_t e0, e1;
    HIP_CHECK(hipEventCreate(&e0));
    HIP_CHECK(hipEventCreate(&e1));
    float best = 1e30f, sum = 0.f;
    for (int i = 0; i < iters; ++i) {
        HIP_CHECK(hipEventRecord(e0, stream));
        inkling::launch_fa_sparse_global_prefill(d_q, kv, d_sinks, inkling::N_HEAD,
                                                 rel_table, r_prep, d_bias,
                                                 d_qg, inkling::N_Q, d_out,
                                                 sscr.selected_pages, sscr.n_selected,
                                                 n_tokens, pos_start, n_kv, stream);
        HIP_CHECK(hipEventRecord(e1, stream));
        HIP_CHECK(hipEventSynchronize(e1));
        float ms = 0.f;
        HIP_CHECK(hipEventElapsedTime(&ms, e0, e1));
        if (ms < best) best = ms;
        sum += ms;
    }
    std::vector<float> got(nq);
    HIP_CHECK(hipMemcpy(got.data(), d_out, nq * sizeof(float), hipMemcpyDeviceToHost));

    size_t diff_cnt = 0;
    double max_rel = 0.0;
    for (size_t i = 0; i < nq; ++i) {
        if (got[i] != ref[i]) {
            ++diff_cnt;
            const double d = std::abs((double)got[i] - (double)ref[i])
                           / (1e-6 + std::abs((double)ref[i]));
            if (d > max_rel) max_rel = d;
        }
    }

    std::printf("RESULT ms_min=%.4f ms_avg=%.4f  attn_tok_per_s=%.1f  bitdiff=%zu maxrel=%.3e\n",
                best, sum / iters, (double)n_tokens / best * 1000.0, diff_cnt, max_rel);

    if (argc > 8) {
        FILE * f = std::fopen(argv[8], "wb");
        if (f) { std::fwrite(got.data(), sizeof(float), nq, f); std::fclose(f); }
    }

    HIP_CHECK(hipEventDestroy(e0));
    HIP_CHECK(hipEventDestroy(e1));
    inkling::sparse_global_page_meta_free(meta);
    inkling::sparse_global_scratch_free(sscr);
    HIP_CHECK(hipFree(kv.k_qs)); HIP_CHECK(hipFree(kv.v_qs));
    HIP_CHECK(hipFree(kv.k_d));  HIP_CHECK(hipFree(kv.v_d));
    HIP_CHECK(hipFree(kv.k_pos));
    HIP_CHECK(hipFree(d_q)); HIP_CHECK(hipFree(d_sinks)); HIP_CHECK(hipFree(d_rel));
    HIP_CHECK(hipFree(d_rprep)); HIP_CHECK(hipFree(d_bias)); HIP_CHECK(hipFree(d_qg));
    HIP_CHECK(hipFree(d_out));
    HIP_CHECK(hipStreamDestroy(stream));
    return 0;
}
