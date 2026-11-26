#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>

static double to_mib(size_t b) { return (double)b / (1024.0 * 1024.0); }
static double to_gib(size_t b) { return (double)b / (1024.0 * 1024.0 * 1024.0); }

static void print_mem(const char* tag) {
    size_t f = 0, t = 0;
    hipError_t e = hipMemGetInfo(&f, &t);
    if (e != hipSuccess) {
        std::printf("[mem:%s] hipMemGetInfo error: %s\n", tag, hipGetErrorString(e));
    } else {
        std::printf("[mem:%s] free=%.1f MiB (%.3f GiB) | used=%.1f MiB (%.3f GiB) | total=%.1f MiB (%.3f GiB)\n",
                    tag, to_mib(f), to_gib(f), to_mib(t - f), to_gib(t - f), to_mib(t), to_gib(t));
    }
    std::fflush(stdout);
}

static size_t probe_chunks(const char* label, size_t initial_chunk_mib, bool touch_pages, int pre_chunks, size_t pre_chunk_mib) {
    std::vector<void*> ptrs;
    size_t total = 0;
    size_t next_report_gib = 16;

    auto try_alloc = [&](size_t bytes) -> bool {
        void* p = nullptr;
        hipError_t e = hipMalloc(&p, bytes);
        if (e != hipSuccess || !p) {
            (void)hipGetLastError();
            return false;
        }
        if (touch_pages) {
            e = hipMemset(p, 0x5A, bytes);
            if (e == hipSuccess) e = hipDeviceSynchronize();
            if (e != hipSuccess) {
                (void)hipGetLastError();
                (void)hipFree(p);
                return false;
            }
        }
        ptrs.push_back(p);
        total += bytes;
        return true;
    };

    for (int i = 0; i < pre_chunks; ++i) {
        if (!try_alloc(pre_chunk_mib * 1024ULL * 1024ULL)) {
            std::printf("[%s] pre-alloc failed at chunk %d (%.3f GiB)\n", label, i, to_gib(total));
            break;
        }
    }
    if (pre_chunks > 0) {
        std::printf("[%s] after pre-alloc (%d x %zu MiB): total=%.1f MiB (%.3f GiB)\n",
                    label, (int)ptrs.size(), pre_chunk_mib, to_mib(total), to_gib(total));
        print_mem(label);
    }

    size_t step_mib = initial_chunk_mib;
    while (step_mib >= 1) {
        while (try_alloc(step_mib * 1024ULL * 1024ULL)) {
            if (total >= next_report_gib * 1024ULL * 1024ULL * 1024ULL) {
                std::printf("[%s] reached %.1f MiB (%.3f GiB) in %zu allocs\n",
                            label, to_mib(total), to_gib(total), ptrs.size());
                print_mem(label);
                next_report_gib += 1;
            }
            if (total > 22ULL * 1024ULL * 1024ULL * 1024ULL) {
                std::printf("[%s] safety cap 22 GiB hit (overcommit detected!)\n", label);
                step_mib = 0;
                break;
            }
        }
        if (step_mib == 0) break;
        step_mib /= 2;
    }

    std::printf("RESULT [%s] max_alloc=%.1f MiB (%.4f GiB) across %zu allocations (touch=%d)\n",
                label, to_mib(total), to_gib(total), ptrs.size(), touch_pages ? 1 : 0);
    print_mem("at_cliff");

    for (void* p : ptrs) (void)hipFree(p);
    (void)hipDeviceSynchronize();
    return total;
}

int main() {
    int dev = 0;
    hipError_t e = hipSetDevice(dev);
    if (e != hipSuccess) {
        std::fprintf(stderr, "hipSetDevice failed: %s\n", hipGetErrorString(e));
        return 1;
    }
    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, dev);
    std::printf("[device] name=%s gcnArch=%s CUs=%d totalGlobalMem=%.1f MiB (%.4f GiB)\n",
                prop.name, prop.gcnArchName, prop.multiProcessorCount,
                to_mib(prop.totalGlobalMem), to_gib(prop.totalGlobalMem));
    print_mem("idle_start");

    // 1. Chunked 16 MiB + hipMemset touch + 1 MiB refinement
    size_t max_touched = probe_chunks("chunk16_touched", 16, true, 0, 0);
    print_mem("after_chunk16_free");

    // 2. Realistic Nib-like fragmentation: 512 x 32 MiB (~16.0 GiB weights) + 16..1 MiB chunks touched
    size_t max_niblike = probe_chunks("niblike_512x32M_touched", 16, true, 512, 32);
    print_mem("after_niblike_free");

    std::printf("=== STAGE 1 SUMMARY ===\n");
    std::printf("  chunk16_touched          = %.1f MiB (%.4f GiB)\n", to_mib(max_touched), to_gib(max_touched));
    std::printf("  niblike_512x32M_touched  = %.1f MiB (%.4f GiB)\n", to_mib(max_niblike), to_gib(max_niblike));
    std::fflush(stdout);
    return 0;
}
