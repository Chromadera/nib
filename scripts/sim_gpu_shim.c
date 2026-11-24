// sim_gpu_shim.c — LD_PRELOAD HIP shim for simulating other AMD gfx1100 GPUs
// (RX 7900 XT 20GB, RX 7900 GRE 16GB, Radeon PRO W7800 32/48GB 70CU, W7900 48GB, etc.)
// on a single host RX 7900 XTX.
//
// Controls via environment variables (set by scripts/sim_gpu.sh):
//   NIB_SIM_GPU_NAME          Display name override
//   NIB_SIM_WGP               Multiprocessor (WGP) count override (CUs / 2 on RDNA3)
//   NIB_SIM_VRAM_MIB          Simulated total VRAM in MiB (enforces hard OOM cap when <= host VRAM)
//   NIB_SIM_VRAM_RESERVE_MIB  Simulated OS/desktop reserve in MiB (default: 0)
//   NIB_SIM_BW_GBS            Target memory bandwidth in GB/s (for throughput scaling note)
//   NIB_SIM_HOST_BW_GBS       Host GPU memory bandwidth in GB/s (default: 960)
//   NIB_SIM_QUIET             If "1", suppress banner/summary stderr output

#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define __HIP_PLATFORM_AMD__
#include <hip/hip_runtime_api.h>

// Open-addressed hash table tracking live hipMalloc pointers -> allocation sizes
#define ALLOC_TABLE_CAP 65536
typedef struct {
    uintptr_t ptr;
    size_t    size;
    uint8_t   state; // 0 = empty, 1 = occupied, 2 = tombstone
} AllocEntry;

static pthread_mutex_t g_mu = PTHREAD_MUTEX_INITIALIZER;
static AllocEntry      g_tab[ALLOC_TABLE_CAP];
static size_t          g_live_bytes = 0;
static size_t          g_peak_bytes = 0;
static size_t          g_oom_count  = 0;

static int             g_inited          = 0;
static int             g_quiet           = 0;
static int             g_sim_wgp         = 0;
static size_t          g_sim_vram_bytes  = 0;
static size_t          g_sim_resv_bytes  = 0;
static size_t          g_sim_usable      = 0;
static double          g_sim_bw_gbs      = 0.0;
static double          g_host_bw_gbs     = 960.0;
static char            g_sim_name[256]   = "";

typedef hipError_t (*fn_hipMalloc)(void **, size_t);
typedef hipError_t (*fn_hipMallocManaged)(void **, size_t, unsigned int);
typedef hipError_t (*fn_hipExtMallocWithFlags)(void **, size_t, unsigned int);
typedef hipError_t (*fn_hipFree)(void *);
typedef hipError_t (*fn_hipMemGetInfo)(size_t *, size_t *);
typedef hipError_t (*fn_hipDeviceGetAttribute)(int *, hipDeviceAttribute_t, int);
typedef hipError_t (*fn_hipGetDeviceProperties)(hipDeviceProp_t *, int);

static fn_hipMalloc              real_hipMalloc              = NULL;
static fn_hipMallocManaged       real_hipMallocManaged       = NULL;
static fn_hipExtMallocWithFlags  real_hipExtMallocWithFlags  = NULL;
static fn_hipFree                real_hipFree                = NULL;
static fn_hipMemGetInfo          real_hipMemGetInfo          = NULL;
static fn_hipDeviceGetAttribute  real_hipDeviceGetAttribute  = NULL;
static fn_hipGetDeviceProperties real_hipGetDeviceProperties = NULL;

static inline uint32_t ptr_hash(uintptr_t p) {
    uint64_t x = (uint64_t)p >> 4;
    x ^= x >> 33;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33;
    return (uint32_t)(x & (ALLOC_TABLE_CAP - 1));
}

static void track_insert_locked(void * p, size_t sz) {
    if (!p) return;
    uintptr_t key = (uintptr_t)p;
    uint32_t idx = ptr_hash(key);
    int first_tomb = -1;
    for (uint32_t i = 0; i < ALLOC_TABLE_CAP; ++i) {
        uint32_t slot = (idx + i) & (ALLOC_TABLE_CAP - 1);
        uint8_t st = g_tab[slot].state;
        if (st == 1 && g_tab[slot].ptr == key) {
            g_live_bytes = g_live_bytes - g_tab[slot].size + sz;
            g_tab[slot].size = sz;
            if (g_live_bytes > g_peak_bytes) g_peak_bytes = g_live_bytes;
            return;
        }
        if (st == 2 && first_tomb < 0) first_tomb = (int)slot;
        if (st == 0) {
            uint32_t dst = (first_tomb >= 0) ? (uint32_t)first_tomb : slot;
            g_tab[dst].ptr   = key;
            g_tab[dst].size  = sz;
            g_tab[dst].state = 1;
            g_live_bytes += sz;
            if (g_live_bytes > g_peak_bytes) g_peak_bytes = g_live_bytes;
            return;
        }
    }
}

static void track_remove_locked(void * p) {
    if (!p) return;
    uintptr_t key = (uintptr_t)p;
    uint32_t idx = ptr_hash(key);
    for (uint32_t i = 0; i < ALLOC_TABLE_CAP; ++i) {
        uint32_t slot = (idx + i) & (ALLOC_TABLE_CAP - 1);
        uint8_t st = g_tab[slot].state;
        if (st == 0) return;
        if (st == 1 && g_tab[slot].ptr == key) {
            if (g_live_bytes >= g_tab[slot].size) g_live_bytes -= g_tab[slot].size;
            else g_live_bytes = 0;
            g_tab[slot].state = 2;
            g_tab[slot].ptr   = 0;
            g_tab[slot].size  = 0;
            return;
        }
    }
}

static void sim_init_once(void) {
    if (g_inited) return;
    real_hipMalloc              = (fn_hipMalloc)dlsym(RTLD_NEXT, "hipMalloc");
    real_hipMallocManaged       = (fn_hipMallocManaged)dlsym(RTLD_NEXT, "hipMallocManaged");
    real_hipExtMallocWithFlags  = (fn_hipExtMallocWithFlags)dlsym(RTLD_NEXT, "hipExtMallocWithFlags");
    real_hipFree                = (fn_hipFree)dlsym(RTLD_NEXT, "hipFree");
    real_hipMemGetInfo          = (fn_hipMemGetInfo)dlsym(RTLD_NEXT, "hipMemGetInfo");
    real_hipDeviceGetAttribute  = (fn_hipDeviceGetAttribute)dlsym(RTLD_NEXT, "hipDeviceGetAttribute");
    real_hipGetDeviceProperties = (fn_hipGetDeviceProperties)dlsym(RTLD_NEXT, "hipGetDevicePropertiesR0600");
    if (!real_hipGetDeviceProperties) {
        real_hipGetDeviceProperties = (fn_hipGetDeviceProperties)dlsym(RTLD_NEXT, "hipGetDeviceProperties");
    }

    const char * env_q = getenv("NIB_SIM_QUIET");
    g_quiet = (env_q && strcmp(env_q, "1") == 0);

    const char * env_name = getenv("NIB_SIM_GPU_NAME");
    if (env_name && *env_name) {
        snprintf(g_sim_name, sizeof(g_sim_name), "%s", env_name);
    }

    const char * env_wgp = getenv("NIB_SIM_WGP");
    if (env_wgp && *env_wgp) g_sim_wgp = atoi(env_wgp);

    const char * env_vram = getenv("NIB_SIM_VRAM_MIB");
    if (env_vram && *env_vram) {
        g_sim_vram_bytes = (size_t)strtoull(env_vram, NULL, 10) * 1024ULL * 1024ULL;
    }

    const char * env_resv = getenv("NIB_SIM_VRAM_RESERVE_MIB");
    if (env_resv && *env_resv) {
        g_sim_resv_bytes = (size_t)strtoull(env_resv, NULL, 10) * 1024ULL * 1024ULL;
    }

    if (g_sim_vram_bytes > 0) {
        g_sim_usable = (g_sim_vram_bytes > g_sim_resv_bytes)
            ? (g_sim_vram_bytes - g_sim_resv_bytes)
            : 0;
    }

    const char * env_bw = getenv("NIB_SIM_BW_GBS");
    if (env_bw && *env_bw) g_sim_bw_gbs = atof(env_bw);

    const char * env_hbw = getenv("NIB_SIM_HOST_BW_GBS");
    if (env_hbw && *env_hbw) g_host_bw_gbs = atof(env_hbw);

    g_inited = 1;

    if (!g_quiet) {
        const char * cu_mask = getenv("HSA_CU_MASK");
        double bw_scale = (g_sim_bw_gbs > 0.0 && g_host_bw_gbs > 0.0)
            ? (g_sim_bw_gbs / g_host_bw_gbs) : 1.0;
        fprintf(stderr,
            "[sim_gpu] Active: %s | WGPs=%d (%d CUs, HSA_CU_MASK=%s) | VRAM=%.2f GiB (usable=%.2f GiB, resv=%.0f MiB) | BW=%.0f GB/s (decode scale ~%.3fx vs %.0f GB/s host)\n",
            g_sim_name[0] ? g_sim_name : "custom",
            g_sim_wgp, g_sim_wgp * 2,
            cu_mask ? cu_mask : "unset",
            (double)g_sim_vram_bytes / (1024.0 * 1024.0 * 1024.0),
            (double)g_sim_usable / (1024.0 * 1024.0 * 1024.0),
            (double)g_sim_resv_bytes / (1024.0 * 1024.0),
            g_sim_bw_gbs, bw_scale, g_host_bw_gbs);
    }
}

__attribute__((destructor))
static void sim_fini(void) {
    if (!g_inited || g_quiet) return;
    double peak_gib   = (double)g_peak_bytes / (1024.0 * 1024.0 * 1024.0);
    double usable_gib = (double)g_sim_usable / (1024.0 * 1024.0 * 1024.0);
    double total_gib  = (double)g_sim_vram_bytes / (1024.0 * 1024.0 * 1024.0);
    const char * verdict = (g_oom_count > 0) ? "OOM-REJECTED"
                         : (g_sim_usable > 0 && g_peak_bytes <= g_sim_usable) ? "FITS"
                         : "EXCEEDS-CAP";
    double bw_scale = (g_sim_bw_gbs > 0.0 && g_host_bw_gbs > 0.0)
        ? (g_sim_bw_gbs / g_host_bw_gbs) : 1.0;
    fprintf(stderr,
        "[sim_gpu] Summary (%s): peak_vram=%.2f GiB / %.2f GiB usable (%.2f GiB total) -> %s (oom_rejects=%zu) | est real-card decode = measured * %.3fx\n",
        g_sim_name[0] ? g_sim_name : "custom",
        peak_gib, usable_gib, total_gib, verdict, g_oom_count, bw_scale);
}

static hipError_t check_and_alloc(void ** ptr, size_t size, int kind, unsigned int flags) {
    pthread_mutex_lock(&g_mu);
    sim_init_once();
    if (g_sim_usable > 0 && (g_live_bytes + size > g_sim_usable)) {
        g_oom_count++;
        if (!g_quiet) {
            fprintf(stderr,
                "[sim_gpu] OOM: request %.2f MiB would push live VRAM to %.2f GiB > %.2f GiB cap (%s)\n",
                (double)size / (1024.0 * 1024.0),
                (double)(g_live_bytes + size) / (1024.0 * 1024.0 * 1024.0),
                (double)g_sim_usable / (1024.0 * 1024.0 * 1024.0),
                g_sim_name[0] ? g_sim_name : "custom");
        }
        pthread_mutex_unlock(&g_mu);
        if (ptr) *ptr = NULL;
        return hipErrorOutOfMemory;
    }
    pthread_mutex_unlock(&g_mu);

    hipError_t err = hipErrorUnknown;
    if (kind == 0 && real_hipMalloc) {
        err = real_hipMalloc(ptr, size);
    } else if (kind == 1 && real_hipMallocManaged) {
        err = real_hipMallocManaged(ptr, size, flags);
    } else if (kind == 2 && real_hipExtMallocWithFlags) {
        err = real_hipExtMallocWithFlags(ptr, size, flags);
    }

    if (err == hipSuccess && ptr && *ptr) {
        pthread_mutex_lock(&g_mu);
        track_insert_locked(*ptr, size);
        pthread_mutex_unlock(&g_mu);
    }
    return err;
}

hipError_t hipMalloc(void ** ptr, size_t size) {
    return check_and_alloc(ptr, size, 0, 0);
}

hipError_t hipMallocManaged(void ** dev_ptr, size_t size, unsigned int flags) {
    return check_and_alloc(dev_ptr, size, 1, flags);
}

hipError_t hipExtMallocWithFlags(void ** ptr, size_t size, unsigned int flags) {
    return check_and_alloc(ptr, size, 2, flags);
}

hipError_t hipFree(void * ptr) {
    pthread_mutex_lock(&g_mu);
    sim_init_once();
    track_remove_locked(ptr);
    pthread_mutex_unlock(&g_mu);
    return real_hipFree ? real_hipFree(ptr) : hipSuccess;
}

hipError_t hipMemGetInfo(size_t * free_bytes, size_t * total_bytes) {
    pthread_mutex_lock(&g_mu);
    sim_init_once();
    size_t live   = g_live_bytes;
    size_t total  = g_sim_vram_bytes;
    size_t usable = g_sim_usable;
    pthread_mutex_unlock(&g_mu);

    hipError_t err = real_hipMemGetInfo ? real_hipMemGetInfo(free_bytes, total_bytes) : hipSuccess;
    if (err != hipSuccess) return err;

    if (total > 0) {
        if (total_bytes) *total_bytes = total;
        if (free_bytes) {
            size_t sim_free = (usable > live) ? (usable - live) : 0;
            // If simulating a card larger than host VRAM, still report simulated free
            // (real hipMalloc will hit hardware OOM if it exceeds physical 24 GiB).
            *free_bytes = sim_free;
        }
    }
    return hipSuccess;
}

hipError_t hipDeviceGetAttribute(int * pi, hipDeviceAttribute_t attr, int deviceId) {
    pthread_mutex_lock(&g_mu);
    sim_init_once();
    int wgp = g_sim_wgp;
    pthread_mutex_unlock(&g_mu);

    hipError_t err = real_hipDeviceGetAttribute
        ? real_hipDeviceGetAttribute(pi, attr, deviceId)
        : hipErrorUnknown;
    if (err == hipSuccess && pi && attr == hipDeviceAttributeMultiprocessorCount && wgp > 0) {
        *pi = wgp;
    }
    return err;
}

// Note: ROCm 6+ maps hipGetDeviceProperties -> hipGetDevicePropertiesR0600 via header macro
hipError_t hipGetDeviceProperties(hipDeviceProp_t * prop, int deviceId) {
    pthread_mutex_lock(&g_mu);
    sim_init_once();
    int wgp = g_sim_wgp;
    size_t vram = g_sim_vram_bytes;
    char name[256];
    memcpy(name, g_sim_name, sizeof(name));
    pthread_mutex_unlock(&g_mu);

    hipError_t err = real_hipGetDeviceProperties
        ? real_hipGetDeviceProperties(prop, deviceId)
        : hipErrorUnknown;
    if (err == hipSuccess && prop) {
        if (wgp > 0) prop->multiProcessorCount = wgp;
        if (vram > 0) prop->totalGlobalMem = vram;
        if (name[0]) {
            snprintf(prop->name, sizeof(prop->name), "%s", name);
        }
    }
    return err;
}
