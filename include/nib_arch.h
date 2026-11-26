#pragma once

// ============================================================================
// nib_arch.h — Single source of truth for AMD RDNA3 (gfx11xx) & RDNA4 (gfx12xx)
// ============================================================================
// Provides:
//   - Device-side arch macros (NIB_ARCH_RDNA3, NIB_ARCH_RDNA4, NIB_ARCH_RDNA)
//   - Hard compile-time failure on unsupported GPU targets (zero silent stubs)
//   - Cross-workgroup coherence & LDS barrier primitives (RDNA3 & RDNA4 ISA)
//   - Unified Wave32 FP16 WMMA 16x16x16 fragment & layout abstraction
//   - Host-side cached device info (gcnArchName, MP/WGP count, VRAM, family)
// ============================================================================

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>
#include <string>

// ----------------------------------------------------------------------------
// Compile-time target architecture classification (device pass)
// ----------------------------------------------------------------------------
#if defined(__GFX11__) || defined(__gfx1100__) || defined(__gfx1101__) || \
    defined(__gfx1102__) || defined(__gfx1103__)
#  define NIB_ARCH_RDNA3 1
#else
#  define NIB_ARCH_RDNA3 0
#endif

#if defined(__GFX12__) || defined(__gfx1200__) || defined(__gfx1201__)
#  define NIB_ARCH_RDNA4 1
#else
#  define NIB_ARCH_RDNA4 0
#endif

#define NIB_ARCH_RDNA (NIB_ARCH_RDNA3 || NIB_ARCH_RDNA4)

#if defined(__HIP_DEVICE_COMPILE__) && !NIB_ARCH_RDNA
#  error "Nib device kernels require AMD RDNA3 (gfx1100-gfx1103) or RDNA4 (gfx1200-gfx1201)."
#endif

// Tight amdgpu_num_vgpr() caps tuned for RDNA3's 512-VGPR/SIMD occupancy grid.
// On RDNA4 (gfx12), slightly different ISA/scheduling causes two RDNA3 pins to
// spill 1-4 VGPRs under ROCm 7.2 clang 22, so omit those hard caps on gfx12.
#if NIB_ARCH_RDNA3
#  define NIB_RDNA3_VGPR_CAP(n) __attribute__((amdgpu_num_vgpr(n)))
#else
#  define NIB_RDNA3_VGPR_CAP(n)
#endif

namespace inkling {

// ============================================================================
// Host-side runtime GPU info (cached per device, HIP-graph capture safe)
// ============================================================================
struct NibDeviceInfo {
    int    device_id        = 0;
    char   name[256]        = {};
    char   gcn_arch[64]     = {};  // stripped of :sramecc/:xnack suffixes (e.g. "gfx1201")
    char   arch_family[16]  = {};  // "RDNA3", "RDNA4", or "AMD GPU"
    int    mp_count         = 48;  // HIP multiProcessorCount (WGPs on RDNA3/RDNA4)
    int    warp_size        = 32;
    size_t total_vram_bytes = 0;
    bool   is_rdna3         = false;
    bool   is_rdna4         = false;
    bool   valid            = false;
};

inline const NibDeviceInfo & nib_device_info(int dev = -1) {
    static NibDeviceInfo cache[64] = {};
    if (dev < 0) {
        if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) {
            dev = 0;
        }
    }
    if (dev >= 64) {
        dev = 0;
    }
    NibDeviceInfo & info = cache[dev];
    if (!info.valid) {
        info.device_id = dev;
        hipDeviceProp_t prop{};
        if (hipGetDeviceProperties(&prop, dev) == hipSuccess) {
            std::snprintf(info.name, sizeof(info.name), "%s", prop.name);

            // Copy gcnArchName up to first ':' (strip :sramecc- / :xnack- feature tags)
            size_t len = 0;
            while (len + 1 < sizeof(info.gcn_arch) &&
                   prop.gcnArchName[len] != '\0' &&
                   prop.gcnArchName[len] != ':') {
                info.gcn_arch[len] = prop.gcnArchName[len];
                ++len;
            }
            info.gcn_arch[len] = '\0';

            info.mp_count = (prop.multiProcessorCount > 0) ? prop.multiProcessorCount : 48;
            info.warp_size = (prop.warpSize > 0) ? prop.warpSize : 32;
            info.total_vram_bytes = prop.totalGlobalMem;

            if (std::strncmp(info.gcn_arch, "gfx12", 5) == 0) {
                info.is_rdna4 = true;
                std::strcpy(info.arch_family, "RDNA4");
            } else if (std::strncmp(info.gcn_arch, "gfx11", 5) == 0) {
                info.is_rdna3 = true;
                std::strcpy(info.arch_family, "RDNA3");
            } else {
                std::strcpy(info.arch_family, "AMD GPU");
            }
        } else {
            (void)hipGetLastError();
            std::strcpy(info.name, "AMD Radeon GPU");
            std::strcpy(info.gcn_arch, "unknown");
            std::strcpy(info.arch_family, "AMD GPU");
            info.mp_count = 48;
            info.warp_size = 32;
        }
        info.valid = true;
    }
    return info;
}

inline int nib_device_mp_count(int dev = -1) {
    return nib_device_info(dev).mp_count;
}

// Check whether a compiled NIB_GPU_TARGET list (semicolon or comma separated)
// contains the runtime GPU architecture.
inline bool nib_target_list_contains(const char * compiled_targets, const char * arch) {
    if (!compiled_targets || !arch || arch[0] == '\0') {
        return true;
    }
    const size_t arch_len = std::strlen(arch);
    const char * p = compiled_targets;
    while (*p != '\0') {
        while (*p == ';' || *p == ',' || *p == ' ') {
            ++p;
        }
        const char * start = p;
        while (*p != '\0' && *p != ';' && *p != ',' && *p != ' ') {
            ++p;
        }
        const size_t tok_len = (size_t)(p - start);
        if (tok_len == arch_len && std::strncmp(start, arch, arch_len) == 0) {
            return true;
        }
    }
    return false;
}

// Verify Wave32 + compiled arch compatibility once at startup.
// Returns false only on fatal incompatibilities (e.g. non-Wave32 device).
inline bool nib_check_runtime_device(const char * compiled_targets, bool verbose = true) {
    const NibDeviceInfo & info = nib_device_info(0);
    if (verbose) {
        std::fprintf(stderr, "[nib] GPU: %s (%s, %s, %d MPs, wave%d, %.2f GiB VRAM)\n",
                     info.name[0] ? info.name : "unknown",
                     info.gcn_arch[0] ? info.gcn_arch : "unknown",
                     info.arch_family,
                     info.mp_count,
                     info.warp_size,
                     (double)info.total_vram_bytes / (1024.0 * 1024.0 * 1024.0));
    }
    if (info.warp_size != 32) {
        std::fprintf(stderr,
                     "[nib] ERROR: Nib requires Wave32 execution (RDNA3/RDNA4); device '%s' (%s) reports wave%d\n",
                     info.name, info.gcn_arch, info.warp_size);
        return false;
    }
    if (info.gcn_arch[0] != '\0' && std::strcmp(info.gcn_arch, "unknown") != 0 &&
        compiled_targets != nullptr && compiled_targets[0] != '\0' &&
        !nib_target_list_contains(compiled_targets, info.gcn_arch)) {
        std::fprintf(stderr,
                     "[nib] WARNING: binary compiled for '%s', but runtime GPU is '%s' (%s). "
                     "Rebuild with -DGPU_TARGET=%s (or a semicolon-separated list).\n",
                     compiled_targets, info.gcn_arch, info.name, info.gcn_arch);
    }
    return true;
}

// ============================================================================
// Device-side ISA & WMMA primitives (HIP compilation only)
// ============================================================================
#ifdef __HIPCC__

// ----------------------------------------------------------------------------
// 1. Cross-workgroup memory ordering & cache coherence (RDNA3 & RDNA4)
// ----------------------------------------------------------------------------
// RDNA3 (gfx11): L0/L1 are write-through to L2; producer waits vscnt(0),
//                consumer invalidates GL1+GL0.
// RDNA4 (gfx12): L0/L1 are write-through to L2; producer waits storecnt(0),
//                consumer executes global_inv scope:SCOPE_DEV.
static __device__ __forceinline__ void nib_flush_stores() {
    __scoped_atomic_thread_fence(__ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
#if NIB_ARCH_RDNA3
    asm volatile("s_waitcnt_vscnt null, 0x0" ::: "memory");
#elif NIB_ARCH_RDNA4
    asm volatile("s_wait_storecnt 0x0" ::: "memory");
#endif
}

static __device__ __forceinline__ void nib_inv_l0_l1() {
    __scoped_atomic_thread_fence(__ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
#if NIB_ARCH_RDNA3
    asm volatile("buffer_gl1_inv\n\tbuffer_gl0_inv" ::: "memory");
#elif NIB_ARCH_RDNA4
    asm volatile("global_inv scope:SCOPE_DEV" ::: "memory");
#endif
}

// Volatile LDS wait + workgroup barrier.
// Uses explicit inline asm on both RDNA3 and RDNA4 so that pin_wmma_acc's
// volatile asm cannot sink across the barrier (clang 22 ROCm 7.2.x scheduling).
static __device__ __forceinline__ void nib_wait_lds_barrier() {
#if NIB_ARCH_RDNA3
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#elif NIB_ARCH_RDNA4
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait 0xffff" ::: "memory");
#else
    __syncthreads();
#endif
}

// Variant with vector-memory load-counter hint (used after K*Q^T in sparse FA).
static __device__ __forceinline__ void nib_wait_lds_barrier_vmhint() {
#if NIB_ARCH_RDNA3
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier\n\ts_waitcnt vmcnt(63)" ::: "memory");
#elif NIB_ARCH_RDNA4
    asm volatile("s_wait_dscnt 0x0\n\ts_barrier_signal -1\n\ts_barrier_wait 0xffff\n\ts_wait_loadcnt 0x3f" ::: "memory");
#else
    __syncthreads();
#endif
}

// ----------------------------------------------------------------------------
// 2. Unified Wave32 FP16 WMMA 16x16x16 -> FP32 abstraction
// ----------------------------------------------------------------------------
// Wave32 lane l in 0..31: c = l & 15, half_w = l >> 4 (0 or 1).
//
// RDNA3 (gfx11xx, wmma-256b-insts):
//   A (16x16): lane l holds row c, K = 0..15 (16 halves; lanes 16..31 replicate 0..15)
//   B (16x16): lane l holds col c, K = 0..15 (16 halves; lanes 16..31 replicate 0..15)
//   C/D (FP32): lane l, elem i (0..7) -> row (2*i + half_w), col c
//
// RDNA4 (gfx12xx, wmma-128b-insts):
//   A (16x16): lane l holds row c, K = 8*half_w + 0..7 (8 halves; NO replication)
//   B (16x16): lane l holds col c, K = 8*half_w + 0..7 (8 halves; NO replication)
//   C/D (FP32): lane l, elem i (0..7) -> row (8*half_w + i), col c
// ----------------------------------------------------------------------------
typedef float nib_float8_t __attribute__((ext_vector_type(8)));

#if NIB_ARCH_RDNA4
typedef _Float16 nib_wmma_hfrag_t __attribute__((ext_vector_type(8)));
static constexpr int NIB_WMMA_HFRAG_U32S = 4;

static __device__ __forceinline__ int nib_wmma_k_off(int half_w) {
    return half_w << 3;
}

static __device__ __forceinline__ int nib_wmma_c_row(int half_w, int idx) {
    return (half_w << 3) + idx;
}

static __device__ __forceinline__ nib_float8_t nib_wmma_f32_16x16x16_f16(
        nib_wmma_hfrag_t a, nib_wmma_hfrag_t b, nib_float8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12(a, b, c);
}

static __device__ __forceinline__ nib_wmma_hfrag_t nib_lds_wmma_frag(
        const _Float16 * row_ptr, int half_w) {
    const uint4 v = *reinterpret_cast<const uint4 *>(row_ptr + (half_w << 3));
    nib_wmma_hfrag_t r;
    __builtin_memcpy(&r, &v, 16);
    return r;
}

static __device__ __forceinline__ nib_wmma_hfrag_t nib_pack_p_frag(
        const nib_float8_t & p, int /*half_w*/) {
    // On RDNA4, C_frag[i] already holds row (8*half_w + i) for col (lane & 15),
    // which is the exact B_frag layout of O^T = V^T @ P — zero shuffles needed!
    union {
        nib_wmma_hfrag_t h8;
        uint32_t         u[4];
    } res;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const __half2 m = __floats2half2_rn(p[2 * t], p[2 * t + 1]);
        res.u[t] = __builtin_bit_cast(uint32_t, m);
    }
    return res.h8;
}

static __device__ __forceinline__ nib_wmma_hfrag_t nib_abs_wmma_hfrag(nib_wmma_hfrag_t x) {
    union {
        nib_wmma_hfrag_t h8;
        uint32_t         u[4];
    } v;
    v.h8 = x;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        v.u[i] &= 0x7FFF7FFFu;
    }
    return v.h8;
}

#else // NIB_ARCH_RDNA3 (or host parse pass)
typedef _Float16 nib_wmma_hfrag_t __attribute__((ext_vector_type(16)));
static constexpr int NIB_WMMA_HFRAG_U32S = 8;

static __device__ __forceinline__ int nib_wmma_k_off(int /*half_w*/) {
    return 0;
}

static __device__ __forceinline__ int nib_wmma_c_row(int half_w, int idx) {
    return (idx << 1) + half_w;
}

static __device__ __forceinline__ nib_float8_t nib_wmma_f32_16x16x16_f16(
        nib_wmma_hfrag_t a, nib_wmma_hfrag_t b, nib_float8_t c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(a, b, c);
}

static __device__ __forceinline__ nib_wmma_hfrag_t nib_lds_wmma_frag(
        const _Float16 * row_ptr, int /*half_w*/) {
    const uint4 v0 = *reinterpret_cast<const uint4 *>(row_ptr + 0);
    const uint4 v1 = *reinterpret_cast<const uint4 *>(row_ptr + 8);
    nib_wmma_hfrag_t r;
    __builtin_memcpy(&r, &v0, 16);
    __builtin_memcpy(reinterpret_cast<char *>(&r) + 16, &v1, 16);
    return r;
}

static __device__ __forceinline__ nib_wmma_hfrag_t nib_pack_p_frag(
        const nib_float8_t & p, int half_w) {
    const uint32_t perm0 = (half_w == 0) ? 0x05040100u : 0x01000504u;
    const uint32_t perm1 = (half_w == 0) ? 0x07060302u : 0x03020706u;
    union {
        nib_wmma_hfrag_t h16;
        uint32_t         u[8];
    } res;
#pragma unroll
    for (int t = 0; t < 4; ++t) {
        const __half2 m = __floats2half2_rn(p[2 * t], p[2 * t + 1]);
        const uint32_t mu = __builtin_bit_cast(uint32_t, m);
        const uint32_t ou = __shfl_xor(mu, 16, 32);
        res.u[2 * t + 0] = __builtin_amdgcn_perm(ou, mu, perm0);
        res.u[2 * t + 1] = __builtin_amdgcn_perm(ou, mu, perm1);
    }
    return res.h16;
}

static __device__ __forceinline__ nib_wmma_hfrag_t nib_abs_wmma_hfrag(nib_wmma_hfrag_t x) {
    union {
        nib_wmma_hfrag_t h16;
        uint32_t         u[8];
    } v;
    v.h16 = x;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        v.u[i] &= 0x7FFF7FFFu;
    }
    return v.h16;
}
#endif

// Cached co-residency check for cooperative spin-barrier kernels.
// Uses min(hipOccupancyMaxActiveBlocksPerMultiprocessor, hw_cap_per_mp) * mp_count >= grid,
// keyed by (device, kernel_ptr, block_threads) so distinct kernels never collide.
inline bool nib_grid_coresident(const void * kernel_ptr, int block_threads, int grid_blocks,
                                int hw_cap_per_mp = 0, size_t dyn_smem = 0) {
    struct Entry {
        const void * ptr;
        int          dev;
        int          threads;
        int          occ;
    };
    static Entry cache[128] = {};
    static int   n_entries  = 0;

    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0) {
        dev = 0;
    }
    int occ = -1;
    for (int i = 0; i < n_entries; ++i) {
        if (cache[i].ptr == kernel_ptr && cache[i].dev == dev && cache[i].threads == block_threads) {
            occ = cache[i].occ;
            break;
        }
    }
    if (occ < 0) {
        int q = 0;
        if (hipOccupancyMaxActiveBlocksPerMultiprocessor(&q, kernel_ptr, block_threads, dyn_smem) == hipSuccess && q > 0) {
            occ = q;
        } else {
            occ = 0;
        }
        if (n_entries < 128) {
            cache[n_entries++] = Entry{kernel_ptr, dev, block_threads, occ};
        }
    }
    if (occ <= 0) {
        return false;
    }
    const int eff = (hw_cap_per_mp > 0 && occ > hw_cap_per_mp) ? hw_cap_per_mp : occ;
    return eff * nib_device_mp_count(dev) >= grid_blocks;
}

#endif // __HIPCC__

} // namespace inkling
