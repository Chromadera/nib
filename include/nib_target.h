// ============================================================================
// include/nib_target.h — which RDNA generation is this TU being compiled for?
//
// nib was written for gfx1100..gfx1103 (RDNA3, 96 CU, wmma-256b-insts). The RDNA4
// (gfx1200/gfx1201, 64 CU) port keeps the gfx11 code paths byte-for-byte identical
// and only changes what cannot work there:
//   NIB_RDNA3   - native RDNA3 WMMA + the gfx11-only cache-flush asm are available.
//   NIB_RDNA4   - portable GEMM tile (include/nib_mma.h) and CU-aware grid sizing.
// ============================================================================
#pragma once

#if defined(__gfx1100__) || defined(__gfx1101__) || defined(__gfx1102__) || defined(__gfx1103__)
#define NIB_RDNA3 1
#else
#define NIB_RDNA3 0
#endif

#if defined(__gfx1200__) || defined(__gfx1201__)
#define NIB_RDNA4 1
#else
#define NIB_RDNA4 0
#endif

// LDS/smem drain + block barrier, as the RDNA3 kernels spell it in inline asm.
// gfx11: the exact instruction pair the kernels were tuned against (unchanged).
// gfx12: the same two waits through the compiler -- RDNA4 has no bare `s_barrier`
// (it assembles to s_barrier_signal/s_barrier_wait) and splits the wait counters,
// so `s_waitcnt 0` + __builtin_amdgcn_s_barrier() is the equivalent sequence.
static __device__ __forceinline__ void nib_lds_fence_barrier() {
#if NIB_RDNA3
    asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
#else
    __builtin_amdgcn_s_waitcnt(0);
    __builtin_amdgcn_s_barrier();
#endif
}
