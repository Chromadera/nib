#pragma once

// ============================================================================
// kernels_wmma_fp8.h — RDNA4 FP8 WMMA prefill host API (src/kernels_wmma_fp8_rdna4.hip)
// ============================================================================
// The FP8 kernels are built only when GPU_TARGET contains gfx1200 or gfx1201; other builds link
// the host-only src/kernels_wmma_fp8_stub.cpp, which provides this API with
// rdna4_fp8_prefill_built() == false.  Host code therefore needs no build flag: it checks
// rdna4_fp8_prefill_supported() at runtime, and the launchers abort with a contract violation
// rather than let HIP fail on a kernel absent from the running GPU's code object.
// ============================================================================

#include "inkling_engine.h"
#include "nib_arch_fp8.h"

#include <cstdint>

namespace inkling {

// FP8 prefill modes (EngineConfig::fp8_prefill, --fp8-prefill, INKLING_FP8_PREFILL).
enum Fp8PrefillMode : int32_t {
    FP8_PREFILL_OFF = 0,   // every prefill GEMM on FP16 WMMA
    FP8_PREFILL_MOE = 1,   // NVFP4 expert GEMMs of large prefill micro-batches on FP8 WMMA
};

// FP8 prefill engages for prefill micro-batches with M > this many tokens (INKLING_FP8_MIN_M).
static constexpr int FP8_PREFILL_DEFAULT_MIN_M = 256;

// Maximum single-chunk micro-batch supported by the RDNA4 FP8 MoE prefill kernels
// (route_kernel keeps s_mask[128][32] + s_pref[128][32] = 34.3 KB <= 64 KB LDS).
static constexpr int FP8_MAX_UBATCH = 1024;

// Static scale applied to the SITU activations before they are rounded to E4M3 for the down
// projection: |SITU(g, u)| = |4 tanh(g/4) sigmoid(g)| * |25 tanh(u/25)| < 4 * 25 = 100, so the
// scaled value stays strictly inside the E4M3 range (|x| < 400 <= 448) with no per-row amax pass.
static constexpr float FP8_MOE_ACT_SCALE = 4.0f;

// The FP8 MoE activation operands (X_norm rows and SITU rows) are stored K-interleaved so that
// one 16-byte LDS load feeds two consecutive 16-deep WMMAs: inside every 32-element block the
// four 8-element sub-blocks are stored in the order 0, 2, 1, 3, i.e. bits 3 and 4 of the K index
// are swapped.  The mapping is an involution (logical <-> physical).
static inline constexpr int fp8_moe_kpos(int k) {
    return (k & ~0x18) | ((k & 0x08) << 1) | ((k & 0x10) >> 1);
}

// True when this binary was built with the RDNA4 FP8 kernels (a gfx1200 / gfx1201 target).
bool rdna4_fp8_prefill_built();

// True when the FP8 kernels are built and the current device is RDNA4 (gfx12).
bool rdna4_fp8_prefill_supported();

// Grouped NVFP4 MoE prefill on RDNA4 FP8 WMMA.  Same contract, scratch and outputs as
// launch_moe_nvfp4_prefill_wmma (include/inkling_engine.h); differences:
//   - each X_norm row is rounded to E4M3 with its own scale 448 / amax(|row|),
//   - each NVFP4 weight is folded to one E4M3 byte (nib_nvfp4_fp8_table, fold / 8),
//   - the SITU activations are rounded to E4M3 at FP8_MOE_ACT_SCALE,
// with FP32 accumulation and FP32 outputs.  Uses no memory beyond MoePrefillScratch.
// scratch_cap is the token capacity of `scratch` (default MAX_UBATCH=512; up to FP8_MAX_UBATCH=1024).
void launch_moe_nvfp4_prefill_fp8_wmma(
    const SoaNVFP4Experts & gate_up,
    const SoaNVFP4Experts & down,
    const float           * X_norm,    // [M, N_EMBD]
    const int32_t         * topk_ids,  // [M, TOP_K]
    const float           * topk_w,    // [M, TOP_K]
    MoePrefillScratch     & scratch,
    float                 * Y_out,     // [M, N_EMBD]
    int                     M,
    hipStream_t             stream,
    const float           * post_conv_w  = nullptr, // [N_EMBD, 4] optional fused sconv_post_res
    float                 * post_sconv_h = nullptr, // [N_EMBD, 3]
    const float           * post_res_add = nullptr, // [M, N_EMBD]
    int                     scratch_cap  = MAX_UBATCH);

// Building block of the FP8 MoE (exposed for the kernel tests): X [M, N_EMBD] FP32 ->
//   X8 [M, N_EMBD] E4M3, K-interleaved (fp8_moe_kpos), x8 = RNE(clamp(x * 448 / amax_row)),
//   xs [M]           dequantisation multiplier amax_row / 448 (x ~= e4m3(x8) * xs).
// An all-zero row gets xs = 0; a row holding Inf/NaN gets xs = NaN and zero bytes, so it
// poisons exactly its own outputs (as it would in the FP16 path).
void launch_x_to_fp8_rowscale(const float * X, uint8_t * X8, float * xs, int M, hipStream_t stream);

} // namespace inkling
