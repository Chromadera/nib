// ============================================================================
// kernels_wmma_fp8_stub.cpp — host-only RDNA4 FP8 prefill API for builds without gfx12 targets
// ============================================================================
// Linked instead of src/kernels_wmma_fp8_rdna4.hip when GPU_TARGET has no gfx1200 / gfx1201
// (see CMakeLists.txt), so host code can query FP8 support at runtime with no build flag.  The
// engine never calls the launchers when rdna4_fp8_prefill_supported() is false; reaching one
// anyway is a programming error and aborts.
// ============================================================================
#include "kernels_wmma_fp8.h"

#include <cstdio>
#include <cstdlib>

namespace inkling {
namespace {

[[noreturn]] void not_built(const char * who) {
    std::fprintf(stderr,
                 "[inkling kernels_wmma_fp8] %s: this Nib binary was built without RDNA4 FP8 kernels "
                 "(GPU_TARGET has no gfx1200 / gfx1201)\n", who);
    std::abort();
}

} // namespace

bool rdna4_fp8_prefill_built() {
    return false;
}

bool rdna4_fp8_prefill_supported() {
    return false;
}

void launch_moe_nvfp4_prefill_fp8_wmma(const SoaNVFP4Experts &, const SoaNVFP4Experts &, const float *,
                                       const int32_t *, const float *, MoePrefillScratch &, float *, int,
                                       hipStream_t, const float *, float *, const float *, int) {
    not_built("launch_moe_nvfp4_prefill_fp8_wmma");
}

void launch_x_to_fp8_rowscale(const float *, uint8_t *, float *, int, hipStream_t) {
    not_built("launch_x_to_fp8_rowscale");
}

} // namespace inkling
