#pragma once

// ============================================================================
// kernels_iq.h — GSQ-RCO / IQ / K-quant & Q2_0 MoE support for Nib
// ============================================================================
// Completely isolated from all existing NVFP4 / Q8_0 RDNA3/RDNA4 HIP files:
//   - Load-time GPU dequant+repack of dense BF16/Q4_K/Q5_K/Q6_K/IQ4_XS into
//     SoaQ8Matrix (src/repack_iq.hip)
//   - Load-time GPU SoA repack + per-expert amax for all 9 GSQ-RCO MoE expert
//     formats: Q2_0, IQ4_NL (down) and IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
//     IQ3_XXS, IQ3_S (gate_up) (src/repack_iq.hip)
//   - Native Wave32 decode, batched verify, and grouped FP16 WMMA prefill
//     (src/kernels_moe_iq.hip)
//   - Native RDNA4 FP8 WMMA grouped prefill (src/kernels_wmma_fp8_iq_rdna4.hip)
// ============================================================================

#include "inkling_engine.h"
#include "iq_types.h"

#include <cstddef>
#include <cstdint>

namespace inkling {

// ============================================================================
// Load-Time GPU Repack API (src/repack_iq.hip)
// ============================================================================
namespace detail {

// Dequantizes dense BF16(30), Q4_K(12), Q5_K(13), Q6_K(14), or IQ4_XS(23) rows
// directly on GPU into SoaQ8Matrix at row offset dst_row_offset.
void repack_dense_to_soa_q8(
    uint32_t ggml_type, const void * d_raw,
    int8_t * d_qs, __half * d_d,
    int rows, int K, int dst_row_offset, hipStream_t stream);

// Repacks raw GGML MoE expert blocks (any of the 9 GSQ-RCO types) into 16B-aligned
// SoA planes and computes per-expert max(|w|) into dst.exp_amax[0..n_exp-1].
void repack_iq_experts_to_soa(
    const void * d_raw, const SoaIQExperts & dst, hipStream_t stream);

// Unit-test helper: dequantizes one [K] expert row from SoaIQExperts to FP32 on GPU.
void dequant_iq_expert_row_gpu(
    const SoaIQExperts & src, int exp, int row, float * d_out, hipStream_t stream);

} // namespace detail

// ============================================================================
// Runtime GSQ-RCO MoE Dispatch API (src/kernels_moe_iq.hip)
// ============================================================================

// Single-token decode (2-launch split: gate_up + SITU -> tmp_act, then down + post_conv + res).
void launch_moe_iq_decode(
    const SoaIQExperts & gate_up,
    const SoaIQExperts & down,
    const float        * x_norm,        // [N_EMBD]
    const int32_t      * topk_ids,      // [TOP_K]
    const float        * topk_w,        // [TOP_K]
    float              * tmp_act,       // [TOP_K, N_FF_EXP]
    float              * out,           // [N_EMBD]
    hipStream_t          stream,
    const float        * post_conv_w  = nullptr, // [N_EMBD, 4]
    float              * post_sconv_h = nullptr, // [N_EMBD, 3]
    const float        * post_res_add = nullptr, // [N_EMBD]
    float              * raw_out      = nullptr);// [N_EMBD] optional MTP pre-conv dump

// Batched verify / multi-slot decode: unique-expert gate_up + SITU into act[(r*TOP_K+slot)*N_FF_EXP+row].
void launch_verify_moe_gu_iq(
    const SoaIQExperts & gate_up,
    const float        * x_norm, int xstride,
    const int32_t      * unique_exp, const uint32_t * exp_tok_mask,
    const int32_t      * exp_tok_slot, const float * exp_tok_w,
    const int32_t      * exp_tok_cnt, const int32_t * exp_tok_list,
    const int32_t      * d_num_unique, int m,
    float              * act, hipStream_t stream);

// Batched verify / multi-slot decode: unique-expert down into buf_slot[(r*TOP_K+slot)*N_EMBD+row].
void launch_verify_moe_dn_iq(
    const SoaIQExperts & down,
    const float        * act,
    const int32_t      * unique_exp, const uint32_t * exp_tok_mask,
    const int32_t      * exp_tok_slot, const float * exp_tok_w,
    const int32_t      * exp_tok_cnt, const int32_t * exp_tok_list,
    const int32_t      * d_num_unique, int m,
    float              * buf_slot, hipStream_t stream);

// Grouped MoE Prefill using RDNA3/RDNA4 FP16 WMMA + fused SITU + weighted combine + optional sconv.
void launch_moe_iq_prefill_wmma(
    const SoaIQExperts & gate_up,
    const SoaIQExperts & down,
    const float        * X_norm,       // [M, N_EMBD]
    const int32_t      * topk_ids,     // [M, TOP_K]
    const float        * topk_w,       // [M, TOP_K]
    MoePrefillScratch  & scratch,
    float              * Y_out,        // [M, N_EMBD]
    int                  M,
    hipStream_t          stream,
    const float        * post_conv_w  = nullptr,
    float              * post_sconv_h = nullptr,
    const float        * post_res_add = nullptr,
    int                  scratch_cap  = MAX_UBATCH);

// Grouped MoE Prefill using RDNA4 FP8 WMMA (src/kernels_wmma_fp8_iq_rdna4.hip; stubbed on non-gfx12).
void launch_moe_iq_prefill_fp8_wmma(
    const SoaIQExperts & gate_up,
    const SoaIQExperts & down,
    const float        * X_norm,       // [M, N_EMBD]
    const int32_t      * topk_ids,     // [M, TOP_K]
    const float        * topk_w,       // [M, TOP_K]
    MoePrefillScratch  & scratch,
    float              * Y_out,        // [M, N_EMBD]
    int                  M,
    hipStream_t          stream,
    const float        * post_conv_w  = nullptr,
    float              * post_sconv_h = nullptr,
    const float        * post_res_add = nullptr,
    int                  scratch_cap  = MAX_UBATCH);

} // namespace inkling
