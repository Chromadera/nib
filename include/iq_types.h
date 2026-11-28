#pragma once

// ============================================================================
// iq_types.h — Lightweight GSQ-RCO MoE quant enums & SoaIQExperts descriptor
// ============================================================================
// Kept separate from kernels_iq.h so inkling_engine.h can embed SoaIQExperts
// in PhysLayerWeights without circular header dependencies or touching any
// existing struct used by the 19 existing HIP kernel files.
// ============================================================================

#include <cstdint>
#include <hip/hip_fp16.h>

namespace inkling {

enum MoEQuantType : int32_t {
    MOE_Q_NONE    = 0,
    MOE_Q_Q8_0    = 1,
    MOE_Q_NVFP4   = 2,
    MOE_Q_Q2_0    = 3,  // ggml 42: QK=64,  18 B/blk -> p0=qs[16], d[2]
    MOE_Q_IQ4_NL  = 4,  // ggml 20: QK=32,  18 B/blk -> p0=qs[16], d[2]
    MOE_Q_IQ1_S   = 5,  // ggml 19: QK=256, 50 B/blk -> p0=qs[32]+qh[16]=48, d[2]
    MOE_Q_IQ1_M   = 6,  // ggml 29: QK=256, 56 B/blk -> p0=qs[32]+qh[16]=48, p1=scales[8] (d packed in p1)
    MOE_Q_IQ2_XXS = 7,  // ggml 16: QK=256, 66 B/blk -> p0=qs[64], d[2]
    MOE_Q_IQ2_XS  = 8,  // ggml 17: QK=256, 74 B/blk -> p0=qs[64], p1=scales[8], d[2]
    MOE_Q_IQ2_S   = 9,  // ggml 22: QK=256, 82 B/blk -> p0=qs[64]+qh[8]+scales[8]=80, d[2]
    MOE_Q_IQ3_XXS = 10, // ggml 18: QK=256, 98 B/blk -> p0=qs[96], d[2]
    MOE_Q_IQ3_S   = 11, // ggml 21: QK=256, 110 B/blk -> p0=qs[64]+signs[32]=96, p1=qh[8]+scales[4]=12, d[2]
};

static inline bool is_iq_moe_type(MoEQuantType t) {
    return t >= MOE_Q_Q2_0 && t <= MOE_Q_IQ3_S;
}

static inline MoEQuantType ggml_type_to_moe_quant(uint32_t ggml_t) {
    switch (ggml_t) {
        case 8:  return MOE_Q_Q8_0;
        case 40: return MOE_Q_NVFP4;
        case 42: return MOE_Q_Q2_0;
        case 20: return MOE_Q_IQ4_NL;
        case 19: return MOE_Q_IQ1_S;
        case 29: return MOE_Q_IQ1_M;
        case 16: return MOE_Q_IQ2_XXS;
        case 17: return MOE_Q_IQ2_XS;
        case 22: return MOE_Q_IQ2_S;
        case 18: return MOE_Q_IQ3_XXS;
        case 21: return MOE_Q_IQ3_S;
        default: return MOE_Q_NONE;
    }
}

static inline const char * moe_quant_name(MoEQuantType t) {
    switch (t) {
        case MOE_Q_Q8_0:    return "Q8_0";
        case MOE_Q_NVFP4:   return "NVFP4";
        case MOE_Q_Q2_0:    return "Q2_0";
        case MOE_Q_IQ4_NL:  return "IQ4_NL";
        case MOE_Q_IQ1_S:   return "IQ1_S";
        case MOE_Q_IQ1_M:   return "IQ1_M";
        case MOE_Q_IQ2_XXS: return "IQ2_XXS";
        case MOE_Q_IQ2_XS:  return "IQ2_XS";
        case MOE_Q_IQ2_S:   return "IQ2_S";
        case MOE_Q_IQ3_XXS: return "IQ3_XXS";
        case MOE_Q_IQ3_S:   return "IQ3_S";
        default:            return "NONE";
    }
}

static inline int moe_quant_block_k(MoEQuantType t) {
    switch (t) {
        case MOE_Q_Q8_0:
        case MOE_Q_IQ4_NL:  return 32;
        case MOE_Q_NVFP4:
        case MOE_Q_Q2_0:    return 64;
        case MOE_Q_IQ1_S:
        case MOE_Q_IQ1_M:
        case MOE_Q_IQ2_XXS:
        case MOE_Q_IQ2_XS:
        case MOE_Q_IQ2_S:
        case MOE_Q_IQ3_XXS:
        case MOE_Q_IQ3_S:   return 256;
        default:            return 0;
    }
}

// Primary 16-byte-aligned plane bytes per block.
static inline int moe_quant_p0_bytes(MoEQuantType t) {
    switch (t) {
        case MOE_Q_Q2_0:    return 16;
        case MOE_Q_IQ4_NL:  return 16;
        case MOE_Q_IQ1_S:   return 48; // qs[32] + qh[16]
        case MOE_Q_IQ1_M:   return 48; // qs[32] + qh[16]
        case MOE_Q_IQ2_XXS: return 64; // qs[64]
        case MOE_Q_IQ2_XS:  return 64; // qs[64]
        case MOE_Q_IQ2_S:   return 80; // qs[64] + qh[8] + scales[8]
        case MOE_Q_IQ3_XXS: return 96; // qs[96]
        case MOE_Q_IQ3_S:   return 96; // qs[64] + signs[32]
        default:            return 0;
    }
}

// Secondary aligned plane bytes per block (0 if unused).
static inline int moe_quant_p1_bytes(MoEQuantType t) {
    switch (t) {
        case MOE_Q_IQ1_M:   return 8;  // scales[8] (also packs fp16 d)
        case MOE_Q_IQ2_XS:  return 8;  // scales[8]
        case MOE_Q_IQ3_S:   return 12; // qh[8] + scales[4]
        default:            return 0;
    }
}

static inline bool moe_quant_has_d(MoEQuantType t) {
    return is_iq_moe_type(t) && (t != MOE_Q_IQ1_M);
}

struct SoaIQExperts {
    MoEQuantType type     = MOE_Q_NONE;
    uint8_t    * p0       = nullptr; // [n_exp, rows, (K/QK) * p0_bytes] (16B-aligned per block!)
    uint8_t    * p1       = nullptr; // [n_exp, rows, (K/QK) * p1_bytes] (optional, 4B/8B-aligned)
    __half     * d        = nullptr; // [n_exp, rows, (K/QK)] (nullptr for IQ1_M)
    float      * exp_amax = nullptr; // [n_exp] max(|w|) per expert (for RDNA4 FP8 WMMA scaling)
    int          n_exp    = 128;
    int          rows     = 0;       // 2304 for gate_up, 2304 for down
    int          K        = 0;       // 2304 for gate_up, 1152 for down
};

} // namespace inkling
