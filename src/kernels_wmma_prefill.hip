// ============================================================================
// kernels_wmma_prefill.hip — Hardware RDNA3 FP16 WMMA prefill GEMM + grouped MoE
// Target: AMD RDNA3 gfx1100 (RX 7900 XTX), Wave32, __builtin_amdgcn_wmma_f32_16x16x16_f16_w32.
//
// WMMA Wave32 register mapping (gfx11, verified by the GPU unit test):
//   A (16x16, M x K) : lane l holds row (l % 16), all 16 K values, as 16 halves.
//                      Lanes 16..31 replicate lanes 0..15.
//   B (16x16, K x N) : lane l holds column (l % 16), all 16 K values, as 16 halves.
//   C/D (16x16 FP32) : lane l, element i -> row (2*i + l/16), column (l % 16).
// B column n of W^T is row n of W, so a W tile is stored in LDS row-major [n][k] and each lane
// reads its 16 contiguous K halves with two ds_load_b128.
//
// Mainloop: BK = 64, global -> registers -> dequantise to FP16 -> LDS -> WMMA with FP32 accumulation.
//   dense GEMM  : B chunks spread over the block, step k+1 in flight during step k; LDS row stride
//                 72 halves (144 B); 8-warp (4x2) and 16-warp (4x4) tiles keep VGPRs <= 157.
//                 launch_soa_q8_gemm_wmma_qkv3 fuses 3 QKV projections sharing K into one launch.
//   grouped MoE : Zero-padding XOR-swizzled LDS layout (LDK_XOR = 64 halves = 128 B/row, 100%
//                 bank-conflict-free) cuts TM=48, BN=192 tile LDS from 62,208 B to 30,720 B
//                 (30.0 KB <= 32 KB -> 2 blocks/CU = 4 blocks/WGP!). One thread per weight row
//                 streams full 128-byte lines (4 NVFP4 / 2 Q8 k-steps) with in-place single-SReg
//                 prefetch overlapped with compute() and single-bf fragment streaming, keeping all
//                 MoE WMMA kernels at 112..140 VGPRs (< 160, 0 spills, 9..12 waves/SIMD).
// Note: gfx1100 WMMA FP32 accumulation is not bit-exact even for small-integer FP16 inputs
// (measured: ~1 ulp on some outputs), so results match an FP64 reference of the FP16 operands to
// ~1e-5 relative, not bitwise.
//
// Weight dequant to FP16:
//   Q8_0  : q (exact in FP16 via the 0x6400|u trick) * d, rounded once (RNE) — same as fp16(q*d).
//   NVFP4 : kvalues_mxfp4 * ue4m3/2 is EXACT in FP16 (<= 6 significant bits, range 2^-10..2688);
//           the per-expert scale2 is applied to the FP32 accumulator, so NVFP4 weights are exact.
//
// Grouped MoE (zero host syncs):
//   route kernel (1 block): per-expert token bitmasks in LDS -> counts, offsets, deterministic
//     slot order (tokens ascending inside each expert), tile worklist, perm_tok, perm_w.
//     A token listing the same expert twice gets ONE slot with the summed router weight
//     (mathematically identical). Out-of-range expert ids contribute nothing.
//   convert : X_norm -> FP16 once (into scratch.perm_x, reinterpreted as FP16 [M, 2304]).
//   stage 1 : grouped WMMA gate+up (same block, same lanes) + scale2 + SITU -> perm_act (FP16).
//             A rows are gathered from the FP16 X_norm through perm_tok (no [M*8, 2304] copy).
//   stage 2 : grouped WMMA down * scale2 * perm_w -> perm_down.
//   combine : per token, sum its <= 8 slots in k order (deterministic, no float atomics).
// Grids are sized from the host-known worst-case tile count; blocks past *d_num_tiles exit.
// ============================================================================
#include "inkling_engine.h"

#include <cstdio>
#include <cstdlib>

namespace inkling {
namespace {

typedef int      v4i      __attribute__((ext_vector_type(4)));
typedef _Float16 half16_t __attribute__((ext_vector_type(16)));
typedef float    float8_t __attribute__((ext_vector_type(8)));

constexpr int BK      = 64;      // K per mainloop step (2 Q8 blocks / 1 NVFP4 block)
constexpr int LDK_XOR = BK;      // Grouped MoE & dense GEMM XOR-swizzled LDS row stride in halves (128 B, 0 pad)

// ---------------------------------------------------------------------------
// Device helpers
// ---------------------------------------------------------------------------
__device__ __forceinline__ float situ_scalar(float gate, float up, float beta, float inv_beta, float lb, float inv_lb) {
    const float act = beta * tanhf(gate * inv_beta) / (1.0f + expf(-gate));
    const float u   = lb > 0.0f ? lb * tanhf(up * inv_lb) : up;
    return act * u;
}

// Bank-conflict-free XOR-swizzled LDS fragment load (LDK_XOR = 64 halves = 8 v4i per row).
// For any kc in 0..3, lanes l16 in 0..15 access subchunk (2*kc + sub) ^ (l16 & 7) across rows,
// hitting all 32 LDS banks without conflicts while using zero padding bytes.
__device__ __forceinline__ half16_t lds_frag_xor(const _Float16 * row_ptr, int rlow, int kc) {
    const v4i * d4 = reinterpret_cast<const v4i *>(row_ptr);
    const v4i a = d4[(2 * kc + 0) ^ rlow];
    const v4i b = d4[(2 * kc + 1) ^ rlow];
    half16_t r;
    __builtin_memcpy(&r, &a, 16);
    __builtin_memcpy(reinterpret_cast<char *>(&r) + 16, &b, 16);
    return r;
}

__device__ __forceinline__ uint32_t bits_of(__half2 h) { return __builtin_bit_cast(uint32_t, h); }
__device__ __forceinline__ __half2  h2_of(uint32_t u)   { return __builtin_bit_cast(__half2, u); }

// ---------------------------------------------------------------------------
// Weight formats: global -> registers (Reg / SReg2L) -> FP16 LDS row segment
// ---------------------------------------------------------------------------
struct FmtQ8 {
    static constexpr int CHUNKS = 4;   // 16-byte chunks per row per BK step
    struct Reg { v4i q; __half d; };

    // row_q: start of the weight row; row_d: start of its fp16 scales.
    __device__ static __forceinline__ Reg load(const int8_t * row_q, const __half * row_d, int k0, int c, int K) {
        Reg r;
        const int kk = k0 + c * 16;
        if (row_q != nullptr && kk < K) {
            r.q = *reinterpret_cast<const v4i *>(row_q + kk); // cacheable: the line is reused next k-step
            r.d = row_d[kk >> 5];
        } else {
            r.q = v4i{0, 0, 0, 0};
            r.d = __float2half(0.0f);
        }
        return r;
    }

    __device__ static __forceinline__ void dequant16(uint32_t (&o)[8], const Reg & r) {
        const __half2 d2   = __half2half2(r.d);
        const __half2 bias = h2_of(0x64806480u); // (1152, 1152) = 1024 + 128
#pragma unroll
        for (int w = 0; w < 4; ++w) {
            const uint32_t u = (uint32_t) r.q[w] ^ 0x80808080u;                   // q + 128 in 0..255
            const uint32_t h01 = __builtin_amdgcn_perm(0x64646464u, u, 0x04010400u); // 1024+u0, 1024+u1
            const uint32_t h23 = __builtin_amdgcn_perm(0x64646464u, u, 0x04030402u); // 1024+u2, 1024+u3
            o[2 * w + 0] = bits_of(__hmul2(__hsub2(h2_of(h01), bias), d2));
            o[2 * w + 1] = bits_of(__hmul2(__hsub2(h2_of(h23), bias), d2));
        }
    }

    __device__ static __forceinline__ void store_xor(_Float16 * dst_row, int row_low3, int c, const Reg & r) {
        uint32_t o[8];
        dequant16(o, r);
        v4i * d4 = reinterpret_cast<v4i *>(dst_row);
        d4[(2 * c + 0) ^ row_low3] = v4i{(int) o[0], (int) o[1], (int) o[2], (int) o[3]};
        d4[(2 * c + 1) ^ row_low3] = v4i{(int) o[4], (int) o[5], (int) o[6], (int) o[7]};
    }

    // 2-lane coalesced super-step streaming (grouped MoE): 2 adjacent threads (bchk in 0..1) share
    // one weight row and coalesce 64 B each (full 128 B cacheline across SUPER = 2 k-steps),
    // cutting SReg VGPRs per lane in half and halving per-warp cacheline spread.
    static constexpr int SUPER_GU = 2;
    static constexpr int SUPER_DN = 2;
    template <int SU> struct SReg2L { v4i q[2 * SU]; uint32_t d; };

    template <int SU>
    __device__ static __forceinline__ SReg2L<SU> sload_2l(const int8_t * row_q, const __half * row_d, int bchk, int ks0, int K) {
        static_assert(SU <= 2, "FmtQ8 SReg2L packs SU <= 2 fp16 scales into uint32_t");
        SReg2L<SU> r;
        r.d = 0u;
        const uint16_t * d16 = reinterpret_cast<const uint16_t *>(row_d + (ks0 >> 5));
#pragma unroll
        for (int t = 0; t < SU; ++t) {
            const int kk = ks0 + t * 64 + bchk * 32;
            if (kk < K) {
                r.q[2 * t + 0] = *reinterpret_cast<const v4i *>(row_q + kk);
                r.q[2 * t + 1] = *reinterpret_cast<const v4i *>(row_q + kk + 16);
                const uint32_t dv = (uint32_t) d16[2 * t + bchk];
                r.d |= (dv << (16 * t));
            } else {
                r.q[2 * t + 0] = v4i{0, 0, 0, 0};
                r.q[2 * t + 1] = v4i{0, 0, 0, 0};
            }
        }
        return r;
    }

    template <int SU>
    __device__ static __forceinline__ void sstore_xor_2l(_Float16 * dst_row, int row_low3, int bchk, const SReg2L<SU> & r, int t) {
        const __half dh = __builtin_bit_cast(__half, (uint16_t) ((r.d >> (16 * t)) & 0xFFFFu));
#pragma unroll
        for (int sub = 0; sub < 2; ++sub) {
            Reg x;
            x.q = r.q[2 * t + sub];
            x.d = dh;
            store_xor(dst_row, row_low3, 2 * bchk + sub, x);
        }
    }
};

struct FmtNVFP4 {
    struct Reg { v4i q; uint32_t sc; };

    // Exact bit-level UE4M3 (with *0.5 scale) -> FP16x2 conversion (0 branches, 0 FP32 ops).
    __device__ static __forceinline__ __half2 ue4m3_to_h2(uint32_t x) {
        const uint32_t u = x & 0x7Fu;
        uint32_t h;
        if (u == 0x7Fu) {
            h = 0u;
        } else if (u >= 8u) {
            h = (u << 7) + 0x1C00u;
        } else {
            h = (__builtin_amdgcn_perm(0x1F1E1D1Cu, 0x1A181400u, u) & 0xFFu) << 8;
        }
        return h2_of(h | (h << 16));
    }

    __device__ static __forceinline__ void hbytes(uint32_t q, uint32_t & lo, uint32_t & hi) {
        constexpr uint32_t T0 = 0x42403C00u; // fp16 high bytes of 0,1,2,3
        constexpr uint32_t T1 = 0x4A484644u; // fp16 high bytes of 4,6,8,12
        lo = __builtin_amdgcn_perm(T1, T0, q & 0x07070707u)        | ((q & 0x08080808u) << 4);
        hi = __builtin_amdgcn_perm(T1, T0, (q >> 4) & 0x07070707u) | (q & 0x80808080u);
    }

    __device__ static __forceinline__ void store_xor(_Float16 * dst_row, int row_low3, int c, const Reg & r) {
        v4i * d4 = reinterpret_cast<v4i *>(dst_row);
#pragma unroll
        for (int s = 0; s < 2; ++s) {
            const __half2 sc2 = ue4m3_to_h2((r.sc >> (8 * s)) & 0xFFu);
            uint32_t lo0, hi0, lo1, hi1;
            hbytes((uint32_t) r.q[2 * s + 0], lo0, hi0); // elems 0..3, 8..11
            hbytes((uint32_t) r.q[2 * s + 1], lo1, hi1); // elems 4..7, 12..15
            const uint32_t o0 = bits_of(__hmul2(h2_of(__builtin_amdgcn_perm(lo0, 0u, 0x050C040Cu)), sc2));
            const uint32_t o1 = bits_of(__hmul2(h2_of(__builtin_amdgcn_perm(lo0, 0u, 0x070C060Cu)), sc2));
            const uint32_t o2 = bits_of(__hmul2(h2_of(__builtin_amdgcn_perm(lo1, 0u, 0x050C040Cu)), sc2));
            const uint32_t o3 = bits_of(__hmul2(h2_of(__builtin_amdgcn_perm(lo1, 0u, 0x070C060Cu)), sc2));
            const uint32_t o4 = bits_of(__hmul2(h2_of(__builtin_amdgcn_perm(hi0, 0u, 0x050C040Cu)), sc2));
            const uint32_t o5 = bits_of(__hmul2(h2_of(__builtin_amdgcn_perm(hi0, 0u, 0x070C060Cu)), sc2));
            const uint32_t o6 = bits_of(__hmul2(h2_of(__builtin_amdgcn_perm(hi1, 0u, 0x050C040Cu)), sc2));
            const uint32_t o7 = bits_of(__hmul2(h2_of(__builtin_amdgcn_perm(hi1, 0u, 0x070C060Cu)), sc2));
            d4[(4 * c + 2 * s + 0) ^ row_low3] = v4i{(int) o0, (int) o1, (int) o2, (int) o3};
            d4[(4 * c + 2 * s + 1) ^ row_low3] = v4i{(int) o4, (int) o5, (int) o6, (int) o7};
        }
    }

    // 2-lane coalesced super-step streaming:
    //   Gate+Up (K = 2304 = 9 * 256): SUPER_GU = 4 (128 B/row = 4 k-steps, 0 tail iterations)
    //   Down    (K = 1152 = 6 * 192): SUPER_DN = 3 (96 B/row  = 3 k-steps, 0 tail iterations)
    static constexpr int SUPER_GU = 4;
    static constexpr int SUPER_DN = 3;
    template <int SU> struct SReg2L { v4i q[SU]; uint16_t sc[SU]; };

    template <int SU>
    __device__ static __forceinline__ SReg2L<SU> sload_2l(const uint8_t * row_q, const uint8_t * row_s, int bchk, int ks0, int K) {
        SReg2L<SU> r;
        const uint16_t * s16 = reinterpret_cast<const uint16_t *>(row_s + (ks0 >> 4));
#pragma unroll
        for (int t = 0; t < SU; ++t) {
            const int kk = ks0 + t * 64 + bchk * 32;
            r.q[t]  = (kk < K) ? *reinterpret_cast<const v4i *>(row_q + (kk >> 1)) : v4i{0, 0, 0, 0};
            r.sc[t] = (kk < K) ? s16[2 * t + bchk] : (uint16_t) 0u;
        }
        return r;
    }

    template <int SU>
    __device__ static __forceinline__ void sstore_xor_2l(_Float16 * dst_row, int row_low3, int bchk, const SReg2L<SU> & r, int t) {
        Reg x;
        x.q  = r.q[t];
        x.sc = (uint32_t) r.sc[t];
        store_xor(dst_row, row_low3, bchk, x);
    }
};

template <class Fmt> struct WPtr;
template <> struct WPtr<FmtQ8>    { typedef int8_t  Q; typedef __half  S; };
template <> struct WPtr<FmtNVFP4> { typedef uint8_t Q; typedef uint8_t S; };

// ---------------------------------------------------------------------------
// A-operand source for dense GEMM (FP32 rows converted to FP16 while staging)
// ---------------------------------------------------------------------------
struct AF32 {
    typedef float  T;
    typedef float4 Reg;
    static constexpr int EPL = 4; // elements per 16-byte load
    __device__ static __forceinline__ Reg load(const T * p) { return *reinterpret_cast<const float4 *>(p); }
    __device__ static __forceinline__ Reg zero() { return make_float4(0.0f, 0.0f, 0.0f, 0.0f); }
    __device__ static __forceinline__ void store_xor(_Float16 * dst_row, int rlow, int c4, const Reg & r) {
        const __half2 h01 = __floats2half2_rn(r.x, r.y);
        const __half2 h23 = __floats2half2_rn(r.z, r.w);
        reinterpret_cast<uint2 *>(dst_row)[c4 ^ (rlow << 1)] = make_uint2(bits_of(h01), bits_of(h23));
    }
};

// ---------------------------------------------------------------------------
// Dense WMMA tile mainloop (BM x BN tile, WARPS_M x WARPS_N warps, LDK_XOR = 64)
// ---------------------------------------------------------------------------
// Zero-padding XOR-swizzled LDS (LDK_XOR = 64 halves = 128 B/row) + single-bf fragment streaming:
//   - <128, 128, 4, 4>: LDS drops 36,864 B -> 32,768 B (exact 32 KB -> 2 blocks/CU!), 124 VGPRs
//   - <64,  128, 4, 2>: LDS drops 27,648 B -> 24,576 B (24 KB -> 2 blocks/CU!), 124 VGPRs
template <int BM, int BN, int WARPS_M, int WARPS_N, bool PAIRED, class Fmt, class AS = AF32>
struct TileGemm {
    static constexpr int NT    = WARPS_M * WARPS_N * 32;
    static constexpr int WTM   = BM / WARPS_M;
    static constexpr int FM    = WTM / 16;
    static constexpr int OUTN  = PAIRED ? BN / 2 : BN;
    static constexpr int WTN   = OUTN / WARPS_N;
    static constexpr int FNO   = WTN / 16;
    static constexpr int FN    = PAIRED ? 2 * FNO : FNO;
    static constexpr int A_PR  = BK / AS::EPL;                 // A loads per tile row (16)
    static constexpr int A_N   = BM * A_PR;                    // A loads per k-step
    static constexpr int A_IT  = (A_N + NT - 1) / NT;
    static constexpr bool A_EXACT = (A_N % NT) == 0;           // else the last round is partial
    static constexpr int B_IT  = (BN * Fmt::CHUNKS) / NT;
    static_assert(BM % (16 * WARPS_M) == 0 && OUTN % (16 * WARPS_N) == 0, "warp tiling");
    static_assert((BN * Fmt::CHUNKS) % NT == 0 && B_IT >= 1, "B load tiling");

    typedef typename WPtr<Fmt>::Q WQ;
    typedef typename WPtr<Fmt>::S WS;

    struct Smem {
        _Float16 a[BM * LDK_XOR];
        _Float16 b[BN * LDK_XOR];
    };

    const typename AS::T * a_ptr[A_IT];
    const WQ *    b_q[B_IT];
    const WS *    b_s[B_IT];

    typename AS::Reg   ar[A_IT];
    typename Fmt::Reg  br[B_IT];

    __device__ __forceinline__ static bool a_live(int p) { return A_EXACT || threadIdx.x + p * NT < A_N; }
    __device__ __forceinline__ static int a_row(int p) { return a_live(p) ? (threadIdx.x + p * NT) / A_PR : 0; }
    __device__ __forceinline__ static int a_c4(int p)  { return (threadIdx.x + p * NT) % A_PR; }
    __device__ __forceinline__ static int a_col(int p) { return a_c4(p) * AS::EPL; }
    __device__ __forceinline__ static int b_row(int p) { return (threadIdx.x + p * NT) / Fmt::CHUNKS; }
    __device__ __forceinline__ static int b_chk(int p) { return (threadIdx.x + p * NT) % Fmt::CHUNKS; }

    __device__ __forceinline__ void loadA(int k0, int K) {
#pragma unroll
        for (int p = 0; p < A_IT; ++p) {
            const int kk = k0 + a_col(p);
            ar[p] = (a_ptr[p] != nullptr && kk < K) ? AS::load(a_ptr[p] + kk) : AS::zero();
        }
    }

    __device__ __forceinline__ void load(int k0, int K) {
        loadA(k0, K);
#pragma unroll
        for (int p = 0; p < B_IT; ++p) {
            br[p] = Fmt::load(b_q[p], b_s[p], k0, b_chk(p), K);
        }
    }

    __device__ __forceinline__ void storeA(Smem & sm) {
#pragma unroll
        for (int p = 0; p < A_IT; ++p) {
            if (a_live(p)) {
                const int r = a_row(p);
                AS::store_xor(&sm.a[r * LDK_XOR], r & 7, a_c4(p), ar[p]);
            }
        }
    }

    __device__ __forceinline__ void store(Smem & sm) {
        storeA(sm);
#pragma unroll
        for (int p = 0; p < B_IT; ++p) {
            const int r = b_row(p);
            Fmt::store_xor(&sm.b[r * LDK_XOR], r & 7, b_chk(p), br[p]);
        }
    }

    __device__ __forceinline__ static int b_frag_row(int wn, int f) {
        if (PAIRED) {
            return (f < FNO) ? (wn * WTN + f * 16) : (OUTN + wn * WTN + (f - FNO) * 16);
        }
        return wn * WTN + f * 16;
    }

    __device__ __forceinline__ void run(Smem & sm, int K, float8_t (&acc)[FM][FN]) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int wm   = warp / WARPS_N;
        const int wn   = warp % WARPS_N;
        const int l16  = lane & 15;

#pragma unroll
        for (int i = 0; i < FM; ++i) {
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                acc[i][j] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            }
        }

        const int nk = (K + BK - 1) / BK;
        load(0, K);
        for (int kt = 0; kt < nk; ++kt) {
            store(sm);
            __syncthreads();
            if (kt + 1 < nk) {
                load((kt + 1) * BK, K); // in flight during the WMMAs below
            }
            compute(sm, acc, wm, wn, l16);
            __syncthreads();
        }
    }

    __device__ __forceinline__ void compute(Smem & sm, float8_t (&acc)[FM][FN], int wm, int wn, int l16) {
        const int rlow = l16 & 7;
#pragma unroll
        for (int kc = 0; kc < BK / 16; ++kc) {
            __builtin_amdgcn_sched_barrier(0);
            half16_t af[FM];
#pragma unroll
            for (int i = 0; i < FM; ++i) {
                af[i] = lds_frag_xor(&sm.a[(wm * WTM + i * 16 + l16) * LDK_XOR], rlow, kc);
            }
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                __builtin_amdgcn_sched_barrier(0);
                const half16_t bf = lds_frag_xor(&sm.b[(b_frag_row(wn, j) + l16) * LDK_XOR], rlow, kc);
#pragma unroll
                for (int i = 0; i < FM; ++i) {
                    acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[i], bf, acc[i][j]);
                }
            }
        }
    }

    __device__ __forceinline__ static int out_row(int i, int v) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        return (warp / WARPS_N) * WTM + i * 16 + 2 * v + (lane >> 4);
    }
    __device__ __forceinline__ static int out_col(int j) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        return (warp % WARPS_N) * WTN + j * 16 + (lane & 15);
    }
};

// ---------------------------------------------------------------------------
// Grouped MoE WMMA tile mainloop (2-lane coalesced super-step + XOR-swizzled LDS)
// ---------------------------------------------------------------------------
// 2 adjacent lanes (brow = threadIdx.x >> 1, bchk = threadIdx.x & 1) share each B row:
//   - Cuts SReg VGPRs per lane in half (36 -> 18 VGPRs), dropping total kernel VGPRs from
//     151..156 to 88..90 VGPRs (12..15 waves/SIMD at 2 blocks/CU!).
//   - Halves the simultaneous cachelines touched per warp (16 rows/warp instead of 32), eliminating
//     GL0/L2 MSHR thrashing when 4 blocks/WGP stream cold GDDR6 expert weights.
//   - 12 warps (3 x 4 = 384 threads = 3 warps/SIMD) balances all 4 SIMDs per WGP and gives 100%
//     thread utilization during B load/dequant for BN = 192 (384 == 2 * 192).
template <int BM, int BN, int WARPS_M, int WARPS_N, bool PAIRED, class Fmt, int SU>
struct TileGemmMoe {
    static constexpr int NT    = WARPS_M * WARPS_N * 32;
    static constexpr int WTM   = BM / WARPS_M;
    static constexpr int FM    = WTM / 16;
    static constexpr int OUTN  = PAIRED ? BN / 2 : BN;
    static constexpr int WTN   = OUTN / WARPS_N;
    static constexpr int FNO   = WTN / 16;
    static constexpr int FN    = PAIRED ? 2 * FNO : FNO;
    static constexpr int A_PR  = BK / 8;                       // 8 FP16 elements per 16-byte v4i load
    static constexpr int A_N   = BM * A_PR;
    static constexpr int A_IT  = (A_N + NT - 1) / NT;
    static constexpr bool A_EXACT = (A_N % NT) == 0;
    static_assert(BM % (16 * WARPS_M) == 0 && OUTN % (16 * WARPS_N) == 0, "warp tiling");
    static_assert(NT >= 2 * BN, "at least 2 threads per B row");

    typedef typename WPtr<Fmt>::Q WQ;
    typedef typename WPtr<Fmt>::S WS;

    struct Smem {
        _Float16 a[BM * LDK_XOR];
        _Float16 b[BN * LDK_XOR];
    };

    const _Float16 * a_ptr[A_IT];
    const WQ *       s_rowq = nullptr;
    const WS *       s_rows = nullptr;
    typename Fmt::template SReg2L<SU> sr;
    v4i              ar[A_IT];

    __device__ __forceinline__ static bool a_live(int p) { return A_EXACT || threadIdx.x + p * NT < A_N; }
    __device__ __forceinline__ static int a_row(int p) { return a_live(p) ? (threadIdx.x + p * NT) / A_PR : 0; }
    __device__ __forceinline__ static int a_sub(int p) { return (threadIdx.x + p * NT) % A_PR; }
    __device__ __forceinline__ static int a_col(int p) { return a_sub(p) * 8; }

    __device__ __forceinline__ void loadA(int k0, int K) {
#pragma unroll
        for (int p = 0; p < A_IT; ++p) {
            const int kk = k0 + a_col(p);
            ar[p] = (a_ptr[p] != nullptr && kk < K) ? *reinterpret_cast<const v4i *>(a_ptr[p] + kk) : v4i{0, 0, 0, 0};
        }
    }

    __device__ __forceinline__ void storeA(Smem & sm) {
#pragma unroll
        for (int p = 0; p < A_IT; ++p) {
            if (a_live(p)) {
                const int r = a_row(p);
                const int s = a_sub(p) ^ (r & 7);
                reinterpret_cast<v4i *>(&sm.a[r * LDK_XOR])[s] = ar[p];
            }
        }
    }

    __device__ __forceinline__ static int b_frag_row(int wn, int f) {
        if (PAIRED) {
            return (f < FNO) ? (wn * WTN + f * 16) : (OUTN + wn * WTN + (f - FNO) * 16);
        }
        return wn * WTN + f * 16;
    }

    __device__ __forceinline__ void compute(Smem & sm, float8_t (&acc)[FM][FN], int wm, int wn, int l16, int m_valid) {
        bool live[FM];
        bool any = false;
#pragma unroll
        for (int i = 0; i < FM; ++i) {
            live[i] = wm * WTM + i * 16 < m_valid;
            any |= live[i];
        }
        if (!any) {
            return;
        }
        const int rlow = l16 & 7;
#pragma unroll
        for (int kc = 0; kc < BK / 16; ++kc) {
            __builtin_amdgcn_sched_barrier(0);
            half16_t af[FM];
#pragma unroll
            for (int i = 0; i < FM; ++i) {
                af[i] = lds_frag_xor(&sm.a[(wm * WTM + i * 16 + l16) * LDK_XOR], rlow, kc);
            }
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                __builtin_amdgcn_sched_barrier(0);
                const half16_t bf = lds_frag_xor(&sm.b[(b_frag_row(wn, j) + l16) * LDK_XOR], rlow, kc);
#pragma unroll
                for (int i = 0; i < FM; ++i) {
                    if (FM == 1 || live[i]) {
                        acc[i][j] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(af[i], bf, acc[i][j]);
                    }
                }
            }
        }
    }

    __device__ __forceinline__ void run_super(Smem & sm, int K, float8_t (&acc)[FM][FN], int m_valid) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        const int wm   = warp / WARPS_N;
        const int wn   = warp % WARPS_N;
        const int l16  = lane & 15;
        const int brow = threadIdx.x >> 1;
        const int bchk = threadIdx.x & 1;
#pragma unroll
        for (int i = 0; i < FM; ++i) {
#pragma unroll
            for (int j = 0; j < FN; ++j) {
                acc[i][j] = float8_t{0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
            }
        }
        const int nk   = (K + BK - 1) / BK;
        const int nsup = (nk + SU - 1) / SU;
        loadA(0, K);
        if (NT == 2 * BN || brow < BN) {
            sr = Fmt::template sload_2l<SU>(s_rowq, s_rows, bchk, 0, K);
        }
#pragma unroll 1
        for (int sp = 0; sp < nsup; ++sp) {
#pragma unroll
            for (int t = 0; t < SU; ++t) {
                const int kt = sp * SU + t;
                __builtin_amdgcn_sched_barrier(0);
                storeA(sm);
                if (NT == 2 * BN || brow < BN) {
                    Fmt::template sstore_xor_2l<SU>(&sm.b[brow * LDK_XOR], brow & 7, bchk, sr, t);
                }
                __syncthreads();
                if (kt + 1 < nk) {
                    loadA((kt + 1) * BK, K);
                }
                if (t + 1 == SU && sp + 1 < nsup) {
                    if (NT == 2 * BN || brow < BN) {
                        sr = Fmt::template sload_2l<SU>(s_rowq, s_rows, bchk, (sp + 1) * SU * BK, K);
                    }
                }
                compute(sm, acc, wm, wn, l16, m_valid);
                __syncthreads();
            }
        }
    }

    __device__ __forceinline__ static int out_row(int i, int v) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        return (warp / WARPS_N) * WTM + i * 16 + 2 * v + (lane >> 4);
    }
    __device__ __forceinline__ static int out_col(int j) {
        const int warp = threadIdx.x >> 5;
        const int lane = threadIdx.x & 31;
        return (warp % WARPS_N) * WTN + j * 16 + (lane & 15);
    }
};

// ---------------------------------------------------------------------------
// Dense SoA Q8_0 GEMM: Y[M, N] = X[M, K] @ W[N, K]^T
// ---------------------------------------------------------------------------
// grid.x = M tiles (fastest: the M tiles sharing a W tile run together -> W streamed once),
// grid.y = N tiles.
template <int BM, int BN, int WARPS_M, int WARPS_N>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, 2)
dense_q8_wmma_kernel(const int8_t * __restrict__ qs, const __half * __restrict__ d,
                     const float * __restrict__ X, float * __restrict__ Y, int M, int N, int K) {
    typedef TileGemm<BM, BN, WARPS_M, WARPS_N, false, FmtQ8> G;
    __shared__ __align__(16) typename G::Smem sm;

    const int m0 = blockIdx.x * BM;
    const int n0 = blockIdx.y * BN;
    const int nb = K >> 5;

    G g;
#pragma unroll
    for (int p = 0; p < G::A_IT; ++p) {
        const int m = m0 + G::a_row(p);
        g.a_ptr[p] = (G::a_live(p) && m < M) ? X + (size_t) m * K : nullptr;
    }
#pragma unroll
    for (int p = 0; p < G::B_IT; ++p) {
        const int n = n0 + G::b_row(p);
        g.b_q[p] = (n < N) ? qs + (size_t) n * K : nullptr;
        g.b_s[p] = (n < N) ? d + (size_t) n * nb : nullptr;
    }

    float8_t acc[G::FM][G::FN];
    g.run(sm, K, acc);

#pragma unroll
    for (int i = 0; i < G::FM; ++i) {
#pragma unroll
        for (int j = 0; j < G::FN; ++j) {
            const int n = n0 + G::out_col(j);
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int m = m0 + G::out_row(i, v);
                if (m < M && n < N) {
                    Y[(size_t) m * N + n] = acc[i][j][v];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Fused 3-matrix QKV Dense SoA Q8_0 WMMA GEMM: Y_i[M, N_i] = X_i[M, K] @ W_i[N_i, K]^T
// ---------------------------------------------------------------------------
struct Qkv3WmmaSeg {
    const int8_t * qs;
    const __half * d;
    const float  * X;
    float        * Y;
    int            N;
    int            nt_end;
};

struct Qkv3WmmaArgs {
    Qkv3WmmaSeg s[3];
    int         M;
    int         K;
};

template <int BM, int BN, int WARPS_M, int WARPS_N>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, 2)
dense_q8_wmma_qkv3_kernel(const Qkv3WmmaArgs a) {
    typedef TileGemm<BM, BN, WARPS_M, WARPS_N, false, FmtQ8> G;
    __shared__ __align__(16) typename G::Smem sm;

    const int by  = blockIdx.y;
    const int si  = (by >= a.s[0].nt_end) + (by >= a.s[1].nt_end);
    const int nt0 = (si == 0) ? 0 : a.s[si - 1].nt_end;
    const Qkv3WmmaSeg sg = a.s[si];

    const int m0 = blockIdx.x * BM;
    const int n0 = (by - nt0) * BN;
    const int M  = a.M;
    const int N  = sg.N;
    const int K  = a.K;
    const int nb = K >> 5;

    G g;
#pragma unroll
    for (int p = 0; p < G::A_IT; ++p) {
        const int m = m0 + G::a_row(p);
        g.a_ptr[p] = (G::a_live(p) && m < M) ? sg.X + (size_t) m * K : nullptr;
    }
#pragma unroll
    for (int p = 0; p < G::B_IT; ++p) {
        const int n = n0 + G::b_row(p);
        g.b_q[p] = (n < N) ? sg.qs + (size_t) n * K : nullptr;
        g.b_s[p] = (n < N) ? sg.d + (size_t) n * nb : nullptr;
    }

    float8_t acc[G::FM][G::FN];
    g.run(sm, K, acc);

#pragma unroll
    for (int i = 0; i < G::FM; ++i) {
#pragma unroll
        for (int j = 0; j < G::FN; ++j) {
            const int n = n0 + G::out_col(j);
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int m = m0 + G::out_row(i, v);
                if (m < M && n < N) {
                    sg.Y[(size_t) m * N + n] = acc[i][j][v];
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// MoE routing (single block): counts, offsets, deterministic slots, tile worklist, tok_slots
// ---------------------------------------------------------------------------
constexpr int ROUTE_THREADS = 1024;
constexpr int MASK_WORDS    = MAX_UBATCH / 32; // 16

__global__ void __launch_bounds__(ROUTE_THREADS)
moe_route_kernel(const int32_t * __restrict__ topk_ids, const float * __restrict__ topk_w, int M, int TM,
                 int32_t * __restrict__ exp_counts, int32_t * __restrict__ exp_offsets,
                 int32_t * __restrict__ exp_cursors, int32_t * __restrict__ perm_tok,
                 float * __restrict__ perm_w, int32_t * __restrict__ tile_exp,
                 int32_t * __restrict__ tile_m0, int32_t * __restrict__ d_num_tiles,
                 int32_t * __restrict__ tok_slots) {
    __shared__ uint32_t s_mask[N_EXPERT][MASK_WORDS];
    __shared__ int32_t  s_pref[N_EXPERT][MASK_WORDS];
    __shared__ int32_t  s_cnt[N_EXPERT];
    __shared__ int32_t  s_off[N_EXPERT + 1];
    __shared__ int32_t  s_toff[N_EXPERT + 1];

    const int tid   = threadIdx.x;
    const int npair = M * TOP_K;
    const int nw    = (M + 31) >> 5;

    for (int i = tid; i < N_EXPERT * MASK_WORDS; i += ROUTE_THREADS) {
        (&s_mask[0][0])[i] = 0u;
    }
    __syncthreads();

    for (int p = tid; p < npair; p += ROUTE_THREADS) {
        const int e = topk_ids[p];
        if ((unsigned) e < (unsigned) N_EXPERT) {
            const int tok = p / TOP_K;
            atomicOr(&s_mask[e][tok >> 5], 1u << (tok & 31));
        }
    }
    __syncthreads();

    if (tid < N_EXPERT) {
        int run = 0;
        for (int w = 0; w < nw; ++w) {
            s_pref[tid][w] = run;
            run += __popc(s_mask[tid][w]);
        }
        s_cnt[tid] = run;
    }
    __syncthreads();

    if (tid == 0) {
        int off = 0, toff = 0;
        for (int e = 0; e < N_EXPERT; ++e) {
            s_off[e]  = off;
            s_toff[e] = toff;
            off  += s_cnt[e];
            toff += (s_cnt[e] + TM - 1) / TM;
        }
        s_off[N_EXPERT]  = off;
        s_toff[N_EXPERT] = toff;
        *d_num_tiles = toff;
    }
    __syncthreads();

    if (tid <= N_EXPERT) {
        exp_offsets[tid] = s_off[tid];
    }
    if (tid < N_EXPERT) {
        exp_counts[tid]  = s_cnt[tid];
        exp_cursors[tid] = s_cnt[tid]; // fill cursor after routing (all slots of the expert placed)
        const int nt = (s_cnt[tid] + TM - 1) / TM;
        for (int j = 0; j < nt; ++j) {
            tile_exp[s_toff[tid] + j] = tid;
            tile_m0[s_toff[tid] + j]  = j * TM; // M offset within the expert's slot segment
        }
    }

    for (int p = tid; p < npair; p += ROUTE_THREADS) {
        const int e = topk_ids[p];
        if ((unsigned) e >= (unsigned) N_EXPERT) {
            tok_slots[p] = -1;
            continue;
        }
        const int tok = p / TOP_K;
        const int k   = p - tok * TOP_K;
        bool first = true;
        float w = 0.0f;
        for (int kk = 0; kk < TOP_K; ++kk) {
            if (topk_ids[tok * TOP_K + kk] == e) {
                if (kk < k) {
                    first = false;
                }
                w += topk_w[tok * TOP_K + kk];
            }
        }
        if (!first) {
            tok_slots[p] = -1;
            continue;
        }
        const int wi   = tok >> 5;
        const int rank = s_pref[e][wi] + __popc(s_mask[e][wi] & ((1u << (tok & 31)) - 1u));
        const int slot = s_off[e] + rank;
        perm_tok[slot] = tok;
        perm_w[slot]   = w;
        tok_slots[p]   = slot;
    }
}

// ---------------------------------------------------------------------------
// Grouped stage 1: gate_up WMMA + scale2 + SITU -> perm_act
// ---------------------------------------------------------------------------
// Expert-contiguous grid: blockIdx.x = N-tile (0..N_FF_EXP/S1_OUT - 1), blockIdx.y = M-tile (0..tiles-1)
// so all 12 N-tiles of the same expert launch together while its A rows and scale2 are hot in cache.

template <class Fmt> struct ExpertArgs {
    const typename WPtr<Fmt>::Q * qs;
    const typename WPtr<Fmt>::S * sc;    // Q8: fp16 d; NVFP4: UE4M3 bytes
    const float *                 scale2; // NVFP4 only (nullptr for Q8)
};

template <class Fmt> __device__ __forceinline__ size_t w_row_bytes(int K);
template <> __device__ __forceinline__ size_t w_row_bytes<FmtQ8>(int K)    { return (size_t) K; }
template <> __device__ __forceinline__ size_t w_row_bytes<FmtNVFP4>(int K) { return (size_t) K / 2; }
template <class Fmt> __device__ __forceinline__ size_t w_row_scales(int K);
template <> __device__ __forceinline__ size_t w_row_scales<FmtQ8>(int K)    { return (size_t) K / 32; }
template <> __device__ __forceinline__ size_t w_row_scales<FmtNVFP4>(int K) { return (size_t) K / 16; }

template <int TM, int WARPS_M, int WARPS_N, int S1_OUT, class Fmt>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, 2)
moe_gate_up_wmma_kernel(const ExpertArgs<Fmt> W, const _Float16 * __restrict__ Xh,
                        const int32_t * __restrict__ exp_offsets, const int32_t * __restrict__ perm_tok,
                        const int32_t * __restrict__ tile_exp, const int32_t * __restrict__ tile_m0,
                        const int32_t * __restrict__ d_num_tiles, _Float16 * __restrict__ perm_act) {
    typedef TileGemmMoe<TM, 2 * S1_OUT, WARPS_M, WARPS_N, true, Fmt, Fmt::SUPER_GU> G;
    __shared__ __align__(16) typename G::Smem sm;

    const int tile = blockIdx.y;
    if (tile >= *d_num_tiles) {
        return; // uniform for the whole block
    }
    constexpr int K = N_EMBD;
    const int e     = tile_exp[tile];
    const int slot0 = exp_offsets[e] + tile_m0[tile];
    const int mrem  = exp_offsets[e + 1] - slot0;
    const int mcnt  = mrem < TM ? mrem : TM;
    const int n0    = blockIdx.x * S1_OUT;

    G g;
#pragma unroll
    for (int p = 0; p < G::A_IT; ++p) {
        const int i = G::a_row(p);
        g.a_ptr[p] = (G::a_live(p) && i < mcnt) ? Xh + (size_t) perm_tok[slot0 + i] * K : nullptr;
    }
    const int brow = threadIdx.x >> 1;
    if (G::NT == 4 * S1_OUT || brow < 2 * S1_OUT) {
        // B tile row brow: rows [0, S1_OUT) = gate rows n0.., rows [S1_OUT, 2 * S1_OUT) = matching up rows
        const int row = (brow < S1_OUT) ? (n0 + brow) : (N_FF_EXP + n0 + (brow - S1_OUT));
        const size_t grow = (size_t) e * N_GATE_UP_EXP + row;
        g.s_rowq = W.qs + grow * w_row_bytes<Fmt>(K);
        g.s_rows = W.sc + grow * w_row_scales<Fmt>(K);
    }

    float8_t acc[G::FM][G::FN];
    g.run_super(sm, K, acc, mcnt);

    const float s2 = (W.scale2 != nullptr) ? W.scale2[e] : 1.0f;
#pragma unroll
    for (int i = 0; i < G::FM; ++i) {
#pragma unroll
        for (int j = 0; j < G::FNO; ++j) {
            const int col = n0 + G::out_col(j);
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const int m = G::out_row(i, v);
                if (m < mcnt) {
                    const float gv = acc[i][j][v] * s2;
                    const float uv = acc[i][j + G::FNO][v] * s2;
                    // stored as FP16: stage 2 feeds it to FP16 WMMA, so this is the one RNE rounding
                    perm_act[(size_t) (slot0 + m) * N_FF_EXP + col] = (_Float16)
                        situ_scalar(gv, uv, SITU_BETA, 1.0f / SITU_BETA, SITU_LINEAR_BETA, 1.0f / SITU_LINEAR_BETA);
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Grouped stage 2: down WMMA * scale2 * perm_w -> perm_down
// ---------------------------------------------------------------------------
// Expert-contiguous grid: blockIdx.x = N-tile (0..N_EMBD/S2_BN - 1), blockIdx.y = M-tile (0..tiles-1).
template <int TM, int WARPS_M, int WARPS_N, int S2_BN, class Fmt>
__global__ void __launch_bounds__(WARPS_M * WARPS_N * 32, 2)
moe_down_wmma_kernel(const ExpertArgs<Fmt> W, const _Float16 * __restrict__ perm_act,
                     const int32_t * __restrict__ exp_offsets, const float * __restrict__ perm_w,
                     const int32_t * __restrict__ tile_exp, const int32_t * __restrict__ tile_m0,
                     const int32_t * __restrict__ d_num_tiles, float * __restrict__ perm_down) {
    typedef TileGemmMoe<TM, S2_BN, WARPS_M, WARPS_N, false, Fmt, Fmt::SUPER_DN> G;
    __shared__ __align__(16) typename G::Smem sm;

    const int tile = blockIdx.y;
    if (tile >= *d_num_tiles) {
        return;
    }
    constexpr int K = N_FF_EXP;
    const int e     = tile_exp[tile];
    const int slot0 = exp_offsets[e] + tile_m0[tile];
    const int mrem  = exp_offsets[e + 1] - slot0;
    const int mcnt  = mrem < TM ? mrem : TM;
    const int n0    = blockIdx.x * S2_BN;

    G g;
#pragma unroll
    for (int p = 0; p < G::A_IT; ++p) {
        const int i = G::a_row(p);
        g.a_ptr[p] = (G::a_live(p) && i < mcnt) ? perm_act + (size_t) (slot0 + i) * K : nullptr;
    }
    const int brow = threadIdx.x >> 1;
    if (G::NT == 2 * S2_BN || brow < S2_BN) {
        const size_t grow = (size_t) e * N_EMBD + n0 + brow;
        g.s_rowq = W.qs + grow * w_row_bytes<Fmt>(K);
        g.s_rows = W.sc + grow * w_row_scales<Fmt>(K);
    }

    float8_t acc[G::FM][G::FN];
    g.run_super(sm, K, acc, mcnt);

    const float s2 = (W.scale2 != nullptr) ? W.scale2[e] : 1.0f;
#pragma unroll
    for (int i = 0; i < G::FM; ++i) {
#pragma unroll
        for (int v = 0; v < 8; ++v) {
            const int m = G::out_row(i, v);
            if (m < mcnt) {
                const float wsc = s2 * perm_w[slot0 + m];
                float * dst = perm_down + (size_t) (slot0 + m) * N_EMBD + n0;
#pragma unroll
                for (int j = 0; j < G::FN; ++j) {
                    dst[G::out_col(j)] = acc[i][j][v] * wsc;
                }
            }
        }
    }
}

// ---------------------------------------------------------------------------
// X_norm [M, 2304] FP32 -> FP16 (RNE) once per MoE call; stage 1 gathers rows from it.
// ---------------------------------------------------------------------------
constexpr int CVT_THREADS = N_EMBD / 8; // 288: 8 elements (one 16-byte store) per thread

__global__ void __launch_bounds__(CVT_THREADS)
x_to_f16_kernel(const float * __restrict__ X, _Float16 * __restrict__ Xh) {
    const size_t base = (size_t) blockIdx.x * N_EMBD + threadIdx.x * 8;
    const float4 a = *reinterpret_cast<const float4 *>(X + base);
    const float4 b = *reinterpret_cast<const float4 *>(X + base + 4);
    const v4i o = v4i{(int) bits_of(__floats2half2_rn(a.x, a.y)), (int) bits_of(__floats2half2_rn(a.z, a.w)),
                      (int) bits_of(__floats2half2_rn(b.x, b.y)), (int) bits_of(__floats2half2_rn(b.z, b.w))};
    *reinterpret_cast<v4i *>(Xh + base) = o;
}

// ---------------------------------------------------------------------------
// Deterministic combine: Y[tok] = sum over its distinct experts (k order) of perm_down[slot]
// Uses precomputed tok_slots[tok * TOP_K + k] from moe_route_kernel (zero binary search).
// ---------------------------------------------------------------------------
constexpr int COMBINE_THREADS = 192; // 576 float4 per row -> 3 per thread

__global__ void __launch_bounds__(COMBINE_THREADS)
moe_combine_kernel(const int32_t * __restrict__ tok_slots, const float * __restrict__ perm_down,
                   float * __restrict__ Y) {
    const int tok = blockIdx.x;
    const int32_t * ts = tok_slots + tok * TOP_K;
    const int s0 = ts[0], s1 = ts[1], s2 = ts[2], s3 = ts[3];
    const int s4 = ts[4], s5 = ts[5], s6 = ts[6], s7 = ts[7];
    const int s[TOP_K] = {s0, s1, s2, s3, s4, s5, s6, s7};

    float4 * y4 = reinterpret_cast<float4 *>(Y + (size_t) tok * N_EMBD);
    for (int c = threadIdx.x; c < N_EMBD / 4; c += COMBINE_THREADS) {
        float4 acc = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
        for (int k = 0; k < TOP_K; ++k) {
            const int slot = s[k];
            if (slot >= 0) {
                const float4 v = reinterpret_cast<const float4 *>(perm_down + (size_t) slot * N_EMBD)[c];
                acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
            }
        }
        y4[c] = acc;
    }
}

// ---------------------------------------------------------------------------
// Fused deterministic MoE combine + causal 4-tap static shortconv + residual add
// Matches sconv_post_res_kernel / sconv_post_res_one bit-for-bit within FP32 FMA.
// Vectorized 128-bit float4 per thread (4 channels/thread -> 256 channels/block = 1 KB burst/slot):
//   blockIdx.x = channel blocks (9 blocks * 64 threads * 4 ch = 2304 channels),
//   blockIdx.y = time chunk (T_CHUNK = 8 tokens/chunk -> 576 blocks = 12 warps/CU at M = 512).
// Race-free history update: ONLY t_blk == 0 reads and updates sconv_hist (for n_tchunks > 1,
// t_blk == 0 computes the final 3 tokens M-3..M-1 after its loop and writes them to sconv_hist).
// ---------------------------------------------------------------------------
constexpr int COMBINE_SCONV_THREADS = 64;
constexpr int COMBINE_SCONV_BLOCKS  = (N_EMBD / 4) / COMBINE_SCONV_THREADS; // 9
constexpr int COMBINE_SCONV_TCHUNK  = 8;

__device__ __forceinline__ float4 load_tok_sum_c4(
        const int32_t * __restrict__ tok_slots, const float * __restrict__ perm_down, int tok, int c4) {
    const int4 ts0 = reinterpret_cast<const int4 *>(tok_slots + tok * TOP_K)[0];
    const int4 ts1 = reinterpret_cast<const int4 *>(tok_slots + tok * TOP_K)[1];
    const int s[TOP_K] = {ts0.x, ts0.y, ts0.z, ts0.w, ts1.x, ts1.y, ts1.z, ts1.w};
    const float4 * p4 = reinterpret_cast<const float4 *>(perm_down);
    float4 x = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma unroll
    for (int k = 0; k < TOP_K; ++k) {
        const int slot = s[k];
        if (slot >= 0) {
            const float4 v = p4[(size_t) slot * (N_EMBD / 4) + c4];
            x.x += v.x; x.y += v.y; x.z += v.z; x.w += v.w;
        }
    }
    return x;
}

__device__ __forceinline__ float sconv_step_one(float x, float & h0, float & h1, float & h2,
                                                float4 w, bool has_res, float res_val) {
    constexpr float kScale = 0.44721359549995793928f; // 1 / sqrt(5)
    const float acc = x + h0 * w.x + h1 * w.y + h2 * w.z + x * w.w;
    h0 = h1;
    h1 = h2;
    h2 = x;
    return has_res ? fmaf(acc, kScale, res_val) : (acc * kScale);
}

__global__ void __launch_bounds__(COMBINE_SCONV_THREADS)
moe_combine_sconv_kernel(const int32_t * __restrict__ tok_slots, const float * __restrict__ perm_down,
                         const float * __restrict__ conv_w, float * __restrict__ sconv_hist,
                         const float * __restrict__ res_add, float * __restrict__ Y, int M) {
    const int c4    = blockIdx.x * COMBINE_SCONV_THREADS + threadIdx.x;
    const int c0    = c4 * 4;
    const int t_blk = blockIdx.y;
    const int t0    = t_blk * COMBINE_SCONV_TCHUNK;
    if (t0 >= M) {
        return;
    }
    const int t1 = (t0 + COMBINE_SCONV_TCHUNK < M) ? (t0 + COMBINE_SCONV_TCHUNK) : M;

    const float4 * cw4 = reinterpret_cast<const float4 *>(conv_w);
    const float4 w0 = cw4[c0 + 0];
    const float4 w1 = cw4[c0 + 1];
    const float4 w2 = cw4[c0 + 2];
    const float4 w3 = cw4[c0 + 3];

    float * h_base = sconv_hist + c0 * 3;
    float4 h0, h1, h2;
    if (t_blk == 0) {
        h0 = make_float4(h_base[0], h_base[3], h_base[6], h_base[9]);
        h1 = make_float4(h_base[1], h_base[4], h_base[7], h_base[10]);
        h2 = make_float4(h_base[2], h_base[5], h_base[8], h_base[11]);
    } else {
        // t0 >= 8 >= 3, so t0-3, t0-2, t0-1 are strictly in [0, M-1]
        h0 = load_tok_sum_c4(tok_slots, perm_down, t0 - 3, c4);
        h1 = load_tok_sum_c4(tok_slots, perm_down, t0 - 2, c4);
        h2 = load_tok_sum_c4(tok_slots, perm_down, t0 - 1, c4);
    }

    const bool has_res = (res_add != nullptr);
    const float4 * r4_ptr = reinterpret_cast<const float4 *>(res_add);
    float4 * y4_ptr       = reinterpret_cast<float4 *>(Y);

    for (int t = t0; t < t1; ++t) {
        const float4 x = load_tok_sum_c4(tok_slots, perm_down, t, c4);
        const float4 r = has_res ? r4_ptr[(size_t) t * (N_EMBD / 4) + c4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        float4 out;
        out.x = sconv_step_one(x.x, h0.x, h1.x, h2.x, w0, has_res, r.x);
        out.y = sconv_step_one(x.y, h0.y, h1.y, h2.y, w1, has_res, r.y);
        out.z = sconv_step_one(x.z, h0.z, h1.z, h2.z, w2, has_res, r.z);
        out.w = sconv_step_one(x.w, h0.w, h1.w, h2.w, w3, has_res, r.w);
        y4_ptr[(size_t) t * (N_EMBD / 4) + c4] = out;
    }

    if (t_blk == 0) {
        if (t1 < M) {
            h0 = load_tok_sum_c4(tok_slots, perm_down, M - 3, c4);
            h1 = load_tok_sum_c4(tok_slots, perm_down, M - 2, c4);
            h2 = load_tok_sum_c4(tok_slots, perm_down, M - 1, c4);
        }
        h_base[0]  = h0.x; h_base[1]  = h1.x; h_base[2]  = h2.x;
        h_base[3]  = h0.y; h_base[4]  = h1.y; h_base[5]  = h2.y;
        h_base[6]  = h0.z; h_base[7]  = h1.z; h_base[8]  = h2.z;
        h_base[9]  = h0.w; h_base[10] = h1.w; h_base[11] = h2.w;
    }
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------
int device_cu_count() {
    static int cache[64] = {};
    int dev = 0;
    if (hipGetDevice(&dev) != hipSuccess || dev < 0 || dev >= 64) {
        return 96;
    }
    if (cache[dev] == 0) {
        int n = 0;
        if (hipDeviceGetAttribute(&n, hipDeviceAttributeMultiprocessorCount, dev) != hipSuccess || n <= 0) {
            n = 96;
        }
        cache[dev] = n;
    }
    return cache[dev];
}

[[noreturn]] void wmma_contract_fail(const char * what) {
    std::fprintf(stderr, "[inkling kernels_wmma_prefill] contract violation: %s\n", what);
    std::abort();
}

template <int BM, int BN, int WM, int WN>
void launch_dense_cfg(const SoaQ8Matrix & W, const float * X, float * Y, int M, hipStream_t stream) {
    const dim3 grid((M + BM - 1) / BM, (W.rows + BN - 1) / BN);
    dense_q8_wmma_kernel<BM, BN, WM, WN><<<grid, WM * WN * 32, 0, stream>>>(W.qs, W.d, X, Y, M, W.rows, W.K);
}

template <int BM, int BN, int WM, int WN>
void launch_qkv3_cfg(Qkv3WmmaArgs a, int nseg, hipStream_t stream) {
    int nt = 0;
    for (int i = 0; i < 3; ++i) {
        if (i < nseg) {
            nt += (a.s[i].N + BN - 1) / BN;
            a.s[i].nt_end = nt;
        } else {
            a.s[i].nt_end = 0x7FFFFFFF;
        }
    }
    const dim3 grid((a.M + BM - 1) / BM, nt);
    dense_q8_wmma_qkv3_kernel<BM, BN, WM, WN><<<grid, WM * WN * 32, 0, stream>>>(a);
}

inline long long n_blocks(int M, int N, int BM, int BN) {
    return (long long) ((M + BM - 1) / BM) * (long long) ((N + BN - 1) / BN);
}

// Worst-case number of M tiles: sum_e ceil(c_e / TM) <= floor(P / TM) + min(P, N_EXPERT)
inline int max_tiles(int M, int TM) {
    const int P = M * TOP_K;
    return P / TM + (P < N_EXPERT ? P : N_EXPERT);
}

// Grouped MoE launch configuration (tile height shared by both stages: one tile worklist).
template <int TM_, int GWM_, int GWN_, int GOUT_, int DWM_, int DWN_, int DBN_>
struct MoeCfg {
    static constexpr int TM = TM_;
    static constexpr int GWM = GWM_, GWN = GWN_, GOUT = GOUT_; // gate_up: warps, act cols
    static constexpr int DWM = DWM_, DWN = DWN_, DBN  = DBN_;  // down:    warps, out cols
};

template <class C, class Fmt>
void launch_moe_grouped(const ExpertArgs<Fmt> & gu, const ExpertArgs<Fmt> & dn, const float * X_norm,
                        const int32_t * topk_ids, const float * topk_w, MoePrefillScratch & s,
                        float * Y_out, int M, hipStream_t stream,
                        const float * post_conv_w, float * post_sconv_h, const float * post_res_add) {
    // Scratch reuse (internal to these kernels):
    //   - s.perm_x (allocated as [MAX_UBATCH * TOP_K, N_EMBD] FP32 = 36 MiB) holds X_norm as FP16
    //     [M, 2304] in its first 2.25 MiB, and tok_slots [M * TOP_K] int32_t right after MAX_UBATCH * N_EMBD halves.
    //   - s.perm_act holds post-SITU activations as FP16 [slots, 1152].
    _Float16 * xh       = reinterpret_cast<_Float16 *>(s.perm_x);
    int32_t * tok_slots = reinterpret_cast<int32_t *>(xh + (size_t) MAX_UBATCH * N_EMBD);
    _Float16 * acth     = reinterpret_cast<_Float16 *>(s.perm_act);

    moe_route_kernel<<<1, ROUTE_THREADS, 0, stream>>>(topk_ids, topk_w, M, C::TM, s.exp_counts, s.exp_offsets,
                                                      s.exp_cursors, s.perm_tok, s.perm_w, s.tile_exp,
                                                      s.tile_m0, s.d_num_tiles, tok_slots);
    x_to_f16_kernel<<<M, CVT_THREADS, 0, stream>>>(X_norm, xh);
    const int tiles = max_tiles(M, C::TM);
    moe_gate_up_wmma_kernel<C::TM, C::GWM, C::GWN, C::GOUT, Fmt>
        <<<dim3(N_FF_EXP / C::GOUT, tiles), C::GWM * C::GWN * 32, 0, stream>>>(
        gu, xh, s.exp_offsets, s.perm_tok, s.tile_exp, s.tile_m0, s.d_num_tiles, acth);
    moe_down_wmma_kernel<C::TM, C::DWM, C::DWN, C::DBN, Fmt>
        <<<dim3(N_EMBD / C::DBN, tiles), C::DWM * C::DWN * 32, 0, stream>>>(
        dn, acth, s.exp_offsets, s.perm_w, s.tile_exp, s.tile_m0, s.d_num_tiles, s.perm_down);
    if (post_conv_w != nullptr) {
        if (post_sconv_h == nullptr) {
            wmma_contract_fail("launch_moe_*_prefill_wmma: post_sconv_h must be non-null when post_conv_w is provided");
        }
        const int n_tchunks = (M + COMBINE_SCONV_TCHUNK - 1) / COMBINE_SCONV_TCHUNK;
        moe_combine_sconv_kernel<<<dim3(COMBINE_SCONV_BLOCKS, n_tchunks), COMBINE_SCONV_THREADS, 0, stream>>>(
            tok_slots, s.perm_down, post_conv_w, post_sconv_h, post_res_add, Y_out, M);
    } else {
        moe_combine_kernel<<<M, COMBINE_THREADS, 0, stream>>>(tok_slots, s.perm_down, Y_out);
    }
}

// Tile configuration on RX 7900 XTX (96 CUs / 48 WGPs, 64 KB LDS/CU):
//   M > 256 : TM = 64 (holds skewed experts <= 64 tokens in 1 M-tile -> zero duplicate weight reads),
//             8 warps (4 x 2 = 256 threads = 2 * 128 -> 100% thread utilization during B load/dequant),
//             24,576 B LDS ((64 + 128) * 128 B = 24 KB -> 2 blocks/CU):
//             - gate_up: S1_OUT = 64 (BN = 128, 18 N-tiles)
//             - down   : S2_BN  = 128 (18 N-tiles)
//   M <= 256: TM = 32, 8 warps (2 x 4 = 256 threads = 2 * 128), gate_up S1_OUT = 64 (BN = 128),
//             down S2_BN = 128 (20,480 B LDS <= 32 KB -> 3 blocks/CU, dead wm warps skip compute).
template <class Fmt>
void launch_moe_dispatch(const ExpertArgs<Fmt> & gu, const ExpertArgs<Fmt> & dn, const float * X_norm,
                         const int32_t * topk_ids, const float * topk_w, MoePrefillScratch & s,
                         float * Y_out, int M, hipStream_t stream,
                         const float * post_conv_w, float * post_sconv_h, const float * post_res_add) {
    if (M > 256) {
        launch_moe_grouped<MoeCfg<64, 4, 2, 64, 4, 2, 128>, Fmt>(
            gu, dn, X_norm, topk_ids, topk_w, s, Y_out, M, stream, post_conv_w, post_sconv_h, post_res_add);
    } else {
        launch_moe_grouped<MoeCfg<32, 2, 4, 64, 2, 4, 128>, Fmt>(
            gu, dn, X_norm, topk_ids, topk_w, s, Y_out, M, stream, post_conv_w, post_sconv_h, post_res_add);
    }
}

void check_scratch(const MoePrefillScratch & s) {
    if (!s.exp_counts || !s.exp_offsets || !s.exp_cursors || !s.perm_tok || !s.perm_w || !s.perm_x || !s.perm_act ||
        !s.perm_down || !s.tile_exp || !s.tile_m0 || !s.d_num_tiles) {
        wmma_contract_fail("MoePrefillScratch has a null buffer");
    }
}

} // namespace

// ============================================================================
// Public API
// ============================================================================
void launch_soa_q8_gemm_wmma(const SoaQ8Matrix & W, const float * X, float * Y, int M, hipStream_t stream) {
    if (M <= 0 || W.rows <= 0) {
        return;
    }
    if (W.K <= 0 || (W.K % 32) != 0) {
        wmma_contract_fail("SoaQ8Matrix.K must be a positive multiple of 32");
    }
    if (M == 1) {
        launch_soa_q8_gemv(W, X, Y, stream);
        return;
    }
    const int N = W.rows;
    // HIP reports 48 multiprocessors (WGPs) on gfx1100 -> target = 96 blocks (one per CU).
    // With XOR-swizzled LDK_XOR = 64 and single-bf streaming:
    //   - <64, 128, 4, 2> uses 24,576 B LDS (2 blocks/CU = 16 warps/CU) and 124 VGPRs (10 waves/SIMD),
    //     delivering 82+ TFLOPS cold at M = 512 (1.91x faster than 16-warp <128, 128, 4, 4> on proj_qkv_all
    //     and 2.05x faster on wo).
    //   - <128, 128, 4, 4> uses 32,768 B LDS (exact 32 KB -> 2 blocks/CU) and 124 VGPRs for M > 512.
    const long long target = 2LL * device_cu_count();
    if (M > 512 && n_blocks(M, N, 128, 128) >= 2 * target) {
        launch_dense_cfg<128, 128, 4, 4>(W, X, Y, M, stream);
    } else if (M > 64 && n_blocks(M, N, 64, 128) >= target) {
        launch_dense_cfg<64, 128, 4, 2>(W, X, Y, M, stream);
    } else if (M > 64 && n_blocks(M, N, 64, 64) >= target) {
        launch_dense_cfg<64, 64, 2, 2>(W, X, Y, M, stream);
    } else if (M > 64) {
        launch_dense_cfg<32, 64, 2, 2>(W, X, Y, M, stream);
    } else if (n_blocks(M, N, 16, 64) >= target) {
        launch_dense_cfg<16, 64, 1, 4>(W, X, Y, M, stream);
    } else {
        launch_dense_cfg<16, 32, 1, 2>(W, X, Y, M, stream);
    }
}

void launch_soa_q8_gemm_wmma_qkv3(const SoaQ8Matrix & W0, const float * X0, float * Y0,
                                  const SoaQ8Matrix & W1, const float * X1, float * Y1,
                                  const SoaQ8Matrix & W2, const float * X2, float * Y2,
                                  int M, hipStream_t stream) {
    if (W0.K != W1.K || W0.K != W2.K) {
        wmma_contract_fail("launch_soa_q8_gemm_wmma_qkv3: all three matrices must share K");
    }
    if (M <= 0) {
        return;
    }
    if (M == 1) {
        launch_soa_q8_gemv_qkv3(W0, X0, Y0, W1, X1, Y1, W2, X2, Y2, stream);
        return;
    }
    const SoaQ8Matrix * Ws[3] = {&W0, &W1, &W2};
    const float * Xs[3] = {X0, X1, X2};
    float * Ys[3] = {Y0, Y1, Y2};
    Qkv3WmmaArgs a = {};
    a.M = M;
    a.K = W0.K;
    int nseg = 0;
    int total_N = 0;
    for (int i = 0; i < 3; ++i) {
        if (Ws[i]->rows > 0) {
            if (Ws[i]->K <= 0 || (Ws[i]->K % 32) != 0) {
                wmma_contract_fail("SoaQ8Matrix.K must be a positive multiple of 32");
            }
            a.s[nseg].qs = Ws[i]->qs;
            a.s[nseg].d  = Ws[i]->d;
            a.s[nseg].X  = Xs[i];
            a.s[nseg].Y  = Ys[i];
            a.s[nseg].N  = Ws[i]->rows;
            a.s[nseg].nt_end = 0;
            total_N += Ws[i]->rows;
            ++nseg;
        }
    }
    if (nseg == 0) {
        return;
    }
    const long long target = 2LL * device_cu_count();
    if (M > 512 && n_blocks(M, total_N, 128, 128) >= 2 * target) {
        launch_qkv3_cfg<128, 128, 4, 4>(a, nseg, stream);
    } else if (M > 64 && n_blocks(M, total_N, 64, 128) >= target) {
        launch_qkv3_cfg<64, 128, 4, 2>(a, nseg, stream);
    } else if (M > 64 && n_blocks(M, total_N, 64, 64) >= target) {
        launch_qkv3_cfg<64, 64, 2, 2>(a, nseg, stream);
    } else if (M > 64) {
        launch_qkv3_cfg<32, 64, 2, 2>(a, nseg, stream);
    } else if (n_blocks(M, total_N, 16, 64) >= target) {
        launch_qkv3_cfg<16, 64, 1, 4>(a, nseg, stream);
    } else {
        launch_qkv3_cfg<16, 32, 1, 2>(a, nseg, stream);
    }
}

void launch_moe_nvfp4_prefill_wmma(const SoaNVFP4Experts & gate_up, const SoaNVFP4Experts & down,
                                   const float * X_norm, const int32_t * topk_ids, const float * topk_w,
                                   MoePrefillScratch & scratch, float * Y_out, int M, hipStream_t stream,
                                   const float * post_conv_w, float * post_sconv_h, const float * post_res_add) {
    if (M <= 0) {
        return;
    }
    if (gate_up.rows != N_GATE_UP_EXP || gate_up.K != N_EMBD || down.rows != N_EMBD || down.K != N_FF_EXP) {
        wmma_contract_fail("NVFP4 experts must be gate_up [*, 2304, 2304] and down [*, 2304, 1152]");
    }
    check_scratch(scratch);
    const ExpertArgs<FmtNVFP4> gu{gate_up.qs, gate_up.scales, gate_up.scale2};
    const ExpertArgs<FmtNVFP4> dn{down.qs, down.scales, down.scale2};
    // The scratch holds MAX_UBATCH tokens: longer batches run in MAX_UBATCH chunks (stream ordered).
    for (int t0 = 0; t0 < M; t0 += MAX_UBATCH) {
        const int mc = (M - t0) < MAX_UBATCH ? (M - t0) : MAX_UBATCH;
        const float * chunk_res = (post_res_add != nullptr) ? (post_res_add + (size_t) t0 * N_EMBD) : nullptr;
        launch_moe_dispatch<FmtNVFP4>(gu, dn, X_norm + (size_t) t0 * N_EMBD, topk_ids + (size_t) t0 * TOP_K,
                                      topk_w + (size_t) t0 * TOP_K, scratch, Y_out + (size_t) t0 * N_EMBD, mc, stream,
                                      post_conv_w, post_sconv_h, chunk_res);
    }
}

void launch_moe_q8_prefill_wmma(const SoaQ8Experts & gate_up, const SoaQ8Experts & down,
                                const float * X_norm, const int32_t * topk_ids, const float * topk_w,
                                MoePrefillScratch & scratch, float * Y_out, int M, hipStream_t stream,
                                const float * post_conv_w, float * post_sconv_h, const float * post_res_add) {
    if (M <= 0) {
        return;
    }
    if (gate_up.rows != N_GATE_UP_EXP || gate_up.K != N_EMBD || down.rows != N_EMBD || down.K != N_FF_EXP) {
        wmma_contract_fail("Q8 experts must be gate_up [*, 2304, 2304] and down [*, 2304, 1152]");
    }
    check_scratch(scratch);
    const ExpertArgs<FmtQ8> gu{gate_up.qs, gate_up.d, nullptr};
    const ExpertArgs<FmtQ8> dn{down.qs, down.d, nullptr};
    for (int t0 = 0; t0 < M; t0 += MAX_UBATCH) {
        const int mc = (M - t0) < MAX_UBATCH ? (M - t0) : MAX_UBATCH;
        const float * chunk_res = (post_res_add != nullptr) ? (post_res_add + (size_t) t0 * N_EMBD) : nullptr;
        launch_moe_dispatch<FmtQ8>(gu, dn, X_norm + (size_t) t0 * N_EMBD, topk_ids + (size_t) t0 * TOP_K,
                                   topk_w + (size_t) t0 * TOP_K, scratch, Y_out + (size_t) t0 * N_EMBD, mc, stream,
                                   post_conv_w, post_sconv_h, chunk_res);
    }
}

} // namespace inkling
