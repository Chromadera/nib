# nib on RDNA4 — gfx1201 (Radeon AI PRO R9700) port

Status: **builds and is believed correct; not yet runtime-validated on silicon.**
This directory documents the `messerli-wallace/rdna4` branch: what was changed to
make the engine compile and produce *correct* math on gfx1200/gfx1201, what is
verified statically, what is still slow, and how to run the first real test.

Target: **AMD Radeon AI PRO R9700** — `gfx1201` / Navi 48, **64 CU** (32 WGP),
32 GB GDDR6 @ 640 GB/s, 64 MB Infinity Cache. ROCm 7.2.0 in WSL2 on Windows 11
(Adrenalin 26.2.2, `/dev/dxg`, ROCDXG), Ubuntu 24.04.

## Why it did not build, and what was done about it

| Blocker (see `../nib-gfx1201-assessment.md`) | Fix in this branch |
|---|---|
| 17 `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32` sites (RDNA3-only, `wmma-256b-insts`) | `include/nib_mma.h` supplies a layout-independent 16x16x16 tile (`v_dot2_f32_f16`, FP32 accumulate). gfx11 keeps the native builtin, untouched. |
| Prefill FA compiled to `(void)q;` on non-gfx11 (silent garbage) | `kernels_fa_relbias.hip` and `kernels_sparse_global.hip` now compute K·Qᵀ and P·Vᵀ for real on gfx12; P is rebuilt in ascending-key order from the two half-waves instead of the RDNA3-packed P fragment. |
| `kernels_core.hip` ctrl_down / ctrl_apply / router prefill had empty gfx12 bodies | All three are real kernels on gfx12 now. |
| 96-block cooperative kernels deadlock on 64 CU | CU-aware co-residency gates: `decode_fused_prime_occupancy()` / `s7i_a_v3_coresident()`. Where a grid cannot be fully resident, the launchers fall back to the multi-launch composition (same math, no cross-block barrier). gfx11 short-circuits to the old unconditional fused launch. |
| gfx11-only cache-flush asm (`s_waitcnt_vscnt`, `buffer_gl0_inv/gl1_inv`) | Verified absent on gfx12 (they do not assemble) and that the scoped fences the kernels already use emit the RDNA4 equivalents: `fence(RELEASE, AGENT)` → `s_waitcnt` + `global_wb scope:SCOPE_DEV`, acquire-scope atomics → `global_inv scope:SCOPE_DEV`. No hand-written cache asm is needed; documented at all 13 helper copies. |
| gfx12 has no bare `s_barrier` | `nib_lds_fence_barrier()` in `include/nib_target.h` (identical asm on gfx11, `s_waitcnt 0` + `__builtin_amdgcn_s_barrier()` → `s_barrier_signal/wait` on gfx12). |
| Split-K / decode-split heuristics claimed 96 CU | Grids are 64 CTAs × split, so 384/192/256 are exact multiples of 64 too; only the comments were wrong. Re-labelled. |

## The tile primitive (`include/nib_mma.h`)

The native RDNA3 wave32 16x16x16 f16→f32 fragment layout (documented and unit
tested in `kernels_wmma_prefill.hip`) is

```
A (M x K): lane l holds row (l % 16) as 16 contiguous K halves; lanes 16..31 replicate
B (K x N): lane l holds row (l % 16) of the row-major [n][k] tile, 16 contiguous K halves
D        : lane l, element i -> row (2*i + l/16), column (l % 16)
```

so one tile needs **no cross-lane traffic**: each lane reads the 8 A rows it owns
as D rows (`m = 2*i + l/16`) from the *same shared-memory tile* the RDNA3 path
stages, and reuses the B row it already holds. `dot16()` is 8 `v_dot2_f32_f16`
(FP16 inputs, FP32 accumulation — the tensor-core operand semantics), so the
numerics differ from RDNA3 WMMA only by ~1e-5 relative, which the upstream header
already documents for the native path.

Because it consumes rows rather than fragments, no LDS staging, XOR swizzle or
lane→element mapping was re-derived, and every epilogue (online softmax, the
in-register FWHT-128, the coalesced stores) is unchanged.

## Verified statically (no GPU run)

* `cmake -DGPU_TARGET=gfx1201` builds clean (ROCm 7.2.0, clang 22), and so does
  `-DGPU_TARGET=gfx1100`.
* **gfx1100 is unchanged**: all 18 kernel objects of the gfx1100 build are
  instruction-for-instruction identical to upstream `93507de` (v0.1.2, built with
  the same toolchain), compared object by object with `llvm-objdump -d`.
* Every ported kernel has a real body on gfx1201 with the expected amount of
  `v_dot2_f32_f16` and zero `v_wmma`:
  FA prefill 3072, ctrl_apply 1024, ctrl_down 128, router 256, dense GEMM 1024,
  MoE gate/up 4096, Quest scorer 4096, vision FA 2048, patch-embed 64.
* `rocminfo` reports `gfx1201`, 64 compute units, `AMD Radeon AI PRO R9700`.
* gfx12 WMMA exists and assembles: `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`
  emits `v_wmma_f32_16x16x16_f16 v[0:7], v[8:11], v[8:11], 0` (8-half fragments).
  That is the fast path for a follow-up commit; it is deliberately **not** used
  yet because its fragment/accumulator layout cannot be validated in this
  session (no GPU runs allowed).

## What is slow (expected)

* `spill_audit.sh GPU_TARGET=gfx1201` fails with VGPR spills in the emulated
  kernels — the tile math needs more live registers than the WMMA fragments:
  `kernels_fa.o` 1600 spills total over its 6 prefill instantiations (~240–290
  each, 920 B private), `kernels_core.o` 493 (ctrl_apply prefill, 1976 B
  private), `kernels_wmma.o` 278 (dense GEMM 65, MoE 32–40), `kernels_vision.o`
  0, `kernels_sparse_global.o` 2 (pre-existing). This is the price of "correct
  first"; register budgets and LDS blocking need a re-tune pass.
* Prefill GEMM/attention throughput will be far below the RDNA3 numbers
  (wmma → 64 dot2 + 16 LDS loads per tile per lane). Decode should be much less
  affected: the decode path (GEMV / FA decode v2-v4 / MoE decode) uses no WMMA.
* The A step of decode (dual mix + ctrl QKV + norm) runs as 2–3 launches on
  64 CU instead of 1, and MTP verify rows run one launch per row.

## First runtime test

```bash
cd ~/nib
# model pack (~17.3 GiB) in models/:
#   Inkling-2-Mini-NVFP4-Mixed.gguf, mtp.safetensors, mmproj-Inkling-2-Mini-Q8_0.gguf
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGPU_TARGET=gfx1201
cmake --build build -j 8

# 0. static checks, no GPU kernels involved (safe next to a busy GPU)
./scripts/verify_rdna4.sh

# 1. does the loader + repack + one decode step work at all
./build/nib selftest -m models/Inkling-2-Mini-NVFP4-Mixed.gguf

# 2. decode-only generation (exercises the GEMV/MoE/FA-decode path, ~50 tokens)
./build/nib gen -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
    --prompt-file tests/prompts/prose.txt -n 32 -c 4096

# 3. PREFILL gate: a ~2k-token prompt pushes every token through the gfx12 tile
#    math (dense GEMM, MoE grouped GEMM, prefill FA). If the tile emulation were
#    wrong the continuation would be obvious garbage / repeated tokens.
for i in $(seq 40); do cat tests/prompts/prose.txt; done > /tmp/long_prompt.txt
./build/nib gen -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
    --prompt-file /tmp/long_prompt.txt -n 32 -c 8192

# 4. optional: MTP drafter path (verify-batch exercises the m-row fallbacks)
./build/nib selftest -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
    --mtp models/mtp.safetensors --bench-mtp
```

There is no `--spill` flag in v0.1.2; spills are audited with
`GPU_TARGET=gfx1201 ./scripts/spill_audit.sh`.

Sanity signals to watch:
* `[step7i] occupancy blocks/WGP (wgp=64): ... a_v3=...` — on the R9700 the A
  step should report the fallback path (`occ * 64 < 96`), not the fused launch.
* `NIB_DUMP_OCC=1` prints the decode_fused occupancy cache at init.
* Prefill logits/perplexity must match the RDNA3 numbers (`PPL 3.5788` on the
  reference pack); if prefill is wrong the tile emulation is wrong, not the model.

## Honest limitations

* Nothing here has been executed on the GPU by the author of this port; all
  correctness claims are by construction (layout argument above) plus static
  verification. A `ppl`/`oracle` diff against a gfx1100 run is the real gate.
* The gfx12 WMMA fast path is not implemented — only its existence is verified.
* `kernels_step7j_fdot2.hip`'s opt-in 192/384-block MoE fdot2 kernels and
  `kernels_step7h_fused.hip`'s v3 kernels are cooperative and sized for 96 CU;
  they are off by default (`fdot2_moe = 0`, `b_v3 = 0`, `c_v3 = 0`) and would
  need the same gate before they are safe on 64 CU.
