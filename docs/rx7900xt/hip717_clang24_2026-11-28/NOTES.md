# RX 7900 XT — v0.3.0-dev on ROCm core 10.2 / HIP 7.17 (native Linux), spill fix — 2026-11-28

Toolchain: native Linux (Ubuntu, no WSL2 / no ROCDXG), `/dev/kfd` accessed directly.
GPU: Sapphire Nitro+ RX 7900 XT, `gfx1100`, 20 GB.
ROCm: **core 10.2** (`/opt/rocm/core-10.2`, `amdrocm-core10.2-gfx1100 10.2.0-2`),
`hipcc` reports **HIP 7.17.26434-0000000** + **AMD clang 24.0.0git**
(llvm-project `bdc35841c97f`), HSA runtime 1.21.
Build: `-DGPU_TARGET=gfx1100`, `nib` at `d0f4517` (branch `optimize-for-new-toolchain`,
parent `f6b3694` = v0.3.0 version bump).
Model: `Inkling-2-Mini-NVFP4-23.gguf` (16,099,143,008 B) + `mtp.safetensors`
+ `mmproj-Inkling-2-Mini-Q8_0.gguf`.
Server flags as deployed: `nib serve --mtp --mmproj --slots 2 -c 132768 --kv-type q4_1
--sparse-pages 224` (the `run.sh` config now carries `262848`; every server number below
was measured on the running `-c 132768` instance, so single-request context maxes at
~115k there — see the v0.2.1 folder for the 262,848 ladder).

## Findings (this toolchain)

1. **`v0.2.1` source does not pass `scripts/spill_audit.sh` here.** `kernels_sparse_global.o`
   reports 6 instantiations of `inkling_fa_prefill_gqa_wmma_sparse_global_kernel`, each at
   `maxVGPR=256` with 40-44 `vgprSpills` (252 for the object) and `maxPriv=96` bytes per
   work-item. Every other object in the tree, and every decode kernel in this object, is
   clean. Same source passes 0/0/0/0 on ROCm 7.2.4 / HIP-Clang 22
   ([`../v0.2.1_2026-11-27/NOTES.md`](../v0.2.1_2026-11-27/NOTES.md) finding 3) and
   `git diff v0.2.1..f6b3694 -- src/kernels_sparse_global.hip` is empty, so this is
   **toolchain drift, not a source regression**.
2. **Fix: cap the RelBias unroll.** Both relative-bias nests (`NK16 × 8` = 32 iterations)
   were fully unrolled, which lets the compiler keep all 32 bias rows live (512 loads on
   the `r_prep` path) and sink them into the O-accumulation WMMA phase, where `O_acc`
   (64 VGPRs) + `S` (32) + KV fragments already own the 256-VGPR budget that
   `__launch_bounds__(384, 1)` imposes on RDNA3. Capping to `2 × 4`
   (`src/kernels_sparse_global.hip:2242,2244` interior, `:2295,2297` boundary) clears it:
   `scripts/spill_audit.sh` → **AUDIT PASS**, 19 objects, 0 SGPR / 0 VGPR spills,
   0 private bytes, 0 stubs; that kernel now compiles at `maxVGPR` 205-215.
   `build/nib_kernel_test`: **9/9 suites** pass (suite 6 `max_abs=0.0000 cos=0.9999999`).
   A cap cannot reorder per-element math, and output is bit-identical (§ Kernel A/B).
3. **Kernel-level effect is large, engine-level effect is not.** Isolated FA-prefill
   launches get 17-29% faster (§ Kernel A/B); end to end the change moves prefill by
   **less than 1%** and decode not at all (§ Server A/B). With `--sparse-pages 224` the
   global-attention read set is fixed, so attention is a small slice of prefill on this
   MoE model. The v0.2.1 note "spill regressions land on prefill, not decode" was observed
   with **1,286** vgprSpills in a single instantiation; a 252-VGPR regression spread over
   six instantiations is below engine noise.
4. **Absolute performance matches the published WSL2 baseline** within ~1.5% across the
   prefill ladder and decode (§ Cross-check), so the native-Linux / newer-toolchain path
   loses nothing relative to the documented 7900 XT numbers.
5. `gfx1102` / `gfx1103` (cross-target compile of the same TU): this change takes those
   prefill kernels from 780 spills to **0**. `inkling_fa_decode_sparse_global_kernel<3, …>`
   still spills on those two targets, independent of this change — left untouched, worth a
   follow-up.

## Kernel A/B — `scripts/bench_fa_sparse.hip`

Exclusive GPU (server stopped), `gfx1100`, d=128, 8:1 GQA, WMMA_M=64, 384 threads,
`n_tokens=512`, `sparse_pages=224`, `kv_type=0` (Q8_0), `meta_fmt=0`, min over 12 launches
after 3 warmups. `base` = `f6b3694` (spilling), `fixed` = `d0f4517`.

```
# from the repo root; prints RESULT lines with ms_min / attn_tok_per_s / bitdiff.
# Step 1 compiles the bench TU, step 2 links it against the SAME object that
# scripts/spill_audit.sh inspects, so the bench cannot drift from the audited device code.
/opt/rocm/bin/hipcc --offload-arch=gfx1100 -O3 --std=c++17 -DGPU_TARGET=gfx1100 \
    -DWITH_ROCM -DWMMA_M=64 -I include -I src \
    -c docs/rx7900xt/hip717_clang24_2026-11-28/scripts/bench_fa_sparse.hip -o bench.o
/opt/rocm/bin/hipcc bench.o build/kernels_sparse_global.o -o bench_fa_sparse -lamdhip64
./bench_fa_sparse <n_kv> <n_tokens> <sparse_pages> <iters> <rel_mode> <kv_type> <meta_fmt> [out_dump]
```

`base` column: `git checkout f6b3694 -- src/kernels_sparse_global.hip && cmake --build build`,
then relink the bench against that object (the two bench binaries then differ in nothing
else), and restore with `git checkout HEAD -- src/kernels_sparse_global.hip`. A fresh smoke
run of the recipe above reproduces the `fixed` column: `8192 512 224 6 0 0 0` ->
`ms_min=1.9677`, `bitdiff=0` (vs 2.0015 in the table; session spread < 2%).
Verbatim `RESULT` lines for all ten shapes: `raw/kernel_ab_gfx1100.txt`.

`rel_mode=0` (`rel_table` bias path):

| `n_kv` | base ms | fixed ms | delta | base tok/s | fixed tok/s |
|---|---|---|---|---|---|
| 8,192 | 2.7179 | 2.0015 | **−26.4%** | 188,378 | 255,804 |
| 16,384 | 4.3064 | 3.5733 | −17.0% | 118,892 | 143,283 |
| 32,768 | 4.4693 | 3.5528 | −20.5% | 114,560 | 144,112 |
| 65,536 | 4.4015 | 3.3821 | −23.2% | 116,323 | 151,387 |
| 131,072 | 4.2445 | 3.3753 | −20.5% | 120,627 | 151,690 |

`rel_mode=1` (`r_prep` bias path):

| `n_kv` | base ms | fixed ms | delta | base tok/s | fixed tok/s |
|---|---|---|---|---|---|
| 8,192 | 2.4418 | 1.9128 | −21.7% | 209,680 | 267,674 |
| 16,384 | 4.4832 | 3.5785 | −20.2% | 114,205 | 143,075 |
| 32,768 | 4.4220 | 3.2638 | −26.2% | 115,785 | 156,873 |
| 65,536 | 4.5603 | 3.3309 | −27.0% | 112,274 | 153,714 |
| 131,072 | 4.6596 | 3.3231 | **−28.7%** | 109,882 | 154,073 |

Across all 10 shapes the two builds write **byte-identical** output (md5 of the dumped
tensor) and the harness's CPU reference reports `bitdiff=0` / `maxrel=0.0` for both.
A shallower sweep run during development (`n_kv` 2,048-131,072, other shapes) landed in
the same −21% … −37% band.

## Server A/B — live service, binary swap

Method: `sudo /opt/nib/scripts/ab_use.sh {pre-fix|fixed}` (see `scripts/ab_use.sh`),
`sudo systemctl restart load_gpu0`, ~65 s to load, 5-rep warmup, then 4 reps per prompt.
Same host, same GPU, same flags, same model; input prompt hashes identical across runs
(`prompt_sha256_prefix` in `raw/probe_prefix.json` / `raw/probe_fixed.json`).

`scripts/srv_probe.py` — prefill ms (median of 4) / tok/s:

| prompt tok | base ms | fixed ms | base tok/s | fixed tok/s |
|---|---|---|---|---|
| 114 | 52.43 | 52.98 | 778.9 | 770.9 |
| 455 | 81.96 | 82.09 | 5,035.8 | 5,027.9 |
| 1,668 | 146.64 | 146.66 | 10,784.8 | 10,783.4 |
| 10,145 | 688.39 | 692.71 | 13,962.9 | 13,884.2 |
| 26,140 | 2,359.31 | 2,360.90 | 10,508.0 | 10,490.5 |

A first sample (`raw/ab_prefix.json` vs `raw/ab_fixed.json`, 26,140-token prefill) showed
1.0-1.8% faster prefill on the fixed build; it did not replicate on the repeat
(`raw/ab_fixed2.json`) or on the ladder above. Decode: 122.91 → 121.84 tok/s (−0.9%,
within session noise), `tok_with_rounds` 192.59 → 190.92, `mtp_tok_per_round` 1.567 → 1.600.
Two simultaneous requests (`scripts/srv_conc.py`, aggregate prefill tok/s, median of 2-3
rounds) — also unchanged:

| concurrent load | base agg tok/s | fixed agg tok/s |
|---|---|---|
| 2 x ~6.4k tok (`raw/conc_prefix_10k.json` / `conc_fixed_10k.json`) | 3,921.4 | 3,876.2 |
| 2 x ~12.8k tok (`raw/conc_prefix_20k.json` / `conc_fixed_20k.json`) | 3,864.2 | 3,833.7 |

**Determinism:** at `temperature=0`, 3 prompts × 2 repeats, both builds return identical
completion `sha256` (`b626fb2b40f73f75…`, `14325c6fed6a316d…`, `90c98c13bd1670be…`) and
identical `mtp_tok_per_round` (1.575 / 2.783 / 1.600).

## Cross-check against the published v0.2.1 7900 XT numbers

Same GPU and same flags, but WSL2 + ROCDXG and ROCm 7.2.4 / HIP-Clang 22
([`../v0.2.1_2026-11-27/NOTES.md`](../v0.2.1_2026-11-27/NOTES.md)), vs native Linux on
ROCm core 10.2 / clang 24 with this fix:

| prompt tok | docs v0.2.1 tok/s | native base | native fixed |
|---|---|---|---|
| 114 | 758 | 778.9 | 770.9 |
| 455 | 5,014 | 5,035.8 | 5,027.9 |
| 1,668 | 10,731 | 10,784.8 | 10,783.4 |
| 10,145 | 13,741 | 13,962.9 | 13,884.2 |
| 26,140 | 10,590 | 10,508.0 | 10,490.5 |
| decode (100 tok, MTP) | 120.96 | 122.91 | 121.84 |

Native Linux matches the documented WSL2 baseline within ~1.5% everywhere. Deep ladder on
the fixed build, single request, 224 pages, `max_ctx` 132,768
(`raw/deep_fixed.json`):

| prompt tok | ttft ms | prefill tok/s |
|---|---|---|
| 15,397 | 1,228.2 | 12,800.5 |
| 29,145 | 2,909.6 | 10,165.5 |
| 57,800 | 7,274.3 | 8,124.5 |
| 115,126 | 17,410.2 | 6,714.5 |

## Caveats / follow-ups

- `scripts/bench_fa_sparse.hip` has a harness bug: with `kv_type=1` (Q4_1 KV) it dies on an
  illegal memory access at `n_kv >= 16384`, and its CPU reference disagrees for Q4_1 at any
  depth. This is the harness, not the kernel — the engine serves `q4_1` at 262,848
  context. Use `kv_type=0` when re-running.
- Server A/B costs two root service restarts (~65 s load each) and needs a long warmup:
  per-prompt medians have ~1.5% session-to-session scatter, which is larger than the effect
  being measured. Keep kernel-level A/B as the primary evidence and server A/B as a
  no-regression check.
- `scripts/ab_use.sh` deliberately leaves `.keep-*` copies next to the deployed binaries;
  the deployed set is `nib`, `nib.pre-fix`, `nib.fixed`.
- Server numbers here are on the `-c 132768` instance (slot cap 66,368), not the 262,848
  config in `run.sh`; for 220k-class numbers see the v0.2.1 folder and
  `docs/amd/inkling2-rx9070xt-vs-rx7900xt-2026-11-27.md`.

## Repro

```
./scripts/spill_audit.sh                    # AUDIT PASS: 19 objects, 0/0/0/0
cmake --build build -j && ./build/nib_kernel_test        # 9/9 suites
# kernel A/B: see "Kernel A/B" above (needs the GPU to itself)
# server A/B: sudo /opt/nib/scripts/ab_use.sh pre-fix && sudo systemctl restart load_gpu0
#             then scripts/srv_probe.py / srv_bench.py / srv_conc.py, repeat with `fixed`
cat raw/kernel_ab_gfx1100.txt        # verbatim RESULT lines behind the Kernel A/B tables
```
