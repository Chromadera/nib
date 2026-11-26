# AMD Radeon RX 9070 XT (16 GB, RDNA4 `gfx1201`) — Nib `NVFP4-23` vs `NVFP4-All`

Real-silicon evaluation of **Nib 0.1.2** (RDNA4 + dynamic NVFP4 layer loading tree, built `-DGPU_TARGET=gfx1201`) running **Inkling 2 Mini** in two bit-exact NVFP4 repacks, on an **AMD Radeon RX 9070 XT** (`gfx1201` / Navi 48, 64 CU / 32 WGP, 16 GB GDDR6, 256-bit) under Windows/WSL2 (`ROCDXG`, ROCm 7.2.4), **2026-11-26**.

No `gbench` was run. All runs: `--cpu-embd`, `--sparse-pages 224`, greedy, localhost `127.0.0.1`.

> **TL;DR**
> - **Both quants run text-only on 16 GB with no throughput loss from GTT spill. They decode at the same speed; NVFP4-All just has more room.** NVFP4-All's 0.48 GiB of smaller weights buys **3× the context**: **73,728 tokens** max (q4_0) vs **24,576** for NVFP4-23, plus **`--slots 4 @ 48K`** and **`--slots 8 @ 8K`**, where NVFP4-23 stops at `--slots 2 @ 16K`.
> - **Decode speed is identical across the two quants:** `nib bench` tg128 **96.2 tok/s at d0 → 92.2 tok/s at 72.7K** (NVFP4-All) vs **96.0 → 92.2** (NVFP4-23). Full BetterBench weighted single-stream decode is **94.5–95.7 tok/s** (All) and **94.2–95.2 tok/s** (23), with TTFT p50 76–79 ms.
> - **Prefill:** pp512 **5,019–5,137 tok/s at d0**. BetterBench prefill held **4,006 tok/s on a real 70,026-token prompt** (All, `-c 73728`).
> - **Concurrency (All, `--slots 4 -c 49152 q4_0`, 32/32 OK per level):** **92.7 / 126.8 / 162.0 tok/s aggregate** at c = 1 / 2 / 4, reproduced by a rerun at 93.9 / 127.3 / 160.4.
> - **DSpark MTP does not fit on either quant.** `DSparkDrafter: hipMalloc failed` even at the minimum `-c 4096 q4_0 --draft-ctx 512`, so no MTP sweep or MTP BetterBench was possible.
> - **Vision fits only on NVFP4-All:** up to **1440p** at `-c ≤ 8192`, **1080p** at `-c 16384`, and only the small 1024×512 card image at 32K. 4K / 2560² return a clean `HTTP 400`. Post-image text decode is unchanged. **NVFP4-23 can't load the ViT at all** (OOM at `-c 4096`).
> - **GPU kernels on gfx1201 / ROCm 7.2.4: 0 SGPR / 0 VGPR spills across all 19 objects** (RDNA3 gfx1100 spilled 1,312 VGPRs in `kernels_fa.o` on the same compiler).
> - **GTT spill flag:** the `--cpu-embd` pinned `tok_embd` buffer itself reads as **+514 MB USED_GTT**. That's intentional host memory. Beyond it, every Stage 2 / sweep run was **0–6 MB**. Four BetterBench serves and two vision serves showed **+30 to +133 MB**, mostly appearing at model load (see §8). Solo decode measured after each of those runs is unchanged at 95–98 tok/s.

---

## Folder contents

| File | What it is |
|---|---|
| `README.md` | This report |
| [`benchmarks.json`](benchmarks.json) | Machine-readable dataset: selftests, 113 Stage 2 probes, max-ctx matrix, `nib bench` sweeps, BetterBench summaries plus per-run VRAM/GTT, and vision steps (built by `scripts/build_benchmarks.py`) |
| [`betterbench_all_s4_q40_48k.json`](betterbench_all_s4_q40_48k.json) / [`.html`](betterbench_all_s4_q40_48k.html) | Full BetterBench, **NVFP4-All `--slots 4 -c 49152 q4_0`** (single-stream + prefill + c=1,2,4) |
| [`betterbench_all_s1_q40_72k.json`](betterbench_all_s1_q40_72k.json) / [`.html`](betterbench_all_s1_q40_72k.html) | Full BetterBench, **NVFP4-All `--slots 1 -c 73728 q4_0`** (max context; prefill ladder to 70K real tokens) |
| [`betterbench_23_s2_q40_16k.json`](betterbench_23_s2_q40_16k.json) / [`.html`](betterbench_23_s2_q40_16k.html) | Full BetterBench, **NVFP4-23 `--slots 2 -c 16384 q4_0`** (single-stream + prefill + c=1,2) |
| [`betterbench_23_s1_q40_24k.json`](betterbench_23_s1_q40_24k.json) / [`.html`](betterbench_23_s1_q40_24k.html) | Full BetterBench, **NVFP4-23 `--slots 1 -c 24576 q4_0`** (max context; prefill ladder to 23K real tokens) |
| [`scripts/`](scripts/) | Benchmark scripts: `bootstrap.sh`, `build.sh`, `repack.sh`, `bb_install.sh`, `vram_probe.cpp`, `selftests.sh`, `lib.sh`, `client.py`, `probe.sh`, `search.sh`, `s2_*.sh`, `bench.sh`, `bb.sh`, `concheck.sh`, `vis.sh`, queue scripts, `build_benchmarks.py` |

---

## 1. Environment

| Layer | Value |
|---|---|
| GPU | AMD Radeon RX 9070 XT, ASRock Navi48 XTX (IFWI `023.008.000.068`), `gfx1201`, 64 CU (HIP reports 32 MPs = WGPs), 64 MiB L3, PCIe Gen 5 x16 |
| VRAM | `16,304 MB` (`amd-smi`) / `16,188.5 MiB = 15.81 GiB` (`hipGetDeviceProperties`) |
| Host | Windows + WDDM, Adrenalin `26.10.44-260817a-204192C-AIB`; WSL2 kernel `6.18.58.1-microsoft-standard-WSL2`, GPU via `/dev/dxg` (no `/dev/kfd`) |
| ROCm / Image | `rocm/dev-ubuntu-24.04:7.2.4` (HIP 7.2, AMD clang 22) |
| Nib | `nib 0.1.2 (built: gfx1201, runtime: gfx1201 [AMD Radeon RX 9070 XT])`, `cmake -DCMAKE_BUILD_TYPE=Release -DGPU_TARGET=gfx1201 -DROCM_PATH=/opt/rocm -DCMAKE_EXE_LINKER_FLAGS="-no-pie"` |
| BetterBench | `0.6.1` (`GGZ14/BetterBench@f72bb901`), corpus hash `e332dceff176d033` |
| Co-residency | `[step7i] occupancy blocks/WGP (wgp=32): wo_b=2 wo_c=2 wo_c_h2=3 mix_rms_qkv=4 a_qkv=4` |

### Spill audit (`scripts/spill_audit.sh build`, gfx1201) — **PASS**

All 19 objects: `sgprSpills=0 vgprSpills=0 maxPriv=0 stubs=0`. Highest VGPR counts: `kernels_verify_batch.o` 251, `kernels_sparse_global.o` 217, `kernels_fa.o` 197.

---

## 2. Weights — bit-exact repacks

`repack_nvfp4.cpp` compiled with `g++ -O3 -march=native -fopenmp` from `ggml-org/Inkling-2-Mini-GGUF` `Inkling-2-Mini-Q8_0.gguf` (`27,820,099,264` B, sha256 `d57542f8…`).

| Quant | Repack arg (`max_nvfp4_layer`) | Expert tensors → NVFP4 | Bytes (verified) | sha256 | Weights in VRAM w/ `--cpu-embd` |
|---|---:|---|---:|---|---:|
| **NVFP4-23** | `22` | `blk.0–22` (46), `blk.23` stays Q8_0 | **`16,099,143,008`** ✅ | `1ed0b242…5d7981a9` | **14.46 GiB** |
| **NVFP4-All** | `23` | `blk.0–23` (48) | **`15,589,536,192`** ✅ | `f661c5b2…d07ef5636f` | **13.98 GiB** |

Per-tensor relRMSE for `blk.13–23`: 0.007–0.019 for `blk.13–19`, ~0.086 for `blk.20–23`.

Pack files: `mtp.safetensors` (`7c3b6513…`), `mmproj-Inkling-2-Mini-Q8_0.gguf` (`23328727…`), and `draft_shortlist.bin` (`6ce69b27…`, required for MTP initialization).

### `nib selftest` (`--cpu-embd --sparse-pages 224`, repack verify `max abs err = 0.000000` on `token_embd` and `blk.0.ffn_gate_up_exps`)

| Quant | `q8_0` KV | `q4_1` KV | `q4_0` KV | Samplers (4 tests) | `--mtp` |
|---|---|---|---|---|---|
| NVFP4-All | **PASS** `cos_sim 0.9999909` | rc=2 `cos_sim 0.9979789` | rc=2 `cos_sim 0.9980436` | 4/4 PASS | `DSparkDrafter: hipMalloc failed` |
| NVFP4-23 | **PASS** `cos_sim 0.9999912` | rc=2 `cos_sim 0.9980591` | rc=2 `cos_sim 0.9981440` | 4/4 PASS | `DSparkDrafter: hipMalloc failed` |

Prefill and decode agree on the next token (`111`) in every case. The `rc=2` on 4-bit KV comes from the selftest's fixed `cos_sim > 0.999` gate meeting 4-bit KV quantization noise. It's the same on both quants and both 4-bit KV formats, and not specific to gfx1201: the RDNA3 engineering log records `cos_sim` 0.996–0.998 on similar paths. Not treated as a defect, but it is a selftest exit code ≠ 0.

---

## 3. VRAM ceiling

### Stage 1 — raw `hipMalloc` + `hipMemset`

- Idle at boot: `amd-smi` USED 1,920 MB / FREE 14,384 MB / GTT 260 MB. `hipMemGetInfo` free = 14,641 MiB; the Windows desktop holds ~1.5 GiB.
- Under pressure, WDDM evicts the desktop. After the probe, USED dropped to 277 MB and stayed there for the rest of the session (idle USED 273–487 MB, GTT 55–113 MB).
- **Hard ceiling: `15,672.0 MiB` (15.30 GiB)**, the same for 16 MiB chunks and for 489 × 32 MiB + refinement (0 MiB fragmentation loss). That leaves **~0.84 GiB** beyond NVFP4-23's weights and **~1.32 GiB** beyond NVFP4-All's, which has to cover KV, scratch (~0.46 GiB fixed at 4K), and the HIP runtime.

### Stage 2 — real `nib serve` ceiling matrix (`--cpu-embd --sparse-pages 224`)

Each probe (`scripts/probe.sh`) starts `nib serve`, polls `amd-smi` every second, sends one 128-token greedy request, then an N-way burst at `--slots N`, then a 64-token request. **PASS** = server healthy, every request OK, solo decode ≥ 60 tok/s, and peak GTT ≤ idle + 514 MB (`--cpu-embd`) + 16 MB. The search (`scripts/search.sh`) bisects to 4 K (2 K for NVFP4-23) granularity. Every failure above the max was a hard `hipMalloc` OOM at init (`engine.cpp:145`, `vision_encoder.cpp:314`, or `DSparkDrafter`). There were no stalls or TDRs.

**Max clean `-c` (all 0 MB GTT over the `--cpu-embd` baseline, ≤ 6 MB with multi-slot batch graphs):**

| Config | **NVFP4-All** (13.98 GiB) | **NVFP4-23** (14.46 GiB) |
|---|---:|---:|
| `q4_0`, `--slots 1` | **`73,728`** (15.06 GiB engine, 241 MB free) | **`24,576`** (15.10 GiB, 236 MB free) |
| `q4_1`, `--slots 1` | **`65,536`** (15.06 GiB) | **`16,384`** (15.05 GiB) |
| `q8_0`, `--slots 1` | **`32,768`** (15.01 GiB) | **`8,192`** (15.08 GiB) |
| `q4_0`, `--slots 2` | `65,536` (burst 128.0 tok/s agg) | `16,384` (125.3 agg) |
| `q4_1`, `--slots 2` | `61,440` (131.2 agg) | `12,288` (128.8 agg) |
| `q8_0`, `--slots 2` | `24,576` (130.6 agg) | — |
| `q4_0`, `--slots 4` | **`49,152`** (165.3 agg, 4/4 OK) | ✗ (OOM even at `4,096`) |
| `q4_1`, `--slots 4` | `45,056` (169.2 agg) | ✗ (OOM at `4,096`) |
| `q8_0`, `--slots 4` | `8,192` (172.2 agg) | — |
| `q4_0` / `q4_1`, `--slots 8` | **`8,192`** only (1K/slot; 179.5 / 177.3 agg, 8/8 OK); `16,384` OOM | ✗ (OOM at `8,192`) |
| **Max `--slots`** | **8** (at 8K), **4** (at 48K) | **2** |
| `+ --mtp` (any KV, any `-c ≥ 4096`) | ✗ `DSparkDrafter: hipMalloc failed` | ✗ `DSparkDrafter: hipMalloc failed` |
| `+ --mmproj`, `q4_0`/`q4_1`, `--slots 1` | `32,768` boots (images: see §7) | ✗ (`vision_encoder.cpp:314` OOM at `4,096`) |
| `+ --mmproj`, `q4_1`, `--slots 2` | `20,480` boots | ✗ |

Marginal KV cost (measured): **q4_0 ≈ +0.29 GiB / 32K**, **q4_1 ≈ +0.32 GiB / 32K**, **q8_0 ≈ +0.54 GiB / 32K**, matching the RX 7900 XT tables. The 0.48 GiB weight difference between the two quants therefore costs NVFP4-23 about 49K tokens of q4_0 context.

**Why MTP doesn't fit:** the DSpark drafter needs ~697 MiB (606 MiB `mtp.safetensors` + state) on top of ~22 MiB MTP I/O buffers. At `-c 4096 q4_0`, NVFP4-All still had 919 MB free (`amd-smi`) when drafter init failed, and NVFP4-23 had 281 MB. The engine's fixed prefill scratch is sized by the compile-time `MAX_UBATCH` (0.46 GiB at 4K), so no runtime flag frees enough room; shrinking it requires a compile-time/code change.

---

## 4. Non-MTP `nib bench` sweeps (`-p 512 -n 128 -r 2 --cpu-embd --sparse-pages 224`, at each max `-c`)

All six sweeps: rc=0, **static decode HIP graph = 404 kernel nodes**, peak GTT = idle 55 + 514 MB → **0 MB spill**, min free VRAM 236–321 MB.

### Prefill `pp512` (tok/s)

| Depth | All `q4_0 @ 73,728` | All `q4_1 @ 65,536` | All `q8_0 @ 32,768` | 23 `q4_0 @ 24,576` | 23 `q4_1 @ 16,384` | 23 `q8_0 @ 8,192` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 5,019.0 | 5,092.1 | 5,087.2 | 5,036.2 | **5,137.2** | 5,072.2 |
| 7,424 | — | — | — | — | — | 4,453.1 |
| 8,192 | 4,480.7 | 4,482.9 | 4,467.7 | 4,475.4 | 4,476.2 | — |
| 15,616 / 16,384 | 4,178.7 | 4,257.3 | 4,252.8 | 4,237.9 | 4,273.3 (15,616) | — |
| 23,552 | — | — | — | 4,051.3 | — | — |
| 32,000 / 32,768 | 3,864.4 | 3,866.4 | 3,866.1 (32,000) | — | — | — |
| 64,000 / 65,536 | 4,047.9 | 3,399.1 (64,000) | — | — | — | — |
| 72,704 | **4,057.1** | — | — | — | — | — |

### Decode `tg128` (tok/s)

| Depth | All `q4_0 @ 73,728` | All `q4_1 @ 65,536` | All `q8_0 @ 32,768` | 23 `q4_0 @ 24,576` | 23 `q4_1 @ 16,384` | 23 `q8_0 @ 8,192` |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | **96.24** | 96.15 | 95.50 | **95.99** | 95.87 | 95.26 |
| 7,424 / 8,192 | 94.00 | 93.80 | 92.49 | 93.69 | 93.44 | 92.43 (7,424) |
| 15,616 / 16,384 | 92.60 | 92.33 | 90.53 | 92.44 | 91.98 (15,616) | — |
| 23,552 | — | — | — | 92.24 | — | — |
| 32,000 / 32,768 | 92.42 | 92.12 | 90.43 (32,000) | — | — | — |
| 64,000 / 65,536 | 92.30 | 92.01 (64,000) | — | — | — | — |
| 72,704 | **92.22** | — | — | — | — | — |

- **Same-depth comparison, NVFP4-All vs NVFP4-23:** within ±0.3 tok/s decode and ±2% prefill at every shared depth. Converting the last layer's experts (`blk.23`) to NVFP4 costs nothing in speed on gfx1201; its only effect is the VRAM it frees.
- `q8_0` KV is ~1.8% slower in decode at 16K+ than 4-bit KV. Sparse-224 decode is flat from 16K to 73K (−0.4%).
- One oddity, not investigated further: **All `q4_1 @ 64,000`** pp512 is **3,399** vs **4,048** for q4_0 at 65,536 (each ±10, r=2).
- **Reference context (different GPU / quant):** RX 7900 XT (`gfx1100`, NVFP4-Mixed, 800 GB/s) gave tg64 124.6 / pp512 4,240 at d0 ([`../../rx7900xt/README.md`](../../rx7900xt/README.md)). The 9070 XT decodes at ~77% of that, close to the 640 / 800 GB/s spec-sheet bandwidth ratio, but prefills **~18–21% faster**, consistent with the 0-spill gfx1201 FA/sparse kernels (gfx1100 spills 1,312 VGPRs on ROCm 7.2.4).

---

## 5. MTP `nib gen --mtp` sweeps — **not possible on 16 GB**

`DSparkDrafter: hipMalloc failed` on **both** quants at **every** tried config:

| Attempt | NVFP4-All | NVFP4-23 |
|---|---|---|
| `selftest --mtp` (ctx 4096, q8_0 and q4_1) | FAIL (drafter hipMalloc) | FAIL (drafter hipMalloc) |
| `serve -c 8192 q4_0 --mtp --draft-ctx 1024` | FAIL | — |
| `serve -c 4096 q4_0 --mtp --draft-ctx 512` | FAIL (919 MB free before drafter) | FAIL (281 MB free) |

So there are no MTP decode, `tok/round`, `avg_m` or `draft`/`verify` numbers, and no MTP BetterBench. The code+docs prompt sweep (`bench_mtp.sh` style) was skipped because no MTP config can even boot.

---

## 6. Full BetterBench 0.6.1 (localhost, `--greedy`, cold nonce prefix cache, `runs_per_category: 5`, `warmup: 1`, 32 requests/level, `prefill_runs: 2`)

Four full runs, two per quant: the **max-slots** config and the **max-context** config. The base config uses the standard 8-slot BetterBench profile with `concurrency_levels` trimmed to the slots that fit and `prefill_depths` trimmed and extended to each config's real max context. See *ladder note* below.

| Run | Serve flags (`--cpu-embd --sparse-pages 224 --no-vision`) | Engine / device used | Duration | Server after | Batch graphs |
|---|---|---|---:|---|---:|
| `all_s4_q40_48k` | NVFP4-All `--slots 4 -c 49152 --kv-type q4_0` | 15.06 / 15.63 GiB | 588 s | alive | 11 |
| `all_s1_q40_72k` | NVFP4-All `--slots 1 -c 73728 --kv-type q4_0` | 15.06 / 15.77 GiB | 470 s | alive | 0 |
| `23_s2_q40_16k` | NVFP4-23 `--slots 2 -c 16384 --kv-type q4_0` | 15.10 / 15.70 GiB | 466 s | alive | 1 |
| `23_s1_q40_24k` | NVFP4-23 `--slots 1 -c 24576 --kv-type q4_0` | 15.10 / 15.75 GiB | 380 s | alive | 0 |

### Single-stream decode (median tok/s per category, 5 runs each, 40/40 OK in every run)

| Category | All s4 48K | All s1 72K | 23 s2 16K | 23 s1 24K |
|---|---:|---:|---:|---:|
| chat | 96.2 | 95.0 | 94.9 | 95.9 |
| code | 95.3 | 94.3 | 93.9 | 95.0 |
| file_edit | 95.4 | 94.1 | 94.3 | 95.0 |
| json | 96.8 | 95.2 | 94.7 | 96.0 |
| math | 95.8 | 94.4 | 94.2 | 95.5 |
| prose | 95.4 | 94.4 | 94.0 | 95.0 |
| reasoning | 95.3 | 94.1 | 94.1 | 94.9 |
| summarization | 96.3 | 95.1 | 95.1 | 96.0 |
| **Weighted (BetterBench)** | **95.7** | **94.5** | **94.2** | **95.2** |
| Update p99 / TTFT p50 | 10.7 ms / 78 ms | 12.7 ms / 76 ms | 12.6 ms / 78 ms | 10.8 ms / 79 ms |

The spread between runs (94.2–95.7) is session-to-session variation of ~1.5%, not a quant effect: the order isn't All > 23, and the two NVFP4-23 runs bracket NVFP4-All s1. 27/40 runs per config stop at `max_tokens` (BetterBench's thinking-model note), the same on every run.

### Concurrency sweep (32 requests/level, all **32/32 OK**)

| Level | All s4 48K: agg / per-req med / TTFT p50 | All s4 rerun (`concheck`) | 23 s2 16K: agg / per-req / TTFT | All s1 72K | 23 s1 24K |
|--:|---|---|---|---|---|
| c = 1 | 92.7 / 94.6 / 76.5 ms | 93.9 / 95.7 / 77.7 ms | 92.4 / 94.3 / 77.4 ms | 92.7 / 94.6 / 77.0 ms | 92.4 / 94.2 / 78.1 ms |
| c = 2 | **126.8** / 65.1 / 96.1 ms | 127.3 / 65.6 / 98.3 ms | **126.1** / 65.0 / 96.6 ms | — | — |
| c = 4 | **162.0** / 42.1 / 122.9 ms | 160.4 / 41.8 / 125.8 ms | — (slots 4 doesn't fit) | — | — |

The `--slots 8 -c 8192` config (NVFP4-All only) handled an 8-way burst at **177–180 tok/s aggregate** in Stage 2. It wasn't BetterBench'd: at 1K tokens per slot, BetterBench's longer concurrency prompts plus generation would overflow a slot.

### Prefill ladder (median of 2 runs, cold cache; *real* prompt tokens shown)

| NVFP4-All s1 `-c 73728` | | NVFP4-All s4 `-c 49152` | | NVFP4-23 s1 `-c 24576` | | NVFP4-23 s2 `-c 16384` | |
|---:|---:|---:|---:|---:|---:|---:|---:|
| **tokens** | **PP tok/s** | **tokens** | **PP tok/s** | **tokens** | **PP tok/s** | **tokens** | **PP tok/s** |
| 2,035 | 4,271.9 | 2,026 | 4,327.4 | 2,058 | 4,091.8 | 2,075 | 4,178.4 |
| 8,000 | **4,536.7** | 8,106 | 4,367.3 | 8,103 | 4,318.1 | 8,060 | **4,539.6** |
| 16,040 | 4,468.7 | 16,139 | 4,286.4 | 16,043 | **4,455.1** | 15,725 | 4,481.6 |
| 32,184 | 4,329.5 | 32,140 | 4,327.9 | 23,252 | 4,392.1 | | |
| 42,915 | 4,217.6 | 37,746 | 4,270.2 | | | | |
| **70,026** | **4,006.4** (TTFT 17.5 s) | | | | | | |

**Ladder note:** BetterBench's `target_depth` turns into ~0.78× as many real tokens (48,128 → 37,746 on the first run). To make the ladder actually reach each config's max context, runs 2–4 add a top target of `0.975·ctx / 0.80` and pass `--max-model-len` = top target + 512, recorded as `--note max_model_len_note=…` and `--note served_ctx=<ctx>`. So the `max_model_len` field in those three JSONs is *not* the served context; the served context is in the `served_ctx` note. Run 1 (`all_s4_q40_48k`) used the plain `--max-model-len 49152`, so its ladder stopped at 37.7K of 49K.

---

## 7. Vision (`--mmproj`, `mmproj-Inkling-2-Mini-Q8_0.gguf`, 329.9 MiB) quick matrix

Sequence per config: text → 1024×512 card → 1920×1080 → 2560×1440 → 3840×2160 → 2560×2560 → text. Every 1080p/1440p answer correctly named the quadrants (red / blue / green / yellow). Failed images return `HTTP 400 "Failed to decode/encode image"` from the ViT-workspace OOM guard, and the server stayed alive in every case.

| Config (NVFP4-All unless noted, `--slots 1`) | Free @ ready | card | 1080p | 1440p | 4K | 2560² | Text before → after | GTT over baseline |
|---|---:|---|---|---|---|---|---|---:|
| `-c 4096 q4_0` | 500 MB | ✅ 95.9 | ✅ 94.8 (TTFT 539 ms) | ✅ **94.6** (1,059 ms) | 400 (needs 414 MiB, 440 free) | 400 | 96.3 → 96.7 | 0 MB |
| `-c 8192 q4_1` | 402 MB | ✅ 95.7 | ✅ 94.7 (547 ms) | ✅ **94.1** (1,064 ms) | 400 (414 MiB vs 382 free) | 400 | 96.4 → 96.6 | 0 MB |
| `-c 16384 q4_1` | 275 MB | ✅ 95.9 | ✅ **94.6** (571 ms) | 400 (234 MiB vs 258 free) | 400 | 400 | 96.6 → 96.8 | +30 MB ⚠ |
| `-c 32768 q4_0` | 240 MB | ✅ 95.8 | 400 (132 MiB vs 228 free) | 400 | 400 | 400 | 96.4 → 96.6 | +6 MB |
| `-c 32768 q4_1` | 171 MB | 400 (34 MiB vs 53 free) | 400 | 400 | 400 | 400 | 96.5 → 96.7 | +89 MB ⚠ (present at load) |
| `--slots 2 -c 20480 q4_1` | 218 MB | ✅ 95.8 | 400 | 400 | 400 | 400 | 96.5 → 96.7 | +1 MB |
| **NVFP4-23** `-c 4096 q4_0` | — | **boot OOM** `vision_encoder.cpp:314` | | | | | — | — |

- **No post-image text regression** in any config (Δ ≤ +0.4 tok/s). `vision_workspace_mib` is back to **0** after every image.
- Under WDDM, a ViT workspace allocation fails even when `hipMemGetInfo` shows slightly *more* free than requested (414 MiB vs 440–487 MiB at 4K ctx). Plan on **≥ ~1.2× the workspace** being free.
- **Practical vision configs:** NVFP4-All `-c 8192` (up to 1440p) or `-c 16384` (up to 1080p), both text-only otherwise. With MTP impossible, vision + MTP can't be combined.

---

## 8. GTT accounting and spill flags (read before quoting "0 MB")

- **`--cpu-embd` pinned host RAM is counted in `USED_GTT`.** At the same `-c 8192`, GTT reads **569 MB with `--cpu-embd`** and **100 MB with `--no-cpu-embd`** (idle 55). The +514 MB is the 469 MiB `tok_embd` buffer living in GPU-mapped host RAM, which is the point of `--cpu-embd`, not spill. Spill is therefore reported as **peak − idle − 514 MB**.
- **`amd-smi` on WSL2 is GPU-wide**, including Windows host allocations. Idle GTT moved between **55 and 113 MB** during the session, and the desktop's 1.5 GiB was evicted from VRAM in Stage 1 and never came back.
- **Clean (0–6 MB over baseline):** every Stage 2 PASS probe (113 probes), all six `nib bench` sweeps, and 4 of 6 vision configs. The 1–6 MB in multi-slot probes coincides with static batch HIP-graph capture.
- **⚠ Flagged (> 16 MB over baseline):**

| Run | GTT over baseline | When it appeared | Solo decode measured afterwards |
|---|---:|---|---|
| BB `all_s4_q40_48k` | **+133 MB** | first concurrency level (c=2), during capture of 9 more batch graphs; VRAM used +78 MB at the same moment | concheck rerun: GTT already +90 MB **at ready**, whole concurrency sweep added **+3 MB**; solo decode 96.3/97.8 before → 95.3/97.3 after |
| BB `all_s1_q40_72k` | **+60 MB** | **at model load** (GTT 113 → 669 MB as VRAM filled to 151 MB free), before any request | 95.2 / 97.3 tok/s |
| BB `23_s2_q40_16k` | **+90 MB** | at load | 96.4 / 98.3 tok/s |
| BB `23_s1_q40_24k` | **+89 MB** | at load | 95.3 / 96.6 tok/s |
| Vision `-c 32768 q4_1` | **+89 MB** | at load | text 96.5 → 96.7 |
| Vision `-c 16384 q4_1` | **+30 MB** | during image encodes | text 96.6 → 96.8 |

  **Interpretation (measured, but the cause is unproven):** the extra GTT is either WDDM demoting host/desktop or HIP-runtime allocations once VRAM is ~150–250 MB from full, or a small part of nib's footprint landing in GTT. `amd-smi` can't tell which process owns it. None of these runs lost throughput: decode, prefill and concurrency in every flagged run match the 0-MB-GTT probes and sweeps to within 1–2%. **If a strict 0 MB rule is required, step one notch down from max** (for example NVFP4-All `-c 65536`, which ran its `q4_0` probe and q4_1 sweep at exactly +0 MB).

---

## 9. Head-to-head summary

| Metric (best clean config) | **NVFP4-All** | **NVFP4-23** | Winner |
|---|---:|---:|---|
| File size / weights in VRAM (`--cpu-embd`) | 15.59 GB / **13.98 GiB** | 16.10 GB / 14.46 GiB | All |
| Max context, text, `--slots 1` | **73,728** (q4_0) / 65,536 (q4_1) / 32,768 (q8_0) | 24,576 / 16,384 / 8,192 | **All (3×)** |
| Max `--slots` | **4 @ 49,152** · **8 @ 8,192** | 2 @ 16,384 | **All** |
| Concurrency aggregate (BetterBench, highest level that fits) | **162.0 tok/s @ c=4** | 126.1 tok/s @ c=2 | **All** |
| Decode tg128 d0 / deepest | 96.24 / 92.22 @ 72.7K | 95.99 / 92.24 @ 23.6K | tie |
| BetterBench weighted single-stream | 94.5–95.7 | 94.2–95.2 | tie (±1.5% run-to-run) |
| Prefill pp512 d0 | 5,019–5,092 | 5,036–5,137 | tie |
| BetterBench prefill at deepest point | **4,006 tok/s @ 70,026** | 4,392 @ 23,252 | — (different depths) |
| MTP | ✗ | ✗ | — |
| Vision | ✅ up to 1440p (`-c ≤ 8192`) | ✗ (ViT OOM) | **All** |
| Quality (see [`../README.md`](../README.md)) | experts of all 24 layers NVFP4 (48 tensors, mean relRMSE 0.0248, 40×512 KLD `0.00529`) | `blk.0–22` NVFP4 (46 tensors, mean relRMSE 0.0221, **`blk.23` kept Q8_0**, 40×512 KLD `0.00483`) | 23 (`-0.00046` KLD) |

**Recommendation for a 16 GB RX 9070 XT:** **NVFP4-All**. It is the only quant that delivers useful context (64–72K), more than 2 slots, or vision on this card, at the same speed as NVFP4-23. NVFP4-23 only makes sense for short-context (≤24K), single-or-dual-slot text serving where its slight accuracy edge (`-0.00046` KLD) matters.

### Recommended `nib serve` configs (all `--cpu-embd --sparse-pages 224`)

| Goal | Flags | Measured |
|---|---|---|
| Max context, text | NVFP4-All `--slots 1 -c 73728 --kv-type q4_0 --no-vision` | 94.5 tok/s weighted, 4,006 tok/s prefill @ 70K |
| Strict 0 MB GTT, long context | NVFP4-All `--slots 1 -c 65536 --kv-type q4_0 --no-vision` | Stage 2 PASS +0 MB, 96.3 tok/s |
| 4-way API | NVFP4-All `--slots 4 -c 49152 --kv-type q4_0 --no-vision` | 162 tok/s agg @ c=4, 12K/slot, 48K solo |
| 8-way short chat | NVFP4-All `--slots 8 -c 8192 --kv-type q4_0 --no-vision` | 179.5 tok/s agg (8-burst), 1K/slot |
| Vision | NVFP4-All `--slots 1 -c 8192 --kv-type q4_1 --mmproj …` | ≤1440p at 94 tok/s, 4K → HTTP 400 |
| NVFP4-23, text | `--slots 1 -c 24576 --kv-type q4_0 --no-vision` (or `--slots 2 -c 16384`) | 95.2 tok/s weighted / 126 agg @ c=2 |

---

## 10. Methodology & Notes

1. **MTP sweeps and MTP BetterBench were not run:** MTP cannot initialize on either quant on 16 GB (§5).
2. **The BetterBench prefill ladder uses inflated `--max-model-len`** to reach real max context (§6 ladder note).
3. **4-bit-KV selftest exit 2** reflects the 0.999 cosine gate, not a functional failure (§2). The q8_0-KV selftests pass on both quants.
4. **"0 MB GTT"** is defined net of the intentional `--cpu-embd` buffer, and each flagged exception is listed (§8).
5. **`draft_shortlist.bin`** is included alongside `mtp.safetensors` as required for MTP initialization.
