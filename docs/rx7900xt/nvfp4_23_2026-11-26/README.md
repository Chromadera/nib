# RX 7900 XT (20 GB) — `NVFP4-23` full evaluation (2026-11-26)

**Question:** how much extra context, concurrency, MTP and Vision headroom does `Inkling-2-Mini-NVFP4-23.gguf` (NVFP4 experts on layers 0–22) give on a 20 GB RX 7900 XT, compared with `NVFP4-Mixed`?

Every number below is recorded in [`benchmarks.json`](benchmarks.json) and the BetterBench reports in this folder. Where a comparison uses the earlier `NVFP4-Mixed` report ([`../README.md`](../README.md)), it says so. That earlier report was measured with an older build (which had compiler spills) and without `--cpu-embd`. To remove those differences, **`NVFP4-Mixed` was also re-run on the same system with this same build** (§7). That same-host control is the fair comparison.

> **TL;DR**
> - **Full native 256K (`262,848`) now fits with MTP, and with up to 8 slots.** `q4_1`/`q4_0 + MTP` boot and run at `262,848` with `--slots 1, 4, 8` and **0 MB GTT spill**. `--slots 8 -c 262848 q4_1 + MTP` has **502 MB free** after boot. On the same system, `NVFP4-Mixed` tops out at **212,992** (`s1`), **188,416** (`s4`) and **155,648** (`s8`) for the same config.
> - **MTP decodes a 251,041-token real code prompt at `162.8 tok/s`** (`q4_1`; `q4_0`: `156.5`). Cumulative prefill is `3,006 tok/s`. The earlier Mixed report could only reach `163,840` (`q4_1`) / `180,224` (`q4_0`) with MTP.
> - **Full BetterBench 0.6.1 at `--slots 8 -c 262848 q4_1 + MTP`** (5 runs/category, 32 req/level, prefill ladder up to a 204K-token prompt): weighted single-stream **`157.2 tok/s`**; **`199.0` / `207.6 tok/s` aggregate at `c = 4` / `c = 8`**, 128/128 requests OK; prefill **`3,218 tok/s` at 204,096 tokens**. That is about the same throughput as Mixed's `--slots 8 -c 65536` run (`156.2` / `192.7` / `213.0`) with **4× the context**.
> - **Vision + MTP at full 256K:** `--slots 1 -c 262848 q4_1 + MTP + mmproj` passes **1080p → 1440p → 4K → 2560²**, plus a **248,796-token fill with a 1080p image**, with 257 MB minimum free and 0 GTT rise. `--slots 4 -c 262848 + MTP + Vision` passes up to 1440p and returns a clean `HTTP 400` at 4K. On the same system, Mixed managed `154,624` (`s1`) and `131,072` (`s4`) for the same config.
> - **Speed is unchanged.** On the same system and build, NVFP4-23 matches Mixed: `tg128` is +1.4% (`118.75` vs `117.17 tok/s` at 128K), MTP is within ±2%, and prefill is within 1%. The quant buys **headroom (−1.42 GiB), not speed**. The larger prefill numbers compared with the earlier README (`~3,200` vs `~2,300 tok/s` at 256K) come from the build (0 spills now) and the host, not from NVFP4-23.
> - **q8_0 KV** gains the most context: `256,000` without MTP (same-host Mixed: `166,912`), and `201,728` with MTP (Mixed: `123,904`).

---

## 1. Environment, build, model

| | |
|---|---|
| GPU | AMD Radeon RX 7900 XT, `gfx1100`, 84 CU / 42 WGP, `20,464 MB` GDDR6 (`800 GB/s`, 80 MiB L3) |
| Host / Virtualization | Windows + WDDM; WSL2 kernel `6.18.58.1-microsoft-standard-WSL2`, `/dev/dxg` (`ROCDXG`, no `/dev/kfd`) |
| ROCm / Image | `rocm/dev-ubuntu-24.04:7.2.4` (ROCm 7.2.4, HIP 7.2, AMD clang 22) |
| Idle desktop VRAM | **367 MB used / 20,097 MB free, 40 MB GTT** at start. It dropped to 71 MB used / 2 MB GTT before the control run (WDDM eviction), so the control had ~300 MB *more* free than the NVFP4-23 matrix. |
| nib | `0.1.2` (RDNA4 + dynamic NVFP4 layer loading tree), built with `-DGPU_TARGET=gfx1100 -DCMAKE_EXE_LINKER_FLAGS="-no-pie"` |
| Spill audit | **`AUDIT PASS: 0 SGPR spills, 0 VGPR spills, 0 private segment, 0 stubs (gfx1100)`** across all 19 objects. The earlier report had `1312 VGPR` spills in `kernels_fa.o` and `1286` in `kernels_sparse_global.o` on this same ROCm 7.2.4; those are gone. |
| `NVFP4-23` | `repack_nvfp4 Inkling-2-Mini-Q8_0.gguf … 22` → **16,099,143,008 bytes (exact)**, sha256 `1ed0b242…d7981a9`. 46 tensors converted, mean relRMSE 0.022135. Layers 20–22 show relRMSE ≈ 0.086 against ≈ 0.007–0.021 for layers 0–19. |
| Weights in VRAM | **14.46 GiB** (`14,802.8 MiB` with `--cpu-embd`) against Mixed **15.88 GiB** (`16,260.8 MiB`): **−1,458 MiB** |
| Aux | `mtp.safetensors`, `mmproj-Inkling-2-Mini-Q8_0.gguf`, `draft_shortlist.bin`: SHA-256 match the HF pack |
| `nib selftest` (`q4_1`, sp224, `--mtp`, `--cpu-embd`) | **PASS** (rc 0): sampler χ² tests, `verify(m)` bit-identical for m = 2/4/8, rollback exact, **64/64 greedy tokens identical MTP vs non-MTP** |

**All runs use `--sparse-pages 224 --cpu-embd`.** The 469 MiB pinned host `tok_embd` buffer shows up as ~515 MB of `USED_GTT` at boot. That is a host allocation, not a spill. **GTT spill below means a rise in `USED_GTT` above the post-boot value while requests run** (sampled every 0.5 s, or every 0.2 s in the vision runs).

---

## 2. VRAM ceiling

### Stage 1: raw HIP (`hipMalloc` + `hipMemset`)

- Idle: `20,200.4 MiB` free by `hipMemGetInfo`. The earlier Mixed baseline host had `18,591.2 MiB` free.
- **Hard ceiling: `19,908.0 MiB` (`19.4414 GiB`)** for both `1,245 × 16 MiB` and `512 × 32 MiB + tail`. That is identical to the earlier Mixed host, with 0 fragmentation loss.

### Stage 2: real `nib serve` matrix (`NVFP4-23`)

Method: nib's own VRAM estimator gives the projected free VRAM at two context sizes, from which candidate contexts were picked. Each candidate was then really booted, from the largest down. **PASS** means: boots, `n = slots` concurrent text requests (128 tokens) pass, the 1920×1080 image passes when vision is on, and GTT rises ≤ 8 MB. Every failure was a clean `hipMalloc` OOM at startup or an `HTTP 400` vision guard. There were **0 WDDM stalls and 0 GPU losses**. "Max" is the largest **verified** pass; the true cliff is between it and the next failed candidate (see [`benchmarks.json`](benchmarks.json)).

| KV | `--slots` | No MTP, no Vision | **+ MTP** | + Vision | **+ MTP + Vision** |
|---|--:|---|---|---|---|
| `q4_0` | 1 | **262,848** (17.21 GiB, 2,049 MB free) | **262,848** (17.92 GiB, 1,327 free) | **262,848** (1,740 free) | **262,848** (977 free) |
| `q4_0` | 4 | **262,848** (1,846 free) | **262,848** (1,087 free) | **262,848** (1,509 free) | **262,848** (750 free) |
| `q4_0` | 8 | **262,848** (1,526 free) | **262,848** (766 free) | **262,848** (1,189 free) | **262,848** (429 free) |
| `q4_1` | 1 | **262,848** (17.47 GiB, 1,791 free) | **262,848** (18.18 GiB, 1,067 free) | **262,848** (1,476 free) | **262,848** (712 free) |
| `q4_1` | 4 | **262,848** (1,586 free) | **262,848** (824 free) | **262,848** (1,231 free) | **262,848** (469 free) |
| `q4_1` | 8 | **262,848** (1,263 free) | **262,848** (502 free) | **262,848** (908 free) | 253,952 (275 free); 262,848 boots but 1080p → `HTTP 400` |
| `q8_0` | 1 | 256,000 (117 free); 262,848 OOM | 201,728 (223 free); 212,992 OOM | 225,280 (294 free) | 182,272 (239 free) |
| `q8_0` | 4 | 229,376 (85 free); 241,664 OOM | 176,128 (191 free) | 184,320 (486 free) | 143,360 (401 free) |
| `q8_0` | 8 | 188,416 (117 free); 196,608 OOM | 139,264 (161 free) | 147,456 (456 free) | 106,496 (410 free) |

*Free = `amd-smi` `FREE_VRAM` after boot, in MB. GTT rise during requests was 0–5 MB in every PASS. The 4–5 MB at `--slots 8` comes from static batch-graph capture.*

### NVFP4-23 vs NVFP4-Mixed: maximum verified context

| Config | Mixed, earlier report (WSL2, no `--cpu-embd`) | **Mixed, same host & build** (§7) | **NVFP4-23** |
|---|---|---|---|
| `q4_1`, no MTP, `s1` | 229,376 verified (`262,848` OOM) | **262,848** (627 MB free) | **262,848** (1,791 MB free) |
| `q4_0`, no MTP, `s1` | 262,848 (19.10 GiB) | 262,848 (885 free) | 262,848 (17.21 GiB, 2,049 free) |
| `q8_0`, no MTP, `s1` | 131,072 verified | **166,912** | **256,000** |
| `q4_1` + MTP, `s1` | 163,840 (`196,608` FAIL) | **212,992** (`232,448` FAIL) | **262,848** |
| `q4_0` + MTP, `s1` | 180,224 (`262,848` FAIL) | **237,568** (`258,048` FAIL) | **262,848** |
| `q8_0` + MTP, `s1` | 90,112 | **123,904** (`135,168` FAIL) | **201,728** |
| `q4_1` + MTP, `s4` | 131,072 | **188,416** (`208,896` FAIL) | **262,848** |
| `q4_1` + MTP, `s8` | 65,536 (`131,072` FAIL) | **155,648** (`172,032` FAIL) | **262,848** |
| `q4_1` + MTP + Vision (1080p), `s1` | — | **154,624** (`180,224` → 1080p `HTTP 400`) | **262,848** (and 4K / 2560²) |
| `q4_1` + MTP + Vision (1080p), `s4` | 131,072 (1080p only, 212 MB free) | **131,072** (`155,648` → `HTTP 400`) | **262,848** (and 1440p) |

**Same-host gain: +50K to +107K tokens of context for every MTP configuration, and every `q4_x` MTP / Vision configuration now reaches the full 256K window.** The control's candidate grid is coarse, so Mixed's true cliff lies between the PASS and FAIL values shown.

---

## 3. Non-MTP `nib bench` depth sweeps (`-p 512 -n 128 -r 2`, `--slots 1`)

| Depth | `q4_1` @ 262,848 (17.47 GiB) pp512 / tg128 | `q4_0` @ 262,848 (17.21 GiB) | `q8_0` @ 256,000 (19.16 GiB) | Mixed earlier report `q4_0` @ 262,848 |
|--:|--:|--:|--:|--:|
| 0 | 4,426 / **125.98** | 4,479 / **126.15** | 4,441 / 124.97 | 4,102 / 121.17 |
| 8,192 | 3,940 / 122.19 | 3,963 / 122.64 | 3,937 / 120.49 | 3,074 / 118.12 |
| 32,768 | 3,299 / 119.69 | 3,322 / 120.40 | 3,300 / 117.61 | 2,243 / 116.30 |
| 65,536 | 3,381 / 119.46 | 3,380 / 120.12 | 3,374 / 117.40 | 2,375 / 115.74 |
| 131,072 | 3,296 / 118.80 | 3,309 / 119.54 | 3,311 / 116.67 | 2,385 / 115.07 (128K) |
| 196,608 | 3,251 / 117.90 | 3,254 / 118.43 | 3,229 / 115.69 | — |
| 229,376 | 3,208 / 117.93 | 3,223 / 118.39 | 3,219 / 115.69 | 2,291 / 114.70 (224K) |
| **256,000** | **3,182 / 117.60** | **3,189 / 118.12** | — (ctx 256,000) | 2,278 / 114.20 |

Spread across the `-r 2` repeats: ≤ ±98 tok/s for pp512 and ≤ ±0.10 tok/s for tg128. All three runs had **0 MB GTT rise** (max `USED_GTT` stayed at the 555 MB boot value), and `q8_0` ran the whole sweep with **117 MB free**. *The prefill difference against the earlier Mixed column is mostly the build and host, not the quant; see §7.*

---

## 4. MTP `nib gen --mtp -n 256` sweeps (real code+docs prompts)

The prompts are built the same way as in [`../scripts/bench_mtp.sh`](../scripts/bench_mtp.sh) (README + `engine.cpp` + `server.cpp` + a C++17 `RingBuffer` task), with sizes extended to 251K tokens. One `nib gen` process per row, `--slots 1`, `--cpu-embd`.

| Config (VRAM, free) | Prompt tokens | Cumul. prefill tok/s | **MTP decode tok/s** | tok/round | avg_m | draft / verify ms |
|---|--:|--:|--:|--:|--:|--:|
| **`q4_1` @ 262,848** (17.47 + 0.68 GiB, 1,067 MB free) | 2,759 | 3,521.9 | **177.02** | 4.44 | 5.07 | 1.43 / 22.30 |
| | 38,993 | 3,247.3 | **161.39** | 3.91 | 4.82 | 1.63 / 21.58 |
| | 75,455 | 3,029.3 | **196.26** | 5.02 | 5.24 | 1.64 / 23.24 |
| | 143,078 | 3,041.9 | **173.70** | 4.06 | 4.54 | 1.62 / 21.08 |
| | 183,231 | 3,030.6 | **166.42** | 3.89 | 4.55 | 1.63 / 20.93 |
| | **251,041** | **3,006.4** | **162.83** | 3.88 | 4.58 | 1.62 / 21.34 |
| **`q4_0` @ 262,848** (17.21 + 0.68 GiB, 1,327 MB free) | 2,759 | 3,524.0 | **166.28** | 3.71 | 4.58 | 1.43 / 19.98 |
| | 38,993 | 3,248.6 | **164.79** | 3.86 | 4.65 | 1.62 / 21.11 |
| | 75,455 | 3,031.7 | **189.20** | 4.78 | 5.04 | 1.63 / 22.46 |
| | 143,078 | 3,044.3 | **159.79** | 3.84 | 4.60 | 1.62 / 21.21 |
| | 183,231 | 3,031.2 | **154.38** | 3.66 | 4.59 | 1.63 / 21.10 |
| | **251,041** | **3,008.0** | **156.52** | 3.78 | 4.52 | 1.62 / 21.15 |
| **`q8_0` @ 201,728** (18.27 + 0.68 GiB, 223 MB free) | 2,759 | 3,523.9 | **173.34** | 4.25 | 5.00 | 1.43 / 21.84 |
| | 38,993 | 3,242.7 | **159.44** | 3.70 | 4.42 | 1.62 / 20.80 |
| | 75,455 | 3,025.5 | **152.88** | 3.38 | 4.11 | 1.62 / 19.62 |
| | 143,078 | 3,037.2 | **162.46** | 4.14 | 4.89 | 1.62 / 22.66 |
| | 183,231 | 3,024.1 | **163.88** | 3.94 | 4.66 | 1.63 / 21.69 |

- **`q4_1` and `q4_0` + MTP at 262,848 now work**, including a 251K-token prompt. On Mixed these failed with `DSparkDrafter: hipMalloc failed`. Every row had **0 MB GTT rise**.
- Each number is a single 256-token greedy run, so differences of ±10% between rows track acceptance (tok/round), not depth. I did **not** reproduce the old report's claim that `q8_0` beats `q4_1` by 10–13%: here `q8_0` averaged 162.4 and `q4_1` 175.0 tok/s over the shared prompt sizes. Treat that ordering as unresolved.

---

## 5. Full BetterBench 0.6.1, `--slots 8 -c 262848 --kv-type q4_1 --mtp --no-vision --cpu-embd`

Localhost `127.0.0.1`, `--greedy`, cold nonce prefix cache, **no `--quick`**. Config: `runs_per_category: 5`, warmup 1, `concurrency_levels [1,2,4,8]` × 32 requests, prefill ladder `[2560, 10240, 20480, 40960, 81920, 163840, 261000]`, 2 runs + 1 warmup each. Reports: [`betterbench_slots8_262k_q4_1_mtp.html`](betterbench_slots8_262k_q4_1_mtp.html) · [`betterbench_slots8_262k_q4_1_mtp.json`](betterbench_slots8_262k_q4_1_mtp.json).

`amd-smi`: boot **19,962 MB used / 502 MB free / 559 MB GTT**. During the run (1,443 samples): **min free 502 MB**, max GTT 570 MB (+11 MB, which accompanied the capture of 32 static batch HIP graphs; free VRAM never moved). rc 0, 14 min wall.

### Single-stream decode (8 categories × 5 runs)

| Category | **NVFP4-23 `s8` 262K** | Mixed `s8` 64K local | Mixed `s4` 128K WAN |
|---|--:|--:|--:|
| `json` | **179.1** | 189.3 | 192.9 |
| `file_edit` | **177.2** | 181.5 | 187.5 |
| `summarization` | **176.0** | 166.0 | 178.8 |
| `math` | **169.3** | 172.9 | 179.5 |
| `code` | **160.0** | 157.9 | 171.8 |
| `reasoning` | **143.7** | 139.7 | 163.2 |
| `chat` | **143.1** | 130.5 | 157.4 |
| `prose` | **122.3** | 118.7 | 123.2 |
| **Weighted** | **157.2** (TTFT p50 78 ms, update p99 31.1 ms) | 156.2 (TTFT 83 ms) | 168.2 |

28 of 40 runs stopped at `max_tokens` (thinking phase), as in earlier runs. The CV per category is 4–12%, so category gaps under ~10% are noise at 5 runs.

### Concurrency (true 8-way, 32 requests per level, **128/128 OK**)

| `c` | Aggregate tok/s | Per-stream median | TTFT p50 / p99 (ms) | Mixed `s8` 64K local (agg / med / TTFT) |
|--:|--:|--:|--:|--:|
| 1 | 146.8 | 162.3 | 76.3 / 95.5 | 148.2 / 170.0 / 82.6 |
| 2 | 153.7 | 78.7 | 91.0 / 152.6 | 147.9 / 75.3 / 97.5 |
| 4 | **199.0** | 51.7 | 111.6 / 307.6 | 192.7 / 50.0 / 118.7 |
| 8 | **207.6** | 27.0 | 165.9 / 705.2 | 213.0 / 30.4 / 162.5 |

Same throughput as Mixed's `--slots 8 -c 65536`, but each request can now use up to the full 262,848 window (`slot_ctx=32768` when all 8 are busy, versus 8K/slot before). No WDDM TDR or reset occurred; this host had only 367 MB of desktop VRAM, and the earlier report's TDR warning applied to a host with ~1.1–1.4 GB pinned.

### Prefill ladder

| Target | Prompt tokens | TTFT p50 | **PP tok/s (median)** | Mixed earlier `s4` 128K WAN |
|--:|--:|--:|--:|--:|
| 2,560 | 2,026 | 0.56 s | 3,619.1 | 1,073 (WAN RTT) |
| 10,240 | 8,017 | 2.14 s | 3,748.4 | 1,937 |
| 20,480 | 16,088 | 4.35 s | 3,700.0 | 1,709 |
| 40,960 | 32,065 | 9.07 s | 3,536.1 | 2,199 |
| 81,920 | 64,170 | 19.87 s | 3,229.9 | 1,851 |
| 163,840 | 128,290 | 39.66 s | 3,234.6 | — |
| 261,000 | **204,096** | 63.42 s | **3,217.9** | — |

*BetterBench's 261,000-token target produced a 204,096-token prompt by the server's tokenizer, so the deepest measured point is 204K. The 251K-token point is in §4.*

---

## 6. Vision (`--mmproj`) matrix

The client is [`scripts/vis_client.py`](scripts/vis_client.py). It sends a text request, then 1080p / 1440p / 4K / 2560² images (512 tokens each), fills the context to `ctx − 8K` with text plus a 1080p image (256 tokens), and finishes with a post-fill 4K image. `amd-smi` is sampled every 200 ms. Decode tok/s is measured client-side as (completion − 1) / stream time.

| Config (`--cpu-embd`) | Boot free | text | 1080p | 1440p | 4K | 2560² | Fill (prompt tok → decode) | After-fill 4K | Min free | GTT rise |
|---|--:|--:|--:|--:|--:|--:|---|--:|--:|--:|
| **`q4_1` 262,848 `s1` + MTP** | 712 MB | 121.6 | **147.1** | 128.4 | **134.9** | **134.5** | **248,796** in 79.4 s → 107.6 | **137.2** | 257 MB | **0** |
| `q4_1` 262,848 `s1`, no MTP | 1,475 MB | 123.5 | 121.9 | 121.4 | 121.1 | 121.1 | 248,796 in 76.8 s → 115.4 | 121.4 | 1,020 MB | **+32 MB on 1440p only** |
| **`q4_0` 262,848 `s1` + MTP** | 977 MB | 125.7 | 133.7 | 132.0 | **138.8** | 129.9 | 248,796 in 79.4 s → **141.8** | 123.1 | 522 MB | **0** |
| `q4_1` 262,848 `s4` + MTP | 470 MB | 126.6 | **154.0** | 131.4 | `HTTP 400` | `HTTP 400` | 248,796 in 79.4 s → 124.6 | `HTTP 400` | 211 MB | **+42 MB on 1440p only** |
| `q4_1` 131,072 `s4` + MTP | 1,854 MB | 130.8 | 129.6 | 134.3 | 130.8 | 128.9 | 120,761 in 38.2 s → 125.1 | 148.1 | 1,400 MB | **+32 MB on 1440p only** |

- **`HTTP 400` is the intended guard, not a crash.** The serve log shows `[vision] hipMalloc(ViT workspace, 414.4 MiB) failed (503.2 MiB free)`. The server stayed healthy, and later text and 1080p requests passed. The ViT workspace peaks at 131.7 / 234.0 / 414.4 / 416.0 MiB for 1080p / 1440p / 4K / 2560² and goes back to 0 after each encode.
- **⚠ GTT flag:** in 3 of 5 configs the **1440p** request alone raised `USED_GTT` by **+32 to +42 MB**, which dropped back afterwards. It also happened with 1.4–1.8 GB of VRAM free (`q4_1` 131K `s4`; `q4_1` 262K no-MTP), so it does not look like VRAM-pressure spill. It is more likely a transient host-side staging allocation for that image size. **Cause not verified.** The 1440p decode in those configs (121–134 tok/s) shows no slowdown.
- Compared with Mixed (earlier report, §7.2): `--slots 4 -c 131072 + MTP + Vision` used to be 1080p-only, with 212 MB free. With NVFP4-23 the same config has **1,854 MB free** and passes **4K and 2560²**. The full-256K vision configs had never run with MTP on Mixed.

---

## 7. Same-host, same-build control: `NVFP4-Mixed` vs `NVFP4-23`

`Inkling-2-Mini-NVFP4-Mixed.gguf` (17,627,963,360 bytes, sha256 `304d4fe3…` matches HF) was run on the same host with the same binary and the same `--cpu-embd`. This removes host, build, and `--cpu-embd` differences from the comparison. Ceiling results are in the §2 comparison table.

### Throughput: `q4_1` @ 131,072, `--slots 1`

| Metric | NVFP4-Mixed | **NVFP4-23** | Δ |
|---|--:|--:|--:|
| Engine `total_vram` | 17.60 GiB | **16.18 GiB** | **−1.42 GiB** |
| `pp512` @ 0 / 32K / 64K / 128K | 4,407 / 3,294 / 3,359 / 3,299 | 4,457 / 3,310 / 3,379 / 3,314 | +0.5 to +1.1% |
| `tg128` @ 0 / 32K / 64K / 128K | 124.08 / 117.91 / 117.70 / 117.17 | **125.78 / 119.68 / 119.36 / 118.75** | **+1.4 to +1.5%** |
| `gen --mtp` 2,759 tok: prefill / decode | 3,491.3 / 178.82 (4.44 tok/rd) | 3,516.7 / 175.51 (4.44 tok/rd) | +0.7% / −1.9% |
| `gen --mtp` 75,455 tok: prefill / decode | 3,012.0 / 191.16 (5.00 tok/rd) | 3,030.8 / 192.72 (5.02 tok/rd) | +0.6% / +0.8% |

- NVFP4-23 decodes **~1.4% faster** without MTP (fewer bytes to read in layers 20–22); prefill and MTP are equal within noise. MTP acceptance is identical (4.44 / 5.0 tok/round), which suggests the extra quantisation in layers 20–22 does not hurt greedy draft agreement on these prompts. (40×512 WikiText-2 KLD/PPL is reported in root [`README.md`](../../../README.md).)
- The 143,078-token prompt exceeds `-c 131072` and was omitted from both models in the 131K control.

---

## 8. Recommended RX 7900 XT configs with NVFP4-23

| Goal | `nib serve` flags (all with `--sparse-pages 224 --cpu-embd`) | App VRAM | Measured |
|---|---|--:|---|
| **Full 256K + MTP + 8-way API (text)** | `--slots 8 -c 262848 --kv-type q4_1 --mtp models/mtp.safetensors --no-vision` | 18.71 GiB (502 MB free) | BB weighted 157.2, 207.6 agg @ c=8, 0 GTT |
| **Full 256K + MTP + Vision up to 4K (solo)** | `--slots 1 -c 262848 --kv-type q4_1 --mtp … --mmproj …` | 18.50 GiB (712 MB free) | 4K 134.9 tok/s, 248K fill + image OK |
| **Full 256K + MTP + Vision, 4 slots (≤1440p)** | `--slots 4 -c 262848 --kv-type q4_1 --mtp … --mmproj …` | 18.73 GiB (470 MB free) | 1080p 154.0, 1440p 131.4; 4K → clean `HTTP 400` |
| **4-slot Vision up to 4K + MTP, 128K** | `--slots 4 -c 131072 --kv-type q4_1 --mtp … --mmproj …` | ~17.44 GiB (1,854 MB free) | all sizes pass, 129–148 tok/s |
| **Most headroom at 256K** | `--kv-type q4_0` instead of `q4_1` (−0.26 GiB) | — | `s8` + MTP + Vision at 262,848 passes 1080p |
| **Highest-precision KV** | `--slots 1 -c 201728 --kv-type q8_0 --mtp …` | 18.97 GiB (223 MB free) | 152.9–173.3 tok/s MTP up to 183K prompts |

These margins were measured on a host whose desktop held only 71–367 MB of VRAM. On Windows hosts that keep 1–1.8 GB of desktop VRAM pinned, subtract that much free VRAM. The `s8 262K + MTP` config (502 MB free) would then OOM or page, so drop to `q4_0`, `--slots 4`, or ~196K.

---

## Folder contents

| Path | What |
|---|---|
| [`benchmarks.json`](benchmarks.json) | Machine-readable dataset: environment/build/model, Stage 1, all 63 Stage 2 probes (36 configs) + per-config max, nonmtp rows, MTP rows, BB config/smi, vision per-request rows, same-host control |
| [`betterbench_slots8_262k_q4_1_mtp.json`](betterbench_slots8_262k_q4_1_mtp.json) / [`.html`](betterbench_slots8_262k_q4_1_mtp.html) | Full BetterBench 0.6.1 report |
| [`scripts/`](scripts/) | Benchmark scripts: `setup.sh`, `vram_probe.sh`, `project.sh` + `ceiling.sh` + `probe_client.py`, `bench_nonmtp.sh`, `bench_mtp.sh`, `bench_bb.sh`, `bench_vision.sh` + `vis_client.py`, `pipeline.sh`, `control.sh` |
