# Nib on AMD RDNA 4 (`gfx1201`) — RX 9070 XT Silicon Verification & Radeon AI PRO R9700 (32 GB) Projection

This document summarizes **verified real-silicon RDNA 4 (`gfx1201`) support** on the **AMD Radeon RX 9070 XT (16 GB)**, explains the **16 GB VRAM capacity ceiling**, and provides a **theoretical performance & capacity projection** for the **AMD Radeon AI PRO R9700 (32 GB)** — which shares the exact same Navi 48 (`gfx1201`) silicon with double the VRAM.

> **Measurement vs. Projection Notice:** Section 1 and Section 2 report **real-silicon measurements** (RX 9070 XT under Windows/WSL2 ROCDXG, ROCm 7.2.4, and 40×512 WikiText-2 KLD on RX 7900 XTX). Section 3 (**Radeon AI PRO R9700**) is an **unmeasured theoretical projection** extrapolated from measured `gfx1201` silicon behavior and RDNA 3 MTP scaling; it will be replaced with real measurements once R9700 hardware is tested.

---

## 1. Verified RDNA 4 (`gfx1201`) Silicon Support — RX 9070 XT (16 GB)

Full scripts, `benchmarks.json`, and BetterBench `0.6.1` HTML/JSON reports live in [`nvfp4_23_all_2026-11-26/README.md`](nvfp4_23_all_2026-11-26/README.md).

### Build & Correctness (`-DGPU_TARGET=gfx1201`)
- **Spill audit (`scripts/spill_audit.sh`):** **PASS — `0` SGPR spills, `0` VGPR spills, `0` private segment, `0` stubs** across all **19** HIP objects on ROCm 7.2.4 (AMD clang 22).
- **Co-residency & HIP Graph:** Verified `wgp=32` occupancy guards (`wo_b=2 wo_c=2 wo_c_h2=3 mix_rms_qkv=4 a_qkv=4`) and **404-node static decode HIP graph**.
- **Parity (`nib selftest`):** **PASS** on `q8_0` KV (`cos_sim = 0.999991`, `4/4` sampler tests pass, exact repack parity `max_abs_err = 0.000000`) for both `NVFP4-23` and `NVFP4-All`.

### Measured `NVFP4-All` vs. `NVFP4-23` on RX 9070 XT (16 GB)

| Metric | **`NVFP4-All`** (`L0–23` NVFP4) | **`NVFP4-23`** (`L0–22` NVFP4, `L23` Q8_0) |
|---|---:|---:|
| GGUF File Size | `15.59 GB` (`14.52 GiB`) | `16.10 GB` (`14.99 GiB`) |
| Weights in VRAM (`--cpu-embd` ON) | **`13.98 GiB`** | `14.46 GiB` |
| 40×512 WikiText-2 Mean KLD vs. `Q8_0` | **`0.00529 ± 0.00016`** (`96.98%` top-1) | `0.00483 ± 0.00016` (`97.10%` top-1) |
| Max Context, `--slots 1` (`q4_0` / `q4_1` / `q8_0`) | **`73,728`** / **`65,536`** / **`32,768`** | `24,576` / `16,384` / `8,192` |
| Max Multi-Slot Pool (`q4_0`) | **`--slots 4` @ `49,152`** (`s8` @ `8,192`) | `--slots 2` @ `16,384` |
| Non-MTP `nib bench` Decode (`tg128`, d0 → max) | `96.24 → 92.22 tok/s` (`@ 72.7K`) | `95.99 → 92.24 tok/s` (`@ 23.6K`) |
| Non-MTP `nib bench` Prefill (`pp512`, d0) | `5,019–5,092 tok/s` | `5,036–5,137 tok/s` |
| Full BetterBench `0.6.1` Single-Stream Weighted | `95.7` / `94.5 tok/s` | `94.2` / `95.2 tok/s` |
| Full BetterBench Concurrency (`c = 1 / 2 / 4` agg) | **`92.7` / `126.8` / `162.0 tok/s`** | `92.4` / `126.1` / `OOM` |
| Full BetterBench Prefill at Deepest Prompt | **`4,006 tok/s`** (`70,026` real tokens) | `4,392 tok/s` (`23,252` real tokens) |
| DSpark MTP (`--mtp`) | `✗` OOM (`DSparkDrafter: hipMalloc failed`) | `✗` OOM |
| Vision (`--mmproj`) | **Up to `1440p` @ `≤ 8K`** (clean `HTTP 400` at 4K) | `✗` ViT fails to load even at `4K` |

**Takeaway for 16 GB cards:** Quantizing `blk.23` (`NVFP4-23` → `NVFP4-All`) costs only **`+0.00046` Mean KLD** (`-0.12%` top-1, `+0.0034` PPL) at identical speed, while freeing `0.48 GiB` VRAM — delivering **3–4× the context**, **2× the slots**, and the only working Vision configuration on 16 GB.

---

## 2. Why 16 GB Cards Top Out Below 128K–262K (and How to Push Higher)

Under Windows/WSL2 (`ROCDXG`), the 16 GB RX 9070 XT reports `15.81 GiB` (`16,188.5 MiB`) via `hipGetDeviceProperties` and reaches a hard raw `hipMalloc` + `hipMemset` ceiling of **`15.30 GiB`** (`15,672 MiB`) after WDDM desktop eviction:

| VRAM Budget Component | `NVFP4-All` | `NVFP4-23` |
|---|---:|---:|
| Hard usable `hipMalloc` ceiling (WSL2/WDDM) | **`15.30 GiB`** | **`15.30 GiB`** |
| Model weights in VRAM (`--cpu-embd` ON) | `-13.98 GiB` | `-14.46 GiB` |
| Fixed compile-time prefill scratch (`MAX_UBATCH`) | `-0.46 GiB` | `-0.46 GiB` |
| HIP runtime + code objects + HIP graph metadata | `-0.22 GiB` | `-0.22 GiB` |
| **Net VRAM available for KV cache** | **`~0.64 GiB`** | **`~0.16 GiB`** |

At Nib's measured marginal KV footprint (**`~0.29 GiB / 32K`** for `q4_0`, **`~0.32 GiB / 32K`** for `q4_1`, **`~0.54 GiB / 32K`** for `q8_0`):
- `~0.64 GiB` yields **`73,728`** (`q4_0`), `65,536` (`q4_1`), or `32,768` (`q8_0`).
- Reaching **`262,848`** (`q4_0`) requires **`~2.38 GiB`** of KV alone (`~16.8+ GiB` total), physically exceeding 16 GB even on native Linux.
- **Levers to extend 16 GB capacity:**
  1. **Native Linux** (reclaiming `~0.5 GiB` of WDDM overhead) should stretch `NVFP4-All` `q4_0` from `72K` to **`~128K`**.
  2. **Reducing compile-time `MAX_UBATCH` prefill scratch** (`0.46 GiB` fixed) by `~250–300 MiB` would add `~32K` of `q4_0` KV — or open enough headroom on `NVFP4-All` to initialize DSpark MTP (`~697 MiB`) at short context.
  3. **Further weight offloading/quantization** (e.g., `lm_head` or attention weights) is required for `262K` on 16 GB.

---

## 3. Theoretical Projection — AMD Radeon AI PRO R9700 (32 GB, `gfx1201`)

### Silicon Comparison: RX 9070 XT vs. Radeon AI PRO R9700

| Specification | RX 9070 XT (16 GB, measured) | Radeon AI PRO R9700 (32 GB, projected) | Impact on Nib |
|---|---|---|---|
| **Die / ROCm ISA Target** | Navi 48 / **`gfx1201`** | Navi 48 XTW / **`gfx1201`** | **Drop-in binary compatibility** (zero code changes; same 0-spill build) |
| **Compute Units / WGPs** | `64 CU` / `32 WGP` | `64 CU` / `32 WGP` | Identical co-residency guards (`wgp=32`) and compute throughput |
| **Boost / Game Clock** | `2,970` / `2,400 MHz` | `2,920` / `2,350 MHz` | `-1.7%` peak boost; dual-slot blower may settle `2–4%` lower under sustained load |
| **VRAM Capacity** | `16 GB` GDDR6 | **`32 GB` GDDR6** *(Linux ECC optional)* | **2× VRAM** — removes every 16 GB context, MTP, Vision, and slot bottleneck |
| **Memory Bus & Bandwidth** | `256-bit`, `20 Gbps` (**`640 GB/s`**) | `256-bit`, `20 Gbps` (**`640 GB/s`**) | Identical bandwidth-bound decode speed (with ECC off) |
| **Infinity Cache (L3)** | `64 MB` | `64 MB` | Identical L3 hit behavior |
| **AI Matrix Peak (Dense)** | `191 TFLOPS` FP16 · `383 TFLOPS` FP8 | `191 TFLOPS` FP16 · `383 TFLOPS` FP8 | Identical WMMA prefill compute |
| **TBP / Cooling** | `304 W` open-air | `300 W` dual-slot active blower | Designed for multi-GPU workstations |

### Projected VRAM Budget on 32 GB (Native Linux, ECC Off)

Because the R9700 exceeds Nib's `22 GiB` threshold, `--cpu-embd` stays **OFF** (keeping `token_embd` in VRAM with `0 MB` host GTT mapping):

| Component (`--slots 8 -c 262848 --kv-type q8_0 --mtp --mmproj --sparse-pages 224`) | Estimated VRAM |
|---|---:|
| **`Inkling-2-Mini-NVFP4-Mixed.gguf`** weights (incl. `token_embd` in VRAM, `0.0033` KLD) | `~16.42 GiB` |
| `q8_0` KV cache at full **`262,848`** context | `~4.32 GiB` |
| Engine scratch + `--slots 8` batched state + HIP graph metadata | `~1.25 GiB` |
| DSpark MTP drafter (`mtp.safetensors` + shortlist + verify buffers) | `~0.70 GiB` |
| Vision encoder (`mmproj-Q8_0` weights `0.32 GiB` + transient `4K` workspace `0.42 GiB`) | `~0.74 GiB` peak (`0.33 GiB` steady) |
| **Total Peak Footprint** | **`~23.4 GiB`** (**`~8.0–8.5 GiB` free!**) |

With `~8 GiB` still free at full `NVFP4-Mixed` + `262K q8_0` + MTP + 4K Vision + `--slots 8`, the R9700 beats even a 24 GB RX 7900 XTX on headroom (allowing higher concurrency such as `--slots 16` or full `Q8_0` weights at medium context).

### Projected Performance (Theoretical Estimates)

Anchored directly to our measured `gfx1201` (RX 9070 XT) bandwidth/WMMA rates and RDNA 3 (`gfx1100`) MTP scaling ratios (`~1.38–1.48×` solo decode speedup with `q8_0` KV):

| Benchmark / Workload | RX 9070 XT (16 GB, measured) | **Radeon AI PRO R9700 (32 GB, est.)** | Basis / Scaling Notes |
|---|---:|---:|---|
| **Recommended Quant** | `NVFP4-All` (`14.52 GiB`) | **`NVFP4-Mixed`** (`16.42 GiB`) | No need to quantize `L20–23` on 32 GB |
| **Max Full-Feature Config** | `72K q4_0`, no MTP/Vision (`s1`) | **`262,848 q8_0` + MTP + 4K Vision (`s8`)** | `~23.4 GiB` peak vs. `32 GB` VRAM |
| **Non-MTP Decode (`tg128`, `d0 → 256K`)** | `96.2 → 92.2 tok/s` (`@ 72.7K`) | **`~95–96 → 92 tok/s`** | Memory-bandwidth-bound on identical `640 GB/s` bus (`+0.5–1%` possible from VRAM `token_embd`) |
| **MTP Solo Decode (BetterBench weighted)** | `✗` (OOM) | **`~130–142 tok/s`** | `~1.38–1.48×` over `95 tok/s` base; `q8_0` KV maintains high acceptance (`~4.2–4.7 tok/round`); `json`/`code` peaks `~155–165 tok/s`, `prose` `~100–110 tok/s` |
| **Deep-Context MTP (`256K` prompt)** | `✗` (OOM) | **`~125–138 tok/s`** | Sparse-224 keeps attention bandwidth flat from `16K → 256K` |
| **Prefill (`pp512` at d0)** | `5,019–5,137 tok/s` | **`~4,950–5,100 tok/s`** | Compute-bound on identical 64-CU / 191 TFLOPS FP16 WMMA (`-1.7%` boost clock delta) |
| **Long-Context Prefill (`64K → 256K`)** | `4,006 tok/s` (`@ 70K`) | **`~3,700–4,000 tok/s`** | **~15–25% faster than RX 7900 XT** (`~3,220 tok/s` @ `204K`) and close to RX 7900 XTX |
| **Concurrency (`c = 1 / 2 / 4 / 8` agg)** | `92.7 / 126.8 / 162.0 / 179.5*` *(no MTP)* | **`~130–140 / ~128 / ~162 / ~180–192 tok/s`** | `c = 1` jumps ~45% from MTP; `c = 2..8` tracks 9070 XT scaling now across full `262K` context |

*\*RX 9070 XT `c = 8` measured at `179.5 tok/s` during burst probe at `-c 8192`.*

### Caveats & Future RDNA 4 Upside
1. **ECC overhead (Linux):** Turning GPU ECC **on** reduces usable VRAM slightly and costs roughly **`~5–8%`** effective memory bandwidth on decode-heavy workloads; the projections above assume ECC **off** (matching consumer cards).
2. **Blower thermals:** Under multi-hour sustained prefill or `c = 4–8` batched serving, the dual-slot `300 W` blower may clock `~2–4%` below an open-air triple-fan RX 9070 XT.
3. **Untapped native FP8 WMMA (`383 TFLOPS`):** Nib's current RDNA 4 prefill path uses **FP16 WMMA (`191 TFLOPS`)** via `include/nib_arch.h`. Because Navi 48 (`gfx1201`) supports native **`2×`-rate FP8 WMMA (`383 TFLOPS` dense)**, a future FP8 prefill kernel on RDNA 4 could push both RX 9070 XT and R9700 prefill substantially higher.
