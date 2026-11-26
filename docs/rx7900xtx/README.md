# AMD Radeon RX 7900 XTX (24 GB) — Real-Hardware Nib Evaluation & Vision-Fix Verification

Real-silicon characterization and baseline-vs-patched A/B verification of **Nib** running **Inkling 2 Mini** (`Inkling-2-Mini-NVFP4-Mixed.gguf`, `16.34 GiB` weights) on **AMD Radeon RX 7900 XTX** (`gfx1100` / Navi 31, `96 CU` / `48 WGP`, `24 GB` GDDR6 @ `960 GB/s`, `96 MiB` L3 Infinity Cache), tested on both native Linux (ROCm 7.16) and Windows/WSL2 ROCDXG (`rocm/dev-ubuntu-24.04:7.2.4`) on **2026-11-26**.

> **TL;DR**
> - **Zero regression on the 24 GB RX 7900 XTX hot path:** `< 22 GiB` auto-defaults are a complete no-op on 24 GB GPUs (`kv=q8_0`, `ctx=262144`, `cpu_embd=False`, identical `296`-node non-MTP / `300`-node MTP decode HIP graphs). `nib bench` matches baseline within `0.1%` (`141.99 → 141.80 tok/s` non-MTP; `233.13 → 232.18 tok/s` MTP, `verify = 13.21 → 13.22 ms`).
> - **BetterBench (`0.6.1`, `--greedy`, `5 passes/cat`, `c = 1, 2, 4, 8 × 32 reqs`, localhost zero-RTT, `--slots 8 -c 196608 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj`):** **`191.4 tok/s`** weighted solo (`TTFT p50: 70.0 ms`, `update p99: 22.7 ms`), **`224.7 tok/s` agg** at `c = 4` (`58.7 tok/s`/stream), and **`264.0 tok/s` agg** at `c = 8` (`35.2 tok/s`/stream, `168/168 OK`, `34` static batch HIP graphs captured) — matching the native Linux flagship baseline (`186.8` solo, `223.9` `c = 4`, `268.0` `c = 8`).
> - **Aliased single-arena ViT scratch + post-encode release is a major win even on 24 GB XTX:**
>   - **Bit-identical** embeddings (`bit_diffs = 0`, `max_abs_diff = 0.0`) across `2,985,984` floats at `1080p` and `9,400,320` floats at `4K 3840×2160`.
>   - **Eliminated a real `1440p`/`4K` OOM crash on 24 GB XTX** at flagship `256K q8_0 + MTP + Vision` (`build_base` crashed with `Connection refused` on `1440p`; patched `build` passed `1080p → 1440p → 4K → 2560×2560` max-res at `209–230 tok/s` with `cpu_embd=False`).
>   - **Saved `-939 MiB` post-`4K` resident Vision VRAM** (`1.274 → 0.357 GiB`), **`-544` to `-609 MB` peak VRAM**, and trimmed **`11–34 ms` off Vision TTFT** (`1080p`–`4K`).
> - **Live post-concurrency Vision verified on `--slots 8`:** Running a `1080p` image request immediately after BetterBench `c = 1..8` (with all `34` static batch HIP graphs still resident) decoded at **`221.7 tok/s`** (`618.9 ms` TTFT) and left only **`+12 MB`** VRAM delta (`0 MB` GTT delta, `1,105 MB` still free).

---

## Folder Contents

| File | Description |
|---|---|
| [`README.md`](README.md) | This curated report (environments, A/B zero-regression, bit-exact Vision parity, `256K`/`128K` Vision A/B, BetterBench concurrency, recommended configs) |
| [`benchmarks.json`](benchmarks.json) | Machine-readable dataset of all RX 7900 XTX baseline-vs-patched A/B benchmarks, `--dump-embeds` parity, flagship Vision A/B, and BetterBench summaries |
| [`betterbench_slots8_196k_visionfix.json`](betterbench_slots8_196k_visionfix.json) | Zero-WAN-RTT localhost BetterBench JSON for patched Vision-fixed build (`--slots 8 -c 196608 q8_0 + MTP + Vision`, `5 passes/cat`, `c = 1, 2, 4, 8 × 32 reqs`) |
| [`betterbench_slots8_196k_visionfix.html`](betterbench_slots8_196k_visionfix.html) | Interactive BetterBench HTML report for the patched Vision-fixed build |
| [`../betterbench_slots8_262k.json`](../betterbench_slots8_262k.json) | Native Linux flagship BetterBench JSON (`--slots 8 -c 262848 q8_0 + MTP + Vision`, reference baseline) |
| [`../betterbench_slots8_262k.html`](../betterbench_slots8_262k.html) | Native Linux flagship BetterBench HTML report (reference baseline) |
| [`scripts/bench_ab_vision.sh`](scripts/bench_ab_vision.sh) | Reproducible Baseline vs Patched A/B suite (`nib bench` non-MTP & MTP, `--dump-embeds` bit parity at `1080p` & `4K`, and `128K`/`256K q8_0` Vision serve A/B) |
| [`scripts/bench_concurrency_local.sh`](scripts/bench_concurrency_local.sh) | Reproducible localhost BetterBench (`5 passes/cat`, `c = 1, 2, 4, 8 × 32 reqs`) + live post-concurrency `1080p` Vision smoke script |

---

## 1. Test Environments & Platform Notes

| Layer | Native Linux Flagship Baseline | Windows/WSL2 ROCDXG A/B Host |
|---|---|---|
| **GPU** | AMD Radeon RX 7900 XTX (`gfx1100`, `96 CU` / `48 WGP`, `384-bit`, `960 GB/s`, `96 MiB` L3) | AMD Radeon RX 7900 XTX (`gfx1100`, `96 CU` / `48 WGP`, `384-bit`, `960 GB/s`, `96 MiB` L3) |
| **Physical VRAM** | `24,560 MB` (`23.94 GiB` usable) | `24,560 MB` (`24,387 MB` free when idle; `~23.0 GiB` single-process WSL2 WDDM DXG cap) |
| **OS / Virtualization** | Native Linux (`/dev/kfd`) | Windows WDDM + WSL2 (`6.18.58.1-microsoft-standard-WSL2`), `/dev/dxg` (`ROCDXG`) |
| **ROCm / Compiler** | ROCm 7.16 (clang 24, **0 spills** across all 18 objects) | `rocm/dev-ubuntu-24.04:7.2.4` (clang 22; `-DCMAKE_EXE_LINKER_FLAGS="-no-pie"`, 0 decode spills) |

### Key WSL2 vs Native Linux Finding on 24 GB RX 7900 XTX
- On **native Linux**, `--slots 8 -c 262848 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj` (`22.06 GiB` LLM + `0.68 GiB` MTP + `0.32 GiB` Vision = `23.08 GiB` app + `~0.42 GiB` HIP runtime) boots and runs with full `262K` context.
- On **Windows/WSL2 (ROCDXG)**, WDDM caps a single WSL2 process around **`~23.0 GiB`**, causing `--slots 8 -c 262848 q8_0 + MTP + Vision` (with `cpu_embd=False`) to hit `HIP error 2 (out of memory)` at `vision_encoder.cpp:314`.
- Windows/WSL2 XTX users have two clean options:
  1. **`--slots 8 -c 196608 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj`** (`22.00 GiB` app, `cpu_embd=False`, **`1.1 GiB` free** after 34 batch HIP graphs + `1080p` Vision), or
  2. **`--slots 8 -c 262848 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj --cpu-embd`** (frees `469.3 MiB` of VRAM at `< 0.8%` plain decode cost to clear WDDM's `~23.0 GiB` ceiling).

---

## 2. Zero-Regression A/B Verification (`build_base/nib` vs `build/nib`)

Both binaries were compiled from the exact same checkout (`build_base/nib` unpatched vs `build/nib` with all 9 host-only C++ changes: aliased single-arena ViT scratch + post-encode release, `--cpu-embd`, `< 22 GiB` auto defaults, accurate VRAM accounting) and benchmarked back-to-back on the same Windows/WSL2 ROCDXG system.

### 24 GB Default Policy Check
Launching `build/nib serve -m ... --mtp ... --mmproj ...` with **no** `-c`, `--kv-type`, or `--cpu-embd` flags left 24 GB defaults completely untouched:
```text
[nib] Config: ctx=262144 slots=4(65536/slot, req_max=262144, elastic) kv=q8_0 sparse_pages=0 temp=0.60 mtp=ON vision=ON vram=22.49 GiB
[nib] Device VRAM: 23.35 / 23.94 GiB used (0.59 GiB free; app=22.49 GiB: llm=21.49 + mtp=0.68 + vision=0.32)
```
`cpu_embd` stayed **`false`**, `kv` stayed **`q8_0`**, and `ctx` stayed **`262144`**.

### `nib bench` A/B (`-c 16384 --kv-type q8_0`, depth 0, 48 WGPs)

| Benchmark | Baseline (`build_base/nib`) | Patched (`build/nib`) | Delta |
|---|---:|---:|---:|
| Non-MTP `pp512` | `4,805.21 ± 129.97 tok/s` | `4,799.67 ± 147.47 tok/s` | `-0.1%` (within noise) |
| Non-MTP `tg128` (`296` static HIP graph nodes) | `141.99 ± 0.23 tok/s` | `141.80 ± 0.18 tok/s` | `-0.1%` (within noise) |
| MTP `tg128` (`300` decode / `690..744` verify nodes) | `233.13 tok/s` (`avg_acc=2.46`, `avg_m=3.78`, `draft=1.16 ms`, `verify=13.21 ms`) | `232.18 tok/s` (`avg_acc=2.46`, `avg_m=3.78`, `draft=1.16 ms`, `verify=13.22 ms`) | `+0.01 ms` verify |

Local native ROCm 7.16 `./scripts/spill_audit.sh` confirmed **0 SGPR spills, 0 VGPR spills, 0 private segment across all 18 objects (`gfx1100`)**.

---

## 3. Vision Single-Arena Aliasing & Post-Encode Release A/B

### A. Bit-Exact `--dump-embeds` Parity (`build_base/nib` vs `build/nib`)

| Resolution | Output Shape | Total Floats | `bit_diffs` | `max_abs_diff` | Peak VRAM (`BASE → NEW`) |
|---|---|---:|---:|---:|---:|
| **`1920×1080` (`1080p`)** | `[1296, 2304]` | `2,985,984` | **`0`** | **`0.0`** | `20,259 → 19,994 MB` (`-265 MB`) |
| **`3840×2160` (`4K`)** | `[4080, 2304]` | `9,400,320` | **`0`** | **`0.0`** | `20,911 → 20,302 MB` (**`-609 MB`**) |

### B. Flagship `256K q8_0` Serve A/B (`--slots 4 -c 262144 --kv-type q8_0 --mtp --mmproj`, `cpu_embd=False`)

| Step | Baseline (`build_base/nib`, `sp224`) | Patched (`build/nib`, default `262144 q8_0`) |
|---|---|---|
| Boot (`H0`) | `vram=21.88 GiB` *(omitted `0.68 GiB` MTP)* | `vram=22.49 GiB` (`llm=21.49 + mtp=0.68 + vis=0.32`), `free=0.592 GiB`, `cpu_embd=False` |
| **`1080p`** | PASS (`585.7 ms` / `205.8 tok/s`), Vision held **`0.623 GiB`**, peak `24,222 MB` (`338 MB` free) | **PASS** (`208.8 tok/s`), `ws=0 MiB` (`peak=131.65 MiB`), Vision held **`0.333 GiB`** |
| **`1440p`** | **OOM CRASH (`Connection refused`)** | **PASS** (`1,114.8 ms` / `230.3 tok/s`), `ws=0 MiB` (`peak=234.03 MiB`), `563 MB` min free |
| **`4K 3840×2160`** | *(dead)* | **PASS** (`2,360.5 ms` / `219.5 tok/s`), `ws=0 MiB` (`peak=414.40 MiB`), `280 MB` min free |
| **`2560×2560` max (`4,096` img toks)** | *(dead)* | **PASS** (`2,259.4 ms` / `222.8 tok/s`), `ws=0 MiB` (`peak=416.03 MiB`), `278 MB` min free |
| **`TEXT_AFTER`** | *(dead)* | **PASS** (`64.0 ms` / `163.0 tok/s`), **`696 MB` free** |

### C. Flagship `128K q8_0` Serve A/B (`--slots 4 -c 131072 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj`, `cpu_embd=False`)

| Metric | Baseline (`build_base/nib`) | Patched (`build/nib`) | Net Change |
|---|---:|---:|---:|
| Post-`4K` Vision VRAM held | `1.274 GiB` | **`0.357 GiB`** (`ws = 0 MiB`) | **`-939 MiB`** |
| Peak VRAM during `4K` / `2560×2560` | `22,672 MB` | **`22,128 MB`** | **`-544 MB`** |
| Free VRAM after `4K` (`TEXT_AFTER`) | `1,888 MB` | **`2,851 MB`** | **`+963 MB` free** |
| `1080p` TTFT / Decode | `587.4 ms` / `204.7 tok/s` | **`576.0 ms` / `205.9 tok/s`** | **`-11.4 ms`** |
| `1440p` TTFT / Decode | `1,117.7 ms` / `230.7 tok/s` | **`1,100.8 ms` / `230.0 tok/s`** | **`-16.9 ms`** |
| `4K 3840×2160` TTFT / Decode | `2,361.2 ms` / `223.2 tok/s` | **`2,327.5 ms` / `222.4 tok/s`** | **`-33.7 ms`** |
| `2560×2560` Max TTFT / Decode | `2,275.6 ms` / `226.6 tok/s` | **`2,253.1 ms` / `224.5 tok/s`** | **`-22.5 ms`** |
| `TEXT_AFTER` Decode | `152.7 tok/s` | **`164.9 tok/s`** | **`+12.2 tok/s`** |

---

## 4. BetterBench Single-Stream & Parallel Concurrency (`--slots 8`, Vision-Fixed)

Measured with **BetterBench `0.6.1`** (`corpus v1.0`, `--greedy`, cold nonce, `5 passes/cat`, `c = 1, 2, 4, 8 × 32 reqs`, localhost `127.0.0.1:4444/v1`) on patched `build/nib` (`--slots 8 -c 196608 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj`, `cpu_embd=False`, `22.00 GiB` app VRAM, `34` static batch HIP graphs captured), compared against the native Linux flagship baseline ([`../betterbench_slots8_262k.json`](../betterbench_slots8_262k.json)):

### Single-Stream Decode (`batch = 1`, DSpark MTP, `--slots 8`, `q8_0`, `sparse-224`, Vision Loaded)

| Category | Native Linux Baseline (`262K`) | Patched Vision-Fixed (`196K`, WSL2) | TTFT p50 | Update p99 |
|---|---:|---:|---:|---:|
| `json` | `223.8 tok/s` | **`225.5 tok/s`** | `69.4 ms` | `23.8 ms` |
| `summarization` | `188.0 tok/s` | **`212.4 tok/s`** | `74.5 ms` | `23.5 ms` |
| `file_edit` | `213.1 tok/s` | **`210.9 tok/s`** | `71.9 ms` | `24.0 ms` |
| `math` | `206.0 tok/s` | **`204.0 tok/s`** | `63.4 ms` | `22.7 ms` |
| `code` | `191.2 tok/s` | **`197.1 tok/s`** | `67.9 ms` | `23.3 ms` |
| `reasoning` | `175.3 tok/s` | **`177.0 tok/s`** | `72.3 ms` | `22.3 ms` |
| `chat` | `158.5 tok/s` | **`141.0 tok/s`** | `74.4 ms` | `22.5 ms` |
| `prose` | `137.9 tok/s` | **`138.2 tok/s`** | `70.7 ms` | `19.3 ms` |
| **Combined (weighted)** | **`186.8 tok/s`** (`69.0 ms` / `23.2 ms`) | **`191.4 tok/s`** (`70.0 ms` / `22.7 ms`) | **`70.0 ms`** | **`22.7 ms`** |

### Multi-Slot Parallel Concurrency Sweep (`c = 1, 2, 4, 8`, `32/32` OK at Every Level)

| Concurrency | Completed | Baseline Wall / Agg / Med-Stream / TTFT p50 | **Patched Wall / Agg / Med-Stream / TTFT p50** | TTFT p99 |
|--:|--:|---|---|--:|
| **`c = 1`** | `32/32` | `73.4 s` / `172.7` / `185.8 tok/s` / `71.6 ms` | **`73.3 s` / `172.2` / `191.8 tok/s` / `71.8 ms`** | `88.7 ms` |
| **`c = 2`** | `32/32` | `75.4 s` / `168.5` / `87.1 tok/s` / `87.1 ms` | **`73.7 s` / `170.9` / `88.0 tok/s` / `83.5 ms`** | `139.3 ms` |
| **`c = 4`** | `32/32` | `56.1 s` / `223.9` / `58.4 tok/s` / `104.7 ms` | **`56.3 s` / `224.7` / `58.7 tok/s` / `101.8 ms`** | `288.7 ms` |
| **`c = 8`** | `32/32` | `47.3 s` / `268.0` / `36.0 tok/s` / `138.2 ms` | **`47.0 s` / `264.0` / `35.2 tok/s` / `137.4 ms`** | `617.0 ms` |

### Post-Concurrency Live `1080p` Vision Smoke (Same `--slots 8` Server, 34 Batch Graphs Resident)
- **Result:** `ok=True`, `1,321` prompt tokens, **`618.9 ms` TTFT**, **`221.7 tok/s`** decode.
- **VRAM before → after `1080p`:** `23,443 MB` → **`23,455 MB`** (`+12 MB` for `d_proj_out_` only; `131.65 MiB` workspace immediately released back to `0 MiB`; **`1,105 MB` still free**; `USED_GTT: 77 MB → 77 MB`, **`0 MB` GTT delta**).
- **Full interactive report:** [`betterbench_slots8_196k_visionfix.html`](betterbench_slots8_196k_visionfix.html) · [`betterbench_slots8_196k_visionfix.json`](betterbench_slots8_196k_visionfix.json).

---

## 5. Recommended Production Configs (RX 7900 XTX, 24 GB)

| Use Case | Environment | Command | App VRAM | Notes |
|---|---|---|---:|---|
| **Flagship 262K + MTP + 4K Vision** | Native Linux (`/dev/kfd`) | `nib serve -m ... --mtp ... --mmproj ... --slots 4 -c 262848 --kv-type q8_0 --sparse-pages 224` | `22.56 GiB` | Handles `4K` & `2560×2560` max-res Vision cleanly (`ws=0` after encode) |
| **Flagship 8-Way Parallel + 262K** | Native Linux (`/dev/kfd`) | `nib serve -m ... --mtp ... --mmproj ... --slots 8 -c 262848 --kv-type q8_0 --sparse-pages 224` | `23.08 GiB` | `186–191 tok/s` solo, `264–268 tok/s` agg @ `c = 8` |
| **8-Way Parallel + Hi-Res Vision (Windows/WSL2)** | Windows WDDM / WSL2 | `nib serve -m ... --mtp ... --mmproj ... --slots 8 -c 196608 --kv-type q8_0 --sparse-pages 224` | `22.00 GiB` | Leaves `1.1 GiB` free under WSL2's `~23 GiB` cap (`168/168` BetterBench + `1080p` Vision verified) |
| **8-Way + Full 262K on Windows/WSL2** | Windows WDDM / WSL2 | Add `--cpu-embd` to `--slots 8 -c 262848 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj` | `22.62 GiB` | Saves `469.3 MiB` VRAM (`< 0.8%` plain decode delta) to clear WDDM ceiling |
| **4-Slot 128K + 4K Vision + Desktop Headroom** | Linux / WSL2 desktop | `nib serve -m ... --mtp ... --mmproj ... --slots 4 -c 131072 --kv-type q8_0 --sparse-pages 224` | `20.40 GiB` | `2.85 GiB` free after `4K` / `2560×2560` Vision (`222–230 tok/s`) |
