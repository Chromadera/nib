# AMD Radeon RX 7900 XT (20 GB) — Real-Hardware Nib Evaluation

Real-silicon characterization of **Nib** running **Inkling 2 Mini** (`Inkling-2-Mini-NVFP4-Mixed.gguf`, `16.34 GiB` weights) on **AMD Radeon RX 7900 XT** (`gfx1100` / Navi 31, `84 CU` / `42 WGP`, `20 GB` GDDR6 @ `800 GB/s`), tested across SaladCloud community nodes on **2026-11-25 – 2026-11-26** (including pre-fix reproduction and post-fix verification of [`Chromadera/nib#5`](https://github.com/Chromadera/nib/issues/5)).

> **TL;DR**
> - **Decode matches 90–92% of RX 7900 XTX** (`168.2 tok/s` weighted BetterBench MTP on clean host `b121017c`, `114–125 tok/s` non-MTP, `143–166 tok/s` MTP up to `142K` tokens) — beating the raw 83.3% memory-bandwidth ratio (`800 / 960 GB/s`) thanks to 80 MiB Infinity Cache.
> - **Plug-and-play Vision up to `4K` / `2560×2560` max-res with `0 MB` GTT spill:** With single-arena aliased ViT scratch (freed immediately post-encode, `-56%` workspace), `--cpu-embd` (`-469.3 MiB` VRAM, `< 0.8%` plain decode delta), and `< 22 GiB` VRAM-aware auto-defaults, launching `nib serve --mtp --mmproj` with **zero `-c`/`--kv-type` flags** automatically configures `--slots 4 -c 65536 --kv-type q4_1 --cpu-embd` and passes **`1080p → 1440p → 4K 3840×2160 → 2560×2560` max (`4,096` img toks)** at **`143–173 tok/s`** with **`0 MB` GTT** and **`561–908 MB` free at peak**. Even `mandrav`'s `--slots 4 -c 131072 q4_1 + MTP + Vision` config (previously hard OOM / `38–53 tok/s` GTT spill) now passes **`1080p` at `168.3 tok/s` (`0 MB` GTT)** and cleanly returns `HTTP 400` on oversize images.
> - **True 4-Way & 8-Way Parallel Streaming (`--slots 8 -c 65536 --kv-type q4_1 --mtp`, localhost zero-RTT, `48/48 OK`):** **`192.7 tok/s` agg** at `c = 4` (`50.0 tok/s`/stream, `118.7 ms` TTFT p50) and **`213.0 tok/s` agg** at `c = 8` (`30.4 tok/s`/stream, `162.5 ms` TTFT p50, 33 static batch HIP graphs captured) — with up to **`55.2 tok/s`/stream** at `c = 4` on lower-desktop-VRAM hosts (`b121017c`).
> - **Full native 256K context (`262,848`) fits and runs cleanly on 20 GB even under Windows/WSL2** using `--kv-type q4_0 --sparse-pages 224 --no-vision` (no MTP): **`19.10 GiB`**, holding **`114.2 tok/s` decode** and **`2,278 tok/s` prefill** at `256,000` depth.
> - **With DSpark MTP enabled (text-only, `--no-vision`)**, the WSL2 safe ceiling (`≤19.10 GiB` solo; `≤18.7 GiB` recommended for multi-slot to avoid Windows WDDM 2s TDR when host desktop VRAM is pinned) supports up to **`90K` (`q8_0`)**, **`160K` (`q4_1`)**, or **`176K` (`q4_0`)** solo, **`128K` @ `--slots 4`** (`q4_1`), and **`64K` @ `--slots 8`** (`q4_1`, `8K/slot`) — with an additional `+469 MiB` headroom available via `--cpu-embd`.

---

## Folder Contents

| File | Description |
|---|---|
| [`README.md`](README.md) | This curated report (environment, VRAM ceiling, KV tables, non-MTP, MTP, BetterBench) |
| [`benchmarks.json`](benchmarks.json) | Machine-readable dataset of all Stage 1/2 VRAM probes, non-MTP `nib bench`, MTP `nib gen`, and BetterBench summaries (both `--slots 4` WAN and `--slots 8` localhost) |
| [`betterbench_slots8_64k_local.json`](betterbench_slots8_64k_local.json) | Zero-WAN-RTT localhost BetterBench JSON (`--slots 8 -c 65536 q4_1 + MTP`, single-stream + true `c=1,2,4,8` parallel streaming, `48/48` reqs/level) |
| [`betterbench_slots8_64k_local.html`](betterbench_slots8_64k_local.html) | Interactive localhost `--slots 8` BetterBench HTML report |
| [`betterbench_slots4_128k.json`](betterbench_slots4_128k.json) | Complete unified BetterBench JSON (`--slots 4 -c 131072 q4_1 + MTP`, over SaladCloud WAN gateway: single-stream + `c=1,2,4,8` + prefill ladder) |
| [`betterbench_slots4_128k.html`](betterbench_slots4_128k.html) | Interactive `--slots 4` BetterBench HTML report |
| [`rx7900xt_vs_xtx_chart.html`](rx7900xt_vs_xtx_chart.html) | Visual RX 7900 XT vs. RX 7900 XTX comparison dashboard |
| [`scripts/salad_bootstrap.sh`](scripts/salad_bootstrap.sh) | Container bootstrap for SaladCloud RX 7900 XT (`nib` build, `aria2c` weights, IPv6↔IPv4 `socat` bridge, multi-slot fallback) |
| [`scripts/vram_ceiling_probe.sh`](scripts/vram_ceiling_probe.sh) | Stage 1 raw HIP `hipMalloc`/`hipMemset` + Stage 2 real `nib` VRAM ceiling matrix |
| [`scripts/bench_nonmtp.sh`](scripts/bench_nonmtp.sh) | Non-MTP `nib bench` depth sweep (`q4_1`, `q8_0`, `q4_0`) |
| [`scripts/bench_q40_262k.sh`](scripts/bench_q40_262k.sh) | Focused same-GPU `q4_0 @ 131K` vs. `q4_0 @ 262,848` (`0 → 256,000`) benchmark |
| [`scripts/bench_mtp.sh`](scripts/bench_mtp.sh) | MTP `nib gen --mtp` prefill + decode benchmark across `q8_0 @ 90K`, `q4_1 @ 163K`, `q4_0 @ 180K` |
| [`scripts/bench_concurrency_local.sh`](scripts/bench_concurrency_local.sh) | Zero-WAN-RTT localhost BetterBench `--decode --concurrency` sweep for `--slots 8 -c 65536 q4_1 + MTP` |
| [`scripts/bench_vision_matrix.sh`](scripts/bench_vision_matrix.sh) | Vision (`--mmproj`) + MTP matrix (`1024×512` + `1920×1080` + post-image regression) reproducing Issue #5 |

---

## 1. Test Environment & Platform Notes

### Hardware & Stack (3-Layer SaladCloud AMD Architecture)

| Layer | Specification |
|---|---|
| **GPU** | AMD Radeon RX 7900 XT (`gfx1100`, `84 CU` / `42 WGP`, `320-bit`, `800 GB/s`, `80 MiB` L3 Infinity Cache, PCIe Gen 4) |
| **Reported VRAM** | `20,464 MB` (`amd-smi`) / `20,420.8 MiB` = `19.942 GiB` (`hipGetDeviceProperties`) |
| **Host OS & Driver** | Windows host (WDDM) + AMD Software Adrenalin `26.20.29.02-261116a-206168C` |
| **Virtualization** | WSL2 (`6.18.58.1-microsoft-standard-WSL2 #1 SMP PREEMPT_DYNAMIC x86_64`), GPU exposed via `/dev/dxg` (`ROCDXG` / `librocdxg`; no `/dev/kfd` or `/dev/dri`) |
| **Container Image** | `rocm/dev-ubuntu-24.04:7.2.4` (`1.22 GB` compressed; HIP `7.2.53211`, AMD clang `22.0.0git`) |
| **Container Resources** | `4 vCPU`, `16,384 MB` RAM, `8,192 MB` `/dev/shm`, `40 GiB` ephemeral disk, `medium` priority (`$0.122/hr`) |
| **Nodes Tested** | `b121017c` (BetterBench), `6eb9be11` (VRAM probe + MTP matrix), `dddf7d23` (Non-MTP sweep), `b840ff22` (Non-MTP `q4_0` 256K) |

### Key Deployment & Compiler Findings

1. **ROCm >= 7.1 required on WSL2/ROCDXG:** ROCm 7.0 fails under `/dev/dxg`; `rocm/dev-ubuntu-24.04:7.2.4` works out of the box without installing any kernel drivers inside the container.
2. **Non-PIE linker flag required on ROCm 7.2.4:** Clang 22 in `7.2.4` emits non-PIE HIP host stubs, requiring `-DCMAKE_EXE_LINKER_FLAGS="-no-pie"` in CMake.
3. **ROCm 7.2.4 compiler spills affect long-context prefill (not decode):**
   - Local native Linux **ROCm 7.16 (clang 24)** compiles all 18 `nib` objects with **0 spills**.
   - Container **ROCm 7.2.4 (clang 22)** spills `1312 VGPR / 4 SGPR` in `kernels_fa.o` and `1286 VGPR` in `kernels_sparse_global.o`, while **all decode kernels remain 0-spill**.
   - Result: decode hits **90–92% of RX 7900 XTX**, while dense `32K` and sparse `64K–256K` prefill land at **~72–75% of XTX** (estimated ~85% on native Linux ROCm 7.16 / `7.14.1-full`).
4. **IPv6 Gateway Bridge:** `nib serve` binds IPv4 (`AF_INET`), whereas Salad Container Gateway connects over IPv6 — bridged cleanly with:
   ```bash
   socat TCP6-LISTEN:4444,ipv6only=0,fork,reuseaddr,nodelay,backlog=128 TCP4:127.0.0.1:4445
   ```
5. **Co-Residency Guards Verified on 42 WGPs:** Every node logged `occupancy blocks/WGP (wgp=42): wo_b=2 wo_c=2 wo_c_h2=3 mix_rms_qkv=4 a_qkv=4` and captured the 312-node static decode HIP graph with zero hangs or parity issues.

---

## 2. Exact VRAM Ceiling & WDDM Behavior

### Stage 1 — Raw HIP Allocation Ceiling (`6eb9be11`)

- **Idle at boot:** `18,591.2 MiB` (`18.155 GiB`) free (`1,829.6 MiB` held by Windows host).
- **WDDM eviction under pressure:** Crossing `18 GiB` caused Windows WDDM to page ~`1.65 GiB` of host desktop VRAM out to system RAM (`used` dropped to `173.7 MiB` after freeing).
- **Hard `hipMalloc` + `hipMemset` ceiling:** **`19,908.0 MiB` (`19.4414 GiB`)** — identical for `1,245 × 16 MiB` and `512 × 32 MiB + 16..1 MiB` (`0 MiB` fragmentation loss).

### Stage 2 — Real `nib` Ceiling Matrix (`--sparse-pages 224`, `--no-vision`)

> **Accounting note:** In the pre-fix binary used during the initial Stage 2 ceiling probe, `InklingEngine`'s `total_vram` and `/health`'s `vram_gib` omitted DSpark MTP's **`697 MiB` (`0.681 GiB`)**. Adding MTP plus ~**`0.30–0.41 GiB`** of HIP runtime / code objects / HIP graph metadata reconciles the `nib` cliff (`19.10 GiB`) with Stage 1 (`19.44 GiB`) to the megabyte. **Fixed in current `nib`:** startup logs and `/health` now report exact `llm + mtp + vision` VRAM, live `hipMemGetInfo` device free VRAM, and an explicit PCIe GTT spillover warning when free VRAM drops below `512 MiB`.

| Config (`--slots 1` unless noted) | Engine VRAM | + MTP | Engine + MTP | Result on SaladCloud WSL2 |
|---|---:|---:|---:|---|
| `q8_0` @ `65,536` + MTP | `17.91 GiB` | `0.68 GiB` | **`18.59 GiB`** | **PASS** |
| `q4_1` @ `131,072` + MTP | `18.06 GiB` | `0.68 GiB` | **`18.74 GiB`** | **PASS** |
| `q4_1` @ `131,072` + MTP (`--slots 4`) | `18.29 GiB` | `0.68 GiB` | **`18.97 GiB`** | **PASS** (full BetterBench verified) |
| `q8_0` @ `90,112` + MTP | `18.31 GiB` | `0.68 GiB` | **`18.99 GiB`** | **PASS** |
| `q8_0` @ `131,072` (no MTP) | `18.99 GiB` | — | **`18.99 GiB`** | **PASS** |
| `q4_1` @ `229,376` (no MTP) | `19.02 GiB` | — | **`19.02 GiB`** | **PASS** |
| `q4_0` @ `180,224` + MTP | `18.36 GiB` | `0.68 GiB` | **`19.04 GiB`** | **PASS** |
| `q4_1` @ `131,072` + MTP (`--slots 5`) | `18.37 GiB` | `0.68 GiB` | **`19.05 GiB`** | **PASS** |
| `q4_1` @ `163,840` + MTP | `18.38 GiB` | `0.68 GiB` | **`19.06 GiB`** | **PASS** |
| `q4_0` @ `262,848` (no MTP) | `19.10 GiB` | — | **`19.10 GiB`** | **PASS** (verified `0 → 256,000`) |
| `q8_0` @ `98,304` + MTP | `18.45 GiB` | `0.68 GiB` | `19.13 GiB` | **STALL** (WDDM paging cliff) |
| `q4_1` @ `131,072` + MTP (`--slots 6`) | `18.45 GiB` | `0.68 GiB` | `19.13 GiB` | **STALL** (WDDM paging cliff) |
| `q4_1` @ `131,072` + MTP (`--slots 8`) | `18.60 GiB` | `0.68 GiB` | `19.28 GiB` | **FAIL** (`DSparkDrafter: hipMalloc failed`) |
| `q4_1` @ `262,848` (no MTP) | `~19.35 GiB` | — | `~19.35 GiB` | **FAIL** (`engine.cpp:145` OOM; fits native Linux) |
| `q4_1` @ `196,608` + MTP | `18.70 GiB` | `0.68 GiB` | `19.38 GiB` | **FAIL** (`DSparkDrafter: hipMalloc failed`) |
| `q8_0` @ `131,072` + MTP | `18.99 GiB` | `0.68 GiB` | `19.67 GiB` | **FAIL** (`DSparkDrafter: hipMalloc failed`) |
| `q4_0` @ `262,848` + MTP | `19.10 GiB` | `0.68 GiB` | `19.78 GiB` | **FAIL** (`DSparkDrafter: hipMalloc failed`) |

---

## 3. KV Cache Capacity Tables (`--sparse-pages 224`, `--no-vision`)

Calibrated to all measured hardware points (`≤0.007 GiB` error). Extra concurrent slots only add SWA ring buffers (`+0.128 GiB`/slot for `q8_0`, `+0.077 GiB`/slot for `q4_1`, `+0.069 GiB`/slot for `q4_0`); global KV is partitioned zero-copy.

### Non-MTP Maximum Context Length

| KV Type | Bits / val | Cost / 32K ctx | `--slots` | Max Context on WSL2 (`≤19.10 GiB`) | VRAM | Max Context on Native Linux Headless (`≤19.55 GiB`) | VRAM |
|---|---:|---:|---:|---|---:|---|---:|
| **`q8_0`** | `8.5` | `+0.54 GiB` | `1` | **`137,216`** (`134K`, `131,072` verified) | `19.09 GiB` | **`164,864`** (`161K`) | `19.55 GiB` |
| **`q8_0`** | `8.5` | `+0.54 GiB` | `4` | **`113,664`** (`111K`) | `19.09 GiB` | **`141,312`** (`138K`) | `19.54 GiB` |
| **`q4_1`** | `5.0` | `+0.32 GiB` | `1` | **`236,544`** (`231K`, `229,376` verified) | `19.09 GiB` | **`262,848`** (**full 256K**) | `19.35 GiB` |
| **`q4_1`** | `5.0` | `+0.32 GiB` | `4` | **`212,992`** (`208K`) | `19.09 GiB` | **`259,072`** (`253K`) | `19.55 GiB` |
| **`q4_0`** | `4.5` | `+0.29 GiB` | `1` | **`262,848`** (**full 256K**, verified) | `19.10 GiB` | **`262,848`** (**full 256K**) | `19.10 GiB` |
| **`q4_0`** | `4.5` | `+0.29 GiB` | `4` | **`239,616`** (`234K`) | `19.09 GiB` | **`262,848`** (**full 256K**) | `19.30 GiB` |

### MTP Maximum Context Length (`--slots 1`, `+0.681 GiB` DSpark Drafter)

| KV Type | Max Verified on WSL2 (`≤19.10 GiB`) | Engine + MTP VRAM | Est. Max on Native Linux Headless (`≤19.55 GiB`) |
|---|---|---:|---|
| **`q8_0` + MTP** | **`90,112`** (`88K`) | `18.31 + 0.68 = 18.99 GiB` | `~122,880` (`120K`) |
| **`q4_1` + MTP** | **`163,840`** (`160K`; or `131K` @ `--slots 5`) | `18.38 + 0.68 = 19.06 GiB` | `~212,992` (`208K`) |
| **`q4_0` + MTP** | **`180,224`** (`176K`) | `18.36 + 0.68 = 19.04 GiB` | `~237,568` (`232K`) |

---

## 4. Non-MTP Benchmarks (`nib bench -p 512 -n 64 -r 2 --sparse-pages 224`)

### Prefill (`pp512`, tok/s)

| Depth | `q4_1` @ `131K` (`18.06 GiB`) | `q8_0` @ `131K` (`18.99 GiB`) | `q4_1` @ `229K` (`19.02 GiB`) | `q4_0` @ `131K` (`17.93 GiB`)* | `q4_0` @ `262K` (`19.10 GiB`)* | XTX Ref (`q8_0` 256K, ROCm 7.16) |
|---:|---:|---:|---:|---:|---:|---:|
| **`0`** | `4,240.5` | `4,255.7` | `4,239.1` | `4,144.5` | `4,102.4` | `4,985.0` |
| **`8,192`** | `3,233.3` | `3,234.2` | `3,238.4` | — | `3,074.0` | — |
| **`32,768`** | `2,357.0` | `2,312.0` | `2,358.8` | `2,248.5` | `2,243.4` | — |
| **`65,536`** | `2,532.8` | `2,528.6` | `2,516.0` | — | `2,374.7` | `3,520.5` |
| **`128,000`** | `2,504.0` | `2,476.5` | `2,501.0` | `2,361.3` | `2,385.3` | — |
| **`224,000`** | — | — | `2,444.5` | — | `2,291.2` | — |
| **`256,000`** | — | — | — | — | **`2,278.4`** | `3,279.5` |

### Decode (`tg64`, tok/s)

| Depth | `q4_1` @ `131K` (`18.06 GiB`) | `q8_0` @ `131K` (`18.99 GiB`) | `q4_1` @ `229K` (`19.02 GiB`) | `q4_0` @ `131K` (`17.93 GiB`)* | `q4_0` @ `262K` (`19.10 GiB`)* | XTX Ref (`q8_0` 256K, ROCm 7.16) |
|---:|---:|---:|---:|---:|---:|---:|
| **`0`** | `124.55` | `122.80` | `123.58` | `122.41` | `121.17` | `136.95` |
| **`8,192`** | `121.05` | `118.96` | `120.59` | — | `118.12` | — |
| **`32,768`** | `118.48` | `116.17` | `116.93` | `116.83` | `116.30` | — |
| **`65,536`** | `118.14` | `115.48` | `118.37` | — | `115.74` | `128.21` |
| **`128,000`** | `117.77` | `115.46` | `117.74` | `116.00` | `115.07` | — |
| **`224,000`** | — | — | `116.14` | — | `114.70` | — |
| **`256,000`** | — | — | — | — | **`114.20`** (`90.5%` XTX) | `126.22` |

*\*`q4_0` measured on node `b840ff22` (which clocked ~3% lower than `dddf7d23` across both `131K` and `262K`, normal AIB clock variance). Comparing `131K` vs `262K` on `b840ff22` confirms `<1%` difference — zero WDDM throttling at `19.10 GiB`.*

---

## 5. MTP Benchmarks (`nib gen --mtp -n 256 --sparse-pages 224`)

Measured on `6eb9be11` using identical real code+docs prompts (`README.md` + `src/engine.cpp` + `src/server.cpp` + C++17 `RingBuffer<T, N>` coding task):

| Config (`--slots 1`, MTP ON) | Engine + MTP VRAM | Prompt Tokens | Cumulative Prefill (`tok/s`) | MTP Decode (`tok/s`) | `tok/round` (`avg_m`) | `draft` / `verify` ms |
|---|---:|---:|---:|---:|---:|---:|
| **`q8_0` @ `90,112` (`88K`)** | `18.99 GiB` | `2,740` | `2,108.0` | **`160.84`** | `4.47` (`5.21`) | `1.89` / `24.83` |
| | | `38,178` | `1,791.9` | **`157.51`** | `4.23` (`4.44`) | `2.08` / `23.08` |
| | | `75,106` | `1,534.3` | **`165.69`** | `4.72` (`5.22`) | `2.21` / `25.26` |
| **`q4_1` @ `163,840` (`160K`)** | `19.06 GiB` | `2,740` | `2,439.3` | **`153.24`** | `3.92` (`4.64`) | `1.79` / `22.25` |
| | | `38,178` | `1,768.4` | **`143.22`** | `3.67` (`4.43`) | `1.94` / `22.68` |
| | | `75,106` | `1,529.5` | **`146.56`** | `4.00` (`4.95`) | `2.13` / `24.09` |
| | | `142,265` | `1,693.4` | **`144.71`** | `3.97` (`4.58`) | `2.29` / `23.49` |
| **`q4_0` @ `180,224` (`176K`)** | `19.04 GiB` | `2,740` | `2,509.6` | **`156.70`** | `4.06` (`4.86`) | `1.72` / `22.38` |
| | | `38,178` | `1,688.4` | **`164.90`** | `4.44` (`4.58`) | `1.97` / `23.34` |
| | | `75,106` | `1,554.2` | **`149.87`** | `4.06` (`5.05`) | `2.02` / `24.18` |
| | | `142,265` | `1,727.4` | **`144.10`** | `3.91` (`4.43`) | `2.02` / `22.75` |

- **`q8_0` boosts MTP decode by +10–13% over `q4_1`** (`158–166` vs `143–153 tok/s`) because Q8 KV preserves higher speculative acceptance (`4.23–4.72` vs `3.67–4.00 tok/round`).
- **Cumulative prefill rebounds from `75K` → `142K`** (`1,530 → 1,727 tok/s`) once Quest sparse-224 replaces dense `0 → 64K` prefill.

---

## 6. BetterBench End-to-End API Results (`nib serve`)

Two complementary BetterBench (`0.6.1`, corpus `v1.0`, `--greedy`, cold nonce prefix cache) runs were executed on RX 7900 XT:

1. **True 8-Slot Localhost Run (`127.0.0.1`, zero WAN RTT)** on `6eb9be11`: `--slots 8 -c 65536 --kv-type q4_1 --sparse-pages 224 --mtp --no-vision` (`17.98 GiB` engine / `18.66 GiB` total, `8K/slot` concurrent · `64K` solo, **48 requests/level**). Full reports: [`betterbench_slots8_64k_local.html`](betterbench_slots8_64k_local.html) · [`betterbench_slots8_64k_local.json`](betterbench_slots8_64k_local.json).
2. **4-Slot 128K WAN Gateway Run** on `b121017c` (lower host desktop VRAM): `--slots 4 -c 131072 --kv-type q4_1 --sparse-pages 224 --mtp --no-vision` (`18.31 GiB` engine / `19.00 GiB` total, `32K/slot` concurrent · `128K` solo, **32 requests/level**, includes `~1.0–1.3s` SaladCloud WAN RTT). Full reports: [`betterbench_slots4_128k.html`](betterbench_slots4_128k.html) · [`betterbench_slots4_128k.json`](betterbench_slots4_128k.json).

### Parallel Concurrency — True 8-Way Localhost (`--slots 8 -c 65536`, Zero WAN RTT, `48/48 OK`)

| Concurrency | Completed | XT Aggregate (`tok/s`) | XT Per-Stream Median (`tok/s`) | XT TTFT p50 (`ms`) | XTX Ref (`agg` / `med` / `TTFT p50`) | XT / XTX (`agg` / `med`) |
|--:|--:|--:|--:|--:|--:|--:|
| **`c = 1`** *(Solo MTP)* | `48/48` | **`148.2`** | **`170.0`** | **`82.6 ms`** | `172.7` / `185.8` / `71.6 ms` | `85.8%` / `91.5%` |
| **`c = 2`** *(Batched 2-slot)* | `48/48` | **`147.9`** | **`75.3`** | **`97.5 ms`** | `168.5` / `87.1` / `87.1 ms` | `87.8%` / `86.5%` |
| **`c = 4`** *(Batched 4-slot)* | `48/48` | **`192.7`** | **`50.0`** | **`118.7 ms`** | `223.9` / `58.4` / `104.7 ms` | `86.1%` / `85.6%` |
| **`c = 8`** *(Batched 8-slot)* | `48/48` | **`213.0`** | **`30.4`** | **`162.5 ms`** | `268.0` / `36.0` / `138.2 ms` | `79.5%` / `84.4%` |

- **33 dynamic static batch HIP graphs (`B = 2..8`)** were captured across the sweep, adding only **`~88 MB` total (`~2.8 MB/graph`)** (`amd-smi`: `USED_GTT 623 → 711 MB`).
- **WSL2 WDDM multi-slot headroom & TDR finding:** Node `6eb9be11` kept `~1.1–1.4 GB` of Windows host desktop VRAM pinned even at idle, so `18.66 GiB` still had `~217–305 MB` of GTT spill (shaving ~8–10% off throughput vs. clean host `b121017c`, which hit `83.6 tok/s` at `c = 2` and `55.2 tok/s` at `c = 4`). Pushing `≥18.98 GiB` under multi-slot load on `6eb9be11` (`--slots 8 -c 98304` or `--slots 4 -c 131072`) triggered Windows' **2-second WDDM TDR** GPU reset during `B = 2 → B = 1` tail transitions. On native Linux (no Windows desktop VRAM, no 2s WDDM TDR), full `c = 4`/`c = 8` throughput scales cleanly to ~`88–92%` of XTX (`~205–210` agg @ `c = 4`, `~235–245` agg @ `c = 8`).

### Parallel Concurrency — 4-Slot 128K WAN Gateway (`b121017c`, `32/32 OK`)

| Concurrency | XT Per-Stream Median (`tok/s`) | XT Aggregate (`tok/s`) | XT Median TTFT | XTX Ref (`med` / `agg`) |
|---:|---:|---:|---:|---:|
| **`c = 1`** | **`170.7`** | `97.8` *(WAN RTT limited)* | `1,301 ms` | `185.8` / `172.7` |
| **`c = 2`** | **`83.6`** | `120.0` | `1,126 ms` | `87.1` / `168.5` |
| **`c = 4`** | **`55.2`** | **`184.8`** | `1,072 ms` | `58.4` / `223.9` |
| **`c = 8`** *(4 active + 4 queued)* | **`51.4`** | **`193.7`** | `7,908 ms` | `36.0` / `268.0` |

### Single-Stream Decode (8 Categories, Greedy, Cold Nonce Prefix)

| Category | XT `b121017c` (`--slots 4`, `128K`) | XT `6eb9be11` (`--slots 8`, `64K` local) | RX 7900 XTX Ref (`tok/s`) | Best XT / XTX |
|---|---:|---:|---:|---:|
| `json` | **`192.9`** | `189.3` | `223.8` | `86.2%` |
| `file_edit` | **`187.5`** | `181.5` | `213.1` | `88.0%` |
| `math` | **`179.5`** | `172.9` | `206.0` | `87.1%` |
| `summarization` | **`178.8`** | `166.0` | `188.0` | `95.1%` |
| `code` | **`171.8`** | `157.9` | `191.2` | `89.9%` |
| `reasoning` | **`163.2`** | `139.7` | `175.3` | `93.1%` |
| `chat` | **`157.4`** | `130.5` | `158.5` | `99.3%` |
| `prose` | **`123.2`** | `118.7` | `137.9` | `89.3%` |
| **Weighted Overall** | **`168.2`** | `156.2` *(TTFT `83 ms`, p99 `26.8 ms`)* | **`186.8`** | **`90.0%`** |

### BetterBench Prefill Ladder (`q4_1 + MTP`, ROCm 7.2.4 + WAN RTT on `b121017c`)

| Prompt Tokens | XT Prefill Median (`tok/s`) | XT Median TTFT | XTX Ref (`tok/s`, ROCm 7.16 local) |
|---:|---:|---:|---:|
| `2,032` | `1,073.4` *(WAN RTT dominated)* | `1.89 s` | `3,974.6` |
| `8,072` | `1,937.1` | `4.17 s` | `4,234.8` |
| `16,092` | `1,708.5` | `10.05 s` | `4,127.6` |
| `32,218` | **`2,198.5`** | `14.65 s` | `3,968.4` |
| `64,020` | `1,851.4` | `34.78 s` | `3,645.3` |

---

## 7. Vision (`--mmproj`) + MTP Matrix & Issue #5 Resolution

Tested on clean RX 7900 XT node `f90c2c79` (`292 MB` idle VRAM, **`20,172 MB` / `19.70 GiB` free**, `0 MB` GTT — matching native Linux desktop headroom) with `mmproj-Inkling-2-Mini-Q8_0.gguf` (`329.9 MiB` weights), reproducing and verifying the fix for [`Chromadera/nib#5`](https://github.com/Chromadera/nib/issues/5).

### 7.1 Pre-Fix Reproduction (Issue #5 Root Cause)

On the unpatched binary, each booting config ran 4 steps: **text baseline** → **`1024×512` card PNG** → **`1920×1080` PNG** → **post-`1080p` text check**.

| Config (`--mmproj` ON, `--sparse-pages 224`) | True App VRAM *(incl. MTP)* | `amd-smi` Boot (`USED / FREE / GTT`) | `amd-smi` Post-`1080p` (`USED / FREE / GTT`) | `1080p` TTFT / Decode | Outcome |
|---|---:|---|---|---:|---|
| **`--slots 1 -c 32768 q4_1` + MTP** | **`18.12 GiB`** | `19,346 / 1,118 / 0 MB` | `19,673 / 791 / 0 MB` | `792 ms` / **`157.0 tok/s`** | **PASS** (`0 MB` GTT) |
| **`--slots 1 -c 65536 q4_1` + MTP** | **`18.44 GiB`** | `19,675 / 789 / 0 MB` | `20,001 / 463 / 0 MB` | `793 ms` / **`149.7 tok/s`** | **PASS** (`0 MB` GTT) |
| **`--slots 4 -c 65536 q4_1` + MTP** *(Issue #5 `181 tg32`)* | **`18.67 GiB`** | `19,917 / 547 / 0 MB` | **`20,244 / 220 / 0 MB`** | `788 ms` / **`166.1 tok/s`** | **PASS** (`0 MB` GTT, only `220 MB` left after `1080p`) |
| **`--slots 1 -c 131072 q4_1`** *(no MTP)* | **`18.38 GiB`** | `19,570 / 894 / 0 MB` | `19,897 / 567 / 0 MB` | `765 ms` / `91.4 tok/s` | **PASS** (`0 MB` GTT, `567 MB` free after `1080p`) |
| **`--slots 1 -c 65536 q8_0`** *(no MTP)* | **`18.23 GiB`** | `19,415 / 1,049 / 0 MB` | `19,742 / 722 / 0 MB` | `771 ms` / `90.8 tok/s` | **PASS** (`0 MB` GTT, `722 MB` free after `1080p`) |
| **`--slots 2 -c 131072 q4_1` + MTP** *(Issue #5 text-only)* | **`19.16 GiB`** | Loaded `329.9 MiB` ViT → stalled at HIP graph warmup | — | — | **STALL (WSL2)**; native Linux fits text-only with `~0` room for `1080p` scratch |
| **`--slots 4 -c 131072 q4_1` + MTP** *(Issue #5 `38–53 tok/s`)* | **`19.31 GiB`** | **`HIP error 2 (out of memory) at vision_encoder.cpp:313`** | — | — | **OOM (WSL2)** / **`amdgpu` TTM GTT spill (`38–53 tok/s`) on native Linux** |

- **Old lazy ViT scratch growth (`VisionEncoder::ensure_scratch`):** `0.322 GiB` startup → `0.401 GiB` (`+79 MiB`) on `1024×512` → `0.623 GiB` (`+301 MiB`, `amd-smi` `+327 MB` total) on `1920×1080` → up to `1.274 GiB` peak on `4K` images, never freed until shutdown.
- **Issue #5 root cause:** `--slots 4 -c 131072 q4_1` is **not** a multi-slot bug (it runs at `168.2 tok/s` solo / `55.2 tok/s` `c = 4` with `--no-vision`). Adding `--mmproj` (`+0.32 GiB`) alongside `--mtp` (`+0.68 GiB`, previously omitted from `vram=`) pushed true GPU footprint to `~19.72 GiB` (incl. `~0.41 GiB` HIP runtime), exceeding free VRAM on a desktop 20 GB card and causing Linux `amdgpu` TTM to silently spill buffers over PCIe into GTT.

### 7.2 Post-Fix Verification (Aliased ViT Scratch, `--cpu-embd`, `< 22 GiB` Auto Defaults)

Four host-only C++ improvements were implemented (zero `.hip` kernel edits, `0` spills across all 18 objects in `spill_audit.sh`, and zero regression on 24 GB RX 7900 XTX — see [`../rx7900xtx/README.md`](../rx7900xtx/README.md)):

1. **Aliased single-arena ViT workspace + immediate post-encode release (`src/vision_encoder.{hpp,cpp}`):** Packs non-overlapping transient ViT buffers (`hidden | norm | shared phase pool`, with `d_attn_out_` aliasing `d_qkv_`) into one 256-byte-aligned allocation freed immediately after `encode_rgb_u8()` (leaving only the small `d_proj_out_` buffer: `11.4 MiB` at `1080p`, `36 MiB` at `4K`).
   - **Bit-exact parity:** `max_abs_diff = 0`, `0 / 2,985,984` differing floats at `1920×1080` and `0 / 778,752` at `1024×512` (plus `0 / 9,400,320` at `4K` on XTX); warmed `1080p` encode time unchanged (`202.53 ms` vs `202.96 ms`).
   - **Scratch reduction:** `1080p` transient workspace **`~301 MiB → 131.7 MiB` (`-56%`)**, `4K` max **`~950 MiB → 416.0 MiB`**, freed back to **`0.0 MiB`** immediately after each encode (`0.62 → 0.33 GiB` post-`1080p` Vision VRAM).
   - **Graceful OOM guard:** Oversize images return `HTTP 400` instead of crashing or spilling to GTT.
2. **`--cpu-embd` / `--no-cpu-embd` (`src/gguf_loader.cpp`, `src/engine.{hpp,cpp}`):** Places `token_embd.weight` (`469.3 MiB`) in GPU-mapped pinned host RAM (`hipHostMalloc(hipHostMallocMapped)`), keeping `lm_head` 100% in VRAM.
   - **Parity & cost:** Repack `max abs err = 0.000000`, identical greedy tokens (`256/256` MTP, `128/128` plain); saves **`469.3 MiB`** VRAM (`16,730.2 → 16,260.8 MiB`) at **`-0.7% to -0.8%`** plain `tg128`, **`-0.1% to -0.3%`** `pp512`, and within run-to-run noise on `gen --mtp`.
3. **VRAM-aware `< 22 GiB` auto-defaults (`src/main.cpp`):** Complete no-op on `>= 22 GiB` cards. Below `22 GiB`, only for flags not explicitly passed: defaults `--kv-type` to `q4_1`, auto-enables `--cpu-embd` when MTP + Vision are both loaded (or projected free VRAM `< 1 GiB`), and sizes `-c` via exact `InklingEngine::estimate_state_bytes()` to preserve `>= 768 MiB` free (`>= ~928 MiB` with Vision).
4. **Accurate `llm + mtp + vision` accounting & GTT warning (`src/server.cpp`):** Startup logs and `/health` now report `mtp_vram_gib`, `device_free_vram_gib`, `device_total_vram_gib`, `cpu_embd`, `cpu_embd_host_mib`, `vision_workspace_mib`, and `vision_workspace_peak_mib`.

#### High-Res Escalation Matrix (`1080p → 1440p → 4K → 2560×2560` Max, 200 ms `amd-smi` Polling)

Every run below recorded **`peak_gtt_mb = 0 MB`** across all samples, and `vision_workspace_mib` returned to **`0.0 MiB`** immediately after each image:

| Config (`mmproj=ON`) | `cpu_embd` | Free @ Boot | `1080p` (`131.7 MiB` ws) | `1440p` (`234.0 MiB` ws) | `4K 3840×2160` (`414.4 MiB` ws) | `2560×2560` Max (`416.0 MiB` ws, `4,096` img toks) | Post-HiRes Free | Peak GTT |
|---|---:|---:|---|---|---|---|---:|---:|
| **`auto_default`** (`--mtp --mmproj` only → auto `s4 64K q4_1`) | Auto ON | `1,016 MiB` | **PASS** `159.6 tok/s` (`890 MB` min free) | **PASS** `143.8 tok/s` (`776 MB` min free) | **PASS** `142.7 tok/s` (`582 MB` min free) | **PASS** `154.6 tok/s` (`580 MB` min free) | **`999 MB`** | **`0 MB`** |
| **`s4_64k_mtp_auto`** (`--slots 4 -c 65536 q4_1 sp224 + MTP`) | Auto ON | `997 MiB` | **PASS** `173.1 tok/s` (`871 MB` min free) | **PASS** `149.5 tok/s` (`757 MB` min free) | **PASS** `145.4 tok/s` (`563 MB` min free) | **PASS** `143.1 tok/s` (`561 MB` min free) | **`980 MB`** | **`0 MB`** |
| **`s1_128k_nomtp`** (`--slots 1 -c 131072 q4_1 sp224`, no MTP) | Auto ON | `1,343 MiB` | **PASS** `91.9 tok/s` (`1,218 MB` min free) | **PASS** `92.2 tok/s` (`1,104 MB` min free) | **PASS** `91.5 tok/s` (`910 MB` min free) | **PASS** `91.9 tok/s` (`908 MB` min free) | **`1,327 MB`** | **`0 MB`** |
| **`s4_64k_mtp_nocpu`** (`--no-cpu-embd` forced) | OFF | `527 MiB` | **PASS** `162.6 tok/s` | **PASS** `153.2 tok/s` (`287 MB` min free) | `HTTP 400` *(clean guard)* | `HTTP 400` *(alive)* | `510 MB` | **`0 MB`** |
| **`s2_128k_mtp_auto`** (`--slots 2 -c 131072 q4_1 sp224 + MTP`) | Auto ON | `498 MiB` | **PASS** `168.6 tok/s` (`373 MB` min free) | **PASS** `151.6 tok/s` (`258 MB` min free) | `HTTP 400` *(clean guard)* | `HTTP 400` *(alive)* | `481 MB` | **`0 MB`** |
| **`s4_128k_mtp_auto`** (`--slots 4 -c 131072 q4_1 sp224 + MTP`) | Auto ON | `338 MiB` | **PASS** `168.3 tok/s` (`212 MB` min free)* | `HTTP 400` *(clean guard)* | `HTTP 400` *(alive)* | `HTTP 400` *(alive)* | `320 MB` | **`0 MB`** |

*\*`s4_128k_mtp_auto` also passed a 9-request mixed stress test including **4 concurrent requests (2 image + 2 text)** at **`151.8 tok/s` combined** with `0 MB` GTT across all 167 `amd-smi` samples (`auto_default` passed the same 4-concurrent mixed test at **`153.6 tok/s` combined** with `891 MB` min free).*

#### Before vs. After Impact on Issue #5 Matrix

| Matrix Case | Before (Pre-Fix Binary) | After (Patched Binary) | Net Improvement |
|---|---|---|---|
| **`mandrav_s4_128k_mtp`** (`s4 128K q4_1 + MTP + Vision`) | **HARD OOM** at `vision_encoder.cpp:313` (GTT spill `38–53 tok/s` on Linux) | **PASS** `1080p` (`168.3 tok/s`, `781.6 ms` TTFT, `212 MB` min free, `0 MB` GTT); graceful `HTTP 400` on `1440p+` | **OOM / Spill → Working (`1080p`)** |
| **`mandrav_s2_128k_mtp`** (`s2 128K q4_1 + MTP + Vision`) | **STALLED** at HIP graph warmup | **PASS** up to **`1440p`** (`168.6 tok/s` `1080p` / `151.6 tok/s` `1440p`, `0 MB` GTT) | **Stall → Working (`1440p`)** |
| **`mandrav_s4_64k_mtp`** (`s4 64K q4_1 + MTP + Vision`) | `1080p` PASS (`166.1 tok/s`, `788 ms`), only `220 MB` free after | **PASS up to `4K` & `2560×2560` max** (`173.1 tok/s` `1080p` / `145.4 tok/s` `4K`, **`980 MB` free** after) | **`+760 MB` headroom + `4K` max-res** |
| **`solo_s1_128k_nomtp`** (`s1 128K q4_1 + Vision`, no MTP) | `1080p` PASS (`91.4 tok/s`, `765 ms`), `567 MB` free after | **PASS up to `4K` & `2560×2560` max** (`91.9 tok/s`, `762.6 ms`, **`1,327 MB` free** after) | **`+760 MB` headroom + `4K` max-res** |
| **Zero-flag default** (`--mtp --mmproj` only) | Instant OOM (attempted `256K q8_0`) | Auto `s4 64K q4_1 + --cpu-embd` → **PASS up to `4K` & `2560×2560`** (`143–160 tok/s`, `0 MB` GTT) | **OOM → Plug-and-play `4K`** |

---

## 8. Recommended RX 7900 XT Production Configs

| Workload Goal | Recommended `nib serve` Flags (`--sparse-pages 224`) | True App VRAM | Notes |
|---|---|---:|---|
| **Plug-and-play Multimodal (Vision up to `4K` + MTP, 4 slots)** | `--mtp models/mtp.safetensors --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf` *(auto-picks `--slots 4 -c 65536 --kv-type q4_1 --cpu-embd`)* | `~18.21 GiB` | Passes `1080p → 4K → 2560×2560` max with `~980 MiB` free after encode, `0 MB` GTT |
| **4-way Multimodal + `128K` context (Vision up to `1080p` + MTP)** | `--slots 4 -c 131072 --kv-type q4_1 --mtp models/mtp.safetensors --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf` *(auto-enables `--cpu-embd`)* | `18.87 GiB` | `168.3 tok/s` at `1080p`, `0 MB` GTT; clean `HTTP 400` on `1440p+` (use `--slots 2` for `1440p`) |
| **Multimodal `128K` context up to `4K` (Vision, no MTP)** | `--slots 1 -c 131072 --kv-type q4_1 --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf` *(auto-enables `--cpu-embd`)* | `~17.92 GiB` | Passes `4K` / `2560×2560` with `1,327 MB` free after encode |
| **Safe 8-way parallel streaming (text-only)** | `--slots 8 -c 65536 --kv-type q4_1 --mtp models/mtp.safetensors --no-vision` | `18.66 GiB` | `213.0 tok/s` agg @ `c = 8` (add `--cpu-embd` for `~18.20 GiB` on busy desktop hosts) |
| **4-way parallel API w/ `128K` solo (text-only)** | `--slots 4 -c 131072 --kv-type q4_1 --mtp models/mtp.safetensors --no-vision` *(add `--cpu-embd` for `~18.51 GiB`)* | `18.97 GiB` | `168.2 tok/s` weighted solo, `55.2 tok/s`/stream @ `c = 4` |
| **Fastest & highest-precision solo coding (`≤88K`, text-only)** | `--slots 1 -c 90112 --kv-type q8_0 --mtp models/mtp.safetensors --no-vision` | `18.99 GiB` | `158–166 tok/s` MTP (`q8_0` KV gives +10–13% acceptance over `q4_1`) |
| **Max solo context with MTP (WSL2, text-only)** | `--slots 1 -c 163840 --kv-type q4_1 --mtp models/mtp.safetensors --no-vision` *(or `180224` `q4_0`)* | `19.06 GiB` | Add `--cpu-embd` for even higher context ceiling |
| **Full native 256K context on 20 GB (WSL2)** | `--slots 1 -c 262848 --kv-type q4_0 --no-vision` *(no MTP)* | `19.10 GiB` | Verified `0 → 256,000` (`114.2 tok/s` decode, `2,278 tok/s` prefill) |
| **Full native 256K context on 20 GB (Native Linux)** | `--slots 1 -c 262848 --kv-type q4_1 --no-vision` *(no MTP)* | `19.35 GiB` | Or `~18.89 GiB` with `--cpu-embd` |
