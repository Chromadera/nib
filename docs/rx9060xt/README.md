# AMD Radeon RX 9060 XT (16 GB, `gfx1200`) — Real-Hardware Nib Evaluation

Real-silicon evaluation of **Nib `v0.2.1`** (`9de6314`) running **Inkling 2 Mini** (`Inkling-2-Mini-NVFP4-All.gguf`, `14.52 GiB` disk / `13.98 GiB` weights in VRAM with `--cpu-embd`) on **AMD Radeon RX 9060 XT (16 GB)** (`gfx1200` / Navi 44, `32 CU` / `16 WGP`, `128-bit`, `320 GB/s`, `32 MiB` L3 Infinity Cache, `16 GB` GDDR6), tested under Windows/WSL2 (`ROCDXG`, ROCm 7.2.4) on **2026-11-27**.

> **TL;DR**
> - **Zero spills, 9/9 kernel tests, zero hangs:** `./scripts/spill_audit.sh` passes (`0` SGPR spills, `0` VGPR spills, `0` private segment, `0` stubs) on `gfx1200`, `./build/nib_kernel_test` passes **9/9** (`cos = 0.9999998–1.0000000`, `0` coherence errors, `0` co-residency deadlocks on 16 MPs), and the **452-node** co-residency fallback static decode HIP graph completed the entire benchmark suite (`nib bench` + two full BetterBench `0.6.1` sweeps, `40/40 + 32/32 OK`) with **0 hangs**.
> - **Same 16 GB context envelope as RX 9070 XT:** **`73,728`** solo (`--slots 1`, `q4_0`, `15.06 GiB` VRAM) and **`49,152`** multi-slot (`--slots 4`, `12,288`/slot, `q4_0`, `15.06 GiB` VRAM) with `--cpu-embd` ON (`469.3 MiB` `token_embd` in GPU-mapped host RAM).
> - **Outperforms raw bandwidth/CU ratio:** On half the memory bus (`320` vs `640 GB/s`) and half the CUs (`32` vs `64 CU`) of RX 9070 XT, RX 9060 XT hits **`56.1 tok/s`** BetterBench single-stream decode (**`57.8%`** of RX 9070 XT), **`93.9 tok/s`** 4-way aggregate (**`57.3%`**), and **`2,425 → 2,243 tok/s`** BetterBench prefill (`8K → 56K`, **`54.5–54.7%`** of RX 9070 XT; `2,852 tok/s` `pp512` at `d0`).

Raw run notes and curated BetterBench `0.6.1` reports live in [`v0.2.1_2026-11-27/NOTES.md`](v0.2.1_2026-11-27/NOTES.md):
- Single-stream 72K (`--slots 1 -c 73728 q4_0`): [`bb_All_s1_q4_0_c73728.html`](v0.2.1_2026-11-27/bb_All_s1_q4_0_c73728.html) · [`bb_All_s1_q4_0_c73728.json`](v0.2.1_2026-11-27/bb_All_s1_q4_0_c73728.json)
- 4-way concurrency 48K (`--slots 4 -c 49152 q4_0`): [`bb_All_s4_q4_0_c49152.html`](v0.2.1_2026-11-27/bb_All_s4_q4_0_c49152.html) · [`bb_All_s4_q4_0_c49152.json`](v0.2.1_2026-11-27/bb_All_s4_q4_0_c49152.json)

---

## 1. Test Environment & Verification (`-DGPU_TARGET=gfx1200`)

| Layer | Specification |
|---|---|
| **GPU** | AMD Radeon RX 9060 XT (`gfx1200` / Navi 44, `32 CU` / `16 WGP`, `128-bit`, `320 GB/s`, `32 MiB` L3 Infinity Cache) |
| **Reported VRAM** | `15.81 GiB` (`hipGetDeviceProperties`, `16,304 MB` total via `amd-smi`, `670 MB` idle host VRAM) |
| **Virtualization** | Windows/WSL2 (`6.18.58.1-microsoft-standard-WSL2`, `/dev/dxg` via `ROCDXG`) |
| **ROCm / Compiler** | ROCm `7.2.4` (`HIP 7.2.53211`, AMD clang `22.0.0git`), `-DGPU_TARGET=gfx1200` |
| **Nib Commit** | `v0.2.1` (`9de6314`) |
| **Model** | `Inkling-2-Mini-NVFP4-All.gguf` (`14.52 GiB` file; `13.98 GiB` VRAM with `--cpu-embd`, no MTP) |

### Build, Kernel Suite & Self-Test
- **Spill audit (`scripts/spill_audit.sh`):** **PASS — `0` SGPR spills, `0` VGPR spills, `0` private segment, `0` stubs** across all HIP objects (`gfx1200`).
- **Kernel verification (`./build/nib_kernel_test`):** **9/9 PASS** before, mid-run, and after all benchmarks (`Wave32 WMMA cos=1.000000`, `Dense SoA-Q8/QKV3 cos=0.9999999`, `Grouped MoE NVFP4/Q8 cos=0.9999999/0.9999998`, `FA Prefill Q8_0/Q4_1/Q4_0 cos=1.0000000`, `Sparse Quest+FA Q8/Q4 cos=0.9999999`, `Vision WMMA cos=1.0000000`, `Cross-WG Coherence 16 WGs × 200 rounds: 0 errors`, `Cooperative Co-Residency Safety 16 MPs: 0 NaNs / 0 deadlocks`).
- **Co-residency & HIP Graph (`wgp=16`):** Startup logged `wo_b=2 wo_c=2 wo_c_h2=3 mix_rms_qkv=4 a_qkv=4 a_v3=4` and `qkv_384=7`, capturing a **452-node** static decode HIP graph (co-residency fallback path for 16 WGPs vs. 404 nodes on 32-WGP `gfx1201`).
- **Parity (`nib selftest`):**
  - `q8_0` KV: **PASS** (`next_pf=111 next_dec=111`, `cos_sim = 0.9999909`, `4/4` sampler tests pass, repack `max_abs_err = 0.000000`).
  - `q4_0` / `q4_1` KV note: `q4_0` reports `cos_sim = 0.9980436` and `q4_1` reports `0.9979789` (slightly below the `0.999` strict prefill-vs-decode selftest gate, deterministic across runs and `--cpu-embd` settings). These values are **bit-identical** to RX 9070 XT (`gfx1201`) on both `v0.2.0` and `v0.2.1` — a pre-existing RDNA4-wide 4-bit KV tolerance difference, not specific to `gfx1200` or the 16-WGP fallback path.

---

## 2. Non-MTP Microbenchmarks (`nib bench`, `NVFP4-All`)

Config: `-m Inkling-2-Mini-NVFP4-All.gguf -c 73728 --kv-type q4_0 --sparse-pages 224 --cpu-embd -p 512 -n 128` (`15.06 GiB` total engine VRAM = `13.98 GiB` weights + `1.08 GiB` KV/scratch; `rc=0`, `0` hangs).

| Depth | `pp512` Prefill (`tok/s`) | `tg128` Decode (`tok/s`) |
|---:|---:|---:|
| **`0`** | **`2,851.59 ± 15.75`** | **`55.77 ± 0.27`** |
| **`8,192`** | `2,541.10 ± 12.05` | `54.55 ± 0.32` |
| **`16,384`** | `2,360.12 ± 13.55` | `53.82 ± 0.25` |
| **`32,768`** | `2,200.74 ± 4.67` | `53.75 ± 0.28` |
| **`49,152`** | `2,036.51 ± 8.15` | `53.71 ± 0.25` |
| **`65,536`** | `2,240.85 ± 15.64` | `53.72 ± 0.25` |
| **`72,704`** | **`2,281.74 ± 0.80`** | **`53.74 ± 0.26`** |

Decode remains flat within `0.1 tok/s` from `16K → 72.7K` (`53.82 → 53.74 tok/s`), and prefill rebounds above `64K` (`2,037 → 2,282 tok/s`) once Quest `--sparse-pages 224` activates.

---

## 3. BetterBench `0.6.1` End-to-End API Results (`nib serve`)

Measured on `127.0.0.1` (zero WAN RTT, corpus `v1.0`, `--greedy`, cold nonce prefix cache, `NVFP4-All` + `--cpu-embd`, no MTP).

### Single-Stream Decode (8 Categories, 40/40 OK per run)

| Category | `s1 -c 73728 q4_0` (`tok/s`) | TTFT p50 (`ms`) | `s4 -c 49152 q4_0` (`tok/s`) | TTFT p50 (`ms`) |
|---|---:|---:|---:|---:|
| `chat` | `56.6` | `134.3` | `56.5` | `134.6` |
| `summarization` | `56.6` | `137.1` | `56.5` | `137.2` |
| `json` | `56.5` | `128.1` | `56.6` | `129.1` |
| `math` | `56.0` | `124.4` | `56.1` | `124.6` |
| `prose` | `56.0` | `122.1` | `55.9` | `122.5` |
| `file_edit` | `56.0` | `140.6` | `55.9` | `140.7` |
| `code` | `55.9` | `127.8` | `55.8` | `128.1` |
| `reasoning` | `55.9` | `128.9` | `55.8` | `128.3` |
| **Weighted Overall** | **`56.1`** | **`128.8`** | **`56.0`** | **`128.9`** |

### Multi-Slot Continuous Batching (`--slots 4 -c 49152 --kv-type q4_0`, `12K/slot`, `32/32 OK` per level)

| Concurrency | Completed | Aggregate (`tok/s`) | Per-Stream Median (`tok/s`) | TTFT p50 (`ms`) | Scaling vs. `c = 1` |
|--:|--:|---:|---:|---:|---:|
| **`c = 1`** | `32/32` | **`55.1`** | **`56.0`** | **`129.1`** | `1.00×` |
| **`c = 2`** | `32/32` | **`77.2`** | **`39.7`** | **`161.6`** | `1.40×` |
| **`c = 4`** | `32/32` | **`93.9`** | **`24.2`** | **`211.0`** | **`1.70×`** |

### Prompt Processing (Prefill) Ladder

| Actual Prompt Tokens | `s1 -c 73728` Prefill (`tok/s`) | TTFT p50 (`ms`) | `s4 -c 49152` Prefill (`tok/s`) | TTFT p50 (`ms`) |
|---:|---:|---:|---:|---:|
| `~2,035` | `2,282.6` | `887.1` | `2,223.1` | `923.0` |
| `~8,035` | **`2,424.5`** | `3,320.9` | **`2,417.6`** | `3,317.3` |
| `~16,075` | `2,407.5` | `6,665.2` | `2,416.5` | `6,665.0` |
| `~31,995` | `2,356.5` | `13,577.8` | `2,351.1` | `13,608.0` |
| `36,841` | — | — | **`2,331.7`** | `15,800.5` |
| `56,272` | **`2,242.6`** | `25,092.8` | — | — |

---

## 4. RDNA 4 Scaling: RX 9060 XT (`gfx1200`) vs. RX 9070 XT (`gfx1201`)

Same Nib `v0.2.1` build, same `NVFP4-All` quant, same ROCm `7.2.4` / WSL2 environment:

| Metric | RX 9060 XT (`32 CU`, `320 GB/s`) | RX 9070 XT (`64 CU`, `640 GB/s`) | 9060 XT / 9070 XT |
|---|---:|---:|---:|
| Max Clean Context (`s1` / `s4`, `q4_0`) | `73,728` / `49,152` | `73,728` / `49,152` | `100%` |
| BetterBench Single-Stream (`s1 73K`) | **`56.1 tok/s`** | `97.0 tok/s` | **`57.8%`** *(vs. 50% bus)* |
| BetterBench `c = 4` Aggregate (`s4 49K`) | **`93.9 tok/s`** | `163.8 tok/s` | **`57.3%`** |
| BetterBench Prefill (`8K` / `56K` real toks) | **`2,425` / `2,243 tok/s`** | `4,449` / `4,099 tok/s` | **`54.5%` / `54.7%`** *(vs. 50% CUs)* |
| `nib bench` Deep Decode (`tg128 @ 72.7K`) | **`53.74 tok/s`** | `93.40 tok/s` | **`57.5%`** |

---

## 5. Recommended RX 9060 XT (16 GB) Production Configs

Build with `-DGPU_TARGET=gfx1200` and use `Inkling-2-Mini-NVFP4-All.gguf` (DSpark MTP does not fit in 16 GB):

```bash
# Maximum single-user context (72K, 15.06 GiB VRAM)
./build/nib serve \
  -m models/Inkling-2-Mini-NVFP4-All.gguf \
  --slots 1 -c 73728 --kv-type q4_0 --sparse-pages 224 --cpu-embd \
  --host 127.0.0.1 --port 4444

# 4-way concurrent serving (48K total / 12K per slot, 93.9 tok/s aggregate)
./build/nib serve \
  -m models/Inkling-2-Mini-NVFP4-All.gguf \
  --slots 4 -c 49152 --kv-type q4_0 --sparse-pages 224 --cpu-embd \
  --host 127.0.0.1 --port 4444
```
