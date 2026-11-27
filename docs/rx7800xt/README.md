# AMD Radeon RX 7800 XT (16 GB, `gfx1101`) — Real-Hardware Nib Evaluation

Real-silicon verification of **Nib** on **AMD Radeon RX 7800 XT** (`gfx1101` / Navi 32, `60 CU` / `30 WGP`, `256-bit`, `624 GB/s`, `64 MiB` L3 Infinity Cache, `16 GB` GDDR6) evaluating **`Inkling-2-Mini-NVFP4-All.gguf`** (`14.52 GiB`) and **`Inkling-2-Mini-NVFP4-23.gguf`** (`14.99 GiB`).

Full report, `benchmarks.json`, scripts, and BetterBench `0.6.1` HTML/JSON reports live in **[`nvfp4_23_all_2026-11-26/README.md`](nvfp4_23_all_2026-11-26/README.md)**.

## Highlights (`-DGPU_TARGET=gfx1101`, ROCm 7.2.4, Windows/WSL2 ROCDXG)

- **Zero spills / zero stubs:** `./scripts/spill_audit.sh` passes with `0` SGPR spills, `0` VGPR spills, `0` private segment, `0` stubs across all **19** HIP objects on `gfx1101` (`wgp=30` co-residency guards, `404`-node static decode HIP graph).
- **`NVFP4-All` is the 16 GB quant (`13.98 GiB` weights in VRAM with `--cpu-embd`):**
  - **Max clean context (`0 MB` excess GTT spill):** **`69,632`** (`q4_0`), **`65,536`** (`q4_1`), **`32,768`** (`q8_0`) at `--slots 1` (vs. `20,480` / `16,384` / `8,192` on `NVFP4-23`).
  - **Multi-slot concurrency:** **`--slots 4` @ `45,056` (`q4_0`)** → **`159.4 tok/s`** aggregate at `c = 4`; **`--slots 8` @ `12,288` (`q4_1`)** → **`174.9 tok/s`** aggregate at `c = 8` (`22.9 tok/s`/stream, `32/32 OK`).
  - **Single-stream speed:** **`97.9 → 93.5 tok/s`** non-MTP `tg128` (`d0 → 68.6K`), **`98.0 tok/s`** full BetterBench weighted median, **`3,200 → 2,365 tok/s`** `pp512` prefill (`2,471 tok/s` at `53K` real tokens).
  - **Vision (`--mmproj`):** Works on `NVFP4-All` up to **`1080p` @ `≤ 20K`** and **`1440p` @ `≤ 8K`** (`0 MB` GTT spill).
  - **DSpark MTP (`--mtp`):** Does not fit in 16 GB (`DSparkDrafter: hipMalloc failed`).

---

## Nib `v0.2.1` Rerun (`2026-11-27`, `NVFP4-All`)

Re-verified on commit `9de6314` (`-DGPU_TARGET=gfx1101`, ROCm `7.2.4`, Windows/WSL2 ROCDXG). Build, `spill_audit.sh` (`0/0/0/0`), `nib_kernel_test` (**9/9**), and `nib selftest` (`q4_1` `cos_sim = 0.9996072`, `q8_0` PASS) all passed cleanly. See [`v0.2.1_2026-11-27/NOTES.md`](v0.2.1_2026-11-27/NOTES.md).

### Non-MTP `nib bench` (`-p 512 -n 128 --sparse-pages 224 --cpu-embd`)

| Depth | `q4_0 @ 69,632` `pp512` / `tg128` (`15.03 GiB`) | `q4_1 @ 65,536` `pp512` / `tg128` (`15.06 GiB`) |
|---:|---:|---:|
| **`0`** | `3,250.7` / **`97.87`** | `3,201.2` / **`98.46`** |
| **`8,192`** | `2,823.3` / `94.91` | `2,809.0` / `94.77` |
| **`16,384`** | `2,644.7` / `93.47` | `2,621.2` / `93.11` |
| **`32,768`** | `2,380.0` / `93.17` | `2,380.7` / `92.80` |
| **`49,152`** | `2,172.2` / `93.26` | `2,152.8` / `92.90` |
| **`64.5K / 65.5K`** | `2,295.0` / `93.27` | `1,989.2` / `92.79` |
| **`68,608`** | **`2,341.4`** / **`93.12`** | — |

### BetterBench `0.6.1` (`nib serve`, `32/32 OK` Every Level)

| Config (`NVFP4-All`, `--cpu-embd`) | Single-Stream Weighted | Concurrency Aggregate (`c = 1 / 2 / 4 / 8`) | Deepest Prefill (`tok/s`) | Report |
|---|---:|---:|---:|---|
| **`--slots 1 -c 69632 q4_0`** | **`97.8 tok/s`** (`89.4 ms` TTFT) | `95.9` (`c = 1`) | **`2,451.5`** @ `52,994` toks (`2,737.7` @ `8K`) | [`html`](v0.2.1_2026-11-27/bb_All_s1_q4_0_c69632.html) · [`json`](v0.2.1_2026-11-27/bb_All_s1_q4_0_c69632.json) |
| **`--slots 4 -c 45056 q4_0`** | **`97.9 tok/s`** (`90.5 ms` TTFT) | **`95.8` / `130.2` / `159.1`** (`41.0`/stream @ `c = 4`) | `2,587.8` @ `33,739` toks | [`html`](v0.2.1_2026-11-27/bb_All_s4_q4_0_c45056.html) · [`json`](v0.2.1_2026-11-27/bb_All_s4_q4_0_c45056.json) |
| **`--slots 8 -c 12288 q4_1`** | **`97.8 tok/s`** (`89.9 ms` TTFT) | **`95.9` / `130.4` / `159.2` / `174.8`** (`22.8`/stream @ `c = 8`) | `2,726.9` @ `8,054` toks | [`html`](v0.2.1_2026-11-27/bb_All_s8_q4_1_c12288.html) · [`json`](v0.2.1_2026-11-27/bb_All_s8_q4_1_c12288.json) |

