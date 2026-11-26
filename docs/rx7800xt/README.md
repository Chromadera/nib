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
