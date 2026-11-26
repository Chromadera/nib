<p align="center">
  <img src="docs/nib-logo.svg" alt="nib" width="220" />
</p>

# Nib

**Nib** is a native, zero-framework C++17 / AMD RDNA3 & RDNA4 HIP inference engine for **Inkling 2 Mini** (26B looped hybrid-attention MoE + Vision + DSpark MTP), built from scratch for **AMD RDNA3 (`gfx1100`–`gfx1103`) and RDNA4 (`gfx1200`, `gfx1201`) GPUs**.

It runs end-to-end on **16 GB, 20 GB, and 24–48 GB AMD GPUs** via three native-NVFP4 GGUF tiers (`NVFP4-Mixed`, `NVFP4-23`, `NVFP4-All`) with **zero SGPR/VGPR/private-segment register spills and zero hollow stubs** across every kernel object:
- **24–48 GB (`RX 7900 XTX`, `Radeon AI PRO R9700`, `W7900`, `W7800`):** Fits **full `262,848` context**, **`Q8_0` KV + centroid-damped Quest sparse attention (`--sparse-pages 224`)**, **lossless DSpark Multi-Token Prediction (MTP)**, and the **23-layer Vision Encoder** simultaneously inside **22.18–23.08 GiB true VRAM** (`--slots 1..8`) with a transient single-arena ViT workspace (`~132 MiB` at `1080p`, `~416 MiB` max at `4K`, freed immediately after each encode).
- **20 GB (`RX 7900 XT`):** With `NVFP4-23` (`14.99 GiB`, `0.00483` KLD), unlocks **full `262,848` `q4_1`/`q4_0` context + DSpark MTP across `--slots 1..8`** (`157.2` solo / `207.6 tok/s` `c = 8` agg) and **`4K` / `2560²` Vision + MTP at `262,848`** with `0 MB` GTT spill.
- **16 GB (`RX 9070 XT` RDNA4, `RX 7800 XT` RDNA3, `RX 9060 XT` RDNA4):** With `NVFP4-All` (`14.52 GiB`, `0.00529` KLD), runs up to **`73,728`** (`RX 9070 XT`, `5,092 tok/s` FP16 WMMA prefill, `96.2 tok/s` decode, `162.0 tok/s` `c = 4` agg) and **`69,632`** (`RX 7800 XT`, `97.9 tok/s` decode, `174.9 tok/s` `c = 8` agg) plus `1080p/1440p` Vision with `0 MB` excess GTT spill.

---

## Performance Benchmarks (RX 7900 XTX, 262k Context)

Measured natively on an **AMD Radeon RX 7900 XTX** with BetterBench (`0.6.1`, corpus `v1.0`, `--greedy`, cold nonce prefix cache, `--slots 8 -c 262848 --kv-type q8_0 --sparse-pages 224`, lossless DSpark MTP + Vision loaded, `23.08 GiB` total app VRAM):

- **Single-Stream Weighted Decode (`c = 1`)**: **186.8 tok/s** median (`TTFT p50: 69.0 ms`, `update p99: 23.2 ms`) via static HIP Graph + lossless DSpark MTP fast path
- **4-Way Parallel Streaming (`c = 4`)**: **223.9 tok/s aggregate** (**58.4 tok/s** per stream median, **104.7 ms** TTFT p50, `32/32` OK)
- **8-Way Parallel Streaming (`c = 8`)**: **268.0 tok/s aggregate** (**36.0 tok/s** per stream median, **138.2 ms** TTFT p50, `32/32` OK, `32k`/slot)
- **Peak Category Decode**: **223.8 tok/s** (`json`), **213.1 tok/s** (`file_edit`), **206.0 tok/s** (`math`), **191.2 tok/s** (`code`), **188.0 tok/s** (`summarization`)
- **262k Long-Context Prefill (`sparse-224`)**: **4,234.8 tok/s** peak (`8k`) → **3,645.3 tok/s** (`64k`, exact dense FA) → **3,389.5 tok/s** at **261,526 actual tokens** (`77.16 s` TTFT, exact WMMA Quest scorer)
- **256k Noisy Needle Recall (`--sparse-pages 224`)**: **3/3 exact needles** at **251,659 noisy tokens** (`3%`, `48%`, `91%` depth) at **152.4 tok/s** MTP decode

### Single-Stream Decode (`batch = 1`, lossless DSpark MTP, `--slots 8`, `sparse-224`)

| Category | TTFT p50 (ms) | Update p99 (ms) | Decode Median (tok/s) |
|---|--:|--:|--:|
| `json` | 68.1 | 24.0 | **223.8** |
| `file_edit` | 70.7 | 24.3 | **213.1** |
| `math` | 62.6 | 23.2 | **206.0** |
| `code` | 66.3 | 23.7 | **191.2** |
| `summarization` | 73.7 | 24.0 | **188.0** |
| `reasoning` | 72.0 | 22.3 | **175.3** |
| `chat` | 73.3 | 23.8 | **158.5** |
| `prose` | 67.5 | 21.3 | **137.9** |
| **Combined (weighted)** | **69.0** | **23.2** | **186.8** |

### Multi-Slot Continuous Batching Concurrency Sweep (`--slots 8`, `32k/slot` concurrent · `262k` solo)

| Concurrency | Completed | Aggregate (tok/s) | Per-Stream Median (tok/s) | TTFT p50 (ms) | Execution Path |
|--:|--:|--:|--:|--:|---|
| `c = 1` | 32/32 | **172.7** | **185.8** | **71.6** | Solo fast path: static HIP Graph + DSpark MTP (up to 262k ctx) |
| `c = 2` | 32/32 | **168.5** | **87.1** | **87.1** | Batched 2-slot static HIP Graph (1× weight stream) |
| `c = 4` | 32/32 | **223.9** | **58.4** | **104.7** | Batched 4-slot static HIP Graph (1× weight stream) |
| `c = 8` | 32/32 | **268.0** | **36.0** | **138.2** | Batched 8-slot static HIP Graph (1× weight stream) |

### Prompt Processing (Prefill) Ladder — 2k to 262k (`--sparse-pages 224`)

| Actual Prompt Tokens | TTFT p50 (ms) | Prefill Median (tok/s) | Attention Mode |
|--:|--:|--:|---|
| 2,079 | 523.1 | **3,974.6** | Exact dense FA |
| 8,054 | 1,902.0 | **4,234.8** | Exact dense FA |
| 16,061 | 3,891.2 | **4,127.6** | Exact dense FA |
| 32,142 | 8,099.3 | **3,968.4** | Exact dense FA |
| 64,042 | 17,568.3 | **3,645.3** | Exact dense FA |
| 128,168 | 36,611.9 | **3,500.8** | Centroid-damped Quest (`sp224`, WMMA scorer) |
| 261,526 | 77,157.4 | **3,389.5** | Centroid-damped Quest (`sp224`, WMMA scorer) |

Full BetterBench report (RX 7900 XTX): [`docs/betterbench_slots8_262k.html`](docs/betterbench_slots8_262k.html) · [`docs/betterbench_slots8_262k.json`](docs/betterbench_slots8_262k.json).

### Multi-GPU Real-Silicon Summary (RDNA3 & RDNA4, Full BetterBench `0.6.1`)

| GPU | Arch / VRAM | Quant & Config | Non-MTP `tg128` (d0 → deep) | BetterBench Single-Stream | Peak Aggregate | Prefill (`pp512` d0 → deep) | Full Report |
|---|---|---|---:|---:|---:|---:|---|
| **RX 7900 XTX** | `gfx1100` · 24 GB | `NVFP4-Mixed` · `s8 -c 262848 q8_0` + MTP | `~141 → 134 tok/s` | **`186.8 tok/s`** *(MTP)* | **`268.0 tok/s`** (`c = 8`) | `4,235 → 3,390 tok/s` (`261K`) | [`docs/rx7900xtx/`](docs/rx7900xtx/README.md) |
| **RX 7900 XT** | `gfx1100` · 20 GB | **`NVFP4-23`** · `s8 -c 262848 q4_1` + MTP | `126.0 → 117.6 tok/s` (`256K`) | **`157.2 tok/s`** *(MTP)* | **`207.6 tok/s`** (`c = 8`) | `4,479 → 3,218 tok/s` (`204K`) | [`docs/rx7900xt/`](docs/rx7900xt/README.md) · [`NVFP4-23`](docs/rx7900xt/nvfp4_23_2026-11-26/README.md) |
| **RX 9070 XT** | `gfx1201` · 16 GB | **`NVFP4-All`** · `s1 -c 73728` / `s4 -c 49152 q4_0` | `96.2 → 92.2 tok/s` (`72.7K`) | **`95.7 tok/s`** *(no MTP)* | **`162.0 tok/s`** (`c = 4`) | **`5,092 → 4,048 tok/s`** (`70K`) | [`docs/rx9070xt/`](docs/rx9070xt/README.md) |
| **RX 7800 XT** | `gfx1101` · 16 GB | **`NVFP4-All`** · `s1 -c 69632` / `s8 -c 12288` | `97.9 → 93.5 tok/s` (`68.6K`) | **`98.0 tok/s`** *(no MTP)* | **`174.9 tok/s`** (`c = 8`) | `3,200 → 2,365 tok/s` (`68.6K`) | [`docs/rx7800xt/`](docs/rx7800xt/README.md) |

---

## Capability & Long-Context Verification (`gbench`, RX 7900 XTX, `sparse-224`)

Verified end-to-end against [`google-deepmind/gbench`](https://github.com/google-deepmind/gbench) (`v0.5.0.post1`) over an **8h 12m** continuous soak (**2,289 requests**, **30.97M prompt tokens**, **2.80M completion tokens**, **0 HTTP/engine errors**, **0 empty replies**, **21.50 GiB** startup → **22.45 GiB** high-res multimodal peak VRAM with `--slots 1 -c 262848 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj`):

| Suite | Mode / Scope | Nib (`NVFP4-Mixed` + `Q8_0` `sp224`) | Median Decode | Reference / Notes |
|---|---|---:|---:|---|
| **`arc_agi`** | Thinking · 20 tasks (21 pairs × 2) | **85.00%** *(85.71% pair pass@2)* | **197.5 tok/s** | Official ARC-AGI v1: **84.5%** |
| **`screenspot`** | Thinking · 200 targets (ScreenSpot-v2) | **88.50%** (`177/200`) | **114.3 tok/s** | Official ScreenSpot-Pro: **76.5%** |
| **`mrcr`** | Thinking · 120 stratified (`4K–256K`) | **83.44%** pooled | **94.0 tok/s** | `2-needle 64K–128K`: **100.0%**; `4-needle 64K–128K`: **99.74%**; `8-needle 128K–256K`: **37.28%** *(skips >256K)* |
| **`mmmu_pro`** | Thinking · 50 Qs × 4 modes (200 reqs) | **69.00%** *(72.0% Vision / 66.0% Std-10)* | **145.9 tok/s** | Official full 1,730 Std-10: **74.9%** |
| **`longbench_v2_code`** | Thinking · **Full 50/50** Code Repo | **62.00%** non-CoT / **60.00%** CoT | **123.1 tok/s** | Official all-domain LongBench-v2: **66.9%** |
| **`gsm8k`** | Non-thinking greedy · 200 (8-shot CoT) | **93.00%** (`186/200`) | **138.5 tok/s** | Canonical `lm-eval` 8-shot CoT |
| **`ifeval`** | Non-thinking greedy · **Full 541/541** | **71.53%** strict / **77.08%** loose | **126.4 tok/s** | **79.38%** instr-strict / **83.69%** instr-loose |
| **`serving perf`** | `--preset quick` (`b=1`, Text / Vision) | **68.4 ms** / **313.9 ms** TTFT p50 | **123.6** / **95.6 tok/s** | TPOT p50: **7.96 ms** (Text) / **7.82 ms** (Vision) |
| **`golden`** | Smoke test (15 applicable cases) | **14 / 15 PASS** (**93.3%**) | — | `chat_template 4/4`, `tool_use 2/2`, `vision 1/1` |

### Quantization & Engine Fidelity (WikiText-2, `40 × 512`)

**Near-lossless to `Q8_0` across all three NVFP4 tiers** (`0.0033–0.0053` Mean KLD, `97.0–97.5%` same top-1 token while saving `9.49–11.39 GiB` vs. `Q8_0`), and **Nib** matches `Q8_0` perplexity to within `0.0027` (`6.7727` vs `6.7754`). Even the smallest **`NVFP4-All`** (`14.52 GiB`) delivers **nearly half the KLD** of `16.42 GiB` standard 4-bit GGUFs (`IQ4_NL`, `UD-Q4_K_XL`).

| Quant | NVFP4 / Q8_0 MoE | Disk Size | Mean KLD | Same Top-1 | WikiText-2 PPL | Reference |
|---|---|---:|---:|---:|---:|---|
| **Nib Pack (`NVFP4-Mixed`)** | `L0–19` / `L20–23` | **17.63 GB** (`16.42 GiB`) | **`0.0033`** *(3-way: `0.00347`)* | **`97.46%`** | **`6.7696`** *(Nib: **`6.7727`**)* | vs. `ggml-org` `Q8_0` (`40 × 512`) |
| **Nib Pack (`NVFP4-23`)** | `L0–22` / `L23` | **16.10 GB** (`14.99 GiB`) | **`0.00483`** | **`97.10%`** | **`6.7758`** | vs. `ggml-org` `Q8_0` (`40 × 512`) |
| **Nib Pack (`NVFP4-All`)** | `L0–23` / none | **15.59 GB** (`14.52 GiB`) | **`0.00529`** | **`96.98%`** | **`6.7791`** | vs. `ggml-org` `Q8_0` (`40 × 512`) |
| [`ggml-org` `IQ4_NL`](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) | standard 4-bit | 17.63 GB (`16.42 GiB`) | `0.0095` | `95.5%` | — | vs. `BF16` upcast (`40 × 512`) |
| [`unsloth` `UD-Q4_K_XL`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) | dynamic 4-bit | 17.63 GB (`16.42 GiB`) | `0.0107` | `98.0%` | — | vs. `UD-Q8_K_XL` (Unsloth held-out) |
| [`ggml-org` `Q8_0`](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) *(ref)* | all 8-bit | 27.82 GB (`25.91 GiB`) | `0.0030` | `97.6%` | `6.7754` | vs. `BF16` upcast (`40 × 512`) |

<details>
<summary>Full KLD distributions &amp; methodology</summary>

- **Setup**: `llama-perplexity --kl-divergence` on WikiText-2 test (`40` chunks × `512` ctx, `10,200` scored tokens), comparing all three NVFP4 GGUFs head-to-head against identical `ggml-org` `Inkling-2-Mini-Q8_0.gguf` reference logits (`PPL = 6.7754 ± 0.1661`), plus `nib ppl` (`q8_0` KV, `--sparse-pages 224`) on the identical 40 chunks.
- **`NVFP4-Mixed` (local XTX / 3-way control)**: Mean `0.003289 ± 0.000086` (`0.003470 ± 0.000166` 3-way) · Median `0.001262` (`0.001255`) · p95 `0.012395` (`0.012520`) · p99 `0.032347` (`0.032821`) · Log-PPL Corr **`99.86%`** · RMS Δp `1.416%` (`1.521%`).
- **`NVFP4-23` (`-1.43 GiB` vs. Mixed)**: Mean `0.004832 ± 0.000159` · Median `0.001933` · p90 `0.010844` · p95 `0.018106` · p99 `0.043524` · Log-PPL Corr **`99.84%`** · ΔPPL `+0.0097 ± 0.0095` · RMS Δp `1.757%`.
- **`NVFP4-All` (`-1.90 GiB` vs. Mixed)**: Mean `0.005289 ± 0.000161` · Median `0.002264` · p90 `0.011927` · p95 `0.019255` · p99 `0.046761` · Log-PPL Corr **`99.83%`** · ΔPPL `+0.0131 ± 0.0098` · RMS Δp `1.803%`.
- **Linear layer slope**: Quantizing `L20–22` adds `+0.00136` Mean KLD (`+0.00095` KLD/GiB); quantizing `L23` adds only **`+0.00046`** Mean KLD (`+0.00097` KLD/GiB), making `NVFP4-All` an easy win on 16 GB cards (`3–4×` more context and working Vision).
- **Note**: `BF16` (`52.2 GB`) is an upcast of Thinking Machines' native `NVFP4 (L0..19) + MXFP8 (L20..23)` release, and `Q8_0` sits at `0.0030` KLD against it. Cross-repo figures are shown for context only (`ggml-org` measures against `BF16`; this pack and Unsloth measure against 8-bit references; Unsloth evaluates on a separate held-out set).

</details>

Full breakdown (including MRCR per-bin heatmap, `nexusraven_api_eval`, `docvqa`, `chartqa`, and trace notes): [`docs/GBENCH_EVAL.md`](docs/GBENCH_EVAL.md).

---

## Architecture

- **Custom RDNA3 & RDNA4 HIP Kernels (`gfx1100`–`gfx1103`, `gfx1200`, `gfx1201`)**: Hand-written Wave32 kernels (`v_dot4_i32_i8`, `v_dot2_f32_f16`, native RDNA3 256-bit and RDNA4 128-bit FP16 WMMA prefill via [`include/nib_arch.h`](include/nib_arch.h), fused RMSNorm+QKV, fused MoE gate/up/down, split-K FlashAttention with relative bias and sinks, and arch-native L0/L1 cache coherence) with **WGP-aware co-residency guards** (safe across 48/42/35/32/30/16-WGP GPUs) and **0 register spills / 0 hollow stubs** verified via `scripts/spill_audit.sh`.
- **16-Byte-Aligned GPU SoA Repacking (Dynamic Per-Layer NVFP4)**: Loads `NVFP4-Mixed` (`L0..19` NVFP4), `NVFP4-23` (`L0..22` NVFP4), or `NVFP4-All` (`L0..23` NVFP4) directly via `mmap` and dynamically detects each MoE layer's precision (`NVFP4` or `Q8_0`) while repacking weights into coalesced Structure-of-Arrays layouts on GPU in ~1.6s.
- **Full-Graph Static HIP Graphs**: Single-token decode (288–404 nodes), batched MTP verification (`m=2..8`, 698–704 nodes), DSpark draft steps (4 static graphs), and per-slot-combination multi-stream decode (`B=2..8`) execute with near-zero CPU launch overhead.
- **Multi-Slot Continuous Batching (`--slots 1..8`)**: Zero-copy Global KV + Quest page partitioning (`65,536` tokens/slot at 4 slots, `32,768` tokens/slot at 8 slots, full `262,848` context on slot 0 when solo), per-slot SWA rings + short-conv histories + prefix KV caches, chunked prefill interleaving, and automatic switch between solo DSpark MTP and batched multi-slot HIP Graphs. Prefix caches cover **multimodal** prompts too: image segments are keyed by a 128-bit content hash, so a conversation that keeps re-sending the same image(s) replays the cached KV instead of re-running the ViT and prefill on every turn.
- **262k Centroid-Damped Quest Sparse Global Attention + SWA**: Sliding-window attention (`1536` cap) paired with native NoPE-basis centroid-damped Quest (`k_mean`, `k_rad`, $\alpha = 0.25$, up to 254 pages; recommended `--sparse-pages 224`) on global attention layers and FWHT-rotated `Q8_0` / `Q4_1` KV cache. Prefill runs **exact dense FA** up to 64K tokens and an exact per-token FP16 WMMA scorer (`>= 224` pages) beyond 64K.
- **Lossless DSpark MTP Speculative Decoding (Any Temperature)**: 3-layer DSpark drafter (`mtp.safetensors`) with a 32,834-token shortlist head (`models/draft_shortlist.bin`), adaptive confidence-threshold drafting (1–7 draft tokens/round), exact target-model verification + rollback, and lossless rejection sampling at both `temperature = 0` and `temperature > 0` (`top_k`, `min_p`, `top_p`) via a zero-spill GPU verify top-256 kernel.
- **Native Vision Encoder (Aliased Single-Arena Workspace)**: 23-layer ViT (`mmproj-Inkling-2-Mini-Q8_0.gguf`) wired directly into both the CLI and OpenAI multimodal chat API. Non-overlapping ViT activation buffers are packed into a single 256-byte-aligned transient workspace (`~132 MiB` at `1080p`, `~416 MiB` max at `4K`/`2560×2560` — `-56%` peak scratch, bit-identical) that is freed immediately after each encode (leaving only `11–36 MiB` for the projected embeddings), with graceful `HTTP 400` protection if an image exceeds free VRAM.
- **16–20 GB Tight-VRAM Awareness (`--cpu-embd` & `< 22 GiB` Auto-Defaults)**: Zero behavioral change on `>= 22 GiB` GPUs. On `< 22 GiB` cards (such as 20 GB RX 7900 XT or 16 GB RX 9070 XT / RX 7800 XT), unset flags automatically select `q4_1` KV, size `-c` to preserve safe headroom (including worst-case ViT workspace), and enable `--cpu-embd` (GPU-mapped pinned host RAM for `token_embd.weight`, saving `469.3 MiB` VRAM at `< 0.8%` plain decode delta while keeping `lm_head` in VRAM). Startup logs and `/health` report exact `llm + mtp + vision` VRAM, live `hipMemGetInfo` free VRAM, and PCIe GTT spillover warnings.
- **Built-in OpenAI-Compatible HTTP Server & Web UI**: Zero-dependency C++ HTTP/SSE server (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/api/activity`, `/api/logs`, `/api/outputs`, `/api/reset-kv`) with live per-slot telemetry and an embedded SPA (Activity, Playground Chat with HTML Live Preview + Save-As folder picker + Files drawer, Load Test, Models, Logs, Performance, Settings, Light/Dark + 11 theme presets).

---

## Hardware Requirements & Supported GPUs

| GPU | Arch | CUs / VRAM | Recommended Quant | Verification | Recommended Config |
|---|---|---:|---|---|---|
| **RX 7900 XTX** | `gfx1100` (RDNA3) | 96 CU / 24 GiB | `NVFP4-Mixed` | **Measured on hardware** ([report](docs/rx7900xtx/README.md)) | Full `262,848` ctx + `Q8_0` KV + `--sparse-pages 224` + MTP + 4K Vision (`22.18–23.08 GiB` @ `--slots 1..8`; WSL2 WDDM: `-c 196608` or `--cpu-embd`) |
| **RX 7900 XT** | `gfx1100` (RDNA3) | 84 CU / 20 GiB | **`NVFP4-23`** *(or `Mixed`)* | **Measured on hardware** ([Mixed](docs/rx7900xt/README.md) · [NVFP4-23](docs/rx7900xt/nvfp4_23_2026-11-26/README.md)) | **`NVFP4-23`:** full **`262,848`** `q4_1`/`q4_0` + MTP across `--slots 1, 4, 8` (`207.6 tok/s` `c = 8` agg) + **`4K`/`2560²` Vision + MTP at `262K` (`s1`)**; `q8_0` up to `256K` (`no MTP`) / `201K` (`+ MTP`). **`NVFP4-Mixed`:** auto-defaults `65K q4_1` `s4` + MTP + `4K` Vision (`128K` `1080p`) |
| **RX 9070 XT** | `gfx1201` (RDNA4) | 64 CU / 16 GiB | **`NVFP4-All`** | **Measured on hardware** ([report](docs/rx9070xt/README.md)) | Build with `-DGPU_TARGET=gfx1201`; up to **`73,728`** `q4_0` / `65,536` `q4_1` / `32,768` `q8_0` (`s1`, `96.2 tok/s` decode, `5,092 tok/s` prefill), **`--slots 4` @ `49,152`** (`162.0 tok/s` agg), `1440p` Vision @ `≤ 8K` *(no MTP on 16 GB)* |
| **RX 7800 XT** | `gfx1101` (RDNA3) | 60 CU / 16 GiB | **`NVFP4-All`** | **Measured on hardware** ([report](docs/rx7800xt/README.md)) | Build with `-DGPU_TARGET=gfx1101`; up to **`69,632`** `q4_0` / `65,536` `q4_1` / `32,768` `q8_0` (`s1`, `97.9 tok/s` decode), **`--slots 4` @ `45,056`** (`159.4 tok/s` agg), **`--slots 8` @ `12,288`** (`174.9 tok/s` agg), `1080p` Vision @ `≤ 20K` *(no MTP on 16 GB)* |
| **Radeon AI PRO R9700** | `gfx1201` (RDNA4) | 64 CU / 32 GiB | `NVFP4-Mixed` | **Silicon-verified `gfx1201`** (RX 9070 XT + [projection](docs/rx9070xt/README.md#3-theoretical-projection--amd-radeon-ai-pro-r9700-32-gb-gfx1201)) | Drop-in `-DGPU_TARGET=gfx1201`; fits full `262,848` `Q8_0` KV + `--sparse-pages 224` + MTP + 4K Vision + `--slots 8` (`~23.4 GiB` peak, ~8 GiB free) |
| **Radeon PRO W7900 / W7800** | `gfx1100` (RDNA3) | 96 / 70 CU · 48 / 32 GiB | `NVFP4-Mixed` | **Simulated** (`sim_gpu.sh`) | Full `262,848` ctx + `Q8_0` KV + `--sparse-pages 224` + MTP + 4K Vision (`22.18–23.08 GiB`) |
| **RX 9060 XT (16 GB)** | `gfx1200` (RDNA4) | 32 CU / 16 GiB | `NVFP4-All` | **Kernel-verified** (`nib_kernel_test` 9/9) | Build with `-DGPU_TARGET=gfx1200`; same 16 GB VRAM envelope as RX 9070 XT (`NVFP4-All`, `--cpu-embd`, no MTP) |

> **Note on RDNA3 & RDNA4 GPUs**: Four distinct AMD GPUs spanning RDNA3 and RDNA4 — **RX 7900 XTX (`gfx1100`, 24 GB)**, **RX 7900 XT (`gfx1100`, 20 GB)**, **RX 9070 XT (`gfx1201`, 16 GB)**, and **RX 7800 XT (`gfx1101`, 16 GB)** — have been verified and benchmarked end-to-end on real silicon (decode, full BetterBench `0.6.1` multi-slot concurrency, long-context prefill, and Vision), alongside native `gfx1200` kernel verification on **RX 9060 XT** (`./build/nib_kernel_test` 9/9 PASS, `cos = 0.9999998–1.0000000`, `0` coherence errors, `0` deadlocks). Multi-arch fatbin builds (`gfx1100;gfx1101;gfx1200;gfx1201`) pass `./scripts/spill_audit.sh` with **0 spills / 0 private bytes / 0 stubs across all 19 kernel objects** on both **ROCm 7.2.4 / AMD clang 22** and **HIP 7.16 / AMD clang 24**. See [`docs/rx9070xt/README.md`](docs/rx9070xt/README.md) for the **Radeon AI PRO R9700 (32 GB)** theoretical projection — real-hardware R9700 reports are welcome!

- **Vision VRAM**: `mmproj-Inkling-2-Mini-Q8_0.gguf` takes **`0.32 GiB`** at startup. During each image encode, [`VisionEncoder`](src/vision_encoder.cpp) allocates a single aliased workspace (**`~132 MiB`** at `1080p`, **`~234 MiB`** at `1440p`, **`~416 MiB`** max at `4K`/`2560×2560`) and frees it immediately after encoding (`0 MiB` retained workspace; only the `11–36 MiB` projected embedding buffer stays resident). Pass `--vision-keep-scratch` to keep the workspace resident across images.
- **16–20 GB Tight-VRAM Tuning**: Passing `--cpu-embd` places `token_embd.weight` (`469.3 MiB`) in GPU-mapped pinned host RAM (`lm_head` stays in VRAM) at `< 0.8%` plain decode cost; pass `--no-cpu-embd` to keep embeddings in VRAM on `< 22 GiB` cards.
- **System RAM**: **~390 MiB** steady-state RSS (**~860 MiB** with `--cpu-embd`; ~`2.0–2.5 GiB` transient during startup staging). Runs on **8 GiB** systems; **32 GiB+** keeps the `14.5–16.4 GiB` `mmap`'d GGUF warm in Linux page cache for fast (~2s) restarts.

---

## Building

### Prerequisites
- Linux x86_64 with AMD ROCm (`hipcc`, `amdhip64`, `llvm-objcopy`, `clang-offload-bundler`, `llvm-readobj`)
- CMake ≥ 3.21 and a C++17 host compiler
- AMD RDNA3 (`gfx1100`–`gfx1103`) or RDNA4 (`gfx1200`, `gfx1201`) GPU (default target `gfx1100`)

> **ROCm / Compiler note:** Zero-spill, zero-stub builds (`./scripts/spill_audit.sh` passing across all 19 kernel objects: 0 SGPR/VGPR spills, 0 private bytes, 0 hollow stubs) are verified on both **HIP 7.16 / AMD clang 24** and stable **ROCm 7.2.4 / AMD clang 22** (on Ubuntu 24.04 ROCm 7.2.4 PIE toolchains, pass `-DCMAKE_EXE_LINKER_FLAGS="-no-pie"` if clang emits non-PIE HIP host stubs).

### Compile & Spill Audit
```bash
# RDNA3 (default: gfx1100; use -DGPU_TARGET=gfx1101 for RX 7800 XT)
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./scripts/spill_audit.sh
./build/nib_kernel_test
```

Build for **AMD RDNA4** (e.g. Radeon AI PRO R9700 / RX 9070 XT = `gfx1201`, RX 9060 XT = `gfx1200`), or a multi-arch fatbin:
```bash
# Single RDNA4 target
cmake -B build -DROCM_PATH=/opt/rocm -DGPU_TARGET=gfx1201
cmake --build build -j
./scripts/spill_audit.sh
./build/nib_kernel_test

# Multi-arch RDNA3 + RDNA4 fatbin (all 4 verified targets)
cmake -B build -DGPU_TARGET="gfx1100;gfx1101;gfx1200;gfx1201"
cmake --build build -j
./scripts/spill_audit.sh
```

---

## Weights Setup

Download **only the GGUF tier for your GPU's VRAM** from Hugging Face ([`degenmeowster/Inkling-2-Mini-Nib-Pack`](https://huggingface.co/degenmeowster/Inkling-2-Mini-Nib-Pack)) into `models/`:

```bash
# 24–48 GB GPUs (RX 7900 XTX, R9700, W7900, W7800) — NVFP4-Mixed + MTP + Vision
hf download degenmeowster/Inkling-2-Mini-Nib-Pack --local-dir models \
  --include "Inkling-2-Mini-NVFP4-Mixed.gguf" "mtp.safetensors" "draft_shortlist.bin" "mmproj-Inkling-2-Mini-Q8_0.gguf"

# 20 GB GPUs (RX 7900 XT) — NVFP4-23 + MTP + Vision
hf download degenmeowster/Inkling-2-Mini-Nib-Pack --local-dir models \
  --include "Inkling-2-Mini-NVFP4-23.gguf" "mtp.safetensors" "draft_shortlist.bin" "mmproj-Inkling-2-Mini-Q8_0.gguf"

# 16 GB GPUs (RX 9070 XT, RX 7800 XT, RX 9060 XT) — NVFP4-All + Vision
hf download degenmeowster/Inkling-2-Mini-Nib-Pack --local-dir models \
  --include "Inkling-2-Mini-NVFP4-All.gguf" "mmproj-Inkling-2-Mini-Q8_0.gguf"
```

Or place/symlink the files manually in `models/` (or pass paths via CLI flags `-m`, `--mtp`, `--mmproj` or env vars `NIB_MODEL`, `NIB_MMPROJ`):

| File | Precision | Size | Mean KLD | Target VRAM Tier |
|---|---|--:|--:|---|
| `models/Inkling-2-Mini-NVFP4-Mixed.gguf` | NVFP4 (`L0..19` MoE) + Q8_0 (`L20..23` MoE, attn, embd, head) | 16.42 GiB | `0.0033` | **24–48 GB** (native precision grid) |
| `models/Inkling-2-Mini-NVFP4-23.gguf` | NVFP4 (`L0..22` MoE) + Q8_0 (`L23` MoE, attn, embd, head) | 14.99 GiB | `0.00483` | **20 GB** (`-1.43 GiB`; unlocks 262K + MTP + Vision) |
| `models/Inkling-2-Mini-NVFP4-All.gguf` | NVFP4 (`L0..23` MoE) + Q8_0 (attn, embd, head) | 14.52 GiB | `0.00529` | **16 GB** (`-1.90 GiB`; unlocks 70–74K + Vision) |
| `models/mtp.safetensors` | BF16 (repacked to SoA Q8_0 on load) | 606 MiB | — | Official DSpark MTP drafter (`20 GB+`) |
| `models/draft_shortlist.bin` | int32 vocab map (32,834 rows) | 256 KiB | — | Included in repo; looked up next to `mtp.safetensors` or `models/` |
| `models/mmproj-Inkling-2-Mini-Q8_0.gguf` | Q8_0 + F16/F32 | 330 MiB | — | Optional 23-layer Vision Encoder |

> **Why three NVFP4 tiers?** Thinking Machines trained and released `Inkling-2-Mini` natively with layers `0..19` MoE experts in NVFP4 and layers `20..23` in MXFP8. `NVFP4-Mixed` matches that native grid for 24 GB+ GPUs, while `NVFP4-23` (`-1.43 GiB`) and `NVFP4-All` (`-1.90 GiB`) extend NVFP4 into `L20..22` / `L20..23` at minimal KLD cost (`0.00483` / `0.00529`, still nearly half the KLD of standard 4-bit GGUFs) so 20 GB and 16 GB RDNA3/RDNA4 cards run smoothly with zero GTT spillover.

---

## Usage

### 1. Start the OpenAI API Server & Web UI
```bash
# 24–48 GB GPUs (RX 7900 XTX, R9700, W7900): NVFP4-Mixed + 262K Q8_0 + MTP + 4K Vision
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
./build/nib serve \
  -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
  --mtp models/mtp.safetensors \
  --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf \
  --slots 4 -c 262848 --kv-type q8_0 --sparse-pages 224 \
  --host 127.0.0.1 --port 4444

# 20 GB GPUs (RX 7900 XT): NVFP4-23 + full 262K Q4_1 + MTP + 4K Vision (--cpu-embd auto-on)
./build/nib serve \
  -m models/Inkling-2-Mini-NVFP4-23.gguf \
  --mtp models/mtp.safetensors \
  --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf \
  --slots 1 -c 262848 --kv-type q4_1 --sparse-pages 224 \
  --host 127.0.0.1 --port 4444

# 16 GB GPUs (RX 9070 XT, RX 7800 XT): NVFP4-All up to 64K–72K solo or 48K @ 4 slots (no MTP)
./build/nib serve \
  -m models/Inkling-2-Mini-NVFP4-All.gguf \
  --slots 1 -c 65536 --kv-type q4_0 --sparse-pages 224 --cpu-embd \
  --host 127.0.0.1 --port 4444
```
Open **http://127.0.0.1:4444/** for the built-in Web UI, or connect any OpenAI-compatible client to `http://127.0.0.1:4444/v1`.

### 2. CLI Generation (with DSpark MTP)
```bash
./build/nib gen \
  -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
  --mtp models/mtp.safetensors \
  --prompt-file tests/prompts/coding.txt \
  -n 512 -c 262848 --kv-type q8_0 --sparse-pages 224
```

### 3. Vision Inference
```bash
./build/nib vision \
  -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
  --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf \
  --image tests/test_vision_card.png \
  --prompt "Describe this image in detail."
```

### 4. Built-in Microbenchmarks, Self-Test & GPU Simulator
```bash
./build/nib bench -m models/Inkling-2-Mini-NVFP4-Mixed.gguf -d 0,8192,32768 -p 512 -n 128 --sparse-pages 224
./build/nib selftest -m models/Inkling-2-Mini-NVFP4-Mixed.gguf --mtp models/mtp.safetensors --sparse-pages 224 --bench-mtp
./scripts/sim_gpu.sh 7900xt ./build/nib selftest -m models/Inkling-2-Mini-NVFP4-Mixed.gguf --mtp models/mtp.safetensors -c 131072 --kv-type q4_1 --sparse-pages 224
```

---

## Repository Layout

```text
nib/
├── CMakeLists.txt           # Zero-dependency CMake + hipcc build (RDNA3 & RDNA4)
├── LICENSE                  # Apache-2.0
├── README.md
├── docs/                    # Logo, BetterBench, gBench, rx7900xtx/, rx7900xt/, rx9070xt/, rx7800xt/
├── include/                 # Engine, arch abstractions (nib_arch.h), sparse-global, MTP headers
├── models/                  # draft_shortlist.bin (large weights gitignored)
├── scripts/
│   ├── sim_gpu.sh           # RDNA3 & RDNA4 multi-GPU VRAM/CU simulator wrapper
│   ├── sim_gpu_shim.c       # LD_PRELOAD HIP property & VRAM-cap shim
│   └── spill_audit.sh       # Automated SGPR/VGPR/private-segment & stub audit
├── src/                     # C++17 host runtime, HTTP server/UI, RDNA3/RDNA4 HIP kernels
├── tests/                   # Parity prompts, vision fixture, & test_rdna_kernels.hip suite
└── third_party/
    └── stb_image.h          # Single-header image decoder for vision
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
