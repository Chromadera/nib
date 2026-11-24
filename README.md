<p align="center">
  <img src="docs/nib-logo.svg" alt="nib" width="220" />
</p>

# Nib

**Nib** is a native, zero-framework C++17 / AMD RDNA3 HIP inference engine for **Inkling 2 Mini** (26B looped hybrid-attention MoE + Vision + DSpark MTP), built from scratch for **AMD RDNA3 (`gfx1100`, Navi 31) GPUs**.

It fits **full `262,848` context**, **`Q8_0` KV + centroid-damped Quest sparse attention (`--sparse-pages 224`)**, **lossless DSpark Multi-Token Prediction (MTP)**, and the **23-layer Vision Encoder** simultaneously inside **21.50 GiB VRAM** (`--slots 1`), **21.89 GiB** (`--slots 4`), or **22.40 GiB** (`--slots 8`), with **zero SGPR/VGPR/private-segment register spills** across every GPU kernel.

---

## Performance Benchmarks (RX 7900 XTX, 262k Context)

Measured natively on an **AMD Radeon RX 7900 XTX** with BetterBench (`0.6.1`, corpus `v1.0`, `--greedy`, cold nonce prefix cache, `--slots 8 -c 262848 --kv-type q8_0 --sparse-pages 224`, lossless DSpark MTP + Vision loaded, `22.40 GiB` VRAM):

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

Full BetterBench report: [`docs/betterbench_slots8_262k.html`](docs/betterbench_slots8_262k.html) · [`docs/betterbench_slots8_262k.json`](docs/betterbench_slots8_262k.json).

---

## Capability & Long-Context Verification (`gbench`, RX 7900 XTX, `sparse-224`)

Verified end-to-end against [`google-deepmind/gbench`](https://github.com/google-deepmind/gbench) (`v0.5.0.post1`) over an **8h 12m** continuous soak (**2,289 requests**, **30.97M prompt tokens**, **2.80M completion tokens**, **0 HTTP/engine errors**, **0 empty replies**, steady **22.45 GiB** VRAM with `--slots 1 -c 262848 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj`):

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

### Quantization & Engine Fidelity (WikiText-2)

**Near-lossless to `Q8_0` (`0.0033` KLD, `97.4%` same top token) while saving 9.48 GiB**, and **Nib** matches `Q8_0` perplexity to within `0.0027` (`6.7727` vs `6.7754`).

| Quant | Size | Mean KLD | Same Top-1 | WikiText-2 PPL | Reference |
|---|---:|---:|---:|---:|---|
| **Nib Pack (`NVFP4-Mixed`)** | **17.63 GB** (`16.42 GiB`) | **`0.0033`** | **`97.4%`** | **`6.7696`** *(Nib: **`6.7727`**)* | vs. `ggml-org` `Q8_0` (`40 × 512`) |
| [`ggml-org` `IQ4_NL`](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) | 17.63 GB (`16.42 GiB`) | `0.0095` | `95.5%` | — | vs. `BF16` upcast (`40 × 512`) |
| [`unsloth` `UD-Q4_K_XL`](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) | 17.63 GB (`16.42 GiB`) | `0.0107` | `98.0%` | — | vs. `UD-Q8_K_XL` (Unsloth held-out) |
| [`ggml-org` `Q8_0`](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) *(ref)* | 27.82 GB (`25.91 GiB`) | `0.0030` | `97.6%` | `6.7754` | vs. `BF16` upcast (`40 × 512`) |

<details>
<summary>Full KLD distribution &amp; methodology</summary>

- **Setup**: `llama-perplexity --kl-divergence` on WikiText-2 test (`40` chunks × `512` ctx, `10,200` scored tokens), comparing `Inkling-2-Mini-NVFP4-Mixed.gguf` against `ggml-org` `Inkling-2-Mini-Q8_0.gguf`, plus `nib ppl` (`q8_0` KV, `--sparse-pages 224`) on the identical 40 chunks.
- **KLD distribution**: Mean `0.003289 ± 0.000086` · Median `0.001262` · p90 `0.007472` · p95 `0.012395` · p99 `0.032347` · Max `0.385739`.
- **Probability & PPL agreement**: Log-PPL correlation **`99.86%`** · ΔPPL `+0.0035 ± 0.0087` · Mean Δp `+0.011%` · Median Δp `0.000%` · RMS Δp `1.416%`.
- **Note**: `BF16` (`52.2 GB`) is an upcast of Thinking Machines' native `NVFP4 (L0..19) + MXFP8 (L20..23)` release, and `Q8_0` sits at `0.0030` KLD against it. Storing `L0..19` directly in native `NVFP4` lets Nib match the base checkpoint's precision grid on RDNA3, whereas universal GGUFs (`IQ4_NL`, `UD-Q4_K_XL`) target broad cross-backend compatibility across standard `llama.cpp` formats. Cross-repo figures are shown for context only (`ggml-org` measures against `BF16`; this pack and Unsloth measure against 8-bit references; Unsloth evaluates on a separate held-out set).

</details>

Full breakdown (including MRCR per-bin heatmap, `nexusraven_api_eval`, `docvqa`, `chartqa`, and trace notes): [`docs/GBENCH_EVAL.md`](docs/GBENCH_EVAL.md).

---

## Architecture

- **Custom `gfx1100` RDNA3 HIP Kernels**: Hand-written Wave32/Wave64 kernels (`v_dot4_i32_i8`, `v_dot2_f32_f16`, WMMA prefill, fused RMSNorm+QKV, fused MoE gate/up/down, split-K FlashAttention with relative bias and sinks) with **WGP-aware co-residency guards** (safe across 96-CU, 84-CU, and 70-CU `gfx1100` GPUs) and **0 register spills** verified via `scripts/spill_audit.sh`.
- **16-Byte-Aligned GPU SoA Repacking**: Loads `Inkling-2-Mini-NVFP4-Mixed.gguf` directly via `mmap` and repacks `NVFP4` (layers `0..19` MoE experts) and `Q8_0` (layers `20..23` experts + attention/embeddings/LM head) into coalesced Structure-of-Arrays layouts on GPU in ~1.6s.
- **Full-Graph Static HIP Graphs**: Single-token decode (288–292 nodes), batched MTP verification (`m=2..8`, 698–704 nodes), DSpark draft steps (4 static graphs), and per-slot-combination multi-stream decode (`B=2..8`) execute with near-zero CPU launch overhead.
- **Multi-Slot Continuous Batching (`--slots 1..8`)**: Zero-copy Global KV + Quest page partitioning (`65,536` tokens/slot at 4 slots, `32,768` tokens/slot at 8 slots, full `262,848` context on slot 0 when solo), per-slot SWA rings + short-conv histories + prefix KV caches, chunked prefill interleaving, and automatic switch between solo DSpark MTP and batched multi-slot HIP Graphs.
- **262k Centroid-Damped Quest Sparse Global Attention + SWA**: Sliding-window attention (`1536` cap) paired with native NoPE-basis centroid-damped Quest (`k_mean`, `k_rad`, $\alpha = 0.25$, up to 254 pages; recommended `--sparse-pages 224`) on global attention layers and FWHT-rotated `Q8_0` / `Q4_1` KV cache. Prefill runs **exact dense FA** up to 64K tokens and an exact per-token RDNA3 FP16 WMMA scorer (`>= 224` pages) beyond 64K.
- **Lossless DSpark MTP Speculative Decoding (Any Temperature)**: 3-layer DSpark drafter (`mtp.safetensors`) with a 32,834-token shortlist head (`models/draft_shortlist.bin`), adaptive confidence-threshold drafting (1–7 draft tokens/round), exact target-model verification + rollback, and lossless rejection sampling at both `temperature = 0` and `temperature > 0` (`top_k`, `min_p`, `top_p`) via a zero-spill GPU verify top-256 kernel.
- **Native Vision Encoder**: 23-layer ViT (`mmproj-Inkling-2-Mini-Q8_0.gguf`) wired directly into both the CLI and OpenAI multimodal chat API.
- **Built-in OpenAI-Compatible HTTP Server & Web UI**: Zero-dependency C++ HTTP/SSE server (`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/health`, `/api/activity`, `/api/logs`, `/api/outputs`, `/api/reset-kv`) with live per-slot telemetry and an embedded SPA (Activity, Playground Chat with HTML Live Preview + Save-As folder picker + Files drawer, Load Test, Models, Logs, Performance, Settings, Light/Dark + 11 theme presets).

---

## Hardware Requirements & Supported GPUs

| GPU (`gfx1100` / Navi 31) | CUs / VRAM | Verification | Recommended Config |
|---|---:|---|---|
| **RX 7900 XTX** | 96 CU / 24 GiB | **Measured on hardware** | Full `262,848` ctx + `Q8_0` KV + `--sparse-pages 224` + MTP + Vision (`21.50 GiB` @ `--slots 1`, `21.89 GiB` @ `--slots 4`, `22.40 GiB` @ `--slots 8`) |
| **Radeon PRO W7900** | 96 CU / 48 GiB | **Simulated** (`sim_gpu.sh`) | Full `262,848` ctx + `Q8_0` KV + `--sparse-pages 224` + MTP + Vision (`21.50–22.40 GiB`) |
| **Radeon PRO W7800 32 GB** | 70 CU / 32 GiB | **Simulated** (`sim_gpu.sh`) | Full `262,848` ctx + `Q8_0` KV + `--sparse-pages 224` + MTP + Vision (`21.50–22.40 GiB`) |
| **RX 7900 XT** | 84 CU / 20 GiB | **Simulated** (`sim_gpu.sh`) | Up to `131,072` ctx + `--kv-type q4_1 --sparse-pages 224 --mtp --no-vision` (`~18.35 GiB`) |
| **RX 7900 GRE / RX 7800 XT** | 16 GiB | **Not supported** | Does not fit the `~16.4 GiB` `NVFP4-Mixed` base weights |

> **Note on other RDNA3 GPUs**: All benchmarks were measured natively on an **RX 7900 XTX**. Because I could not find anywhere to rent other RDNA3 cards (`W7900`, `W7800`, `7900 XT`), compatibility, VRAM budgets, and WGP co-residency guards for those GPUs were validated using an `LD_PRELOAD` HIP CU/VRAM simulator ([`scripts/sim_gpu.sh`](scripts/sim_gpu.sh)) on the 7900 XTX. Real-hardware reports from owners of those cards are welcome!

- **System RAM**: **~390 MiB** steady-state RSS (~`2.0–2.5 GiB` transient during startup staging). Runs on **8 GiB** systems; **32 GiB+** keeps the `16.4 GiB` `mmap`'d GGUF warm in Linux page cache for fast (~2s) restarts.

---

## Building

### Prerequisites
- Linux x86_64 with AMD ROCm (`hipcc`, `amdhip64`, `llvm-objcopy`, `clang-offload-bundler`, `llvm-readobj`)
- CMake ≥ 3.21 and a C++17 host compiler
- AMD RDNA3 GPU (default target `gfx1100`)

### Compile & Spill Audit
```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./scripts/spill_audit.sh
```

Override ROCm path or GPU target if needed:
```bash
cmake -B build -DROCM_PATH=/opt/rocm -DGPU_TARGET=gfx1100
```

---

## Weights Setup

Download the ready-to-run pack directly from Hugging Face ([`degenmeowster/Inkling-2-Mini-Nib-Pack`](https://huggingface.co/degenmeowster/Inkling-2-Mini-Nib-Pack)) into `models/`:

```bash
hf download degenmeowster/Inkling-2-Mini-Nib-Pack --local-dir models
```

Or place/symlink the files manually in `models/` (or pass paths via CLI flags `-m`, `--mtp`, `--mmproj` or env vars `NIB_MODEL`, `NIB_MMPROJ`):

| File | Precision | Size | Purpose |
|---|---|--:|---|
| `models/Inkling-2-Mini-NVFP4-Mixed.gguf` | NVFP4 (`L0..19` MoE) + Q8_0 (`L20..23` MoE, attn, embd, head) | ~16.4 GiB | Main LLM |
| `models/mtp.safetensors` | BF16 (repacked to SoA Q8_0 on load) | 607 MiB | Official DSpark MTP drafter |
| `models/draft_shortlist.bin` | int32 vocab map (32,834 rows) | 256 KiB | Included in repo |
| `models/mmproj-Inkling-2-Mini-Q8_0.gguf` | Q8_0 + F16/F32 | 330 MiB | Optional vision encoder |

> **Why NVFP4-Mixed?** Thinking Machines trained and released `Inkling-2-Mini` natively with layers `0..19` MoE experts in NVFP4 and layers `20..23` in MXFP8. Matching that precision split in GGUF avoids bloating those 20 layers to Q8_0/BF16 while preserving native quality and fitting `262k` context + MTP + Vision into 24 GiB VRAM.

---

## Usage

### 1. Start the OpenAI API Server & Web UI
```bash
HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0 \
./build/nib serve \
  -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
  --mtp models/mtp.safetensors \
  --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf \
  --slots 1 \
  -c 262848 \
  --kv-type q8_0 \
  --sparse-pages 224 \
  --host 127.0.0.1 \
  --port 4444
```
Open **http://127.0.0.1:4444/** for the built-in Web UI, or connect any OpenAI-compatible client to `http://127.0.0.1:4444/v1`. Pass `--slots 4` to enable 4-way concurrent streaming (`+0.39 GiB` VRAM, `65k/slot` concurrent, full `262k` when solo).

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
├── CMakeLists.txt           # Zero-dependency CMake + hipcc build
├── LICENSE                  # Apache-2.0
├── README.md
├── docs/                    # Logo, BetterBench & gBench evaluation reports
├── include/                 # Engine, sparse-global, MTP, and kernel headers
├── models/                  # draft_shortlist.bin (large weights gitignored)
├── scripts/
│   ├── sim_gpu.sh           # gfx1100 multi-GPU VRAM/CU simulator wrapper
│   ├── sim_gpu_shim.c       # LD_PRELOAD HIP property & VRAM-cap shim
│   └── spill_audit.sh       # Automated SGPR/VGPR/private-segment spill audit
├── src/                     # C++17 host runtime, HTTP server/UI, RDNA3 HIP kernels
├── tests/                   # Parity prompts & vision test fixture
└── third_party/
    └── stb_image.h          # Single-header image decoder for vision
```

---

## License

Apache License 2.0 — see [LICENSE](LICENSE).
