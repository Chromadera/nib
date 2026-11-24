# Nib — `google-deepmind/gbench` Evaluation Report

End-to-end capability, long-context, multimodal, and serving verification of **Nib** running **`Inkling-2-Mini-NVFP4-Mixed.gguf`** against [`google-deepmind/gbench`](https://github.com/google-deepmind/gbench) (`v0.5.0.post1`).

## Test Configuration & Reliability Summary

- **GPU**: AMD Radeon RX 7900 XTX (`gfx1100`, 24 GiB VRAM)
- **Server Command**:
  ```bash
  ./build/nib serve \
    -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
    --mtp models/mtp.safetensors \
    --mmproj models/mmproj-Inkling-2-Mini-Q8_0.gguf \
    --slots 1 -c 262848 --kv-type q8_0 --sparse-pages 224 \
    --host 127.0.0.1 --port 4444
  ```
- **VRAM Footprint (`--slots 1`)**: **21.50 GiB** at startup (`21.18 GiB` LLM + KV + MTP + `0.32 GiB` Vision weights), rising to **22.45 GiB** steady-state peak (`1.27 GiB` total Vision = `0.32 GiB` weights + `0.95 GiB` lazily allocated high-resolution ViT activation scratch after ScreenSpot / MMMU-Pro images)
- **Continuous Soak Telemetry** (across Golden Smoke + 5-Suite Reasoning Sweep + Serving Perf + 5-Suite Fast Bundle):
  - **Uptime**: **8h 12m** continuous (`29,568 s`), zero restarts
  - **Requests Served**: **2,289** total (`0` failed HTTP/engine requests, `0` timeouts, `0` empty replies, `0` `answer_in_thinking` drops)
  - **Tokens Processed**: **30,971,652** prompt tokens / **2,800,448** completion tokens

---

## 0. Quantization & Engine Fidelity (WikiText-2)

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

---

## 1. Golden Set Smoke Test (`--golden-only`)

- **Score**: **14 / 15** applicable cases **PASS** (**93.33%**, 1 audio case `N/A`)
- **Passing categories**: `chat_template` (**4/4**, zero template-token leaks), `tool_use` (**2/2**), `multimodal_vision` (**1/1**), `code_generation`, `structured_output`, `math`, `exact_knowledge`, `entity_identification`, `location_identification`, `translation`
- **1 non-pass**: `safety_boundary` (`0/1` — refusal phrasing did not match gBench's literal refusal substring filter)

---

## 2. Serving Performance (`--serving-only --preset quick`, `batch=1`)

Measured on `nib` (`128→512` text, `336→512` multimodal, `30` requests each, lossless DSpark MTP active):

| Mode | Completed | TTFT p50 | TPOT p50 | ITL mean | Output Throughput |
|---|---:|---:|---:|---:|---:|
| **Text** | `30/30` | **68.4 ms** | **7.96 ms** | **7.96 ms** | **123.6 tok/s** |
| **Multimodal (Vision)** | `30/30` | **313.9 ms** | **7.82 ms** | **7.69 ms** | **95.6 tok/s** *(incl. vision prefill)* |

*(MTP emits multi-token bursts per verify round, so p50 inter-chunk latency is ~12.4 ms while mean ITL / TPOT is **7.69–7.96 ms/tok**.)*

---

## 3. Reasoning & Model-Card Overlap Sweep (`--eval-thinking`)

Five canonical `gbench` suites matching Thinking Machines' `Inkling-2-Mini` release card (reasoning enabled, `--eval-concurrency 1`, total wall time ~6h 36m):

| Suite | Scope / Samples | Nib (`NVFP4-Mixed` + `Q8_0` `sp224`) | Median Decode | Reference (Official BF16/FP4 Card) | Protocol Notes |
|---|---|---:|---:|---|---|
| **`arc_agi`** | 20 tasks (21 pairs × 2 attempts, 32K cap) | **85.00%** task<br>*(85.71% pair pass@2, 69.05% avg@2)* | **197.5 tok/s** | **84.5%** (ARC-AGI v1) | `17/20` tasks (`18/21` pairs); `6/42` attempts hit 32K cap (`5/11` retries recovered) |
| **`screenspot`** | 200 stratified targets (ScreenSpot-v2) | **88.50%** (`177/200`) | **114.3 tok/s** | **76.5%** (ScreenSpot-Pro orig) | `mobile/text 93.9%`, `mobile/icon 90.9%`, `desktop/icon 90.9%`, `web/text 90.9%`, `web/icon 85.3%`, `desktop/text 79.4%` |
| **`mrcr`** | 120 stratified rows (`4K–256K` × 2/4/8 needles) | **83.44%** pooled mean | **94.0 tok/s** | **46.4%** (8-needle `256K–512K`) | Not apples-to-apples: `-c 262848` evaluates `4K–256K` and skips `>256K` (`679/2400` skipped); see bin table below |
| **`mmmu_pro`** | 50 stratified Qs × 4 modes (200 reqs) | **69.00%** headline<br>*(72.0% Vision Direct / 66.0% Std-10 CoT)* | **145.9 tok/s** | **74.9%** (full 1,730 Std-10) | Std-10 = 10-option MCQ (`66.0%` CoT / `64.0%` Direct); Vision = screenshot (`72.0%` Direct / `66.0%` CoT) |
| **`longbench_v2_code`** | **Full 50/50** Code Repo Understanding | **62.00%** non-CoT (`31/50`)<br>**60.00%** w/ CoT (`30/50`) | **123.1 tok/s** | **66.9%** (all-domain LongBench-v2) | Full canonical code split; 16K output budget leaves 246K context window |

### MRCR (`openai/mrcr`) Per-Bin Breakdown (`4K–256K`, `Q8_0` KV + `--sparse-pages 224`)

Mean canonical `SequenceMatcher` ratio per `(needles, length bin)` cell (`6–7` stratified rows per cell; `128K–256K` partial fit inside `262,848` window):

| Needles | `4K–8K` | `8K–16K` | `16K–32K` | `32K–64K` | `64K–128K` | `128K–256K` |
|---|---:|---:|---:|---:|---:|---:|
| **2-needle** | 98.96% | 99.60% | 99.99% | 99.99% | **100.00%** | 85.33% |
| **4-needle** | 99.72% | 71.56% | 88.49% | 99.99% | **99.74%** | 58.67% |
| **8-needle** | 85.25% | 66.07% | 84.45% | 66.11% | 58.60% | 37.28% |

---

## 4. Fast Capability Bundle (Greedy Non-Thinking, `enable_thinking: false`)

Five fast suites run at `temperature = 0.0` with thinking disabled (~48 min total):

| Suite | Scope / Samples | Accuracy | Median Decode | Notes & Trace Findings |
|---|---|---:|---:|---|
| **`gsm8k`** | 200 (8-shot CoT) | **93.00%** (`186/200`) | **138.5 tok/s** | Only `1/200` hit upstream's 256-token cap |
| **`ifeval`** | **Full 541/541** | **71.53%** prompt-strict<br>**79.38%** instr-strict<br>**77.08%** prompt-loose<br>**83.69%** instr-loose | **126.4 tok/s** | `18/541` hit 4096-token cap; `postscript`, `multiple_sections`, and `constrained_response` all **100%** |
| **`nexusraven_api_eval`** | **Full 339/339** | **38.41%** headline<br>*(Generic: **49.44%**, Cyber: **27.37%**)* | **68.0 tok/s** | **`82.12%`** on `virustotal` (`124/151`) and **`60.78%`** on `toolalpaca` (`31/51`); `cve_cpe` (`0/78`) & `emailrep` (`0/38`) emitted valid raw JSON without markdown fences, rejected by upstream's fenced-block parser |
| **`docvqa`** | 200 (ANLS) | **30.40%** (`65/200`) | **22.2 tok/s** | `115/200` truncated at `lmms-eval`'s strict `max_tokens=32` cap (see non-thinking note below) |
| **`chartqa`** | 200 | **20.00%** (`40/200`) | **38.6 tok/s** | Scorer + cap artifact: vision read exact chart values (`6%` vs gold `6`, `2%` vs `2`, `77%` vs `77`), which ChartQA's parser divides by 100 (`0.06 ≠ 6`); `75/200` also hit `lmms-eval`'s `max_tokens=16` cap |

> **Note on Non-Thinking (`effort = 0`) vs Thinking (`--eval-thinking`)**:
> Inkling 2 Mini is natively RL-trained as a reasoning model (`effort = 0.9` default). On short-cap (`16–32` token) VQA benchmarks with `enable_thinking: false`, forcing direct text output (`<|message_model|><|content_text|>`) can cause the model to emit reasoning commentary inside `content` and hit the upstream `max_tokens` cap. For multimodal and complex reasoning tasks, running with thinking enabled (default `reasoning_effort` or `--eval-thinking`) matches Thinking Machines' intended usage and yields substantially stronger results (e.g., **88.5%** ScreenSpot, **85.0%** ARC-AGI, **69.0%** MMMU-Pro).
