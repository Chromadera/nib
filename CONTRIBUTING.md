# Contributing to Nib

Thanks for your interest in contributing to **Nib**! Nib is a focused, native AMD RDNA3 (`gfx1100`) inference engine for **Inkling 2 Mini** where correctness, predictable VRAM, and zero-spill GPU performance matter.

## Building Locally

```bash
cmake -B build
cmake --build build -j$(nproc)
./build/nib --version
```

## Engineering Invariants

Before opening a pull request, please keep these core invariants in mind:

1. **Zero compiler warnings** on both `hipcc` (`.hip`) and C++ (`src/*.cpp`).
2. **Zero GPU register spills** when modifying any `.hip` kernel — verify with:
   ```bash
   ./scripts/spill_audit.sh
   ```
3. **WGP-aware co-residency:** Avoid hardcoding 96-CU assumptions. Kernels and occupancy guards must remain safe across 96-CU (7900 XTX), 84-CU (7900 XT), and 70-CU (7900 GRE / W7800) `gfx1100` GPUs. Use `./scripts/sim_gpu.sh` if testing lower-CU profiles.
4. **Bit-exact / lossless parity:**
   - Run `nib selftest --mtp` and confirm prefill/decode parity, GPU sampler chi-square/batch tests, MTP `verify` / `verify_rollback`, and 64-token greedy parity all pass.
   - Changes to the scheduler, prefix cache, or sampler should not regress the solo slot-0 HIP Graph / DSpark MTP fast path or `B >= 2` batched decode throughput.
5. **Concurrent slot & shared-buffer safety:**
   - Multi-slot changes (`--slots 2..8`) must keep per-slot KV, elastic multi-segment spans, SWA rings, Quest sparse metadata, and shared scratch buffers (such as `VisionEncoder` scratch) isolated across interleaved prefill/decode ticks.

## Verification Checklist

```bash
# 1. Build (0 warnings)
cmake -B build && cmake --build build -j$(nproc)

# 2. Core engine + GPU sampler + MTP selftest
./build/nib selftest -m models/Inkling-2-Mini-NVFP4-Mixed.gguf --mtp models/mtp.safetensors

# 3. Kernel spill audit (required if any src/*.hip changed)
./scripts/spill_audit.sh
```

For `src/server.cpp` changes, also test live HTTP traffic under both `--slots 1` and `--slots 4` (text, tool calls / streaming, and multimodal with `--mmproj` when relevant).

## Pull Request Process

- Keep PRs focused on a single bug fix, feature, or optimization.
- Keep **"Allow edits by maintainers"** enabled when opening PRs from a fork so small edge-case/hardening fixes can be pushed cleanly on top of your commit(s) while preserving your authorship.
- Include hardware/ROCm info and any relevant before/after measurements in the PR description.
- For large architectural changes or new GPU targets, open an issue first to align on the approach before investing heavy kernel work.
