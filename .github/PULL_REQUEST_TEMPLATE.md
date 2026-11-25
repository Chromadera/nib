## Summary
- What does this PR change and why?
- Related issue (if any): Closes #

## Verification
Please check what you ran locally (and note GPU / ROCm version):
- **Hardware / ROCm:** _(e.g. RX 7900 XTX `gfx1100`, ROCm 6.x)_
- [ ] Clean build with **0 compiler warnings** (`cmake -B build && cmake --build build -j$(nproc)`)
- [ ] `./build/nib selftest -m models/Inkling-2-Mini-NVFP4-Mixed.gguf --mtp models/mtp.safetensors` passes
- [ ] If `.hip` kernels changed: `./scripts/spill_audit.sh` reports **0 register spills**
- [ ] If scheduler / server / KV cache changed: verified single-slot (`--slots 1`) and concurrent multi-slot (`--slots 4`+, including multimodal if applicable)

## Performance / Behavior Notes
_(Optional: before/after tok/s, TTFT, VRAM delta, or behavioral edge cases.)_
