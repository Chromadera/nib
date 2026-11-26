#!/usr/bin/env bash
cd /workspace/nib
for Q in All 23; do for M in "" "--mtp /workspace/models/mtp.safetensors"; do T=selftest_${Q}$([ -n "$M" ] && echo _mtp)
  amd-smi metric -m | grep -E "USED_VRAM|USED_GTT" | head -2 > /workspace/logs/$T.log
  timeout 900 ./build/nib selftest -m /workspace/models/Inkling-2-Mini-NVFP4-$Q.gguf --kv-type q4_1 --sparse-pages 224 --cpu-embd $M >> /workspace/logs/$T.log 2>&1; echo "rc=$?" >> /workspace/logs/$T.log
done; done
