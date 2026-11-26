#!/usr/bin/env bash
cd /workspace; mkdir -p logs
{ echo "=== STAGE 1 raw HIP ceiling $(date -u +%FT%TZ)"; amd-smi metric -m | grep -E "VRAM|GTT";
/opt/rocm/bin/hipcc -O2 --offload-arch=gfx1101 -no-pie /workspace/vram_probe.cpp -o /workspace/vram_probe && timeout 120 /workspace/vram_probe; echo "rc=$?";
echo "--- after"; sleep 3; amd-smi metric -m | grep -E "VRAM|GTT"; } > logs/stage1_vram_probe.log 2>&1
