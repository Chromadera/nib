#!/usr/bin/env bash
# Stage 1: raw HIP hipMalloc+hipMemset ceiling (same probe as docs/rx7900xt/scripts/vram_ceiling_probe.sh)
source /workspace/common.sh
sed -n '/^cat > \/tmp\/vram_probe.cpp/,/^EOF$/p' /workspace/vram_ceiling_probe.sh | sed '1d;$d' > /tmp/vram_probe.cpp
/opt/rocm/bin/hipcc -O2 --offload-arch=gfx1100 -no-pie /tmp/vram_probe.cpp -o /tmp/vram_probe
{ echo "pre $(ts): $(smi)"; timeout 120 /tmp/vram_probe; echo "post $(ts): $(smi)"; } 2>&1 | tee $R/stage1_hip_probe.txt
