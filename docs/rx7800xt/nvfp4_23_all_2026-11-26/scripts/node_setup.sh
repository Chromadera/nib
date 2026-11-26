#!/usr/bin/env bash
# Node setup for RX 7800 XT (gfx1101): build local nib, repack NVFP4-23 / NVFP4-All, fetch MTP + mmproj, BetterBench 0.6.1.
set -uo pipefail
mkdir -p /workspace/logs; exec > >(tee -a /workspace/logs/setup.log) 2>&1
ts(){ date -u +%FT%TZ; }; log(){ echo "[$(ts)] $*"; }
log "=== SETUP START ==="; uname -a; ls -l /dev/dxg /dev/kfd 2>&1; free -h; df -h /workspace / 2>&1; nproc
/opt/rocm/bin/hipcc --version 2>&1 | head -4
amd-smi static > /workspace/logs/amd-smi_static.txt 2>&1; amd-smi metric -m > /workspace/logs/amd-smi_metric_idle.txt 2>&1
grep -E "MARKET_NAME|NUM_COMPUTE_UNITS|TARGET_GRAPHICS_VERSION|SIZE:|BIT_WIDTH|DRIVER|VERSION:" /workspace/logs/amd-smi_static.txt | head -20
grep -E "VRAM|GTT" /workspace/logs/amd-smi_metric_idle.txt
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq; apt-get install -y --no-install-recommends git cmake socat ca-certificates curl python3 python3-venv python3-pip aria2 jq g++ libgomp1 >/dev/null 2>&1 || log "apt rc=$?"
cd /workspace && tar xzf /workspace/nib_src.tgz && cp rdna3-hillclimb/kernels/repack_nvfp4.cpp /workspace/ || { log "FATAL untar"; exit 1; }
# --- downloads (background) ---
mkdir -p /workspace/models
( aria2c -x 16 -s 16 -k 8M --file-allocation=none --continue=true --max-tries=10 --retry-wait=5 --summary-interval=60 --console-log-level=warn \
    -d /workspace/models -o Inkling-2-Mini-Q8_0.gguf https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/main/Inkling-2-Mini-Q8_0.gguf > /workspace/logs/dl_q8.log 2>&1; echo "q8 rc=$?" >> /workspace/logs/dl_q8.log ) &
DLQ8=$!
HF=https://huggingface.co/degenmeowster/Inkling-2-Mini-Nib-Pack/resolve/main
( for f in mtp.safetensors mmproj-Inkling-2-Mini-Q8_0.gguf; do aria2c -x 16 -s 16 -k 8M --file-allocation=none --continue=true --max-tries=10 --console-log-level=warn -d /workspace/models -o $f $HF/$f; done > /workspace/logs/dl_pack.log 2>&1 ) &
DLP=$!
# --- build ---
cd /workspace/nib
log "cmake configure gfx1101"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGPU_TARGET=gfx1101 -DROCM_PATH=/opt/rocm -DCMAKE_EXE_LINKER_FLAGS="-no-pie" > /workspace/logs/cmake_configure.log 2>&1 || { log "FATAL configure"; tail -30 /workspace/logs/cmake_configure.log; }
log "cmake build -j$(nproc)"
cmake --build build -j"$(nproc)" > /workspace/logs/cmake_build.log 2>&1 && log "build OK" || { log "FATAL build"; tail -40 /workspace/logs/cmake_build.log; }
ls -l build/nib; ./build/nib --version 2>&1 | head -3
log "spill audit"; ROCM_PATH=/opt/rocm bash scripts/spill_audit.sh build > /workspace/logs/spill_audit.log 2>&1; echo "spill_audit rc=$?" >> /workspace/logs/spill_audit.log; tail -5 /workspace/logs/spill_audit.log
# --- repack tool ---
g++ -O3 -march=native -fopenmp /workspace/repack_nvfp4.cpp -o /workspace/repack_nvfp4 && log "repack built"
# --- betterbench venv ---
python3 -m venv /workspace/bbvenv && /workspace/bbvenv/bin/pip install -q "numpy>=1.24" "git+https://github.com/GGZ14/BetterBench.git@f72bb9011a830fd98e82ab89d01e549efa76e9c5" > /workspace/logs/bb_install.log 2>&1
log "betterbench: $(/workspace/bbvenv/bin/betterbench --version 2>&1 | head -1)"
wait $DLP; ls -l /workspace/models
wait $DLQ8; tail -2 /workspace/logs/dl_q8.log; ls -l /workspace/models/Inkling-2-Mini-Q8_0.gguf
cd /workspace/models; curl -sfL -o draft_shortlist.bin https://huggingface.co/degenmeowster/Inkling-2-Mini-Nib-Pack/resolve/main/draft_shortlist.bin
bash /workspace/repack.sh
cat /workspace/logs/sizes.txt
log "=== SETUP DONE ==="
