#!/usr/bin/env bash
# NVFP4-23 RX 7900 XT setup: build local nib, repack NVFP4-23, fetch mtp+mmproj, BetterBench 0.6.1 venv.
set -uo pipefail
exec > >(tee -a /workspace/setup.log) 2>&1
log() { echo "[$(date -u +%FT%TZ)] $*"; }
export DEBIAN_FRONTEND=noninteractive
log "uname: $(uname -a)"; ls -l /dev/dxg /dev/kfd 2>&1; free -h; df -h /workspace /
log "rocm: $(cat /opt/rocm/.info/version 2>/dev/null)"
apt-get update -qq; apt-get install -y --no-install-recommends git cmake curl ca-certificates python3 python3-venv python3-pip aria2 g++ libgomp1 jq socat >/dev/null || log "apt warn"
cd /workspace
[ -x nib/build/nib ] && SKIPBUILD=1; [ -d nib ] || tar xzf /workspace/nib_src.tgz
cd nib
log "cmake build"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGPU_TARGET=gfx1100 -DROCM_PATH=/opt/rocm -DCMAKE_EXE_LINKER_FLAGS="-no-pie" > /workspace/cmake_config.log 2>&1
cmake --build build -j"$(nproc)" > /workspace/cmake_build.log 2>&1; tail -3 /workspace/cmake_build.log
[ -x build/nib ] || { log "FATAL build failed"; tail -40 /workspace/cmake_build.log; exit 1; }
./build/nib --version 2>&1 | head -3
log "spill audit:"; ./scripts/spill_audit.sh > /workspace/spill_audit.log 2>&1; tail -30 /workspace/spill_audit.log
log "repack build"
g++ -O3 -march=native -fopenmp /workspace/repack_nvfp4.cpp -o /workspace/repack_nvfp4 || { log "FATAL repack build"; exit 1; }
mkdir -p /workspace/models; cd /workspace/models
Q8=Inkling-2-Mini-Q8_0.gguf; OUT=Inkling-2-Mini-NVFP4-23.gguf
if [ "$(stat -c%s $OUT 2>/dev/null)" != 16099143008 ]; then
  for a in 1 2 3 4; do
    aria2c -x 16 -s 16 -k 8M --file-allocation=none -c --summary-interval=60 --console-log-level=warn -o $Q8 \
      https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/resolve/main/$Q8 && break
  done
  ls -l $Q8
  log "repack NVFP4-23 (max_layer=22)"
  time /workspace/repack_nvfp4 $Q8 $OUT 22 > /workspace/repack.log 2>&1; tail -25 /workspace/repack.log
  sz=$(stat -c%s $OUT); log "NVFP4-23 size=$sz (want 16099143008)"
  [ "$sz" = 16099143008 ] || { log "FATAL size mismatch"; exit 1; }
  sha256sum $OUT | tee /workspace/nvfp4_23.sha256
  rm -f $Q8; log "deleted Q8_0"
fi
HF=https://huggingface.co/degenmeowster/Inkling-2-Mini-Nib-Pack/resolve/main
for f in mtp.safetensors mmproj-Inkling-2-Mini-Q8_0.gguf; do
  want=$(curl -sIL "$HF/$f" | awk 'tolower($1)=="x-linked-size:"{gsub("\r","",$2);v=$2} END{print v}')
  sha=$(curl -sIL "$HF/$f" | awk 'tolower($1)=="x-linked-etag:"{gsub("[\r\"]","",$2);v=$2} END{print v}')
  for a in 1 2 3 4; do [ "$(stat -c%s $f 2>/dev/null)" = "$want" ] && break; aria2c -x16 -s16 -k8M --file-allocation=none -c --console-log-level=warn -o $f "$HF/$f"; done
  [ "$(sha256sum $f | cut -d' ' -f1)" = "$sha" ] && log "$f sha256 OK ($want)" || log "WARN sha mismatch $f"
done
ls -l /workspace/models; df -h /workspace
log "betterbench venv"
python3 -m venv /workspace/bbenv && /workspace/bbenv/bin/pip install -q "numpy>=1.24" "git+https://github.com/GGZ14/BetterBench.git@f72bb9011a830fd98e82ab89d01e549efa76e9c5"
log "betterbench: $(/workspace/bbenv/bin/betterbench --version 2>&1)"
log "SETUP DONE"
