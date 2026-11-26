#!/usr/bin/env bash
# Real hardware non-MTP prefill + decode benchmark for fitting RX 7900 XT KV configs.
set -uo pipefail

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*"; }

log "=== NIB RX 7900 XT NON-MTP BENCHMARK START ==="
log "uname: $(uname -a)"
/opt/rocm/bin/hipcc --version 2>&1 | head -3 || true
if command -v amd-smi >/dev/null 2>&1; then
    amd-smi static 2>&1 | grep -E "MARKET_NAME|NUM_COMPUTE_UNITS|SIZE:|VERSION:" | head -12 || true
    amd-smi metric -m 2>&1 | grep -E "VRAM:|GTT:" | head -10 || true
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || true
apt-get install -y --no-install-recommends git cmake ca-certificates curl aria2 || true

mkdir -p /workspace && cd /workspace
if [ ! -d nib/.git ]; then
    git clone --depth 1 https://github.com/Chromadera/nib.git nib
fi
cd /workspace/nib
log "nib commit: $(git rev-parse --short HEAD)"
if [ ! -x build/nib ]; then
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGPU_TARGET=gfx1100 -DROCM_PATH=/opt/rocm \
        -DCMAKE_EXE_LINKER_FLAGS="-no-pie"
    cmake --build build -j"$(nproc)"
fi

mkdir -p models
HF=https://huggingface.co/degenmeowster/Inkling-2-Mini-Nib-Pack/resolve/main
file_size() { stat -c%s "$1" 2>/dev/null || echo 0; }
hf_meta() {
    local hdrs
    hdrs=$(curl -sIL --max-time 60 "$HF/$1" 2>/dev/null)
    HF_WANT=$(printf '%s' "$hdrs" | awk 'tolower($1)=="x-linked-size:"{gsub("\r","",$2); v=$2} END{print v}')
    [ -n "$HF_WANT" ] || HF_WANT=$(printf '%s' "$hdrs" | awk 'tolower($1)=="content-length:"{gsub("\r","",$2); v=$2} END{print v}')
}

f=Inkling-2-Mini-NVFP4-Mixed.gguf
HF_WANT=""
hf_meta "$f"
log "downloading $f (want=${HF_WANT:-unknown})"
for attempt in 1 2 3 4; do
    [ -n "$HF_WANT" ] && [ "$(file_size "models/$f")" = "$HF_WANT" ] && break
    aria2c -x 16 -s 16 -k 8M --file-allocation=none --continue=true \
        --max-tries=5 --retry-wait=5 --timeout=60 --connect-timeout=30 \
        --summary-interval=30 --console-log-level=notice \
        -d models -o "$f" "$HF/$f" || log "aria2c rc=$?"
done
log "$f ready ($(file_size "models/$f") bytes)"

run_bench() {
    local LABEL="$1"
    local KV="$2"
    local CTX="$3"
    local DEPTHS="$4"
    log "=== BENCH [$LABEL]: kv=$KV ctx=$CTX depths=$DEPTHS (pp512 / tg64 / r2 / sparse224 / no-mtp) ==="
    if timeout 600 ./build/nib bench \
        -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
        -c "$CTX" \
        -d "$DEPTHS" \
        -p 512 -n 64 -r 2 \
        --kv-type "$KV" \
        --sparse-pages 224; then
        log "=== DONE [$LABEL] OK ==="
        return 0
    else
        log "=== DONE [$LABEL] FAILED (rc=$?) ==="
        return 1
    fi
}

# 1. Baseline q4_1 @ 128k (18.06 GiB) — clean headroom reference
run_bench "q4_1_131k" "q4_1" 131072 "0,8192,32768,65536,128000" || true

# 2. q8_0 @ 128k (18.99 GiB) — max standard q8_0 context on WSL2
run_bench "q8_0_131k" "q8_0" 131072 "0,8192,32768,65536,128000" || true

# 3. q4_1 @ 224k (19.02 GiB) — max standard q4_1 context on WSL2
run_bench "q4_1_229k" "q4_1" 229376 "0,8192,32768,65536,128000,224000" || true

# 4. q4_0 @ full 256k (19.10 GiB) — full native context window on WSL2
if ! run_bench "q4_0_262k" "q4_0" 262848 "0,8192,32768,65536,128000,224000,256000"; then
    log "q4_0_262k hit node WDDM cap; testing fallback q4_0_245k"
    run_bench "q4_0_245k" "q4_0" 245760 "0,8192,32768,65536,128000,224000,240000" || true
fi

log "=== ALL NON-MTP BENCHMARKS COMPLETE ==="
sleep 86400
