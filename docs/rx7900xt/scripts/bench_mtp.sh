#!/usr/bin/env bash
# Real hardware RX 7900 XT MTP prefill + decode benchmark across all fitting KV formats:
#   1. q8_0 @ 90,112  (88K,  ~19.00 GiB; fallback 81,920)
#   2. q4_1 @ 163,840 (160K, ~19.06 GiB; fallback 131,072)
#   3. q4_0 @ 180,224 (176K, ~19.04 GiB; fallback 163,840)
set -uo pipefail

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*"; }

log "=== NIB RX 7900 XT MTP BENCHMARK START ==="
log "uname: $(uname -a)"
/opt/rocm/bin/hipcc --version 2>&1 | head -3 || true
if command -v amd-smi >/dev/null 2>&1; then
    amd-smi static 2>&1 | grep -E "MARKET_NAME|NUM_COMPUTE_UNITS|SIZE:|VERSION:" | head -12 || true
    amd-smi metric -m 2>&1 | grep -E "VRAM:|GTT:" | head -10 || true
fi

export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || true
apt-get install -y --no-install-recommends git cmake ca-certificates curl aria2 python3 || true

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

for f in Inkling-2-Mini-NVFP4-Mixed.gguf mtp.safetensors; do
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
done

# Build deterministic realistic code+docs prompts of increasing lengths from /workspace/nib sources.
python3 - <<'PY'
import pathlib

src = (
    pathlib.Path("/workspace/nib/README.md").read_text(errors="ignore")
    + "\n\n// --- engine.cpp ---\n"
    + pathlib.Path("/workspace/nib/src/engine.cpp").read_text(errors="ignore")
    + "\n\n// --- server.cpp ---\n"
    + pathlib.Path("/workspace/nib/src/server.cpp").read_text(errors="ignore")
)
tail = (
    "\n\n### Task\n"
    "Using C++17, write a clean, self-contained template class `RingBuffer<T, N>` "
    "supporting `push`, `pop`, `peek`, `size`, `empty`, `full`, and `clear`, "
    "followed by a short `main()` demonstrating usage and a 3-sentence complexity summary.\n"
    "```cpp\n"
)

# Conservative ~3.1 bytes/token on C++ code so prompts stay safely below max_ctx.
targets = {
    "2k":   6200,
    "32k":  100000,
    "64k":  200000,
    "80k":  248000,
    "128k": 395000,
    "152k": 470000,
    "168k": 520000,
}
for name, nbytes in targets.items():
    reps = (nbytes // len(src)) + 2
    body = (src * reps)[:nbytes]
    p = pathlib.Path(f"/tmp/prompt_{name}.txt")
    p.write_text(body + tail)
    print(f"wrote {p} ({p.stat().st_size} bytes)")
PY

SUMMARY=/tmp/mtp_bench_summary.txt
: > "$SUMMARY"

run_mtp_case() {
    local CFG="$1"
    local KV="$2"
    local CTX="$3"
    local DEPTH="$4"
    local PFILE="/tmp/prompt_${DEPTH}.txt"
    local OUT="/tmp/out_${CFG}_${DEPTH}.log"
    log "--- MTP CASE [$CFG @ $DEPTH]: kv=$KV ctx=$CTX mtp=1 ---"
    if timeout 360 ./build/nib gen \
        -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
        --mtp models/mtp.safetensors \
        --prompt-file "$PFILE" \
        -n 256 \
        -c "$CTX" \
        --kv-type "$KV" \
        --sparse-pages 224 > "$OUT" 2>&1; then
        local ENG MTP PF DEC STATS
        ENG=$(grep -m1 "total_vram=" "$OUT" | sed -n 's/.*total_vram=\([0-9.]* GiB\).*/\1/p')
        MTP=$(grep -m1 "drafter ready:" "$OUT" | sed -n 's/.*vram=\([0-9]* MiB\).*/\1/p')
        PF=$(grep -m1 "\[gen\] prefill:" "$OUT" || true)
        DEC=$(grep -m1 -- "--- END (" "$OUT" || true)
        STATS=$(grep -E "^\[mtp\] " "$OUT" | grep -v "drafter online" | tr '\n' ' ' || true)
        log "PASS [$CFG @ $DEPTH]: eng=${ENG:-?} mtp=${MTP:-?} | ${PF} | ${DEC} | ${STATS}"
        echo "PASS | $CFG | $DEPTH | kv=$KV | ctx=$CTX | eng=${ENG:-?} | mtp=${MTP:-?} | ${PF} | ${DEC} | ${STATS}" >> "$SUMMARY"
        return 0
    else
        local RC=$?
        local ERR
        ERR=$(grep -Ei "hipMalloc|OOM|failed|error|exceeds" "$OUT" | tail -n 3 | tr '\n' ' ')
        log "FAIL [$CFG @ $DEPTH] (rc=$RC): ${ERR}"
        echo "FAIL | $CFG | $DEPTH | kv=$KV | ctx=$CTX | rc=$RC | ${ERR}" >> "$SUMMARY"
        return 1
    fi
}

# 1. q8_0 + MTP (max ~88K = 90,112 -> 19.00 GiB; fallback 81,920 -> 18.86 GiB)
Q8_CTX=90112
if ! run_mtp_case "q8_0_90k" "q8_0" "$Q8_CTX" "2k"; then
    Q8_CTX=81920
    run_mtp_case "q8_0_82k" "q8_0" "$Q8_CTX" "2k" || true
fi
for D in 32k 64k 80k; do
    run_mtp_case "q8_0_${Q8_CTX}" "q8_0" "$Q8_CTX" "$D" || true
done

# 2. q4_1 + MTP (max 160K = 163,840 -> 19.06 GiB; fallback 131,072 -> 18.74 GiB)
Q41_CTX=163840
if ! run_mtp_case "q4_1_163k" "q4_1" "$Q41_CTX" "2k"; then
    Q41_CTX=131072
    run_mtp_case "q4_1_131k" "q4_1" "$Q41_CTX" "2k" || true
fi
for D in 32k 64k 128k 152k; do
    [ "$Q41_CTX" = "131072" ] && [ "$D" = "152k" ] && continue
    run_mtp_case "q4_1_${Q41_CTX}" "q4_1" "$Q41_CTX" "$D" || true
done

# 3. q4_0 + MTP (max 176K = 180,224 -> 19.04 GiB; fallback 163,840 -> 18.90 GiB)
Q40_CTX=180224
if ! run_mtp_case "q4_0_180k" "q4_0" "$Q40_CTX" "2k"; then
    Q40_CTX=163840
    run_mtp_case "q4_0_163k" "q4_0" "$Q40_CTX" "2k" || true
fi
for D in 32k 64k 128k 168k; do
    [ "$Q40_CTX" = "163840" ] && [ "$D" = "168k" ] && continue
    run_mtp_case "q4_0_${Q40_CTX}" "q4_0" "$Q40_CTX" "$D" || true
done

log "=== ALL MTP BENCHMARKS COMPLETE ==="
cat "$SUMMARY"
sleep 86400
