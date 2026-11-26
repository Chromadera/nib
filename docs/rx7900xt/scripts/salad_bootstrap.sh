#!/usr/bin/env bash
# Nib bootstrap for SaladCloud AMD RX 7900 XT (20GB, gfx1100) on WSL2/ROCDXG.
# Installs omp, Claude Code, and betterbench, builds nib, pulls weights via aria2c,
# and serves the nib OpenAI API with multi-slot concurrency on dual-stack [::]:4444 -> 127.0.0.1:4445.
set -uo pipefail

LOG=/var/log/nib-bootstrap.log
mkdir -p /var/log
exec > >(tee -a "$LOG") 2>&1

export HOME="${HOME:-/root}"
export PATH="/root/.local/bin:/usr/local/bin:$PATH"

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*"; }
die() { log "FATAL: $*"; log "Keeping container alive for log inspection."; sleep 86400; exit 1; }

PROXY_PID=""
NIB_PID=""
SERVING_SPEC=""

log "=== NIB SALAD BOOTSTRAP v3 START ==="
log "uname: $(uname -a)"
log "HSA_ENABLE_DXG_DETECTION=${HSA_ENABLE_DXG_DETECTION:-unset}"
log "LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-unset}"
ls -l /dev/dxg 2>&1 || true
free -h 2>&1 || true
df -h / /tmp 2>&1 || true

log "hipcc:"; /opt/rocm/bin/hipcc --version 2>&1 | head -6 || true
if command -v amd-smi >/dev/null 2>&1; then
    log "amd-smi static (summary):"
    amd-smi static 2>&1 | grep -E "MARKET_NAME|NUM_COMPUTE_UNITS|TARGET_GRAPHICS_VERSION|VERSION:|SIZE:|BIT_WIDTH|PCIE_INTERFACE" | head -20 || true
fi

log "installing minimal build/download/CLI tools (no ROCm packages)"
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq || log "apt update returned non-zero (continuing)"
apt-get install -y --no-install-recommends git cmake socat ca-certificates curl python3 aria2 unzip jq ripgrep || \
    log "apt install returned non-zero (continuing with what is present)"
command -v git >/dev/null 2>&1 || die "git unavailable"
command -v cmake >/dev/null 2>&1 || die "cmake unavailable"
command -v curl >/dev/null 2>&1 || die "curl unavailable"
command -v socat >/dev/null 2>&1 || die "socat unavailable"

# Persist PATH for interactive/SSH shells
cat > /etc/profile.d/local-bin.sh <<'PROF'
export PATH="/root/.local/bin:/usr/local/bin:$PATH"
PROF

log "=== installing omp (Oh My Pi) ==="
if ! command -v omp >/dev/null 2>&1; then
    curl -fsSL https://omp.sh/install | sh || log "WARN: omp install returned $?"
fi
[ -x /root/.local/bin/omp ] && ln -sf /root/.local/bin/omp /usr/local/bin/omp || true
log "omp check: $(command -v omp 2>/dev/null || echo MISSING) | $(omp --version 2>&1 | head -2 || true)"

log "=== installing Claude Code ==="
if ! command -v claude >/dev/null 2>&1; then
    curl -fsSL https://claude.ai/install.sh | bash || log "WARN: claude install returned $?"
fi
[ -x /root/.local/bin/claude ] && ln -sf /root/.local/bin/claude /usr/local/bin/claude || true
log "claude check: $(command -v claude 2>/dev/null || echo MISSING) | $(claude --version 2>&1 | head -2 || true)"

log "=== installing uv + betterbench ==="
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh || log "WARN: uv install returned $?"
fi
[ -x /root/.local/bin/uv ] && ln -sf /root/.local/bin/uv /usr/local/bin/uv || true
if command -v uv >/dev/null 2>&1 && ! command -v betterbench >/dev/null 2>&1; then
    uv tool install betterbench || log "WARN: betterbench install returned $?"
fi
[ -x /root/.local/bin/betterbench ] && ln -sf /root/.local/bin/betterbench /usr/local/bin/betterbench || true
log "betterbench check: $(command -v betterbench 2>/dev/null || echo MISSING) | $(betterbench --version 2>&1 | head -2 || true)"

mkdir -p /workspace
cd /workspace || die "cd /workspace failed"

if [ ! -d nib/.git ]; then
    for i in 1 2 3; do
        log "clone attempt $i"
        git clone --depth 1 https://github.com/Chromadera/nib.git nib && break
        sleep 5
    done
fi
[ -d nib/.git ] || die "git clone of nib failed"
cd /workspace/nib || die "cd nib failed"
log "commit: $(git rev-parse HEAD) $(git log -1 --format=%s)"

if [ ! -x build/nib ]; then
    log "cmake configure"
    cmake -B build -DCMAKE_BUILD_TYPE=Release -DGPU_TARGET=gfx1100 -DROCM_PATH=/opt/rocm \
        -DCMAKE_EXE_LINKER_FLAGS="-no-pie" || die "cmake configure failed"
    log "cmake build -j$(nproc)"
    cmake --build build -j"$(nproc)" || die "build failed"
else
    log "existing build/nib found, skipping build"
fi
ls -l build/nib || die "build/nib missing"

mkdir -p models
HF=https://huggingface.co/degenmeowster/Inkling-2-Mini-Nib-Pack/resolve/main
file_size() { stat -c%s "$1" 2>/dev/null || echo 0; }

hf_meta() {
    local hdrs
    hdrs=$(curl -sIL --max-time 60 "$HF/$1" 2>/dev/null)
    HF_WANT=$(printf '%s' "$hdrs" | awk 'tolower($1)=="x-linked-size:"{gsub("\r","",$2); v=$2} END{print v}')
    [ -n "$HF_WANT" ] || HF_WANT=$(printf '%s' "$hdrs" | awk 'tolower($1)=="content-length:"{gsub("\r","",$2); v=$2} END{print v}')
    HF_ETAG=$(printf '%s' "$hdrs" | awk 'tolower($1)=="x-linked-etag:"{gsub("\r","",$2); gsub("\"","",$2); v=$2} END{print v}')
}

for f in Inkling-2-Mini-NVFP4-Mixed.gguf mtp.safetensors; do
    HF_WANT=""; HF_ETAG=""
    hf_meta "$f"
    have=$(file_size "models/$f")
    log "weights $f: have=$have want=${HF_WANT:-unknown} sha256=${HF_ETAG:-unknown}"
    if [ -n "$HF_WANT" ] && [ "$have" = "$HF_WANT" ]; then
        log "$f already complete"
        continue
    fi
    for attempt in 1 2 3 4 5 6; do
        if command -v aria2c >/dev/null 2>&1 && [ "$attempt" -le 4 ]; then
            log "aria2c $f attempt $attempt (-x16 -s16)"
            aria2c -x 16 -s 16 -k 8M --file-allocation=none --continue=true \
                --max-tries=5 --retry-wait=5 --timeout=60 --connect-timeout=30 \
                --summary-interval=15 --console-log-level=notice \
                -d models -o "$f" "$HF/$f" || log "aria2c rc=$?"
        else
            log "curl $f attempt $attempt"
            curl -fL --retry 3 --retry-delay 5 --connect-timeout 30 -C - -o "models/$f" "$HF/$f" || log "curl rc=$?"
        fi
        have=$(file_size "models/$f")
        if [ -n "$HF_WANT" ] && [ "$have" = "$HF_WANT" ]; then
            log "$f download complete ($have bytes)"
            break
        fi
        log "size after attempt: $have / ${HF_WANT:-unknown}; retrying"
        sleep 5
    done
    have=$(file_size "models/$f")
    if [ -n "$HF_WANT" ]; then
        [ "$have" = "$HF_WANT" ] || die "download incomplete: $f ($have/$HF_WANT)"
    else
        [ "$have" -gt 100000000 ] || die "download failed: $f (size $have)"
    fi
    if [ -n "$HF_ETAG" ]; then
        log "verifying sha256 for $f"
        got=$(sha256sum "models/$f" | awk '{print $1}')
        [ "$got" = "$HF_ETAG" ] && log "$f sha256 OK" || die "sha256 mismatch $f: $got != $HF_ETAG"
    fi
done
ls -lh models/ || true

start_proxy() {
    if [ -n "$PROXY_PID" ]; then kill "$PROXY_PID" 2>/dev/null || true; fi
    sleep 1
    socat TCP6-LISTEN:4444,ipv6only=0,fork,reuseaddr,nodelay,backlog=128 TCP4:127.0.0.1:4445 &
    PROXY_PID=$!
    sleep 1
    log "socat dual-stack proxy pid=$PROXY_PID on [::]:4444 -> 127.0.0.1:4445"
}

try_serve() {
    local S="$1"
    local C="$2"
    log "=== launching nib serve with --slots $S -c $C ==="
    ./build/nib serve \
        -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
        --mtp models/mtp.safetensors \
        --no-vision \
        --slots "$S" \
        -c "$C" \
        --kv-type q4_1 \
        --sparse-pages 224 \
        --host 127.0.0.1 \
        --port 4445 &
    NIB_PID=$!
    log "nib pid=$NIB_PID"
    for i in $(seq 1 120); do
        if ! kill -0 "$NIB_PID" 2>/dev/null; then
            log "nib exited before becoming ready (slots=$S ctx=$C)"
            return 1
        fi
        if curl -fsS --max-time 3 http://127.0.0.1:4445/health -o /tmp/health.json 2>/dev/null; then
            log "health OK (slots=$S ctx=$C): $(head -c 900 /tmp/health.json)"
            echo "${S}:${C}" > /workspace/nib/.serving_spec
            return 0
        fi
        sleep 5
    done
    log "timed out waiting for health (slots=$S ctx=$C)"
    return 1
}

while true; do
    if [ -n "$SERVING_SPEC" ]; then
        SPEC_LIST="$SERVING_SPEC"
    else
        SPEC_LIST="8:131072 4:131072 8:65536 4:65536 1:131072 1:65536"
    fi
    GOT=0
    for SPEC in $SPEC_LIST; do
        S="${SPEC%%:*}"
        C="${SPEC##*:}"
        start_proxy
        if try_serve "$S" "$C"; then GOT=1; SERVING_SPEC="$SPEC"; break; fi
        kill "$NIB_PID" 2>/dev/null || true
        wait "$NIB_PID" 2>/dev/null || true
        sleep 3
    done
    [ "$GOT" = 1 ] || die "all (slots:ctx) configs failed to start"

    log "SERVING with spec=$SERVING_SPEC"
    if curl -fsS --max-time 5 "http://[::1]:4444/health" >/dev/null 2>&1; then
        log "dual-stack IPv6 check OK ([::1]:4444/health)"
    fi

    while kill -0 "$NIB_PID" 2>/dev/null; do
        if [ -z "$PROXY_PID" ] || ! kill -0 "$PROXY_PID" 2>/dev/null; then
            log "socat proxy died; restarting"
            start_proxy
        fi
        sleep 10
    done
    log "nib exited after becoming ready; restarting serve loop"
    wait "$NIB_PID" 2>/dev/null || true
done
