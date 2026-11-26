#!/usr/bin/env bash
# Two-stage VRAM ceiling probe for AMD RX 7900 XT (20GB, gfx1100, WSL2/ROCDXG).
# Stage 1 (~30-60s): raw HIP hipMemGetInfo + hipMalloc + hipMemset ceiling probe (no weights download).
# Stage 2: real nib gen + serve config matrix across q4_1 / q4_0 / q8_0, 64k..262k, +/- MTP, slots.
set -uo pipefail

ts()  { date -u +%Y-%m-%dT%H:%M:%SZ; }
log() { echo "[$(ts)] $*"; }

log "=== NIB RX 7900 XT VRAM CEILING PROBE START ==="
log "uname: $(uname -a)"
log "HSA_ENABLE_DXG_DETECTION=${HSA_ENABLE_DXG_DETECTION:-unset}"
ls -l /dev/dxg 2>&1 || true
free -h 2>&1 || true
df -h / /tmp 2>&1 || true

log "hipcc:"; /opt/rocm/bin/hipcc --version 2>&1 | head -4 || true
if command -v amd-smi >/dev/null 2>&1; then
    log "amd-smi static:"
    amd-smi static 2>&1 | grep -E "MARKET_NAME|NUM_COMPUTE_UNITS|TARGET_GRAPHICS_VERSION|VERSION:|SIZE:|BIT_WIDTH|PCIE_INTERFACE" | head -20 || true
    log "amd-smi metric -m (idle):"
    amd-smi metric -m 2>&1 | head -30 || true
fi

log "=== STAGE 1: RAW HIP / ROCDXG VRAM CEILING PROBE ==="
cat > /tmp/vram_probe.cpp <<'EOF'
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <cstdint>

static double to_mib(size_t b) { return (double)b / (1024.0 * 1024.0); }
static double to_gib(size_t b) { return (double)b / (1024.0 * 1024.0 * 1024.0); }

static void print_mem(const char* tag) {
    size_t f = 0, t = 0;
    hipError_t e = hipMemGetInfo(&f, &t);
    if (e != hipSuccess) {
        std::printf("[mem:%s] hipMemGetInfo error: %s\n", tag, hipGetErrorString(e));
    } else {
        std::printf("[mem:%s] free=%.1f MiB (%.3f GiB) | used=%.1f MiB (%.3f GiB) | total=%.1f MiB (%.3f GiB)\n",
                    tag, to_mib(f), to_gib(f), to_mib(t - f), to_gib(t - f), to_mib(t), to_gib(t));
    }
    std::fflush(stdout);
}

static size_t probe_chunks(const char* label, size_t initial_chunk_mib, bool touch_pages, int pre_chunks, size_t pre_chunk_mib) {
    std::vector<void*> ptrs;
    size_t total = 0;
    size_t next_report_gib = 16;

    auto try_alloc = [&](size_t bytes) -> bool {
        void* p = nullptr;
        hipError_t e = hipMalloc(&p, bytes);
        if (e != hipSuccess || !p) {
            (void)hipGetLastError();
            return false;
        }
        if (touch_pages) {
            e = hipMemset(p, 0x5A, bytes);
            if (e == hipSuccess) e = hipDeviceSynchronize();
            if (e != hipSuccess) {
                (void)hipGetLastError();
                (void)hipFree(p);
                return false;
            }
        }
        ptrs.push_back(p);
        total += bytes;
        return true;
    };

    for (int i = 0; i < pre_chunks; ++i) {
        if (!try_alloc(pre_chunk_mib * 1024ULL * 1024ULL)) {
            std::printf("[%s] pre-alloc failed at chunk %d (%.3f GiB)\n", label, i, to_gib(total));
            break;
        }
    }
    if (pre_chunks > 0) {
        std::printf("[%s] after pre-alloc (%d x %zu MiB): total=%.1f MiB (%.3f GiB)\n",
                    label, (int)ptrs.size(), pre_chunk_mib, to_mib(total), to_gib(total));
        print_mem(label);
    }

    size_t step_mib = initial_chunk_mib;
    while (step_mib >= 1) {
        while (try_alloc(step_mib * 1024ULL * 1024ULL)) {
            if (total >= next_report_gib * 1024ULL * 1024ULL * 1024ULL) {
                std::printf("[%s] reached %.1f MiB (%.3f GiB) in %zu allocs\n",
                            label, to_mib(total), to_gib(total), ptrs.size());
                print_mem(label);
                next_report_gib += 1;
            }
            if (total > 22ULL * 1024ULL * 1024ULL * 1024ULL) {
                std::printf("[%s] safety cap 22 GiB hit (overcommit detected!)\n", label);
                step_mib = 0;
                break;
            }
        }
        if (step_mib == 0) break;
        step_mib /= 2;
    }

    std::printf("RESULT [%s] max_alloc=%.1f MiB (%.4f GiB) across %zu allocations (touch=%d)\n",
                label, to_mib(total), to_gib(total), ptrs.size(), touch_pages ? 1 : 0);
    print_mem("at_cliff");

    for (void* p : ptrs) (void)hipFree(p);
    (void)hipDeviceSynchronize();
    return total;
}

int main() {
    int dev = 0;
    hipError_t e = hipSetDevice(dev);
    if (e != hipSuccess) {
        std::fprintf(stderr, "hipSetDevice failed: %s\n", hipGetErrorString(e));
        return 1;
    }
    hipDeviceProp_t prop{};
    hipGetDeviceProperties(&prop, dev);
    std::printf("[device] name=%s gcnArch=%s CUs=%d totalGlobalMem=%.1f MiB (%.4f GiB)\n",
                prop.name, prop.gcnArchName, prop.multiProcessorCount,
                to_mib(prop.totalGlobalMem), to_gib(prop.totalGlobalMem));
    print_mem("idle_start");

    // 1. Chunked 16 MiB + hipMemset touch + 1 MiB refinement
    size_t max_touched = probe_chunks("chunk16_touched", 16, true, 0, 0);
    print_mem("after_chunk16_free");

    // 2. Realistic Nib-like fragmentation: 512 x 32 MiB (~16.0 GiB weights) + 16..1 MiB chunks touched
    size_t max_niblike = probe_chunks("niblike_512x32M_touched", 16, true, 512, 32);
    print_mem("after_niblike_free");

    std::printf("=== STAGE 1 SUMMARY ===\n");
    std::printf("  chunk16_touched          = %.1f MiB (%.4f GiB)\n", to_mib(max_touched), to_gib(max_touched));
    std::printf("  niblike_512x32M_touched  = %.1f MiB (%.4f GiB)\n", to_mib(max_niblike), to_gib(max_niblike));
    std::fflush(stdout);
    return 0;
}
EOF

/opt/rocm/bin/hipcc -O2 --offload-arch=gfx1100 -no-pie /tmp/vram_probe.cpp -o /tmp/vram_probe 2>&1 || \
    /opt/rocm/bin/hipcc -O2 --offload-arch=gfx1100 /tmp/vram_probe.cpp -o /tmp/vram_probe 2>&1
if [ -x /tmp/vram_probe ]; then
    timeout 60 /tmp/vram_probe 2>&1 || log "WARN: vram_probe exited $?"
else
    log "ERROR: failed to compile /tmp/vram_probe"
fi

log "=== STAGE 2: REAL NIB CONFIG MATRIX ==="
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
for f in Inkling-2-Mini-NVFP4-Mixed.gguf mtp.safetensors; do
    HF_WANT=""
    hf_meta "$f"
    have=$(file_size "models/$f")
    log "downloading $f (have=$have want=${HF_WANT:-unknown})"
    for attempt in 1 2 3 4; do
        [ -n "$HF_WANT" ] && [ "$(file_size "models/$f")" = "$HF_WANT" ] && break
        aria2c -x 16 -s 16 -k 8M --file-allocation=none --continue=true \
            --max-tries=5 --retry-wait=5 --timeout=60 --connect-timeout=30 \
            --summary-interval=15 --console-log-level=notice \
            -d models -o "$f" "$HF/$f" || log "aria2c rc=$?"
    done
    log "$f ready ($(file_size "models/$f") bytes)"
done

cat > /tmp/probe_prompt.txt <<'EOF'
Write a single sentence explaining what an AMD RX 7900 XT GPU is.
EOF

SUMMARY=/tmp/nib_matrix_summary.txt
: > "$SUMMARY"

run_gen_case() {
    local NAME="$1"
    local KV="$2"
    local CTX="$3"
    local USE_MTP="$4"
    local EXTRA=()
    if [ "$USE_MTP" = "1" ]; then
        EXTRA=(--mtp models/mtp.safetensors)
    fi
    log "--- PROBE CASE: $NAME (kv=$KV ctx=$CTX mtp=$USE_MTP) ---"
    local OUT=/tmp/case_${NAME}.log
    if timeout 240 ./build/nib gen \
        -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
        --prompt-file /tmp/probe_prompt.txt \
        -n 16 \
        -c "$CTX" \
        --kv-type "$KV" \
        --sparse-pages 224 \
        "${EXTRA[@]}" > "$OUT" 2>&1; then
        local ENG_VRAM MTP_VRAM GEN_LINE
        ENG_VRAM=$(grep -m1 "total_vram=" "$OUT" | sed -n 's/.*total_vram=\([0-9.]* GiB ([^)]*)\).*/\1/p')
        MTP_VRAM=$(grep -m1 "drafter ready:" "$OUT" | sed -n 's/.*vram=\([0-9]* MiB\).*/\1/p')
        GEN_LINE=$(grep -E "decoded|mtp|tok/s" "$OUT" | tail -n 1)
        log "PASS [$NAME]: eng=${ENG_VRAM:-?} | mtp=${MTP_VRAM:-none} | ${GEN_LINE}"
        echo "PASS | $NAME | kv=$KV | ctx=$CTX | mtp=$USE_MTP | eng=${ENG_VRAM:-?} | mtp=${MTP_VRAM:-none}" >> "$SUMMARY"
    else
        local RC=$?
        local ERR_LINE ENG_LINE
        ENG_LINE=$(grep -m1 "total_vram=" "$OUT" || true)
        ERR_LINE=$(grep -Ei "hipMalloc|OOM|failed|error|out of memory" "$OUT" | tail -n 2 | tr '\n' ' ')
        log "FAIL [$NAME] (rc=$RC): ${ENG_LINE} | err=${ERR_LINE}"
        echo "FAIL | $NAME | kv=$KV | ctx=$CTX | mtp=$USE_MTP | rc=$RC | ${ENG_LINE} | err=${ERR_LINE}" >> "$SUMMARY"
    fi
}

# Core sim_gpu.sh & README comparison matrix
run_gen_case "q4_1_262k_nomtp"  "q4_1" 262848 0
run_gen_case "q4_1_262k_mtp"    "q4_1" 262848 1
run_gen_case "q4_0_262k_mtp"    "q4_0" 262848 1
run_gen_case "q8_0_131k_nomtp"  "q8_0" 131072 0
run_gen_case "q8_0_131k_mtp"    "q8_0" 131072 1
run_gen_case "q8_0_196k_nomtp"  "q8_0" 196608 0
run_gen_case "q8_0_229k_nomtp"  "q8_0" 229376 0
run_gen_case "q8_0_262k_nomtp"  "q8_0" 262848 0
run_gen_case "q8_0_98k_mtp"     "q8_0" 98304  1
run_gen_case "q8_0_65k_mtp"     "q8_0" 65536  1
run_gen_case "q4_1_229k_mtp"    "q4_1" 229376 1
run_gen_case "q4_1_196k_mtp"    "q4_1" 196608 1
run_gen_case "q4_1_163k_mtp"    "q4_1" 163840 1
run_gen_case "q4_1_131k_mtp"    "q4_1" 131072 1

# Also check multi-slot serve startup ceiling for 131k q4_1 + MTP (slots 4..8)
for S in 4 5 6 7 8; do
    NAME="serve_q4_1_131k_mtp_s${S}"
    log "--- PROBE CASE: $NAME ---"
    OUT=/tmp/case_${NAME}.log
    ./build/nib serve \
        -m models/Inkling-2-Mini-NVFP4-Mixed.gguf \
        --mtp models/mtp.safetensors \
        --no-vision \
        --slots "$S" \
        -c 131072 \
        --kv-type q4_1 \
        --sparse-pages 224 \
        --host 127.0.0.1 \
        --port 4455 > "$OUT" 2>&1 &
    PID=$!
    OK=0
    for i in $(seq 1 60); do
        if ! kill -0 "$PID" 2>/dev/null; then break; fi
        if curl -fsS --max-time 2 http://127.0.0.1:4455/health >/dev/null 2>&1; then
            OK=1; break
        fi
        sleep 2
    done
    kill "$PID" 2>/dev/null || true
    wait "$PID" 2>/dev/null || true
    if [ "$OK" = "1" ]; then
        ENG_VRAM=$(grep -m1 "total_vram=" "$OUT" | sed -n 's/.*total_vram=\([0-9.]* GiB ([^)]*)\).*/\1/p')
        MTP_VRAM=$(grep -m1 "drafter ready:" "$OUT" | sed -n 's/.*vram=\([0-9]* MiB\).*/\1/p')
        log "PASS [$NAME]: eng=${ENG_VRAM:-?} | mtp=${MTP_VRAM:-none}"
        echo "PASS | $NAME | kv=q4_1 | ctx=131072 | slots=$S | mtp=1 | eng=${ENG_VRAM:-?} | mtp=${MTP_VRAM:-none}" >> "$SUMMARY"
    else
        ENG_LINE=$(grep -m1 "total_vram=" "$OUT" || true)
        ERR_LINE=$(grep -Ei "hipMalloc|OOM|failed|error" "$OUT" | tail -n 2 | tr '\n' ' ')
        log "FAIL [$NAME]: ${ENG_LINE} | err=${ERR_LINE}"
        echo "FAIL | $NAME | kv=q4_1 | ctx=131072 | slots=$S | mtp=1 | ${ENG_LINE} | err=${ERR_LINE}" >> "$SUMMARY"
    fi
done

log "=== STAGE 2 MATRIX COMPLETE ==="
cat "$SUMMARY"
log "Probe complete; sleeping so logs remain inspectable until deleted."
sleep 86400
