#!/usr/bin/env bash
# Reproducible zero-WAN-RTT localhost BetterBench concurrency sweep for RX 7900 XT (20 GB).
# Runs single-stream decode + true c=1,2,4,8 parallel streaming on --slots 8 -c 65536 q4_1 + MTP.
set -euo pipefail

export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

NIB="${NIB:-/workspace/nib/build/nib}"
MODEL="${MODEL:-/workspace/nib/models/Inkling-2-Mini-NVFP4-Mixed.gguf}"
MTP="${MTP:-/workspace/nib/models/mtp.safetensors}"
BB="${BB:-betterbench}"
OUT_PREFIX="${OUT_PREFIX:-/workspace/bb_s8_64k_local}"

echo "=== IDLE VRAM ==="
amd-smi metric -m | grep -E "VRAM|GTT" || true

"$NIB" serve \
  -m "$MODEL" \
  --mtp "$MTP" \
  --no-vision \
  --slots 8 \
  -c 65536 \
  --kv-type q4_1 \
  --sparse-pages 224 \
  --host 127.0.0.1 \
  --port 4444 \
  > /tmp/srv_s8_64k.log 2>&1 &
SRV_PID=$!
trap 'kill -9 "$SRV_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 90); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "Server exited early:"
    tail -n 30 /tmp/srv_s8_64k.log
    exit 1
  fi
  if curl -sf http://127.0.0.1:4444/health > /tmp/health.json 2>/dev/null; then
    echo "SERVER_READY (${i}s): $(cat /tmp/health.json)"
    break
  fi
  sleep 1
done

echo "=== POST-START VRAM ==="
amd-smi metric -m | grep -E "VRAM|GTT" || true

"$BB" run \
  --no-update-check \
  --endpoint http://127.0.0.1:4444/v1 \
  --model inkling-2-mini \
  --quick \
  --greedy \
  --decode \
  --concurrency \
  --max-model-len 65536 \
  --name rx7900xt-slots8-65536-local \
  --note engine=nib \
  --note gpu=RX_7900_XT \
  --note quant=NVFP4-Mixed \
  --note slots=8 \
  --note ctx=65536 \
  --note kv=q4_1 \
  --note sparse_pages=224 \
  --note mtp=dspark \
  --note vision=none \
  --out "${OUT_PREFIX}.json" \
  --html-out "${OUT_PREFIX}.html"

echo "=== POST-CONCURRENCY VRAM ==="
amd-smi metric -m | grep -E "VRAM|GTT" || true
echo -n "Static batch HIP graphs captured: "
grep -c "Captured static batch" /tmp/srv_s8_64k.log || true
