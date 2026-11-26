#!/usr/bin/env bash
# Reproducible zero-WAN-RTT localhost BetterBench concurrency sweep + live post-concurrency 1080p Vision check
# for AMD Radeon RX 7900 XTX (24 GB).
# Default CTX=196608 fits comfortably under Windows/WSL2 WDDM's ~23 GiB single-process DXG cap;
# set CTX=262848 on native Linux (/dev/kfd).
set -euo pipefail

export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

NIB="${NIB:-/workspace/nib/build/nib}"
MODEL="${MODEL:-/workspace/nib/models/Inkling-2-Mini-NVFP4-Mixed.gguf}"
MTP="${MTP:-/workspace/nib/models/mtp.safetensors}"
MMPROJ="${MMPROJ:-/workspace/nib/models/mmproj-Inkling-2-Mini-Q8_0.gguf}"
BB="${BB:-betterbench}"
CTX="${CTX:-196608}"
OUT_PREFIX="${OUT_PREFIX:-/workspace/bb_xtx_s8_${CTX}_visionfix}"

echo "=== IDLE VRAM ==="
amd-smi metric -m | grep -E "USED_VRAM|FREE_VRAM|USED_GTT" || true

"$NIB" serve \
  -m "$MODEL" \
  --mtp "$MTP" \
  --mmproj "$MMPROJ" \
  --slots 8 \
  -c "$CTX" \
  --kv-type q8_0 \
  --sparse-pages 224 \
  --host 127.0.0.1 \
  --port 4444 \
  > /tmp/srv_xtx_s8.log 2>&1 &
SRV_PID=$!
trap 'kill -9 "$SRV_PID" 2>/dev/null || true' EXIT

for i in $(seq 1 90); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    echo "Server exited early:"
    tail -n 30 /tmp/srv_xtx_s8.log
    exit 1
  fi
  if curl -sf http://127.0.0.1:4444/health > /tmp/xtx_health0.json 2>/dev/null; then
    echo "SERVER_READY (${i}s): $(cat /tmp/xtx_health0.json)"
    break
  fi
  sleep 1
done

echo "=== POST-START VRAM ==="
amd-smi metric -m | grep -E "USED_VRAM|FREE_VRAM|USED_GTT" || true

cat << 'CFG' > /tmp/bb_xtx_cfg.json
{
  "temperature": 0.0,
  "top_p": 0.95,
  "top_k": 20,
  "greedy": true,
  "warmup": 1,
  "runs_per_category": 5,
  "unique_nonce": true,
  "run_single_stream": true,
  "run_concurrency": true,
  "concurrency_levels": [1, 2, 4, 8],
  "concurrency_requests": 32,
  "run_prefill": false,
  "timeout_s": 600.0
}
CFG

"$BB" run \
  --no-update-check \
  --endpoint http://127.0.0.1:4444/v1 \
  --model inkling-2-mini \
  --config /tmp/bb_xtx_cfg.json \
  --max-model-len "$CTX" \
  --name "rx7900xtx-slots8-${CTX}-visionfix" \
  --note engine=nib \
  --note gpu=RX_7900_XTX \
  --note quant=NVFP4-Mixed \
  --note slots=8 \
  --note "ctx=${CTX}" \
  --note kv=q8_0 \
  --note sparse_pages=224 \
  --note mtp=dspark \
  --note vision=aliased-released \
  --out "${OUT_PREFIX}.json" \
  --html-out "${OUT_PREFIX}.html"

echo "=== POST-CONCURRENCY VRAM ==="
amd-smi metric -m | grep -E "USED_VRAM|FREE_VRAM|USED_GTT" || true
echo -n "Static batch HIP graphs captured: "
grep -c "Captured static batch" /tmp/srv_xtx_s8.log || true

echo "=== LIVE POST-CONCURRENCY 1080p VISION SMOKE ==="
python3 - << 'PYEOF'
import base64, json, struct, time, urllib.request, zlib
w, h = 1920, 1080
hw, hh = w // 2, h // 2
rt = b"\x00" + (b"\xdc\x26\x26" * hw) + (b"\x25\x63\xeb" * (w - hw))
rb = b"\x00" + (b"\x16\xa3\x4a" * hw) + (b"\xea\xb3\x08" * (w - hw))
raw = rt * hh + rb * (h - hh)
def ck(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
png = b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ck(b"IDAT", zlib.compress(raw, 1)) + ck(b"IEND", b"")
url = "data:image/png;base64," + base64.b64encode(png).decode()
body = json.dumps({
    "model": "inkling-2-mini",
    "messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": "List the four quadrant colors."}]}],
    "temperature": 0.0, "max_tokens": 96, "stream": True, "stream_options": {"include_usage": True}
}).encode()
req = urllib.request.Request("http://127.0.0.1:4444/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
t0 = time.perf_counter(); tf = None; ch = 0; u = {}
with urllib.request.urlopen(req, timeout=120) as resp:
    for line in resp:
        s = line.decode("utf-8", errors="ignore").strip()
        if not s.startswith("data: "): continue
        p = s[6:]
        if p == "[DONE]": break
        o = json.loads(p)
        if o.get("usage"): u = o["usage"]
        cs = o.get("choices") or []
        if cs:
            d = cs[0].get("delta") or {}
            if d.get("content") or d.get("reasoning_content"):
                if tf is None: tf = time.perf_counter()
                ch += 1
te = time.perf_counter()
ct = u.get("completion_tokens", ch)
print(f"POST_BB_1080P: ok=True pt={u.get('prompt_tokens',0)} ct={ct} ttft_ms={(tf-t0)*1000:.1f} tps={ct/(te-tf):.1f}")
h_end = json.loads(urllib.request.urlopen("http://127.0.0.1:4444/health", timeout=10).read())
print("POST_BB_HEALTH:", json.dumps(h_end))
PYEOF

echo "=== FINAL POST-VISION VRAM ==="
amd-smi metric -m | grep -E "USED_VRAM|FREE_VRAM|USED_GTT" || true
