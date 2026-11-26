#!/usr/bin/env bash
# Reproducible RX 7900 XT Vision (--mmproj) + MTP matrix probe (reproduces Issue #5).
# Tests text baseline -> 1024x512 card -> 1920x1080 PNG -> post-1080p text regression with amd-smi telemetry.
set -uo pipefail

export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

NIB="${NIB:-/workspace/nib/build/nib}"
MODEL="${MODEL:-/workspace/nib/models/Inkling-2-Mini-NVFP4-Mixed.gguf}"
MTP="${MTP:-/workspace/nib/models/mtp.safetensors}"
MMPROJ="${MMPROJ:-/workspace/nib/models/mmproj-Inkling-2-Mini-Q8_0.gguf}"
CARD_IMG="${CARD_IMG:-/workspace/nib/tests/test_vision_card.png}"

cat << 'PYEOF' > /tmp/nib_vision_probe_client.py
import base64, json, os, struct, sys, time, urllib.request, zlib

def make_png(path, w, h):
    half_w, half_h = w // 2, h // 2
    row_top = b"\x00" + (b"\xdc\x26\x26" * half_w) + (b"\x25\x63\xeb" * (w - half_w))
    row_bot = b"\x00" + (b"\x16\xa3\x4a" * half_w) + (b"\xea\xb3\x08" * (w - half_w))
    raw = row_top * half_h + row_bot * (h - half_h)
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 1)) + chunk(b"IEND", b"")
    with open(path, "wb") as f:
        f.write(png)

def health():
    with urllib.request.urlopen("http://127.0.0.1:4444/health", timeout=10) as r:
        return json.loads(r.read())

def chat_stream(messages, max_tokens=128):
    body = json.dumps({
        "model": "inkling-2-mini", "messages": messages,
        "temperature": 0.0, "max_tokens": max_tokens,
        "stream": True, "stream_options": {"include_usage": True},
    }).encode()
    req = urllib.request.Request("http://127.0.0.1:4444/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t_first = None; chunks = 0; usage = {}
    with urllib.request.urlopen(req, timeout=120) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line.startswith("data: "): continue
            payload = line[6:]
            if payload == "[DONE]": break
            obj = json.loads(payload)
            if obj.get("usage"): usage = obj["usage"]
            choices = obj.get("choices") or []
            if choices:
                delta = choices[0].get("delta") or {}
                if delta.get("content") or delta.get("reasoning_content"):
                    if t_first is None: t_first = time.perf_counter()
                    chunks += 1
    t_end = time.perf_counter()
    ttft_ms = (t_first - t0) * 1000.0 if t_first else (t_end - t0) * 1000.0
    gen_s = (t_end - t_first) if t_first else (t_end - t0)
    comp = usage.get("completion_tokens", chunks)
    return {"prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": comp, "ttft_ms": round(ttft_ms, 1), "decode_tps": round(comp / gen_s, 1) if gen_s > 0 else 0.0}

def img_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()

if sys.argv[1] == "genpng":
    make_png("/tmp/test_1080p.png", 1920, 1080)
elif sys.argv[1] == "run":
    tag, card = sys.argv[2], sys.argv[3]
    h0 = health()
    print(f"[{tag}] INIT: vram={h0['vram_gib']:.3f} llm={h0['llm_vram_gib']:.3f} mtp={h0.get('mtp_vram_gib',0):.3f} vis={h0['vision_vram_gib']:.3f}", flush=True)
    print(f"[{tag}] TEXT_BEFORE:", chat_stream([{"role": "user", "content": f"[{tag}-{time.time_ns()}] Write a compact C++17 ring buffer."}], 128), flush=True)
    rB = chat_stream([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_url(card)}}, {"type": "text", "text": "Briefly describe this image."}]}], 128)
    print(f"[{tag}] CARD_IMG:", rB, "vis_vram=", round(health()["vision_vram_gib"], 3), flush=True)
    rC = chat_stream([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": img_url('/tmp/test_1080p.png')}}, {"type": "text", "text": "List the four quadrant colors."}]}], 128)
    print(f"[{tag}] 1080P_IMG:", rC, "vis_vram=", round(health()["vision_vram_gib"], 3), flush=True)
    print(f"[{tag}] TEXT_AFTER:", chat_stream([{"role": "user", "content": f"[{tag}-post-{time.time_ns()}] Write a compact C++17 ring buffer."}], 128), flush=True)
PYEOF

python3 /tmp/nib_vision_probe_client.py genpng
amd-smi metric -m | grep -E "USED_VRAM|FREE_VRAM|USED_GTT" || true

run_case() {
  local tag="$1" slots="$2" ctx="$3" kv="$4" use_mtp="$5"
  local logf="/tmp/srv_vis_${tag}.log"
  pkill -9 -x nib 2>/dev/null || true
  sleep 2
  echo "=== CASE: $tag (slots=$slots ctx=$ctx kv=$kv mtp=$use_mtp mmproj=ON) ==="
  local args=(-m "$MODEL" --mmproj "$MMPROJ" --slots "$slots" -c "$ctx" --kv-type "$kv" --sparse-pages 224 --host 127.0.0.1 --port 4444)
  [ "$use_mtp" = "1" ] && args+=(--mtp "$MTP")
  "$NIB" serve "${args[@]}" > "$logf" 2>&1 &
  local pid=$!
  local ready=0
  for i in $(seq 1 65); do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[$tag] START_FAILED (${i}s):"; tail -n 10 "$logf"; return 0
    fi
    if curl -sf http://127.0.0.1:4444/health > /dev/null 2>&1; then
      ready=1; echo "[$tag] READY (${i}s)"; break
    fi
    sleep 1
  done
  if [ "$ready" -eq 0 ]; then
    echo "[$tag] START_TIMEOUT"; tail -n 10 "$logf"; kill -9 "$pid" 2>/dev/null || true; return 0
  fi
  echo "[$tag] BOOT_SMI: $(amd-smi metric -m | grep -E 'USED_VRAM|FREE_VRAM|USED_GTT' | tr '\n' ' ')"
  python3 /tmp/nib_vision_probe_client.py run "$tag" "$CARD_IMG" || tail -n 15 "$logf"
  echo "[$tag] END_SMI:  $(amd-smi metric -m | grep -E 'USED_VRAM|FREE_VRAM|USED_GTT' | tr '\n' ' ')"
  kill -9 "$pid" 2>/dev/null || true
}

run_case "solo_s1_32k_mtp"      1  32768 q4_1 1
run_case "solo_s1_64k_mtp"      1  65536 q4_1 1
run_case "mandrav_s4_64k_mtp"   4  65536 q4_1 1
run_case "solo_s1_128k_nomtp"   1 131072 q4_1 0
run_case "solo_s1_64k_q8_nomtp" 1  65536 q8_0 0
run_case "mandrav_s2_128k_mtp"  2 131072 q4_1 1
run_case "mandrav_s4_128k_mtp"  4 131072 q4_1 1
