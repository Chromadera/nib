#!/usr/bin/env bash
# Reproducible RX 7900 XTX (24 GB) Baseline vs Patched A/B + Hi-Res Vision Verification Suite.
# Compares build_base/nib vs build/nib on bench (non-MTP & MTP), --dump-embeds bit parity (1080p & 4K),
# and flagship 128K/256K q8_0 serve across 1080p -> 1440p -> 4K -> 2560x2560 max-res.
set -uo pipefail

export HIP_VISIBLE_DEVICES=0
export ROCR_VISIBLE_DEVICES=0

BASE="${BASE:-/workspace/nib/build_base/nib}"
NEW="${NEW:-/workspace/nib/build/nib}"
MODEL="${MODEL:-/workspace/nib/models/Inkling-2-Mini-NVFP4-Mixed.gguf}"
MTP="${MTP:-/workspace/nib/models/mtp.safetensors}"
MMPROJ="${MMPROJ:-/workspace/nib/models/mmproj-Inkling-2-Mini-Q8_0.gguf}"

echo "=== 1. BENCH A/B (q8_0, -c 16384) ==="
echo "--- BASE non-MTP ---"
"$BASE" bench -m "$MODEL" -c 16384 --kv-type q8_0 | grep -E "^\|"
echo "--- NEW non-MTP ---"
"$NEW" bench -m "$MODEL" -c 16384 --kv-type q8_0 | grep -E "^\|"
echo "--- BASE MTP ---"
"$BASE" bench -m "$MODEL" --mtp "$MTP" -c 16384 --kv-type q8_0 | grep -E "^\|"
echo "--- NEW MTP ---"
"$NEW" bench -m "$MODEL" --mtp "$MTP" -c 16384 --kv-type q8_0 | grep -E "^\|"

echo "=== 2. VISION --dump-embeds BIT PARITY + FLAGSHIP SERVE A/B ==="
python3 - "$BASE" "$NEW" "$MODEL" "$MTP" "$MMPROJ" << 'PYEOF'
import array, base64, json, os, struct, subprocess, sys, threading, time, urllib.request, zlib

BASE, NEW, MODEL, MTP, MMPROJ = sys.argv[1:6]

def make_png(path, w, h):
    if os.path.exists(path): return
    hw, hh = w // 2, h // 2
    rt = b"\x00" + (b"\xdc\x26\x26" * hw) + (b"\x25\x63\xeb" * (w - hw))
    rb = b"\x00" + (b"\x16\xa3\x4a" * hw) + (b"\xea\xb3\x08" * (w - hw))
    raw = rt * hh + rb * (h - hh)
    def ck(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", ihdr) + ck(b"IDAT", zlib.compress(raw, 1)) + ck(b"IEND", b""))

def smi():
    out = subprocess.check_output(["amd-smi", "metric", "-m"], text=True, stderr=subprocess.DEVNULL)
    r = {}
    for ln in out.splitlines():
        for k in ("USED_VRAM", "FREE_VRAM", "USED_GTT"):
            if k in ln:
                v = ln.split(":", 1)[1].strip().split()[0]
                if v.isdigit(): r[k] = int(v)
    return r

class Poll:
    def __init__(self):
        self.ev = threading.Event()
        self.pv = 0; self.mf = 10**9; self.pg = 0; self.n = 0
        self.t = threading.Thread(target=self._run, daemon=True)
    def start(self): self.t.start()
    def _run(self):
        while not self.ev.is_set():
            try:
                m = smi()
                if "USED_VRAM" in m:
                    self.pv = max(self.pv, m["USED_VRAM"])
                    self.mf = min(self.mf, m["FREE_VRAM"])
                    self.pg = max(self.pg, m.get("USED_GTT", 0))
                    self.n += 1
            except Exception: pass
            time.sleep(0.2)
    def stop(self):
        self.ev.set(); self.t.join(timeout=2)
        return {"peak_vram": self.pv, "min_free": self.mf if self.n else 0, "peak_gtt": self.pg}

def health():
    with urllib.request.urlopen("http://127.0.0.1:4444/health", timeout=10) as r:
        return json.loads(r.read())

def chat_stream(messages, max_tokens=96):
    body = json.dumps({"model":"inkling-2-mini","messages":messages,"temperature":0.0,"max_tokens":max_tokens,"stream":True,"stream_options":{"include_usage":True}}).encode()
    req = urllib.request.Request("http://127.0.0.1:4444/v1/chat/completions", data=body, headers={"Content-Type":"application/json"})
    t0 = time.perf_counter(); tf = None; ch = 0; u = {}
    try:
        with urllib.request.urlopen(req, timeout=180) as resp:
            for raw in resp:
                ln = raw.decode("utf-8", errors="ignore").strip()
                if not ln.startswith("data: "): continue
                pl = ln[6:]
                if pl == "[DONE]": break
                o = json.loads(pl)
                if o.get("usage"): u = o["usage"]
                cs = o.get("choices") or []
                if cs:
                    d = cs[0].get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        if tf is None: tf = time.perf_counter()
                        ch += 1
        te = time.perf_counter()
        ttft = (tf - t0)*1000 if tf else (te - t0)*1000
        gs = (te - tf) if tf else (te - t0)
        ct = u.get("completion_tokens", ch)
        return {"ok": True, "pt": u.get("prompt_tokens", 0), "ct": ct, "ttft_ms": round(ttft,1), "tps": round(ct/gs,1) if gs>0 else 0}
    except Exception as e:
        return {"ok": False, "err": str(e)}

def img_url(path):
    return "data:image/png;base64," + base64.b64encode(open(path,"rb").read()).decode()

make_png("/tmp/xtx_1080p.png", 1920, 1080)
make_png("/tmp/xtx_1440p.png", 2560, 1440)
make_png("/tmp/xtx_4k.png",    3840, 2160)
make_png("/tmp/xtx_max.png",   2560, 2560)

for tag, img in [("1080p", "/tmp/xtx_1080p.png"), ("4k", "/tmp/xtx_4k.png")]:
    for label, exe in [("BASE", BASE), ("NEW", NEW)]:
        p = Poll(); p.start()
        subprocess.run([exe, "vision", "-m", MODEL, "--mmproj", MMPROJ, "--image", img, "-c", "16384", "--kv-type", "q8_0", "-n", "32", "--bench-vision", "5", "--dump-embeds", f"/tmp/emb_{tag}_{label}.bin"], check=True, stdout=subprocess.DEVNULL)
        print(f"[VIS {tag} {label}] smi={p.stop()}", flush=True)
    b1 = open(f"/tmp/emb_{tag}_BASE.bin", "rb").read()
    b2 = open(f"/tmp/emb_{tag}_NEW.bin", "rb").read()
    a1 = array.array("f"); a1.frombytes(b1)
    a2 = array.array("f"); a2.frombytes(b2)
    diff = sum(1 for x, y in zip(a1, a2) if x != y)
    mad = max((abs(x - y) for x, y in zip(a1, a2)), default=0.0)
    print(f"[PARITY {tag}] floats={len(a1)} bit_diffs={diff} max_abs_diff={mad}", flush=True)

def test_serve(name, exe, args):
    subprocess.run(["pkill", "-9", "-x", "nib"], stderr=subprocess.DEVNULL)
    time.sleep(2)
    logf = f"/tmp/xtx_srv_{name}.log"
    proc = subprocess.Popen([exe, "serve"] + args + ["--host", "127.0.0.1", "--port", "4444"], stdout=open(logf, "w"), stderr=subprocess.STDOUT)
    try:
        for _ in range(70):
            if proc.poll() is not None:
                print(f"[{name}] START_FAIL rc={proc.returncode}", flush=True); return
            try: health(); break
            except Exception: time.sleep(1)
        for lb, path in [("1080p","/tmp/xtx_1080p.png"), ("1440p","/tmp/xtx_1440p.png"), ("4k","/tmp/xtx_4k.png"), ("max2560","/tmp/xtx_max.png")]:
            p = Poll(); p.start()
            r = chat_stream([{"role":"user","content":[{"type":"image_url","image_url":{"url":img_url(path)}},{"type":"text","text":"List the four quadrant colors briefly."}]}], 96)
            m = p.stop()
            try: hi = health()
            except Exception: hi = {}
            print(f"[{name}] {lb}: {r} vis={hi.get('vision_vram_gib',0):.3f} ws={hi.get('vision_workspace_mib',-1)} pk={hi.get('vision_workspace_peak_mib',-1)} smi={m}", flush=True)
    finally:
        proc.kill(); proc.wait()

test_serve("NEW_default_256k", NEW,  ["-m", MODEL, "--mtp", MTP, "--mmproj", MMPROJ])
test_serve("BASE_s4_128k_q8",  BASE, ["-m", MODEL, "--mtp", MTP, "--mmproj", MMPROJ, "--slots", "4", "-c", "131072", "--kv-type", "q8_0", "--sparse-pages", "224"])
test_serve("NEW_s4_128k_q8",   NEW,  ["-m", MODEL, "--mtp", MTP, "--mmproj", MMPROJ, "--slots", "4", "-c", "131072", "--kv-type", "q8_0", "--sparse-pages", "224"])
PYEOF
