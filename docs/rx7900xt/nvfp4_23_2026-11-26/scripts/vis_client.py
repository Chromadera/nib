#!/usr/bin/env python3
"""Vision + no-MTP probe for one running `nib serve`. Writes one JSON line per request to stdout.

Per request: TTFT, decode tok/s = (completion_tokens - 1) / (first streamed token -> end of stream),
prompt tokens, the server's own `timings`, and amd-smi min FREE_VRAM / max USED_GTT sampled every
200 ms while the request runs.

Exit codes: 0 done, 3 a request STALLed (caller must kill nib), 4 server went down.
"""
import base64, json, random, re, socket, struct, subprocess, sys, threading, time, urllib.request, zlib

PORT = 4444
TAG, CTX = sys.argv[1], int(sys.argv[2])
GAP_S = 120          # no streamed token for this long -> STALL
FIRST_S = 120        # first token limit for normal requests
FIRST_FILL_S = 600   # first token limit for the fill test (prefill of ~250k tokens)

def png(path, w, h):
    # Smooth two-axis gradient with a grid, so the encoder sees real structure.
    rows = []
    for y in range(h):
        g = (y * 255) // max(h - 1, 1)
        row = bytearray(b"\x00")
        for x in range(w):
            r = (x * 255) // max(w - 1, 1)
            on = (x // 160 + y // 160) % 2
            row += bytes((r, g, 200 if on else 40))
        rows.append(bytes(row))
    raw = b"".join(rows)
    def ck(t, d): return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + ck(b"IDAT", zlib.compress(raw, 1)) + ck(b"IEND", b""))

SIZES = {"1080p": (1920, 1080), "1440p": (2560, 1440), "4k": (3840, 2160), "2560sq": (2560, 2560)}

def smi_once():
    out = subprocess.run(["amd-smi", "metric", "-m"], capture_output=True, text=True, timeout=10).stdout
    f = re.search(r"\bFREE_VRAM:\s*(\d+)", out); g = re.search(r"\bUSED_GTT:\s*(\d+)", out)
    u = re.search(r"\bUSED_VRAM:\s*(\d+)", out)
    return (int(f.group(1)) if f else None, int(g.group(1)) if g else None, int(u.group(1)) if u else None)

class Smi(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = False; self.min_free = None; self.max_gtt = None; self.n = 0
    def run(self):
        while not self.stop:
            try:
                f, g, _ = smi_once()
                if f is not None: self.min_free = f if self.min_free is None else min(self.min_free, f)
                if g is not None: self.max_gtt = g if self.max_gtt is None else max(self.max_gtt, g)
                self.n += 1
            except Exception:
                pass
            time.sleep(0.2)

def health():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10) as r:
        return json.loads(r.read())

def chat(content, max_tokens, first_s=FIRST_S):
    body = json.dumps({"model": "inkling-2-mini", "messages": [{"role": "user", "content": content}],
                       "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                                 headers={"Content-Type": "application/json"})
    smi = Smi(); smi.start()
    t0 = time.perf_counter(); t1 = None; usage = {}; timings = {}; finish = None; n = 0
    st = {"last": t0, "resp": None, "fired": False, "done": False}

    def watchdog():
        # Prefill (no headers yet) is covered by the socket timeout; this covers the gaps between tokens.
        while not st["done"]:
            time.sleep(1)
            lim = first_s if t1 is None else GAP_S
            if time.perf_counter() - st["last"] > lim and st["resp"] is not None:
                st["fired"] = True
                try: st["resp"].fp.raw._sock.shutdown(socket.SHUT_RDWR)
                except Exception: pass
                return
    threading.Thread(target=watchdog, daemon=True).start()
    try:
        with urllib.request.urlopen(req, timeout=first_s) as resp:
            st["resp"] = resp
            try: resp.fp.raw._sock.settimeout(GAP_S)
            except Exception: pass
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: "): continue
                if line[6:] == "[DONE]": break
                o = json.loads(line[6:])
                if o.get("usage"): usage = o["usage"]
                if o.get("timings"): timings = o["timings"]
                for c in o.get("choices") or []:
                    if c.get("finish_reason"): finish = c["finish_reason"]
                    d = c.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        now = time.perf_counter()
                        if t1 is None: t1 = now
                        st["last"] = now; n += 1
        status = "STALL" if st["fired"] else ("PASS" if usage else "ERR_NO_USAGE")
    except urllib.error.HTTPError as e:
        status = f"HTTP_{e.code}"
    except (socket.timeout, TimeoutError):
        status = "STALL"
    except Exception as e:
        status = "STALL" if st["fired"] else f"ERR_{type(e).__name__}"
    t2 = time.perf_counter(); st["done"] = True; smi.stop = True; smi.join(1)
    comp = usage.get("completion_tokens", n)
    gen = (t2 - t1) if t1 else 0
    return {"status": status, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": comp,
            "streamed_chunks": n, "finish_reason": finish,
            "ttft_ms": round(((t1 or t2) - t0) * 1000, 1), "wall_s": round(t2 - t0, 2),
            "decode_tps": round((comp - 1) / gen, 2) if gen > 0 and comp > 1 else None,
            "srv_predicted_tps": timings.get("predicted_per_second"), "srv_prompt_ms": timings.get("prompt_ms"),
            "srv_prompt_tps": timings.get("prompt_per_second"), "cache_hit_tokens": timings.get("cache_hit_tokens"),
            "min_free_mb": smi.min_free, "max_gtt_mb": smi.max_gtt, "smi_samples": smi.n}

def img(name):
    path = f"/tmp/probe_{name}.png"
    try: open(path, "rb").close()
    except FileNotFoundError: png(path, *SIZES[name])
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()}}

BOOT_GTT = None
def emit(step, r, **kw):
    h = {}
    try:
        hh = health(); h = {k: hh.get(k) for k in ("vram_gib", "vision_vram_gib", "vision_workspace_mib",
                                                     "vision_workspace_peak_mib", "device_free_vram_gib", "cpu_embd")}
    except Exception as e:
        h = {"health": f"DOWN {type(e).__name__}"}
    if BOOT_GTT is not None and r.get("max_gtt_mb") is not None:
        r["gtt_rise_mb"] = r["max_gtt_mb"] - BOOT_GTT
    print(json.dumps({"tag": TAG, "ctx": CTX, "step": step, **kw, **r, **h}), flush=True)
    if r["status"] == "STALL": sys.exit(3)
    if "health" in h: sys.exit(4)

def filler(n_chars, seed):
    rnd = random.Random(seed); out = []; size = 0
    words = "sensor relay buffer cache shard ingest commit replay vector kernel queue packet frame".split()
    while size < n_chars:
        s = (f"{rnd.randint(10**5, 10**6)} {rnd.choice(words)}-{rnd.randint(0, 999)} value={rnd.random():.4f} "
             f"status={rnd.choice(['ok', 'warn', 'retry'])} note={rnd.choice(words)} {rnd.choice(words)}\n")
        out.append(s); size += len(s)
    return "".join(out)

nonce = f"[{TAG}-{time.time_ns()}] "
for name in SIZES: img(name)  # build PNGs before timing anything
try:
    hh = health(); f, g, u = smi_once(); BOOT_GTT = g
    print(json.dumps({"tag": TAG, "ctx": CTX, "step": "boot", "boot_free_mb": f, "boot_gtt_mb": g,
                      "boot_used_mb": u, **hh}), flush=True)
except Exception as e:
    print(json.dumps({"tag": TAG, "step": "boot", "error": str(e)}), flush=True); sys.exit(1)

STORY = "Describe this image in detail, then write a 400-word short story inspired by its colours and grid."
emit("text_512", chat(nonce + "Explain in about 500 words how a lock-free ring buffer works in C++17.", 512))
img_prompt = {}
for name in SIZES:
    r = chat([img(name), {"type": "text", "text": nonce + STORY}], 512)
    img_prompt[name] = r.get("prompt_tokens")
    emit(f"img_{name}_512", r, image=name)

# Fill the context: calibrate chars/token, then send ~(ctx - 8k) tokens of text plus a 1080p image.
cal = filler(40000, 1)
r = chat(nonce + "cal\n" + cal + "\nReply with OK.", 4)
cpt = 40000 / max((r.get("prompt_tokens") or 10000) - 30, 1)
emit("calibrate", r, chars_per_token=round(cpt, 3))

# Extra (diagnostic, not part of pass/fail): text prompt padded to the 2560x2560 image request's prompt
# length, so the image cost can be separated from plain context depth.
if img_prompt.get("2560sq"):
    pad = filler(int(max(img_prompt["2560sq"] - 60, 0) * cpt), 3)
    emit("x_text_depthmatch_512", chat(nonce + "log\n" + pad + "\nIgnore the log above. " +
                                      "Explain in about 500 words how a lock-free ring buffer works in C++17.", 512),
         target_tokens=img_prompt["2560sq"])

target = CTX - 8192
txt = filler(int(target * cpt * 0.97), 7)
emit("fill_1080p_256", chat([img("1080p"), {"type": "text", "text": nonce + "fill\n" + txt +
                                             "\nWhat is in the image? Answer in 200 words."}], 256, first_s=FIRST_FILL_S),
     target_tokens=target, chars_per_token=round(cpt, 3))
emit("after_fill_4k_256", chat([img("4k"), {"type": "text", "text": nonce + "after " + STORY}], 256), image="4k")
