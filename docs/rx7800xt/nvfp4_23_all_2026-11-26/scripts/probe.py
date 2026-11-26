#!/usr/bin/env python3
"""Stage 2 ceiling probe: boot one `nib serve` config, exercise it, record VRAM/GTT, tear it down.

Usage: probe.py TAG QUANT KV CTX [--mtp] [--vision] [--slots N] [--img 1080p,4k] [--keep]
QUANT is 23 or All. Appends one JSON line to /workspace/results/probe.jsonl and prints it.

Per probe:
  * amd-smi idle snapshot before launch, 0.5 s amd-smi sampler from launch to teardown
  * boot (health within 300 s), 128-token greedy text decode, N concurrent 64-token requests when --slots>1,
    optional image requests followed by a post-image text decode
  * verdict: BOOT_FAIL / STALL / ERR / PASS; GTT rise = max USED_GTT during run - idle USED_GTT
"""
import base64, json, os, re, signal, socket, struct, subprocess, sys, threading, time, urllib.request, zlib

PORT = 4455
W = "/workspace"
MODELS = {"23": f"{W}/models/Inkling-2-Mini-NVFP4-23.gguf", "All": f"{W}/models/Inkling-2-Mini-NVFP4-All.gguf"}

a = sys.argv[1:]
TAG, QUANT, KV, CTX = a[0], a[1], a[2], int(a[3])
MTP = "--mtp" in a; VIS = "--vision" in a; KEEP = "--keep" in a
SLOTS = int(a[a.index("--slots") + 1]) if "--slots" in a else 1
IMGS = a[a.index("--img") + 1].split(",") if "--img" in a else (["1080p"] if VIS else [])
EXTRA = a[a.index("--extra") + 1].split() if "--extra" in a else []
os.makedirs(f"{W}/results", exist_ok=True); os.makedirs(f"{W}/logs/probe", exist_ok=True)
LOG = f"{W}/logs/probe/{TAG}.log"

def smi():
    out = subprocess.run(["amd-smi", "metric", "-m"], capture_output=True, text=True, timeout=15).stdout
    g = lambda k: (int(m.group(1)) if (m := re.search(rf"\b{k}:\s*(\d+)", out)) else None)
    return {"used": g("USED_VRAM"), "free": g("FREE_VRAM"), "gtt": g("USED_GTT")}

class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = False; self.s = []
    def run(self):
        while not self.stop:
            try: self.s.append((time.time(), smi()))
            except Exception: pass
            time.sleep(0.5)
    def window(self, t0, t1=None):
        v = [x for t, x in self.s if t >= t0 and (t1 is None or t <= t1)]
        if not v: return {}
        return {"max_used": max(x["used"] for x in v), "min_free": min(x["free"] for x in v),
                "max_gtt": max(x["gtt"] for x in v), "n": len(v)}

def png(path, w, h):
    rows = []
    for y in range(h):
        g = (y * 255) // max(h - 1, 1); row = bytearray(b"\x00")
        for x in range(w):
            on = (x // 160 + y // 160) % 2
            row += bytes(((x * 255) // max(w - 1, 1), g, 200 if on else 40))
        rows.append(bytes(row))
    ck = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
                           + ck(b"IDAT", zlib.compress(b"".join(rows), 1)) + ck(b"IEND", b""))
SIZES = {"1080p": (1920, 1080), "1440p": (2560, 1440), "4k": (3840, 2160), "2560sq": (2560, 2560)}
def img(name):
    p = f"/tmp/probe_{name}.png"
    if not os.path.exists(p): png(p, *SIZES[name])
    return {"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(open(p, "rb").read()).decode()}}

def health():
    with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=10) as r: return json.loads(r.read())

def chat(content, max_tokens, timeout=180):
    body = json.dumps({"model": "inkling-2-mini", "messages": [{"role": "user", "content": content}],
                       "temperature": 0.0, "max_tokens": max_tokens, "stream": True,
                       "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t1 = None; usage = {}; timings = {}; n = 0; status = "PASS"
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: "): continue
                if line[6:] == "[DONE]": break
                o = json.loads(line[6:])
                if o.get("usage"): usage = o["usage"]
                if o.get("timings"): timings = o["timings"]
                for c in o.get("choices") or []:
                    d = c.get("delta") or {}
                    if d.get("content") or d.get("reasoning_content"):
                        if t1 is None: t1 = time.perf_counter()
                        n += 1
        if not usage: status = "ERR_NO_USAGE"
    except urllib.error.HTTPError as e: status = f"HTTP_{e.code}"
    except (socket.timeout, TimeoutError): status = "STALL"
    except Exception as e: status = f"ERR_{type(e).__name__}"
    t2 = time.perf_counter(); comp = usage.get("completion_tokens", n); gen = (t2 - t1) if t1 else 0
    return {"status": status, "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": comp,
            "ttft_ms": round(((t1 or t2) - t0) * 1000, 1),
            "decode_tps": round((comp - 1) / gen, 2) if gen > 0 and comp > 1 else None,
            "srv_tps": timings.get("predicted_per_second")}

def grep(pat, text, grp=0, last=False):
    m = list(re.finditer(pat, text))
    return (m[-1] if last else m[0]).group(grp) if m else None

idle = smi()
cmd = [f"{W}/nib/build/nib", "serve", "-m", MODELS[QUANT], "--cpu-embd", "--kv-type", KV, "-c", str(CTX),
       "--slots", str(SLOTS), "--sparse-pages", "224", "--host", "127.0.0.1", "--port", str(PORT)]
cmd += ["--mtp", f"{W}/models/mtp.safetensors"] if MTP else []
cmd += ["--mmproj", f"{W}/models/mmproj-Inkling-2-Mini-Q8_0.gguf"] if VIS else ["--no-vision"]
cmd += EXTRA
samp = Sampler(); samp.start()
t_launch = time.time()
lf = open(LOG, "w"); lf.write("CMD: " + " ".join(cmd) + "\n"); lf.flush()
proc = subprocess.Popen(cmd, cwd=f"{W}/nib", stdout=lf, stderr=subprocess.STDOUT, start_new_session=True)
res = {"tag": TAG, "quant": QUANT, "kv": KV, "ctx": CTX, "mtp": MTP, "vision": VIS, "slots": SLOTS,
       "extra": EXTRA, "idle": idle, "steps": {}}
boot_ok = False
for i in range(600):
    if proc.poll() is not None: break
    try: res["health_boot"] = health(); boot_ok = True; break
    except Exception: time.sleep(0.5)
res["boot_s"] = round(time.time() - t_launch, 1)
t_ready = time.time()
verdict = "PASS"
if not boot_ok:
    verdict = "BOOT_FAIL" if proc.poll() is not None else "BOOT_STALL"
else:
    time.sleep(1.0)
    res["post_boot_smi"] = smi()
    nonce = f"[{TAG}-{time.time_ns()}] "
    TXT = nonce + "Explain in about 300 words how a lock-free single-producer ring buffer works in C++17."
    r = chat(TXT, 128); res["steps"]["text"] = r
    if r["status"] != "PASS": verdict = r["status"]
    if verdict == "PASS" and SLOTS > 1:
        out = [None] * SLOTS
        def w(k): out[k] = chat(f"[{TAG}-c{k}-{time.time_ns()}] Write a haiku about GPU memory number {k}, then explain it.", 64)
        th = [threading.Thread(target=w, args=(k,)) for k in range(SLOTS)]
        tc0 = time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]; tc = time.perf_counter() - tc0
        ok = [o for o in out if o and o["status"] == "PASS"]
        res["steps"]["concurrent"] = {"n": SLOTS, "ok": len(ok), "wall_s": round(tc, 2),
                                      "agg_tps": round(sum(o["completion_tokens"] for o in ok) / tc, 1) if ok else None,
                                      "per_stream_tps": sorted(o["decode_tps"] or 0 for o in ok)}
        if len(ok) < SLOTS: verdict = "CONC_FAIL"
    if verdict == "PASS" and VIS:
        for name in IMGS:
            r = chat([img(name), {"type": "text", "text": nonce + "Describe this image in detail."}], 128)
            try: r["vision_ws_peak_mib"] = health().get("vision_workspace_peak_mib"); r["smi_now"] = smi()
            except Exception: pass
            res["steps"][f"img_{name}"] = r
            if r["status"] != "PASS" and verdict == "PASS": verdict = f"IMG_{name}_{r['status']}"
            if r["status"] != "PASS" and not r["status"].startswith("HTTP_"): break
        r = chat(nonce + "after " + TXT, 128); res["steps"]["post_image_text"] = r
        if r["status"] != "PASS" and verdict == "PASS": verdict = "POST_IMG_" + r["status"]
    try: res["health_end"] = health()
    except Exception as e: res["health_end"] = {"down": str(e)}; verdict = verdict if verdict != "PASS" else "DOWN"
t_end = time.time()
if not KEEP or not boot_ok:
    try: os.killpg(proc.pid, signal.SIGTERM)
    except Exception: pass
    try: proc.wait(15)
    except Exception:
        os.killpg(proc.pid, signal.SIGKILL); proc.wait(10)
    time.sleep(3)
samp.stop = True; samp.join(2)
res["after"] = smi()
res["smi_boot"] = samp.window(t_launch, t_ready)
res["smi_run"] = samp.window(t_ready, t_end)
mg = max([x.get("max_gtt") or 0 for x in (res["smi_boot"], res["smi_run"])])
res["gtt_rise_mb"] = mg - (idle["gtt"] or 0)
log = open(LOG, errors="ignore").read()
res["ready_line"] = grep(r"\[InklingEngine\] Ready!.*", log)
res["total_vram_gib"] = (lambda v: float(v) if v else None)(grep(r"total_vram=([0-9.]+) GiB", log, 1))
res["kv_scratch_gib"] = (lambda v: float(v) if v else None)(grep(r"kv\+scratch=([0-9.]+) GiB", log, 1))
res["mtp_mib"] = (lambda v: float(v) if v else None)(grep(r"drafter ready:.*vram=([0-9.]+) MiB", log, 1))
res["device_line"] = grep(r"\[nib\] Device VRAM:.*", log)
res["vram_profile"] = grep(r"\[vram\]   profile:.*", log)
res["occupancy"] = grep(r"\[step7i\] occupancy.*", log)
res["decode_graph"] = grep(r"Captured 1-shot static decode HIP Graph: (\d+)", log, 1)
res["batch_graphs"] = len(re.findall(r"Captured static batch", log))
res["warn_lowfree"] = bool(re.search(r"WARNING: Only .* MiB free GPU VRAM", log))
res["errors"] = [l for l in log.splitlines() if re.search(r"hipMalloc|out of memory|OOM|failed|error|TDR|lost", l, re.I)][-4:]
h = res.get("health_end") or res.get("health_boot") or {}
res["cpu_embd_host_mib"] = h.get("cpu_embd_host_mib")
res["device_free_vram_gib"] = h.get("device_free_vram_gib")
res["verdict"] = verdict
open(f"{W}/results/probe.jsonl", "a").write(json.dumps(res) + "\n")
s = res["steps"].get("text", {})
print(f"{verdict:10s} {TAG:40s} eng={res['total_vram_gib']} mtp={res['mtp_mib']} boot={res['boot_s']}s "
      f"tps={s.get('decode_tps')} idleUsed={idle['used']} maxUsed={max([x.get('max_used') or 0 for x in (res['smi_boot'], res['smi_run'])])} "
      f"minFree={min([x.get('min_free') or 99999 for x in (res['smi_boot'], res['smi_run'])])} gttRise={res['gtt_rise_mb']} "
      f"conc={res['steps'].get('concurrent', {}).get('agg_tps')} imgs={[res['steps'].get('img_'+n, {}).get('status') for n in IMGS]}", flush=True)
