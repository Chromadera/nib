#!/usr/bin/env python3
"""probe_client.py <port> <n_concurrent> <vision 0|1>  -> one JSON line.
Sends n concurrent nonce text requests (128 tok), then (vision) one 1920x1080 image request (128 tok)."""
import base64, json, struct, sys, threading, time, urllib.request, zlib
port, n, vis = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3] == "1"
U = f"http://127.0.0.1:{port}"
def png(w, h):
    rows = []
    for y in range(h):
        g = (y * 255) // (h - 1); row = bytearray(b"\x00")
        for x in range(w):
            row += bytes(((x * 255) // (w - 1), g, 200 if (x // 160 + y // 160) % 2 else 40))
        rows.append(bytes(row))
    ck = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ck(b"IDAT", zlib.compress(b"".join(rows), 1)) + ck(b"IEND", b"")
def chat(content, mt=128):
    body = json.dumps({"model": "inkling-2-mini", "messages": [{"role": "user", "content": content}], "temperature": 0.0,
                       "max_tokens": mt, "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(U + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t1 = None; usage = {}
    try:
        with urllib.request.urlopen(req, timeout=180) as r:
            for raw in r:
                l = raw.decode("utf-8", "ignore").strip()
                if not l.startswith("data: ") or l[6:] == "[DONE]": continue
                o = json.loads(l[6:]); usage = o.get("usage") or usage
                for c in o.get("choices") or []:
                    d = c.get("delta") or {}
                    if (d.get("content") or d.get("reasoning_content")) and t1 is None: t1 = time.perf_counter()
        st = "PASS" if usage else "NO_USAGE"
    except urllib.error.HTTPError as e: st = f"HTTP_{e.code}"
    except Exception as e: st = f"ERR_{type(e).__name__}"
    t2 = time.perf_counter(); ct = usage.get("completion_tokens", 0)
    return {"status": st, "pt": usage.get("prompt_tokens"), "ct": ct, "ttft_ms": round(((t1 or t2) - t0) * 1e3, 1),
            "tps": round((ct - 1) / (t2 - t1), 1) if t1 and ct > 1 and t2 > t1 else None}
res = [None] * n
def w(i): res[i] = chat(f"[probe-{time.time_ns()}-{i}] Write a compact C++17 ring buffer class with push/pop.")
th = [threading.Thread(target=w, args=(i,)) for i in range(n)]
t0 = time.perf_counter(); [t.start() for t in th]; [t.join() for t in th]; wall = time.perf_counter() - t0
out = {"text": res, "text_agg_tps": round(sum((r["ct"] or 0) for r in res) / wall, 1)}
if vis:
    url = "data:image/png;base64," + base64.b64encode(png(1920, 1080)).decode()
    out["img1080"] = chat([{"type": "image_url", "image_url": {"url": url}}, {"type": "text", "text": f"[{time.time_ns()}] Describe this image."}])
try: out["health"] = json.loads(urllib.request.urlopen(U + "/health", timeout=10).read())
except Exception as e: out["health"] = {"err": str(e)}
print(json.dumps(out))
