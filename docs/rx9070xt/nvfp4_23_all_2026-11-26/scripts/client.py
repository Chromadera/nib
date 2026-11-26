import base64, json, struct, sys, time, urllib.request, zlib, threading
URL = "http://127.0.0.1:4444"
def health():
    with urllib.request.urlopen(URL + "/health", timeout=10) as r: return json.loads(r.read())
def chat(messages, max_tokens=128, timeout=600):
    body = json.dumps({"model": "inkling-2-mini", "messages": messages, "temperature": 0.0, "max_tokens": max_tokens,
                       "stream": True, "stream_options": {"include_usage": True}}).encode()
    req = urllib.request.Request(URL + "/v1/chat/completions", data=body, headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); t1 = None; n = 0; usage = {}; text = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data: "): continue
                p = line[6:]
                if p == "[DONE]": break
                o = json.loads(p)
                if o.get("usage"): usage = o["usage"]
                ch = o.get("choices") or []
                if ch:
                    d = ch[0].get("delta") or {}
                    c = d.get("content") or d.get("reasoning_content")
                    if c:
                        if t1 is None: t1 = time.perf_counter()
                        n += 1; text.append(c)
    except urllib.error.HTTPError as e:
        return {"ok": False, "http": e.code, "err": e.read()[:200].decode("utf-8", "ignore")}
    except Exception as e:
        return {"ok": False, "err": repr(e)[:200]}
    t2 = time.perf_counter()
    comp = usage.get("completion_tokens", n); g = (t2 - t1) if t1 else (t2 - t0)
    return {"ok": True, "prompt_tokens": usage.get("prompt_tokens", 0), "completion_tokens": comp,
            "ttft_ms": round(((t1 or t2) - t0) * 1000, 1), "decode_tps": round((comp - 1) / g, 1) if g > 0 and comp > 1 else 0.0,
            "text": "".join(text)[:160]}
def make_png(path, w, h):
    hw, hh = w // 2, h // 2
    top = b"\x00" + b"\xdc\x26\x26" * hw + b"\x25\x63\xeb" * (w - hw)
    bot = b"\x00" + b"\x16\xa3\x4a" * hw + b"\xea\xb3\x08" * (w - hw)
    raw = top * hh + bot * (h - hh)
    ck = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xffffffff)
    open(path, "wb").write(b"\x89PNG\r\n\x1a\n" + ck(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + ck(b"IDAT", zlib.compress(raw, 1)) + ck(b"IEND", b""))
def img_msg(path, q):
    u = "data:image/png;base64," + base64.b64encode(open(path, "rb").read()).decode()
    return [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": u}}, {"type": "text", "text": q}]}]
def txt(tag): return [{"role": "user", "content": f"[{tag}-{time.time_ns()}] Write a compact C++17 ring buffer class with push/pop."}]
if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "probe":   # probe <tag> <slots>
        tag, slots = sys.argv[2], int(sys.argv[3])
        h = health(); print("HEALTH " + json.dumps(h), flush=True)
        r = chat(txt(tag), 128); print("SOLO " + json.dumps(r), flush=True)
        if slots > 1:
            res = [None] * slots
            def w(i): res[i] = chat(txt(f"{tag}-b{i}"), 96)
            th = [threading.Thread(target=w, args=(i,)) for i in range(slots)]; t0 = time.perf_counter()
            [t.start() for t in th]; [t.join() for t in th]; el = time.perf_counter() - t0
            ok = sum(1 for x in res if x and x.get("ok")); tot = sum(x.get("completion_tokens", 0) for x in res if x and x.get("ok"))
            print("BURST " + json.dumps({"n": slots, "ok": ok, "agg_tps": round(tot / el, 1), "per_stream": [x.get("decode_tps") for x in res if x]}), flush=True)
        r2 = chat(txt(tag + "-post"), 64); print("POST " + json.dumps(r2), flush=True)
    elif cmd == "genpng":
        for name, w, hgt in [("1080p", 1920, 1080), ("1440p", 2560, 1440), ("4k", 3840, 2160), ("max2560", 2560, 2560), ("card", 1024, 512)]:
            make_png(f"/workspace/img_{name}.png", w, hgt)
    elif cmd == "vision":  # vision <tag> <img names...>
        tag = sys.argv[2]
        h = health(); print("HEALTH " + json.dumps(h), flush=True)
        print("TEXT_BEFORE " + json.dumps(chat(txt(tag), 128)), flush=True)
        for nm in sys.argv[3:]:
            r = chat(img_msg(f"/workspace/img_{nm}.png", "List the four quadrant colors."), 128)
            try: hv = health()
            except Exception as e: hv = {"err": repr(e)}
            r["vision_workspace_mib"] = hv.get("vision_workspace_mib"); r["dev_free_gib"] = hv.get("device_free_vram_gib")
            print(f"IMG_{nm} " + json.dumps(r), flush=True)
        print("TEXT_AFTER " + json.dumps(chat(txt(tag + "-post"), 128)), flush=True)
