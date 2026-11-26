#!/usr/bin/env python3
"""Binary-search the largest -c (multiple of STEP, capped at 262848) that PASSes probe.py with no GTT spill.

Usage: search.py QUANT KV [--mtp] [--vision] [--slots N] [--lo L] [--hi H] [--step S]
A probe counts as clean when verdict == PASS and spill_mb <= SPILL_TOL, where
spill_mb = gtt_rise_mb - cpu_embd_host_mib - GTT_BASE (GTT_BASE = pinned-host/staging GTT measured on a
known-fitting config; env NIB_GTT_BASE, default 64 MB). Appends a summary line to /workspace/results/search.jsonl.
"""
import json, os, subprocess, sys
a = sys.argv[1:]
QUANT, KV = a[0], a[1]
flags = [x for x in ("--mtp", "--vision") if x in a]
opt = lambda k, d: int(a[a.index(k) + 1]) if k in a else d
SLOTS, LO, HI, STEP = opt("--slots", 1), opt("--lo", 4096), opt("--hi", 262848), opt("--step", 4096)
IMG = a[a.index("--img") + 1] if "--img" in a else None
GTT_BASE = float(os.environ.get("NIB_GTT_BASE", "64")); SPILL_TOL = float(os.environ.get("NIB_SPILL_TOL", "32"))
MAXC = 262848
name = f"s2_{QUANT}_{KV}_{'mtp' if '--mtp' in flags else 'nomtp'}_{'vis' if '--vision' in flags else 'novis'}_s{SLOTS}"
tried = {}

def run(ctx):
    if ctx in tried: return tried[ctx]
    tag = f"{name}_c{ctx}"
    cmd = ["python3", "/workspace/probe.py", tag, QUANT, KV, str(ctx), "--slots", str(SLOTS)] + flags
    if IMG: cmd += ["--img", IMG]
    subprocess.run(cmd, timeout=1500)
    r = json.loads(open("/workspace/results/probe.jsonl").read().strip().splitlines()[-1])
    spill = (r.get("gtt_rise_mb") or 0) - (r.get("cpu_embd_host_mib") or 0) - GTT_BASE
    ok = r["verdict"] == "PASS" and spill <= SPILL_TOL
    tried[ctx] = (ok, r, spill)
    print(f"   -> {name} c={ctx}: {'CLEAN' if ok else 'NOFIT'} verdict={r['verdict']} spill_est={spill:.0f}MB", flush=True)
    return tried[ctx]

def snap(c): return min(MAXC, max(STEP, (c // STEP) * STEP))
best = None
hi = MAXC if HI >= MAXC else snap(HI)
lo = LO
if run(hi)[0]: best = hi
elif hi > lo and run(lo)[0]:
    best = lo
    while hi - lo > STEP:
        mid = snap((lo + hi) // 2)
        if mid <= lo: mid = lo + STEP
        if mid >= hi: break
        if run(mid)[0]: lo = best = mid
        else: hi = mid
summary = {"search": name, "quant": QUANT, "kv": KV, "flags": flags, "slots": SLOTS, "max_clean_ctx": best,
           "tried": {c: {"ok": v[0], "verdict": v[1]["verdict"], "spill_est_mb": round(v[2]), "eng_gib": v[1]["total_vram_gib"],
                         "mtp_mib": v[1]["mtp_mib"], "tps": v[1]["steps"].get("text", {}).get("decode_tps")} for c, v in sorted(tried.items())}}
open("/workspace/results/search.jsonl", "a").write(json.dumps(summary) + "\n")
print(f"=== {name}: max clean ctx = {best}", flush=True)
