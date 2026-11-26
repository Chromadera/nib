#!/usr/bin/env python3
"""Build benchmarks.json + markdown tables (tables.md) from raw."""
import json, re, statistics as st, sys, glob, os
D = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
R = f"{D}/raw"
out = {"node": {}, "stage1": {}, "stage2_searches": [], "stage2_probes": [], "nonmtp_bench": {}, "vision": [], "betterbench": {}}
md = []

# Stage 1
s1 = open(f"{R}/logs/stage1_vram_probe.log").read()
out["stage1"] = {k: float(v) for k, v in re.findall(r"RESULT \[(\w+)\] max_alloc=([0-9.]+) MiB", s1)}
out["stage1"]["idle_free_mib"] = float(re.search(r"idle_start\] free=([0-9.]+)", s1).group(1))
out["stage1"]["total_mib"] = float(re.search(r"totalGlobalMem=([0-9.]+)", s1).group(1))

# Stage 2
probes = [json.loads(l) for l in open(f"{R}/results/probe.jsonl")]
out["stage2_probes"] = probes
S = [json.loads(l) for l in open(f"{R}/results/search.jsonl")]
out["stage2_searches"] = S
md.append("### Stage 2 ceiling matrix (max clean `-c`, step 4096, `--cpu-embd --sparse-pages 224`)\n")
md.append("| Config | NVFP4-All max ctx | NVFP4-All engine VRAM | NVFP4-23 max ctx | NVFP4-23 engine VRAM |\n|---|---:|---:|---:|---:|")
def key(r): return (r["kv"], tuple(r["flags"]), r["slots"])
byq = {q: {key(r): r for r in S if r["quant"] == q} for q in ("All", "23")}
def eng(r):
    c = r["max_clean_ctx"]
    return r["tried"].get(str(c), r["tried"].get(c, {})).get("eng_gib") if c else None
for k in sorted(byq["All"], key=lambda k: (len(k[1]), k[2], k[0])):
    a, b = byq["All"].get(k), byq["23"].get(k)
    lab = f"`{k[0]}` {'+ vision (1080p) ' if k[1] else ''}`--slots {k[2]}`"
    f = lambda r: (f"**{r['max_clean_ctx']:,}**" if r and r["max_clean_ctx"] else "**does not fit**")
    g = lambda r: (f"{eng(r)} GiB" if r and eng(r) else "—")
    md.append(f"| {lab} | {f(a)} | {g(a)} | {f(b)} | {g(b)} |")
md.append("")

# MTP probes
mtp = [p for p in probes if p["mtp"]]
md.append("### MTP feasibility\n\n| Probe | Engine VRAM (before drafter) | Result |\n|---|---:|---|")
for p in mtp:
    md.append(f"| `{p['tag']}` | {p['total_vram_gib']} GiB | {p['verdict']}: `{(p['errors'] or [''])[-1][:90]}` |")
md.append("")

# Non-MTP bench
md.append("### Non-MTP `nib bench` (pp512 / tg128, r2)\n")
for f in sorted(glob.glob(f"{R}/logs/bench/bench_*.log")):
    t = os.path.basename(f)[:-4]; txt = open(f).read()
    rows = re.findall(r"\|\s+(\d+) \|\s+([0-9.]+) ± ([0-9.]+) \|\s+([0-9.]+) ± ([0-9.]+) \|", txt)
    eng_l = re.search(r"total_vram=([0-9.]+) GiB", txt)
    smi = [l.split() for l in open(f[:-4] + ".smi") if len(l.split()) == 4]
    out["nonmtp_bench"][t] = {"engine_vram_gib": float(eng_l.group(1)) if eng_l else None,
                              "graph_nodes": (re.search(r"HIP Graph: (\d+)", txt) or [None, None])[1],
                              "max_gtt_mb": max(int(s[3]) for s in smi), "min_free_mb": min(int(s[2]) for s in smi),
                              "rows": [{"depth": int(r[0]), "pp512": float(r[1]), "pp_sd": float(r[2]), "tg128": float(r[3]), "tg_sd": float(r[4])} for r in rows]}
# side-by-side table by depth
cfgs = list(out["nonmtp_bench"])
depths = sorted({r["depth"] for c in cfgs for r in out["nonmtp_bench"][c]["rows"]})
for metric, lab in (("pp512", "Prefill `pp512` (tok/s)"), ("tg128", "Decode `tg128` (tok/s)")):
    md.append(f"#### {lab}\n")
    md.append("| Depth | " + " | ".join(f"`{c.replace('bench_', '').replace('_', ' ')}` ({out['nonmtp_bench'][c]['engine_vram_gib']} GiB)" for c in cfgs) + " |")
    md.append("|---:|" + "---:|" * len(cfgs))
    for d in depths:
        cells = []
        for c in cfgs:
            v = [r for r in out["nonmtp_bench"][c]["rows"] if r["depth"] == d]
            cells.append(f"{v[0][metric]:,.2f}" if v else "—")
        md.append(f"| {d:,} | " + " | ".join(cells) + " |")
    md.append("")

# Vision
vis = [p for p in probes if p["tag"].startswith("vis_")]
out["vision"] = vis
md.append("### Vision matrix (`--mmproj`, no MTP, `--slots 1`)\n")
md.append("| Config | Engine VRAM | Text tok/s | 1080p | 1440p | 4K 3840×2160 | 2560×2560 | Post-image text tok/s | min free (MB) | GTT rise (MB) |\n|---|---:|---:|---|---|---|---|---:|---:|---:|")
for p in vis:
    s = p["steps"]
    def im(n):
        r = s.get(f"img_{n}")
        if not r: return "—"
        return f"PASS {r['decode_tps']} tok/s (TTFT {r['ttft_ms']:.0f} ms)" if r["status"] == "PASS" else f"`{r['status']}`"
    mf = min([x.get("min_free") or 99999 for x in (p["smi_boot"], p["smi_run"])])
    md.append(f"| `{p['tag']}` | {p['total_vram_gib']} GiB | {s.get('text', {}).get('decode_tps')} | {im('1080p')} | {im('1440p')} | {im('4k')} | {im('2560sq')} | "
              f"{s.get('post_image_text', {}).get('decode_tps')} | {mf} | {p['gtt_rise_mb']} |")
md.append("")

# BetterBench
W = None
for f in sorted(glob.glob(f"{R}/bb/bb_*.json")):
    if re.search(r"_(config|health_boot|health_end)\.json$", f): continue
    n = os.path.basename(f)[:-5]; d = json.load(open(f)); W = d["config"]["weights"]
    ss = {}
    for cat, runs in (d.get("single_stream") or {}).items():
        ok = [r for r in runs if r.get("ok")]
        if ok: ss[cat] = {"decode_tps_med": st.median(r["decode_tps"] for r in ok), "ttft_ms_med": st.median(r["ttft_ms"] for r in ok), "n": len(ok)}
    wsum = sum(W.get(c, 0) for c in ss)
    weighted = sum(W.get(c, 0) * v["decode_tps_med"] for c, v in ss.items()) / wsum if wsum else None
    pf = [{"target": p["target_depth"], "prompt_tokens": st.median(p["prompt_tokens"]), "pp_tps_med": st.median(p["pp_tps"]), "ttft_s_med": st.median(p["ttft_ms"]) / 1000}
          for p in (d.get("prefill") or []) if not p.get("skipped") and p.get("pp_tps")]
    cc = [{"level": c["level"], "ok": c["ok"], "requests": c["requests"], "agg_tps": c["aggregate_tps"],
           "per_stream_med": st.median(c["decode_tps"]) if c.get("decode_tps") else None,
           "ttft_p50_ms": st.median(c["ttft_ms"]) if c.get("ttft_ms") else None} for c in (d.get("concurrency") or [])]
    smi = [l.split() for l in open(f"{R}/logs/bb/{n}.smi") if len(l.split()) == 4]
    lg = open(f"{R}/logs/bb/{n}.log").read()
    out["betterbench"][n] = {"single_stream": ss, "weighted_decode_tps": weighted, "prefill": pf, "concurrency": cc,
                             "notes": d["env"].get("notes"), "max_model_len": d["config"].get("max_model_len"),
                             "smi_max_used_mb": max(int(s[1]) for s in smi), "smi_min_free_mb": min(int(s[2]) for s in smi),
                             "smi_max_gtt_mb": max(int(s[3]) for s in smi), "smi_min_gtt_mb": min(int(s[3]) for s in smi),
                             "batch_graphs": int(re.search(r"batch graphs captured: (\d+)", lg).group(1)) if "batch graphs captured" in lg else None}
B = out["betterbench"]
md.append("### BetterBench single-stream decode (median tok/s, 5 runs/category, greedy, cold nonce)\n")
ssn = [n for n in B if B[n]["single_stream"]]
md.append("| Category | weight | " + " | ".join(f"`{n[3:]}`" for n in ssn) + " |\n|---|---:|" + "---:|" * len(ssn))
cats = sorted({c for n in ssn for c in B[n]["single_stream"]}, key=lambda c: -W.get(c, 0))
for c in cats:
    md.append(f"| `{c}` | {W.get(c, 0)} | " + " | ".join(f"{B[n]['single_stream'][c]['decode_tps_med']:.1f}" if c in B[n]["single_stream"] else "—" for n in ssn) + " |")
md.append("| **Weighted** | | " + " | ".join(f"**{B[n]['weighted_decode_tps']:.1f}**" for n in ssn) + " |")
md.append("| TTFT median (ms, all cats) | | " + " | ".join(f"{st.median(v['ttft_ms_med'] for v in B[n]['single_stream'].values()):.0f}" for n in ssn) + " |\n")
md.append("### BetterBench prefill ladder (median)\n")
for n in ssn:
    md.append(f"`{n[3:]}`: " + " · ".join(f"{p['prompt_tokens']:,.0f} tok → **{p['pp_tps_med']:,.0f} tok/s** ({p['ttft_s_med']:.2f} s)" for p in B[n]["prefill"]) + "\n")
md.append("### BetterBench concurrency\n\n| Run | c | OK | Aggregate tok/s | Per-stream median tok/s | TTFT p50 (ms) |\n|---|--:|--:|--:|--:|--:|")
for n in B:
    for c in B[n]["concurrency"]:
        md.append(f"| `{n[3:]}` | {c['level']} | {c['ok']}/{c['requests']} | {c['agg_tps']:.1f} | {c['per_stream_med']:.1f} | {c['ttft_p50_ms']:.0f} |")
md.append("\n### BetterBench amd-smi (1 s sampling across whole run)\n\n| Run | max USED_VRAM | min FREE_VRAM | max USED_GTT | batch graphs |\n|---|--:|--:|--:|--:|")
for n in B:
    b = B[n]; md.append(f"| `{n[3:]}` | {b['smi_max_used_mb']} MB | {b['smi_min_free_mb']} MB | {b['smi_max_gtt_mb']} MB | {b['batch_graphs']} |")
json.dump(out, open(f"{D}/benchmarks.json", "w"), indent=1, default=str)
open(f"{D}/tables.md", "w").write("\n".join(md) + "\n")
print("\n".join(md))
