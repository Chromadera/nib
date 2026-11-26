#!/usr/bin/env python3
"""Build benchmarks.json for the RX 9070 XT NVFP4-23 vs NVFP4-All evaluation from the raw logs.

Usage: python3 -I scripts/build_benchmarks.py <eval_dir>
"""
import json, re, statistics, sys
from pathlib import Path

D = Path(sys.argv[1])
L = D / "logs"
CPU_EMBD_GTT_MB = 514  # measured: --cpu-embd pinned tok_embd shows as USED_GTT (+514 MB vs +45 MB with --no-cpu-embd)


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_stage2():
    rows = []
    for line in (L / "stage2_results.txt").read_text().splitlines():
        if not line.startswith(("PASS", "FAIL", "TIMEOUT")):
            continue
        p = [x.strip() for x in line.split("|")]
        m = dict(re.findall(r"(\w+)=(\S+)", p[2]))
        idle = re.search(r"idle\[usedVRAM freeVRAM GTT\]=(\d+) (\d+) (\d+)", line)
        st = re.search(r"peakUsed=(\d*) minFree=(\d*) peakGTT=(\d*)", line)
        eng = re.search(r"eng=([\d.]+) GiB \(weights=([\d.]+) GiB, kv\+scratch=([\d.]+) GiB\)", line)
        dev = re.search(r"dev=([\d.]+) / ([\d.]+) GiB used \(([\d.]+) GiB free", line)
        tps = re.search(r"solo_tps=([\d.]*) burst_ok=(\d*) agg=([\d.]*)", line)
        err = line.split("err=", 1)[1].strip() if "err=" in line else ""
        r = {
            "status": p[0], "tag": p[1], "quant": "NVFP4-" + m["q"], "kv": m["kv"], "ctx": int(m["ctx"]),
            "slots": int(m["slots"]), "mtp": m["mtp"] == "1", "vision": m["vis"] == "1",
            "boot_s": int(p[3].split("=")[1].rstrip("s")),
            "idle_used_vram_mb": int(idle.group(1)), "idle_gtt_mb": int(idle.group(3)),
            "peak_used_vram_mb": int(st.group(1)), "min_free_vram_mb": int(st.group(2)), "peak_gtt_mb": int(st.group(3)),
        }
        if r["status"] == "PASS":
            r["gtt_over_cpu_embd_baseline_mb"] = r["peak_gtt_mb"] - r["idle_gtt_mb"] - CPU_EMBD_GTT_MB
        if eng:
            r.update(engine_vram_gib=float(eng.group(1)), weights_gib=float(eng.group(2)), kv_scratch_gib=float(eng.group(3)))
        if dev:
            r.update(device_used_gib=float(dev.group(1)), device_free_gib_hipmeminfo=float(dev.group(3)))
        if tps:
            r.update(solo_decode_tps=num(tps.group(1)), burst_ok=int(tps.group(2)) if tps.group(2) else None, burst_agg_tps=num(tps.group(3)))
        if "WARNING: Only" not in err and err:
            r["error"] = err[:240]
        rows.append(r)
    return rows


def parse_max():
    out = {}
    for line in (L / "stage2_max.txt").read_text().splitlines():
        p = [x.strip() for x in line.split("|")]
        m = re.match(r"(\d+)", p[2])
        out[p[1]] = {"max_ctx": int(m.group(1)) if m else None, "detail": p[2]}
    return out


def parse_bench():
    res, cur = [], None
    for line in (L / "bench_results.txt").read_text().splitlines():
        if line.startswith("BENCH "):
            lb = line.split()[1]
            q, kv, ctx = {"all_q40_73k": ("All", "q4_0", 73728), "all_q41_64k": ("All", "q4_1", 65536), "all_q80_32k": ("All", "q8_0", 32768),
                          "23_q40_24k": ("23", "q4_0", 24576), "23_q41_16k": ("23", "q4_1", 16384), "23_q80_8k": ("23", "q8_0", 8192)}[lb]
            st = re.search(r"idle=\[(\d+) (\d+) (\d+)\] peakUsed=(\d+) minFree=(\d+) peakGTT=(\d+)", line)
            eng = re.search(r"\| ([\d.]+) GiB", line)
            gn = re.search(r"Graph: (\d+) kernel nodes", line)
            cur = {"label": lb, "quant": "NVFP4-" + q, "kv": kv, "ctx": ctx, "rc": int(re.search(r"rc=(\d+)", line).group(1)),
                   "engine_vram_gib": float(eng.group(1)), "static_decode_graph_nodes": int(gn.group(1)) if gn else None,
                   "idle_gtt_mb": int(st.group(3)), "peak_used_vram_mb": int(st.group(4)), "min_free_vram_mb": int(st.group(5)),
                   "peak_gtt_mb": int(st.group(6)), "gtt_over_cpu_embd_baseline_mb": int(st.group(6)) - int(st.group(3)) - CPU_EMBD_GTT_MB,
                   "rows": []}
            res.append(cur)
        m = re.match(r"\|\s*(\d+) \|\s*([\d.]+) ±\s*([\d.]+) \|\s*([\d.]+) ±\s*([\d.]+) \|", line)
        if m and cur is not None:
            cur["rows"].append({"depth": int(m.group(1)), "pp512_tps": float(m.group(2)), "pp512_sd": float(m.group(3)),
                                "tg128_tps": float(m.group(4)), "tg128_sd": float(m.group(5))})
    return res


def bb_summary(path):
    d = json.loads(path.read_text())
    w = d["config"]["weights"]
    cats = {}
    for c, runs in d["single_stream"].items():
        ok = [r for r in runs if r.get("ok")]
        cats[c] = {"n": len(ok), "decode_tps_median": round(statistics.median(r["decode_tps"] for r in ok), 1),
                   "ttft_ms_median": round(statistics.median(r["ttft_ms"] for r in ok), 1)}
    wsum = sum(w.values())
    weighted = sum(cats[c]["decode_tps_median"] * w[c] for c in w if c in cats) / wsum if cats else None
    conc = [{"level": c["level"], "ok": c["ok"], "requests": c["requests"], "aggregate_tps": round(c["aggregate_tps"], 1),
             "ttft_ms_p50": round(statistics.median(c["ttft_ms"]), 1), "per_req_decode_tps_median": round(statistics.median(c["decode_tps"]), 1)}
            for c in d.get("concurrency") or []]
    pf = [{"target_depth": p["target_depth"], "prompt_tokens_median": int(statistics.median(p["prompt_tokens"])),
           "ttft_ms_median": round(statistics.median(p["ttft_ms"]), 1), "pp_tps_median": round(statistics.median(p["pp_tps"]), 1)}
          for p in d.get("prefill") or []]
    return {"file": path.name, "notes": d["env"].get("notes"), "max_model_len": d["env"].get("max_model_len"),
            "config": {k: d["config"][k] for k in ("runs_per_category", "warmup", "greedy", "unique_nonce", "concurrency_levels",
                                                    "concurrency_requests", "prefill_depths", "prefill_runs")},
            "single_stream": cats, "weighted_decode_tps_median_of_category_medians": round(weighted, 1) if weighted else None,
            "concurrency": conc, "prefill": pf}


def parse_bb_results():
    out = {}
    for line in (L / "bb_results.txt").read_text().splitlines():
        lb = line.split()[1]
        if "rc=143" in line:
            continue  # aborted first attempt (ladder too short) — superseded, see README
        st = re.search(r"idle=\[(\d+) (\d+) (\d+)\] peakUsed=(\d+) minFree=(\d+) peakGTT=(\d+) batch_graphs=(\d+)", line)
        out[lb] = {"rc": int(re.search(r"rc=(\d+)", line).group(1)), "server_after": re.search(r"server=(\w+)", line).group(1),
                   "duration_s": int(re.search(r"dur=(\d+)s", line).group(1)), "idle_gtt_mb": int(st.group(3)),
                   "peak_used_vram_mb": int(st.group(4)), "min_free_vram_mb": int(st.group(5)), "peak_gtt_mb": int(st.group(6)),
                   "gtt_over_cpu_embd_baseline_mb": int(st.group(6)) - int(st.group(3)) - CPU_EMBD_GTT_MB,
                   "static_batch_graphs_captured": int(st.group(7))}
        pc = L / f"bb_{lb}_postcheck.txt"
        if pc.exists():
            t = pc.read_text()
            s = [json.loads(x[5:]) for x in t.splitlines() if x.startswith("SOLO ")]
            p = [json.loads(x[5:]) for x in t.splitlines() if x.startswith("POST ")]
            out[lb]["post_run_solo_decode_tps"] = [r.get("decode_tps") for r in s + p]
    return out


def parse_vis():
    p = L / "vis_results.txt"
    if not p.exists():
        return []
    res, cur = [], None
    for line in p.read_text().splitlines():
        if line.startswith("VIS "):
            m = re.match(r"VIS (\S+) q=(\S+) kv=(\S+) ctx=(\d+) slots=(\d+) server=(\S+) idle=\[([\d ]*)\] ready=\[([^\]]*)\] end=\[([\d ]*)\] peakUsed=(\d*) minFree=(\d*) peakGTT=(\d*)", line)
            idle = m.group(7).split()
            cur = {"tag": m.group(1), "quant": "NVFP4-" + m.group(2), "kv": m.group(3), "ctx": int(m.group(4)), "slots": int(m.group(5)),
                   "server_after": m.group(6), "idle_gtt_mb": int(idle[2]), "peak_used_vram_mb": num(m.group(10)),
                   "min_free_vram_mb": num(m.group(11)), "peak_gtt_mb": num(m.group(12)), "steps": {}}
            if cur["peak_gtt_mb"] is not None:
                cur["gtt_over_cpu_embd_baseline_mb"] = cur["peak_gtt_mb"] - cur["idle_gtt_mb"] - CPU_EMBD_GTT_MB
            res.append(cur)
        elif cur is not None:
            m = re.match(r"(TEXT_BEFORE|TEXT_AFTER|IMG_\w+) (\{.*\})$", line)
            if m:
                cur["steps"][m.group(1)] = json.loads(m.group(2))
    return res


def parse_selftests():
    out = []
    for f in sorted(L.glob("selftest_*.log")):
        t = f.read_text(errors="ignore")
        m = re.search(r"max_diff=([\d.]+) \| cos_sim=([\d.]+)", t)
        out.append({"log": f.name, "prefill_vs_decode_cos_sim": float(m.group(2)) if m else None,
                    "max_diff": float(m.group(1)) if m else None, "sampler_pass": t.count("-> PASS"),
                    "dspark_error": (re.search(r"\[dspark\] init failed: (.*)", t).group(1) if "[dspark] init failed" in t else None)})
    return out


doc = {
    "gpu": "AMD Radeon RX 9070 XT (16 GB GDDR6, gfx1201 / Navi 48, 64 CU / 32 WGP)",
    "host": "Windows/WSL2 (/dev/dxg, WDDM), Adrenalin 26.10.44-260817a-204192C-AIB, rocm/dev-ubuntu-24.04:7.2.4",
    "date": "2026-11-26",
    "nib": "0.1.2 (local uncommitted tree, built -DGPU_TARGET=gfx1201)",
    "models": {
        "NVFP4-23": {"bytes": 16099143008, "weights_gib_cpu_embd": 14.46, "repack_max_layer_arg": 22},
        "NVFP4-All": {"bytes": 15589536192, "weights_gib_cpu_embd": 13.98, "repack_max_layer_arg": 23},
    },
    "gtt_accounting": {"cpu_embd_pinned_gtt_mb": CPU_EMBD_GTT_MB,
                       "rule": "spill = peak USED_GTT - idle USED_GTT - 514 MB (--cpu-embd pinned tok_embd); amd-smi is GPU-wide under WSL2 and includes Windows host allocations"},
    "stage1_hip_ceiling_mib": 15672.0,
    "selftests": parse_selftests(),
    "stage2_probes": parse_stage2(),
    "stage2_max_ctx": parse_max(),
    "nonmtp_bench": parse_bench(),
    "betterbench": {},
    "betterbench_run_meta": parse_bb_results(),
    "vision": parse_vis(),
    "mtp": {"status": "does_not_fit", "detail": "DSparkDrafter: hipMalloc failed on both quants at the minimum -c 4096 (q4_0, --draft-ctx 512); see stage2_probes mtp_* and selftest_*_mtp logs"},
}
for f in sorted(D.glob("betterbench_*.json")):
    doc["betterbench"][f.stem.replace("betterbench_", "")] = bb_summary(f)
(D / "benchmarks.json").write_text(json.dumps(doc, indent=2))
print("wrote", D / "benchmarks.json")
