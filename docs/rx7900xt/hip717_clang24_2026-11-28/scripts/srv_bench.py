#!/usr/bin/env python3
"""End-to-end throughput probe for a running nib server (OpenAI-ish API).

Prefill is measured with tiny completions so prompt processing dominates; each
prompt starts with a unique random head so the prefix cache cannot inflate the
numbers.  Decode is measured on a short prompt with a long completion.

usage: srv_bench.py [base_url] [--sizes 1024,4096,...] [--rounds 3] [--decode-toks 128]
"""
import argparse, json, random, string, time, urllib.request

WORDS = ("attention key value query page sparse kernel register spill wavefront "
         "accumulator tile block head group tensor stride offset mask logits softmax "
         "engine infer token context window slot cache verify draft accept latency").split()


def prompt_of(n_tok, rnd):
    """Roughly n_tok tokens (~4 chars/token for this tokenizer) with a unique head."""
    head = "req-" + "".join(rnd.choices(string.ascii_lowercase + string.digits, k=12)) + " "
    body = []
    total = len(head)
    while total < n_tok * 4:
        chunk = " ".join(rnd.choices(WORDS, k=12))
        body.append(chunk)
        total += len(chunk) + 1
    return head + " ".join(body)


def post(url, payload, timeout=600):
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read())
    return body, time.perf_counter() - t0


def get(url, path, timeout=30):
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://myai:8081")
    ap.add_argument("--sizes", default="1024,4096,8192,16384,32768,49152")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--decode-toks", type=int, default=128)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default="", help="also dump results to this path")
    a = ap.parse_args()
    rnd = random.Random(a.seed)

    results = {"url": a.url, "sizes": [], "decode": [], "seed": a.seed}
    h = get(a.url, "/health")
    results["health"] = {k: h[k] for k in ("model", "gpu_name", "gpu_arch", "max_ctx", "max_slots",
                                           "kv_type", "sparse_pages", "mtp", "version")}
    print(f"# {h['model']} on {h['gpu_name']} ({h['gpu_arch']}), ctx={h['max_ctx']}, "
          f"slots={h['max_slots']}, kv={h['kv_type']}, sparse_pages={h['sparse_pages']}, "
          f"mtp={h['mtp']}, free_vram={h['device_free_vram_gib']:.2f} GiB")

    print("\n# prefill: prompt_tps (server-measured) and wall seconds")
    for n in [int(x) for x in a.sizes.split(",")]:
        rows = []
        for r in range(a.rounds):
            p = prompt_of(n, rnd)
            body, wall = post(a.url, {"model": h["model"], "prompt": p, "max_tokens": 1,
                                      "temperature": 0.0, "stream": False})
            u = body.get("usage", {})
            t = body.get("timings", {})
            pt, pm = u.get("prompt_tokens", 0), t.get("prompt_ms", 0.0)
            cached = t.get("cache_hit_tokens", 0)
            fresh = max(0, pt - cached)
            rows.append((pt, pm, fresh / (pm / 1000.0) if pm else 0.0, cached, wall))
        best_tps = max(r[2] for r in rows)
        med_pm = sorted(r[1] for r in rows)[len(rows) // 2]
        tok = rows[0][0]
        cached = ",cached=%d" % rows[-1][3] if rows[-1][3] else ""
        results["sizes"].append({"prompt_tokens": tok, "prompt_ms_median": med_pm,
                                 "prefill_tps_best": best_tps,
                                 "rounds": [round(r[2], 1) for r in rows],
                                 "round_ms": [round(r[1], 1) for r in rows]})
        print(f"  prompt={tok:>6} tok  prompt_ms median={med_pm:>9.1f}  "
              f"prefill_tps best={best_tps:>9.1f}  (rounds: "
              + " ".join(f"{r[2]:.0f}" for r in rows) + f"){cached}")

    print("\n# decode: short prompt, %d tokens out" % a.decode_toks)
    drows = []
    for r in range(a.rounds):
        p = prompt_of(128, rnd) + " Write a detailed paragraph about RDNA3 kernel scheduling."
        body, wall = post(a.url, {"model": h["model"], "prompt": p, "max_tokens": a.decode_toks,
                                  "temperature": 0.0, "stream": False})
        u, t = body.get("usage", {}), body.get("timings", {})
        drows.append((u.get("completion_tokens", 0), t.get("predicted_ms", 0.0),
                      t.get("prompt_per_second", 0.0), wall))
        print(f"  round {r+1}: gen={u.get('completion_tokens',0):>4} tok  "
              f"decode_ms={t.get('predicted_ms',0.0):>8.1f}  "
              f"decode_tps={t.get('predicted_per_second',0.0):>7.1f}  "
              f"mtp_tok/round={t.get('mtp_tok_per_round',0.0):.2f}  "
              f"wall={wall:>6.2f}s")
    tps = [d[0] / (d[1] / 1000.0) for d in drows if d[1]]
    print(f"  decode_tps best={max(tps):.1f}  mean={sum(tps)/len(tps):.1f}")

    results["decode"].append({"best": max(tps), "mean": sum(tps) / len(tps),
                             "rounds": [round(d[1], 1) for d in drows]})
    if a.json:
        json.dump(results, open(a.json, "w"), indent=1)
        print(f"# wrote {a.json}")
    act = get(a.url, "/api/activity")["items"][-1]
    print("\n# last activity record:", json.dumps(act))


main()
