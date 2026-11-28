#!/usr/bin/env python3
"""Concurrent-prefill probe: fire N prefills at once so several slots occupy the
GPU together, which is when scratch (private segment) traffic is most expensive.

usage: srv_conc.py [url] [--n 2] [--tokens 10240] [--rounds 3] [--json out]
"""
import argparse, json, random, string, threading, time, urllib.request

WORDS = ("attention key value query page sparse kernel register spill wavefront "
         "accumulator tile block head group tensor stride offset mask logits softmax "
         "engine infer token context window slot cache verify draft accept latency").split()


def prompt_of(n_tok, rnd):
    head = "req-" + "".join(rnd.choices(string.ascii_lowercase + string.digits, k=12)) + " "
    body, total = [], len(head)
    while total < n_tok * 4:
        c = " ".join(rnd.choices(WORDS, k=12))
        body.append(c)
        total += len(c) + 1
    return head + " ".join(body)


def one(url, payload, out, key):
    req = urllib.request.Request(url.rstrip("/") + "/v1/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=900) as r:
            b = json.loads(r.read())
        out[key] = (b["usage"]["prompt_tokens"], b["timings"]["prompt_ms"], time.perf_counter() - t0)
    except Exception as e:
        out[key] = (0, 0.0, time.perf_counter() - t0)
        print("  !! request failed:", e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://myai:8081")
    ap.add_argument("--n", type=int, default=2, help="concurrent requests")
    ap.add_argument("--tokens", type=int, default=10240)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--json", default="")
    a = ap.parse_args()
    rnd = random.Random(a.seed)
    model = json.load(urllib.request.urlopen(a.url.rstrip("/") + "/health"))["model"]

    res = []
    for r in range(a.rounds):
        out, ths = {}, []
        for i in range(a.n):
            p = prompt_of(a.tokens, rnd)
            t = threading.Thread(target=one, args=(a.url, {"model": model, "prompt": p,
                                      "max_tokens": 1, "temperature": 0.0, "stream": False}, out, i))
            ths.append(t)
        t0 = time.perf_counter()
        for t in ths:
            t.start()
        for t in ths:
            t.join()
        wall = time.perf_counter() - t0
        rows = [out[i] for i in sorted(out)]
        pt = rows[0][0]
        per = [round(x[1], 1) for x in rows]
        agg = sum(x[0] for x in rows) / wall
        res.append({"prompt_tokens": pt, "prompt_ms": per, "wall_s": round(wall, 3),
                    "agg_prefill_tps": round(agg, 1)})
        print(f"  round {r+1}: {a.n}x{pt:>6} tok  prompt_ms={per}  "
              f"wall={wall:>6.2f}s  aggregate={agg:>8.1f} tok/s")
    best = max(x["agg_prefill_tps"] for x in res)
    med = sorted(x["agg_prefill_tps"] for x in res)[len(res) // 2]
    print(f"# n={a.n} tokens~{a.tokens}: aggregate prefill_tps best={best} median={med}")
    if a.json:
        json.dump({"n": a.n, "target_tokens": a.tokens, "rounds": res,
                   "best": best, "median": med}, open(a.json, "w"), indent=1)
        print(f"# wrote {a.json}")


main()
