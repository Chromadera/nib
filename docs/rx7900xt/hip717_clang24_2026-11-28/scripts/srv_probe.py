#!/usr/bin/env python3
"""Deterministic output capture for cross-build equivalence checks.

Same prompts, temperature=0, same max_tokens -> the generated text and the
per-request timings/MTP acceptance must match between two builds when a change
is purely a code-generation change.

usage: srv_probe.py [url] --json out.json
"""
import argparse, hashlib, json, urllib.request

PROMPTS = [
    "Q: A GPU kernel keeps 256 vector registers live and the compiler reports 44 spilled "
    "VGPRs. Explain in three sentences what happens at runtime and why it is usually bad.\nA:",
    "Translate to Python and explain the complexity: for each query token we score every "
    "page of the key/value cache, keep the top 224 pages, then attend only to those.\nA:",
    "Write exactly eight bullet points listing risks of fully unrolling a nested loop that "
    "loads per-element bias values in a fused attention kernel.\nA:",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("url", nargs="?", default="http://myai:8081")
    ap.add_argument("--json", required=True)
    ap.add_argument("--max-tokens", type=int, default=64)
    ap.add_argument("--repeat", type=int, default=2, help="repeat set to expose nondeterminism")
    a = ap.parse_args()
    model = json.load(urllib.request.urlopen(a.url.rstrip("/") + "/health"))["model"]

    out = {"model": model, "items": []}
    for r in range(a.repeat):
        for i, p in enumerate(PROMPTS):
            req = urllib.request.Request(a.url.rstrip("/") + "/v1/completions",
                                         data=json.dumps({"model": model, "prompt": p,
                                                         "max_tokens": a.max_tokens,
                                                         "temperature": 0.0, "seed": 1234,
                                                         "stream": False}).encode(),
                                         headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                b = json.loads(resp.read())
            text = b["choices"][0]["text"]
            t = b["timings"]
            rec = {"prompt": i, "repeat": r, "text": text,
                   "sha256": hashlib.sha256(text.encode()).hexdigest()[:16],
                   "completion_tokens": b["usage"]["completion_tokens"],
                   "prompt_ms": round(t["prompt_ms"], 2),
                   "decode_tps": round(t["predicted_per_second"], 2),
                   "mtp_tok_per_round": round(t.get("mtp_tok_per_round", 0.0), 4)}
            out["items"].append(rec)
            print(f"  p{i} r{r}: sha={rec['sha256']} gen={rec['completion_tokens']:>3} "
                  f"decode_tps={rec['decode_tps']:>7} mtp/round={rec['mtp_tok_per_round']}")
    json.dump(out, open(a.json, "w"), indent=1)
    print(f"# wrote {a.json}")
    print("# sha set:", " ".join(sorted({i["sha256"] for i in out["items"]})))


main()
