#!/usr/bin/env bash
# After Stage 2: non-MTP bench sweeps -> vision matrix -> full BetterBench (single-stream+prefill at s1 max ctx; concurrency at max slots).
cd /workspace; L=/workspace/logs/phase3.log
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $L; }

log "=== PHASE 3 START"
bash /workspace/bench_nonmtp.sh
python3 - <<'PY' > /workspace/results/plan.txt
import json
S = [json.loads(l) for l in open("/workspace/results/search.jsonl")]
for q in ("All", "23"):
    s1 = [r for r in S if r["quant"] == q and r["slots"] == 1 and not r["flags"] and r["max_clean_ctx"]]
    b = max(s1, key=lambda r: (r["max_clean_ctx"], r["kv"] == "q4_1"))
    print("BB1", q, b["kv"], b["max_clean_ctx"], 1)
    ms = [r for r in S if r["quant"] == q and r["slots"] > 1 and not r["flags"] and r["max_clean_ctx"]]
    if ms:
        top = max(r["slots"] for r in ms)
        c = max((r for r in ms if r["slots"] == top), key=lambda r: (r["max_clean_ctx"], r["kv"] == "q4_1"))
        print("BB2", q, c["kv"], c["max_clean_ctx"], top)
    for r in S:
        if r["quant"] == q and r["flags"] == ["--vision"] and r["max_clean_ctx"]:
            print("VIS", q, r["kv"], r["max_clean_ctx"], 1)
            print("VIS", q, r["kv"], 8192, 1)
PY
cat /workspace/results/plan.txt | tee -a $L
grep ^VIS /workspace/results/plan.txt | sort -u | while read _ Q KV C S; do
  python3 /workspace/probe.py vis_${Q}_${KV}_c${C} $Q $KV $C --vision --img 1080p,1440p,4k,2560sq 2>&1 | tee -a $L
done
grep ^BB /workspace/results/plan.txt | while read K Q KV C S; do
  if [ $K = BB1 ]; then bash /workspace/bb_run.sh $Q $KV $C 1 bb_${Q}_s1_${KV}_c${C}; fi
  if [ $K = BB2 ]; then bash /workspace/bb_run.sh $Q $KV $C $S bb_${Q}_s${S}_${KV}_c${C}; fi
done
log "=== PHASE 3 DONE"
