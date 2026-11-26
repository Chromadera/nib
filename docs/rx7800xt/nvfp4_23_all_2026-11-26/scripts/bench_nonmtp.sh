#!/usr/bin/env bash
# Non-MTP nib bench depth sweeps (pp512 + tg128, r2, sparse224, --cpu-embd) at each quant/KV's Stage-2 max clean ctx.
cd /workspace/nib; mkdir -p /workspace/logs/bench /workspace/results
L=/workspace/logs/bench_nonmtp.log
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $L; }
smi_loop(){ while :; do amd-smi metric -m 2>/dev/null | awk -v t=$(date +%s) '/USED_VRAM:/{u=$2}/FREE_VRAM:/{f=$2}/USED_GTT:/{g=$2}END{print t,u,f,g}' | head -1; sleep 1; done; }
run_bench(){ # quant kv ctx
  local Q=$1 KV=$2 C=$3 T=bench_${1}_${2}_c${3}
  local MAXD=$(( (C - 1024) / 1024 * 1024 ))
  local DEP="0"; for d in 8192 16384 32768 49152 65536 98304 131072; do [ $d -lt $MAXD ] && DEP="$DEP,$d"; done; DEP="$DEP,$MAXD"
  log "=== $T depths=$DEP"
  { echo "--- amd-smi before"; amd-smi metric -m | grep -E "VRAM|GTT"; } > /workspace/logs/bench/$T.log
  smi_loop > /workspace/logs/bench/$T.smi & SP=$!
  timeout 1800 ./build/nib bench -m /workspace/models/Inkling-2-Mini-NVFP4-$Q.gguf --cpu-embd -c $C -d $DEP -p 512 -n 128 -r 2 --kv-type $KV --sparse-pages 224 >> /workspace/logs/bench/$T.log 2>&1
  local RC=$?; kill $SP; sleep 3
  { echo "rc=$RC"; echo "--- amd-smi after"; amd-smi metric -m | grep -E "VRAM|GTT"; } >> /workspace/logs/bench/$T.log
  log "$T rc=$RC maxGTT=$(awk '{print $4}' /workspace/logs/bench/$T.smi | sort -n | tail -1) minFree=$(awk '{print $3}' /workspace/logs/bench/$T.smi | sort -n | head -1)"
  grep -E "^\|" /workspace/logs/bench/$T.log | tee -a $L
}
python3 - <<'PY' > /tmp/bench_cfgs.txt
import json
best = {}
for l in open("/workspace/results/search.jsonl"):
    r = json.loads(l)
    if r["slots"] == 1 and not r["flags"] and r["max_clean_ctx"]:
        best[(r["quant"], r["kv"])] = r["max_clean_ctx"]
for (q, kv), c in sorted(best.items(), key=lambda x: (x[0][0] != "All", x[0][1])):
    print(q, kv, c)
PY
cat /tmp/bench_cfgs.txt | tee -a $L
while read Q KV C; do run_bench $Q $KV $C; done < /tmp/bench_cfgs.txt
log "=== NONMTP DONE"
