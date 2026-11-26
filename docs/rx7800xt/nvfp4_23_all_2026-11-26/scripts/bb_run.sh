#!/usr/bin/env bash
# Full BetterBench 0.6.1 (runs_per_category 5, greedy, cold nonce, localhost) against nib serve.
# Usage: bb_run.sh QUANT KV CTX SLOTS NAME [phase flags e.g. --decode --prefill | --concurrency]
Q=$1 KV=$2 C=$3 S=$4 N=$5; shift 5; PH="$*"
cd /workspace/nib; mkdir -p /workspace/bb /workspace/logs/bb
L=/workspace/logs/bb/$N.log; SRV=/workspace/logs/bb/${N}_serve.log
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $L; }
smi_loop(){ while :; do amd-smi metric -m 2>/dev/null | awk -v t=$(date +%s) '/USED_VRAM:/{u=$2}/FREE_VRAM:/{f=$2}/USED_GTT:/{g=$2}END{print t,u,f,g}' | head -1; sleep 1; done; }
python3 - "$C" "$S" > /workspace/bb/${N}_config.json <<'PY'
import json, sys
c, s = int(sys.argv[1]), int(sys.argv[2])
base = json.load(open("/workspace/betterbench_slots8_262k_config.json"))
base["concurrency_levels"] = [x for x in [1, 2, 4, 8] if x <= max(s, 1)]
d = [x for x in [2560, 10240, 20480, 40960, 81920, 163840, 334000] if x + 2048 <= c]
top = (c - 2048) // 1024 * 1024
if top not in d and top > d[-1]: d.append(top)
base["prefill_depths"] = d
print(json.dumps(base, indent=2))
PY
log "=== BB $N: quant=$Q kv=$KV ctx=$C slots=$S phases='${PH:-all}'"; cat /workspace/bb/${N}_config.json >> $L
{ echo "--- amd-smi before"; amd-smi metric -m | grep -E "VRAM|GTT"; } >> $L
smi_loop > /workspace/logs/bb/$N.smi & SP=$!
./build/nib serve -m /workspace/models/Inkling-2-Mini-NVFP4-$Q.gguf --cpu-embd --no-vision --slots $S -c $C --kv-type $KV \
   --sparse-pages 224 --host 127.0.0.1 --port 4444 > $SRV 2>&1 &
NP=$!
for i in $(seq 1 300); do kill -0 $NP 2>/dev/null || { log "serve died"; tail -20 $SRV | tee -a $L; kill $SP; exit 1; }
  curl -sf http://127.0.0.1:4444/health > /workspace/bb/${N}_health_boot.json && break; sleep 1; done
log "health: $(cat /workspace/bb/${N}_health_boot.json)"
{ echo "--- amd-smi post-boot"; amd-smi metric -m | grep -E "VRAM|GTT"; } >> $L
/workspace/bbvenv/bin/betterbench run --no-update-check --endpoint http://127.0.0.1:4444/v1 --model inkling-2-mini \
  --config /workspace/bb/${N}_config.json --greedy $PH --max-model-len $C --name $N \
  --note engine=nib --note gpu=RX_7800_XT --note quant=NVFP4-$Q --note slots=$S --note ctx=$C --note kv=$KV \
  --note sparse_pages=224 --note mtp=off --note vision=none --note cpu_embd=on --note host=windows_wsl2 \
  --out /workspace/bb/$N.json --html-out /workspace/bb/$N.html >> $L 2>&1
RC=$?; log "betterbench rc=$RC"
curl -sf http://127.0.0.1:4444/health > /workspace/bb/${N}_health_end.json
{ echo "--- amd-smi post-run"; amd-smi metric -m | grep -E "VRAM|GTT"; } >> $L
log "batch graphs captured: $(grep -c 'Captured static batch' $SRV) | decode graph: $(grep -o 'static decode HIP Graph: [0-9]* kernel nodes' $SRV | head -1)"
log "smi: maxUsed=$(awk '{print $2}' /workspace/logs/bb/$N.smi | sort -n | tail -1) minFree=$(awk '{print $3}' /workspace/logs/bb/$N.smi | sort -n | head -1) maxGTT=$(awk '{print $4}' /workspace/logs/bb/$N.smi | sort -n | tail -1) minGTT=$(awk '{print $4}' /workspace/logs/bb/$N.smi | sort -n | head -1)"
kill $NP; sleep 5; kill -9 $NP 2>/dev/null; kill $SP
{ echo "--- amd-smi after teardown"; amd-smi metric -m | grep -E "VRAM|GTT"; } >> $L
log "=== BB $N DONE"
