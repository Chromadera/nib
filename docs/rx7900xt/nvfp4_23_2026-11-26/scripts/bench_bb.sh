#!/usr/bin/env bash
# Full BetterBench 0.6.1 (5 runs/cat, c=1,2,4,8 x 32 reqs, full prefill ladder) on localhost.
# Env: S (slots) C (ctx) KV NAME
source /workspace/common.sh
OUT=$R/betterbench; mkdir -p $OUT
BB=/workspace/bbenv/bin/betterbench
tag=${NAME:-bb_s${S}_${C}_${KV}_mtp}
python3 - $C > $OUT/${tag}_config.json <<'PY'
import json,sys
c=int(sys.argv[1]); lad=[d for d in [2560,10240,20480,40960,81920,163840,261000] if d+16+256 <= c]
print(json.dumps({"temperature":0.0,"top_p":0.95,"top_k":20,"greedy":True,"warmup":1,"runs_per_category":5,
 "unique_nonce":True,"run_single_stream":True,"run_concurrency":True,"concurrency_levels":[1,2,4,8],
 "concurrency_requests":32,"run_prefill":True,"prefill_depths":lad,"prefill_runs":2,"prefill_warmup":1,
 "prefill_max_tokens":16,"prefill_ctx_margin":256,"timeout_s":900.0},indent=1))
PY
cat $OUT/${tag}_config.json
pkill -9 -x nib; wait_idle 19500
echo "idle $(ts): $(smi)" | tee $OUT/${tag}_smi.txt
amd-smi static > $OUT/${tag}_amdsmi_static.txt 2>&1
$NIB serve -m $M --mtp $MTPF --no-vision --slots $S -c $C --kv-type $KV --sparse-pages 224 --cpu-embd --host 127.0.0.1 --port 4444 > $OUT/${tag}_serve.log 2>&1 &
pid=$!
for i in $(seq 1 240); do kill -0 $pid || { echo START_FAILED; tail $OUT/${tag}_serve.log; exit 1; }; curl -sf 127.0.0.1:4444/health > $OUT/${tag}_health_boot.json && break; sleep 1; done
echo "boot $(ts): $(smi)" | tee -a $OUT/${tag}_smi.txt
smi_start $OUT/${tag}_smi_samples.txt
$BB run --no-update-check --endpoint http://127.0.0.1:4444/v1 --model inkling-2-mini --config $OUT/${tag}_config.json \
  --greedy --max-model-len $C --name $tag \
  --note engine=nib --note gpu=RX_7900_XT --note quant=NVFP4-23 --note slots=$S --note ctx=$C --note kv=$KV \
  --note sparse_pages=224 --note mtp=dspark --note vision=none --note cpu_embd=on --note host=windows_wsl2 \
  --out $OUT/$tag.json --html-out $OUT/$tag.html > $OUT/${tag}_bb.log 2>&1; rc=$?
sm=$(smi_stop)
echo "bb rc=$rc $(ts) during: $sm post: $(smi)" | tee -a $OUT/${tag}_smi.txt
curl -sf 127.0.0.1:4444/health > $OUT/${tag}_health_end.json
echo -n "static batch graphs: "; grep -c "Captured static batch" $OUT/${tag}_serve.log
kill -9 $pid; echo "BB DONE $(ts)"
