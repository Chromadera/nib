#!/usr/bin/env bash
# bb.sh LABEL Q KV CTX SLOTS  -> full BetterBench 0.6.1 (single-stream 8 cat x5 + prefill ladder + concurrency) on localhost
. /workspace/lib.sh
LB=$1 Q=$2 KV=$3 CTX=$4 SL=$5
OUTD=/workspace/bb; mkdir -p $OUTD
CFGF=$OUTD/${LB}_config.json
$PY - "$CTX" "$SL" "$CFGF" <<'PY'
import json, sys
ctx, sl, out = int(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
base = json.load(open("/workspace/betterbench_slots8_262k_config.json"))
lad = [d for d in [2560, 10240, 20480, 40960, 81920, 163840, 334000] if d + 16 + 256 <= ctx]
# BetterBench target depth -> real prompt tokens ratio measured at ~0.784 (48128 -> 37746 on run 1);
# add a top target whose REAL prompt lands ~2% under ctx, so the ladder truly reaches max context.
top = int((ctx * 0.975) / 0.80) // 512 * 512
lad = [d for d in lad if d * 0.80 < ctx * 0.9]
for extra in [int(ctx * 0.6 / 0.80) // 512 * 512, top]:
    if extra > lad[-1] + 2048: lad.append(extra)
base["prefill_depths"] = lad
base["concurrency_levels"] = [c for c in [1, 2, 4, 8] if c <= sl]
json.dump(base, open(out, "w"), indent=2); print(json.dumps(base))
PY
kill_nib; B=$(smi); LF=$LOGD/bb_${LB}_serve.log; PF=$LOGD/bb_${LB}.poll
{ echo "=== amd-smi static ==="; amd-smi static 2>&1 | grep -E "MARKET_NAME|NUM_COMPUTE_UNITS|TARGET_GRAPHICS|SIZE:|VERSION:"; echo "=== idle metric ==="; amd-smi metric -m; } > $LOGD/bb_${LB}_smi_before.txt 2>&1
start_poll $PF
serve_up $LF -m $M/Inkling-2-Mini-NVFP4-$Q.gguf --cpu-embd --slots $SL -c $CTX --kv-type $KV --sparse-pages 224 --no-vision || { echo "BB $LB SERVE_FAILED"; stop_poll; tail -20 $LF; exit 1; }
sleep 2; amd-smi metric -m > $LOGD/bb_${LB}_smi_ready.txt 2>&1
curl -s http://127.0.0.1:4444/health > $LOGD/bb_${LB}_health.json
T0=$(date +%s)
BETTERBENCH_NO_UPDATE_CHECK=1 timeout 14000 /workspace/bbvenv/bin/betterbench run --no-update-check \
  --endpoint http://127.0.0.1:4444/v1 --model inkling-2-mini --config $CFGF --greedy \
  --max-model-len $(( $(python3 -c "import json;print(max(json.load(open(\"$CFGF\"))[\"prefill_depths\"]))") + 512 )) --name rx9070xt-$LB \
  --note engine=nib --note nib_version=0.1.2-local --note gpu=RX_9070_XT_16GB --note arch=gfx1201 \
  --note quant=NVFP4-$Q --note slots=$SL --note ctx=$CTX --note kv=$KV --note sparse_pages=224 \
  --note served_ctx=$CTX --note max_model_len_note=inflated_so_top_ladder_target_reaches_true_ctx --note mtp=none --note vision=none --note cpu_embd=on --note host=windows_wsl2 \
  --out $OUTD/betterbench_${LB}.json --html-out $OUTD/betterbench_${LB}.html > $LOGD/bb_${LB}_client.log 2>&1
RC=$?; T1=$(date +%s)
# post-run solo decode check: does any GTT growth during the run cost single-stream decode?
{ echo "post-run smi: $(smi)"; timeout 300 $PY $CL probe post-$LB 1; echo "post-check smi: $(smi)"; } > $LOGD/bb_${LB}_postcheck.txt 2>&1
alive=$(kill -0 $SRV 2>/dev/null && echo alive || echo DEAD)
amd-smi metric -m > $LOGD/bb_${LB}_smi_after.txt 2>&1
stop_poll
echo "BB $LB rc=$RC server=$alive dur=$((T1-T0))s idle=[$B] $(poll_stats $PF) batch_graphs=$(grep -c 'Captured static batch' $LF) | $(grep -m1 'Device VRAM' $LF)" | tee -a $LOGD/bb_results.txt
kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null
