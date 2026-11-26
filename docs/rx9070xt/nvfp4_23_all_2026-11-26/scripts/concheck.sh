#!/usr/bin/env bash
# concheck.sh: re-run BetterBench concurrency-only on All s4 q4_0 49152, then solo decode post-check (GTT growth impact)
. /workspace/lib.sh
LB=all_s4_q40_48k_concheck; kill_nib; B=$(smi); LF=$LOGD/${LB}_serve.log; PF=$LOGD/${LB}.poll
start_poll $PF
serve_up $LF -m $M/Inkling-2-Mini-NVFP4-All.gguf --cpu-embd --slots 4 -c 49152 --kv-type q4_0 --sparse-pages 224 --no-vision || { echo FAIL; exit 1; }
sleep 2; echo "ready smi: $(smi)" > $LOGD/${LB}.txt
timeout 300 $PY $CL probe pre-$LB 1 >> $LOGD/${LB}.txt 2>&1; echo "after pre solo smi: $(smi)" >> $LOGD/${LB}.txt
BETTERBENCH_NO_UPDATE_CHECK=1 timeout 3000 /workspace/bbvenv/bin/betterbench run --no-update-check --endpoint http://127.0.0.1:4444/v1 --model inkling-2-mini \
  --config /workspace/bb/all_s4_q40_48k_config.json --greedy --concurrency --max-model-len 49152 --name $LB \
  --out /workspace/bb/$LB.json --html-out /workspace/bb/$LB.html > $LOGD/${LB}_client.log 2>&1
echo "after concurrency smi: $(smi)" >> $LOGD/${LB}.txt
timeout 300 $PY $CL probe post-$LB 1 >> $LOGD/${LB}.txt 2>&1; echo "after post solo smi: $(smi)" >> $LOGD/${LB}.txt
stop_poll; echo "$(poll_stats $PF) batch_graphs=$(grep -c 'Captured static batch' $LF)" >> $LOGD/${LB}.txt
kill -9 $SRV; cat $LOGD/${LB}.txt
