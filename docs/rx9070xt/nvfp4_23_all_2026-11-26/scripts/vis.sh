#!/usr/bin/env bash
# vis.sh TAG Q KV CTX SLOTS  -> vision escalation (card,1080p,1440p,4k,max2560) + text before/after
. /workspace/lib.sh
TAG=$1 Q=$2 KV=$3 CTX=$4 SL=$5
[ -f /workspace/img_1080p.png ] || $PY $CL genpng
kill_nib; B=$(smi); LF=$LOGD/vis_${TAG}.log; PF=$LOGD/vis_${TAG}.poll
start_poll $PF
if serve_up $LF -m $M/Inkling-2-Mini-NVFP4-$Q.gguf --cpu-embd --slots $SL -c $CTX --kv-type $KV --sparse-pages 224 --mmproj $MMPROJ; then
  sleep 2; A=$(smi)
  timeout 900 $PY $CL vision $TAG card 1080p 1440p 4k max2560 > $LF.client 2>&1
  alive=$(kill -0 $SRV 2>/dev/null && echo alive || echo DEAD)
else A="-"; alive=START_FAILED; fi
sleep 1; stop_poll; E=$(smi); kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null
echo "VIS $TAG q=$Q kv=$KV ctx=$CTX slots=$SL server=$alive idle=[$B] ready=[$A] end=[$E] $(poll_stats $PF) | $(grep -m1 'Device VRAM' $LF)" | tee -a $LOGD/vis_results.txt
cat $LF.client 2>/dev/null | sed 's/"text": "[^"]*"/"text":"…"/' | tee -a $LOGD/vis_results.txt
grep -Ei "error|HTTP 400|too large|workspace" $LF | tail -5 | tee -a $LOGD/vis_results.txt
