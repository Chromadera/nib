#!/usr/bin/env bash
# bench.sh LABEL Q KV CTX DEPTHS  (non-MTP nib bench pp512/tg128 r2 sparse224 cpu-embd)
. /workspace/lib.sh
LB=$1 Q=$2 KV=$3 CTX=$4 D=$5
kill_nib; B=$(smi); LF=$LOGD/bench_${LB}.log; PF=$LOGD/bench_${LB}.poll
echo "=== amd-smi static (abridged) ===" > $LF; amd-smi static 2>&1 | grep -E "MARKET_NAME|NUM_COMPUTE_UNITS|TARGET_GRAPHICS|SIZE:|DRIVER|VERSION:" >> $LF
echo "=== idle [used free gtt] $B ===" >> $LF
start_poll $PF
timeout 3000 $NIB bench -m $M/Inkling-2-Mini-NVFP4-$Q.gguf --cpu-embd -c $CTX -d $D -p 512 -n 128 -r 2 --kv-type $KV --sparse-pages 224 >> $LF 2>&1; RC=$?
stop_poll; E=$(smi)
echo "=== end [used free gtt] $E | $(poll_stats $PF) | rc=$RC ===" >> $LF
echo "BENCH $LB rc=$RC idle=[$B] $(poll_stats $PF) | $(grep -m1 total_vram= $LF | sed 's/.*total_vram=//') | $(grep -m1 -o 'static decode HIP Graph: [0-9]* kernel nodes' $LF)" | tee -a $LOGD/bench_results.txt
grep -E "^\|" $LF | tee -a $LOGD/bench_results.txt
