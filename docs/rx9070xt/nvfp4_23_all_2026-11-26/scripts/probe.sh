#!/usr/bin/env bash
# probe.sh TAG QUANT KV CTX SLOTS MTP(0/1) VIS(0/1) [extra serve args...]
# -> appends one RESULT line to $LOGD/stage2_results.txt
. /workspace/lib.sh
TAG=$1 Q=$2 KV=$3 CTX=$4 SL=$5 MT=$6 VI=$7; shift 7
kill_nib
B=$(smi)
ARGS=(-m $M/Inkling-2-Mini-NVFP4-$Q.gguf --cpu-embd --slots $SL -c $CTX --kv-type $KV --sparse-pages 224)
[ "$MT" = 1 ] && ARGS+=(--mtp $MTPF)
if [ "$VI" = 1 ]; then ARGS+=(--mmproj $MMPROJ); else ARGS+=(--no-vision); fi
ARGS+=("$@")
LF=$LOGD/s2_${TAG}.log; PF=$LOGD/s2_${TAG}.poll
start_poll $PF
T0=$(date +%s); serve_up $LF "${ARGS[@]}"; RC=$?; T1=$(date +%s)
OUT=""; ST=FAIL
if [ $RC = 0 ]; then
  sleep 2; A=$(smi)
  OUT=$(timeout 300 $PY $CL probe $TAG $SL 2>&1)
  SOLO=$(grep '^SOLO' <<<"$OUT" | sed -n 's/.*"decode_tps": \([0-9.]*\).*/\1/p')
  BOK=$(grep '^BURST' <<<"$OUT" | sed -n 's/.*"ok": \([0-9]*\).*/\1/p')
  AGG=$(grep '^BURST' <<<"$OUT" | sed -n 's/.*"agg_tps": \([0-9.]*\).*/\1/p')
  POK=$(grep -c '^POST {"ok": true' <<<"$OUT")
  kill -0 $SRV 2>/dev/null && [ -n "$SOLO" ] && [ "$POK" = 1 ] && { [ "$SL" = 1 ] || [ "$BOK" = "$SL" ]; } && ST=PASS
else
  A="-"; ST=$([ $RC = 2 ] && echo TIMEOUT || echo FAIL)
fi
sleep 1; stop_poll; E=$(smi)
kill -9 $SRV 2>/dev/null; wait $SRV 2>/dev/null
CFG=$(grep -m1 "\[nib\] Config:" $LF | sed 's/.*vram=//'); DEV=$(grep -m1 "Device VRAM" $LF | sed 's/\[nib\] Device VRAM: //')
ENG=$(grep -m1 "total_vram=" $LF | sed 's/.*total_vram=//')
ERR=$(grep -Ei "hipMalloc|out of memory|failed|error|WARNING: Only|abort" $LF | tail -n 2 | tr '\n' ' ' | cut -c1-240)
GN=$(grep -m1 -o "static decode HIP Graph: [0-9]* kernel nodes" $LF)
L="$ST | $TAG | q=$Q kv=$KV ctx=$CTX slots=$SL mtp=$MT vis=$VI | boot=$((T1-T0))s | idle[usedVRAM freeVRAM GTT]=$B | ready=$A | end=$E | $(poll_stats $PF) | eng=$ENG | dev=$DEV | solo_tps=${SOLO:-} burst_ok=${BOK:-} agg=${AGG:-} | graph=$GN | err=$ERR"
echo "$L" | tee -a $LOGD/stage2_results.txt
echo "$OUT" >> $LF.client
