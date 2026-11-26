#!/usr/bin/env bash
# search.sh NAME Q KV SLOTS MTP VIS LO HI STEP [extra]  -> bisects max clean ctx
. /workspace/lib.sh
N=$1 Q=$2 KV=$3 SL=$4 MT=$5 VI=$6 LO=$7 HI=$8 STEP=$9; shift 9
R=$LOGD/stage2_results.txt
ok() { # run probe at ctx $1; return 0 iff PASS and no spill and decode healthy
  /workspace/probe.sh ${N}_c$1 $Q $KV $1 $SL $MT $VI "$@" >/dev/null 2>&1
  local L; L=$(grep "| ${N}_c$1 |" $R | tail -1)
  local idle=$(sed -n 's/.*idle\[usedVRAM freeVRAM GTT\]=[0-9]* [0-9]* \([0-9]*\).*/\1/p' <<<"$L")
  local pg=$(sed -n 's/.*peakGTT=\([0-9]*\).*/\1/p' <<<"$L")
  local tps=$(sed -n 's/.*solo_tps=\([0-9.]*\).*/\1/p' <<<"$L")
  local spill=$(( ${pg:-0} - ${idle:-0} - 514 ))
  echo "[search $N] ctx=$1 -> ${L%% |*} gtt_over_cpuembd=${spill}MB tps=${tps:-}" | tee -a $LOGD/stage2_search.txt
  [[ "$L" == PASS* ]] || return 1
  [ $spill -le 16 ] || return 1
  awk -v t="${tps:-0}" 'BEGIN{exit !(t>=60)}' || return 1
}
shift 0
EXTRA=("$@")
if ! ok $LO "${EXTRA[@]}"; then
  echo "MAX | $N | none (even ctx=$LO fails)" | tee -a $LOGD/stage2_max.txt; exit 0; fi
if ok $HI "${EXTRA[@]}"; then echo "MAX | $N | >= $HI (upper bound passes)" | tee -a $LOGD/stage2_max.txt; exit 0; fi
while [ $((HI-LO)) -gt $STEP ]; do
  MID=$(( (LO+HI)/2 / STEP * STEP )); [ $MID -le $LO ] && MID=$((LO+STEP))
  if ok $MID "${EXTRA[@]}"; then LO=$MID; else HI=$MID; fi
done
echo "MAX | $N | $LO (first fail $HI)" | tee -a $LOGD/stage2_max.txt
