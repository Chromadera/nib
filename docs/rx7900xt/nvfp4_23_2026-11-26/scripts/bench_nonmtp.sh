#!/usr/bin/env bash
# Non-MTP nib bench depth sweep: SPECS="kv:ctx ..." ; pp512 + tg128, -r 2, sparse224, --cpu-embd, slots 1.
source /workspace/common.sh
OUT=$R/nonmtp; mkdir -p $OUT
for spec in $SPECS; do
  kv=${spec%%:*}; c=${spec##*:}; tag=${kv}_$c
  d=""; for x in 0 8192 32768 65536 131072 196608 229376 256000; do [ $x -le $((c-1024)) ] && d="$d,$x"; done; d=${d#,}
  pkill -9 -x nib; wait_idle 19500
  echo "=== $tag depths=$d $(ts) pre: $(smi)" | tee -a $OUT/summary.txt
  smi_start /tmp/smi_nm.txt
  timeout 3000 $NIB bench -m $M -c $c -d $d -p 512 -n 128 -r 2 --kv-type $kv --sparse-pages 224 --cpu-embd > $OUT/bench_$tag.txt 2>&1; rc=$?
  sm=$(smi_stop)
  echo "rc=$rc $(ts) during: $sm post: $(smi)" | tee -a $OUT/summary.txt
  grep -E 'total_vram|^\|' $OUT/bench_$tag.txt | tee -a $OUT/summary.txt
done
echo "NONMTP DONE $(ts)" | tee -a $OUT/summary.txt
