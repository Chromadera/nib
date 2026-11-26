#!/usr/bin/env bash
# Vision matrix: CONFIGS="kv:ctx:slots:mtp ..." -> vis_client.py (text, 1080p/1440p/4K/2560sq, fill ctx-8k + 1080p, after-fill 4K)
source /workspace/common.sh
OUT=$R/vision; mkdir -p $OUT
for spec in $CONFIGS; do
  IFS=: read kv c s mtp <<<"$spec"; tag=${kv}_${c}_s${s}_mtp${mtp}
  pkill -9 -x nib; wait_idle 19500
  echo "=== $tag $(ts) pre: $(smi)" | tee -a $OUT/summary.txt
  a=(-m $M --mmproj $MMP --slots $s -c $c --kv-type $kv --sparse-pages 224 --cpu-embd --host 127.0.0.1 --port 4444)
  [ $mtp = 1 ] && a+=(--mtp $MTPF)
  $NIB serve "${a[@]}" > $OUT/serve_$tag.log 2>&1 & pid=$! ok=0
  for i in $(seq 1 240); do kill -0 $pid 2>/dev/null || break; curl -sf -m 3 127.0.0.1:4444/health >/dev/null && { ok=1; break; }; sleep 1; done
  if [ $ok != 1 ]; then echo "$tag START_FAIL"; tail -5 $OUT/serve_$tag.log | tee -a $OUT/summary.txt; kill -9 $pid; continue; fi
  echo "boot: $(smi) | $(grep -m1 'Device VRAM' $OUT/serve_$tag.log)" | tee -a $OUT/summary.txt
  timeout 3600 python3 /workspace/vis_client.py $tag $c | tee -a $OUT/results.jsonl
  echo "end $(ts): $(smi)" | tee -a $OUT/summary.txt
  grep -E 'hipMalloc|GTT|WARNING' $OUT/serve_$tag.log | head -8 >> $OUT/summary.txt
  kill -9 $pid; wait $pid 2>/dev/null
done
echo "VISION DONE $(ts)" | tee -a $OUT/summary.txt
