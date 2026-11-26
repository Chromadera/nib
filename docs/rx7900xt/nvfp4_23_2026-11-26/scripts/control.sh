#!/usr/bin/env bash
# Same-node, same-build control: NVFP4-Mixed vs NVFP4-23.
source /workspace/common.sh
MIX=/workspace/models/Inkling-2-Mini-NVFP4-Mixed.gguf; N23=/workspace/models/Inkling-2-Mini-NVFP4-23.gguf
OUT=$R/control; mkdir -p $OUT
until grep -q 'got=' /workspace/dl_mixed.log 2>/dev/null; do sleep 10; done; cat /workspace/dl_mixed.log | tail -1 | tee $OUT/mixed_sha.txt
echo "CONTROL START $(ts): $(smi)" | tee $OUT/summary.txt
# 1) ceiling probes on Mixed for the headline configs
M_OVERRIDE=$MIX CEIL_OUT=$OUT/ceiling_mixed CONFIGS="q4_1:1:0:0 q4_0:1:0:0 q8_0:1:0:0 q4_1:1:1:0 q4_0:1:1:0 q8_0:1:1:0 q4_1:4:1:0 q4_1:8:1:0 q4_1:1:1:1 q4_1:4:1:1" bash /workspace/ceiling.sh > $OUT/ceiling_mixed.nohup 2>&1
# 2) matched nib bench + gen --mtp, both models, q4_1 @ 131072 (fits both)
for tag in mixed:$MIX n23:$N23; do
  n=${tag%%:*}; m=${tag#*:}
  pkill -9 -x nib; wait_idle 19500
  smi_start /tmp/smi_c.txt
  timeout 1500 $NIB bench -m $m -c 131072 -d 0,32768,65536,128000 -p 512 -n 128 -r 2 --kv-type q4_1 --sparse-pages 224 --cpu-embd > $OUT/bench_${n}_q4_1_131k.txt 2>&1
  echo "$n bench rc=$? $(smi_stop)" | tee -a $OUT/summary.txt; grep -E 'total_vram|^\|' $OUT/bench_${n}_q4_1_131k.txt | tee -a $OUT/summary.txt
  for D in 2k 64k 128k; do
    pkill -9 -x nib; wait_idle 19500; smi_start /tmp/smi_c.txt
    timeout 900 $NIB gen -m $m --mtp $MTPF --prompt-file /tmp/prompt_$D.txt -n 256 -c 131072 --kv-type q4_1 --sparse-pages 224 --cpu-embd > $OUT/gen_${n}_q4_1_131k_$D.log 2>&1
    echo "$n gen $D rc=$? $(grep -m1 '\[gen\] prefill' $OUT/gen_${n}_q4_1_131k_$D.log) | $(grep -E '^\[mtp\] (256|per)' $OUT/gen_${n}_q4_1_131k_$D.log | tr '\n' ' ') | $(smi_stop)" | tee -a $OUT/summary.txt
  done
done
echo "CONTROL DONE $(ts)" | tee -a $OUT/summary.txt
