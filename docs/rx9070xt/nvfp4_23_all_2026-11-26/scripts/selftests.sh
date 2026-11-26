#!/usr/bin/env bash
cd /workspace/nib; M=/workspace/models
for q in All 23; do for kv in q8_0 q4_1 q4_0; do
  L=/workspace/logs/selftest_${q}_${kv}.log
  timeout 400 build/nib selftest -m $M/Inkling-2-Mini-NVFP4-$q.gguf --cpu-embd --kv-type $kv --sparse-pages 224 > $L 2>&1; rc=$?
  echo "NVFP4-$q kv=$kv rc=$rc | $(grep -E 'SELFTEST\]' $L) | $(grep -c -- '-> PASS' $L) sampler PASS | $(grep -m1 total_vram= $L | sed 's/.*total_vram=//')"
done; done
# MTP selftest on All
L=/workspace/logs/selftest_All_q4_1_mtp.log
timeout 500 build/nib selftest -m $M/Inkling-2-Mini-NVFP4-All.gguf --cpu-embd --kv-type q4_1 --sparse-pages 224 --mtp $M/mtp.safetensors > $L 2>&1; echo "NVFP4-All mtp rc=$?"; grep -E "selftest|SELFTEST" $L | tail -20
L=/workspace/logs/selftest_23_q4_1_mtp.log
timeout 500 build/nib selftest -m $M/Inkling-2-Mini-NVFP4-23.gguf --cpu-embd --kv-type q4_1 --sparse-pages 224 --mtp $M/mtp.safetensors > $L 2>&1; echo "NVFP4-23 mtp rc=$?"; grep -E "selftest|SELFTEST|rror|fail" $L | tail -20
echo SELFTESTS_DONE
