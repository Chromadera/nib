cd /workspace/nib; M=/workspace/models
for q in All 23; do for kv in q8_0 q4_1; do L=/workspace/logs/selftest_${q}_${kv}_mtp.log
timeout 600 build/nib selftest -m $M/Inkling-2-Mini-NVFP4-$q.gguf --cpu-embd --kv-type $kv --sparse-pages 224 --mtp $M/mtp.safetensors > $L 2>&1; echo "NVFP4-$q kv=$kv MTP rc=$?"; grep -E "SELFTEST|selftest\]|dspark|rror" $L | grep -v sampler | tail -12; done; done
echo DONE_MTP_ST
