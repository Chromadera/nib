source /workspace/common.sh
echo "pre: $(smi)"
timeout 600 $NIB selftest -m $M --kv-type q4_1 --sparse-pages 224 --mtp $MTPF --cpu-embd > $R/selftest_mtp_q4_1.log 2>&1; echo rc=$?
echo "post: $(smi)"
grep -E 'selftest|PASS|FAIL|total_vram|gguf\] (loaded|total)|tok_embd' $R/selftest_mtp_q4_1.log | tail -45
