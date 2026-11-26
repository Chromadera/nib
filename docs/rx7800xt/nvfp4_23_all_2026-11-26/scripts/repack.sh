#!/usr/bin/env bash
cd /workspace/models; L=/workspace/logs
ls -l Inkling-2-Mini-Q8_0.gguf*; tail -3 $L/dl_q8.log
{ time /workspace/repack_nvfp4 Inkling-2-Mini-Q8_0.gguf Inkling-2-Mini-NVFP4-23.gguf 22; echo "rc=$?"; } > $L/repack_23.log 2>&1
{ time /workspace/repack_nvfp4 Inkling-2-Mini-Q8_0.gguf Inkling-2-Mini-NVFP4-All.gguf 23; echo "rc=$?"; } > $L/repack_all.log 2>&1
S23=$(stat -c%s Inkling-2-Mini-NVFP4-23.gguf); SALL=$(stat -c%s Inkling-2-Mini-NVFP4-All.gguf)
{ echo "NVFP4-23=$S23 want 16099143008 $([ "$S23" = 16099143008 ] && echo OK || echo MISMATCH)"
  echo "NVFP4-All=$SALL want 15589536192 $([ "$SALL" = 15589536192 ] && echo OK || echo MISMATCH)"
  sha256sum Inkling-2-Mini-Q8_0.gguf; } > $L/sizes.txt
if [ "$S23" = 16099143008 ] && [ "$SALL" = 15589536192 ]; then rm -f Inkling-2-Mini-Q8_0.gguf; echo "deleted Q8_0" >> $L/sizes.txt; fi
sha256sum Inkling-2-Mini-NVFP4-23.gguf Inkling-2-Mini-NVFP4-All.gguf mtp.safetensors mmproj-Inkling-2-Mini-Q8_0.gguf > $L/sha256.txt
echo REPACK_DONE >> $L/sizes.txt
