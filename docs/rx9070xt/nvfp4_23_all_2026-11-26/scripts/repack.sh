#!/usr/bin/env bash
set -x
cd /workspace
g++ -O3 -march=native -fopenmp /workspace/repack_nvfp4.cpp -o /workspace/repack_nvfp4.bin || exit 1
M=/workspace/models
sha256sum $M/Inkling-2-Mini-Q8_0.gguf; stat -c%s $M/Inkling-2-Mini-Q8_0.gguf
time /workspace/repack_nvfp4.bin $M/Inkling-2-Mini-Q8_0.gguf $M/Inkling-2-Mini-NVFP4-23.gguf 22 | tail -n 20
time /workspace/repack_nvfp4.bin $M/Inkling-2-Mini-Q8_0.gguf $M/Inkling-2-Mini-NVFP4-All.gguf 23 | tail -n 20
S23=$(stat -c%s $M/Inkling-2-Mini-NVFP4-23.gguf); SA=$(stat -c%s $M/Inkling-2-Mini-NVFP4-All.gguf)
echo "SIZE23=$S23 want 16099143008"; echo "SIZEALL=$SA want 15589536192"
if [ "$S23" = 16099143008 ] && [ "$SA" = 15589536192 ]; then echo SIZES_OK; rm -f $M/Inkling-2-Mini-Q8_0.gguf; echo Q8_DELETED; else echo SIZES_MISMATCH; fi
sha256sum $M/Inkling-2-Mini-NVFP4-23.gguf $M/Inkling-2-Mini-NVFP4-All.gguf $M/mtp.safetensors $M/mmproj-Inkling-2-Mini-Q8_0.gguf
echo REPACK_DONE
