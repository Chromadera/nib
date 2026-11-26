#!/usr/bin/env bash
cd /workspace/nib
/opt/rocm/bin/hipcc --version
time cmake -B build -DCMAKE_BUILD_TYPE=Release -DGPU_TARGET=gfx1201 -DROCM_PATH=/opt/rocm -DCMAKE_EXE_LINKER_FLAGS="-no-pie"
time cmake --build build -j"$(nproc)"
ls -la build/nib
g++ -O3 -march=native -fopenmp /workspace/repack_nvfp4.cpp -o /workspace/repack_nvfp4 && echo REPACK_BUILT
/opt/rocm/bin/hipcc -O2 --offload-arch=gfx1201 -no-pie /workspace/vram_probe.cpp -o /workspace/vram_probe && echo PROBE_BUILT
echo BUILD_DONE
