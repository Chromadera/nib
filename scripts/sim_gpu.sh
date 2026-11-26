#!/usr/bin/env bash
# sim_gpu.sh — Simulate other AMD gfx1100 (Navi 31) GPUs on a host RX 7900 XTX.
#
# Combines:
#   1. Hardware CU masking via HSA_CU_MASK (physically runs only the target CU count)
#   2. LD_PRELOAD HIP shim (overrides WGP/CU count, device name, hipMemGetInfo,
#      and enforces a hard hipMalloc VRAM ceiling for <=24 GiB cards)
#   3. Peak VRAM fit verdict + memory-bandwidth decode scaling report
#
# Usage:
#   ./scripts/sim_gpu.sh --list
#   ./scripts/sim_gpu.sh <profile> [--reserve <MiB>] [--] <command...>
#   ./scripts/sim_gpu.sh --custom <CUs>:<VRAM_GiB>:<BW_GBs> [--] <command...>
#
# Examples:
#   ./scripts/sim_gpu.sh 7900xt ./build/nib bench -m models/Inkling-2-Mini-NVFP4-Mixed.gguf -c 262848 -d 0,65536 -p 512 -n 64 --kv-type q4_1 --sparse-pages 224
#   ./scripts/sim_gpu.sh w7800  ./build/nib selftest -m models/Inkling-2-Mini-NVFP4-Mixed.gguf --kv-type q8_0 --sparse-pages 224 --mtp models/mtp.safetensors
#   ./scripts/sim_gpu.sh 7900gre --reserve 0 ./build/nib bench ...

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
SHIM_SRC="${SCRIPT_DIR}/sim_gpu_shim.c"
SHIM_SO="${ROOT_DIR}/build/libsim_gpu.so"

print_list() {
    cat <<'EOF'
Supported AMD RDNA3 / RDNA4 GPU profiles:

  Profile     GPU Name                  Arch     CUs  WGPs   VRAM    BW       Recommended Nib Config (Inkling-2-Mini NVFP4, 16.34 GiB weights)
  ----------  ------------------------  -------  ---  ----  ------  --------  -------------------------------------------------------------------
  w7900       Radeon PRO W7900          gfx1100   96    48  48 GiB  864 GB/s  -c 262848 --slots 4 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj (plenty of headroom, can even run 8+ slots or dense!)
  w7800-48    Radeon PRO W7800 48GB     gfx1100   70    35  48 GiB  576 GB/s  -c 262848 --slots 4 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj
  w7800       Radeon PRO W7800 32GB     gfx1100   70    35  32 GiB  576 GB/s  -c 262848 --slots 4 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj (~21.9 GiB used, 10 GiB free)
  7900xtx     Radeon RX 7900 XTX        gfx1100   96    48  24 GiB  960 GB/s  -c 262848 --slots 1..4 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj (~21.5-21.9 GiB)
  7900xt      Radeon RX 7900 XT         gfx1100   84    42  20 GiB  800 GB/s  -c 262848 --kv-type q4_1 --sparse-pages 224 (~18.8 GiB) OR -c 131072 --kv-type q8_0 --sparse-pages 224 --mtp (~19.4 GiB)
  7900gre     Radeon RX 7900 GRE        gfx1100   80    40  16 GiB  576 GB/s  Weights (16.34 GiB) exceed 16 GiB VRAM — needs smaller quant or host offload
  7800xt      Radeon RX 7800 XT (ref)   gfx1101   60    30  16 GiB  624 GB/s  CU/VRAM stress test only (requires gfx1101 build + smaller quant)
  r9700       Radeon AI PRO R9700       gfx1201   64    32  32 GiB  640 GB/s  -c 262848 --slots 4 --kv-type q8_0 --sparse-pages 224 --mtp --mmproj (build with -DGPU_TARGET=gfx1201 for native RDNA4)
  9070xt      Radeon RX 9070 XT         gfx1201   64    32  16 GiB  640 GB/s  Weights (16.34 GiB) exceed 16 GiB VRAM — native gfx1201 kernel tests / smaller quant
  9070        Radeon RX 9070            gfx1201   56    28  16 GiB  640 GB/s  Weights (16.34 GiB) exceed 16 GiB VRAM — native gfx1201 kernel tests / smaller quant
  9060xt      Radeon RX 9060 XT 16GB    gfx1200   32    16  16 GiB  320 GB/s  CU/WGP co-residency stress test (build with -DGPU_TARGET=gfx1200 for native RDNA4)

Notes:
  * Host XTX has 24 GiB physical VRAM: profiles >24 GiB (w7900, w7800, w7800-48, r9700) accurately simulate
    CU masking, WGP count, and bandwidth scaling, while physical allocations still cannot exceed 24 GiB.
  * Profiles <=24 GiB (7900xt, 7900gre, 7800xt, 9070xt, 9070, 9060xt) enforce a real hipMalloc OOM cap (default reserve: 350 MiB).
  * RDNA4 profiles simulate WGP count, VRAM, and bandwidth on host RDNA3 hardware; for real RDNA4 ISA execution, build with -DGPU_TARGET=gfx1201 (or gfx1200).
  * Override desktop reserve with --reserve <MiB> (use 0 for headless server simulation).
EOF
}

if [[ $# -lt 1 || "$1" == "-h" || "$1" == "--help" ]]; then
    print_list
    exit 0
fi

if [[ "$1" == "--list" ]]; then
    print_list
    exit 0
fi

PROFILE="$1"
shift

RESERVE_MIB=350
CUSTOM_SPEC=""
if [[ "$PROFILE" == "--custom" ]]; then
    CUSTOM_SPEC="${1:-}"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reserve)
            RESERVE_MIB="${2:?missing MiB value}"
            shift 2
            ;;
        --)
            shift
            break
            ;;
        *)
            break
            ;;
    esac
done

if [[ $# -lt 1 ]]; then
    echo "Error: missing command to run." >&2
    echo "Example: $0 7900xt ./build/nib selftest -m models/Inkling-2-Mini-NVFP4-Mixed.gguf --kv-type q4_1 --sparse-pages 224" >&2
    exit 2
fi

GPU_NAME=""
CUS=96
VRAM_GIB=24
BW_GBS=960

if [[ -n "$CUSTOM_SPEC" ]]; then
    IFS=':' read -r CUS VRAM_GIB BW_GBS <<< "$CUSTOM_SPEC"
    GPU_NAME="Custom gfx1100 (${CUS}CU, ${VRAM_GIB}GiB, ${BW_GBS}GB/s)"
else
    case "${PROFILE,,}" in
        w7900|w7900-48)
            GPU_NAME="AMD Radeon PRO W7900 (sim)"
            CUS=96; VRAM_GIB=48; BW_GBS=864
            ;;
        w7800-48)
            GPU_NAME="AMD Radeon PRO W7800 48GB (sim)"
            CUS=70; VRAM_GIB=48; BW_GBS=576
            ;;
        w7800|w7800-32)
            GPU_NAME="AMD Radeon PRO W7800 32GB (sim)"
            CUS=70; VRAM_GIB=32; BW_GBS=576
            ;;
        7900xtx|xtx)
            GPU_NAME="AMD Radeon RX 7900 XTX (sim)"
            CUS=96; VRAM_GIB=24; BW_GBS=960
            ;;
        7900xt|xt)
            GPU_NAME="AMD Radeon RX 7900 XT (sim)"
            CUS=84; VRAM_GIB=20; BW_GBS=800
            ;;
        7900gre|gre)
            GPU_NAME="AMD Radeon RX 7900 GRE (sim)"
            CUS=80; VRAM_GIB=16; BW_GBS=576
            ;;
        7800xt)
            GPU_NAME="AMD Radeon RX 7800 XT (sim)"
            CUS=60; VRAM_GIB=16; BW_GBS=624
            ;;
        r9700|pro-r9700)
            GPU_NAME="AMD Radeon AI PRO R9700 (sim)"
            CUS=64; VRAM_GIB=32; BW_GBS=640
            ;;
        9070xt)
            GPU_NAME="AMD Radeon RX 9070 XT (sim)"
            CUS=64; VRAM_GIB=16; BW_GBS=640
            ;;
        9070)
            GPU_NAME="AMD Radeon RX 9070 (sim)"
            CUS=56; VRAM_GIB=16; BW_GBS=640
            ;;
        9060xt)
            GPU_NAME="AMD Radeon RX 9060 XT (sim)"
            CUS=32; VRAM_GIB=16; BW_GBS=320
            ;;
        *)
            echo "Unknown GPU profile: '$PROFILE'. Run '$0 --list' to see available profiles." >&2
            exit 2
            ;;
    esac
fi

WGPS=$((CUS / 2))
VRAM_MIB=$((VRAM_GIB * 1024))

# Build libsim_gpu.so if missing or out of date
if [[ ! -f "$SHIM_SO" || "$SHIM_SRC" -nt "$SHIM_SO" ]]; then
    mkdir -p "$(dirname "$SHIM_SO")"
    gcc -O2 -shared -fPIC -Wall -Wextra \
        -I/opt/rocm/include \
        "$SHIM_SRC" -ldl -lpthread \
        -o "$SHIM_SO"
fi

export HIP_VISIBLE_DEVICES="${HIP_VISIBLE_DEVICES:-0}"
export ROCR_VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0}"
export HSA_CU_MASK="0:0-$((CUS - 1))"
export NIB_SIM_GPU_NAME="$GPU_NAME"
export NIB_SIM_WGP="$WGPS"
export NIB_SIM_VRAM_MIB="$VRAM_MIB"
export NIB_SIM_VRAM_RESERVE_MIB="$RESERVE_MIB"
export NIB_SIM_BW_GBS="$BW_GBS"
export NIB_SIM_HOST_BW_GBS="${NIB_SIM_HOST_BW_GBS:-960}"
export LD_PRELOAD="${SHIM_SO}${LD_PRELOAD:+:$LD_PRELOAD}"

exec "$@"
