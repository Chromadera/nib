#!/usr/bin/env bash
# Register-spill audit for Nib's RDNA3 HIP kernel objects.
# Extracts .hip_fatbin -> GPU code object -> NT_AMDGPU_METADATA and fails on any
# SGPR spill, VGPR spill, or private-segment allocation.
set -euo pipefail

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
GPU_TARGET="${GPU_TARGET:-gfx1100}"
BIN="${ROCM_PATH}/llvm/bin"

if [ "$#" -gt 0 ]; then
  OBJS=("$@")
else
  shopt -s nullglob
  OBJS=(build/*.o)
  shopt -u nullglob
fi

if [ "${#OBJS[@]}" -eq 0 ]; then
  echo "Usage: $0 [build/*.o ...]" >&2
  echo "No HIP kernel .o files found. Run 'cmake --build build' first." >&2
  exit 1
fi

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

ok=1
checked=0
for o in "${OBJS[@]}"; do
  rm -f "$WORK/fat.bin" "$WORK/h.o" "$WORK/d.o" "$WORK/notes.txt"
  "$BIN/llvm-objcopy" --dump-section .hip_fatbin="$WORK/fat.bin" "$o" 2>/dev/null || continue
  [ -s "$WORK/fat.bin" ] || continue
  checked=$((checked + 1))

  "$BIN/clang-offload-bundler" -type=o -unbundle -input="$WORK/fat.bin" \
      -targets="host-x86_64-unknown-linux-gnu-,hip-amdgcn-amd-amdhsa--${GPU_TARGET}" \
      -output="$WORK/h.o" -output="$WORK/d.o" >/dev/null 2>&1 || true
  if [ ! -s "$WORK/d.o" ]; then
    echo "$(basename "$o"): failed to unbundle ${GPU_TARGET} code object"
    ok=0
    continue
  fi

  "$BIN/llvm-readobj" --notes "$WORK/d.o" > "$WORK/notes.txt" 2>/dev/null
  kern=$(grep -c "\.vgpr_count:" "$WORK/notes.txt" || true)
  sgpr_sp=$(grep -o "\.sgpr_spill_count: *[0-9]*" "$WORK/notes.txt" | awk '{s+=$2} END {print s+0}')
  vgpr_sp=$(grep -o "\.vgpr_spill_count: *[0-9]*" "$WORK/notes.txt" | awk '{s+=$2} END {print s+0}')
  priv=$(grep -o "\.private_segment_fixed_size: *[0-9]*" "$WORK/notes.txt" | awk '{if ($2>m) m=$2} END {print m+0}')
  maxv=$(grep -o "\.vgpr_count: *[0-9]*" "$WORK/notes.txt" | awk '{if ($2>m) m=$2} END {print m+0}')
  maxs=$(grep -o "\.sgpr_count: *[0-9]*" "$WORK/notes.txt" | awk '{if ($2>m) m=$2} END {print m+0}')
  lds=$(grep -o "\.group_segment_fixed_size: *[0-9]*" "$WORK/notes.txt" | awk '{if ($2>m) m=$2} END {print m+0}')
  worstv=$(grep -B6 "\.vgpr_spill_count: *[1-9]" "$WORK/notes.txt" | grep "\.name:" | tail -1 | sed 's/.*\.name: *//' || true)

  printf "%-28s kernels=%3d maxVGPR=%3d maxSGPR=%3d maxLDS=%5d maxPriv=%5d sgprSpills=%d vgprSpills=%d %s\n" \
      "$(basename "$o")" "$kern" "$maxv" "$maxs" "$lds" "$priv" "$sgpr_sp" "$vgpr_sp" "${worstv:+[$worstv]}"

  if [ "$sgpr_sp" != "0" ] || [ "$vgpr_sp" != "0" ] || [ "$priv" != "0" ]; then
    ok=0
  fi
done

if [ "$checked" -eq 0 ]; then
  echo "No HIP fatbin objects found in inputs." >&2
  exit 1
fi

if [ "$ok" = "1" ]; then
  echo "==> AUDIT PASS: 0 SGPR spills, 0 VGPR spills, 0 private segment across ${checked} objects (${GPU_TARGET})"
else
  echo "==> AUDIT FAIL"
  exit 1
fi
