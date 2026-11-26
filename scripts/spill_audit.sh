#!/usr/bin/env bash
# Register-spill & hollow-stub audit for Nib's RDNA3/RDNA4 HIP kernel objects.
# Extracts .hip_fatbin -> GPU code object(s) -> NT_AMDGPU_METADATA + ELF symbols
# and fails on any SGPR spill, VGPR spill, private-segment allocation, or
# hollow kernel stub (function code size <= 16 bytes, e.g. bare s_endpgm).
set -euo pipefail

ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
BIN="${ROCM_PATH}/llvm/bin"

OBJS=()
if [ "$#" -gt 0 ]; then
  shopt -s nullglob
  for arg in "$@"; do
    if [ -d "$arg" ]; then
      OBJS+=("${arg%/}"/*.o)
    else
      OBJS+=("$arg")
    fi
  done
  shopt -u nullglob
else
  shopt -s nullglob
  OBJS=(build/*.o)
  shopt -u nullglob
fi

if [ "${#OBJS[@]}" -eq 0 ]; then
  echo "Usage: $0 [build_dir | build/*.o ...]" >&2
  echo "No HIP kernel .o files found. Run 'cmake --build build' first." >&2
  exit 1
fi

# Auto-detect GPU_TARGET from CMakeCache.txt in the object directory if unset.
if [ -z "${GPU_TARGET:-}" ]; then
  obj_dir="$(dirname "${OBJS[0]}")"
  cache_file="${obj_dir}/CMakeCache.txt"
  if [ -f "$cache_file" ]; then
    cached_tgt="$(grep -E '^GPU_TARGET:STRING=' "$cache_file" | head -1 | cut -d= -f2- || true)"
    if [ -n "$cached_tgt" ]; then
      GPU_TARGET="$cached_tgt"
    fi
  fi
fi
GPU_TARGET="${GPU_TARGET:-gfx1100}"

# Split semicolon- or comma-separated GPU targets into an array.
IFS=';,' read -r -a TARGETS <<< "$GPU_TARGET"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

ok=1
total_checked=0
for tgt in "${TARGETS[@]}"; do
  tgt="${tgt// /}"
  [ -n "$tgt" ] || continue
  checked=0
  echo "=== Auditing GPU target: ${tgt} ==="
  for o in "${OBJS[@]}"; do
    rm -f "$WORK/fat.bin" "$WORK/h.o" "$WORK/d.o" "$WORK/notes.txt" "$WORK/syms.txt"
    "$BIN/llvm-objcopy" --dump-section .hip_fatbin="$WORK/fat.bin" "$o" 2>/dev/null || continue
    [ -s "$WORK/fat.bin" ] || continue
    checked=$((checked + 1))

    "$BIN/clang-offload-bundler" -type=o -unbundle -input="$WORK/fat.bin" \
        -targets="host-x86_64-unknown-linux-gnu-,hip-amdgcn-amd-amdhsa--${tgt}" \
        -output="$WORK/h.o" -output="$WORK/d.o" >/dev/null 2>&1 || true
    if [ ! -s "$WORK/d.o" ]; then
      echo "$(basename "$o"): failed to unbundle ${tgt} code object"
      ok=0
      continue
    fi

    "$BIN/llvm-readobj" --notes "$WORK/d.o" > "$WORK/notes.txt" 2>/dev/null
    "$BIN/llvm-readobj" --symbols "$WORK/d.o" > "$WORK/syms.txt" 2>/dev/null

    kern=$(grep -c "\.vgpr_count:" "$WORK/notes.txt" || true)
    sgpr_sp=$(grep -o "\.sgpr_spill_count: *[0-9]*" "$WORK/notes.txt" | awk '{s+=$2} END {print s+0}')
    vgpr_sp=$(grep -o "\.vgpr_spill_count: *[0-9]*" "$WORK/notes.txt" | awk '{s+=$2} END {print s+0}')
    priv=$(grep -o "\.private_segment_fixed_size: *[0-9]*" "$WORK/notes.txt" | awk '{if ($2>m) m=$2} END {print m+0}')
    maxv=$(grep -o "\.vgpr_count: *[0-9]*" "$WORK/notes.txt" | awk '{if ($2>m) m=$2} END {print m+0}')
    maxs=$(grep -o "\.sgpr_count: *[0-9]*" "$WORK/notes.txt" | awk '{if ($2>m) m=$2} END {print m+0}')
    lds=$(grep -o "\.group_segment_fixed_size: *[0-9]*" "$WORK/notes.txt" | awk '{if ($2>m) m=$2} END {print m+0}')
    worstv=$(grep -B6 "\.vgpr_spill_count: *[1-9]" "$WORK/notes.txt" | grep "\.name:" | tail -1 | sed 's/.*\.name: *//' || true)

    # Detect hollow kernel stubs (STT_FUNC with ELF Size <= 16 bytes, e.g. bare s_endpgm).
    stub_info=$(awk '
      /^[[:space:]]*Symbol \{/ { name=""; size=-1; type=""; binding=""; next }
      /^[[:space:]]*Name:/     { name=$2; next }
      /^[[:space:]]*Size:/     { size=$2+0; next }
      /^[[:space:]]*Binding:/  { binding=$2; next }
      /^[[:space:]]*Type:/     { type=$2; next }
      /^[[:space:]]*\}/ {
        if (type == "Function" && binding == "Global" && size >= 0 && size <= 16 && name != "") {
          cnt++;
          if (first == "") first = name "(" size "B)";
        }
      }
      END { printf "%d %s", cnt+0, first }
    ' "$WORK/syms.txt")
    stubs="${stub_info%% *}"
    first_stub="${stub_info#* }"
    [ "$first_stub" = "$stubs" ] && first_stub=""

    diag=""
    [ -n "$worstv" ] && diag="${diag}[spill:${worstv}] "
    [ "$stubs" != "0" ] && diag="${diag}[STUB:${first_stub}]"

    printf "%-28s kernels=%3d maxVGPR=%3d maxSGPR=%3d maxLDS=%5d maxPriv=%5d sgprSpills=%d vgprSpills=%d stubs=%d %s\n" \
        "$(basename "$o")" "$kern" "$maxv" "$maxs" "$lds" "$priv" "$sgpr_sp" "$vgpr_sp" "$stubs" "$diag"

    if [ "$sgpr_sp" != "0" ] || [ "$vgpr_sp" != "0" ] || [ "$priv" != "0" ] || [ "$stubs" != "0" ]; then
      ok=0
    fi
  done
  total_checked=$((total_checked + checked))
done

if [ "$total_checked" -eq 0 ]; then
  echo "No HIP fatbin objects found in inputs." >&2
  exit 1
fi

if [ "$ok" = "1" ]; then
  echo "==> AUDIT PASS: 0 SGPR spills, 0 VGPR spills, 0 private segment, 0 stubs (${GPU_TARGET})"
else
  echo "==> AUDIT FAIL"
  exit 1
fi
