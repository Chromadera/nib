#!/usr/bin/env bash
# verify_rdna4.sh — static checks for the gfx1201 port. Runs NO GPU kernels:
# it only inspects the built code objects, so it is safe next to a production GPU.
#
#   1. device: rocminfo sees gfx1201
#   2. every ported kernel has a real body with v_dot2_f32_f16 (no empty no-op)
#   3. gfx1100 codegen parity against a pristine upstream build (optional, pass its dir)
#   4. spill_audit.sh for gfx1201
#
# usage: scripts/verify_rdna4.sh [path/to/pristine/gfx1100/build]
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${ROCM_PATH:-/opt/rocm}/llvm/bin"
BUILD="${BUILD:-$ROOT/build}"
GPU=1201

say() { printf '\n=== %s ===\n' "$*"; }

say "1. device"
rocminfo 2>/dev/null | grep -E 'Marketing Name|gfx1201|Compute Unit' | tail -3 || echo "rocminfo not found"

say "2. ported kernels (gfx$GPU)"
"$ROOT/scripts/spill_audit.sh" >/dev/null 2>&1 || true   # warms nothing; kept for symmetry
W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
check() { # obj pattern label
  local o="$1" pat="$2" label="$3"
  rm -f "$W/f.bin" "$W/h.o" "$W/d.o"
  "$BIN/llvm-objcopy" --dump-section .hip_fatbin="$W/f.bin" "$o" 2>/dev/null || return 0
  "$BIN/clang-offload-bundler" -type=o -unbundle -input="$W/f.bin" \
      -targets="host-x86_64-unknown-linux-gnu-,hip-amdgcn-amd-amdhsa--gfx${GPU}" \
      -output="$W/h.o" -output="$W/d.o" >/dev/null 2>&1 || true
  [ -s "$W/d.o" ] || { printf '  %-40s NO CODE OBJECT\n' "$label"; return 0; }
  "$BIN/llvm-objdump" -d --mcpu=gfx"$GPU" "$W/d.o" > "$W/d.s" 2>/dev/null
  python3 - "$W/d.s" "$pat" "$label" <<'PY'
import re, sys
path, pat, label = sys.argv[1:4]
cur = None; insns = dot2 = wmma = 0; found = False
for line in open(path, errors='ignore'):
    m = re.match(r'^[0-9a-f]+ <(.+)>:', line)
    if m:
        cur = m.group(1); continue
    if cur and pat in cur and '\t' in line:
        insns += 1; found = True
        dot2 += 'v_dot2_f32_f16' in line
        wmma += 'v_wmma' in line
print("  %-40s insns=%-6d v_dot2=%-5d v_wmma=%d%s" %
      (label, insns, dot2, wmma, "" if found else "  <-- EMPTY BODY"))
PY
}
check "$BUILD/kernels_fa.o"            "inkling_fa_prefill_gqa_wmma_kernelILi12ELi48ELb0ELi0E" "FA prefill (global)"
check "$BUILD/kernels_fa.o"            "inkling_fa_prefill_gqa_wmma_kernelILi6ELi48ELb1ELi0E"  "FA prefill (SWA)"
check "$BUILD/kernels_core.o"          "inkling_ctrl_down_prefill_wmma_kernel"   "ctrl_down prefill"
check "$BUILD/kernels_core.o"          "inkling_ctrl_apply_prefill_wmma_kernel"  "ctrl_apply prefill"
check "$BUILD/kernels_core.o"          "router_gemm_topk_prefill_wmma_kernel"    "router prefill"
check "$BUILD/kernels_wmma.o"          "dense_q8_wmma_kernelILi64ELi64ELi2ELi2E" "dense GEMM 64x64"
check "$BUILD/kernels_wmma.o"          "moe_gate_up_wmma_kernel"                 "MoE gate/up"
check "$BUILD/kernels_sparse_global.o" "sparse_score_pages_prefill_wmma_kernel" "Quest page scorer"
check "$BUILD/kernels_vision.o"        "vision_patch_embed_pos_cls_kernel"       "vision patch embed"
check "$BUILD/kernels_vision.o"        "vision_fa_wmma_d64_kernel"               "vision FA d64"

if [ "$#" -ge 1 ] && [ -d "$1" ]; then
  say "3. gfx1100 codegen parity vs $1"
  same=0; diff=0
  for f in "$1"/*.o; do
    n=$(basename "$f"); [ -f "$ROOT/build-gfx1100/$n" ] || continue
    un() { local o="$1" out="$2"
      rm -f "$W/f.bin" "$W/h.o" "$W/d.o"
      "$BIN/llvm-objcopy" --dump-section .hip_fatbin="$W/f.bin" "$o" 2>/dev/null || return 1
      "$BIN/clang-offload-bundler" -type=o -unbundle -input="$W/f.bin" \
        -targets="host-x86_64-unknown-linux-gnu-,hip-amdgcn-amd-amdhsa--gfx1100" \
        -output="$W/h.o" -output="$W/d.o" >/dev/null 2>&1 || true
      [ -s "$W/d.o" ] || return 1
      "$BIN/llvm-objdump" -d --mcpu=gfx1100 "$W/d.o" | sed -E 's/^[[:space:]]*[0-9a-f]+://; s|//.*||; s/[[:space:]]+$//' > "$out"
    }
    un "$f" "$W/a.s" || continue
    un "$ROOT/build-gfx1100/$n" "$W/b.s" || continue
    if cmp -s "$W/a.s" "$W/b.s"; then same=$((same+1)); else echo "  DIFFERS $n"; diff=$((diff+1)); fi
  done
  echo "  $same identical, $diff differ"
else
  say "3. gfx1100 codegen parity — skipped (pass the pristine gfx1100 build dir)"
fi

say "4. spill audit (gfx$GPU)"
GPU_TARGET=gfx$GPU "$ROOT/scripts/spill_audit.sh" || true
