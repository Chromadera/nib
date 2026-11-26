#!/usr/bin/env bash
# MTP nib gen --mtp sweep with real code+docs prompts (same construction as docs/rx7900xt/scripts/bench_mtp.sh).
# SPECS="kv:ctx ..." ; each ctx runs the prompt sizes that fit (prompt + 256 gen + margin).
source /workspace/common.sh
OUT=$R/mtp; mkdir -p $OUT
python3 - <<'PY'
import pathlib
src = (pathlib.Path("/workspace/nib/README.md").read_text(errors="ignore") + "\n\n// --- engine.cpp ---\n"
       + pathlib.Path("/workspace/nib/src/engine.cpp").read_text(errors="ignore") + "\n\n// --- server.cpp ---\n"
       + pathlib.Path("/workspace/nib/src/server.cpp").read_text(errors="ignore"))
tail = ("\n\n### Task\nUsing C++17, write a clean, self-contained template class `RingBuffer<T, N>` "
        "supporting `push`, `pop`, `peek`, `size`, `empty`, `full`, and `clear`, "
        "followed by a short `main()` demonstrating usage and a 3-sentence complexity summary.\n```cpp\n")
for name, nb in {"2k":6200,"32k":100000,"64k":200000,"128k":395000,"176k":500000,"240k":690000}.items():
    body = (src * (nb // len(src) + 2))[:nb]
    pathlib.Path(f"/tmp/prompt_{name}.txt").write_text(body + tail)
PY
need() { case $1 in 2k) echo 4096;; 32k) echo 40960;; 64k) echo 81920;; 128k) echo 147456;; 176k) echo 196608;; 240k) echo 258048;; esac; }
for spec in $SPECS; do
  kv=${spec%%:*}; c=${spec##*:}
  for D in 2k 32k 64k 128k 176k 240k; do
    [ $(need $D) -le $c ] || continue
    tag=${kv}_${c}_$D; pkill -9 -x nib; wait_idle 19500
    pre="$(smi)"; smi_start /tmp/smi_mtp.txt
    timeout 1200 $NIB gen -m $M --mtp $MTPF --prompt-file /tmp/prompt_$D.txt -n 256 -c $c --kv-type $kv --sparse-pages 224 --cpu-embd > $OUT/gen_$tag.log 2>&1; rc=$?
    sm=$(smi_stop)
    eng=$(grep -m1 -o 'total_vram=[0-9.]* GiB' $OUT/gen_$tag.log); dr=$(grep -m1 'drafter ready' $OUT/gen_$tag.log)
    pf=$(grep -m1 '\[gen\] prefill' $OUT/gen_$tag.log); dec=$(grep -m1 -- '--- END (' $OUT/gen_$tag.log)
    st=$(grep -E '^\[mtp\] ' $OUT/gen_$tag.log | grep -v 'drafter online' | tr '\n' ' ')
    echo "$tag rc=$rc $eng | $dr | $pf | $dec | $st | pre: $pre | during: $sm" | tee -a $OUT/summary.txt
  done
done
echo "MTP DONE $(ts)" | tee -a $OUT/summary.txt
