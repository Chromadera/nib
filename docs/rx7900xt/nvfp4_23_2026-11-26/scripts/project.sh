#!/usr/bin/env bash
# project.sh kv slots mtp vis ctx -> prints projected free MiB (from nib's own estimator)
source /workspace/common.sh
kv=$1 s=$2 mtp=$3 vis=$4 c=$5
a=(-m $M --slots $s -c $c --kv-type $kv --sparse-pages 224 --cpu-embd --host 127.0.0.1 --port 4499)
[ $mtp = 1 ] && a+=(--mtp $MTPF); [ $vis = 1 ] && a+=(--mmproj $MMP) || a+=(--no-vision)
$NIB serve "${a[@]}" > /tmp/proj.log 2>&1 & p=$!
for i in $(seq 1 60); do grep -q 'projected free after load' /tmp/proj.log && break; sleep 0.25; done
kill -9 $p 2>/dev/null; wait $p 2>/dev/null
grep -o 'projected free after load ~-\?[0-9]*' /tmp/proj.log | grep -o -- '-\?[0-9]*$'
