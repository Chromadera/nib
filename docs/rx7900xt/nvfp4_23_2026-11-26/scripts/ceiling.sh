#!/usr/bin/env bash
# Stage 2: real nib ceiling matrix for NVFP4-23 (--sparse-pages 224, --cpu-embd).
# For each (kv,slots,mtp,vis): fit nib's own linear VRAM projection from 2 quick boots, then real-boot
# candidates (projected free >= 64, 256, 512, 1024 MiB; capped at 262848) descending until one PASSES.
# PASS = boots <=240s, all text reqs PASS (+1080p image when vis), GTT rise during reqs <= 8 MB, tps>=60.
source /workspace/common.sh
OUT=${CEIL_OUT:-$R/ceiling}; mkdir -p $OUT; J=$OUT/ceiling.jsonl
IDLE_FREE=$(smi_val FREE_VRAM); echo "idle $(ts): $(smi)" | tee -a $OUT/summary.txt
round() { local s=$1 c=$2; local st=$((1024*s)); c=$(( c / st * st )); [ $c -gt 262848 ] && c=262848; echo $c; }
probe() { # kv s mtp vis ctx -> 0 pass
  local kv=$1 s=$2 mtp=$3 vis=$4 c=$5 tag="${1}_s${2}_mtp${3}_vis${4}_${5}"
  pkill -9 -x nib 2>/dev/null; wait_idle $((IDLE_FREE-300))
  local pre_gtt=$(smi_val USED_GTT) pre="$(smi)"
  local a=(-m $M --slots $s -c $c --kv-type $kv --sparse-pages 224 --cpu-embd --host 127.0.0.1 --port 4455)
  [ $mtp = 1 ] && a+=(--mtp $MTPF); [ $vis = 1 ] && a+=(--mmproj $MMP) || a+=(--no-vision)
  $NIB serve "${a[@]}" > $OUT/serve_$tag.log 2>&1 & local pid=$! ok=0 t0=$(date +%s)
  while [ $(( $(date +%s)-t0 )) -lt 240 ]; do kill -0 $pid 2>/dev/null || { ok=2; break; }; curl -sf -m 3 127.0.0.1:4455/health >/dev/null && { ok=1; break; }; sleep 1; done
  local bt=$(( $(date +%s)-t0 ))
  if [ $ok != 1 ]; then
    local st=$([ $ok = 2 ] && echo START_FAILED || echo STALL); kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
    local err=$(grep -iE 'hipMalloc|out of memory|failed|error' $OUT/serve_$tag.log | tail -2 | tr '\n"' '  ')
    echo "{\"tag\":\"$tag\",\"kv\":\"$kv\",\"slots\":$s,\"mtp\":$mtp,\"vis\":$vis,\"ctx\":$c,\"status\":\"$st\",\"boot_s\":$bt,\"err\":\"$err\"}" >> $J
    echo "$tag $st (${bt}s) $err" | tee -a $OUT/summary.txt; return 1
  fi
  local boot_gtt=$(smi_val USED_GTT) boot="$(smi)"
  smi_start /tmp/smi_ceil.txt
  local cl=$(timeout 400 python3 /workspace/probe_client.py 4455 $s $vis); local sm=$(smi_stop)
  local max_gtt=$(sed -n 's/.*max_gtt=\([0-9]*\).*/\1/p' <<<"$sm"); local rise=$(( ${max_gtt:-0} - boot_gtt ))
  local cfgline=$(grep -m1 '^\[nib\] Device VRAM' $OUT/serve_$tag.log | tr '"' ' ')
  local eng=$(grep -m1 -o 'total_vram=[0-9.]* GiB' $OUT/serve_$tag.log)
  kill -9 $pid 2>/dev/null; wait $pid 2>/dev/null
  local verdict=$(python3 - "$cl" "$rise" "$vis" <<'PY'
import json,sys
try: d=json.loads(sys.argv[1])
except Exception: print("CLIENT_FAIL"); sys.exit()
rise=int(sys.argv[2]); ok=all(r and r["status"]=="PASS" and (r["tps"] or 0)>=(60 if len(d["text"])==1 else 10) for r in d["text"])
if sys.argv[3]=="1": ok = ok and d.get("img1080",{}).get("status")=="PASS"
print("PASS" if ok and rise<=8 else ("GTT_SPILL" if ok else "REQ_FAIL"))
PY
)
  echo "{\"tag\":\"$tag\",\"kv\":\"$kv\",\"slots\":$s,\"mtp\":$mtp,\"vis\":$vis,\"ctx\":$c,\"status\":\"$verdict\",\"boot_s\":$bt,\"engine\":\"$eng\",\"device\":\"$cfgline\",\"pre_smi\":\"$pre\",\"boot_smi\":\"$boot\",\"req_smi\":\"$sm\",\"gtt_rise_mb\":$rise,\"gtt_boot_minus_idle_mb\":$(( boot_gtt - pre_gtt )),\"client\":${cl:-null}}" >> $J
  echo "$tag $verdict boot=${bt}s $eng | $cfgline | boot: $boot | req: $sm | rise=${rise}MB" | tee -a $OUT/summary.txt
  gpu_ok || { echo "GPU_LOST $(ts)" | tee -a $OUT/summary.txt; exit 2; }
  [ "$verdict" = PASS ]
}
CONFIGS=${CONFIGS:-"$(for vis in 0 1; do for mtp in 0 1; do for s in 1 4 8; do for kv in q4_0 q4_1 q8_0; do echo "$kv:$s:$mtp:$vis"; done; done; done; done)"}
for cf in $CONFIGS; do
  IFS=: read kv s mtp vis <<<"$cf"
  p1=$(/workspace/project.sh $kv $s $mtp $vis 65536); p2=$(/workspace/project.sh $kv $s $mtp $vis 196608)
  # free(c) = p1 + (c-65536)*slope
  cands=""; for thr in -128 64 256 512 1024; do
    c=$(python3 -c "p1,p2,t=$p1,$p2,$thr; sl=(p2-p1)/131072; print(int(65536+(t-p1)/sl) if sl<0 else 262848)")
    c=$(round $s $c); [ $c -lt 8192 ] && continue; case " $cands " in *" $c "*) ;; *) cands="$cands $c";; esac
  done
  echo "=== $cf proj@64k=${p1}MiB proj@192k=${p2}MiB candidates:$cands $(ts)" | tee -a $OUT/summary.txt
  got=""; for c in $cands; do probe $kv $s $mtp $vis $c && { got=$c; break; }; done
  [ -z "$got" ] && echo "NOFIT $cf (no candidate passed)" | tee -a $OUT/summary.txt
  echo "{\"cfg\":\"$cf\",\"max_pass_ctx\":${got:-null},\"proj64k\":$p1,\"proj192k\":$p2,\"cands\":\"$cands\"}" >> $OUT/max.jsonl
done
echo "CEILING DONE $(ts)" | tee -a $OUT/summary.txt
