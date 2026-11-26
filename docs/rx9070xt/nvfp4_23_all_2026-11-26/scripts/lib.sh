# common helpers for nib 9070xt runs
NIB=/workspace/nib/build/nib; M=/workspace/models; MTPF=$M/mtp.safetensors; MMPROJ=$M/mmproj-Inkling-2-Mini-Q8_0.gguf
PY=/workspace/bbvenv/bin/python; CL=/workspace/client.py; LOGD=/workspace/logs
ts() { date -u +%H:%M:%S; }
smi() { amd-smi metric -m 2>/dev/null | awk '/USED_VRAM:/{v=$2}/FREE_VRAM:/{f=$2}/USED_GTT:/{g=$2}END{printf "%s %s %s",v,f,g}'; }
kill_nib() { pkill -9 -x nib 2>/dev/null; for i in $(seq 1 60); do pgrep -x nib >/dev/null || break; sleep 1; done; sleep 3; }
start_poll() { ( while :; do echo "$(date +%s) $(smi)"; sleep 1; done ) > "$1" & POLL=$!; }
stop_poll() { kill $POLL 2>/dev/null; wait $POLL 2>/dev/null; }
# peak used vram / min free / peak gtt from poll file
poll_stats() { awk 'NF==4{if($2>mv)mv=$2; if(mf==""||$3<mf)mf=$3; if($4>mg)mg=$4}END{printf "peakUsed=%s minFree=%s peakGTT=%s",mv,mf,mg}' "$1"; }
# serve_up <logfile> args...  -> sets SRV ; returns 0 when healthy
serve_up() { local lf=$1; shift; "$NIB" serve --host 127.0.0.1 --port 4444 "$@" > "$lf" 2>&1 & SRV=$!
  for i in $(seq 1 240); do kill -0 $SRV 2>/dev/null || return 1; curl -sf --max-time 3 http://127.0.0.1:4444/health >/dev/null 2>&1 && return 0; sleep 1; done; return 2; }
