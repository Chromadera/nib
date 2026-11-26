# sourced by all bench scripts
export HIP_VISIBLE_DEVICES=0 ROCR_VISIBLE_DEVICES=0
NIB=/workspace/nib/build/nib
M=${M_OVERRIDE:-/workspace/models/Inkling-2-Mini-NVFP4-23.gguf}
MTPF=/workspace/models/mtp.safetensors
MMP=/workspace/models/mmproj-Inkling-2-Mini-Q8_0.gguf
R=/workspace/results; mkdir -p $R
ts() { date -u +%FT%TZ; }
smi() { amd-smi metric -m 2>/dev/null | grep -E '^\s*(USED_VRAM|FREE_VRAM|USED_GTT):' | tr -s ' \n' ' '; }
smi_val() { amd-smi metric -m 2>/dev/null | awk -v k="$1:" '$1==k{print $2; exit}'; }
# background sampler: smi_start <file>; smi_stop -> prints "min_free=.. max_gtt=.. max_used=.."
smi_start() { SMI_F=$1; : > $SMI_F; ( while true; do amd-smi metric -m 2>/dev/null | awk '$1=="USED_VRAM:"{u=$2} $1=="FREE_VRAM:"{f=$2} $1=="USED_GTT:"{g=$2} END{print u,f,g}' >> $SMI_F; sleep 0.5; done ) & SMI_PID=$!; }
smi_stop() { kill $SMI_PID 2>/dev/null; wait $SMI_PID 2>/dev/null; awk 'NF==3{if(mf==""||$2<mf)mf=$2; if($3>mg)mg=$3; if($1>mu)mu=$1; n++} END{printf "min_free=%s max_gtt=%s max_used=%s samples=%d\n",mf,mg,mu,n}' $SMI_F; }
wait_idle() { local idle=${1:-19500}; for i in $(seq 1 40); do f=$(smi_val FREE_VRAM); [ -n "$f" ] && [ "$f" -ge "$idle" ] && return 0; sleep 1.5; done; return 1; }
gpu_ok() { [ -n "$(smi_val FREE_VRAM)" ]; }
