#!/usr/bin/env bash
# Stage 2 real-nib ceiling matrix (0 MB GTT spill beyond cpu_embd pinned host RAM) for NVFP4-All and NVFP4-23.
cd /workspace; export NIB_GTT_BASE=64 NIB_SPILL_TOL=32
L=/workspace/logs/stage2.log
log(){ echo "[$(date -u +%FT%TZ)] $*" | tee -a $L; }
log "=== STAGE 2 START"; amd-smi metric -m | grep -E "USED_VRAM|FREE_VRAM|USED_GTT" | head -3 | tee -a $L
# MTP feasibility at the smallest useful context (DSparkDrafter needs ~0.68 GiB)
for Q in All 23; do
  python3 probe.py s2_${Q}_q4_0_mtp_c4096 $Q q4_0 4096 --mtp 2>&1 | tee -a $L
  python3 probe.py s2_${Q}_q4_0_mtp_c2048_dctx1024 $Q q4_0 2048 --mtp --extra "--draft-ctx 1024" 2>&1 | tee -a $L
done
for Q in All 23; do
  for KV in q4_0 q4_1 q8_0; do python3 search.py $Q $KV --hi 131072 2>&1 | tee -a $L; done
  for KV in q4_0 q4_1; do python3 search.py $Q $KV --vision --hi 131072 2>&1 | tee -a $L; done
  for KV in q4_0 q4_1; do for S in 2 4 8; do python3 search.py $Q $KV --slots $S --hi 131072 2>&1 | tee -a $L; done; done
done
log "=== STAGE 2 DONE"
