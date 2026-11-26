#!/usr/bin/env bash
# One-shot pipeline: setup+repack -> Stage 1 -> selftests -> Stage 2 -> Phase 3 (bench, vision, BetterBench).
cd /workspace
[ -f /workspace/logs/sizes.txt ] && grep -q REPACK_DONE /workspace/logs/sizes.txt || bash /workspace/node_setup.sh
bash /workspace/stage1.sh
bash /workspace/selftest.sh
python3 probe.py cal_All_q4_1_c16384 All q4_1 16384
bash /workspace/stage2.sh
bash /workspace/phase3.sh
echo ALL_DONE >> /workspace/logs/run_all.done
