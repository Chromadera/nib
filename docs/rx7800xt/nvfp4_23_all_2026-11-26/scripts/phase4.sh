#!/usr/bin/env bash
until grep -q "PHASE 3 DONE" /workspace/logs/phase3.log; do sleep 20; done
bash /workspace/bb_run.sh All q4_0 45056 4 bb_All_s4_q4_0_c45056 --concurrency
echo PHASE4_DONE >> /workspace/logs/phase4.done
