#!/usr/bin/env bash
until [ -f /workspace/logs/phase4.done ]; do sleep 20; done
cd /workspace
python3 probe.py vis_All_q4_0_c4096 All q4_0 4096 --vision --img 1080p,1440p,4k,2560sq >> /workspace/logs/phase5.log 2>&1
python3 probe.py vis_All_q4_0_c4096_nographs All q4_0 4096 --vision --img 4k,2560sq --extra "--no-graph" >> /workspace/logs/phase5.log 2>&1
python3 probe.py vis_23_q4_0_c2048 23 q4_0 2048 --vision --img 1080p >> /workspace/logs/phase5.log 2>&1
echo PHASE5_DONE >> /workspace/logs/phase5.done
