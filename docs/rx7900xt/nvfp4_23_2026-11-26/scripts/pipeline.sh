#!/usr/bin/env bash
source /workspace/common.sh
cd /workspace
echo "PIPELINE START $(ts)" | tee $R/pipeline.txt
amd-smi static > $R/amdsmi_static_pipeline_start.txt 2>&1
echo "stage nonmtp $(ts): $(smi)" >> $R/pipeline.txt
SPECS="q4_1:262848 q4_0:262848 q8_0:256000" bash bench_nonmtp.sh > $R/nonmtp.nohup 2>&1
echo "stage mtp $(ts): $(smi)" >> $R/pipeline.txt
SPECS="q4_1:262848 q4_0:262848 q8_0:201728" bash bench_mtp.sh > $R/mtp.nohup 2>&1
echo "stage bb $(ts): $(smi)" >> $R/pipeline.txt
S=8 C=262848 KV=q4_1 NAME=betterbench_slots8_262k_q4_1_mtp bash bench_bb.sh > $R/bb.nohup 2>&1
echo "stage vision $(ts): $(smi)" >> $R/pipeline.txt
CONFIGS="q4_1:262848:1:1 q4_1:262848:1:0 q4_0:262848:1:1 q4_1:262848:4:1 q4_1:131072:4:1" bash bench_vision.sh > $R/vision.nohup 2>&1
amd-smi static > $R/amdsmi_static_pipeline_end.txt 2>&1
echo "PIPELINE DONE $(ts): $(smi)" | tee -a $R/pipeline.txt
