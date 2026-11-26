cd /workspace
until grep -q BB_ALL_DONE /workspace/logs/bb_all_runs2.log; do sleep 20; done
bash /workspace/concheck.sh
V=/workspace/vis.sh
$V all_q41_s1_32k All q4_1 32768 1
$V all_q41_s1_16k All q4_1 16384 1
$V all_q41_s1_8k  All q4_1 8192 1
$V all_q40_s1_32k All q4_0 32768 1
$V all_q41_s2_20k All q4_1 20480 2
$V 23_q40_s1_4k   23  q4_0 4096 1
echo POSTQ_DONE
