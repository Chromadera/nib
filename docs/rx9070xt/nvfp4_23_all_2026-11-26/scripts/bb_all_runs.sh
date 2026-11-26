cd /workspace
until grep -q BENCH_DONE /workspace/logs/s2b_bench.log; do sleep 20; done
/workspace/bb.sh all_s4_q40_48k All q4_0 49152 4
/workspace/bb.sh all_s1_q40_72k All q4_0 73728 1
/workspace/bb.sh 23_s2_q40_16k  23  q4_0 16384 2
/workspace/bb.sh 23_s1_q40_24k  23  q4_0 24576 1
echo BB_ALL_DONE
