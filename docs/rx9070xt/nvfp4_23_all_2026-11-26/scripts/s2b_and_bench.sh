cd /workspace; S=/workspace/search.sh; P=/workspace/probe.sh
$P all_q41_s8_c8192 All q4_1 8192 8 0 0
$P all_q40_s8_c8192 All q4_0 8192 8 0 0
$S all_q40_s2 All q4_0 2 0 0 61440 81920 4096
$S all_q40_s4 All q4_0 4 0 0 45056 69632 4096
$S all_q80_s2 All q8_0 2 0 0 8192 32768 4096
$S all_q80_s4 All q8_0 4 0 0 8192 32768 4096
$S 23_q40_s2  23 q4_0 2 0 0 4096 24576 2048
$P 23_q40_s4_c4096 23 q4_0 4096 4 0 0
$S all_q40_s1_vis All q4_0 1 0 1 32768 73728 4096
$S all_q41_s2_vis All q4_1 2 0 1 8192 32768 4096
$P 23_q40_s1_vis_c4096 23 q4_0 4096 1 0 1
echo S2B_DONE
B=/workspace/bench.sh
$B all_q40_73k All q4_0 73728 0,8192,16384,32768,65536,72704
$B all_q41_64k All q4_1 65536 0,8192,16384,32768,64000
$B all_q80_32k All q8_0 32768 0,8192,16384,32000
$B 23_q40_24k  23 q4_0 24576 0,8192,16384,23552
$B 23_q41_16k  23 q4_1 16384 0,8192,15616
$B 23_q80_8k   23 q8_0 8192 0,7424
echo BENCH_DONE
