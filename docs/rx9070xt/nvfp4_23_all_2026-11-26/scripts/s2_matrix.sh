cd /workspace; S=/workspace/search.sh
# MTP definitive minimum attempts
/workspace/probe.sh mtp_all_q40_4k_dc512 All q4_0 4096 1 1 0 --draft-ctx 512
/workspace/probe.sh mtp_23_q40_4k_dc512  23  q4_0 4096 1 1 0 --draft-ctx 512
# non-MTP text, slots 1
$S all_q40_s1 All q4_0 1 0 0 65536 196608 4096
$S all_q41_s1 All q4_1 1 0 0 65536 163840 4096
$S all_q80_s1 All q8_0 1 0 0 8192 98304 4096
$S 23_q40_s1  23 q4_0 1 0 0 8192 49152 2048
$S 23_q41_s1  23 q4_1 1 0 0 8192 32768 2048
$S 23_q80_s1  23 q8_0 1 0 0 4096 24576 2048
# multi-slot q4_1
$S all_q41_s2 All q4_1 2 0 0 16384 131072 4096
$S all_q41_s4 All q4_1 4 0 0 16384 131072 4096
$S all_q41_s8 All q4_1 8 0 0 16384 131072 8192
$S 23_q41_s2  23 q4_1 2 0 0 4096 32768 2048
$S 23_q41_s4  23 q4_1 4 0 0 4096 32768 4096
$S 23_q41_s8  23 q4_1 8 0 0 8192 32768 8192
# vision (no MTP), q4_1
$S all_q41_s1_vis All q4_1 1 0 1 8192 98304 4096
$S 23_q41_s1_vis  23 q4_1 1 0 1 4096 24576 2048
echo MATRIX_DONE
