cd /workspace
P=/workspace/probe.sh
$P cal_all_q41_8k  All q4_1 8192 1 0 0
$P cal_all_q41_8k_gpuembd All q4_1 8192 1 0 0 --no-cpu-embd
$P cal_all_q41_32k All q4_1 32768 1 0 0
$P cal_all_q41_64k All q4_1 65536 1 0 0
$P cal_23_q41_8k   23 q4_1 8192 1 0 0
$P cal_23_q41_32k  23 q4_1 32768 1 0 0
$P cal_all_q40_8k_mtp_dc1k All q4_0 8192 1 1 0 --draft-ctx 1024
echo CALIB_DONE
