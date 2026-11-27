# RX 9070 XT — nib v0.2.1 (9de6314) rerun, 2026-11-27
Final data: node `c4a6a709` (group `nib-v021-9070xt-c`, deleted). Earlier nodes `083424cd` (2.5 MiB/s download) and `-b` were deleted at user request before benchmarking.
- Build gfx1201 OK; `spill_audit` PASS (0/0/0/0); kernel test 9/9; selftest q8_0 PASS 0.9999909, q4_1 rc=2 0.9979789 (bit-identical to v0.2.0 on this GPU — pre-existing).
- **2 intermittent hangs** in `nib bench` q4_0 @ 73728, both at the first pp512 at d0 after graph capture (13:25Z: GFX 100% stuck, VRAM resident; 14:13Z: GFX 0%, host threads spinning, only 747 MB VRAM in use → device context apparently lost). Evidence in `raw/node_c4a6a709/logs/hang/` (+ `bench_hung_1413Z/`, `bench_nonmtp_HUNG_*.log`).
- Same-node A/B (`logs/hang/ab/`): v0.2.1 vs v0.2.0, 3 alternating full-ladder runs each → 6/6 clean, identical throughput (d0 tg128 95.4–95.5 both). Hang did not reproduce on either build; host vs v0.2.1 not separated.
- Final pipeline (scripts/run_rest.sh + hangwatch.sh watchdog): 0 further hangs. NB: watchdog v1 crashed on a pid race at ~14:35 and was replaced (v3 from 14:40:53); BB s1 ran 14:35–14:40 without it.
- `nib bench` q4_0 @ 73728: d0 4,287/95.53 → 72.7K 4,212/93.40; q4_1 @ 65536: 4,308/95.42 → 64.5K 3,481/93.18. d0 pp512 ± ~1,400 = one slow first repeat (same on v0.2.0 on this node).
- BetterBench: s1 73728 q4_0 ss 97.0, prefill 4,099 @ 55,980; s4 49152 q4_0 ss 96.9, agg 95.2/128.2/163.8, prefill 4,271 @ 36,967 (all OK).
