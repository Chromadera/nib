# RX 7800 XT — nib v0.2.1 (9de6314) rerun, 2026-11-27
Node `ce0cd0ba` (group `nib-v021-7800xt`, deleted). Quant `NVFP4-All` (sha256 verified). No MTP.
- Build gfx1101 OK; `spill_audit` PASS (0/0/0/0); kernel test 9/9; selftest q4_1 (cos 0.9996072) and q8_0 rc=0.
- `nib bench` q4_0 @ 69632: d0 3,251/97.87 → 68.6K 2,341/93.12; q4_1 @ 65536: 3,201/98.46 → 64.5K 1,989/92.79.
  q4_0 @ 69632 shows the same ~+70 MB GTT rise as v0.2.0 (flagged there too).
- BetterBench: s1 69632 q4_0 ss 97.8, prefill 2,452 @ 52,994; s4 45056 q4_0 ss 97.9, agg 95.8/130.2/159.1; s8 12288 q4_1 ss 97.8, agg c8 174.8 (32/32 OK every level).
- Pipeline scripts in `scripts/` are the post-DNS-hardening versions; this node ran the original ones (same steps).
