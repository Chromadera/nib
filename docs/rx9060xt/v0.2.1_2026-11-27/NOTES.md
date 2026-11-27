# RX 9060 XT — nib v0.2.1 (9de6314) rerun, 2026-11-27 (first full benchmark on this GPU)
Node `c88f4f05` (group `nib-v021-9060xt`, deleted). gfx1200, 32 CU (wgp=16). Quant `NVFP4-All`. No MTP.
- Build OK; `spill_audit` PASS (0/0/0/0); kernel test 9/9 (before, after selftests, at end).
- Decode HIP graph 452 nodes (co-residency fallback path; 404 on gfx1201).
- **0 hangs** in real workloads: `nib bench` q4_0 @ 73728 rc=0 (d0 2,852/55.77 → 72.7K 2,282/53.74); BetterBench s1 73728 (ss 56.1, prefill 2,243 @ 56,272) and s4 49152 (ss 56.0, agg 55.1/77.2/93.9, prefill 2,332 @ 36,841), all 40/40 + 32/32 OK, server alive.
- Test A harness (deadlock_repro_2026-11-27) NOT executed on v0.2.1: first build failed on its relative #include, second on link (hipcc `-x hip` applied to the .o files). Needs `-c` then link.
- Selftest (prefill-vs-decode gate cos > 0.999): q8_0 PASS 0.9999909; q4_0 0.9980436 and q4_1 0.9979789 rc=2 (deterministic over 3 runs; also with no --cpu-embd; --sparse-pages 0: q4_0 0.9994 PASS, q4_1 0.9969 FAIL). Values are bit-identical to RX 9070 XT v0.2.0 and v0.2.1 → RDNA4-wide 4-bit-KV behaviour, pre-existing, not a 32-CU fallback effect. RDNA3 passes q4_1 at 0.99961. Cause not verified.
