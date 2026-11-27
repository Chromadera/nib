# RX 7900 XT — nib v0.2.1 (9de6314) rerun, 2026-11-27
Node `fb7533a1` (group `nib-v021-7900xt-b`, deleted). Quant `NVFP4-23` (HF Nib-Pack, sha256 verified) + mtp/shortlist/mmproj.
- Build gfx1100 OK; `spill_audit` PASS (0/0/0/0); `nib_kernel_test` 9/9; selftest q4_1, q8_0, q4_1+MTP all rc=0 (q4_1 cos 0.9996158).
- `nib bench` q4_1/q4_0 @ 262848, -r 2: d0 4,577/124.47 → 256K 3,185/116.27 (q4_1); q4_0 4,553/124.49 → 3,201/116.75.
- BetterBench 0.6.1 `--slots 8 -c 262848 q4_1 --mtp`: single-stream 163.2; agg c1/2/4/8 150.0/152.5/200.9/206.4 (128/128 OK); prefill 3,215 tok/s @ 204,108.
- Host note: desktop held 1,276 MB VRAM at idle (v0.2.0 host: 367 MB) → 90 MB free after boot; transient +85 MB GTT in the first ~75 s (boot + 32 batch-graph captures), then flat for the measured run.
- First node `114c8a84` had no DNS (all steps rc=127); its logs are in `raw/node_114c8a84_DNS_FAILED/`.
- Layout: `raw/node_*/logs` (setup, build, audit, selftests, bench, BB client/serve logs, amd-smi samples), `raw/node_*/bb` (BB JSON/HTML/config/health), `scripts/` (pipeline as run; node_setup/run_all/watch were hardened after the DNS failure).
