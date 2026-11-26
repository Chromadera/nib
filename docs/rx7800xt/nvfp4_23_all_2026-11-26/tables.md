### Stage 2 ceiling matrix (max clean `-c`, step 4096, `--cpu-embd --sparse-pages 224`)

| Config | NVFP4-All max ctx | NVFP4-All engine VRAM | NVFP4-23 max ctx | NVFP4-23 engine VRAM |
|---|---:|---:|---:|---:|
| `q4_0` `--slots 1` | **69,632** | 15.03 GiB | **20,480** | 15.07 GiB |
| `q4_1` `--slots 1` | **65,536** | 15.06 GiB | **16,384** | 15.05 GiB |
| `q8_0` `--slots 1` | **32,768** | 15.01 GiB | **8,192** | 15.08 GiB |
| `q4_0` `--slots 2` | **65,536** | 15.06 GiB | **12,288** | 15.06 GiB |
| `q4_1` `--slots 2` | **57,344** | 15.06 GiB | **12,288** | 15.09 GiB |
| `q4_0` `--slots 4` | **45,056** | 15.02 GiB | **does not fit** | — |
| `q4_1` `--slots 4` | **40,960** | 15.05 GiB | **does not fit** | — |
| `q4_0` `--slots 8` | **12,288** | 15.01 GiB | **does not fit** | — |
| `q4_1` `--slots 8` | **12,288** | 15.08 GiB | **does not fit** | — |
| `q4_0` + vision (1080p) `--slots 1` | **20,480** | 14.59 GiB | **does not fit** | — |
| `q4_1` + vision (1080p) `--slots 1` | **16,384** | 14.58 GiB | **does not fit** | — |

### MTP feasibility

| Probe | Engine VRAM (before drafter) | Result |
|---|---:|---|
| `s2_All_q4_0_mtp_c4096` | 14.45 GiB | BOOT_FAIL: `[dspark] init failed: DSparkDrafter: hipMalloc failed` |
| `s2_All_q4_0_mtp_c2048_dctx1024` | 14.43 GiB | BOOT_FAIL: `[dspark] init failed: DSparkDrafter: hipMalloc failed` |
| `s2_23_q4_0_mtp_c4096` | 14.92 GiB | BOOT_FAIL: `[dspark] init failed: DSparkDrafter: hipMalloc failed` |
| `s2_23_q4_0_mtp_c2048_dctx1024` | 14.9 GiB | BOOT_FAIL: `[dspark] init failed: DSparkDrafter: hipMalloc failed` |

### Non-MTP `nib bench` (pp512 / tg128, r2)

#### Prefill `pp512` (tok/s)

| Depth | `23 q4 0 c20480` (15.07 GiB) | `23 q4 1 c16384` (15.05 GiB) | `23 q8 0 c8192` (15.08 GiB) | `All q4 0 c69632` (15.03 GiB) | `All q4 1 c65536` (15.06 GiB) | `All q8 0 c32768` (15.01 GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 3,212.65 | 3,211.94 | 3,203.00 | 3,199.48 | 3,220.79 | 3,216.36 |
| 7,168 | — | — | 2,843.67 | — | — | — |
| 8,192 | 2,862.24 | 2,857.17 | — | 2,842.99 | 2,860.33 | 2,856.73 |
| 15,360 | — | 2,682.90 | — | — | — | — |
| 16,384 | 2,676.50 | — | — | 2,657.58 | 2,668.92 | 2,660.37 |
| 19,456 | 2,610.84 | — | — | — | — | — |
| 31,744 | — | — | — | — | — | 2,413.47 |
| 32,768 | — | — | — | 2,402.06 | 2,398.53 | — |
| 49,152 | — | — | — | 2,195.69 | 2,187.81 | — |
| 64,512 | — | — | — | — | 2,016.48 | — |
| 65,536 | — | — | — | 2,360.24 | — | — |
| 68,608 | — | — | — | 2,364.56 | — | — |

#### Decode `tg128` (tok/s)

| Depth | `23 q4 0 c20480` (15.07 GiB) | `23 q4 1 c16384` (15.05 GiB) | `23 q8 0 c8192` (15.08 GiB) | `All q4 0 c69632` (15.03 GiB) | `All q4 1 c65536` (15.06 GiB) | `All q8 0 c32768` (15.01 GiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 97.43 | 97.47 | 96.77 | 97.94 | 97.94 | 97.02 |
| 7,168 | — | — | 93.09 | — | — | — |
| 8,192 | 94.94 | 94.64 | — | 95.46 | 95.12 | 92.93 |
| 15,360 | — | 92.93 | — | — | — | — |
| 16,384 | 93.24 | — | — | 93.77 | 93.39 | 90.59 |
| 19,456 | 93.49 | — | — | — | — | — |
| 31,744 | — | — | — | — | — | 90.32 |
| 32,768 | — | — | — | 93.49 | 93.20 | — |
| 49,152 | — | — | — | 93.31 | 93.02 | — |
| 64,512 | — | — | — | — | 92.84 | — |
| 65,536 | — | — | — | 93.38 | — | — |
| 68,608 | — | — | — | 93.48 | — | — |

### Vision matrix (`--mmproj`, no MTP, `--slots 1`)

| Config | Engine VRAM | Text tok/s | 1080p | 1440p | 4K 3840×2160 | 2560×2560 | Post-image text tok/s | min free (MB) | GTT rise (MB) |
|---|---:|---:|---|---|---|---|---:|---:|---:|
| `vis_All_q4_0_c20480` | 14.59 GiB | 98.29 | PASS 96.94 tok/s (TTFT 838 ms) | `HTTP_400` | `HTTP_400` | `HTTP_400` | 98.19 | 296 | 514 |
| `vis_All_q4_0_c8192` | 14.48 GiB | 98.18 | PASS 96.82 tok/s (TTFT 844 ms) | PASS 96.65 tok/s (TTFT 1725 ms) | `HTTP_400` | `HTTP_400` | 98.24 | 292 | 514 |
| `vis_All_q4_1_c16384` | 14.58 GiB | 98.26 | PASS 96.77 tok/s (TTFT 841 ms) | `HTTP_400` | `HTTP_400` | `HTTP_400` | 98.33 | 308 | 514 |
| `vis_All_q4_1_c8192` | 14.5 GiB | 98.35 | PASS 96.83 tok/s (TTFT 836 ms) | PASS 96.57 tok/s (TTFT 1723 ms) | `HTTP_400` | `HTTP_400` | 98.33 | 275 | 514 |
| `vis_All_q4_0_c4096` | 14.45 GiB | 98.31 | PASS 96.85 tok/s (TTFT 842 ms) | PASS 96.66 tok/s (TTFT 1738 ms) | `HTTP_400` | `HTTP_400` | 98.23 | 330 | 514 |
| `vis_All_q4_0_c4096_nographs` | 14.45 GiB | 98.16 | — | — | `HTTP_400` | `HTTP_400` | 98.21 | 553 | 510 |
| `vis_23_q4_0_c2048` | 14.9 GiB | None | — | — | — | — | None | 344 | 510 |

### BetterBench single-stream decode (median tok/s, 5 runs/category, greedy, cold nonce)

| Category | weight | `23_s1_q4_0_c20480` | `23_s2_q4_1_c12288` | `All_s1_q4_0_c69632` | `All_s8_q4_1_c12288` |
|---|---:|---:|---:|---:|---:|
| `code` | 0.3 | 97.2 | 97.4 | 97.5 | 98.0 |
| `reasoning` | 0.2 | 97.6 | 97.6 | 98.0 | 98.0 |
| `prose` | 0.15 | 97.7 | 97.7 | 98.2 | 98.1 |
| `json` | 0.15 | 99.0 | 98.5 | 98.5 | 99.3 |
| `summarization` | 0.1 | 98.5 | 98.6 | 98.9 | 99.0 |
| `file_edit` | 0.1 | 97.3 | 97.6 | 97.8 | 98.2 |
| `math` | 0 | 98.1 | 98.0 | 98.1 | 98.4 |
| `chat` | 0 | 97.9 | 97.9 | 98.3 | 98.9 |
| **Weighted** | | **97.8** | **97.8** | **98.0** | **98.3** |
| TTFT median (ms, all cats) | | 90 | 90 | 89 | 89 |

### BetterBench prefill ladder (median)

`23_s1_q4_0_c20480`: 2,063 tok → **2,693 tok/s** (0.77 s) · 8,012 tok → **2,795 tok/s** (2.87 s) · 14,417 tok → **2,747 tok/s** (5.25 s)

`23_s2_q4_1_c12288`: 2,066 tok → **2,682 tok/s** (0.77 s) · 8,086 tok → **2,800 tok/s** (2.89 s)

`All_s1_q4_0_c69632`: 2,050 tok → **2,735 tok/s** (0.75 s) · 8,096 tok → **2,820 tok/s** (2.87 s) · 16,132 tok → **2,750 tok/s** (5.87 s) · 32,132 tok → **2,630 tok/s** (12.22 s) · 53,112 tok → **2,471 tok/s** (21.49 s)

`All_s8_q4_1_c12288`: 2,046 tok → **2,763 tok/s** (0.74 s) · 8,031 tok → **2,807 tok/s** (2.86 s)

### BetterBench concurrency

| Run | c | OK | Aggregate tok/s | Per-stream median tok/s | TTFT p50 (ms) |
|---|--:|--:|--:|--:|--:|
| `23_s1_q4_0_c20480` | 1 | 32/32 | 95.5 | 97.9 | 91 |
| `23_s2_q4_1_c12288` | 1 | 32/32 | 95.5 | 97.9 | 91 |
| `23_s2_q4_1_c12288` | 2 | 32/32 | 130.3 | 67.2 | 110 |
| `All_s1_q4_0_c69632` | 1 | 32/32 | 95.7 | 98.0 | 90 |
| `All_s4_q4_0_c45056` | 1 | 32/32 | 96.1 | 98.3 | 90 |
| `All_s4_q4_0_c45056` | 2 | 32/32 | 130.6 | 67.4 | 110 |
| `All_s4_q4_0_c45056` | 4 | 32/32 | 159.4 | 41.2 | 139 |
| `All_s8_q4_1_c12288` | 1 | 32/32 | 95.8 | 98.1 | 91 |
| `All_s8_q4_1_c12288` | 2 | 32/32 | 131.4 | 67.7 | 111 |
| `All_s8_q4_1_c12288` | 4 | 32/32 | 159.4 | 41.5 | 140 |
| `All_s8_q4_1_c12288` | 8 | 32/32 | 174.9 | 22.9 | 203 |

### BetterBench amd-smi (1 s sampling across whole run)

| Run | max USED_VRAM | min FREE_VRAM | max USED_GTT | batch graphs |
|---|--:|--:|--:|--:|
| `23_s1_q4_0_c20480` | 16062 MB | 306 MB | 526 MB | 0 |
| `23_s2_q4_1_c12288` | 16092 MB | 276 MB | 526 MB | 1 |
| `All_s1_q4_0_c69632` | 16068 MB | 300 MB | 532 MB | 0 |
| `All_s4_q4_0_c45056` | 16069 MB | 299 MB | 529 MB | 9 |
| `All_s8_q4_1_c12288` | 16086 MB | 282 MB | 539 MB | 34 |
