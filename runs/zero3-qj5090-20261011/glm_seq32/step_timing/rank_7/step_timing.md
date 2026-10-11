# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.126

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.370367 | 24.337408 | 12.164721 | 42.903517 | 5.967 |
| 2 | 1 | 6.048587 | 23.852353 | 5.139867 | 35.050319 | 7.304 |
| 3 | 1 | 6.019142 | 23.784121 | 9.143220 | 38.959191 | 6.571 |
| 4 | 1 | 6.022999 | 30.188350 | 8.395429 | 44.618445 | 5.738 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.021070 | 26.986235 | 8.769325 | 41.788818 |
