# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 576.762

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.562544 | 33.318389 | 46.800011 | 89.813214 | 364.846 |
| 2 | 1 | 9.142255 | 35.884897 | 11.247682 | 56.376533 | 581.235 |
| 3 | 1 | 9.121774 | 37.845720 | 11.408614 | 58.471969 | 560.405 |
| 4 | 1 | 8.760185 | 37.107540 | 9.192621 | 55.155476 | 594.102 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 8.940980 | 37.476630 | 10.300617 | 56.813722 |
