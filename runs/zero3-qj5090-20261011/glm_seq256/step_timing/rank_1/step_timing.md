# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 58.541

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.530190 | 23.909160 | 6.356762 | 36.827298 | 55.611 |
| 2 | 1 | 6.179393 | 24.937800 | 5.121822 | 36.249520 | 56.497 |
| 3 | 1 | 6.193916 | 22.696093 | 4.970433 | 33.871250 | 60.464 |
| 4 | 1 | 6.161998 | 25.608022 | 4.314462 | 36.096316 | 56.737 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.177957 | 24.152057 | 4.642447 | 34.983783 |
