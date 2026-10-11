# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 166.298

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.994328 | 34.236124 | 14.836309 | 57.129564 | 143.393 |
| 2 | 1 | 8.140418 | 38.935364 | 13.088702 | 60.245166 | 135.978 |
| 3 | 1 | 7.469897 | 35.645551 | 9.053267 | 52.219174 | 156.877 |
| 4 | 1 | 7.993348 | 31.014363 | 7.241166 | 46.302839 | 176.922 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.731622 | 33.329957 | 8.147216 | 49.261006 |
