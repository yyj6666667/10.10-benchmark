# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.668

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.363382 | 24.321846 | 7.423814 | 38.150673 | 6.710 |
| 2 | 1 | 6.046984 | 23.867196 | 3.528156 | 33.453312 | 7.652 |
| 3 | 1 | 6.021193 | 23.783326 | 5.178272 | 34.993848 | 7.316 |
| 4 | 1 | 6.024255 | 30.170580 | 5.589867 | 41.794787 | 6.125 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.022724 | 26.976953 | 5.384069 | 38.394317 |
