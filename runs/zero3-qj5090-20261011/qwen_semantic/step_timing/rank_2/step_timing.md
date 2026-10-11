# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.228

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.011390 | 27.424957 | 54.638752 | 91.115285 | 2.810 |
| 2 | 1 | 7.398391 | 38.234390 | 7.745244 | 53.426715 | 4.792 |
| 3 | 1 | 7.666259 | 27.960277 | 5.758309 | 41.419518 | 6.181 |
| 4 | 1 | 7.275600 | 28.551884 | 4.923383 | 40.790661 | 6.276 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.470929 | 28.256081 | 5.340846 | 41.105089 |
