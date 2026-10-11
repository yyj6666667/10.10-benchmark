# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 94.460

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.473229 | 25.279820 | 15.538901 | 47.322702 | 86.555 |
| 2 | 1 | 6.114250 | 25.686436 | 3.734516 | 35.544679 | 115.235 |
| 3 | 1 | 6.078181 | 24.812094 | 13.784718 | 44.689048 | 91.656 |
| 4 | 1 | 6.090779 | 28.197525 | 7.737073 | 42.035049 | 97.442 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.084480 | 26.504809 | 10.760895 | 43.362048 |
