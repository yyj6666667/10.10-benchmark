# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 86.803

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.964978 | 25.890764 | 18.081910 | 52.037530 | 78.712 |
| 2 | 1 | 8.042937 | 31.921626 | 7.837415 | 47.876424 | 85.554 |
| 3 | 1 | 7.498878 | 35.835865 | 6.428311 | 49.838366 | 82.186 |
| 4 | 1 | 7.501170 | 30.455189 | 6.518465 | 44.535973 | 91.971 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.500024 | 33.145527 | 6.473388 | 47.187170 |
