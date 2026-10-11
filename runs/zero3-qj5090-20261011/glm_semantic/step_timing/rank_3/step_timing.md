# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.838

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.371920 | 32.277406 | 11.558339 | 51.241408 | 4.996 |
| 2 | 1 | 6.031142 | 31.409116 | 8.361288 | 45.815777 | 5.588 |
| 3 | 1 | 6.109053 | 24.964551 | 6.613804 | 37.698943 | 6.791 |
| 4 | 1 | 5.998778 | 23.945545 | 7.223349 | 37.181908 | 6.885 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.053916 | 24.455048 | 6.918576 | 37.440426 |
