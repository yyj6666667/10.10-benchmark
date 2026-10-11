# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 431.099

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.813253 | 22.124889 | 12.384785 | 41.359395 | 396.137 |
| 2 | 1 | 6.336076 | 25.178984 | 13.521672 | 45.049313 | 363.690 |
| 3 | 1 | 6.304623 | 22.357966 | 8.232769 | 36.905219 | 443.948 |
| 4 | 1 | 6.331170 | 25.701148 | 7.060894 | 39.105251 | 418.972 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.317897 | 24.029557 | 7.646832 | 38.005235 |
