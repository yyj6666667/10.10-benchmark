# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 343.967

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8.624649 | 28.437128 | 49.276596 | 86.420216 | 189.585 |
| 2 | 1 | 7.617187 | 30.111992 | 9.115784 | 46.911819 | 349.251 |
| 3 | 1 | 8.003813 | 31.020916 | 8.516751 | 47.614025 | 344.100 |
| 4 | 1 | 7.978653 | 31.262474 | 8.345766 | 47.650810 | 343.835 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.991233 | 31.141695 | 8.431258 | 47.632418 |
