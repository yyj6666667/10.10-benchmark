# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 765.140

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.945246 | 26.744107 | 12.815369 | 46.543960 | 704.023 |
| 2 | 1 | 6.548736 | 28.102025 | 16.946780 | 51.613243 | 634.876 |
| 3 | 1 | 6.566533 | 29.332081 | 8.364159 | 44.275894 | 740.087 |
| 4 | 1 | 6.544640 | 26.848965 | 7.969777 | 41.376375 | 791.950 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.555587 | 28.090523 | 8.166968 | 42.826134 |
