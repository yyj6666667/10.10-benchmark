# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 424.252

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.811426 | 22.122366 | 8.869076 | 37.837891 | 433.005 |
| 2 | 1 | 6.336499 | 25.190144 | 3.382944 | 34.925890 | 469.108 |
| 3 | 1 | 6.301495 | 22.345189 | 8.751684 | 37.409916 | 437.959 |
| 4 | 1 | 6.331123 | 25.711147 | 7.773741 | 39.827252 | 411.377 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.316309 | 24.028168 | 8.262713 | 38.618584 |
