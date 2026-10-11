# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 813.049

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.945668 | 26.743336 | 6.387084 | 40.117689 | 816.797 |
| 2 | 1 | 6.547782 | 28.103409 | 5.849917 | 40.518399 | 808.719 |
| 3 | 1 | 6.565521 | 29.330028 | 5.905343 | 41.817488 | 783.596 |
| 4 | 1 | 6.543698 | 26.869133 | 5.360120 | 38.787738 | 844.803 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.554609 | 28.099580 | 5.632731 | 40.302613 |
