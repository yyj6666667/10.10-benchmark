# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 52.847

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.751754 | 28.066942 | 16.100403 | 51.959110 | 39.416 |
| 2 | 1 | 7.877511 | 28.290980 | 6.065067 | 42.284549 | 48.434 |
| 3 | 1 | 7.190996 | 27.236394 | 4.024409 | 38.498250 | 53.197 |
| 4 | 1 | 7.192073 | 26.861520 | 4.917621 | 39.008114 | 52.502 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.191535 | 27.048957 | 4.471015 | 38.753182 |
