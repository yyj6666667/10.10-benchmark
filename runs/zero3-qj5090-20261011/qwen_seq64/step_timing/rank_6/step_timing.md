# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 12.036

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.659594 | 33.475146 | 48.016027 | 89.182055 | 5.741 |
| 2 | 1 | 7.342667 | 28.656217 | 10.111919 | 46.153990 | 11.093 |
| 3 | 1 | 7.868211 | 27.117439 | 7.963882 | 42.979727 | 11.913 |
| 4 | 1 | 7.255367 | 28.115691 | 6.689687 | 42.100735 | 12.161 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.561789 | 27.616565 | 7.326784 | 42.540231 |
