# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 99.382

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.477243 | 25.258818 | 9.055250 | 40.824365 | 100.332 |
| 2 | 1 | 6.114104 | 25.698950 | 3.914066 | 35.739128 | 114.608 |
| 3 | 1 | 6.080266 | 24.812321 | 8.188682 | 39.094567 | 104.772 |
| 4 | 1 | 6.088595 | 28.180979 | 9.055488 | 43.334783 | 94.520 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.084431 | 26.496650 | 8.622085 | 41.214675 |
