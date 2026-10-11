# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 361.771

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8.625138 | 28.443047 | 33.450670 | 70.587417 | 232.109 |
| 2 | 1 | 7.617846 | 30.113585 | 8.060544 | 45.862340 | 357.243 |
| 3 | 1 | 8.003565 | 31.005533 | 6.291640 | 45.360095 | 361.199 |
| 4 | 1 | 7.978620 | 31.284210 | 5.891970 | 45.216519 | 362.345 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.991093 | 31.144871 | 6.091805 | 45.288307 |
