# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 101.004

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.476114 | 25.274182 | 9.179190 | 40.960532 | 99.999 |
| 2 | 1 | 6.111381 | 25.683858 | 2.812342 | 34.617702 | 118.321 |
| 3 | 1 | 6.078893 | 24.796524 | 8.308992 | 39.196736 | 104.498 |
| 4 | 1 | 6.089254 | 28.194368 | 7.614744 | 41.909084 | 97.735 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.084073 | 26.495446 | 7.961868 | 40.552910 |
