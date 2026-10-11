# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 51.224

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.751922 | 28.061373 | 42.891610 | 78.747001 | 26.007 |
| 2 | 1 | 7.877429 | 28.291918 | 5.458001 | 41.697224 | 49.116 |
| 3 | 1 | 7.190897 | 27.255677 | 5.584017 | 40.068962 | 51.112 |
| 4 | 1 | 7.190225 | 26.843215 | 5.826368 | 39.892835 | 51.338 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.190561 | 27.049446 | 5.705192 | 39.980899 |
