# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.709

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.369404 | 24.332654 | 7.697620 | 38.434093 | 6.661 |
| 2 | 1 | 6.047365 | 23.848724 | 4.244434 | 34.152400 | 7.496 |
| 3 | 1 | 6.020365 | 23.797984 | 4.355593 | 34.190282 | 7.488 |
| 4 | 1 | 6.022517 | 30.168826 | 5.923008 | 42.124948 | 6.077 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.021441 | 26.983405 | 5.139300 | 38.157615 |
