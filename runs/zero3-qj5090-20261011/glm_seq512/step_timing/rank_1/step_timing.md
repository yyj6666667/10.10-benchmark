# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 98.661

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.476509 | 25.279848 | 12.933671 | 44.722166 | 91.588 |
| 2 | 1 | 6.114312 | 25.686111 | 9.147278 | 40.958787 | 100.003 |
| 3 | 1 | 6.080975 | 24.815801 | 10.089363 | 40.998020 | 99.907 |
| 4 | 1 | 6.089523 | 28.198053 | 7.734391 | 42.034054 | 97.445 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.085249 | 26.506927 | 8.911877 | 41.516037 |
