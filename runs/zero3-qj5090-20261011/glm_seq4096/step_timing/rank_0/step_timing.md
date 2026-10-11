# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 772.862

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.945945 | 26.752051 | 9.916425 | 43.650204 | 750.695 |
| 2 | 1 | 6.549146 | 28.076551 | 2.663076 | 37.302070 | 878.450 |
| 3 | 1 | 6.561615 | 29.337496 | 9.372641 | 45.293398 | 723.461 |
| 4 | 1 | 6.541751 | 26.843146 | 6.097586 | 39.503137 | 829.504 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.551683 | 28.090321 | 7.735114 | 42.398267 |
