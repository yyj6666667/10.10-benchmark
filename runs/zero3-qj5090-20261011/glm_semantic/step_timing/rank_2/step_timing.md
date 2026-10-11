# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.326

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.361902 | 32.291148 | 13.935328 | 53.631058 | 4.773 |
| 2 | 1 | 6.031162 | 31.395244 | 4.387934 | 41.825472 | 6.121 |
| 3 | 1 | 6.106970 | 24.962701 | 10.276377 | 41.356434 | 6.190 |
| 4 | 1 | 5.997420 | 23.957319 | 9.608480 | 39.576593 | 6.468 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.052195 | 24.460010 | 9.942428 | 40.466514 |
