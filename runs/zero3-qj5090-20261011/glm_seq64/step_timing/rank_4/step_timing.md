# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 15.694

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.460295 | 20.284773 | 5.888605 | 32.671404 | 15.671 |
| 2 | 1 | 6.057722 | 23.608986 | 4.822957 | 34.499524 | 14.841 |
| 3 | 1 | 6.069378 | 20.848528 | 4.102798 | 31.029306 | 16.501 |
| 4 | 1 | 6.040825 | 24.361850 | 3.801103 | 34.217290 | 14.963 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.055101 | 22.605189 | 3.951951 | 32.623298 |
