# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.481

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.369145 | 32.292644 | 14.511253 | 54.203748 | 4.723 |
| 2 | 1 | 6.031231 | 31.392033 | 10.334332 | 47.767195 | 5.359 |
| 3 | 1 | 6.038590 | 24.976973 | 7.547681 | 38.647383 | 6.624 |
| 4 | 1 | 5.999275 | 23.943986 | 10.396650 | 40.352595 | 6.344 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.018932 | 24.460479 | 8.972165 | 39.499989 |
