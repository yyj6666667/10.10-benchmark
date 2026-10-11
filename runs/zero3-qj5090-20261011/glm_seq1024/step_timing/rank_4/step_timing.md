# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 205.262

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.546356 | 20.647576 | 12.519432 | 39.745807 | 206.110 |
| 2 | 1 | 6.191396 | 26.008275 | 2.107667 | 34.320264 | 238.693 |
| 3 | 1 | 6.193951 | 24.383490 | 5.749363 | 36.340006 | 225.426 |
| 4 | 1 | 6.191958 | 29.771393 | 7.504531 | 43.479861 | 188.409 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.192955 | 27.077442 | 6.626947 | 39.909933 |
