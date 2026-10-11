# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 54.346

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.528200 | 23.925936 | 10.570344 | 41.055797 | 49.883 |
| 2 | 1 | 6.177714 | 24.939942 | 6.523294 | 37.651675 | 54.393 |
| 3 | 1 | 6.190596 | 22.712420 | 7.958988 | 36.874485 | 55.540 |
| 4 | 1 | 6.159919 | 25.593503 | 6.730744 | 38.494109 | 53.203 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.175258 | 24.152962 | 7.344866 | 37.684297 |
