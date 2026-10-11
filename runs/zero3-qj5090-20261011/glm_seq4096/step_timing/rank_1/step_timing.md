# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 792.574

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.946028 | 26.759697 | 9.800366 | 43.544350 | 752.520 |
| 2 | 1 | 6.547580 | 28.082343 | 5.019904 | 39.665812 | 826.102 |
| 3 | 1 | 6.562054 | 29.338592 | 6.725875 | 42.646938 | 768.355 |
| 4 | 1 | 6.541486 | 26.863101 | 6.619446 | 40.040586 | 818.370 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.551770 | 28.100846 | 6.672661 | 41.343762 |
