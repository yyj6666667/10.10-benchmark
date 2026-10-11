# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 167.114

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.992651 | 34.261180 | 18.517128 | 60.834479 | 134.660 |
| 2 | 1 | 8.139186 | 38.939381 | 14.818282 | 61.968918 | 132.195 |
| 3 | 1 | 7.470739 | 35.647771 | 6.788878 | 49.956731 | 163.982 |
| 4 | 1 | 7.994653 | 31.017245 | 9.017920 | 48.084338 | 170.367 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.732696 | 33.332508 | 7.903399 | 49.020535 |
