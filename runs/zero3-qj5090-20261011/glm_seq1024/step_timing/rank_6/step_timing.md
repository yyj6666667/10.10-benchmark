# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 214.412

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.545689 | 20.633186 | 6.003083 | 33.227277 | 246.544 |
| 2 | 1 | 6.191132 | 26.037452 | 4.297464 | 36.537658 | 224.207 |
| 3 | 1 | 6.195797 | 24.384154 | 4.677431 | 35.269802 | 232.267 |
| 4 | 1 | 6.192879 | 29.768247 | 5.169812 | 41.143787 | 199.107 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.194338 | 27.076200 | 4.923621 | 38.206794 |
