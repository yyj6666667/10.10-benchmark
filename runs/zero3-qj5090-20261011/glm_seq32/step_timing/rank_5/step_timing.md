# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.090

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.369734 | 24.321765 | 11.042002 | 41.770561 | 6.129 |
| 2 | 1 | 6.046576 | 23.863236 | 11.957231 | 41.879556 | 6.113 |
| 3 | 1 | 6.023704 | 23.856329 | 9.673863 | 39.563434 | 6.471 |
| 4 | 1 | 6.021588 | 30.227195 | 8.256055 | 44.515518 | 5.751 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.022646 | 27.041762 | 8.964959 | 42.039476 |
