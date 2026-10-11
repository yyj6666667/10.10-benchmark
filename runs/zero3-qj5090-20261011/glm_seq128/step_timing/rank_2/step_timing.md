# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 30.109

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.491706 | 20.241146 | 8.216124 | 34.980687 | 29.273 |
| 2 | 1 | 6.073841 | 24.053407 | 4.620459 | 34.759375 | 29.460 |
| 3 | 1 | 6.067461 | 23.306679 | 4.577862 | 33.961626 | 30.152 |
| 4 | 1 | 6.074013 | 23.161425 | 4.812026 | 34.058278 | 30.066 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.070737 | 23.234052 | 4.694944 | 34.009952 |
