# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 29.584

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.492202 | 20.246267 | 8.799346 | 35.567548 | 28.790 |
| 2 | 1 | 6.076801 | 24.054685 | 2.343730 | 32.490828 | 31.517 |
| 3 | 1 | 6.069315 | 23.296117 | 6.371685 | 35.746038 | 28.647 |
| 4 | 1 | 6.074600 | 23.166662 | 4.227272 | 33.479813 | 30.586 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.071958 | 23.231389 | 5.299478 | 34.612926 |
