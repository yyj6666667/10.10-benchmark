# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 24.412

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.712178 | 26.109321 | 20.594782 | 54.448958 | 18.807 |
| 2 | 1 | 7.411637 | 28.846876 | 3.806492 | 40.106322 | 25.532 |
| 3 | 1 | 7.266051 | 28.008374 | 6.904719 | 42.222670 | 24.252 |
| 4 | 1 | 7.270574 | 28.060171 | 6.303159 | 41.671628 | 24.573 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.268313 | 28.034272 | 6.603939 | 41.947149 |
