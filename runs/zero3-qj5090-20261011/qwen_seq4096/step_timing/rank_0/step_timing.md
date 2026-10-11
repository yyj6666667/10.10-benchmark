# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 605.951

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.563319 | 33.318363 | 39.396565 | 82.376723 | 397.782 |
| 2 | 1 | 9.141675 | 35.882033 | 9.211743 | 54.341723 | 602.999 |
| 3 | 1 | 9.119966 | 37.846875 | 8.728264 | 55.813233 | 587.101 |
| 4 | 1 | 8.758071 | 37.105814 | 6.352582 | 52.340639 | 626.053 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 8.939018 | 37.476344 | 7.540423 | 54.076936 |
