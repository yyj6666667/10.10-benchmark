# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 15.674

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.446961 | 20.286688 | 7.499541 | 34.277166 | 14.937 |
| 2 | 1 | 6.056380 | 23.592710 | 5.116448 | 34.775917 | 14.723 |
| 3 | 1 | 6.068881 | 20.865687 | 4.003023 | 30.947605 | 16.544 |
| 4 | 1 | 6.041174 | 24.345764 | 3.986613 | 34.383851 | 14.891 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.055027 | 22.605726 | 3.994818 | 32.665728 |
