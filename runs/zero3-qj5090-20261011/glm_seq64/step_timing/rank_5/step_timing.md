# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 15.623

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.460744 | 20.284226 | 6.113643 | 32.896630 | 15.564 |
| 2 | 1 | 6.056689 | 23.588478 | 5.075673 | 34.732218 | 14.741 |
| 3 | 1 | 6.068194 | 20.860933 | 4.407567 | 31.347191 | 16.333 |
| 4 | 1 | 6.041016 | 24.360631 | 3.776474 | 34.195367 | 14.973 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.054605 | 22.610782 | 4.092021 | 32.771279 |
