# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 51.885

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.752360 | 28.085793 | 41.508760 | 77.378612 | 26.467 |
| 2 | 1 | 7.877896 | 28.291686 | 6.259165 | 42.473965 | 48.218 |
| 3 | 1 | 7.190722 | 27.238554 | 4.571401 | 39.056818 | 52.436 |
| 4 | 1 | 7.192023 | 26.861280 | 5.796051 | 39.887309 | 51.345 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.191373 | 27.049917 | 5.183726 | 39.472063 |
