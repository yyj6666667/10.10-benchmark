# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 14.248

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.460341 | 20.300710 | 10.412295 | 37.203728 | 13.762 |
| 2 | 1 | 6.057506 | 23.591918 | 4.692690 | 34.352031 | 14.905 |
| 3 | 1 | 6.068533 | 20.865759 | 8.922226 | 35.866297 | 14.275 |
| 4 | 1 | 6.041614 | 24.347201 | 5.604939 | 36.004989 | 14.220 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.055073 | 22.606480 | 7.263583 | 35.935643 |
