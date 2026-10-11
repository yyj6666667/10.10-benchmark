# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 24.500

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.711176 | 26.103792 | 19.275941 | 53.123926 | 19.276 |
| 2 | 1 | 7.411159 | 28.839355 | 7.585938 | 43.867893 | 23.343 |
| 3 | 1 | 7.265379 | 28.021456 | 6.512747 | 41.851631 | 24.467 |
| 4 | 1 | 7.270235 | 28.053650 | 6.382796 | 41.739329 | 24.533 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.267807 | 28.037553 | 6.447771 | 41.795480 |
