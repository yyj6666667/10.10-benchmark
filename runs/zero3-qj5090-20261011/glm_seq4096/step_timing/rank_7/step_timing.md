# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 758.097

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.943831 | 26.759415 | 9.893943 | 43.638217 | 750.901 |
| 2 | 1 | 6.547057 | 28.165250 | 11.895779 | 46.637625 | 702.609 |
| 3 | 1 | 6.564224 | 29.397239 | 9.203735 | 45.185089 | 725.195 |
| 4 | 1 | 6.544679 | 26.865721 | 7.839268 | 41.262986 | 794.126 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.554451 | 28.131480 | 8.521502 | 43.224037 |
