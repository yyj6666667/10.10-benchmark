# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 12.705

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.658873 | 33.450264 | 24.048121 | 65.642960 | 7.800 |
| 2 | 1 | 7.341789 | 28.653133 | 6.386862 | 42.428466 | 12.067 |
| 3 | 1 | 7.866053 | 27.130645 | 4.248931 | 39.277274 | 13.036 |
| 4 | 1 | 7.254458 | 28.113009 | 5.927942 | 41.323631 | 12.390 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.560255 | 27.621827 | 5.088437 | 40.300453 |
