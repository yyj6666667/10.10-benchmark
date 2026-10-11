# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 13.982

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.458867 | 20.301888 | 11.898973 | 38.691336 | 13.233 |
| 2 | 1 | 6.058507 | 23.594896 | 2.918268 | 32.583925 | 15.713 |
| 3 | 1 | 6.068548 | 20.849084 | 8.943696 | 35.870775 | 14.273 |
| 4 | 1 | 6.042980 | 24.361748 | 6.950480 | 37.367170 | 13.702 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.055764 | 22.605416 | 7.947088 | 36.618973 |
