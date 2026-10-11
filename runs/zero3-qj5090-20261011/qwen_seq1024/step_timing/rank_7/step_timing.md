# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 144.952

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.993424 | 34.267234 | 26.677367 | 68.990514 | 118.741 |
| 2 | 1 | 8.138895 | 38.960596 | 22.462625 | 69.656792 | 117.605 |
| 3 | 1 | 7.471888 | 35.774650 | 14.976975 | 58.288908 | 140.541 |
| 4 | 1 | 7.993899 | 31.006755 | 15.690776 | 54.741874 | 149.648 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.732894 | 33.390702 | 15.333875 | 56.515391 |
