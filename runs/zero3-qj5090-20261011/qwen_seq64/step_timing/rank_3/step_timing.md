# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 12.185

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.659034 | 33.475563 | 35.986253 | 77.157083 | 6.636 |
| 2 | 1 | 7.341263 | 28.654556 | 9.595291 | 45.639374 | 11.218 |
| 3 | 1 | 7.869468 | 27.116999 | 8.301412 | 43.319854 | 11.819 |
| 4 | 1 | 7.257101 | 28.114678 | 5.288138 | 40.715262 | 12.575 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.563284 | 27.615839 | 6.794775 | 42.017558 |
