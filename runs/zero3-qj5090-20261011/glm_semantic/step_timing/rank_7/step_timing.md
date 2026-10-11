# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 7.259

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.372516 | 32.290582 | 6.220364 | 45.916300 | 5.575 |
| 2 | 1 | 6.030364 | 31.393894 | 4.687488 | 42.121970 | 6.078 |
| 3 | 1 | 6.107874 | 24.974744 | 3.683488 | 34.777674 | 7.361 |
| 4 | 1 | 5.998269 | 23.944953 | 5.795648 | 35.752313 | 7.160 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.053071 | 24.459849 | 4.739568 | 35.264994 |
