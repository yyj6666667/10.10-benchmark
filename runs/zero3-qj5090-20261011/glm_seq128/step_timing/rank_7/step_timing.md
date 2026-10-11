# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 30.038

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.491664 | 20.241740 | 6.028023 | 32.801014 | 31.219 |
| 2 | 1 | 6.073785 | 24.053894 | 3.194864 | 33.335380 | 30.718 |
| 3 | 1 | 6.067927 | 23.291528 | 4.333541 | 33.703809 | 30.382 |
| 4 | 1 | 6.073461 | 23.162900 | 5.230127 | 34.477576 | 29.700 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.070694 | 23.227214 | 4.781834 | 34.090692 |
