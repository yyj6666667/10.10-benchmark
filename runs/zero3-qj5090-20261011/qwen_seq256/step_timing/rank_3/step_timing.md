# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 48.039

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.738505 | 28.103771 | 46.844486 | 82.741010 | 24.752 |
| 2 | 1 | 7.878777 | 28.293266 | 8.730877 | 44.957892 | 45.554 |
| 3 | 1 | 7.193464 | 27.256530 | 8.053770 | 42.541533 | 48.141 |
| 4 | 1 | 7.195728 | 26.843239 | 8.645524 | 42.722193 | 47.938 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.194596 | 27.049884 | 8.349647 | 42.631863 |
