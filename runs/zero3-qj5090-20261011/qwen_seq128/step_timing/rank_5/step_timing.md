# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 24.972

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.711818 | 26.107932 | 21.285794 | 55.145705 | 18.569 |
| 2 | 1 | 7.414219 | 28.859617 | 10.024037 | 46.341744 | 22.097 |
| 3 | 1 | 7.267122 | 28.005922 | 5.651067 | 40.991386 | 24.981 |
| 4 | 1 | 7.268917 | 28.074137 | 5.621416 | 41.021650 | 24.962 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.268020 | 28.040029 | 5.636241 | 41.006518 |
