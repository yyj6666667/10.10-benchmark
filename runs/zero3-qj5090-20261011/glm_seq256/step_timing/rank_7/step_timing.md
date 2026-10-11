# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 55.645

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.528474 | 23.908090 | 9.061170 | 39.531609 | 51.807 |
| 2 | 1 | 6.178120 | 24.954151 | 5.147938 | 36.290564 | 56.433 |
| 3 | 1 | 6.190163 | 22.695060 | 6.391287 | 35.288637 | 58.036 |
| 4 | 1 | 6.160774 | 25.590760 | 6.559086 | 38.321260 | 53.443 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.175469 | 24.142910 | 6.475186 | 36.804949 |
