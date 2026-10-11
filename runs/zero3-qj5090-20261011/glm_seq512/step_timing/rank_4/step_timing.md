# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 103.619

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.479070 | 25.259289 | 10.924253 | 42.695957 | 95.934 |
| 2 | 1 | 6.113836 | 25.702775 | 11.022338 | 42.851264 | 95.586 |
| 3 | 1 | 6.080253 | 24.801474 | 7.833954 | 38.726747 | 105.767 |
| 4 | 1 | 6.088918 | 28.198281 | 6.034415 | 40.332257 | 101.556 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.084586 | 26.499877 | 6.934185 | 39.529502 |
