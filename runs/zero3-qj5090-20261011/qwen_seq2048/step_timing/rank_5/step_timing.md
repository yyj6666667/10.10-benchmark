# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 362.450

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8.626664 | 28.444667 | 33.942875 | 71.081822 | 230.495 |
| 2 | 1 | 7.613629 | 30.120223 | 6.735583 | 44.555238 | 367.723 |
| 3 | 1 | 8.004588 | 31.029200 | 6.289077 | 45.397916 | 360.898 |
| 4 | 1 | 7.978833 | 31.269283 | 5.673465 | 45.009085 | 364.015 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.991711 | 31.149241 | 5.981271 | 45.203501 |
