# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.185

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.516162 | 35.598786 | 40.806589 | 83.950434 | 3.049 |
| 2 | 1 | 7.181463 | 25.874388 | 3.160320 | 36.254411 | 7.061 |
| 3 | 1 | 7.681123 | 26.951260 | 7.376224 | 42.038367 | 6.090 |
| 4 | 1 | 7.107596 | 25.846545 | 7.763181 | 40.742459 | 6.283 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.394359 | 26.398902 | 7.569702 | 41.390413 |
