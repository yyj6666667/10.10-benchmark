# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 450.609

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.812783 | 22.140700 | 10.815151 | 39.808052 | 411.575 |
| 2 | 1 | 6.337932 | 25.195552 | 9.798538 | 41.345621 | 396.269 |
| 3 | 1 | 6.302737 | 22.342835 | 6.018765 | 34.678563 | 472.453 |
| 4 | 1 | 6.331579 | 25.700557 | 5.995022 | 38.040781 | 430.696 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.317158 | 24.021696 | 6.006893 | 36.359672 |
