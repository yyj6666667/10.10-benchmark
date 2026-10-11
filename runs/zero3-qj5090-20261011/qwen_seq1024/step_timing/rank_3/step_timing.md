# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 163.256

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.992548 | 34.256410 | 18.457258 | 60.755196 | 134.836 |
| 2 | 1 | 8.141171 | 38.935901 | 15.015197 | 62.157194 | 131.795 |
| 3 | 1 | 7.467707 | 35.648121 | 9.453616 | 52.622013 | 155.676 |
| 4 | 1 | 7.991504 | 30.994868 | 8.697092 | 47.735424 | 171.613 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.729605 | 33.321495 | 9.075354 | 50.178718 |
