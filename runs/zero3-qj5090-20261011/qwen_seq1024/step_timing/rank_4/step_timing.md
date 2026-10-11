# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 155.852

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.992708 | 34.281474 | 27.414187 | 69.748561 | 117.450 |
| 2 | 1 | 8.140439 | 38.940062 | 16.281541 | 63.434209 | 129.142 |
| 3 | 1 | 7.470062 | 35.649944 | 13.085911 | 56.274185 | 145.573 |
| 4 | 1 | 7.992674 | 31.021158 | 9.776502 | 48.851163 | 167.693 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.731368 | 33.335551 | 11.431206 | 52.562674 |
