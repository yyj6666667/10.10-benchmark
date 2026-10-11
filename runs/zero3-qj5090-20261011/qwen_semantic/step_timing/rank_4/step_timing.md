# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 5.858

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.014188 | 27.421944 | 66.779549 | 103.256995 | 2.479 |
| 2 | 1 | 7.398791 | 38.230205 | 10.173732 | 55.852545 | 4.583 |
| 3 | 1 | 7.661607 | 27.981585 | 8.192997 | 43.865491 | 5.836 |
| 4 | 1 | 7.274306 | 28.530645 | 7.704298 | 43.534636 | 5.880 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.467957 | 28.256115 | 7.948648 | 43.700064 |
