# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 102.448

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.476697 | 25.278787 | 14.279248 | 46.068757 | 88.911 |
| 2 | 1 | 6.113517 | 25.685185 | 12.714815 | 44.524508 | 91.994 |
| 3 | 1 | 6.080975 | 24.815280 | 7.519666 | 38.427534 | 106.590 |
| 4 | 1 | 6.089070 | 28.197901 | 7.236330 | 41.535081 | 98.615 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.085022 | 26.506590 | 7.377998 | 39.981307 |
