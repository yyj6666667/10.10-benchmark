# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.423

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.343191 | 32.291156 | 12.290505 | 51.982350 | 4.925 |
| 2 | 1 | 6.030766 | 31.421177 | 10.325016 | 47.787959 | 5.357 |
| 3 | 1 | 6.109771 | 24.980735 | 8.785843 | 39.887508 | 6.418 |
| 4 | 1 | 5.998870 | 23.949984 | 9.863747 | 39.823577 | 6.428 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.054320 | 24.465360 | 9.324795 | 39.855542 |
