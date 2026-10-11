# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 780.997

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.945654 | 26.760824 | 10.911231 | 44.656303 | 733.782 |
| 2 | 1 | 6.550087 | 28.065544 | 15.880519 | 50.510812 | 648.732 |
| 3 | 1 | 6.566037 | 29.329272 | 7.895795 | 43.806659 | 748.014 |
| 4 | 1 | 6.545650 | 26.863535 | 6.682554 | 40.106596 | 817.023 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.555843 | 28.096404 | 7.289174 | 41.956627 |
