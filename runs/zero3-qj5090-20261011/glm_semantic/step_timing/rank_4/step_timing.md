# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 7.102

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.372648 | 32.289829 | 7.263964 | 46.956395 | 5.452 |
| 2 | 1 | 6.030367 | 31.393335 | 2.570520 | 40.006641 | 6.399 |
| 3 | 1 | 6.107972 | 24.961397 | 4.665670 | 35.745579 | 7.162 |
| 4 | 1 | 5.998211 | 23.943989 | 6.389225 | 36.344238 | 7.044 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.053092 | 24.452693 | 5.527448 | 36.044909 |
