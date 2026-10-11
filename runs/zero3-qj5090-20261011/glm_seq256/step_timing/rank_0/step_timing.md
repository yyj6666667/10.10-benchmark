# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 50.822

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.527015 | 23.924695 | 10.389843 | 40.874447 | 50.105 |
| 2 | 1 | 6.178458 | 24.938244 | 3.700882 | 34.830813 | 58.799 |
| 3 | 1 | 6.193939 | 22.696109 | 12.659878 | 41.559618 | 49.279 |
| 4 | 1 | 6.162221 | 25.592453 | 7.270934 | 39.034949 | 52.466 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.178080 | 24.144281 | 9.965406 | 40.297284 |
