# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 51.796

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.525074 | 23.907843 | 12.268900 | 42.736716 | 47.921 |
| 2 | 1 | 6.178789 | 24.951945 | 7.572179 | 38.714218 | 52.900 |
| 3 | 1 | 6.191691 | 22.696467 | 10.337086 | 39.234840 | 52.199 |
| 4 | 1 | 6.161534 | 25.607968 | 8.065691 | 39.844063 | 51.400 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.176613 | 24.152217 | 9.201388 | 39.539452 |
