# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 24.533

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.710997 | 26.108588 | 18.681587 | 52.534221 | 19.492 |
| 2 | 1 | 7.411187 | 28.846635 | 3.988140 | 40.293131 | 25.414 |
| 3 | 1 | 7.264961 | 28.005812 | 6.265753 | 41.582081 | 24.626 |
| 4 | 1 | 7.270475 | 28.058637 | 6.533195 | 41.897235 | 24.441 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.267718 | 28.032224 | 6.399474 | 41.739658 |
