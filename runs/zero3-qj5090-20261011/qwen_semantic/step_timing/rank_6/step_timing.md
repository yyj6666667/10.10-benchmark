# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 5.878

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.009457 | 27.426407 | 65.978655 | 102.453093 | 2.499 |
| 2 | 1 | 7.398227 | 38.233054 | 8.852083 | 54.531777 | 4.695 |
| 3 | 1 | 7.665160 | 27.980992 | 8.052522 | 43.733915 | 5.854 |
| 4 | 1 | 7.273812 | 28.536692 | 7.520008 | 43.363943 | 5.904 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.469486 | 28.258842 | 7.786265 | 43.548929 |
