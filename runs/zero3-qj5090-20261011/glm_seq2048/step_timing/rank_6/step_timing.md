# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 438.009

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.811524 | 22.124772 | 12.667152 | 41.639564 | 393.472 |
| 2 | 1 | 6.337773 | 25.192557 | 10.341204 | 41.885314 | 391.163 |
| 3 | 1 | 6.301947 | 22.343767 | 7.000769 | 35.658962 | 459.464 |
| 4 | 1 | 6.331605 | 25.714587 | 7.095155 | 39.152263 | 418.469 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.316776 | 24.029177 | 7.047962 | 37.405612 |
