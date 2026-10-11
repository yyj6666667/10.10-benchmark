# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.136

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.011265 | 27.402138 | 65.294100 | 101.745483 | 2.516 |
| 2 | 1 | 7.397157 | 38.228733 | 7.792802 | 53.476686 | 4.787 |
| 3 | 1 | 7.663471 | 27.979312 | 6.068386 | 41.744887 | 6.132 |
| 4 | 1 | 7.271688 | 28.529770 | 5.860994 | 41.693670 | 6.140 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.467580 | 28.254541 | 5.964690 | 41.719279 |
