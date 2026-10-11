# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 5.952

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.012932 | 27.442754 | 59.479262 | 95.972064 | 2.667 |
| 2 | 1 | 7.397306 | 38.236245 | 9.909306 | 55.594698 | 4.605 |
| 3 | 1 | 7.664180 | 27.981885 | 8.514701 | 44.200166 | 5.792 |
| 4 | 1 | 7.275037 | 28.532671 | 5.979632 | 41.824258 | 6.121 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.469609 | 28.257278 | 7.247167 | 43.012212 |
