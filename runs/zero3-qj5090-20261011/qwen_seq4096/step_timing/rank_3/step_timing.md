# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 580.366

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.562236 | 33.319400 | 46.885747 | 89.949708 | 364.292 |
| 2 | 1 | 9.142317 | 35.898057 | 11.238358 | 56.453630 | 580.441 |
| 3 | 1 | 9.120135 | 37.903070 | 11.250799 | 58.435611 | 560.754 |
| 4 | 1 | 8.757305 | 37.116289 | 8.450928 | 54.486287 | 601.399 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 8.938720 | 37.509679 | 9.850864 | 56.460949 |
