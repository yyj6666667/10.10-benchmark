# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 24.014

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.710493 | 26.110281 | 20.089247 | 53.945451 | 18.982 |
| 2 | 1 | 7.411774 | 28.846901 | 4.813133 | 41.117289 | 24.904 |
| 3 | 1 | 7.267027 | 28.005687 | 5.843805 | 41.186412 | 24.863 |
| 4 | 1 | 7.271973 | 28.076187 | 8.717460 | 44.095918 | 23.222 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.269500 | 28.040937 | 7.280633 | 42.641165 |
