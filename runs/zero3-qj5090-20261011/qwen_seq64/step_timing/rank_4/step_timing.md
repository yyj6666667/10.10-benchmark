# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 12.241

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.659485 | 33.474961 | 48.944768 | 90.131885 | 5.681 |
| 2 | 1 | 7.343249 | 28.718167 | 8.953469 | 45.077885 | 11.358 |
| 3 | 1 | 7.866206 | 27.118309 | 7.512973 | 42.529881 | 12.039 |
| 4 | 1 | 7.254616 | 28.120599 | 5.712758 | 41.122347 | 12.451 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.560411 | 27.619454 | 6.612865 | 41.826114 |
