# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 24.545

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.709813 | 26.110631 | 17.162201 | 51.021773 | 20.070 |
| 2 | 1 | 7.411199 | 28.846787 | 3.771266 | 40.075023 | 25.552 |
| 3 | 1 | 7.264100 | 28.008783 | 6.477982 | 41.793790 | 24.501 |
| 4 | 1 | 7.270517 | 28.059908 | 6.269117 | 41.643117 | 24.590 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.267309 | 28.034345 | 6.373550 | 41.718454 |
