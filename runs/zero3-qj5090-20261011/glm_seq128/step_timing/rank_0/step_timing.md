# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 28.749

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.490004 | 20.230136 | 12.001244 | 38.755428 | 26.422 |
| 2 | 1 | 6.074823 | 24.072655 | 5.259732 | 35.420011 | 28.910 |
| 3 | 1 | 6.068219 | 23.295647 | 6.153658 | 35.530510 | 28.820 |
| 4 | 1 | 6.074476 | 23.175771 | 6.446334 | 35.706730 | 28.678 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.071347 | 23.235709 | 6.299996 | 35.618620 |
