# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 50.939

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.751956 | 28.084192 | 43.067097 | 78.955812 | 25.939 |
| 2 | 1 | 7.871663 | 28.312675 | 7.704678 | 43.944600 | 46.604 |
| 3 | 1 | 7.191260 | 27.238635 | 6.507569 | 40.986195 | 49.968 |
| 4 | 1 | 7.191579 | 26.861991 | 5.288329 | 39.423181 | 51.949 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.191420 | 27.050313 | 5.897949 | 40.204688 |
