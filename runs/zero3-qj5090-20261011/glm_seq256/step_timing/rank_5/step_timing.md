# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 51.399

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.529295 | 23.926039 | 12.637432 | 43.121677 | 47.494 |
| 2 | 1 | 6.178428 | 24.939628 | 3.457930 | 34.586211 | 59.214 |
| 3 | 1 | 6.193301 | 22.696448 | 12.370716 | 41.269907 | 49.625 |
| 4 | 1 | 6.161209 | 25.593358 | 6.654080 | 38.419657 | 53.306 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.177255 | 24.144903 | 9.512398 | 39.844782 |
