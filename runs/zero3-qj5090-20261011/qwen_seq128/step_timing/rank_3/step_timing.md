# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 24.387

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.711098 | 26.082406 | 18.461533 | 52.299693 | 19.579 |
| 2 | 1 | 7.410995 | 28.859018 | 5.248376 | 41.552382 | 24.644 |
| 3 | 1 | 7.265347 | 28.019004 | 6.367571 | 41.695505 | 24.559 |
| 4 | 1 | 7.271427 | 28.050719 | 6.928167 | 42.282240 | 24.218 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.268387 | 28.034861 | 6.647869 | 41.988873 |
