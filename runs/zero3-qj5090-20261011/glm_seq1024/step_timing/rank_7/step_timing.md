# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 200.903

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.546436 | 20.652698 | 10.475905 | 37.710237 | 217.235 |
| 2 | 1 | 6.189828 | 26.028716 | 6.169074 | 38.399170 | 213.338 |
| 3 | 1 | 6.195733 | 24.403758 | 7.298045 | 37.911282 | 216.083 |
| 4 | 1 | 6.190570 | 29.758594 | 7.679455 | 43.640574 | 187.715 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.193152 | 27.081176 | 7.488750 | 40.775928 |
