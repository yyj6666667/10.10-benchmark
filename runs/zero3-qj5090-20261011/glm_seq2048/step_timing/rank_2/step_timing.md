# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 467.820

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.812420 | 22.121970 | 6.021495 | 34.991604 | 468.227 |
| 2 | 1 | 6.336513 | 25.191174 | 3.856814 | 35.399799 | 462.827 |
| 3 | 1 | 6.302640 | 22.338773 | 4.441034 | 33.094499 | 495.067 |
| 4 | 1 | 6.330768 | 25.712650 | 4.893111 | 36.949567 | 443.415 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.316704 | 24.025711 | 4.667073 | 35.022033 |
