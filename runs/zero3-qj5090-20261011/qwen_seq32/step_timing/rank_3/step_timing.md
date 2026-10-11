# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.399

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.506868 | 35.616378 | 48.522724 | 91.693955 | 2.792 |
| 2 | 1 | 7.179286 | 25.920099 | 8.552873 | 41.699201 | 6.139 |
| 3 | 1 | 7.685742 | 26.954465 | 6.407078 | 41.077881 | 6.232 |
| 4 | 1 | 7.107020 | 25.866280 | 5.920588 | 38.931228 | 6.576 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.396381 | 26.410372 | 6.163833 | 40.004555 |
