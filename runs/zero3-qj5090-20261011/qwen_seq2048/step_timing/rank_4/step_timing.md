# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 364.498

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 8.626621 | 28.447396 | 37.352348 | 74.522428 | 219.853 |
| 2 | 1 | 7.613696 | 30.120270 | 7.115608 | 44.952258 | 364.476 |
| 3 | 1 | 8.003984 | 31.006439 | 6.152264 | 45.252305 | 362.059 |
| 4 | 1 | 7.978857 | 31.290491 | 5.303010 | 44.646732 | 366.970 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.991420 | 31.148465 | 5.727637 | 44.949519 |
