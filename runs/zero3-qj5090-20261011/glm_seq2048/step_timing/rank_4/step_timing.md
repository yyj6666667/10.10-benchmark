# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 464.114

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 6.813717 | 22.138890 | 5.910045 | 34.895452 | 469.517 |
| 2 | 1 | 6.336145 | 25.177614 | 2.168923 | 33.696136 | 486.228 |
| 3 | 1 | 6.301806 | 22.343550 | 4.697410 | 33.355301 | 491.196 |
| 4 | 1 | 6.330769 | 25.714704 | 5.186513 | 37.248033 | 439.862 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 6.316287 | 24.029127 | 4.941962 | 35.301667 |
