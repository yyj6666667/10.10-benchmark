# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.592

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.510904 | 35.603014 | 38.309887 | 81.453540 | 3.143 |
| 2 | 1 | 7.179887 | 25.894365 | 6.325450 | 39.430783 | 6.492 |
| 3 | 1 | 7.684038 | 26.951203 | 5.195378 | 39.863794 | 6.422 |
| 4 | 1 | 7.106753 | 25.864349 | 4.793884 | 37.804769 | 6.772 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.395396 | 26.407776 | 4.994631 | 38.834281 |
