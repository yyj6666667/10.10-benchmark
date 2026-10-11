# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 81.179

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.964412 | 25.910981 | 19.258739 | 53.187300 | 77.011 |
| 2 | 1 | 8.044579 | 31.920891 | 12.225143 | 52.230091 | 78.422 |
| 3 | 1 | 7.501212 | 35.833176 | 9.198604 | 52.580605 | 77.899 |
| 4 | 1 | 7.501595 | 30.435987 | 10.355271 | 48.332192 | 84.747 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.501403 | 33.134581 | 9.776937 | 50.456399 |
