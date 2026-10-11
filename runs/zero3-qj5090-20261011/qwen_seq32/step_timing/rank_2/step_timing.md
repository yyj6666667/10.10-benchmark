# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.546

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 7.510265 | 35.605862 | 46.275003 | 89.423039 | 2.863 |
| 2 | 1 | 7.183194 | 25.895258 | 6.153873 | 39.267289 | 6.519 |
| 3 | 1 | 7.683733 | 26.954792 | 4.901721 | 39.583032 | 6.467 |
| 4 | 1 | 7.106988 | 25.866978 | 5.624185 | 38.631815 | 6.627 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.395361 | 26.410885 | 5.262953 | 39.107423 |
