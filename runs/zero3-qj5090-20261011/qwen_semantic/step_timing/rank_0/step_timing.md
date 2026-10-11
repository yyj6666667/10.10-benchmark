# Step phase timing

- Mode: `coarse_host_wall_no_cuda_sync`
- Separate profiler capture may perturb its selected step; do not use this run for throughput comparison.
- Stable steps: 2 (excluded warmup: 2)
- Stable TPS: 6.249

| Step | Microbatches | Forward (s) | Backward (s) | Optimizer (s) | Total (s) | TPS |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 9.010921 | 27.424302 | 48.874805 | 85.414184 | 2.997 |
| 2 | 1 | 7.399235 | 38.233991 | 7.550246 | 53.241286 | 4.808 |
| 3 | 1 | 7.665664 | 27.984220 | 5.163262 | 40.855184 | 6.266 |
| 4 | 1 | 7.275137 | 28.553700 | 5.213672 | 41.073101 | 6.233 |

## Stable means

| Forward (s) | Backward (s) | Optimizer (s) | Total (s) |
|---:|---:|---:|---:|
| 7.470401 | 28.268960 | 5.188467 | 40.964142 |
