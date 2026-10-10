# GLM-4.5-Air seq4096 KT gradient clipping profile

qj5090, 8 GPUs, BF16 full fine-tuning, global batch 8, `max_grad_norm=1.0`.
Both diagnostics profiled the first clipping call and exited before the
optimizer update. They used zero data-loader workers to avoid forking the
large CPU model. These are clipping diagnostics, not step throughput or loss
checks.

| Rank 0 OpenMP threads | Clip host wall (s) | Fused BF16 sum of squares (s; 135 calls) | `aten::mul_` (s; 1,335 calls) | `aten::item` (ms; 1 call) | Trace |
|---:|---:|---:|---:|---:|---|
| 2 | 37.908 | 30.831 | 5.594 | 0.113 | [Chrome trace](./glm_clip_profile_seq4096_8gpu_20261010_124047/profile/rank_0_clip.trace.json) |
| 64 | 4.303 | 1.382 | 1.247 | 0.131 | [Chrome trace](./glm_clip_profile_seq4096_8gpu_20261010_125103/profile/rank_0_clip.trace.json) |

The BF16 path was used in both runs. The long `aten::item` seen in the older
profile is absent in these runs; clipping still costs 4.3 seconds at 64
threads. The 2-thread run spent most of its time reading CPU BF16 gradients
inside `rank_local_bf16_sumsq`. The other seven ranks spent about 32.3 seconds
in clipping at 2 threads and about 3.0 seconds at 64 rank-0 threads, consistent
with waiting for rank 0. The 64-thread run's remaining CPU time includes
about 1.36 seconds in `aten::stack` and 1.25 seconds in `aten::mul_`.

The only intended configuration difference is rank 0's KT OpenMP thread
count. Other ranks retained two threads. A separate 1 GiB BF16 reduction
check on the same host took 0.162–0.165 seconds with two threads and
0.008–0.012 seconds with 64 threads. This supports the thread-count cause;
the full-model profiler comparison measures its effect on the actual path.

Each run directory includes the configuration, source checksums, resource
summary, per-rank clip timing, operator table, and training log. The raw
`.trace.json` files open directly in Chrome or Perfetto. Each local trace's
SHA-256 was checked against the remote copy. The preceding 15-step seq32
attempt ended with rank 0 receiving SIGKILL before producing step timing;
its evidence is in `../failures/glm_kt_seq32_8gpu_20261010_122800/`.
