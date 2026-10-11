# qj5090 MegaTrain GLM seq256 PyTorch profiler

This is a separate diagnostic, excluded from the throughput chart. The eight
GPU run used the same BF16 text model and `MEGATRAIN_OFFLOAD_EMB_HEAD=1` as
the sweep. It repeated one fixed batch for four optimizer steps; steps 1–2
were warmup and rank 0's complete GPU-worker step 3 was profiled. Loss fell
from 3.2181 to 0.4258 across the four steps (`loss_check.json`).

The raw Chrome trace is `rank0_step3.trace.json` on this local workstation and
under `/mnt/hurry-bench/profiles/megatrain_glm_seq256_20261011_042812/` on
qj5090. Its SHA-256 is
`4b94471058f685a32cee5a58914198977a5fbb1754d23a002abdb6bb9327c976`.
The lossless 17 MB `rank0_step3.trace.json.zst` is tracked in this private
repository; decompress it with `zstd -d` to recover the original 296 MB JSON.
The local trace was verified against the remote SHA-256. Run configuration,
source checksums, software versions, loss evidence, and the training log are
saved alongside it.

## GPU activity in the captured worker step

| Kineto stream | Work | Events | GPU busy time |
|---:|---|---:|---:|
| 7 | Compute kernels | 79,411 | 1.103 s |
| 20 | Pinned layer-weight H2D copies | 138 | 19.144 s |
| 24 | Gradient D2H copies | 690 | 8.191 s |

The trace labels each layer's weight load, forward/recompute/backward compute,
and gradient copy. Based on timestamp interval intersections, weight H2D and
compute kernels overlapped for 0.228 s; gradient D2H and compute kernels for
0.004 s; weight H2D and gradient D2H for 7.975 s. These are observed stream
activity intervals on rank 0, not a complete accounting of CPU stalls or all
eight GPUs. The profiler export slowed its step, so its step time must not be
used as benchmark throughput.

To inspect the complete trace, run Perfetto's native TraceProcessor on the
expanded JSON and open `https://ui.perfetto.dev/#!/?rpc_port=9001`. This
avoids the browser's large JSON import path. The local Perfetto UI has been
opened with the native server connected. The upstream guide is
<https://perfetto.dev/docs/visualization/large-traces>.

The MegaTrain scan was paused before profiling. After the diagnostic, the
remote worker source was restored to its pre-profile SHA-256
`b58e4a01974b297ec642eecd2ff88c55ce6d25eb825e42903a0d2136327f32ef`;
the MegaTrain and ZeRO queues remain paused/waiting as requested.
