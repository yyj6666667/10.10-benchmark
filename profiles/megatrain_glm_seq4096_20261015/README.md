# qj5090 MegaTrain GLM seq4096 PyTorch profiler

This is a diagnostic run, separate from the throughput figure. It used
GLM-4.5-Air on eight RTX 5090 GPUs, BF16 full fine-tuning, global batch 8,
four fixed-batch optimizer steps, and two warmup steps. PyTorch profiler
captured rank 0 GPU worker's complete step 3 with CPU and CUDA activities.
The trace labels layer forward/recompute/backward, weight H2D, and gradient
D2H work. The CPU optimizer runs in the parent process and is not included
in this worker trace.

The loss check passed (4.187525 → 0.225303); see
[loss_check.json](loss_check.json). Profiler overhead affected the captured
step, so use the trace for timeline inspection rather than benchmark TPS.

## Files

- [Compressed Chrome trace](rank0_step3.trace.json.zst): lossless 20 MB copy
  of the 326,870,611-byte JSON. The expanded local JSON is ignored by Git.
- [SHA-256](rank0_step3.sha256): the expanded trace is
  e8c8aed23c7febaa6c96840b40abba38ed5fa56d852ab94af6b329cdfa700cc6;
  local decompression matched the remote source.
- [Stream metadata](rank0_step3.metadata.json) and
  [operator listing](rank0_step3.ops.txt).
- [Run configuration](run_config.json), [step timings](step_timing.json),
  [resource samples](resources.json), [versions](versions.txt),
  [source checksums](checksums.txt), and [launcher log](launcher.log).
- [Profiler source patch](worker_profiler.patch),
  [provenance](provenance.txt), and the exact
  [diagnostic runner](run_qj_megatrain_glm4096_profile_case.sh).

Remote profile:
/mnt/hurry-bench/runs/megatrain-glm4096-profile-20261015

Remote run:
/mnt/hurry-bench/runs/glm45_air_megatrain_alias_slots_patch_seq4096_8gpu_20261015_150013_1129971_semantic

The remote worker source was restored after exit 0 to SHA-256
556200e49eaaecf6e7598cb71faef98519aae39ece8e2315147fc8098f488c62.
No model or optimizer checkpoint was saved.

## Open locally

On this workstation, open [Perfetto](https://ui.perfetto.dev), choose
**Open trace file**, and select `rank0_step3.trace.json` from this directory.
This does not require a local server. If using a fresh clone, first expand
the compressed copy:

~~~bash
zstd -d -k rank0_step3.trace.json.zst -o rank0_step3.trace.json
~~~

The JSON is 327 MB. If browser-side loading runs out of memory or stalls,
start the native Trace Processor instead:

~~~bash
curl -fLsS https://commondatastorage.googleapis.com/perfetto-luci-artifacts/v58.3/linux-amd64/trace_processor_shell \
  -o /tmp/trace_processor_v58_3
chmod u+x /tmp/trace_processor_v58_3
/tmp/trace_processor_v58_3 server http --port 9013 --ip-address 127.0.0.1 \
  rank0_step3.trace.json
~~~

In the Perfetto UI, enable the **Relax CSP** flag once, then open
[the native viewer](https://ui.perfetto.dev/#!/?rpc_port=9013).
The binary above is Perfetto v58.3-11fbaed83 (SHA-256
`f13baf75f5655109a637b041e7f9c78ebe76d7c0f030cd6d611d5679cfcb4fb9`),
matching the UI version used on 2026-10-15. The generic
`get.perfetto.dev/trace_processor` wrapper downloaded v58.2 and triggered a
version-mismatch dialog.
Port 9001 was occupied by an unrelated existing trace server during this run.
The raw trace remains on this workstation for immediate viewing; the
compressed file in the private repository is the durable copy.
