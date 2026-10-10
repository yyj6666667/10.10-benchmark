# sap4 FSDP2 + oneDNN sweep status

Last remote check: 2026-10-10 05:28 UTC. Since then the SSH entry
`117.74.64.224:15016` has timed out. The public IP still responds to ping;
remote training status is unknown until SSH is restored.

- Active case at last check: `fsdp2-glm2048-formal-20261010_0514.service`.
  GLM-4.5-Air, sequence 2048, 8 GPUs, 15 steps, 5 warmup. Steps 1 and 2
  completed with losses 6.258377 and 5.120240. Neither is a stable result.
- Queue service: `fsdp2-remaining-20261010.service`. It waits for GLM 2048
  exit code 0, then runs GLM 4096, Qwen 2048, Qwen 4096 sequentially. It
  stops on any failure or unavailable GPU/RAM.
- Remote output root: `/home/qujing/outputs/onednn_20261010`.
- Remote source: `/home/qujing/outputs/onednn_20261010/fsdp2_review`.
- The exact GLM 2048 source at launch is `source/fsdp2_onednn_full_ft_v1.py`;
  its SHA-256 matches the run's `source_checksums.txt`. The current runner
  releases idle CUDA cache at the end of every step for subsequent cases.

On reconnection, check both systemd units, their case `exit_code.txt` files,
and each case's `step_timing.json`. Only complete 15-step, exit-code-0 cases
belong in the throughput table. Pull small logs/timings/resource CSVs into
this repository; no checkpoint or optimizer weights are produced by the
runner.
