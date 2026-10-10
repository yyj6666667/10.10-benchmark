"""LLaMA Factory entry point for the KT full fine-tuning benchmark."""

from __future__ import annotations

import functools
import json
import os
import time
from pathlib import Path


def main() -> None:
    owner_omp_threads = os.environ.get("KT_BENCH_OWNER_OMP_THREADS") or os.environ.get(
        "KT_CLIP_PROFILE_OWNER_OMP_THREADS"
    )
    if owner_omp_threads and int(os.environ.get("RANK", "0")) == 0:
        if int(owner_omp_threads) < 1:
            raise ValueError("KT_CLIP_PROFILE_OWNER_OMP_THREADS must be positive")
        os.environ["ACCELERATE_KT_OMP_NUM_THREADS"] = owner_omp_threads
        os.environ["OMP_NUM_THREADS"] = owner_omp_threads
        os.environ["MKL_NUM_THREADS"] = owner_omp_threads

    from transformers import Trainer

    from step_phase_timer import install_step_phase_timing

    clip_profile_dir = os.environ.get("KT_CLIP_PROFILE_DIR")
    if os.environ.get("GLM45_PROFILE_DIR") or clip_profile_dir:
        import torch
        from accelerate import Accelerator

        original_clip_grad_norm = Accelerator.clip_grad_norm_

        @functools.wraps(original_clip_grad_norm)
        def profiled_clip_grad_norm(self, *args, **kwargs):
            if not clip_profile_dir:
                with torch.profiler.record_function("KT/gradient_clip"):
                    return original_clip_grad_norm(self, *args, **kwargs)

            rank = int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0")))
            out_dir = Path(clip_profile_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            started = time.perf_counter()
            if rank == 0:
                with torch.profiler.profile(
                    activities=[
                        torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA,
                    ],
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                ) as profiler:
                    with torch.profiler.record_function("KT/gradient_clip"):
                        result = original_clip_grad_norm(self, *args, **kwargs)
                elapsed = time.perf_counter() - started
                profiler.export_chrome_trace(str(out_dir / "rank_0_clip.trace.json"))
                (out_dir / "rank_0_clip.ops.txt").write_text(
                    profiler.key_averages().table(
                        sort_by="self_cpu_time_total", row_limit=80
                    )
                    + "\n",
                    encoding="utf-8",
                )
            else:
                with torch.profiler.record_function("KT/gradient_clip"):
                    result = original_clip_grad_norm(self, *args, **kwargs)
                elapsed = time.perf_counter() - started

            (out_dir / f"rank_{rank}_clip.metadata.json").write_text(
                json.dumps(
                    {
                        "rank": rank,
                        "clip_host_wall_sec": elapsed,
                        "trace": "rank_0_clip.trace.json" if rank == 0 else None,
                        "stops_before_optimizer_update": True,
                    },
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"[kt_clip_profile] rank={rank} clip_sec={elapsed:.3f}", flush=True)
            if os.environ.get("KT_CLIP_PROFILE_STOP_AFTER_CLIP") == "1":
                raise SystemExit(0)
            return result

        Accelerator.clip_grad_norm_ = profiled_clip_grad_norm

    output = os.environ.get("FFT_STEP_TIMING_OUT_DIR")
    if output:
        rank = os.environ.get("RANK", "0")
        os.environ["FFT_STEP_TIMING_OUT_DIR"] = str(Path(output) / f"rank_{rank}")
    install_step_phase_timing()

    # LLaMA Factory calls save_model after train() even when save_strategy=no.
    # Checkpoints are not part of this benchmark and can exceed 200 GiB.
    def skip_model_save(self: Trainer, *args: object, **kwargs: object) -> None:
        print("[kt_bench] final model save skipped", flush=True)

    Trainer.save_model = skip_model_save

    if os.environ.get("FFT_TRAINING_BACKEND") == "deepspeed":
        # torchrun has already created the eight ranks. Calling the CLI here
        # would launch a second torchrun from every rank.
        from llamafactory.train.tuner import run_exp

        run_exp()
    else:
        from llamafactory.cli import main as llamafactory_main

        llamafactory_main()


if __name__ == "__main__":
    main()
