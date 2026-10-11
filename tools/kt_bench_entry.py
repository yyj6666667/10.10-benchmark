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

    if os.environ.get("ZERO_MEMORY_DIAGNOSTIC") == "1":
        install_zero_memory_markers()

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
        expected = int(os.environ["FFT_EXPECTED_TRAINABLE_PARAMS"])
        original_train = Trainer.train

        @functools.wraps(original_train)
        def validated_train(self: Trainer, *args: object, **kwargs: object):
            # ZeRO-3 stores a shard locally; ds_numel is the logical size.
            logical = sum(
                getattr(parameter, "ds_numel", parameter.numel())
                for parameter in self.model.parameters()
                if parameter.requires_grad
            )
            if logical != expected:
                raise RuntimeError(
                    f"trainable parameter contract {logical} != {expected}"
                )
            if int(os.environ.get("RANK", "0")) == 0:
                print(f"[contract] trainable_parameters={logical}", flush=True)
            return original_train(self, *args, **kwargs)

        Trainer.train = validated_train
        # torchrun has already created the eight ranks. Calling the CLI here
        # would launch a second torchrun from every rank.
        from llamafactory.train.tuner import run_exp

        run_exp()
    else:
        from llamafactory.cli import main as llamafactory_main

        llamafactory_main()


def install_zero_memory_markers() -> None:
    """Log loader and engine phase boundaries without changing their behavior."""
    import deepspeed
    from transformers import modeling_utils

    rank = os.environ.get("RANK", "0")

    def mark(name: str) -> None:
        if rank == "0":
            print(f"[zero_memory_phase] {time.time():.6f} {name}", flush=True)

    init_enter = deepspeed.zero.Init.__enter__
    init_exit = deepspeed.zero.Init.__exit__

    @functools.wraps(init_enter)
    def enter(self, *args, **kwargs):
        mark("model_construction.start")
        return init_enter(self, *args, **kwargs)

    @functools.wraps(init_exit)
    def leave(self, *args, **kwargs):
        try:
            return init_exit(self, *args, **kwargs)
        finally:
            mark("model_construction.end")

    deepspeed.zero.Init.__enter__ = enter
    deepspeed.zero.Init.__exit__ = leave

    def wrap_static(name: str, label: str) -> None:
        original = getattr(modeling_utils.PreTrainedModel, name)

        @functools.wraps(original)
        def wrapped(*args, **kwargs):
            mark(f"{label}.start")
            try:
                return original(*args, **kwargs)
            finally:
                mark(f"{label}.end")

        setattr(modeling_utils.PreTrainedModel, name, staticmethod(wrapped))

    wrap_static("_load_pretrained_model", "checkpoint_load")
    wrap_static("_finalize_model_loading", "model_finalize")

    load_state_dict = modeling_utils.load_state_dict
    shard_count = 0

    @functools.wraps(load_state_dict)
    def counted_load(*args, **kwargs):
        nonlocal shard_count
        shard_count += 1
        mark(f"checkpoint_shard.{shard_count}.start")
        try:
            return load_state_dict(*args, **kwargs)
        finally:
            mark(f"checkpoint_shard.{shard_count}.end")

    modeling_utils.load_state_dict = counted_load

    original_initialize = deepspeed.initialize

    @functools.wraps(original_initialize)
    def initialize(*args, **kwargs):
        mark("zero_optimizer_init.start")
        try:
            result = original_initialize(*args, **kwargs)
            if rank == "0":
                try:
                    optimizer = result[0].optimizer
                    parameter = optimizer.fp16_partitioned_groups_flat[0]
                    master = optimizer.fp32_partitioned_groups_flat[0]
                    states = optimizer.optimizer.state.get(master, {})

                    def describe(tensor):
                        return {
                            "dtype": str(tensor.dtype),
                            "device": str(tensor.device),
                            "numel": tensor.numel(),
                            "storage_ptr": tensor.untyped_storage().data_ptr(),
                        }

                    print(
                        "[zero_memory_state] "
                        + json.dumps(
                            {
                                "parameter_partition": describe(parameter),
                                "master_partition": describe(master),
                                "optimizer_state": {
                                    key: describe(value)
                                    for key, value in states.items()
                                    if hasattr(value, "untyped_storage")
                                },
                            }
                        ),
                        flush=True,
                    )
                except (AttributeError, IndexError, KeyError, TypeError) as exc:
                    print(f"[zero_memory_state] unavailable: {exc}", flush=True)
            return result
        finally:
            mark("zero_optimizer_init.end")

    deepspeed.initialize = initialize


if __name__ == "__main__":
    main()
