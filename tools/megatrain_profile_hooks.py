"""Optional rank-local MegaTrain worker trace with explicit stream roles.

Only the selected GPU worker and training step are profiled. The hooks label
the existing weight, compute, and gradient streams without changing their
copy, synchronization, or mathematical operations.
"""

from __future__ import annotations

import functools
import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

import torch


class MegaTrainWorkerProfiler:
    def __init__(self, ctx: Any, worker_globals: dict[str, Any]) -> None:
        self.ctx = ctx
        self.out_dir = Path(os.environ["MEGATRAIN_PROFILE_DIR"])
        self.target_step = int(os.environ.get("MEGATRAIN_PROFILE_STEP", "3"))
        self.active = False
        self.current_layer = -1
        self.layer_calls: dict[int, int] = {}
        self._install_hooks(worker_globals)

    def _range(self, name: str):
        return torch.profiler.record_function(name) if self.active else nullcontext()

    def _install_hooks(self, worker_globals: dict[str, Any]) -> None:
        load = worker_globals["_worker_load_layer_to_buffer_async"]

        @functools.wraps(load)
        def marked_load(layer_idx: int, *args: Any, **kwargs: Any):
            with self._range(f"MegaTrain/PCIe H2D weights layer {layer_idx} [weight_stream]"):
                return load(layer_idx, *args, **kwargs)

        worker_globals["_worker_load_layer_to_buffer_async"] = marked_load

        unflatten = worker_globals["_worker_unflatten_to_layer"]

        @functools.wraps(unflatten)
        def marked_unflatten(layer_idx: int, *args: Any, **kwargs: Any):
            self.current_layer = layer_idx
            return unflatten(layer_idx, *args, **kwargs)

        worker_globals["_worker_unflatten_to_layer"] = marked_unflatten

        collect = worker_globals["_worker_collect_layer_grads_async"]

        @functools.wraps(collect)
        def marked_collect(layer_idx: int, *args: Any, **kwargs: Any):
            with self._range(f"MegaTrain/PCIe D2H gradients layer {layer_idx} [grad_stream]"):
                return collect(layer_idx, *args, **kwargs)

        worker_globals["_worker_collect_layer_grads_async"] = marked_collect

        grad = torch.autograd.grad

        @functools.wraps(grad)
        def marked_grad(*args: Any, **kwargs: Any):
            with self._range(
                f"MegaTrain/GPU backward layer {self.current_layer} [compute_stream]"
            ):
                return grad(*args, **kwargs)

        torch.autograd.grad = marked_grad

        module_to = torch.nn.Module.to

        @functools.wraps(module_to)
        def marked_module_to(module: torch.nn.Module, *args: Any, **kwargs: Any):
            destination = args[0] if args else kwargs.get("device")
            first_parameter = next(module.parameters(), None) if self.active else None
            if (self.active and isinstance(destination, torch.device)
                    and destination.type == "cuda" and first_parameter is not None
                    and first_parameter.device.type == "cpu"):
                with self._range(
                    f"MegaTrain/PCIe H2D module {type(module).__name__} [compute_stream]"
                ):
                    return module_to(module, *args, **kwargs)
            return module_to(module, *args, **kwargs)

        torch.nn.Module.to = marked_module_to

        for templates in self.ctx.gpu_layer_templates.values():
            for layer in templates:
                forward = layer.forward

                @functools.wraps(forward)
                def marked_forward(*args: Any, _forward=forward, **kwargs: Any):
                    if not self.active:
                        return _forward(*args, **kwargs)
                    layer_idx = self.current_layer
                    call = self.layer_calls.get(layer_idx, 0) + 1
                    self.layer_calls[layer_idx] = call
                    phase = ("forward" if call == 1 else
                             "recompute" if call == 2 else "backward replay")
                    with self._range(
                        f"MegaTrain/GPU {phase} layer {layer_idx} [compute_stream]"
                    ):
                        return _forward(*args, **kwargs)

                layer.forward = marked_forward

    @contextmanager
    def capture_step(self, step: int, input_shape: tuple[int, ...]):
        if step != self.target_step:
            yield
            return

        from torch.profiler import ProfilerActivity, profile

        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.layer_calls.clear()
        self.active = True
        print(f"[megatrain_profiler] GPU worker starts step {step}", flush=True)
        try:
            with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as profiler:
                with torch.profiler.record_function("MegaTrain/GPU worker complete step"):
                    yield
                torch.cuda.synchronize(self.ctx.device)
        finally:
            self.active = False

        stem = f"rank0_step{step}"
        trace = self.out_dir / f"{stem}.trace.json"
        profiler.export_chrome_trace(str(trace))
        try:
            operators = profiler.key_averages().table(
                sort_by="self_cuda_time_total", row_limit=80
            )
        except Exception as exc:
            operators = f"operator summary unavailable: {type(exc).__name__}: {exc}"
        (self.out_dir / f"{stem}.ops.txt").write_text(
            operators + "\n",
            encoding="utf-8",
        )
        metadata = {
            "rank": 0,
            "step": step,
            "input_shape": list(input_shape),
            "trace": trace.name,
            "activities": ["CPU", "CUDA"],
            "streams": {
                "compute": self.ctx.compute_stream.cuda_stream,
                "weights_H2D": self.ctx.weight_stream.cuda_stream,
                "gradients_D2H": self.ctx.grad_stream.cuda_stream,
            },
            "record_shapes": False,
            "profile_memory": False,
            "with_stack": False,
        }
        (self.out_dir / f"{stem}.metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        print(f"[megatrain_profiler] saved {trace}", flush=True)
