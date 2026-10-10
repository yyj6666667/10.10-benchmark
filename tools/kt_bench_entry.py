"""LLaMA Factory entry point for the KT full fine-tuning benchmark."""

from __future__ import annotations

import os
from pathlib import Path


def main() -> None:
    from transformers import Trainer

    from step_phase_timer import install_step_phase_timing

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

    from llamafactory.cli import main as llamafactory_main

    llamafactory_main()


if __name__ == "__main__":
    main()
