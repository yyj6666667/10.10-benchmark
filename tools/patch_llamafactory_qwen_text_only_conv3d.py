"""Allow the frozen Qwen3.5 vision tower in this text-only benchmark."""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from pathlib import Path


OLD = '''            if getattr(model_args, "use_kt", False) and getattr(model.config, "model_type", None) in [
                "qwen3_5",
                "qwen3_5_moe",
            ]:
                logger.warning_rank0(
                    "Detected torch 2.9.x with Conv3D in Qwen3.5 vision modules. "
                    "Continuing because KT text LoRA training does not exercise the vision tower."
                )
'''

NEW = '''            if (
                (
                    getattr(model_args, "use_kt", False)
                    or (
                        os.environ.get("FFT_TEXT_ONLY_BENCHMARK") == "1"
                        and finetuning_args.freeze_vision_tower
                    )
                )
                and getattr(model.config, "model_type", None) in ["qwen3_5", "qwen3_5_moe"]
            ):
                logger.warning_rank0(
                    "Detected torch 2.9.x with Conv3D in Qwen3.5 vision modules. "
                    "Continuing because this text-only run does not exercise the vision tower."
                )
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    content = args.source.read_text()
    if NEW in content:
        print("already patched")
        return
    if content.count(OLD) != 1:
        raise RuntimeError("Expected exactly one Qwen3.5 Conv3D guard")
    if args.backup.exists():
        if args.backup.read_text() != content:
            raise RuntimeError("Existing backup does not match current source")
    else:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.backup.write_text(content)
    patched = content.replace(OLD, NEW)
    with tempfile.NamedTemporaryFile("w", dir=args.source.parent, delete=False) as output:
        output.write(patched)
        temporary = output.name
    os.replace(temporary, args.source)
    for label, data in (("before", content), ("after", patched)):
        print(label, hashlib.sha256(data.encode()).hexdigest())


if __name__ == "__main__":
    main()
