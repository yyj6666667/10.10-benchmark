"""Apply the benchmark's bounded-memory ZeRO-3 checkpoint loader patch.

The checkpoint shards used here contain complete conversion groups (one GLM
MoE layer per shard). Loading and converting all shards at once materializes
an extra full model in each rank before ZeRO has partitioned the parameters.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path


OLD = '''        if is_deepspeed_zero3_enabled() and not is_quantized:
            if state_dict is None:
                merged_state_dict = {}
                for ckpt_file in checkpoint_files:
                    merged_state_dict.update(
                        load_state_dict(ckpt_file, map_location="cpu", weights_only=load_config.weights_only)
                    )
                state_dict = merged_state_dict
            error_msgs, missing_keys = _load_state_dict_into_zero3_model(model, state_dict, load_config)
'''

NEW = '''        if is_deepspeed_zero3_enabled() and not is_quantized:
            if state_dict is None:
                # Load one shard at a time. MoE weight conversion can allocate a
                # second full checkpoint per rank if all shards are merged first.
                missing_keys = None
                for ckpt_file in checkpoint_files:
                    shard = load_state_dict(ckpt_file, map_location="cpu", weights_only=load_config.weights_only)
                    shard_errors, shard_missing = _load_state_dict_into_zero3_model(model, shard, load_config)
                    error_msgs.extend(shard_errors)
                    missing_keys = shard_missing if missing_keys is None else missing_keys & shard_missing
                    del shard
                if missing_keys is None:
                    raise ValueError("No checkpoint shards were found for ZeRO-3 loading")
            else:
                error_msgs, missing_keys = _load_state_dict_into_zero3_model(model, state_dict, load_config)
'''

OLD_RENAMING = '''        weight_conversions = get_model_conversion_mapping(model, key_mapping, hf_quantizer)
'''

NEW_RENAMING = '''        weight_conversions = get_model_conversion_mapping(model, key_mapping, hf_quantizer)
        # A checkpoint that already has every expected key needs no renaming.
        # In particular, the nested Qwen3.5 text conversion otherwise strips
        # `language_model` from a matching full-model checkpoint.
        if (
            is_deepspeed_zero3_enabled()
            and sharded_metadata is not None
            and not any(isinstance(item, WeightConverter) for item in weight_conversions)
            and set(model.state_dict()).issubset(sharded_metadata.get("weight_map", {}))
        ):
            weight_conversions = []
'''


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--backup", type=Path, required=True)
    args = parser.parse_args()
    content = args.source.read_text()
    if NEW in content and NEW_RENAMING in content:
        print("already patched")
        return
    if content.count(OLD) != 1 or content.count(OLD_RENAMING) != 1:
        raise RuntimeError("Expected exactly one matching loader and conversion branch")
    if args.backup.exists():
        if args.backup.read_text() != content:
            raise RuntimeError("Existing backup does not match current source")
    else:
        args.backup.parent.mkdir(parents=True, exist_ok=True)
        args.backup.write_text(content)
    patched = content.replace(OLD, NEW).replace(OLD_RENAMING, NEW_RENAMING)
    args.source.write_text(patched)
    for label, data in (("before", content), ("after", patched)):
        print(label, hashlib.sha256(data.encode()).hexdigest())


if __name__ == "__main__":
    main()
