"""Check that a complete Qwen checkpoint contains BF16, not FP8, weights."""

import json
import struct
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    config = json.loads((root / "config.json").read_text())
    if config.get("model_type") != "qwen3_5_moe":
        raise SystemExit(f"unexpected model_type: {config.get('model_type')}")

    index = json.loads((root / "model.safetensors.index.json").read_text())
    shards = sorted({root / name for name in index["weight_map"].values()})
    if not shards:
        raise SystemExit("checkpoint index has no shards")
    dtypes = Counter()
    for shard in shards:
        with shard.open("rb") as stream:
            size_bytes = stream.read(8)
            if len(size_bytes) != 8:
                raise SystemExit(f"empty or truncated shard: {shard}")
            header_size = struct.unpack("<Q", size_bytes)[0]
            if header_size == 0 or header_size > 256 * 1024 * 1024:
                raise SystemExit(f"invalid safetensors header length: {shard}")
            header = stream.read(header_size)
            if len(header) != header_size:
                raise SystemExit(f"truncated safetensors header: {shard}")
        tensors = json.loads(header)
        dtypes.update(v["dtype"] for k, v in tensors.items() if k != "__metadata__")

    if dtypes["BF16"] == 0 or any(dtype.startswith("F8") for dtype in dtypes):
        raise SystemExit(f"checkpoint is not BF16: {dict(dtypes)}")
    print(f"verified {len(shards)} Qwen shards; tensor dtypes: {dict(dtypes)}", flush=True)


if __name__ == "__main__":
    main()
