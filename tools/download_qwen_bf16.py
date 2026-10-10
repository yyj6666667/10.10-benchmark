"""Download the official Qwen3.5-122B-A10B BF16 checkpoint on qj5090."""

import datetime
import json
import sys
import time
import traceback
from pathlib import Path

from modelscope.hub.snapshot_download import snapshot_download


REPO = "Qwen/Qwen3.5-122B-A10B"
TARGET = Path("/mnt/data/models/Qwen3.5-122B-A10B")
WORKERS = 8


def verify_download() -> None:
    config = json.loads((TARGET / "config.json").read_text())
    dtype = str(config.get("torch_dtype", config.get("dtype", ""))).lower()
    if "float8" in dtype or "fp8" in dtype:
        raise RuntimeError(f"Unexpected quantized model dtype: {dtype}")

    index = json.loads((TARGET / "model.safetensors.index.json").read_text())
    shards = {TARGET / filename for filename in index["weight_map"].values()}
    missing = [path.name for path in shards if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty weight shards: {missing[:8]}")
    print("VERIFIED", "dtype", dtype, "shards", len(shards), flush=True)


print("start_utc", datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
print("repo", REPO, "revision", "master", "target", TARGET, "workers", WORKERS, flush=True)
for attempt in range(1, 5):
    try:
        result = snapshot_download(REPO, revision="master", local_dir=str(TARGET), max_workers=WORKERS)
        verify_download()
        print("SUCCESS", datetime.datetime.now(datetime.timezone.utc).isoformat(), result, flush=True)
        sys.exit(0)
    except Exception:
        print("FAILED_ATTEMPT", attempt, datetime.datetime.now(datetime.timezone.utc).isoformat(), flush=True)
        traceback.print_exc()
        if attempt == 4:
            sys.exit(1)
        time.sleep(60 * attempt)
