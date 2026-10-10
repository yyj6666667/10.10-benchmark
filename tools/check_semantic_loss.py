"""Record whether a fixed-batch KT semantic run reduced finite training loss."""

import ast
import json
import math
import re
import sys
from pathlib import Path


log = Path(sys.argv[1])
output = Path(sys.argv[2])
expected_steps = int(sys.argv[3])
losses = []
grad_norms = []
for line in log.read_text(errors="replace").splitlines():
    if not re.match(r"^\s*\{['\"]loss['\"]\s*:", line):
        continue
    values = ast.literal_eval(line.strip())
    losses.append(float(values["loss"]))
    grad_norms.append(float(values["grad_norm"]))

passed = (
    len(losses) == expected_steps
    and all(math.isfinite(value) for value in losses + grad_norms)
    and losses[-1] < losses[0]
)
report = {
    "passed": passed,
    "expected_steps": expected_steps,
    "observed_steps": len(losses),
    "losses": losses,
    "grad_norms": grad_norms,
}
output.write_text(json.dumps(report, indent=2) + "\n")
print(f"semantic loss: {losses[0] if losses else 'missing'} -> {losses[-1] if losses else 'missing'}; passed={passed}")
if not passed:
    raise SystemExit(1)
