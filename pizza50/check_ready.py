#!/usr/bin/env python3
"""Check the premeasurement lock only. Never run a search or a benchmark."""
import hashlib
import json
from pathlib import Path
import sys


def digest(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parent
    freeze = json.loads((root / "FREEZE.json").read_text("utf-8"))
    failures = []
    for relative, spec in freeze["files"].items():
        p = root / relative
        if not p.is_file() or p.stat().st_size != spec["bytes"] or digest(p) != spec["sha256"]:
            failures.append("changed frozen file: " + relative)
    lock = json.loads((root / "INPUT_LOCK.json").read_text("utf-8"))
    for name in ("corpus", "GLYPH_index", "Move-r_index", "oracle"):
        spec = lock[name]
        if not spec.get("path") or not spec.get("sha256"):
            failures.append("missing input lock: " + name)
    if failures:
        print(json.dumps({"ready": False, "performance_runs_started": 0,
                          "reasons": failures}, sort_keys=True))
        return 2
    for name in ("corpus", "GLYPH_index", "Move-r_index", "oracle"):
        spec = lock[name]
        p = Path(spec["path"])
        if not p.is_file() or digest(p) != spec["sha256"]:
            failures.append("input file SHA mismatch or inaccessible: " + name)
    # This is only a necessary lock check, never a correctness/performance PASS.
    print(json.dumps({"ready": not failures, "performance_runs_started": 0,
                      "reasons": failures}, sort_keys=True))
    return 2 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
