#!/usr/bin/env python3
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

VERSION = "GLYPH_STRUCTURAL_FINGERPRINT_REPLAY_V0"

COMPARE_KEYS = [
    "artifact_version",
    "purpose",
    "non_claims",
    "source",
    "byte_stats",
    "entropy_profile",
    "anchor_repeat_profiles",
    "bwt_profile",
]

def load_json(path: Path):
    return json.loads(path.read_text())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("artifact")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    artifact_path = Path(args.artifact).resolve()
    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    original = load_json(artifact_path)

    errors = []

    if original.get("artifact_version") != "GLYPH_STRUCTURAL_FINGERPRINT_V0":
        errors.append("artifact_version_mismatch")

    source_path = Path(original.get("source", {}).get("path", ""))
    if not source_path.exists():
        errors.append(f"source_missing:{source_path}")

    replayed = None

    if not errors:
        with tempfile.TemporaryDirectory() as td:
            tmp = Path(td) / "replayed.json"

            cmd = [
                "python3",
                "tools/glyph_structural_fingerprint_v0.py",
                str(source_path),
                "--out",
                str(tmp),
                "--chunk-size",
                str(original.get("entropy_profile", {}).get("chunk_size", 65536)),
            ]

            bwt = original.get("bwt_profile")
            if bwt and bwt.get("bwt_path"):
                cmd += ["--bwt-path", bwt["bwt_path"]]

            subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            replayed = load_json(tmp)

    comparisons = {}

    if replayed is not None:
        for key in COMPARE_KEYS:
            comparisons[key] = {
                "match": original.get(key) == replayed.get(key)
            }
            if not comparisons[key]["match"]:
                errors.append(f"field_mismatch:{key}")

    result = {
        "replay_version": VERSION,
        "ok": len(errors) == 0,
        "artifact": str(artifact_path),
        "source": str(source_path),
        "errors": errors,
        "comparisons": comparisons,
    }

    out_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))

if __name__ == "__main__":
    main()
