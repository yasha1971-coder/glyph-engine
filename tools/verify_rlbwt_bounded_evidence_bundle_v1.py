#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify portable RLBWT bounded evidence bundle V1.")
    ap.add_argument("--bundle", required=True)
    args = ap.parse_args()

    bundle = Path(args.bundle).resolve()
    manifest_path = bundle / "bundle_manifest_v1.json"

    if not manifest_path.exists():
        raise SystemExit(f"missing bundle manifest: {manifest_path}")

    manifest = json.loads(manifest_path.read_text())
    errors = []

    if manifest.get("bundle_version") != "RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1":
        errors.append("bad bundle_version")

    for rec in manifest.get("files", []):
        p = bundle / rec["path"]
        if not p.exists():
            errors.append(f"missing file: {rec['path']}")
            continue
        if p.stat().st_size != rec["bytes"]:
            errors.append(f"size mismatch: {rec['path']}")
        if sha256_file(p) != rec["sha256"]:
            errors.append(f"sha256 mismatch: {rec['path']}")

    evidence_path = bundle / manifest.get("evidence", "evidence.json")
    if not evidence_path.exists():
        errors.append("missing evidence.json")

    replay_result = None

    if not errors:
        evidence = json.loads(evidence_path.read_text())

        evidence["source_corpus"]["path"] = str(bundle / "corpus.bin")
        evidence["runtime_dir"] = str(bundle / "runtime")

        for rf in evidence["runtime_files"]:
            rf["path"] = str(bundle / rf["path"])

        with tempfile.TemporaryDirectory() as td:
            tmp_artifact = Path(td) / "evidence.abs.json"
            tmp_artifact.write_text(
                json.dumps(evidence, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

            proc = subprocess.run(
                [
                    "python3",
                    str(ROOT / "tools" / "verify_rlbwt_bounded_evidence_v1.py"),
                    "--artifact",
                    str(tmp_artifact),
                ],
                text=True,
                capture_output=True,
            )

            if proc.returncode != 0:
                errors.append("replay verifier failed")
                errors.append(proc.stdout)
                errors.append(proc.stderr)
            else:
                replay_result = json.loads(proc.stdout)

    result = {
        "ok": not errors,
        "bundle": str(bundle),
        "bundle_version": manifest.get("bundle_version"),
        "artifact_version": manifest.get("artifact_version"),
        "profile": manifest.get("profile"),
        "replay_result": replay_result,
        "errors": errors,
    }

    print(json.dumps(result, indent=2, sort_keys=True))

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
