#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def safe_extract(tar_path: Path, out_dir: Path) -> Path:
    with tarfile.open(tar_path, "r:gz") as tar:
        members = tar.getmembers()

        if not members:
            raise RuntimeError("empty tar archive")

        top_levels = set()

        for m in members:
            name = m.name

            if name.startswith("/") or name == "" or "\x00" in name:
                raise RuntimeError(f"unsafe tar member name: {name!r}")

            parts = Path(name).parts
            if not parts or parts[0] in ("", ".", ".."):
                raise RuntimeError(f"unsafe tar member path: {name!r}")

            top_levels.add(parts[0])

            target = (out_dir / name).resolve()
            root = out_dir.resolve()

            if root != target and root not in target.parents:
                raise RuntimeError(f"path traversal in tar member: {name!r}")

            if m.issym() or m.islnk():
                raise RuntimeError(f"links are not allowed in bundle tar: {name!r}")

        if len(top_levels) != 1:
            raise RuntimeError(f"expected one top-level bundle dir, got: {sorted(top_levels)}")

        tar.extractall(out_dir)

        return out_dir / sorted(top_levels)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description="Verify tar.gz RLBWT bounded evidence bundle V1.")
    ap.add_argument("--tar", required=True)
    args = ap.parse_args()

    tar_path = Path(args.tar).resolve()
    errors = []
    replay_result = None
    bundle_dir_s = None

    if not tar_path.exists():
        raise SystemExit(f"missing tar: {tar_path}")

    with tempfile.TemporaryDirectory() as td:
        try:
            bundle_dir = safe_extract(tar_path, Path(td))
            bundle_dir_s = str(bundle_dir)

            proc = subprocess.run(
                [
                    "python3",
                    str(ROOT / "tools" / "verify_rlbwt_bounded_evidence_bundle_v1.py"),
                    "--bundle",
                    str(bundle_dir),
                ],
                check=False,
                text=True,
                capture_output=True,
            )

            if proc.returncode != 0:
                errors.append("bundle verifier failed")
                errors.append(proc.stdout)
                errors.append(proc.stderr)
            else:
                replay_result = json.loads(proc.stdout)

        except Exception as e:
            errors.append(str(e))

    result = {
        "ok": not errors,
        "tar_version": "RLBWT_BOUNDED_EVIDENCE_BUNDLE_TAR_V1",
        "bundle_version": "RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1",
        "tar": str(tar_path),
        "tar_bytes": tar_path.stat().st_size,
        "tar_sha256": sha256_file(tar_path),
        "extracted_bundle": bundle_dir_s,
        "replay_result": replay_result,
        "errors": errors,
    }

    print(json.dumps(result, indent=2, sort_keys=True))

    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
