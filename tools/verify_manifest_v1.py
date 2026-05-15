#!/usr/bin/env python3
import argparse
import hashlib
import json
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def fail(msg: str) -> None:
    print(f"[GLYPH INTEGRITY FAIL] {msg}", file=sys.stderr)
    raise SystemExit(1)


def check_file(label: str, path: Path, expected_bytes: int, expected_sha256: str) -> None:
    if not path.exists():
        fail(f"{label} missing: {path}")

    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        fail(f"{label} size mismatch: expected {expected_bytes}, got {actual_bytes}: {path}")

    actual_sha256 = sha256_file(path)
    if actual_sha256 != expected_sha256:
        fail(f"{label} sha256 mismatch: {path}")

    print(f"[OK] {label}: {path}")


def verify(index_dir: Path) -> None:
    manifest_path = index_dir / "manifest.json"
    if not manifest_path.exists():
        fail(f"manifest.json not found in {index_dir}")

    manifest = json.loads(manifest_path.read_text())

    if manifest.get("format") != "GLYPH_INDEX_MANIFEST_V1":
        fail(f"bad manifest format: {manifest.get('format')!r}")

    raw = manifest["raw_corpus"]
    index_corpus = manifest["index_corpus"]
    artifacts = manifest["artifacts"]

    check_file("raw_corpus", Path(raw["path"]), int(raw["bytes"]), raw["sha256"])
    check_file("index_corpus", Path(index_corpus["path"]), int(index_corpus["bytes"]), index_corpus["sha256"])

    if index_corpus.get("sentinel") != "0x00":
        fail(f"unexpected sentinel value: {index_corpus.get('sentinel')!r}")

    for name in ("sa", "bwt", "fm"):
        path = Path(artifacts[name])
        if not path.exists():
            fail(f"artifact missing: {name}: {path}")
        print(f"[OK] artifact exists: {name}: {path}")

    print("[GLYPH INTEGRITY OK] manifest verified")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("index_dir")
    args = ap.parse_args()
    verify(Path(args.index_dir))


if __name__ == "__main__":
    main()
