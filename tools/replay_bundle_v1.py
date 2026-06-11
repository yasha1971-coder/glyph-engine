#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hash(label, path, expected):
    actual = sha256_file(path)
    print(f"{label}_expected={expected}")
    print(f"{label}_actual={actual}")

    if actual != expected:
        print(f"{label}_MISMATCH")
        return False

    print(f"{label}_VERIFIED")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.bundle)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    ok = True

    ok &= verify_hash(
        "EVIDENCE_SHA256",
        manifest["evidence_path"],
        manifest["evidence_sha256"],
    )

    ok &= verify_hash(
        "CORPUS_SHA256",
        manifest["corpus_path"],
        manifest["corpus_sha256"],
    )

    ok &= verify_hash(
        "FM_SHA256",
        manifest["fm_path"],
        manifest["fm_sha256"],
    )

    ok &= verify_hash(
        "BWT_SHA256",
        manifest["bwt_path"],
        manifest["bwt_sha256"],
    )

    ok &= verify_hash(
        "SA_SHA256",
        manifest["sa_path"],
        manifest["sa_sha256"],
    )

    if not ok:
        print("BUNDLE MISMATCH")
        return 2

    replay_cmd = [
        "tools/replay_evidence_v1.py",
        "--evidence",
        manifest["evidence_path"],
    ]

    p = subprocess.run(
        replay_cmd,
        text=True,
        capture_output=True,
    )

    print(p.stdout, end="")
    if p.stderr:
        print(p.stderr, end="")

    if p.returncode != 0:
        print("QUERY_REPLAY_MISMATCH")
        print("BUNDLE MISMATCH")
        return p.returncode

    print("QUERY_REPLAY_VERIFIED")
    print("BUNDLE VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
