#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import shlex
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def resolve_path(path_text: str) -> Path:
    p = Path(path_text)
    if p.is_absolute():
        return p
    return ROOT / p


def parse_query_output(stdout: str):
    interval = None
    count = None

    m = re.search(r"interval:\s*\[(\d+),\s*(\d+)\)", stdout)
    if m:
        interval = [int(m.group(1)), int(m.group(2))]

    m = re.search(r"count:\s*(\d+)", stdout)
    if m:
        count = int(m.group(1))

    return interval, count


def fail(msg: str) -> int:
    print(f"[audit-v0 verify] FAIL: {msg}", file=sys.stderr)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Verify GLYPH_AUDIT_ARTIFACT_V0."
    )
    ap.add_argument("artifact")
    args = ap.parse_args()

    artifact_path = Path(args.artifact).resolve()

    if not artifact_path.exists():
        return fail(f"artifact not found: {artifact_path}")

    try:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
    except Exception as e:
        return fail(f"invalid JSON: {e}")

    if artifact.get("artifact_version") != "GLYPH_AUDIT_ARTIFACT_V0":
        return fail("artifact_version mismatch")

    corpus_info = artifact.get("corpus", {})
    manifest_info = artifact.get("index_manifest", {})
    query_info = artifact.get("query", {})
    result_info = artifact.get("result", {})
    verification_info = artifact.get("verification", {})

    corpus_path = resolve_path(corpus_info.get("path", ""))
    manifest_path = resolve_path(manifest_info.get("path", ""))

    if not corpus_path.exists():
        return fail(f"corpus not found: {corpus_path}")

    if not manifest_path.exists():
        return fail(f"manifest not found: {manifest_path}")

    actual_corpus_sha = sha256_file(corpus_path)
    expected_corpus_sha = corpus_info.get("sha256")

    if actual_corpus_sha != expected_corpus_sha:
        return fail(
            f"corpus sha256 mismatch: expected={expected_corpus_sha} actual={actual_corpus_sha}"
        )

    actual_manifest_sha = sha256_file(manifest_path)
    expected_manifest_sha = manifest_info.get("sha256")

    if actual_manifest_sha != expected_manifest_sha:
        return fail(
            f"manifest sha256 mismatch: expected={expected_manifest_sha} actual={actual_manifest_sha}"
        )

    if query_info.get("encoding") != "hex":
        return fail("only hex query encoding is supported in V0")

    query_hex = query_info.get("hex")
    if not query_hex:
        return fail("query.hex missing")

    try:
        query_bytes = bytes.fromhex(query_hex)
    except ValueError as e:
        return fail(f"invalid query hex: {e}")

    actual_query_sha = sha256_bytes(query_bytes)
    expected_query_sha = query_info.get("sha256")

    if actual_query_sha != expected_query_sha:
        return fail(
            f"query sha256 mismatch: expected={expected_query_sha} actual={actual_query_sha}"
        )

    expected_len = query_info.get("length_bytes")
    if len(query_bytes) != expected_len:
        return fail(
            f"query length mismatch: expected={expected_len} actual={len(query_bytes)}"
        )

    command = verification_info.get("command")
    if not command:
        return fail("verification.command missing")

    cmd = shlex.split(command)

    run = subprocess.run(
        cmd,
        cwd=str(ROOT),
        text=True,
        capture_output=True,
    )

    interval, count = parse_query_output(run.stdout)

    if run.returncode != 0:
        print(run.stdout)
        print(run.stderr, file=sys.stderr)
        return fail(f"verification command returned {run.returncode}")

    expected_status = verification_info.get("reproduce_status")
    if expected_status != "PASS":
        return fail(f"artifact reproduce_status is not PASS: {expected_status}")

    expected_count = result_info.get("match_count")
    if count != expected_count:
        return fail(f"match_count mismatch: expected={expected_count} actual={count}")

    expected_interval = result_info.get("fm_interval")
    if expected_interval is not None and interval != expected_interval:
        return fail(f"fm_interval mismatch: expected={expected_interval} actual={interval}")

    print("[audit-v0 verify] corpus sha256 OK")
    print("[audit-v0 verify] manifest sha256 OK")
    print("[audit-v0 verify] query sha256 OK")
    print("[audit-v0 verify] verification command OK")
    print(f"[audit-v0 verify] match_count={count}")
    print(f"[audit-v0 verify] fm_interval={interval}")
    print("VERIFY AUDIT ARTIFACT OK")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
