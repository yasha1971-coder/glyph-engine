#!/usr/bin/env python3

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve_path(base_dir, path_value):
    p = Path(path_value)
    if p.is_absolute():
        return p
    return base_dir / p


def verify_hash(label, path, expected):
    actual = sha256_file(path)
    print(f"{label}_expected={expected}")
    print(f"{label}_actual={actual}")

    if actual != expected:
        print(f"{label}_MISMATCH")
        return False

    print(f"{label}_VERIFIED")
    return True


def replay_evidence(evidence, corpus_path):
    corpus = Path(corpus_path).read_bytes()
    expected_sha = evidence.get("corpus_sha256")

    if expected_sha:
        actual_sha = hashlib.sha256(corpus).hexdigest()
        print(f"expected_sha256={expected_sha}")
        print(f"actual_sha256={actual_sha}")

        if actual_sha != expected_sha:
            print("CORPUS_FINGERPRINT_MISMATCH")
            return False

        print("CORPUS_FINGERPRINT_VERIFIED")

    query_bytes = bytes.fromhex(evidence["query_hex"])

    if not evidence["hits"]:
        print("NO_HITS_IN_EVIDENCE")
        return False

    all_ok = True

    for i, hit in enumerate(evidence["hits"], start=1):
        offset = hit["offset"]
        recovered = corpus[offset:offset + len(query_bytes)]
        ok = recovered == query_bytes

        print(f"hit={i} offset={offset} verified={ok}")

        if not ok:
            all_ok = False

    if all_ok:
        print("MATCH VERIFIED")
        return True

    print("MISMATCH")
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bundle", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.bundle).resolve()
    base_dir = manifest_path.parent

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    evidence_path = resolve_path(base_dir, manifest["evidence_path"])
    corpus_path = resolve_path(base_dir, manifest["corpus_path"])
    fm_path = resolve_path(base_dir, manifest["fm_path"])
    bwt_path = resolve_path(base_dir, manifest["bwt_path"])
    sa_path = resolve_path(base_dir, manifest["sa_path"])

    ok = True

    ok &= verify_hash(
        "EVIDENCE_SHA256",
        evidence_path,
        manifest["evidence_sha256"],
    )

    ok &= verify_hash(
        "CORPUS_SHA256",
        corpus_path,
        manifest["corpus_sha256"],
    )

    ok &= verify_hash(
        "FM_SHA256",
        fm_path,
        manifest["fm_sha256"],
    )

    ok &= verify_hash(
        "BWT_SHA256",
        bwt_path,
        manifest["bwt_sha256"],
    )

    ok &= verify_hash(
        "SA_SHA256",
        sa_path,
        manifest["sa_sha256"],
    )

    if not ok:
        print("BUNDLE MISMATCH")
        return 2

    evidence = json.loads(evidence_path.read_text(encoding="utf-8"))

    if not replay_evidence(evidence, corpus_path):
        print("QUERY_REPLAY_MISMATCH")
        print("BUNDLE MISMATCH")
        return 3

    print("QUERY_REPLAY_VERIFIED")
    print("BUNDLE VERIFIED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())