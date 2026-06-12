#!/usr/bin/env python3
"""
GLYPH Commitment V1 verifier.

Loads a Commitment Object V1 JSON, recomputes every committed value from
the raw corpus and SA bytes, and confirms the commitment.

Exit code 0 only if ALL checks pass. Stdlib only. SHA256 only.

The Merkle rules and build-spec canonicalization are restated here
independently (not imported from the builder) so that this file is a
self-contained verifier: agreement between the two programs is itself
part of what a verification run demonstrates.
"""

import argparse
import hashlib
import json
import sys

EXPECTED_VERSION = "commitment-v1"
EXPECTED_HASH_FUNCTION = "sha256"

# Normative format descriptors (must match the Commitment V1 definition).
CORPUS_TREE_LEAF_FORMAT = "sha256(0x00 || chunk_bytes)"
CORPUS_TREE_INTERNAL_FORMAT = "sha256(0x01 || left_digest32 || right_digest32)"
SA_TREE_LEAF_FORMAT = "sha256(0x00 || chunk_bytes)"
SA_TREE_INTERNAL_FORMAT = "sha256(0x01 || left_digest32 || right_digest32)"
ODD_LEAF_RULE = "promote-last-node-unchanged"

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"


def sha256_file_and_leaves(path, chunk_size):
    file_hasher = hashlib.sha256()
    leaves = []
    with open(path, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            file_hasher.update(chunk)
            leaves.append(hashlib.sha256(LEAF_PREFIX + chunk).digest())
    if not leaves:
        leaves.append(hashlib.sha256(LEAF_PREFIX).digest())
    return file_hasher.hexdigest(), leaves


def merkle_root(leaves):
    level = leaves
    while len(level) > 1:
        nxt = []
        i = 0
        while i + 1 < len(level):
            nxt.append(
                hashlib.sha256(NODE_PREFIX + level[i] + level[i + 1]).digest()
            )
            i += 2
        if i < len(level):
            nxt.append(level[i])
        level = nxt
    return level[0].hex()


def build_spec_hash(commitment_version, chunk_size, hash_function):
    spec = {
        "commitment_version": commitment_version,
        "chunk_size": chunk_size,
        "hash_function": hash_function,
        "corpus_tree_leaf_format": CORPUS_TREE_LEAF_FORMAT,
        "corpus_tree_internal_format": CORPUS_TREE_INTERNAL_FORMAT,
        "sa_tree_leaf_format": SA_TREE_LEAF_FORMAT,
        "sa_tree_internal_format": SA_TREE_INTERNAL_FORMAT,
        "odd_leaf_rule": ODD_LEAF_RULE,
    }
    canonical = json.dumps(
        spec, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def commitment_root(corpus_root_hex, sa_root_hex, spec_hash_hex):
    payload = (corpus_root_hex + sa_root_hex + spec_hash_hex).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def check(label, expected, actual, results):
    ok = (expected == actual)
    status = "OK" if ok else "FAIL"
    print(f"[{status}] {label}")
    print(f"       committed : {expected}")
    print(f"       recomputed: {actual}")
    results.append(ok)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Verify GLYPH Commitment Object V1")
    ap.add_argument("--commitment", required=True, help="path to commitment JSON")
    ap.add_argument("--corpus", default=None,
                    help="override corpus path (defaults to path in commitment)")
    ap.add_argument("--sa", default=None,
                    help="override SA path (defaults to path in commitment)")
    args = ap.parse_args()

    with open(args.commitment, "r", encoding="utf-8") as f:
        c = json.load(f)

    required = [
        "commitment_version", "corpus_path", "sa_path",
        "corpus_sha256", "sa_sha256", "corpus_root", "sa_root",
        "build_spec_hash", "commitment_root", "chunk_size",
        "hash_function", "created_at_utc", "method",
    ]
    missing = [k for k in required if k not in c]
    if missing:
        print(f"[FAIL] commitment JSON missing fields: {missing}")
        print("COMMITMENT VERIFICATION FAILED")
        return 1

    if c["commitment_version"] != EXPECTED_VERSION:
        print(f"[FAIL] unsupported commitment_version: {c['commitment_version']}")
        print("COMMITMENT VERIFICATION FAILED")
        return 1
    if c["hash_function"] != EXPECTED_HASH_FUNCTION:
        print(f"[FAIL] unsupported hash_function: {c['hash_function']}")
        print("COMMITMENT VERIFICATION FAILED")
        return 1

    corpus_path = args.corpus or c["corpus_path"]
    sa_path = args.sa or c["sa_path"]
    chunk_size = int(c["chunk_size"])

    corpus_sha256, corpus_leaves = sha256_file_and_leaves(corpus_path, chunk_size)
    sa_sha256, sa_leaves = sha256_file_and_leaves(sa_path, chunk_size)
    corpus_root = merkle_root(corpus_leaves)
    sa_root = merkle_root(sa_leaves)
    spec_hash = build_spec_hash(
        c["commitment_version"], chunk_size, c["hash_function"]
    )
    root = commitment_root(corpus_root, sa_root, spec_hash)

    results = []
    check("corpus_sha256", c["corpus_sha256"], corpus_sha256, results)
    check("sa_sha256", c["sa_sha256"], sa_sha256, results)
    check("corpus_root", c["corpus_root"], corpus_root, results)
    check("sa_root", c["sa_root"], sa_root, results)
    check("build_spec_hash", c["build_spec_hash"], spec_hash, results)
    check("commitment_root", c["commitment_root"], root, results)

    if all(results):
        print("COMMITMENT VERIFIED")
        return 0
    print("COMMITMENT VERIFICATION FAILED")
    return 1


if __name__ == "__main__":
    sys.exit(main())
