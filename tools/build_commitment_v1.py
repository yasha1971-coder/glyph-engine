#!/usr/bin/env python3
"""
GLYPH Commitment V1 builder.

Builds a hash-only Commitment Object V1 binding:
  corpus bytes  -> corpus_root   (Merkle/SHA256)
  SA bytes      -> sa_root       (Merkle/SHA256)
  build spec    -> build_spec_hash
into a single deterministic commitment_root.

Scope: commitment construction only. No proofs of any kind.
Stdlib only. SHA256 only.

Merkle construction (Commitment V1, normative):
  - The input file is split into consecutive chunks of CHUNK_SIZE bytes.
    The final chunk may be shorter. An empty file yields one empty chunk.
  - Leaf hash:          SHA256( 0x00 || chunk_bytes )
  - Internal node hash: SHA256( 0x01 || left_digest || right_digest )
                        (left_digest, right_digest are raw 32-byte digests)
  - Odd-leaf rule:      if a level has an odd number of nodes, the last
                        node is promoted unchanged to the next level
                        (no duplication).
  - The root is the digest of the single node at the top level,
    hex-encoded lowercase.

commitment_root (normative, per spec R7):
  SHA256( utf8(corpus_root_hex) || utf8(sa_root_hex) || utf8(build_spec_hash_hex) )
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone

COMMITMENT_VERSION = "commitment-v1"
HASH_FUNCTION = "sha256"
DEFAULT_CHUNK_SIZE = 4096
METHOD = "merkle-sha256-chunked-v1"

# Normative format descriptors. These strings are committed to via
# build_spec_hash; changing any of them changes the commitment.
CORPUS_TREE_LEAF_FORMAT = "sha256(0x00 || chunk_bytes)"
CORPUS_TREE_INTERNAL_FORMAT = "sha256(0x01 || left_digest32 || right_digest32)"
SA_TREE_LEAF_FORMAT = "sha256(0x00 || chunk_bytes)"
SA_TREE_INTERNAL_FORMAT = "sha256(0x01 || left_digest32 || right_digest32)"
ODD_LEAF_RULE = "promote-last-node-unchanged"

LEAF_PREFIX = b"\x00"
NODE_PREFIX = b"\x01"


def sha256_file_and_leaves(path, chunk_size):
    """Single streaming pass: returns (file_sha256_hex, [leaf digests])."""
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
        # Empty file: one leaf over the empty chunk.
        leaves.append(hashlib.sha256(LEAF_PREFIX).digest())
    return file_hasher.hexdigest(), leaves


def merkle_root(leaves):
    """Deterministic Merkle root per the normative rules above."""
    level = leaves
    while len(level) > 1:
        nxt = []
        i = 0
        while i + 1 < len(level):
            nxt.append(
                hashlib.sha256(NODE_PREFIX + level[i] + level[i + 1]).digest()
            )
            i += 2
        if i < len(level):  # odd node: promote unchanged
            nxt.append(level[i])
        level = nxt
    return level[0].hex()


def build_spec_dict(chunk_size):
    return {
        "commitment_version": COMMITMENT_VERSION,
        "chunk_size": chunk_size,
        "hash_function": HASH_FUNCTION,
        "corpus_tree_leaf_format": CORPUS_TREE_LEAF_FORMAT,
        "corpus_tree_internal_format": CORPUS_TREE_INTERNAL_FORMAT,
        "sa_tree_leaf_format": SA_TREE_LEAF_FORMAT,
        "sa_tree_internal_format": SA_TREE_INTERNAL_FORMAT,
        "odd_leaf_rule": ODD_LEAF_RULE,
    }


def build_spec_hash(chunk_size):
    """SHA256 of the canonical JSON encoding of the build spec."""
    canonical = json.dumps(
        build_spec_dict(chunk_size),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def commitment_root(corpus_root_hex, sa_root_hex, spec_hash_hex):
    payload = (corpus_root_hex + sa_root_hex + spec_hash_hex).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main():
    ap = argparse.ArgumentParser(description="Build GLYPH Commitment Object V1")
    ap.add_argument("--corpus", required=True, help="path to corpus file")
    ap.add_argument("--sa", required=True, help="path to suffix array binary")
    ap.add_argument("--out", required=True, help="output path for commitment JSON")
    ap.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = ap.parse_args()

    if args.chunk_size <= 0:
        print("ERROR: chunk size must be positive", file=sys.stderr)
        return 2

    corpus_sha256, corpus_leaves = sha256_file_and_leaves(args.corpus, args.chunk_size)
    sa_sha256, sa_leaves = sha256_file_and_leaves(args.sa, args.chunk_size)

    corpus_root = merkle_root(corpus_leaves)
    sa_root = merkle_root(sa_leaves)
    spec_hash = build_spec_hash(args.chunk_size)
    root = commitment_root(corpus_root, sa_root, spec_hash)

    commitment = {
        "commitment_version": COMMITMENT_VERSION,
        "corpus_path": args.corpus,
        "sa_path": args.sa,
        "corpus_sha256": corpus_sha256,
        "sa_sha256": sa_sha256,
        "corpus_root": corpus_root,
        "sa_root": sa_root,
        "build_spec_hash": spec_hash,
        "commitment_root": root,
        "chunk_size": args.chunk_size,
        "hash_function": HASH_FUNCTION,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "method": METHOD,
    }

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(commitment, f, indent=2, sort_keys=False)
        f.write("\n")

    print(f"corpus_sha256   : {corpus_sha256}")
    print(f"sa_sha256       : {sa_sha256}")
    print(f"corpus_root     : {corpus_root}")
    print(f"sa_root         : {sa_root}")
    print(f"build_spec_hash : {spec_hash}")
    print(f"commitment_root : {root}")
    print(f"written         : {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
