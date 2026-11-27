#!/usr/bin/env python3
"""Expected count and bounded SA-prefix coordinates from raw bytes only.

No GLYPH, Move-r, index, suffix-array file, or native library is imported.
Full suffix materialization is restricted to tiny self-check fixtures.
"""
import argparse
from array import array
import bisect
import functools
import hashlib
import heapq
import json
from pathlib import Path
import sys

CORPUS_BYTES = 50_000_000
CORPUS_SHA256 = "974837f10883d6d0b4b2d4896ff9c4adc23363fced1f5f9b031f287c54c6b437"
QUERY_SHA256 = "b9776d7ddf459c9ad5b0e1d6ac61e27befb5e99fd62446677600d7cacef544d0"
LIMITS = (1, 10, 100)


def all_matches(data, query):
    if not query:
        raise ValueError("empty query is outside this experiment")
    result = array("I")
    assert result.itemsize == 4
    start = 0
    while True:
        pos = data.find(query, start)
        if pos < 0:
            return result
        result.append(pos)
        start = pos + 1  # overlapping occurrences are included


def suffix_compare(data, a, b):
    if a == b:
        return 0
    # Bound each temporary slice; repeated passages may require several chunks.
    chunk = 64
    while True:
        left, right = data[a:a + chunk], data[b:b + chunk]
        if left != right:
            return -1 if left < right else 1
        if len(left) < chunk:
            return 0
        a += chunk
        b += chunk


def select_sa_prefix(data, positions, limit):
    key = functools.cmp_to_key(lambda a, b: suffix_compare(data, a, b))
    return heapq.nsmallest(limit, positions, key=key)


def fixtures():
    cases = [
        (b"aaaaa", b"aaa", [0, 1, 2]),
        (b"ababa", b"aba", [0, 2]),
        (b"the", b"the", [0]),
        (b"the-tail", b"the", [0]),
        (b"head-the", b"the", [5]),
        (b"prefix-middle-suffix", b"middle", [7]),
        (b"abcdef", b"missing", []),
        (b"", b"the", []),
    ]
    log = []
    for data, query, expected in cases:
        found = list(all_matches(data, query))
        assert found == expected
        brute = [i for i in range(len(data) - len(query) + 1)
                 if data[i:i + len(query)] == query]
        assert found == brute
        assert len(data) < 4096
        reference = sorted(found, key=lambda i: data[i:])
        for limit in LIMITS:
            selected = select_sa_prefix(data, found, limit)
            assert selected == reference[:limit]
        log.append({"data_hex": data.hex(), "query_hex": query.hex(),
                    "positions": found, "sa_order": reference, "ok": True})
    return log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, required=True)
    ap.add_argument("--query", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    fixture_log = fixtures()
    data, query = args.corpus.read_bytes(), args.query.read_bytes()
    corpus_sha, query_sha = hashlib.sha256(data).hexdigest(), hashlib.sha256(query).hexdigest()
    if len(data) != CORPUS_BYTES or corpus_sha != CORPUS_SHA256:
        raise ValueError("pinned corpus size or SHA-256 mismatch")
    if query != b"the" or query_sha != QUERY_SHA256:
        raise ValueError("pinned query bytes or SHA-256 mismatch")
    if b"\0" in data:
        raise ValueError("NUL-free profile precondition failed; do not sanitize")
    positions = all_matches(data, query)
    selected_sa_order = select_sa_prefix(data, positions, max(LIMITS))
    by_limit = {str(k): sorted(selected_sa_order[:k]) for k in LIMITS}
    for offsets in by_limit.values():
        for pos in offsets:
            i = bisect.bisect_left(positions, pos)
            assert i < len(positions) and positions[i] == pos
            assert 0 <= pos <= len(data) - len(query)
            assert data[pos:pos + len(query)] == query
        assert len(offsets) == len(set(offsets))
    assert set(by_limit["1"]) <= set(by_limit["10"]) <= set(by_limit["100"])
    args.out.mkdir(parents=True, exist_ok=False)
    if sys.byteorder != "little":
        positions.byteswap()
    all_file = args.out / "all-matches.u32le"
    with all_file.open("xb") as dest:
        positions.tofile(dest)
    hist10 = [786213, 4736688, 5595028, 7801431, 8761711,
              19383215, 20264073, 46448252, 46448984, 47354405]
    result = {
        "ok": True, "kind": "independent_raw_byte_oracle_not_index_benchmark",
        "corpus": {"path": str(args.corpus.resolve()), "bytes": len(data), "sha256": corpus_sha},
        "query": {"path": str(args.query.resolve()), "bytes": len(query), "hex": query.hex(), "sha256": query_sha},
        "count": len(positions), "overlaps": True,
        "selection": "first K matching suffixes in lexicographic raw-byte order; output selected positions in numeric order",
        "selected_sa_order_first100": selected_sa_order,
        "offsets_by_limit": by_limit, "all_returned_bytes_check": True,
        "all_matches": {"path": str(all_file.resolve()), "bytes": all_file.stat().st_size,
                        "format": "little-endian unsigned u32 in increasing text-offset order",
                        "sha256": hashlib.sha256(all_file.read_bytes()).hexdigest()},
        "historical_count_matches": len(positions) == 761402,
        "historical_k10_matches": by_limit["10"] == hist10,
        "historical_plaintext_sha256_known": False,
        "historical_corpus_identity_claimed": False,
        "fixture_checks": fixture_log,
        "native_engine_queries_started": 0, "performance_samples_started": 0,
    }
    (args.out / "oracle.json").write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
