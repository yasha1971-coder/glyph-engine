#!/usr/bin/env python3
import hashlib
import json
import struct
from pathlib import Path


FORMAT = "GLYPH_BINARY_SAFE_FM_TINY_FIXTURE_V1"
SENTINEL = 256


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_sa(symbols):
    return sorted(range(len(symbols)), key=lambda i: symbols[i:])


def build_bwt(symbols, sa):
    n = len(symbols)
    return [symbols[(i - 1) % n] for i in sa]


def build_C(bwt):
    counts = {}
    for x in bwt:
        counts[x] = counts.get(x, 0) + 1

    C = {}
    total = 0
    for x in sorted(counts):
        C[x] = total
        total += counts[x]
    return C


def occ(bwt, sym, pos):
    return sum(1 for x in bwt[:pos] if x == sym)


def backward_search(bwt, C, query_symbols):
    l, r = 0, len(bwt)
    for c in reversed(query_symbols):
        if c not in C:
            return 0, 0
        l = C[c] + occ(bwt, c, l)
        r = C[c] + occ(bwt, c, r)
        if l >= r:
            return l, r
    return l, r


def main() -> int:
    out_dir = Path("examples/binary-safe-boundary/out/fm_tiny_v1")
    out_dir.mkdir(parents=True, exist_ok=True)

    source = bytes([0x41, 0x00, 0x42, 0x00, 0x43, 0x00, 0x42])
    query = bytes([0x00, 0x42])

    symbols = list(source) + [SENTINEL]
    query_symbols = list(query)

    sa = build_sa(symbols)
    bwt = build_bwt(symbols, sa)
    C = build_C(bwt)
    l, r = backward_search(bwt, C, query_symbols)

    raw_offsets = sa[l:r]
    source_len = len(source)
    query_len = len(query)

    # Evidence semantics: only real, non-wrapping source offsets are valid.
    offsets = []
    rejected = []

    for off in raw_offsets:
        if off + query_len <= source_len and source[off:off + query_len] == query:
            offsets.append(off)
        else:
            rejected.append(off)

    offsets = sorted(offsets)

    expected_offsets = [1, 5]
    expected_count = 2

    byte_check = all(source[o:o + query_len] == query for o in offsets)

    result = {
        "format": FORMAT,
        "source": {
            "bytes": len(source),
            "hex": source.hex(),
            "sha256": sha256_bytes(source),
            "nul_bytes": source.count(b"\x00"),
        },
        "symbol_model": {
            "data_min": 0,
            "data_max": 255,
            "virtual_sentinel": SENTINEL,
            "symbols": symbols,
            "sentinel_is_corpus_byte": False,
        },
        "query": {
            "bytes": len(query),
            "hex": query.hex(),
            "sha256": sha256_bytes(query),
            "symbols": query_symbols,
        },
        "sa": sa,
        "bwt": bwt,
        "C": {str(k): v for k, v in C.items()},
        "retrieval": {
            "fm_interval": [l, r],
            "match_count": r - l,
            "raw_sa_offsets": raw_offsets,
            "offsets": offsets,
            "rejected_non_evidence_offsets": rejected,
        },
        "expected": {
            "match_count": expected_count,
            "offsets": expected_offsets,
        },
        "byte_check": {
            "all_offsets_match_query": byte_check,
        },
        "invariants": {
            "source_contains_nul": source.count(b"\x00") > 0,
            "query_contains_nul": query.count(b"\x00") > 0,
            "sentinel_is_256": symbols[-1] == SENTINEL,
            "sentinel_not_in_source_symbols": SENTINEL not in symbols[:-1],
            "fm_count_matches_expected": (r - l) == expected_count,
            "offsets_match_expected": offsets == expected_offsets,
            "byte_check_true": byte_check is True,
            "no_rejected_offsets": rejected == [],
        },
    }

    ok = all(result["invariants"].values())

    out = out_dir / "binary_safe_fm_tiny_fixture_v1.json"
    out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": ok,
        "out": str(out),
        "source_hex": source.hex(),
        "query_hex": query.hex(),
        "fm_interval": [l, r],
        "match_count": r - l,
        "offsets": offsets,
        "byte_check": byte_check,
        "rejected": rejected,
    }, indent=2, sort_keys=True))

    if not ok:
        print(json.dumps(result["invariants"], indent=2, sort_keys=True))
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
