#!/usr/bin/env python3
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from rlbwt_rank_blocks_v1 import (  # noqa: E402
    RLBWTRank,
    RLB_HEADER_BYTES,
    read_rlb_header,
    read_varint,
)


def build_C_from_rlbwt(rlbwt: Path):
    raw_len, run_count, _raw_sha = read_rlb_header(rlbwt)

    freq = [0] * 256

    with rlbwt.open("rb") as f:
        f.seek(RLB_HEADER_BYTES)

        for _ in range(run_count):
            sym_b = f.read(1)
            if len(sym_b) != 1:
                raise EOFError("symbol eof")
            sym = sym_b[0]
            run_len = read_varint(f)
            freq[sym] += run_len

    if sum(freq) != raw_len:
        raise ValueError(f"frequency sum mismatch: {sum(freq)} != {raw_len}")

    C = [0] * 256
    running = 0
    for c in range(256):
        C[c] = running
        running += freq[c]

    return C, freq, raw_len, run_count


def hex_to_bytes(s: str) -> bytes:
    s = s.strip()
    if len(s) % 2 != 0:
        raise ValueError("hex length must be even")
    return bytes.fromhex(s)


def backward_search(pattern: bytes, C, ranker: RLBWTRank):
    l = 0
    r = ranker.raw_len

    for c in reversed(pattern):
        l = C[c] + ranker.rank(c, l)
        r = C[c] + ranker.rank(c, r)
        if l >= r:
            return l, r

    return l, r


def main():
    ap = argparse.ArgumentParser(description="FM backward search over RLBWT rank blocks.")
    ap.add_argument("--rlbwt", required=True)
    ap.add_argument("--rank-index", required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--query")
    g.add_argument("--query-hex")
    ap.add_argument("--expected-l", type=int)
    ap.add_argument("--expected-r", type=int)
    ap.add_argument("--expected-count", type=int)
    args = ap.parse_args()

    rlbwt = Path(args.rlbwt)
    rank_index = Path(args.rank_index)

    if args.query_hex is not None:
        pattern = hex_to_bytes(args.query_hex)
        query_display = args.query_hex
        query_mode = "hex"
    else:
        pattern = args.query.encode("utf-8")
        query_display = args.query
        query_mode = "utf8"

    if not pattern:
        raise SystemExit("empty query not supported")

    C, freq, raw_len, run_count = build_C_from_rlbwt(rlbwt)
    ranker = RLBWTRank(rlbwt, rank_index)

    l, r = backward_search(pattern, C, ranker)
    count = r - l

    ok = True
    errors = []

    if args.expected_l is not None and l != args.expected_l:
        ok = False
        errors.append(f"expected_l {args.expected_l}, got {l}")

    if args.expected_r is not None and r != args.expected_r:
        ok = False
        errors.append(f"expected_r {args.expected_r}, got {r}")

    if args.expected_count is not None and count != args.expected_count:
        ok = False
        errors.append(f"expected_count {args.expected_count}, got {count}")

    result = {
        "ok": ok,
        "query_mode": query_mode,
        "query": query_display,
        "query_hex": pattern.hex(),
        "raw_bwt_bytes": raw_len,
        "run_count": run_count,
        "rank_step": ranker.rank_step,
        "fm_interval": [l, r],
        "match_count": count,
        "errors": errors,
    }

    print(json.dumps(result, indent=2))

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
