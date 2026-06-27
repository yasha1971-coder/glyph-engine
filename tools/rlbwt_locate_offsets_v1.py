#!/usr/bin/env python3
import argparse
import bisect
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from rlbwt_rank_blocks_v1 import RLBWTRank, read_varint  # noqa: E402
from rlbwt_fm_query_v1 import build_C_from_rlbwt          # noqa: E402


def read_u32(f):
    b = f.read(4)
    if len(b) != 4:
        raise EOFError("read_u32")
    return struct.unpack("<I", b)[0]


def read_u64(f):
    b = f.read(8)
    if len(b) != 8:
        raise EOFError("read_u64")
    return struct.unpack("<Q", b)[0]


def load_locate_core(path: Path):
    sampled = {}

    with path.open("rb") as f:
        magic = f.read(4)
        if magic != b"LOC1":
            raise ValueError(f"bad LOC1 magic: {magic!r}")

        sa_size = read_u64(f)
        sample_step = read_u32(f)
        sampled_count = read_u64(f)

        for _ in range(sampled_count):
            idx = read_u64(f)
            pos = read_u64(f)
            sampled[idx] = pos

    return {
        "sa_size": sa_size,
        "sample_step": sample_step,
        "sampled_count": sampled_count,
        "sampled": sampled,
    }


def rlbwt_symbol_at(ranker: RLBWTRank, pos: int) -> int:
    if not (0 <= pos < ranker.raw_len):
        raise ValueError(f"symbol pos out of range: {pos}")

    rec_i = bisect.bisect_right(ranker.raw_positions, pos) - 1
    if rec_i < 0:
        raise ValueError("no rank checkpoint before pos")

    cp_raw_pos, stream_offset, run_offset, _counts = ranker.records[rec_i]

    cur = cp_raw_pos
    first = True

    with ranker.rlbwt.open("rb") as f:
        f.seek(stream_offset)

        while cur <= pos:
            sym_b = f.read(1)
            if len(sym_b) != 1:
                raise EOFError("symbol_at eof")

            sym = sym_b[0]
            run_len = read_varint(f)

            skip = run_offset if first else 0
            first = False

            if skip > run_len:
                raise ValueError("run_offset > run_len")

            available = run_len - skip
            if available == 0:
                continue

            if pos < cur + available:
                return sym

            cur += available

    raise ValueError(f"symbol_at failed for pos={pos}")


def locate_row(row: int, C, ranker: RLBWTRank, sampled: dict, sa_size: int):
    i = row
    steps = 0
    max_guard = sa_size + 1

    while i not in sampled:
        c = rlbwt_symbol_at(ranker, i)
        i = C[c] + ranker.rank(c, i)
        steps += 1

        if steps > max_guard:
            raise RuntimeError("locate exceeded guard; possible LF/sample bug")

    pos = (sampled[i] + steps) % sa_size
    return pos, steps


def main():
    ap = argparse.ArgumentParser(description="Locate offsets from FM interval using RLBWT rank/LF and LOC1 sampled SA.")
    ap.add_argument("--rlbwt", required=True)
    ap.add_argument("--rank-index", required=True)
    ap.add_argument("--locate-core", required=True)
    ap.add_argument("--l", type=int, required=True)
    ap.add_argument("--r", type=int, required=True)
    ap.add_argument("--expected-offset", type=int, action="append", default=[])
    args = ap.parse_args()

    rlbwt = Path(args.rlbwt)
    rank_index = Path(args.rank_index)
    locate_core = Path(args.locate_core)

    C, _freq, raw_len, run_count = build_C_from_rlbwt(rlbwt)
    ranker = RLBWTRank(rlbwt, rank_index)
    loc = load_locate_core(locate_core)

    if loc["sa_size"] != raw_len:
        raise SystemExit(f"sa_size/raw_len mismatch: {loc['sa_size']} != {raw_len}")

    offsets = []
    total_steps = 0
    max_steps = 0

    for row in range(args.l, args.r):
        pos, steps = locate_row(row, C, ranker, loc["sampled"], loc["sa_size"])
        offsets.append(pos)
        total_steps += steps
        max_steps = max(max_steps, steps)

    offsets = sorted(offsets)

    ok = True
    errors = []

    for x in args.expected_offset:
        if x not in offsets:
            ok = False
            errors.append(f"expected offset {x} not found")

    result = {
        "ok": ok,
        "mode": "rlbwt_locate_offsets_v1",
        "raw_bwt_bytes": raw_len,
        "run_count": run_count,
        "rank_step": ranker.rank_step,
        "sample_step": loc["sample_step"],
        "interval": [args.l, args.r],
        "count": args.r - args.l,
        "offsets": offsets,
        "total_steps": total_steps,
        "max_steps": max_steps,
        "errors": errors,
    }

    print(json.dumps(result, indent=2))

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
