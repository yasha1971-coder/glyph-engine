#!/usr/bin/env python3
import argparse
import bisect
import hashlib
import json
import random
import struct
from pathlib import Path

RLB_MAGIC = b"RLB1"
RANK_MAGIC = b"RLR1"
VERSION = 1
RLB_HEADER_BYTES = 4 + 4 + 8 + 8 + 32


def u32(x): return struct.pack("<I", x)
def u64(x): return struct.pack("<Q", x)


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


def read_varint(f):
    shift = 0
    x = 0
    while True:
        b = f.read(1)
        if not b:
            raise EOFError("varint eof")
        v = b[0]
        x |= (v & 0x7F) << shift
        if (v & 0x80) == 0:
            return x
        shift += 7
        if shift > 63:
            raise ValueError("varint too large")


def sha256_file(path: Path) -> bytes:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1024 * 1024)
            if not b:
                break
            h.update(b)
    return h.digest()


def read_rlb_header(path: Path):
    with path.open("rb") as f:
        magic = f.read(4)
        if magic != RLB_MAGIC:
            raise ValueError(f"bad RLB magic: {magic!r}")
        version = read_u32(f)
        if version != VERSION:
            raise ValueError(f"bad RLB version: {version}")
        raw_len = read_u64(f)
        run_count = read_u64(f)
        raw_sha = f.read(32)
        if len(raw_sha) != 32:
            raise EOFError("raw sha")
    return raw_len, run_count, raw_sha


def build_rank_index(rlbwt: Path, out: Path, rank_step: int):
    if rank_step <= 0:
        raise ValueError("rank_step must be > 0")

    raw_len, run_count, raw_sha = read_rlb_header(rlbwt)
    rlb_sha = sha256_file(rlbwt)

    out.parent.mkdir(parents=True, exist_ok=True)

    records = 0
    counts = [0] * 256
    raw_pos = 0
    next_cp = 0
    run_i = 0

    with rlbwt.open("rb") as f, out.open("wb") as w:
        f.seek(RLB_HEADER_BYTES)

        w.write(RANK_MAGIC)
        w.write(u32(VERSION))
        w.write(u64(raw_len))
        w.write(u32(rank_step))

        num_blocks_pos = w.tell()
        w.write(u64(0))  # patched later

        w.write(rlb_sha)

        def write_record(cp_raw_pos, stream_offset, run_offset, cp_counts):
            nonlocal records
            w.write(u64(cp_raw_pos))
            w.write(u64(stream_offset))
            w.write(u64(run_offset))
            for x in cp_counts:
                w.write(u64(x))
            records += 1

        while run_i < run_count:
            stream_offset = f.tell()
            sym_b = f.read(1)
            if len(sym_b) != 1:
                raise EOFError("symbol eof")
            sym = sym_b[0]
            run_len = read_varint(f)

            run_start = raw_pos
            run_end = raw_pos + run_len

            while next_cp <= run_end and next_cp <= raw_len:
                off = next_cp - run_start
                if off < 0:
                    raise RuntimeError("negative checkpoint offset")

                cp_counts = counts.copy()
                if off > 0:
                    cp_counts[sym] += off

                write_record(next_cp, stream_offset, off, cp_counts)
                next_cp += rank_step

            counts[sym] += run_len
            raw_pos = run_end
            run_i += 1

        if raw_pos != raw_len:
            raise ValueError(f"raw_len mismatch {raw_pos} != {raw_len}")

        if records == 0 or (records - 1) * rank_step < raw_len:
            # Ensure there is always a checkpoint at raw_len.
            write_record(raw_len, stream_offset, run_len, counts.copy())

        end = w.tell()
        w.seek(num_blocks_pos)
        w.write(u64(records))
        w.seek(end)

    result = {
        "ok": True,
        "rlbwt": str(rlbwt),
        "rank_index": str(out),
        "raw_bwt_bytes": raw_len,
        "run_count": run_count,
        "rank_step": rank_step,
        "rank_blocks": records,
        "rank_index_bytes": out.stat().st_size,
        "rlbwt_bytes": rlbwt.stat().st_size,
        "combined_bytes": out.stat().st_size + rlbwt.stat().st_size,
        "combined_ratio_vs_bwt": (out.stat().st_size + rlbwt.stat().st_size) / raw_len,
    }

    print(json.dumps(result, indent=2))
    return result


class RLBWTRank:
    def __init__(self, rlbwt: Path, rank_index: Path):
        self.rlbwt = rlbwt
        self.rank_index = rank_index

        raw_len, run_count, raw_sha = read_rlb_header(rlbwt)
        self.raw_len = raw_len
        self.run_count = run_count

        expected_rlb_sha = sha256_file(rlbwt)

        self.records = []
        self.raw_positions = []

        with rank_index.open("rb") as f:
            magic = f.read(4)
            if magic != RANK_MAGIC:
                raise ValueError(f"bad rank magic: {magic!r}")

            version = read_u32(f)
            if version != VERSION:
                raise ValueError(f"bad rank version: {version}")

            idx_raw_len = read_u64(f)
            self.rank_step = read_u32(f)
            self.num_blocks = read_u64(f)
            idx_rlb_sha = f.read(32)

            if idx_raw_len != raw_len:
                raise ValueError("raw_len mismatch between RLBWT and rank index")
            if idx_rlb_sha != expected_rlb_sha:
                raise ValueError("RLBWT sha mismatch")

            for _ in range(self.num_blocks):
                cp_raw_pos = read_u64(f)
                stream_offset = read_u64(f)
                run_offset = read_u64(f)
                counts = [read_u64(f) for _ in range(256)]
                self.raw_positions.append(cp_raw_pos)
                self.records.append((cp_raw_pos, stream_offset, run_offset, counts))

    def rank(self, c: int, pos: int) -> int:
        if not (0 <= c <= 255):
            raise ValueError("symbol must be 0..255")
        if not (0 <= pos <= self.raw_len):
            raise ValueError("pos out of range")

        i = bisect.bisect_right(self.raw_positions, pos) - 1
        if i < 0:
            raise ValueError("no checkpoint before pos")

        cp_raw_pos, stream_offset, run_offset, counts = self.records[i]
        total = counts[c]
        cur = cp_raw_pos

        if cur == pos:
            return total

        first = True
        with self.rlbwt.open("rb") as f:
            f.seek(stream_offset)

            while cur < pos:
                sym_b = f.read(1)
                if len(sym_b) != 1:
                    raise EOFError("rank scan symbol eof")

                sym = sym_b[0]
                run_len = read_varint(f)

                skip = run_offset if first else 0
                first = False

                if skip > run_len:
                    raise ValueError("run_offset > run_len")

                available = run_len - skip

                if available == 0:
                    continue

                take = min(available, pos - cur)
                if sym == c:
                    total += take
                cur += take

        if cur != pos:
            raise ValueError(f"rank stopped at {cur}, wanted {pos}")

        return total


def verify_rank(bwt: Path, rlbwt: Path, rank_index: Path, trials: int, seed: int):
    data = bwt.read_bytes()
    rr = RLBWTRank(rlbwt, rank_index)

    if len(data) != rr.raw_len:
        raise ValueError("BWT length mismatch")

    rng = random.Random(seed)

    pairs = []
    n = len(data)

    for pos in [0, 1, min(2, n), n // 2, max(0, n - 1), n]:
        for c in [0, 10, 32, 45, 48, 65, 95, 101, 116, 255]:
            pairs.append((pos, c))

    for _ in range(trials):
        pos = rng.randint(0, n)
        c = rng.randrange(256)
        pairs.append((pos, c))

    pairs_sorted = sorted(enumerate(pairs), key=lambda x: x[1][0])

    counts = [0] * 256
    cur = 0
    expected = [None] * len(pairs)

    for idx, (pos, c) in pairs_sorted:
        while cur < pos:
            counts[data[cur]] += 1
            cur += 1
        expected[idx] = counts[c]

    bad = []
    for i, (pos, c) in enumerate(pairs):
        got = rr.rank(c, pos)
        exp = expected[i]
        if got != exp:
            bad.append({
                "pos": pos,
                "symbol": c,
                "expected": exp,
                "got": got,
            })
            if len(bad) >= 10:
                break

    result = {
        "ok": len(bad) == 0,
        "bwt": str(bwt),
        "rlbwt": str(rlbwt),
        "rank_index": str(rank_index),
        "trials": trials,
        "checks": len(pairs),
        "bad": bad,
        "rank_step": rr.rank_step,
        "raw_bwt_bytes": rr.raw_len,
        "rlbwt_bytes": rlbwt.stat().st_size,
        "rank_index_bytes": rank_index.stat().st_size,
        "combined_bytes": rlbwt.stat().st_size + rank_index.stat().st_size,
        "combined_ratio_vs_bwt": (rlbwt.stat().st_size + rank_index.stat().st_size) / rr.raw_len,
    }

    print(json.dumps(result, indent=2))

    if bad:
        raise SystemExit(1)


def main():
    ap = argparse.ArgumentParser(description="RLBWT Rank Blocks V1.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build")
    b.add_argument("--rlbwt", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--rank-step", type=int, default=8192)

    r = sub.add_parser("rank")
    r.add_argument("--rlbwt", required=True)
    r.add_argument("--rank-index", required=True)
    r.add_argument("--symbol", type=int, required=True)
    r.add_argument("--pos", type=int, required=True)

    v = sub.add_parser("verify")
    v.add_argument("--bwt", required=True)
    v.add_argument("--rlbwt", required=True)
    v.add_argument("--rank-index", required=True)
    v.add_argument("--trials", type=int, default=1000)
    v.add_argument("--seed", type=int, default=1)

    args = ap.parse_args()

    if args.cmd == "build":
        build_rank_index(Path(args.rlbwt), Path(args.out), args.rank_step)
    elif args.cmd == "rank":
        rr = RLBWTRank(Path(args.rlbwt), Path(args.rank_index))
        print(rr.rank(args.symbol, args.pos))
    elif args.cmd == "verify":
        verify_rank(Path(args.bwt), Path(args.rlbwt), Path(args.rank_index), args.trials, args.seed)


if __name__ == "__main__":
    main()
