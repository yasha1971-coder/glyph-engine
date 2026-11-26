#!/usr/bin/env python3
import argparse
from array import array
import struct
import sys
from pathlib import Path


def read_sa_u32(path: Path):
    size = path.stat().st_size
    if size % 4 != 0:
        raise ValueError("SA file size is not divisible by 4")
    sa = array("I")
    if sa.itemsize != 4:
        raise ValueError("expected a 32-bit unsigned integer array")
    with path.open("rb") as f:
        sa.fromfile(f, size // 4)
    if sys.byteorder != "little":
        sa.byteswap()
    return sa


def write_u32(f, x: int):
    f.write(struct.pack("<I", x))


def write_u64(f, x: int):
    f.write(struct.pack("<Q", x))


def build_fm_core(bwt: bytes, checkpoint_step: int, out_path: Path):
    n = len(bwt)

    freq = [0] * 256
    for b in bwt:
        freq[b] += 1

    C = [0] * 256
    running = 0
    for c in range(256):
        C[c] = running
        running += freq[c]

    checkpoints = []
    counts = [0] * 256

    for block_start in range(0, n + checkpoint_step, checkpoint_step):
        checkpoints.append(list(counts))
        for b in bwt[block_start:block_start + checkpoint_step]:
            counts[b] += 1
        if block_start >= n:
            break

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        f.write(b"FMV1")
        write_u64(f, n)
        write_u32(f, checkpoint_step)
        write_u64(f, len(checkpoints))

        for x in C:
            write_u64(f, x)
        for x in freq:
            write_u64(f, x)

        for row in checkpoints:
            if len(row) != 256:
                raise ValueError("checkpoint row length != 256")
            for x in row:
                write_u64(f, x)

    return {
        "bwt_bytes": n,
        "checkpoint_step": checkpoint_step,
        "num_checkpoints": len(checkpoints),
    }


def build_locate_core(sa, sample_step: int, out_path: Path):
    sample_step = max(1, sample_step)
    sampled = range(0, len(sa), sample_step)

    with out_path.open("wb") as f:
        f.write(b"LOC1")
        write_u64(f, len(sa))
        write_u32(f, sample_step)
        write_u64(f, len(sampled))

        for idx in sampled:
            write_u64(f, idx)
            write_u64(f, sa[idx])

    return {
        "sa_size": len(sa),
        "sample_step": sample_step,
        "sampled_count": len(sampled),
    }


def build_locate_core_from_file(sa_path: Path, sa_size: int, sample_step: int,
                                out_path: Path, block_bytes: int = 1024 * 1024):
    """Write unchanged LOC1 records while retaining at most one SA block.

    Keep read_sa_u32's array API for glyph-search/LOC2 callers.  The evidence
    CLI only needs every sample_step-th value, not a resident copy of the SA.
    """
    if block_bytes <= 0 or block_bytes % 4:
        raise ValueError("SA block size must be a positive multiple of 4")
    sample_step = max(1, sample_step)
    sampled_count = len(range(0, sa_size, sample_step))
    with sa_path.open("rb") as source, out_path.open("wb") as dest:
        dest.write(b"LOC1")
        write_u64(dest, sa_size)
        write_u32(dest, sample_step)
        write_u64(dest, sampled_count)
        row = 0
        next_sample = 0
        while row < sa_size:
            wanted = min(block_bytes, (sa_size - row) * 4)
            block = source.read(wanted)
            if len(block) != wanted:
                raise ValueError("SA file length changed while reading")
            end = row + len(block) // 4
            for index in range(next_sample, end, sample_step):
                value = struct.unpack_from("<I", block, (index - row) * 4)[0]
                dest.write(struct.pack("<QQ", index, value))
                next_sample = index + sample_step
            row = end
        if source.read(1):
            raise ValueError("SA file length changed while reading")
    return {"sa_size": sa_size, "sample_step": sample_step,
            "sampled_count": sampled_count}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--sa", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=128)
    ap.add_argument("--sample-step", type=int, default=16)
    args = ap.parse_args()

    bwt_path = Path(args.bwt)
    sa_path = Path(args.sa)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bwt = bwt_path.read_bytes()
    sa_bytes = sa_path.stat().st_size
    if sa_bytes % 4 != 0:
        raise ValueError("SA file size is not divisible by 4")
    sa_size = sa_bytes // 4

    if sa_size != len(bwt):
        raise ValueError(f"SA length {sa_size} != BWT length {len(bwt)}")

    fm_info = build_fm_core(
        bwt=bwt,
        checkpoint_step=args.checkpoint_step,
        out_path=out_dir / "fm_core.bin",
    )

    loc_info = build_locate_core_from_file(
        sa_path=sa_path,
        sa_size=sa_size,
        sample_step=args.sample_step,
        out_path=out_dir / f"locate_core_s{args.sample_step}.bin",
    )

    print("fm_core:", fm_info)
    print("locate_core:", loc_info)
    print("done")


if __name__ == "__main__":
    main()
