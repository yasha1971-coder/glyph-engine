#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path


def read_sa_u32(path: Path):
    data = path.read_bytes()
    if len(data) % 4 != 0:
        raise ValueError("SA file size is not divisible by 4")
    return list(struct.unpack("<" + "I" * (len(data) // 4), data))


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
    sampled = [(i, pos) for i, pos in enumerate(sa) if i % sample_step == 0]

    with out_path.open("wb") as f:
        f.write(b"LOC1")
        write_u64(f, len(sa))
        write_u32(f, sample_step)
        write_u64(f, len(sampled))

        for idx, pos in sampled:
            write_u64(f, idx)
            write_u64(f, pos)

    return {
        "sa_size": len(sa),
        "sample_step": sample_step,
        "sampled_count": len(sampled),
    }


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
    sa = read_sa_u32(sa_path)

    if len(sa) != len(bwt):
        raise ValueError(f"SA length {len(sa)} != BWT length {len(bwt)}")

    fm_info = build_fm_core(
        bwt=bwt,
        checkpoint_step=args.checkpoint_step,
        out_path=out_dir / "fm_core.bin",
    )

    loc_info = build_locate_core(
        sa=sa,
        sample_step=args.sample_step,
        out_path=out_dir / f"locate_core_s{args.sample_step}.bin",
    )

    print("fm_core:", fm_info)
    print("locate_core:", loc_info)
    print("done")


if __name__ == "__main__":
    main()
