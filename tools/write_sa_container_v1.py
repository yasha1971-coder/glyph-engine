#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path


MAGIC = b"GLYPHSA1"
VERSION = 1
LITTLE_ENDIAN = 1
RESERVED_FLAGS = 0
HEADER_SIZE = 40


def fail(msg: str):
    raise RuntimeError(msg)


def write_container(sa_path: Path, out_path: Path, corpus_bytes: int, entry_width: int):
    if entry_width not in (4, 8):
        fail("entry_width must be 4 or 8")

    sa_size = sa_path.stat().st_size

    if sa_size == 0:
        fail("SA file is empty")

    if sa_size % entry_width != 0:
        fail(f"SA file size is not divisible by entry_width={entry_width}")

    sa_entries = sa_size // entry_width

    if sa_entries != corpus_bytes:
        fail(
            f"SA entries mismatch: entries={sa_entries}, "
            f"corpus_bytes={corpus_bytes}"
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with sa_path.open("rb") as src, out_path.open("wb") as out:
        out.write(MAGIC)
        out.write(struct.pack("<I", VERSION))
        out.write(struct.pack("<I", entry_width))
        out.write(struct.pack("<Q", corpus_bytes))
        out.write(struct.pack("<Q", sa_entries))
        out.write(struct.pack("<I", LITTLE_ENDIAN))
        out.write(struct.pack("<I", RESERVED_FLAGS))

        while True:
            chunk = src.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)

    return {
        "magic": MAGIC.decode("ascii"),
        "version": VERSION,
        "entry_width": entry_width,
        "corpus_bytes": corpus_bytes,
        "sa_entries": sa_entries,
        "output_bytes": out_path.stat().st_size,
        "header_size": HEADER_SIZE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--corpus-bytes", type=int, required=True)
    ap.add_argument("--entry-width", type=int, default=4)
    args = ap.parse_args()

    info = write_container(
        sa_path=Path(args.sa),
        out_path=Path(args.out),
        corpus_bytes=args.corpus_bytes,
        entry_width=args.entry_width,
    )

    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
