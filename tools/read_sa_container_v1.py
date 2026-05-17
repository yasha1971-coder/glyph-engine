#!/usr/bin/env python3
import argparse
import struct
from pathlib import Path


MAGIC = b"GLYPHSA1"
VERSION = 1
HEADER_SIZE = 40
LITTLE_ENDIAN = 1


def fail(msg: str):
    raise RuntimeError(msg)


def read_header(path: Path):
    file_size = path.stat().st_size

    if file_size < HEADER_SIZE:
        fail(f"SA container too small: {file_size} bytes")

    with path.open("rb") as f:
        header = f.read(HEADER_SIZE)

    magic = header[0:8]
    if magic != MAGIC:
        fail(f"bad SA container magic: {magic!r}")

    version = struct.unpack("<I", header[8:12])[0]
    entry_width = struct.unpack("<I", header[12:16])[0]
    corpus_bytes = struct.unpack("<Q", header[16:24])[0]
    sa_entries = struct.unpack("<Q", header[24:32])[0]
    endian = struct.unpack("<I", header[32:36])[0]
    reserved_flags = struct.unpack("<I", header[36:40])[0]

    if version != VERSION:
        fail(f"unsupported SA container version: {version}")

    if entry_width not in (4, 8):
        fail(f"bad entry_width: {entry_width}")

    if corpus_bytes == 0:
        fail("corpus_bytes must be > 0")

    if sa_entries != corpus_bytes:
        fail(
            f"SA entries mismatch: entries={sa_entries}, "
            f"corpus_bytes={corpus_bytes}"
        )

    if endian != LITTLE_ENDIAN:
        fail(f"unsupported endian marker: {endian}")

    expected_size = HEADER_SIZE + sa_entries * entry_width
    if file_size != expected_size:
        fail(
            f"SA container file size mismatch: expected {expected_size}, "
            f"got {file_size}"
        )

    return {
        "magic": MAGIC.decode("ascii"),
        "version": version,
        "entry_width": entry_width,
        "corpus_bytes": corpus_bytes,
        "sa_entries": sa_entries,
        "endian": endian,
        "reserved_flags": reserved_flags,
        "file_size": file_size,
        "header_size": HEADER_SIZE,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    args = ap.parse_args()

    info = read_header(Path(args.path))

    for k, v in info.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
