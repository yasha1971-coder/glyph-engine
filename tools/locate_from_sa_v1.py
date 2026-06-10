#!/usr/bin/env python3

import argparse
import struct
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa", required=True, help="SA u32 binary file")
    ap.add_argument("--l", type=int, required=True, help="FM interval left")
    ap.add_argument("--r", type=int, required=True, help="FM interval right")
    ap.add_argument("--original-bytes", type=int, default=None, help="Skip sentinel offsets >= original bytes")
    args = ap.parse_args()

    if args.r < args.l:
        raise SystemExit("ERROR: r < l")

    sa_path = Path(args.sa)

    with sa_path.open("rb") as f:
        for idx in range(args.l, args.r):
            f.seek(idx * 4)
            raw = f.read(4)
            if len(raw) != 4:
                raise SystemExit(f"ERROR: failed to read SA at index {idx}")

            pos = struct.unpack("<I", raw)[0]

            if args.original_bytes is not None and pos >= args.original_bytes:
                continue

            print(pos)


if __name__ == "__main__":
    main()