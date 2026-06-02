#!/usr/bin/env python3

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Text query file, one query per line")
    ap.add_argument("--encoding", default="latin1", help="Input encoding: latin1 or utf-8")
    ap.add_argument("--output", default=None, help="Output hex query file")
    args = ap.parse_args()

    src = Path(args.input)
    lines = [
        x.strip()
        for x in src.read_text(encoding=args.encoding).splitlines()
        if x.strip()
    ]

    out_lines = [s.encode(args.encoding).hex() for s in lines]

    if args.output:
        Path(args.output).write_text("\n".join(out_lines) + "\n")
    else:
        print("\n".join(out_lines))


if __name__ == "__main__":
    main()
