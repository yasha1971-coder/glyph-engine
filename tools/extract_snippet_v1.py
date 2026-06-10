#!/usr/bin/env python3

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--offset", type=int, required=True)
    ap.add_argument("--before", type=int, default=80)
    ap.add_argument("--after", type=int, default=160)
    ap.add_argument("--encoding", default="latin1")
    args = ap.parse_args()

    corpus = Path(args.corpus).read_bytes()

    lo = max(0, args.offset - args.before)
    hi = min(len(corpus), args.offset + args.after)

    snippet = corpus[lo:hi].decode(
        args.encoding,
        errors="replace"
    )

    print("offset:", args.offset)
    print("begin:", lo)
    print("end:", hi)
    print()
    print(snippet)


if __name__ == "__main__":
    main()