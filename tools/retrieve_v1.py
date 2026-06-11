#!/usr/bin/env python3

import argparse
import json
import struct
import subprocess
from pathlib import Path


def fm_query(query, fm, bwt):
    hexq = query.encode("latin1").hex() + "\n"

    p = subprocess.run(
        [
            "./build/query_fm_batch_v1",
            fm,
            bwt,
        ],
        input=hexq,
        text=True,
        capture_output=True,
        check=True,
    )

    line = p.stdout.strip()
    if not line:
        raise RuntimeError("empty FM response")

    l, r, cnt = map(int, line.split())
    return l, r, cnt


def locate(sa_path, l, r, original_bytes):
    hits = []

    with open(sa_path, "rb") as f:
        for idx in range(l, r):
            f.seek(idx * 4)

            raw = f.read(4)
            if len(raw) != 4:
                raise RuntimeError(f"bad SA read at {idx}")

            pos = struct.unpack("<I", raw)[0]

            if pos >= original_bytes:
                continue

            hits.append(pos)

    return hits


def snippet(corpus, offset, before=80, after=160):
    lo = max(0, offset - before)
    hi = min(len(corpus), offset + after)

    return corpus[lo:hi].decode(
        "latin1",
        errors="replace"
    )


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--query", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--sa", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--max-hits", type=int, default=5)
    ap.add_argument("--json", action="store_true")

    args = ap.parse_args()

    corpus_bytes = Path(args.corpus).read_bytes()

    l, r, cnt = fm_query(
        args.query,
        args.fm,
        args.bwt
    )

    hits = locate(
        args.sa,
        l,
        r,
        len(corpus_bytes)
    )

    if args.json:

        if not hits:
            out = {
                "query": args.query,
                "query_hex": args.query.encode("latin1").hex(),
                "match": False,
                "count": 0,
                "interval": [l, r],
                "method": "sentinel-safe-fm-sa-v1",
                "index_tag": "retrieval-v1",
                "corpus_path": args.corpus,
                "verified": False,
                "hits": []
            }

            print(json.dumps(out, ensure_ascii=False, indent=2))
            return

        out = {
            "query": args.query,
            "query_hex": args.query.encode("latin1").hex(),
            "match": True,
            "count": cnt,
            "interval": [l, r],
            "method": "sentinel-safe-fm-sa-v1",
            "index_tag": "retrieval-v1",
            "corpus_path": args.corpus,
            "verified": True,
            "hits": []
        }

        for pos in hits[:args.max_hits]:
            out["hits"].append({
                "offset": pos,
                "length": len(args.query.encode("latin1")),
                "snippet": snippet(corpus_bytes, pos)
            })

        print(json.dumps(out, ensure_ascii=False, indent=2))
        return

    print("query:", args.query)
    print("interval:", l, r)
    print("count:", cnt)
    print()

    if not hits:
        print("NO HITS")
        return

    for i, pos in enumerate(hits[:args.max_hits], start=1):
        print("=" * 80)
        print("hit:", i)
        print("offset:", pos)
        print()
        print(snippet(corpus_bytes, pos))
        print()


if __name__ == "__main__":
    main()