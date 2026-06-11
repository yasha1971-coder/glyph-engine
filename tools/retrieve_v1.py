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


def write_evidence(path, obj):
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8"
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
    ap.add_argument("--evidence-out")

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

    evidence = {
        "query": args.query,
        "query_hex": args.query.encode("latin1").hex(),
        "match": bool(hits),
        "count": cnt if hits else 0,
        "interval": [l, r],
        "method": "sentinel-safe-fm-sa-v1",
        "index_tag": "retrieval-v1",
        "corpus_path": args.corpus,
        "fm_path": args.fm,
        "bwt_path": args.bwt,
        "sa_path": args.sa,
        "verified": bool(hits),
        "hits": []
    }

    for pos in hits[:args.max_hits]:
        evidence["hits"].append({
            "offset": pos,
            "length": len(args.query.encode("latin1")),
            "snippet": snippet(corpus_bytes, pos)
        })

    if args.evidence_out:
        write_evidence(args.evidence_out, evidence)

    if args.json:
        print(json.dumps(evidence, ensure_ascii=False, indent=2))
        return

    print("query:", args.query)
    print("interval:", l, r)
    print("count:", cnt)
    print()

    if not hits:
        print("NO HITS")
        return

    for i, hit in enumerate(evidence["hits"], start=1):
        print("=" * 80)
        print("hit:", i)
        print("offset:", hit["offset"])
        print()
        print(hit["snippet"])
        print()


if __name__ == "__main__":
    main()