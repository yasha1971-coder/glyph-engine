#!/usr/bin/env python3

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--evidence", required=True)

    args = ap.parse_args()

    ev = json.loads(
        Path(args.evidence).read_text(encoding="utf-8")
    )

    corpus_path = ev["corpus_path"]

    if not ev["hits"]:
        print("NO_HITS_IN_EVIDENCE")
        return 1

    corpus = Path(corpus_path).read_bytes()

    query_bytes = bytes.fromhex(ev["query_hex"])

    all_ok = True

    for i, hit in enumerate(ev["hits"], start=1):

        offset = hit["offset"]

        recovered = corpus[
            offset:offset + len(query_bytes)
        ]

        ok = recovered == query_bytes

        print(
            f"hit={i} "
            f"offset={offset} "
            f"verified={ok}"
        )

        if not ok:
            all_ok = False

    if all_ok:
        print("MATCH VERIFIED")
        return 0

    print("MISMATCH")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
