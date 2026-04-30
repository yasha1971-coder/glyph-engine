import argparse
import json
import pickle
from collections import Counter, defaultdict


def iter_kmers(data: bytes, k: int):
    if len(data) < k:
        return
    for i in range(0, len(data) - k + 1):
        yield data[i:i + k]


def compute_minimizers(data: bytes, k: int, w: int):
    kmers = list(iter_kmers(data, k))
    if not kmers:
        return []

    if len(kmers) <= w:
        return [min(kmers)]

    mins = []
    for i in range(0, len(kmers) - w + 1):
        window = kmers[i:i + w]
        mins.append(min(window))
    return mins


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--w", type=int, default=24)
    ap.add_argument("--max-df", type=int, default=64,
                    help="keep minimizers with df <= max_df in global_df export")
    ap.add_argument("--top-per-chunk", type=int, default=512,
                    help="optional cap for stored minimizers per chunk")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("=" * 60)
    print("  CHUNK FILTER BUILD V4 (MINIMIZER INDEX)")
    print("=" * 60)
    print(f"  manifest={args.manifest}")
    print(f"  k={args.k}")
    print(f"  w={args.w}")
    print(f"  max_df={args.max_df}")
    print(f"  top_per_chunk={args.top_per_chunk}")
    print(f"  out={args.out}")

    records = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"  chunks={len(records)}")

    per_chunk_counts = {}
    global_df = defaultdict(int)

    # pass 1
    for rec in records:
        chunk_id = rec["chunk_id"]
        raw_path = rec["chunk_dir"] + "/chunk.raw.bin"

        with open(raw_path, "rb") as f:
            data = f.read()

        mins = compute_minimizers(data, args.k, args.w)
        ctr = Counter(mins)
        per_chunk_counts[chunk_id] = ctr

        for tok in ctr.keys():
            global_df[tok] += 1

    total_unique = len(global_df)
    kept_global_df = sum(1 for _, df in global_df.items() if df <= args.max_df)

    print(f"  total_unique_minimizers={total_unique}")
    print(f"  kept_global_df(df<={args.max_df})={kept_global_df}")

    # pass 2
    chunk_rows = []
    total_kept = 0

    for rec in records:
        chunk_id = rec["chunk_id"]
        ctr = per_chunk_counts[chunk_id]

        items = []
        for tok, cnt in ctr.items():
            df = global_df[tok]
            items.append((tok, cnt, df))

        # rank:
        # 1) higher local count
        # 2) lower df
        # 3) token bytes
        items.sort(key=lambda x: (-x[1], x[2], x[0]))

        if args.top_per_chunk > 0:
            items = items[:args.top_per_chunk]

        row = {
            "chunk_id": chunk_id,
            "num_unique_minimizers": len(ctr),
            "minimizers": [(tok.hex(), cnt, df) for tok, cnt, df in items],
        }
        chunk_rows.append(row)
        total_kept += len(items)

    out_obj = {
        "version": "chunk_filter_v4_minimizer",
        "k": args.k,
        "w": args.w,
        "max_df": args.max_df,
        "top_per_chunk": args.top_per_chunk,
        "global_df": {tok.hex(): df for tok, df in global_df.items() if df <= args.max_df},
        "chunks": chunk_rows,
    }

    with open(args.out, "wb") as f:
        pickle.dump(out_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  total_kept_minimizers={total_kept}")
    print("  done")


if __name__ == "__main__":
    main()