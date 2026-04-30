import argparse
import json
import pickle
from collections import Counter, defaultdict


def iter_signatures(data: bytes, sig_len: int, stride: int):
    if len(data) < sig_len:
        return
    for i in range(0, len(data) - sig_len + 1, stride):
        yield data[i:i + sig_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sig-len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=4)
    ap.add_argument("--max-df", type=int, default=16,
                    help="Keep only signatures with df <= max_df as rare anchors")
    ap.add_argument("--top-per-chunk", type=int, default=128,
                    help="Max number of rare anchors to keep per chunk")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("=" * 60)
    print("  CHUNK FILTER BUILD V3 (RARE-ANCHOR INDEX)")
    print("=" * 60)
    print(f"  manifest={args.manifest}")
    print(f"  sig_len={args.sig_len}")
    print(f"  stride={args.stride}")
    print(f"  max_df={args.max_df}")
    print(f"  top_per_chunk={args.top_per_chunk}")
    print(f"  out={args.out}")

    # load manifest
    records = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"  chunks={len(records)}")

    # pass 1: per-chunk counts + global df
    per_chunk_counts = {}
    global_df = defaultdict(int)

    for rec in records:
        chunk_id = rec["chunk_id"]
        raw_path = rec["chunk_dir"] + "/chunk.raw.bin"

        with open(raw_path, "rb") as f:
            data = f.read()

        ctr = Counter(iter_signatures(data, args.sig_len, args.stride))
        per_chunk_counts[chunk_id] = ctr

        for sig in ctr.keys():
            global_df[sig] += 1

    total_signatures = len(global_df)
    rare_signatures = sum(1 for _, df in global_df.items() if df <= args.max_df)

    print(f"  total_unique_signatures={total_signatures}")
    print(f"  rare_signatures(df<={args.max_df})={rare_signatures}")

    # pass 2: choose rare anchors per chunk
    chunk_rows = []
    total_kept = 0

    for rec in records:
        chunk_id = rec["chunk_id"]
        ctr = per_chunk_counts[chunk_id]

        candidates = []
        for sig, cnt in ctr.items():
            df = global_df[sig]
            if df <= args.max_df:
                # sort priority:
                # 1) lower df is better
                # 2) higher local count is better
                candidates.append((sig, cnt, df))

        candidates.sort(key=lambda x: (x[2], -x[1], x[0]))
        top = candidates[:args.top_per_chunk]

        row = {
            "chunk_id": chunk_id,
            "num_unique_signatures": len(ctr),
            "num_rare_candidates": len(candidates),
            "rare_anchors": [(sig.hex(), cnt, df) for sig, cnt, df in top],
        }
        chunk_rows.append(row)
        total_kept += len(top)

    out_obj = {
        "version": "chunk_filter_v3",
        "sig_len": args.sig_len,
        "stride": args.stride,
        "max_df": args.max_df,
        "top_per_chunk": args.top_per_chunk,
        "global_df": {sig.hex(): df for sig, df in global_df.items() if df <= args.max_df},
        "chunks": chunk_rows,
    }

    with open(args.out, "wb") as f:
        pickle.dump(out_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  total_kept_anchors={total_kept}")
    print("  done")


if __name__ == "__main__":
    main()