import argparse
import json
import pickle
from collections import defaultdict, Counter


def iter_signatures(data: bytes, sig_len: int, stride: int):
    if len(data) < sig_len:
        return
    for i in range(0, len(data) - sig_len + 1, stride):
        yield data[i:i + sig_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sig-len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--max-df", type=int, default=32)   # ключевой параметр
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print("=" * 60)
    print("  CHUNK FILTER BUILD V2 (INVERTED DF-AWARE)")
    print("=" * 60)

    # load manifest
    records = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))

    print(f"  chunks={len(records)}")
    print(f"  sig_len={args.sig_len}")
    print(f"  stride={args.stride}")
    print(f"  max_df={args.max_df}")

    # inverted index
    inv = defaultdict(lambda: defaultdict(int))  # sig -> {chunk_id: count}

    for rec in records:
        chunk_id = rec["chunk_id"]
        raw_path = rec["chunk_dir"] + "/chunk.raw.bin"

        with open(raw_path, "rb") as f:
            data = f.read()

        counts = Counter(iter_signatures(data, args.sig_len, args.stride))

        for sig, cnt in counts.items():
            inv[sig][chunk_id] = cnt

    print(f"  total_signatures={len(inv)}")

    # DF filtering
    inv_filtered = {}
    for sig, chunk_map in inv.items():
        df = len(chunk_map)
        if df <= args.max_df:
            inv_filtered[sig] = chunk_map

    print(f"  kept_signatures={len(inv_filtered)}")

    out_obj = {
        "version": "chunk_filter_v2",
        "sig_len": args.sig_len,
        "stride": args.stride,
        "max_df": args.max_df,
        "inv": {
            sig.hex(): chunk_map
            for sig, chunk_map in inv_filtered.items()
        }
    }

    with open(args.out, "wb") as f:
        pickle.dump(out_obj, f)

    print(f"  saved: {args.out}")


if __name__ == "__main__":
    main()