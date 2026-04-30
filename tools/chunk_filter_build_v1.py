import argparse
import json
import pickle
from collections import Counter
from pathlib import Path


def iter_signatures(data: bytes, sig_len: int, stride: int):
    if len(data) < sig_len:
        return
    for i in range(0, len(data) - sig_len + 1, stride):
        yield data[i:i + sig_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sig-len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--top-per-chunk", type=int, default=256)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f]

    print("=" * 60)
    print("  CHUNK FILTER BUILD V1")
    print("=" * 60)
    print(f"  manifest={manifest_path}")
    print(f"  chunks={len(records)}")
    print(f"  sig_len={args.sig_len}")
    print(f"  stride={args.stride}")
    print(f"  top_per_chunk={args.top_per_chunk}")
    print(f"  out={args.out}")

    filter_obj = {
        "version": "chunk_filter_v1",
        "sig_len": args.sig_len,
        "stride": args.stride,
        "top_per_chunk": args.top_per_chunk,
        "chunks": [],
    }

    total_unique = 0
    total_kept = 0

    for rec in records:
        chunk_id = rec["chunk_id"]
        chunk_dir = Path(rec["chunk_dir"])
        raw_path = chunk_dir / "chunk.raw.bin"

        with open(raw_path, "rb") as f:
            raw = f.read()

        ctr = Counter(iter_signatures(raw, args.sig_len, args.stride))
        top = ctr.most_common(args.top_per_chunk)

        row = {
            "chunk_id": chunk_id,
            "raw_len": len(raw),
            "num_unique_signatures": len(ctr),
            "top_signatures": [(sig.hex(), cnt) for sig, cnt in top],
        }

        filter_obj["chunks"].append(row)
        total_unique += len(ctr)
        total_kept += len(top)

    with open(args.out, "wb") as f:
        pickle.dump(filter_obj, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"  total_unique_signatures={total_unique}")
    print(f"  total_kept_signatures={total_kept}")
    print("  done")


if __name__ == "__main__":
    main()