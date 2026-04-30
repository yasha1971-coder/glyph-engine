import argparse
import json
import os
import pickle


def iter_sigs(data: bytes, sig_len: int, stride: int):
    if len(data) < sig_len:
        return
    for i in range(0, len(data) - sig_len + 1, stride):
        yield data[i:i + sig_len].hex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--sig-len", type=int, default=8)
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.manifest, "r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    out = {
        "version": "chunk_coarse_gate_v1",
        "sig_len": args.sig_len,
        "stride": args.stride,
        "chunks": []
    }

    for row in rows:
        cid = row["chunk_id"]
        raw_path = os.path.join(row["chunk_dir"], "chunk.raw.bin")
        with open(raw_path, "rb") as f:
            data = f.read()

        sigs = sorted(set(iter_sigs(data, args.sig_len, args.stride)))
        out["chunks"].append({
            "chunk_id": cid,
            "raw_len": len(data),
            "sigs": sigs
        })

    with open(args.out, "wb") as f:
        pickle.dump(out, f)

    print("=" * 60)
    print(" COARSE GATE BUILD V1")
    print("=" * 60)
    print(f" manifest={args.manifest}")
    print(f" chunks={len(out['chunks'])}")
    print(f" sig_len={args.sig_len}")
    print(f" stride={args.stride}")
    print(f" out={args.out}")


if __name__ == "__main__":
    main()