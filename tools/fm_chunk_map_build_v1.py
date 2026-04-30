#!/usr/bin/env python3
import argparse
import pickle
import struct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sa-pkl", required=True)
    ap.add_argument("--chunk-size", type=int, default=16384)
    ap.add_argument("--num-chunks", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.sa_pkl, "rb") as f:
        sa = pickle.load(f)

    # sa is array/list of suffix starting positions
    # chunk_id = pos // chunk_size
    # cap to [0, num_chunks-1]
    chunk_map = []
    for pos in sa:
        cid = int(pos) // args.chunk_size
        if cid < 0:
            cid = 0
        if cid >= args.num_chunks:
            cid = args.num_chunks - 1
        chunk_map.append(cid)

    with open(args.out, "wb") as f:
        f.write(b"CHMAPV1\0")
        f.write(struct.pack("<Q", len(chunk_map)))
        f.write(struct.pack("<I", args.chunk_size))
        f.write(struct.pack("<I", args.num_chunks))
        for cid in chunk_map:
            f.write(struct.pack("<I", cid))

    print("=" * 60)
    print(" FM CHUNK MAP BUILD V1")
    print("=" * 60)
    print(f" sa_size={len(chunk_map):,}")
    print(f" chunk_size={args.chunk_size}")
    print(f" num_chunks={args.num_chunks}")
    print(f" out={args.out}")


if __name__ == "__main__":
    main()