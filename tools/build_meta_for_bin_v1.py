import argparse
import os
import pickle


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bin", required=True)
    ap.add_argument("--out-meta", required=True)
    ap.add_argument("--chunk-size", type=int, default=16384)
    ap.add_argument("--preview", type=int, default=96)
    args = ap.parse_args()

    corpus_bytes = os.path.getsize(args.bin)
    chunk_starts = list(range(0, corpus_bytes, args.chunk_size))
    num_chunks = len(chunk_starts)

    meta = {
        "glyph": args.bin,
        "num_chunks": num_chunks,
        "corpus_bytes": corpus_bytes,
        "preview": args.preview,
        "chunk_starts": chunk_starts,
        "clean_mode": True,
        "sentinel": 0,
        "byte_shift": 0,
    }

    with open(args.out_meta, "wb") as f:
        pickle.dump(meta, f)

    print("saved:", args.out_meta)
    print("corpus_bytes:", corpus_bytes)
    print("num_chunks:", num_chunks)
    print("first3:", chunk_starts[:3])
    print("last3:", chunk_starts[-3:])


if __name__ == "__main__":
    main()