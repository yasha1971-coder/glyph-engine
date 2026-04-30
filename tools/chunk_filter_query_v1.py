import argparse
import pickle
import struct
import zlib
from collections import Counter


def load_chunks(glyph_path):
    chunks = []
    with open(glyph_path, "rb") as f:
        magic = f.read(6)
        if magic != b"GLYPH1":
            raise ValueError(f"Bad glyph magic: {magic!r}")
        _version, ng = struct.unpack("<BH", f.read(3))
        for _ in range(ng):
            sz = struct.unpack("<I", f.read(4))[0]
            chunks.append(zlib.decompress(f.read(sz)))
    return chunks


def iter_signatures(data: bytes, sig_len: int, stride: int):
    if len(data) < sig_len:
        return
    for i in range(0, len(data) - sig_len + 1, stride):
        yield data[i:i + sig_len]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--query-chunk-id", type=int, required=True)
    ap.add_argument("--top-k", type=int, default=16)
    args = ap.parse_args()

    with open(args.filter, "rb") as f:
        filt = pickle.load(f)

    sig_len = filt["sig_len"]
    stride = filt["stride"]

    chunks = load_chunks(args.glyph)
    qid = args.query_chunk_id
    if qid < 0 or qid >= len(chunks):
        raise ValueError(f"query chunk id out of range: {qid}")

    query = chunks[qid]
    qsig = Counter(iter_signatures(query, sig_len, stride))

    print("=" * 60)
    print("  CHUNK FILTER QUERY V1")
    print("=" * 60)
    print(f"  filter={args.filter}")
    print(f"  glyph={args.glyph}")
    print(f"  query_chunk_id={qid}")
    print(f"  sig_len={sig_len}")
    print(f"  stride={stride}")
    print(f"  query_unique_signatures={len(qsig)}")
    print(f"  top_k={args.top_k}")

    scored = []

    for row in filt["chunks"]:
        chunk_id = row["chunk_id"]
        top_map = {bytes.fromhex(sig_hex): cnt for sig_hex, cnt in row["top_signatures"]}

        overlap = 0
        weighted = 0

        for sig, qcnt in qsig.items():
            if sig in top_map:
                overlap += 1
                weighted += qcnt * top_map[sig]

        scored.append((chunk_id, overlap, weighted))

    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))

    print("")
    print("  shortlist:")
    for rank, (chunk_id, overlap, weighted) in enumerate(scored[:args.top_k], 1):
        marker = " <== TRUE" if chunk_id == qid else ""
        print(f"    #{rank:>2}: chunk={chunk_id} overlap={overlap} weighted={weighted}{marker}")


if __name__ == "__main__":
    main()