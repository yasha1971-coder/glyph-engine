import argparse
import os
import struct
import subprocess
import tempfile
import zlib


def load_chunks(glyph_path):
    chunks = []
    with open(glyph_path, "rb") as f:
        if f.read(6) != b"GLYPH1":
            raise ValueError("bad glyph")
        _v, ng = struct.unpack("<BH", f.read(3))
        for _ in range(ng):
            sz = struct.unpack("<I", f.read(4))[0]
            chunks.append(zlib.decompress(f.read(sz)))
    return chunks


def choose_fragment_starts(L, frag_len, nfrag, min_gap, seed):
    import random
    rng = random.Random(seed)
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    base = rng.randint(0, L - need)
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def write_request(path, frags, topk):
    with open(path, "wb") as f:
        f.write(b"SLREQV1\0")
        f.write(struct.pack("<I", len(frags)))
        f.write(struct.pack("<I", topk))
        for frag in frags:
            f.write(struct.pack("<I", len(frag)))
            f.write(frag)


def read_response(path):
    with open(path, "rb") as f:
        magic = f.read(8)
        n = struct.unpack("<I", f.read(4))[0]
        rows = []
        for _ in range(n):
            cid = struct.unpack("<I", f.read(4))[0]
            score = struct.unpack("<d", f.read(8))[0]
            rows.append((cid, score))
        return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True)
    ap.add_argument("--dense-bin", required=True)
    ap.add_argument("--rare-bin", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--query-chunk-id", type=int, required=True)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    chunks = load_chunks(args.glyph)
    q = chunks[args.query_chunk_id]
    starts = choose_fragment_starts(len(q), args.frag_len, args.nfrag, args.min_gap, args.seed + args.query_chunk_id)
    frags = [q[s:s+args.frag_len] for s in starts]

    with tempfile.TemporaryDirectory() as td:
        req = os.path.join(td, "req.bin")
        resp = os.path.join(td, "resp.bin")
        write_request(req, frags, args.top_k)

        subprocess.run(
            [args.backend, args.dense_bin, args.rare_bin, req, resp],
            check=True
        )

        rows = read_response(resp)

    print("=" * 60)
    print(" SHORTLIST BACKEND SMOKE V1")
    print("=" * 60)
    print(f" query_chunk_id={args.query_chunk_id}")
    print(f" starts={starts}")
    print(" shortlist:")
    for i, (cid, score) in enumerate(rows, 1):
        marker = " <== TRUE" if cid == args.query_chunk_id else ""
        print(f"  #{i:>2}: chunk={cid} score={score:.6f}{marker}")


if __name__ == "__main__":
    main()