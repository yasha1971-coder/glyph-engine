import argparse
import pickle
import random
import struct
import zlib
from collections import defaultdict


# -----------------------------
# LOAD GLYPH
# -----------------------------
def load_chunks(glyph_path):
    chunks = []
    with open(glyph_path, "rb") as f:
        if f.read(6) != b"GLYPH1":
            raise ValueError("bad glyph magic")
        _version, ng = struct.unpack("<BH", f.read(3))
        for _ in range(ng):
            sz = struct.unpack("<I", f.read(4))[0]
            chunks.append(zlib.decompress(f.read(sz)))
    return chunks


# -----------------------------
# FRAGMENTS
# -----------------------------
def choose_fragment_starts(chunk_len, frag_len, nfrag, min_gap, rng):
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    if chunk_len < need:
        return None
    base = rng.randint(0, chunk_len - need)
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def build_fragment_set(chunk, frag_len, nfrag, min_gap, rng):
    starts = choose_fragment_starts(len(chunk), frag_len, nfrag, min_gap, rng)
    if starts is None:
        return None, None
    frags = [chunk[s:s + frag_len] for s in starts]
    return starts, frags


# -----------------------------
# EXACT RERANK
# -----------------------------
def exact_score(query_frags, candidate_chunk):
    present = 0
    total = 0

    for frag in query_frags:
        cnt = candidate_chunk.count(frag)
        if cnt > 0:
            present += 1
            total += cnt

    return present, total


def rerank_exact(chunks, query_frags, candidates):
    scored = []

    for cid in candidates:
        present, total = exact_score(query_frags, chunks[cid])
        scored.append((cid, present, total))

    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return scored


# -----------------------------
# GAP SHORTLIST LOADER (reuse)
# -----------------------------
def load_gap_results(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--glyph", required=True)
    ap.add_argument("--gap-results", required=True)  # <- from multi_gap_diagnostic
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    chunks = load_chunks(args.glyph)
    gap_data = load_gap_results(args.gap_results)

    rng = random.Random(args.seed)

    hit1 = 0
    hit4 = 0
    hit8 = 0
    hit16 = 0
    mrr = 0.0

    executed = 0

    print("=" * 60)
    print(" GAP → EXACT RERANK DIAGNOSTIC")
    print("=" * 60)

    for item in gap_data:
        qid = item["qid"]
        starts = item["starts"]
        gap_top = item["top"][:args.top_k]

        # rebuild frags deterministically
        local_rng = random.Random(args.seed + qid)
        _, frags = build_fragment_set(
            chunks[qid],
            args.frag_len,
            args.nfrag,
            args.min_gap,
            local_rng
        )

        if frags is None:
            continue

        candidates = [cid for cid, *_ in gap_top]

        if not candidates:
            executed += 1
            continue

        reranked = rerank_exact(chunks, frags, candidates)

        rank = None
        for i, (cid, _p, _t) in enumerate(reranked):
            if cid == qid:
                rank = i + 1
                break

        if rank is not None:
            if rank == 1:
                hit1 += 1
            if rank <= 4:
                hit4 += 1
            if rank <= 8:
                hit8 += 1
            if rank <= 16:
                hit16 += 1
            mrr += 1.0 / rank

        executed += 1

    print("\nRESULTS:")
    print(f"  gap_rerank_hit@1  = {hit1 / executed:.2%}")
    print(f"  gap_rerank_hit@4  = {hit4 / executed:.2%}")
    print(f"  gap_rerank_hit@8  = {hit8 / executed:.2%}")
    print(f"  gap_rerank_hit@16 = {hit16 / executed:.2%}")
    print(f"  gap_rerank_MRR    = {mrr / executed:.4f}")


if __name__ == "__main__":
    main()