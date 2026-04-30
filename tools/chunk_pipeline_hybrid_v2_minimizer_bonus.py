import argparse
import math
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
# MINIMIZERS
# -----------------------------
def iter_kmers(data: bytes, k: int):
    for i in range(0, len(data) - k + 1):
        yield data[i:i+k]


def compute_minimizers(data: bytes, k: int, w: int):
    kmers = list(iter_kmers(data, k))
    if not kmers:
        return []
    if len(kmers) <= w:
        return [min(kmers)]
    out = []
    for i in range(0, len(kmers) - w + 1):
        out.append(min(kmers[i:i+w]))
    return out


# -----------------------------
# BUILD MAPS
# -----------------------------
def build_dense_map(filter_obj):
    return filter_obj["dense"]["inv"]


def build_rare_map(filter_obj):
    rare_map = {}
    for row in filter_obj["rare"]["chunks"]:
        cmap = {}
        for sig_hex, cnt, df in row["rare_anchors"]:
            cmap[sig_hex] = (cnt, df)
        rare_map[row["chunk_id"]] = cmap
    return rare_map


def build_minimizer_map(filter_obj):
    mmap = {}
    for row in filter_obj["chunks"]:
        cmap = {}
        for tok_hex, cnt, df in row["minimizers"]:
            cmap[tok_hex] = (cnt, df)
        mmap[row["chunk_id"]] = cmap
    return mmap


# -----------------------------
# HYBRID V2 SHORTLIST
# -----------------------------
def shortlist_hybrid_v2(hybrid_filter, minimizer_filter, frags, top_k, query_max_df):
    dense_inv = build_dense_map(hybrid_filter)
    rare_map = build_rare_map(hybrid_filter)
    minimizer_map = build_minimizer_map(minimizer_filter)

    k = minimizer_filter["k"]
    w = minimizer_filter["w"]
    global_df = minimizer_filter["global_df"]
    N = len(minimizer_filter["chunks"])

    scores = defaultdict(lambda: {
        "dense_present": 0,
        "rare_present": 0,
        "rare_score": 0.0,
        "min_score": 0.0
    })

    # ---- dense + rare ----
    for frag in frags:
        seen_dense = set()
        seen_rare = set()

        for i in range(len(frag) - 7):
            sig = frag[i:i+8].hex()

            # dense
            inv = dense_inv.get(sig)
            if inv:
                for cid in inv:
                    seen_dense.add(int(cid))

            # rare
            for cid, cmap in rare_map.items():
                hit = cmap.get(sig)
                if hit:
                    cnt, df = hit
                    scores[cid]["rare_score"] += cnt / (df + 1.0)
                    seen_rare.add(cid)

        for cid in seen_dense:
            scores[cid]["dense_present"] += 1
        for cid in seen_rare:
            scores[cid]["rare_present"] += 1

    # ---- minimizer bonus ----
    for frag in frags:
        mins = compute_minimizers(frag, k, w)
        seen = set()

        for tok in mins:
            tok_hex = tok.hex()
            if tok_hex in seen:
                continue
            seen.add(tok_hex)

            df = global_df.get(tok_hex)
            if df is None or df > query_max_df:
                continue

            idf = math.log(1.0 + N / df)

            for cid, cmap in minimizer_map.items():
                hit = cmap.get(tok_hex)
                if hit:
                    cnt, _ = hit
                    scores[cid]["min_score"] += cnt * idf

    # ---- final scoring ----
    ranked = []
    for cid, s in scores.items():
        final = (
            s["rare_present"] * 5.0 +
            s["dense_present"] * 2.0 +
            s["rare_score"] * 1.0 +
            s["min_score"] * 0.3
        )
        ranked.append((cid, final))

    ranked.sort(key=lambda x: (-x[1], x[0]))
    return [cid for cid, _ in ranked[:top_k]]


# -----------------------------
# EXACT RERANK
# -----------------------------
def exact_score(frags, chunk):
    present = 0
    total = 0
    for f in frags:
        c = chunk.count(f)
        if c > 0:
            present += 1
            total += c
    return present, total


def rerank_exact(chunks, frags, candidates):
    out = []
    for cid in candidates:
        p, t = exact_score(frags, chunks[cid])
        out.append((cid, p, t))
    out.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return out


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hybrid-filter", required=True)
    ap.add_argument("--minimizer-filter", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-max-df", type=int, default=128)
    args = ap.parse_args()

    with open(args.hybrid_filter, "rb") as f:
        hybrid = pickle.load(f)
    with open(args.minimizer_filter, "rb") as f:
        minim = pickle.load(f)

    chunks = load_chunks(args.glyph)
    rng = random.Random(args.seed)

    hit1 = 0
    total = 0

    print("="*60)
    print(" HYBRID V2 (MINIMIZER BONUS) + EXACT")
    print("="*60)

    for qid in rng.sample(range(128), args.trials):
        starts, frags = build_fragment_set(chunks[qid], 48, 5, 128, rng)
        if frags is None:
            continue

        shortlist = shortlist_hybrid_v2(
            hybrid, minim, frags,
            args.top_k,
            args.query_max_df
        )

        if qid not in shortlist:
            total += 1
            continue

        reranked = rerank_exact(chunks, frags, shortlist)

        for i, (cid, _, _) in enumerate(reranked):
            if cid == qid:
                if i == 0:
                    hit1 += 1
                break

        total += 1

    print(f"\n  hit@1 = {hit1 / total:.2%}")


if __name__ == "__main__":
    main()