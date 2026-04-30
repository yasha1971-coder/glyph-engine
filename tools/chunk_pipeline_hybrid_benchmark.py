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
            raise ValueError("bad glyph")
        _v, ng = struct.unpack("<BH", f.read(3))
        for _ in range(ng):
            sz = struct.unpack("<I", f.read(4))[0]
            chunks.append(zlib.decompress(f.read(sz)))
    return chunks


# -----------------------------
# FRAGMENTS
# -----------------------------
def choose_fragment_starts(L, frag_len, nfrag, min_gap, rng):
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    if L < need:
        return None
    base = rng.randint(0, L - need)
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def build_fragments(chunk, frag_len, nfrag, min_gap, rng):
    starts = choose_fragment_starts(len(chunk), frag_len, nfrag, min_gap, rng)
    if starts is None:
        return None
    return [chunk[s:s+frag_len] for s in starts]


# -----------------------------
# MAPS
# -----------------------------
def build_dense_map(filter_obj):
    return filter_obj["dense"]["inv"]


def build_rare_map(filter_obj):
    rmap = {}
    for row in filter_obj["rare"]["chunks"]:
        cmap = {}
        for sig_hex, cnt, df in row["rare_anchors"]:
            cmap[sig_hex] = (cnt, df)
        rmap[row["chunk_id"]] = cmap
    return rmap


# -----------------------------
# SHORTLIST (BASELINE)
# -----------------------------
def shortlist(hybrid, frags, top_k, max_cid):
    dense_inv = build_dense_map(hybrid)
    rare_map = build_rare_map(hybrid)

    scores = defaultdict(lambda: {
        "dense_present": 0,
        "rare_present": 0,
        "rare_score": 0.0
    })

    for frag in frags:
        seen_dense = set()
        seen_rare = set()

        for i in range(len(frag) - 7):
            sig = frag[i:i+8].hex()

            # dense side
            inv = dense_inv.get(sig)
            if inv:
                for cid_key in inv.keys():
                    cid = int(cid_key)
                    if cid < max_cid:
                        seen_dense.add(cid)

            # rare side
            for cid, cmap in rare_map.items():
                if cid >= max_cid:
                    continue
                hit = cmap.get(sig)
                if hit:
                    cnt, df = hit
                    scores[cid]["rare_score"] += cnt / (df + 1.0)
                    seen_rare.add(cid)

        for cid in seen_dense:
            scores[cid]["dense_present"] += 1
        for cid in seen_rare:
            scores[cid]["rare_present"] += 1

    ranked = []
    for cid, s in scores.items():
        score = (
            s["rare_present"] * 5.0 +
            s["dense_present"] * 2.0 +
            s["rare_score"]
        )
        ranked.append((cid, score))

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


def rerank(chunks, frags, candidates):
    # extra safety
    candidates = [cid for cid in candidates if 0 <= cid < len(chunks)]

    out = []
    for cid in candidates:
        p, t = exact_score(frags, chunks[cid])
        out.append((cid, p, t))
    out.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return out


# -----------------------------
# EVAL
# -----------------------------
def run_eval(chunks, hybrid, trials, top_k, seed):
    rng = random.Random(seed)
    hit1 = 0
    shortlist_hit = 0
    total = 0

    n = len(chunks)
    ids = list(range(n))
    if trials < n:
        ids = rng.sample(ids, trials)

    for qid in ids:
        frags = build_fragments(chunks[qid], 48, 5, 128, rng)
        if frags is None:
            continue

        sl = shortlist(hybrid, frags, top_k, max_cid=n)

        if qid in sl:
            shortlist_hit += 1

        rr = rerank(chunks, frags, sl)
        found = False
        for i, (cid, _, _) in enumerate(rr):
            if cid == qid:
                found = True
                if i == 0:
                    hit1 += 1
                break

        total += 1

    shortlist_recall = (shortlist_hit / total) if total else 0.0
    hit1_rate = (hit1 / total) if total else 0.0
    return shortlist_recall, hit1_rate


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.filter, "rb") as f:
        hybrid = pickle.load(f)

    chunks_all = load_chunks(args.glyph)

    print("=" * 60)
    print(" HYBRID BASELINE SCALING TEST")
    print("=" * 60)

    for size in [64, 128, 256, 512]:
        if size > len(chunks_all):
            continue

        chunks = chunks_all[:size]
        shortlist_recall, hit1 = run_eval(
            chunks=chunks,
            hybrid=hybrid,
            trials=args.trials,
            top_k=args.top_k,
            seed=args.seed
        )

        print(
            f"  chunks={size:4d}  "
            f"shortlist_hit@{args.top_k}={shortlist_recall:.2%}  "
            f"hit@1={hit1:.2%}"
        )


if __name__ == "__main__":
    main()