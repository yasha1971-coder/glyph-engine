import argparse
import pickle
import random
import struct
import zlib
from collections import defaultdict


# ------------------------------------------------------------
# LOAD
# ------------------------------------------------------------
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


def load_filter(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# ------------------------------------------------------------
# FRAGMENTS
# ------------------------------------------------------------
def choose_fragment_starts(L, frag_len, nfrag, min_gap, rng):
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    if L < need:
        return None
    base = rng.randint(0, L - need)
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def build_fragments(chunk, frag_len, nfrag, min_gap, rng):
    starts = choose_fragment_starts(len(chunk), frag_len, nfrag, min_gap, rng)
    if starts is None:
        return None, None
    return [chunk[s:s+frag_len] for s in starts], starts


# ------------------------------------------------------------
# MAPS
# ------------------------------------------------------------
def build_dense_inv(filter_obj):
    return filter_obj["dense"]["inv"]


def build_rare_map(filter_obj):
    rare_map = {}
    for row in filter_obj["rare"]["chunks"]:
        cmap = {}
        anchors = row.get("rare_anchors", row.get("anchors", []))
        for sig_hex, cnt, df in anchors:
            cmap[sig_hex] = (cnt, df)
        rare_map[row["chunk_id"]] = cmap
    return rare_map


# ------------------------------------------------------------
# BRANCH SCORING
# ------------------------------------------------------------
def score_dense_only(frags, dense_inv, max_cid):
    scores = defaultdict(lambda: {
        "dense_present": 0,
        "dense_total": 0,
    })

    for frag in frags:
        seen_dense = set()

        for i in range(len(frag) - 7):
            sig = frag[i:i+8].hex()
            inv = dense_inv.get(sig)
            if not inv:
                continue
            for cid_key, cnt in inv.items():
                cid = int(cid_key)
                if cid >= max_cid:
                    continue
                scores[cid]["dense_total"] += cnt
                seen_dense.add(cid)

        for cid in seen_dense:
            scores[cid]["dense_present"] += 1

    ranked = []
    for cid, s in scores.items():
        # keep simple and stable
        score = s["dense_present"] * 10.0 + s["dense_total"] * 0.01
        ranked.append((cid, score, s["dense_present"], s["dense_total"]))

    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked


def score_rare_only(frags, rare_map, max_cid):
    scores = defaultdict(lambda: {
        "rare_present": 0,
        "rare_score": 0.0,
    })

    for frag in frags:
        seen_rare = set()

        for i in range(len(frag) - 7):
            sig = frag[i:i+8].hex()
            for cid, cmap in rare_map.items():
                if cid >= max_cid:
                    continue
                hit = cmap.get(sig)
                if not hit:
                    continue
                cnt, df = hit
                scores[cid]["rare_score"] += cnt / (df + 1.0)
                seen_rare.add(cid)

        for cid in seen_rare:
            scores[cid]["rare_present"] += 1

    ranked = []
    for cid, s in scores.items():
        score = s["rare_present"] * 10.0 + s["rare_score"]
        ranked.append((cid, score, s["rare_present"], s["rare_score"]))

    ranked.sort(key=lambda x: (-x[1], x[0]))
    return ranked


# ------------------------------------------------------------
# UNION SHORTLIST
# ------------------------------------------------------------
def make_union_shortlist(dense_ranked, rare_ranked, kd, kr):
    dense_top = [cid for cid, *_ in dense_ranked[:kd]]
    rare_top = [cid for cid, *_ in rare_ranked[:kr]]

    dense_set = set(dense_top)
    rare_set = set(rare_top)

    inter = dense_set & rare_set
    dense_only = [cid for cid in dense_top if cid not in inter]
    rare_only = [cid for cid in rare_top if cid not in inter]

    # order:
    # 1) intersection first (consensus)
    # 2) rare-only
    # 3) dense-only
    final = list(sorted(inter)) + rare_only + dense_only

    # dedup preserve order
    seen = set()
    out = []
    for cid in final:
        if cid in seen:
            continue
        seen.add(cid)
        out.append(cid)

    return out, {
        "dense_top": dense_top,
        "rare_top": rare_top,
        "intersection": list(sorted(inter)),
        "dense_only": dense_only,
        "rare_only": rare_only,
    }


# ------------------------------------------------------------
# EXACT RERANK
# ------------------------------------------------------------
def exact_score(frags, chunk):
    present = 0
    total = 0
    for frag in frags:
        c = chunk.count(frag)
        if c > 0:
            present += 1
            total += c
    return present, total


def rerank_exact(chunks, frags, shortlist):
    out = []
    for cid in shortlist:
        if 0 <= cid < len(chunks):
            p, t = exact_score(frags, chunks[cid])
            out.append((cid, p, t))
    out.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return out


# ------------------------------------------------------------
# MAIN
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--dense-k", type=int, default=16)
    ap.add_argument("--rare-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    filt = load_filter(args.filter)
    chunks = load_chunks(args.glyph)[:len(filt["rare"]["chunks"])]

    dense_inv = build_dense_inv(filt)
    rare_map = build_rare_map(filt)

    dense_hit = 0
    rare_hit = 0
    union_hit = 0
    hit1 = 0
    total = 0

    truth_in_intersection = 0
    truth_in_dense_only = 0
    truth_in_rare_only = 0
    truth_in_neither = 0

    examples = []

    print("=" * 60)
    print(" BRANCH UNION SHORTLIST V1")
    print("=" * 60)
    print(f" indexed_chunks={len(chunks)}")
    print(f" dense_k={args.dense_k}")
    print(f" rare_k={args.rare_k}")

    ids = list(range(len(chunks)))
    if args.trials < len(ids):
        ids = rng.sample(ids, args.trials)

    for qid in ids:
        frags, starts = build_fragments(
            chunks[qid],
            args.frag_len,
            args.nfrag,
            args.min_gap,
            random.Random(args.seed + qid),
        )
        if frags is None:
            continue

        dense_ranked = score_dense_only(frags, dense_inv, len(chunks))
        rare_ranked = score_rare_only(frags, rare_map, len(chunks))

        dense_top = [cid for cid, *_ in dense_ranked[:args.dense_k]]
        rare_top = [cid for cid, *_ in rare_ranked[:args.rare_k]]

        if qid in dense_top:
            dense_hit += 1
        if qid in rare_top:
            rare_hit += 1

        union_shortlist, parts = make_union_shortlist(
            dense_ranked, rare_ranked, args.dense_k, args.rare_k
        )

        if qid in union_shortlist:
            union_hit += 1

        in_dense = qid in parts["dense_top"]
        in_rare = qid in parts["rare_top"]

        if in_dense and in_rare:
            truth_in_intersection += 1
        elif in_dense:
            truth_in_dense_only += 1
        elif in_rare:
            truth_in_rare_only += 1
        else:
            truth_in_neither += 1

        rr = rerank_exact(chunks, frags, union_shortlist)
        if rr and rr[0][0] == qid:
            hit1 += 1

        total += 1

        if len(examples) < 5:
            examples.append({
                "qid": qid,
                "starts": starts,
                "dense_top5": dense_top[:5],
                "rare_top5": rare_top[:5],
                "intersection": parts["intersection"][:5],
                "dense_only": parts["dense_only"][:5],
                "rare_only": parts["rare_only"][:5],
                "rerank_top5": rr[:5],
            })

    print()
    print("RESULT:")
    print(f"  trials                 = {total}")
    print(f"  dense_hit@{args.dense_k:<3}         = {dense_hit / total:.4f}")
    print(f"  rare_hit@{args.rare_k:<3}          = {rare_hit / total:.4f}")
    print(f"  union_hit@{args.dense_k + args.rare_k:<3}        = {union_hit / total:.4f}")
    print(f"  hit@1                  = {hit1 / total:.4f}")

    print()
    print("TRUTH LOCATION:")
    print(f"  truth_in_intersection  = {truth_in_intersection / total:.4f}")
    print(f"  truth_in_dense_only    = {truth_in_dense_only / total:.4f}")
    print(f"  truth_in_rare_only     = {truth_in_rare_only / total:.4f}")
    print(f"  truth_in_neither       = {truth_in_neither / total:.4f}")

    print()
    print("EXAMPLES:")
    for ex in examples:
        print(f"  query_chunk={ex['qid']} starts={ex['starts']}")
        print(f"    dense_top5={ex['dense_top5']}")
        print(f"    rare_top5={ex['rare_top5']}")
        print(f"    intersection={ex['intersection']}")
        print(f"    dense_only={ex['dense_only']}")
        print(f"    rare_only={ex['rare_only']}")
        print(f"    rerank_top5={ex['rerank_top5']}")


if __name__ == "__main__":
    main()