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
# BUILD MAPS
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
# HYBRID + COHERENCE BONUS
# ------------------------------------------------------------
def shortlist_with_proximity(frags, filter_obj, top_k):
    dense_inv = build_dense_inv(filter_obj)
    rare_map = build_rare_map(filter_obj)

    # per chunk state
    scores = defaultdict(lambda: {
        "dense_present": 0,
        "dense_total": 0,
        "rare_present": 0,
        "rare_score": 0.0,
        "frag_pattern": [0, 0, 0, 0, 0],   # hit per fragment
        "frag_strength": [0, 0, 0, 0, 0],  # summed support per fragment
    })

    for fi, frag in enumerate(frags):
        seen_dense = set()
        seen_rare = set()

        for i in range(len(frag) - 7):
            sig = frag[i:i+8].hex()

            # dense side
            inv = dense_inv.get(sig)
            if inv:
                for cid_key, cnt in inv.items():
                    cid = int(cid_key)
                    scores[cid]["dense_total"] += cnt
                    scores[cid]["frag_strength"][fi] += cnt
                    seen_dense.add(cid)

            # rare side
            for cid, cmap in rare_map.items():
                hit = cmap.get(sig)
                if hit:
                    cnt, df = hit
                    scores[cid]["rare_score"] += cnt / (df + 1.0)
                    scores[cid]["frag_strength"][fi] += cnt / (df + 1.0)
                    seen_rare.add(cid)

        for cid in seen_dense:
            scores[cid]["dense_present"] += 1
            scores[cid]["frag_pattern"][fi] = 1

        for cid in seen_rare:
            scores[cid]["rare_present"] += 1
            scores[cid]["frag_pattern"][fi] = 1

    ranked = []

    for cid, s in scores.items():
        pattern = s["frag_pattern"]
        strength = s["frag_strength"]

        # surrogate "proximity/coherence":
        # reward chunks that hit multiple adjacent fragments
        adjacent_bonus = 0.0
        for i in range(len(pattern) - 1):
            if pattern[i] and pattern[i + 1]:
                adjacent_bonus += 1.0

        # reward chunks with balanced multi-fragment evidence
        nonzero_strengths = [x for x in strength if x > 0]
        balance_bonus = 0.0
        if len(nonzero_strengths) >= 2:
            mn = min(nonzero_strengths)
            mx = max(nonzero_strengths)
            if mx > 0:
                balance_bonus = mn / mx

        final_score = (
            s["rare_present"] * 5.0 +
            s["dense_present"] * 2.0 +
            s["rare_score"] * 1.0 +
            adjacent_bonus * 3.0 +
            balance_bonus * 2.0
        )

        ranked.append((
            cid,
            final_score,
            s["rare_present"],
            s["dense_present"],
            s["rare_score"],
            adjacent_bonus,
            balance_bonus,
            pattern,
        ))

    ranked.sort(key=lambda x: (-x[1], x[0]))
    return [cid for cid, *_ in ranked[:top_k]], ranked[:top_k]


# ------------------------------------------------------------
# EXACT
# ------------------------------------------------------------
def exact_score(frags, chunk):
    present = 0
    total = 0
    for f in frags:
        c = chunk.count(f)
        if c > 0:
            present += 1
            total += c
    return present, total


def rerank(chunks, frags, shortlist):
    out = []
    for cid in shortlist:
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
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print("=" * 60)
    print(" HYBRID V3 + COHERENCE BONUS + EXACT")
    print("=" * 60)

    filter_obj = load_filter(args.filter)
    chunks = load_chunks(args.glyph)

    # current indexed regime = first 128 chunks
    indexed_limit = 128
    chunks = chunks[:indexed_limit]

    hit1 = 0
    total = 0
    examples = []

    for qid in rng.sample(range(len(chunks)), min(args.trials, len(chunks))):
        frags, starts = build_fragments(chunks[qid], args.frag_len, args.nfrag, args.min_gap, random.Random(args.seed + qid))
        if frags is None:
            continue

        shortlist, ranked = shortlist_with_proximity(frags, filter_obj, args.top_k)
        rr = rerank(chunks, frags, shortlist)

        pred = rr[0][0] if rr else None
        if pred == qid:
            hit1 += 1
        total += 1

        if len(examples) < 5:
            examples.append((qid, starts, ranked[:5], rr[:5]))

    print()
    print("RESULT:")
    print(f"  hit@1 = {hit1 / total:.4f}")

    print()
    print("EXAMPLES:")
    for qid, starts, ranked, rr in examples:
        print(f"  query_chunk={qid} starts={starts}")
        print("    shortlist_top5:")
        for cid, final_score, rare_present, dense_present, rare_score, adjacent_bonus, balance_bonus, pattern in ranked:
            marker = " <== TRUE" if cid == qid else ""
            print(
                f"      chunk={cid} "
                f"final={final_score:.3f} "
                f"rare_present={rare_present} "
                f"dense_present={dense_present} "
                f"rare_score={rare_score:.3f} "
                f"adj={adjacent_bonus:.1f} "
                f"bal={balance_bonus:.3f} "
                f"pattern={pattern}{marker}"
            )
        print("    rerank_top5:")
        for cid, p, t in rr:
            marker = " <== TRUE" if cid == qid else ""
            print(f"      chunk={cid} present={p} total={t}{marker}")


if __name__ == "__main__":
    main()