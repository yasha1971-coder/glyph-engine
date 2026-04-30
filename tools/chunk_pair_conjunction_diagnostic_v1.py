import argparse
import pickle
import random
import struct
import zlib
from collections import Counter
from itertools import combinations
from statistics import median


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
# BASELINE SHORTLIST
# -----------------------------
def build_rare_map(filter_obj):
    rare_map = {}
    for row in filter_obj["rare"]["chunks"]:
        cmap = {}
        for sig_hex, cnt, df in row["rare_anchors"]:
            cmap[sig_hex] = (cnt, df)
        rare_map[row["chunk_id"]] = cmap
    return rare_map


def shortlist_hybrid(filter_obj, frags, top_k):
    dense_inv = filter_obj["dense"]["inv"]
    rare_map = build_rare_map(filter_obj)

    dense_sig_len = filter_obj["dense"].get("sig_len", 8)
    rare_sig_len = filter_obj["rare"].get("sig_len", 8)
    if dense_sig_len != rare_sig_len:
        raise ValueError(f"sig_len mismatch: dense={dense_sig_len} rare={rare_sig_len}")
    sig_len = dense_sig_len

    from collections import defaultdict
    chunk_scores = defaultdict(lambda: {
        "dense_present": 0,
        "dense_total": 0,
        "rare_present": 0,
        "rare_matches": 0,
        "rare_score": 0.0
    })

    # dense
    for frag in frags:
        frag_seen_chunks = set()
        for i in range(len(frag) - sig_len + 1):
            sig = frag[i:i + sig_len].hex()
            chunk_map = dense_inv.get(sig)
            if not chunk_map:
                continue
            for chunk_id_key, cnt in chunk_map.items():
                cid = int(chunk_id_key)
                chunk_scores[cid]["dense_total"] += cnt
                frag_seen_chunks.add(cid)
        for cid in frag_seen_chunks:
            chunk_scores[cid]["dense_present"] += 1

    # rare
    for frag in frags:
        frag_seen_chunks = set()
        for i in range(len(frag) - sig_len + 1):
            sig = frag[i:i + sig_len].hex()
            for cid, cmap in rare_map.items():
                hit = cmap.get(sig)
                if hit is None:
                    continue
                cnt, df = hit
                score = 1.0 / (df + 1.0)
                chunk_scores[cid]["rare_score"] += score * cnt
                chunk_scores[cid]["rare_matches"] += 1
                frag_seen_chunks.add(cid)
        for cid in frag_seen_chunks:
            chunk_scores[cid]["rare_present"] += 1

    scored = []
    for cid, s in chunk_scores.items():
        scored.append((
            cid,
            s["rare_present"],
            s["dense_present"],
            s["rare_score"],
            s["dense_total"],
            s["rare_matches"]
        ))

    scored.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4], -x[5], x[0]))
    return [cid for cid, *_ in scored[:top_k]]


# -----------------------------
# STATS
# -----------------------------
def quantile_sorted(vals, q):
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    idx = int(round((len(vals) - 1) * q))
    return vals[idx]


def summarize(vals):
    if not vals:
        return {
            "count": 0,
            "min": None,
            "median": None,
            "p90": None,
            "max": None,
            "mean": None,
        }
    s = sorted(vals)
    return {
        "count": len(vals),
        "min": s[0],
        "median": median(s),
        "p90": quantile_sorted(s, 0.90),
        "max": s[-1],
        "mean": sum(s) / len(s),
    }


def fmt_stat(x):
    if x is None:
        return "NA"
    if isinstance(x, float):
        return f"{x:.2f}"
    return str(x)


# -----------------------------
# QUERY SIGNATURES
# -----------------------------
def collect_stride_aligned_unique(frags, starts, sig_len, stride):
    sigs = []
    seen = set()
    for frag, frag_start in zip(frags, starts):
        for i in range(len(frag) - sig_len + 1):
            if ((frag_start + i) % stride) == 0:
                sig = frag[i:i + sig_len].hex()
                if sig not in seen:
                    seen.add(sig)
                    sigs.append(sig)
    return sigs


def sig_to_chunkset(sig, dense_inv):
    chunk_map = dense_inv.get(sig)
    if not chunk_map:
        return set()
    return {int(cid) for cid in chunk_map.keys()}


# -----------------------------
# PAIR CONJUNCTION
# -----------------------------
def pair_conjunction_rank(sig_chunksets, pair_limit=128):
    """
    sig_chunksets: list[(sig_hex, set(chunk_ids))]
    Returns:
      ranked_chunks: list[(cid, pair_support, singleton_support)]
      pair_sizes: list[int]
    """
    # smaller DF first is more informative
    sig_chunksets = [(s, cs) for s, cs in sig_chunksets if cs]
    sig_chunksets.sort(key=lambda x: (len(x[1]), x[0]))

    if len(sig_chunksets) < 2:
        return [], []

    # use only most informative signatures to keep pair count bounded
    max_sigs = min(len(sig_chunksets), 16)
    sig_chunksets = sig_chunksets[:max_sigs]

    pair_counter = Counter()
    singleton_counter = Counter()
    pair_sizes = []

    for _sig, cset in sig_chunksets:
        for cid in cset:
            singleton_counter[cid] += 1

    all_pairs = list(combinations(range(len(sig_chunksets)), 2))
    if len(all_pairs) > pair_limit:
        all_pairs = all_pairs[:pair_limit]

    for i, j in all_pairs:
        cset = sig_chunksets[i][1] & sig_chunksets[j][1]
        pair_sizes.append(len(cset))
        for cid in cset:
            pair_counter[cid] += 1

    ranked = []
    candidate_ids = set(pair_counter.keys()) | set(singleton_counter.keys())
    for cid in candidate_ids:
        ranked.append((cid, pair_counter[cid], singleton_counter[cid]))

    ranked.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return ranked, pair_sizes


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-filter", required=True)
    ap.add_argument("--dense-nodf", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=16, help="baseline top-k for defining missed cases")
    ap.add_argument("--pair-top-k", type=int, default=16, help="top-k to evaluate pairwise conjunction shortlist")
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=10)
    ap.add_argument("--pair-limit", type=int, default=128)
    args = ap.parse_args()

    with open(args.baseline_filter, "rb") as f:
        baseline = pickle.load(f)

    with open(args.dense_nodf, "rb") as f:
        dense_nodf = pickle.load(f)

    dense_inv_nodf = dense_nodf["inv"]
    sig_len = dense_nodf.get("sig_len", 8)
    stride = dense_nodf.get("stride", 8)

    chunks = load_chunks(args.glyph)
    rng = random.Random(args.seed)

    indexed_ids = sorted(row["chunk_id"] for row in baseline["rare"]["chunks"])
    eligible_ids = [
        cid for cid in indexed_ids
        if cid < len(chunks)
        and len(chunks[cid]) >= args.nfrag * args.frag_len + (args.nfrag - 1) * args.min_gap
    ]
    if not eligible_ids:
        raise ValueError("no eligible indexed chunks")

    if args.trials >= len(eligible_ids):
        bench_ids = eligible_ids[:args.trials]
    else:
        bench_ids = rng.sample(eligible_ids, args.trials)

    executed = 0
    missed = 0
    pair_hit = 0

    agg_num_stride_sigs = []
    agg_pair_sizes = []
    agg_pairs_le1 = []
    agg_pairs_le4 = []
    agg_pairs_le16 = []

    examples = []

    for qid in bench_ids:
        local_rng = random.Random(args.seed + qid)

        starts, frags = build_fragment_set(
            chunks[qid],
            args.frag_len,
            args.nfrag,
            args.min_gap,
            local_rng
        )
        if frags is None:
            continue

        executed += 1
        baseline_shortlist = shortlist_hybrid(baseline, frags, args.top_k)
        if qid in baseline_shortlist:
            continue

        missed += 1

        stride_sigs = collect_stride_aligned_unique(frags, starts, sig_len, stride)
        sig_chunksets = [(sig, sig_to_chunkset(sig, dense_inv_nodf)) for sig in stride_sigs]

        ranked, pair_sizes = pair_conjunction_rank(sig_chunksets, pair_limit=args.pair_limit)
        pair_shortlist = [cid for cid, _ps, _ss in ranked[:args.pair_top_k]]

        if qid in pair_shortlist:
            pair_hit += 1

        agg_num_stride_sigs.append(len(stride_sigs))
        agg_pair_sizes.extend(pair_sizes)
        agg_pairs_le1.append(sum(1 for x in pair_sizes if x <= 1))
        agg_pairs_le4.append(sum(1 for x in pair_sizes if x <= 4))
        agg_pairs_le16.append(sum(1 for x in pair_sizes if x <= 16))

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "starts": starts,
                "num_stride_sigs": len(stride_sigs),
                "pair_sizes_stats": summarize(pair_sizes),
                "pairs_le1": sum(1 for x in pair_sizes if x <= 1),
                "pairs_le4": sum(1 for x in pair_sizes if x <= 4),
                "pairs_le16": sum(1 for x in pair_sizes if x <= 16),
                "pair_top5": ranked[:5],
                "truth_in_pair_topk": qid in pair_shortlist,
            })

    print("=" * 60)
    print(" PAIR CONJUNCTION DIAGNOSTIC V1")
    print("=" * 60)
    print(f" trials={executed}")
    print(f" baseline_top_k={args.top_k}")
    print(f" pair_top_k={args.pair_top_k}")
    print(f" sig_len={sig_len}")
    print(f" stride={stride}")
    print(f" missed={missed}")
    if executed > 0:
        print(f" missed_rate={missed / executed:.2%}")
    if missed > 0:
        print(f" pair_hit@{args.pair_top_k}_on_missed = {pair_hit / missed:.2%}")

    num_stride_s = summarize(agg_num_stride_sigs)
    pair_size_s = summarize(agg_pair_sizes)
    le1_s = summarize(agg_pairs_le1)
    le4_s = summarize(agg_pairs_le4)
    le16_s = summarize(agg_pairs_le16)

    print("\nAGGREGATES OVER MISSED CASES:")
    print(f"  num_stride_sigs_per_query: "
          f"median={fmt_stat(num_stride_s['median'])} "
          f"mean={fmt_stat(num_stride_s['mean'])} "
          f"min={fmt_stat(num_stride_s['min'])} "
          f"p90={fmt_stat(num_stride_s['p90'])} "
          f"max={fmt_stat(num_stride_s['max'])}")

    print(f"  pair_intersection_size: "
          f"count={pair_size_s['count']} "
          f"min={fmt_stat(pair_size_s['min'])} "
          f"median={fmt_stat(pair_size_s['median'])} "
          f"p90={fmt_stat(pair_size_s['p90'])} "
          f"max={fmt_stat(pair_size_s['max'])} "
          f"mean={fmt_stat(pair_size_s['mean'])}")

    print(f"  num_pairs_with_intersection<=1: "
          f"median={fmt_stat(le1_s['median'])} "
          f"mean={fmt_stat(le1_s['mean'])} "
          f"min={fmt_stat(le1_s['min'])} "
          f"p90={fmt_stat(le1_s['p90'])} "
          f"max={fmt_stat(le1_s['max'])}")

    print(f"  num_pairs_with_intersection<=4: "
          f"median={fmt_stat(le4_s['median'])} "
          f"mean={fmt_stat(le4_s['mean'])} "
          f"min={fmt_stat(le4_s['min'])} "
          f"p90={fmt_stat(le4_s['p90'])} "
          f"max={fmt_stat(le4_s['max'])}")

    print(f"  num_pairs_with_intersection<=16: "
          f"median={fmt_stat(le16_s['median'])} "
          f"mean={fmt_stat(le16_s['mean'])} "
          f"min={fmt_stat(le16_s['min'])} "
          f"p90={fmt_stat(le16_s['p90'])} "
          f"max={fmt_stat(le16_s['max'])}")

    print("\nMISSED EXAMPLES:")
    for ex in examples:
        print(f"  query_chunk={ex['qid']} starts={ex['starts']}")
        print(f"    num_stride_sigs={ex['num_stride_sigs']}")
        s = ex["pair_sizes_stats"]
        print(f"    pair_intersection_size: count={s['count']} min={fmt_stat(s['min'])} "
              f"median={fmt_stat(s['median'])} p90={fmt_stat(s['p90'])} "
              f"max={fmt_stat(s['max'])} mean={fmt_stat(s['mean'])}")
        print(f"    pairs_le1={ex['pairs_le1']} pairs_le4={ex['pairs_le4']} pairs_le16={ex['pairs_le16']}")
        print(f"    truth_in_pair_topk={ex['truth_in_pair_topk']}")
        print(f"    pair_top5={ex['pair_top5']}")


if __name__ == "__main__":
    main()