import argparse
import pickle
import random
import struct
import zlib
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
# CURRENT BASELINE SHORTLIST
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
# DIAGNOSTIC HELPERS
# -----------------------------
def collect_all_offsets(frags, sig_len):
    sigs = []
    for frag in frags:
        for i in range(len(frag) - sig_len + 1):
            sigs.append(frag[i:i + sig_len].hex())
    return sigs


def collect_stride_aligned(frags, starts, sig_len, stride):
    """
    Only keep query signatures whose global offset would align to the index stride.
    Here starts[] are local starts inside the chunk.
    For each fragment-local offset i, global alignment condition is:
        (fragment_start + i) % stride == 0
    """
    sigs = []
    for frag, frag_start in zip(frags, starts):
        for i in range(len(frag) - sig_len + 1):
            if ((frag_start + i) % stride) == 0:
                sigs.append(frag[i:i + sig_len].hex())
    return sigs


def known_and_df(sig_hex_list, dense_inv):
    known = 0
    unknown = 0
    dfs = []
    for sig in sig_hex_list:
        chunk_map = dense_inv.get(sig)
        if chunk_map:
            known += 1
            dfs.append(len(chunk_map))
        else:
            unknown += 1
    return known, unknown, dfs


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
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=10)
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

    agg_all_known = []
    agg_all_unknown = []
    agg_all_df = []

    agg_stride_known = []
    agg_stride_unknown = []
    agg_stride_df = []

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
        shortlist = shortlist_hybrid(baseline, frags, args.top_k)
        if qid in shortlist:
            continue

        missed += 1

        all_sigs = collect_all_offsets(frags, sig_len)
        all_known, all_unknown, all_df = known_and_df(all_sigs, dense_inv_nodf)

        stride_sigs = collect_stride_aligned(frags, starts, sig_len, stride)
        stride_known, stride_unknown, stride_df = known_and_df(stride_sigs, dense_inv_nodf)

        agg_all_known.append(all_known)
        agg_all_unknown.append(all_unknown)
        agg_all_df.extend(all_df)

        agg_stride_known.append(stride_known)
        agg_stride_unknown.append(stride_unknown)
        agg_stride_df.extend(stride_df)

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "starts": starts,
                "all_total": len(all_sigs),
                "all_known": all_known,
                "all_unknown": all_unknown,
                "all_df": summarize(all_df),
                "stride_total": len(stride_sigs),
                "stride_known": stride_known,
                "stride_unknown": stride_unknown,
                "stride_df": summarize(stride_df),
            })

    print("=" * 60)
    print(" MISSED STRIDE/DF DIAGNOSTIC V2")
    print("=" * 60)
    print(f" trials={executed}")
    print(f" top_k={args.top_k}")
    print(f" sig_len={sig_len}")
    print(f" stride={stride}")
    print(f" missed={missed}")
    if executed > 0:
        print(f" missed_rate={missed / executed:.2%}")

    all_known_s = summarize(agg_all_known)
    all_unknown_s = summarize(agg_all_unknown)
    all_df_s = summarize(agg_all_df)

    stride_known_s = summarize(agg_stride_known)
    stride_unknown_s = summarize(agg_stride_unknown)
    stride_df_s = summarize(agg_stride_df)

    print("\nALL-OFFSETS AGAINST NODF DENSE:")
    print(f"  known_per_query: "
          f"median={fmt_stat(all_known_s['median'])} "
          f"mean={fmt_stat(all_known_s['mean'])} "
          f"min={fmt_stat(all_known_s['min'])} "
          f"p90={fmt_stat(all_known_s['p90'])} "
          f"max={fmt_stat(all_known_s['max'])}")
    print(f"  unknown_per_query: "
          f"median={fmt_stat(all_unknown_s['median'])} "
          f"mean={fmt_stat(all_unknown_s['mean'])} "
          f"min={fmt_stat(all_unknown_s['min'])} "
          f"p90={fmt_stat(all_unknown_s['p90'])} "
          f"max={fmt_stat(all_unknown_s['max'])}")
    print(f"  df_over_known: "
          f"count={all_df_s['count']} "
          f"min={fmt_stat(all_df_s['min'])} "
          f"median={fmt_stat(all_df_s['median'])} "
          f"p90={fmt_stat(all_df_s['p90'])} "
          f"max={fmt_stat(all_df_s['max'])} "
          f"mean={fmt_stat(all_df_s['mean'])}")

    print("\nSTRIDE-ALIGNED AGAINST NODF DENSE:")
    print(f"  known_per_query: "
          f"median={fmt_stat(stride_known_s['median'])} "
          f"mean={fmt_stat(stride_known_s['mean'])} "
          f"min={fmt_stat(stride_known_s['min'])} "
          f"p90={fmt_stat(stride_known_s['p90'])} "
          f"max={fmt_stat(stride_known_s['max'])}")
    print(f"  unknown_per_query: "
          f"median={fmt_stat(stride_unknown_s['median'])} "
          f"mean={fmt_stat(stride_unknown_s['mean'])} "
          f"min={fmt_stat(stride_unknown_s['min'])} "
          f"p90={fmt_stat(stride_unknown_s['p90'])} "
          f"max={fmt_stat(stride_unknown_s['max'])}")
    print(f"  df_over_known: "
          f"count={stride_df_s['count']} "
          f"min={fmt_stat(stride_df_s['min'])} "
          f"median={fmt_stat(stride_df_s['median'])} "
          f"p90={fmt_stat(stride_df_s['p90'])} "
          f"max={fmt_stat(stride_df_s['max'])} "
          f"mean={fmt_stat(stride_df_s['mean'])}")

    print("\nMISSED EXAMPLES:")
    for ex in examples:
        print(f"  query_chunk={ex['qid']} starts={ex['starts']}")
        print(f"    all_offsets: total={ex['all_total']} known={ex['all_known']} unknown={ex['all_unknown']}")
        a = ex["all_df"]
        print(f"      df: count={a['count']} min={fmt_stat(a['min'])} median={fmt_stat(a['median'])} "
              f"p90={fmt_stat(a['p90'])} max={fmt_stat(a['max'])} mean={fmt_stat(a['mean'])}")
        print(f"    stride_aligned: total={ex['stride_total']} known={ex['stride_known']} unknown={ex['stride_unknown']}")
        s = ex["stride_df"]
        print(f"      df: count={s['count']} min={fmt_stat(s['min'])} median={fmt_stat(s['median'])} "
              f"p90={fmt_stat(s['p90'])} max={fmt_stat(s['max'])} mean={fmt_stat(s['mean'])}")


if __name__ == "__main__":
    main()