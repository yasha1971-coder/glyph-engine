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
# FILTER HELPERS
# -----------------------------
def build_rare_map(filter_obj):
    rare_map = {}
    for row in filter_obj["rare"]["chunks"]:
        cmap = {}
        for sig_hex, cnt, df in row["rare_anchors"]:
            cmap[sig_hex] = (cnt, df)
        rare_map[row["chunk_id"]] = cmap
    return rare_map


def build_rare_df_map(filter_obj):
    rare_df = {}
    for row in filter_obj["rare"]["chunks"]:
        for sig_hex, _cnt, df in row["rare_anchors"]:
            old = rare_df.get(sig_hex)
            if old is None or df < old:
                rare_df[sig_hex] = df
    return rare_df


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
    return [cid for cid, *_ in scored[:top_k]], scored[:top_k]


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
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=10, help="number of missed examples to print")
    args = ap.parse_args()

    with open(args.filter, "rb") as f:
        filt = pickle.load(f)

    chunks = load_chunks(args.glyph)
    rng = random.Random(args.seed)

    indexed_ids = sorted(row["chunk_id"] for row in filt["rare"]["chunks"])
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

    dense_inv = filt["dense"]["inv"]
    dense_sig_len = filt["dense"].get("sig_len", 8)
    rare_sig_len = filt["rare"].get("sig_len", 8)
    if dense_sig_len != rare_sig_len:
        raise ValueError(f"sig_len mismatch: dense={dense_sig_len} rare={rare_sig_len}")
    sig_len = dense_sig_len

    rare_df_map = build_rare_df_map(filt)

    missed_examples = []

    agg_num_query_sigs = []
    agg_dense_known = []
    agg_dense_unknown = []
    agg_dense_df = []
    agg_rare_known = []
    agg_rare_df = []

    executed = 0
    missed = 0

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

        shortlist, _shortlist_scored = shortlist_hybrid(filt, frags, args.top_k)
        executed += 1

        if qid in shortlist:
            continue

        missed += 1

        query_sigs = []
        dense_df_vals = []
        rare_df_vals = []
        dense_known = 0
        dense_unknown = 0
        rare_known = 0

        for frag in frags:
            for i in range(len(frag) - sig_len + 1):
                sig = frag[i:i + sig_len].hex()
                query_sigs.append(sig)

                chunk_map = dense_inv.get(sig)
                if chunk_map:
                    dense_known += 1
                    dense_df_vals.append(len(chunk_map))
                else:
                    dense_unknown += 1

                rdf = rare_df_map.get(sig)
                if rdf is not None:
                    rare_known += 1
                    rare_df_vals.append(rdf)

        agg_num_query_sigs.append(len(query_sigs))
        agg_dense_known.append(dense_known)
        agg_dense_unknown.append(dense_unknown)
        agg_dense_df.extend(dense_df_vals)
        agg_rare_known.append(rare_known)
        agg_rare_df.extend(rare_df_vals)

        if len(missed_examples) < args.show:
            ex = {
                "qid": qid,
                "starts": starts,
                "num_query_sigs": len(query_sigs),
                "dense_known": dense_known,
                "dense_unknown": dense_unknown,
                "dense_df_stats": summarize(dense_df_vals),
                "rare_known": rare_known,
                "rare_df_stats": summarize(rare_df_vals),
            }
            missed_examples.append(ex)

    print("=" * 60)
    print(" MISSED DF DIAGNOSTIC V1")
    print("=" * 60)
    print(f" trials={executed}")
    print(f" top_k={args.top_k}")
    print(f" sig_len={sig_len}")
    print(f" missed={missed}")
    if executed > 0:
        print(f" missed_rate={missed / executed:.2%}")

    print("\nAGGREGATES OVER MISSED CASES:")
    print(f"  num_query_sigs: count={len(agg_num_query_sigs)} "
          f"median={fmt_stat(summarize(agg_num_query_sigs)['median'])} "
          f"mean={fmt_stat(summarize(agg_num_query_sigs)['mean'])}")

    ds_known = summarize(agg_dense_known)
    ds_unknown = summarize(agg_dense_unknown)
    ddf = summarize(agg_dense_df)
    rs_known = summarize(agg_rare_known)
    rdf = summarize(agg_rare_df)

    print(f"  dense_known_per_query: "
          f"median={fmt_stat(ds_known['median'])} "
          f"mean={fmt_stat(ds_known['mean'])} "
          f"min={fmt_stat(ds_known['min'])} "
          f"p90={fmt_stat(ds_known['p90'])} "
          f"max={fmt_stat(ds_known['max'])}")

    print(f"  dense_unknown_per_query: "
          f"median={fmt_stat(ds_unknown['median'])} "
          f"mean={fmt_stat(ds_unknown['mean'])} "
          f"min={fmt_stat(ds_unknown['min'])} "
          f"p90={fmt_stat(ds_unknown['p90'])} "
          f"max={fmt_stat(ds_unknown['max'])}")

    print(f"  dense_df_over_known_sigs: "
          f"count={ddf['count']} "
          f"min={fmt_stat(ddf['min'])} "
          f"median={fmt_stat(ddf['median'])} "
          f"p90={fmt_stat(ddf['p90'])} "
          f"max={fmt_stat(ddf['max'])} "
          f"mean={fmt_stat(ddf['mean'])}")

    print(f"  rare_known_per_query: "
          f"median={fmt_stat(rs_known['median'])} "
          f"mean={fmt_stat(rs_known['mean'])} "
          f"min={fmt_stat(rs_known['min'])} "
          f"p90={fmt_stat(rs_known['p90'])} "
          f"max={fmt_stat(rs_known['max'])}")

    print(f"  rare_df_over_known_sigs: "
          f"count={rdf['count']} "
          f"min={fmt_stat(rdf['min'])} "
          f"median={fmt_stat(rdf['median'])} "
          f"p90={fmt_stat(rdf['p90'])} "
          f"max={fmt_stat(rdf['max'])} "
          f"mean={fmt_stat(rdf['mean'])}")

    print("\nMISSED EXAMPLES:")
    for ex in missed_examples:
        print(f"  query_chunk={ex['qid']} starts={ex['starts']}")
        print(f"    num_query_sigs={ex['num_query_sigs']}")
        print(f"    dense_known={ex['dense_known']} dense_unknown={ex['dense_unknown']}")
        d = ex["dense_df_stats"]
        print(f"    dense_df: count={d['count']} min={fmt_stat(d['min'])} "
              f"median={fmt_stat(d['median'])} p90={fmt_stat(d['p90'])} "
              f"max={fmt_stat(d['max'])} mean={fmt_stat(d['mean'])}")
        r = ex["rare_df_stats"]
        print(f"    rare_known={ex['rare_known']}")
        print(f"    rare_df: count={r['count']} min={fmt_stat(r['min'])} "
              f"median={fmt_stat(r['median'])} p90={fmt_stat(r['p90'])} "
              f"max={fmt_stat(r['max'])} mean={fmt_stat(r['mean'])}")


if __name__ == "__main__":
    main()