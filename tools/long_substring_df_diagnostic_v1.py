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
# BUILD GLOBAL SUBSTRING DF
# -----------------------------
def build_df_index(chunks, L, stride):
    df_map = defaultdict(set)

    for cid, chunk in enumerate(chunks):
        seen = set()
        for i in range(0, len(chunk) - L + 1, stride):
            sig = chunk[i:i+L]
            seen.add(sig)
        for sig in seen:
            df_map[sig].add(cid)

    df = {k: len(v) for k, v in df_map.items()}
    return df


# -----------------------------
# MAIN
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline-filter", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=10)
    args = ap.parse_args()

    # load
    with open(args.baseline_filter, "rb") as f:
        filt = pickle.load(f)

    chunks = load_chunks(args.glyph)
    rng = random.Random(args.seed)

    indexed_ids = sorted(row["chunk_id"] for row in filt["rare"]["chunks"])

    eligible_ids = [
        cid for cid in indexed_ids
        if cid < len(chunks)
        and len(chunks[cid]) >= args.nfrag * args.frag_len + (args.nfrag - 1) * args.min_gap
    ]

    bench_ids = rng.sample(eligible_ids, min(args.trials, len(eligible_ids)))

    # -----------------------------
    # build DF indexes
    # -----------------------------
    print("Building DF indexes...")
    df24 = build_df_index(chunks, 24, 12)
    df32 = build_df_index(chunks, 32, 16)
    df40 = build_df_index(chunks, 40, 20)

    # -----------------------------
    # collect missed queries
    # -----------------------------
    def shortlist_hybrid(filter_obj, frags, top_k):
        dense_inv = filter_obj["dense"]["inv"]
        scores = defaultdict(int)

        for frag in frags:
            for i in range(len(frag) - 7):
                sig = frag[i:i+8].hex()
                if sig in dense_inv:
                    for cid in dense_inv[sig]:
                        scores[int(cid)] += 1

        ranked = sorted(scores.items(), key=lambda x: -x[1])
        return [cid for cid, _ in ranked[:top_k]]

    missed = []

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

        shortlist = shortlist_hybrid(filt, frags, args.top_k)

        if qid not in shortlist:
            missed.append((qid, frags))

    # -----------------------------
    # analyze
    # -----------------------------
    def analyze(df_map, L, stride):
        all_dfs = []

        for qid, frags in missed:
            for frag in frags:
                for i in range(0, len(frag) - L + 1, stride):
                    sig = frag[i:i+L]
                    if sig in df_map:
                        all_dfs.append(df_map[sig])

        if not all_dfs:
            return None

        all_dfs.sort()
        n = len(all_dfs)

        return {
            "count": n,
            "min": all_dfs[0],
            "median": all_dfs[n//2],
            "p90": all_dfs[int(n*0.9)],
            "max": all_dfs[-1],
            "mean": sum(all_dfs)/n
        }

    res24 = analyze(df24, 24, 12)
    res32 = analyze(df32, 32, 16)
    res40 = analyze(df40, 40, 20)

    # -----------------------------
    # output
    # -----------------------------
    print("=" * 60)
    print(" LONG SUBSTRING DF DIAGNOSTIC")
    print("=" * 60)
    print(f"missed={len(missed)} / {len(bench_ids)}")

    print("\nL=24:")
    print(res24)

    print("\nL=32:")
    print(res32)

    print("\nL=40:")
    print(res40)


if __name__ == "__main__":
    main()