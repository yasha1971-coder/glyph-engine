import argparse
import math
import pickle
import random
import statistics
import struct
import zlib
from collections import defaultdict


def load_chunks(glyph_path):
    chunks = []
    with open(glyph_path, "rb") as f:
        magic = f.read(6)
        if magic != b"GLYPH1":
            raise ValueError(f"Bad glyph magic: {magic!r}")
        _version, ng = struct.unpack("<BH", f.read(3))
        for _ in range(ng):
            sz = struct.unpack("<I", f.read(4))[0]
            chunks.append(zlib.decompress(f.read(sz)))
    return chunks


def iter_signatures(data: bytes, sig_len: int, stride: int):
    if len(data) < sig_len:
        return
    for i in range(0, len(data) - sig_len + 1, stride):
        yield data[i:i + sig_len]


def choose_fragment_starts(chunk_len, frag_len, nfrag, min_gap, rng):
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    if chunk_len < need:
        return None
    base = rng.randint(0, chunk_len - need)
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def build_fragment_set(chunk: bytes, frag_len: int, nfrag: int, min_gap: int, rng):
    starts = choose_fragment_starts(len(chunk), frag_len, nfrag, min_gap, rng)
    if starts is None:
        return None, None
    frags = [chunk[s:s + frag_len] for s in starts]
    return starts, frags


def dense_vote(filt_dense, query_frags):
    sig_len = filt_dense["sig_len"]
    stride = filt_dense["stride"]
    inv = filt_dense["inv"]

    chunk_votes = defaultdict(lambda: [0] * len(query_frags))

    for fi, frag in enumerate(query_frags):
        for sig in iter_signatures(frag, sig_len, stride):
            sig_hex = sig.hex()
            chunk_map = inv.get(sig_hex)
            if not chunk_map:
                continue
            for chunk_id_str, cnt in chunk_map.items():
                chunk_id = int(chunk_id_str)
                chunk_votes[chunk_id][fi] += cnt

    dense_stats = {}
    for chunk_id, votes in chunk_votes.items():
        present = sum(v > 0 for v in votes)
        total = sum(votes)
        dense_stats[chunk_id] = (present, total, votes)

    return dense_stats


def build_rare_maps(filt_rare):
    chunk_anchor_maps = {}
    for row in filt_rare["chunks"]:
        amap = {}
        for sig_hex, cnt, df in row["rare_anchors"]:
            amap[sig_hex] = (cnt, df)
        chunk_anchor_maps[row["chunk_id"]] = amap
    return chunk_anchor_maps


def rare_vote(filt_rare, query_frags, chunk_anchor_maps):
    sig_len = filt_rare["sig_len"]
    stride = filt_rare["stride"]
    global_df = filt_rare["global_df"]
    N = len(filt_rare["chunks"])

    frag_votes = defaultdict(lambda: [0] * len(query_frags))
    weighted = defaultdict(float)

    for fi, frag in enumerate(query_frags):
        seen = set()
        for sig in iter_signatures(frag, sig_len, stride):
            sig_hex = sig.hex()
            if sig_hex in seen:
                continue
            seen.add(sig_hex)

            df = global_df.get(sig_hex)
            if df is None:
                continue

            idf = math.log(1.0 + N / max(1, df))

            for chunk_id, amap in chunk_anchor_maps.items():
                hit = amap.get(sig_hex)
                if hit is None:
                    continue
                local_cnt, _ = hit
                frag_votes[chunk_id][fi] += 1
                weighted[chunk_id] += local_cnt * idf

    rare_stats = {}
    for chunk_id, votes in frag_votes.items():
        present = sum(v > 0 for v in votes)
        matches = sum(votes)
        score = weighted[chunk_id]
        rare_stats[chunk_id] = (present, matches, score, votes)

    return rare_stats


def hybrid_shortlist(filt_dense, filt_rare, query_frags, top_k):
    dense_stats = dense_vote(filt_dense, query_frags)
    chunk_anchor_maps = build_rare_maps(filt_rare)
    rare_stats = rare_vote(filt_rare, query_frags, chunk_anchor_maps)

    all_chunks = set(dense_stats.keys()) | set(rare_stats.keys())
    scored = []

    for cid in all_chunks:
        d_present, d_total, d_votes = dense_stats.get(cid, (0, 0, [0] * len(query_frags)))
        r_present, r_matches, r_score, r_votes = rare_stats.get(cid, (0, 0, 0.0, [0] * len(query_frags)))

        scored.append((
            cid,
            r_present,
            d_present,
            r_score,
            d_total,
            r_matches,
            d_votes,
            r_votes,
        ))

    scored.sort(key=lambda x: (
        -x[1],
        -x[2],
        -x[3],
        -x[4],
        -x[5],
        x[0]
    ))
    return scored[:top_k]


def rank_of_chunk(ids, target):
    for i, cid in enumerate(ids, 1):
        if cid == target:
            return i
    return None


def reciprocal_rank(rank):
    return 0.0 if rank is None else 1.0 / rank


def hit_at(rank, k):
    return rank is not None and rank <= k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter-dense", required=True)
    ap.add_argument("--filter-rare", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open(args.filter_dense, "rb") as f:
        filt_dense = pickle.load(f)
    with open(args.filter_rare, "rb") as f:
        filt_rare = pickle.load(f)

    chunks = load_chunks(args.glyph)
    rng = random.Random(args.seed)

    dense_ids = set()
    for chunk_map in filt_dense["inv"].values():
        for cid in chunk_map.keys():
            dense_ids.add(int(cid))

    rare_ids = set(row["chunk_id"] for row in filt_rare["chunks"])

    indexed_ids = sorted(dense_ids & rare_ids)
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

    print("=" * 60)
    print("  CHUNK BENCH HYBRID V1")
    print("=" * 60)
    print(f"  indexed_chunks={len(indexed_ids)}")
    print(f"  eligible_chunks={len(eligible_ids)}")
    print(f"  trials={len(bench_ids)}")
    print(f"  top_k={args.top_k}")

    hits = {1: 0, 4: 0, 8: 0, 16: 0}
    rr_list = []
    rank_list = []

    true_dense_present = []
    true_dense_total = []
    true_rare_present = []
    true_rare_matches = []
    true_rare_score = []

    examples = []

    for qid in bench_ids:
        local_rng = random.Random(args.seed + qid)
        starts, frags = build_fragment_set(
            chunks[qid], args.frag_len, args.nfrag, args.min_gap, local_rng
        )
        if frags is None:
            continue

        scored = hybrid_shortlist(filt_dense, filt_rare, frags, args.top_k)
        shortlist_ids = [row[0] for row in scored]
        rank = rank_of_chunk(shortlist_ids, qid)

        for k in hits:
            if hit_at(rank, k):
                hits[k] += 1

        rr_list.append(reciprocal_rank(rank))
        rank_list.append(rank if rank is not None else args.top_k + 1)

        for row in scored:
            cid, r_present, d_present, r_score, d_total, r_matches, d_votes, r_votes = row
            if cid == qid:
                true_dense_present.append(d_present)
                true_dense_total.append(d_total)
                true_rare_present.append(r_present)
                true_rare_matches.append(r_matches)
                true_rare_score.append(r_score)
                break

        if len(examples) < 5:
            examples.append({
                "qid": qid,
                "starts": starts,
                "rank": rank,
                "top5": scored[:5],
            })

    n = len(rr_list)
    if n == 0:
        raise ValueError("no valid trials executed")

    print("")
    print("  HYBRID SHORTLIST:")
    for k in (1, 4, 8, 16):
        print(f"    hit@{k} = {100.0 * hits[k] / n:.1f}%")
    print(f"    MRR = {statistics.mean(rr_list):.4f}")
    print(f"    mean_true_rank = {statistics.mean(rank_list):.3f}")

    if true_dense_present:
        print(f"    mean_true_dense_present = {statistics.mean(true_dense_present):.3f}")
    if true_dense_total:
        print(f"    mean_true_dense_total = {statistics.mean(true_dense_total):.3f}")
    if true_rare_present:
        print(f"    mean_true_rare_present = {statistics.mean(true_rare_present):.3f}")
    if true_rare_matches:
        print(f"    mean_true_rare_matches = {statistics.mean(true_rare_matches):.3f}")
    if true_rare_score:
        print(f"    mean_true_rare_score = {statistics.mean(true_rare_score):.3f}")

    print("")
    print("  examples:")
    for ex in examples:
        print(f"    query_chunk={ex['qid']} starts={ex['starts']} rank={ex['rank']}")
        for row in ex["top5"]:
            cid, r_present, d_present, r_score, d_total, r_matches, d_votes, r_votes = row
            marker = " <== TRUE" if cid == ex["qid"] else ""
            print(
                f"      chunk={cid} "
                f"rare_present={r_present}/{args.nfrag} "
                f"dense_present={d_present}/{args.nfrag} "
                f"rare_score={r_score:.2f} "
                f"dense_total={d_total} "
                f"rare_matches={r_matches}{marker}"
            )


if __name__ == "__main__":
    main()