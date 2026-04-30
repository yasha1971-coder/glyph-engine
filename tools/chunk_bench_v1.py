import argparse
import json
import math
import pickle
import random
import statistics
import struct
import zlib
from collections import Counter


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


def shortlist_chunks(filter_obj, query: bytes, top_k: int):
    sig_len = filter_obj["sig_len"]
    stride = filter_obj["stride"]
    qsig = Counter(iter_signatures(query, sig_len, stride))

    scored = []
    for row in filter_obj["chunks"]:
        chunk_id = row["chunk_id"]
        top_map = {bytes.fromhex(sig_hex): cnt for sig_hex, cnt in row["top_signatures"]}

        overlap = 0
        weighted = 0
        for sig, qcnt in qsig.items():
            if sig in top_map:
                overlap += 1
                weighted += qcnt * top_map[sig]

        scored.append((chunk_id, overlap, weighted))

    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return scored[:top_k]


def choose_fragment_starts(chunk_len, frag_len, nfrag, min_gap):
    need = nfrag * frag_len + (nfrag - 1) * min_gap
    if chunk_len < need:
        return None
    base = 0
    return [base + i * (frag_len + min_gap) for i in range(nfrag)]


def count_occurrences(haystack: bytes, needle: bytes) -> int:
    if not needle or len(needle) > len(haystack):
        return 0
    cnt = 0
    start = 0
    while True:
        pos = haystack.find(needle, start)
        if pos == -1:
            break
        cnt += 1
        start = pos + 1
    return cnt


def rerank_shortlist(manifest_by_id, shortlist, query_chunk, frag_len, nfrag, min_gap):
    starts = choose_fragment_starts(len(query_chunk), frag_len, nfrag, min_gap)
    if starts is None:
        raise ValueError("query chunk too short for requested fragment regime")

    query_frags = [query_chunk[s:s + frag_len] for s in starts]
    results = []

    for chunk_id, overlap, weighted in shortlist:
        rec = manifest_by_id[chunk_id]
        raw_path = rec["chunk_dir"] + "/chunk.raw.bin"
        with open(raw_path, "rb") as f:
            raw = f.read()

        frag_counts = [count_occurrences(raw, frag) for frag in query_frags]
        present = sum(c > 0 for c in frag_counts)
        mn = min((c for c in frag_counts if c > 0), default=0)
        total = sum(frag_counts)

        score = (present, mn, total, weighted, overlap)
        results.append({
            "chunk_id": chunk_id,
            "overlap": overlap,
            "weighted": weighted,
            "frag_counts": frag_counts,
            "present": present,
            "mn": mn,
            "total": total,
            "score": score,
        })

    results.sort(key=lambda r: (-r["present"], -r["mn"], -r["total"], -r["weighted"], -r["overlap"], r["chunk_id"]))
    return results


def rank_of_chunk(ids, target):
    for i, cid in enumerate(ids, 1):
        if cid == target:
            return i
    return None


def reciprocal_rank(rank):
    if rank is None:
        return 0.0
    return 1.0 / rank


def hit_at(rank, k):
    return rank is not None and rank <= k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--shortlist-k", type=int, default=16)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--trials", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-id-start", type=int, default=0,
                    help="start chunk id for sequential benchmark window")
    args = ap.parse_args()

    with open(args.filter, "rb") as f:
        filter_obj = pickle.load(f)

    with open(args.manifest, "r", encoding="utf-8") as f:
        manifest_records = [json.loads(line) for line in f]
    manifest_by_id = {r["chunk_id"]: r for r in manifest_records}

    chunks = load_chunks(args.glyph)
    available_ids = sorted(manifest_by_id.keys())
    if not available_ids:
        raise ValueError("empty manifest")

    # restrict queries to manifest-covered chunk ids
    query_ids = [cid for cid in available_ids if cid < len(chunks)]

    if args.query_id_start > 0:
        query_ids = [cid for cid in query_ids if cid >= args.query_id_start]

    if not query_ids:
        raise ValueError("no valid query ids after filtering")

    rng = random.Random(args.seed)
    if args.trials >= len(query_ids):
        bench_ids = query_ids[:args.trials]
    else:
        bench_ids = rng.sample(query_ids, args.trials)

    print("=" * 60)
    print("  CHUNK BENCH V1")
    print("=" * 60)
    print(f"  filter={args.filter}")
    print(f"  manifest={args.manifest}")
    print(f"  glyph={args.glyph}")
    print(f"  shortlist_k={args.shortlist_k}")
    print(f"  frag_len={args.frag_len}")
    print(f"  nfrag={args.nfrag}")
    print(f"  min_gap={args.min_gap}")
    print(f"  trials={len(bench_ids)}")
    print(f"  seed={args.seed}")

    shortlist_hits = {1: 0, 4: 0, 8: 0, 16: 0}
    rerank_hits = {1: 0, 4: 0, 8: 0, 16: 0}

    shortlist_rr = []
    rerank_rr = []

    shortlist_ranks = []
    rerank_ranks = []

    shortlist_overlap_true = []
    shortlist_weighted_true = []

    rerank_present_true = []
    rerank_mn_true = []
    rerank_total_true = []

    examples = []

    for qid in bench_ids:
        query_chunk = chunks[qid]

        shortlist = shortlist_chunks(filter_obj, query_chunk, args.shortlist_k)
        shortlist_ids = [x[0] for x in shortlist]
        s_rank = rank_of_chunk(shortlist_ids, qid)

        for k in shortlist_hits:
            if hit_at(s_rank, k):
                shortlist_hits[k] += 1

        shortlist_rr.append(reciprocal_rank(s_rank))
        shortlist_ranks.append(s_rank if s_rank is not None else args.shortlist_k + 1)

        true_short = None
        for chunk_id, overlap, weighted in shortlist:
            if chunk_id == qid:
                true_short = (overlap, weighted)
                break
        if true_short is not None:
            shortlist_overlap_true.append(true_short[0])
            shortlist_weighted_true.append(true_short[1])

        reranked = rerank_shortlist(
            manifest_by_id=manifest_by_id,
            shortlist=shortlist,
            query_chunk=query_chunk,
            frag_len=args.frag_len,
            nfrag=args.nfrag,
            min_gap=args.min_gap,
        )
        rerank_ids = [r["chunk_id"] for r in reranked]
        r_rank = rank_of_chunk(rerank_ids, qid)

        for k in rerank_hits:
            if hit_at(r_rank, k):
                rerank_hits[k] += 1

        rerank_rr.append(reciprocal_rank(r_rank))
        rerank_ranks.append(r_rank if r_rank is not None else args.shortlist_k + 1)

        true_rerank = None
        for row in reranked:
            if row["chunk_id"] == qid:
                true_rerank = row
                break

        if true_rerank is not None:
            rerank_present_true.append(true_rerank["present"])
            rerank_mn_true.append(true_rerank["mn"])
            rerank_total_true.append(true_rerank["total"])

        if len(examples) < 5:
            examples.append({
                "qid": qid,
                "short_rank": s_rank,
                "rerank_rank": r_rank,
                "short_top5": shortlist[:5],
                "rerank_top5": [
                    {
                        "chunk_id": r["chunk_id"],
                        "present": r["present"],
                        "mn": r["mn"],
                        "total": r["total"],
                        "weighted": r["weighted"],
                        "overlap": r["overlap"],
                    }
                    for r in reranked[:5]
                ],
            })

    n = len(bench_ids)

    print("")
    print("  SHORTLIST STAGE:")
    for k in (1, 4, 8, 16):
        print(f"    hit@{k} = {100.0 * shortlist_hits[k] / n:.1f}%")
    print(f"    MRR = {statistics.mean(shortlist_rr):.4f}")
    print(f"    mean_true_rank = {statistics.mean(shortlist_ranks):.3f}")
    if shortlist_overlap_true:
        print(f"    mean_true_overlap = {statistics.mean(shortlist_overlap_true):.2f}")
    if shortlist_weighted_true:
        print(f"    mean_true_weighted = {statistics.mean(shortlist_weighted_true):.2f}")

    print("")
    print("  RERANK STAGE:")
    for k in (1, 4, 8, 16):
        print(f"    hit@{k} = {100.0 * rerank_hits[k] / n:.1f}%")
    print(f"    MRR = {statistics.mean(rerank_rr):.4f}")
    print(f"    mean_true_rank = {statistics.mean(rerank_ranks):.3f}")
    if rerank_present_true:
        print(f"    mean_true_present = {statistics.mean(rerank_present_true):.3f}")
    if rerank_mn_true:
        print(f"    mean_true_mn = {statistics.mean(rerank_mn_true):.3f}")
    if rerank_total_true:
        print(f"    mean_true_total = {statistics.mean(rerank_total_true):.3f}")

    print("")
    print("  examples:")
    for ex in examples:
        print(f"    query_chunk={ex['qid']} short_rank={ex['short_rank']} rerank_rank={ex['rerank_rank']}")
        print(f"      shortlist_top5={ex['short_top5']}")
        print(f"      rerank_top5={ex['rerank_top5']}")


if __name__ == "__main__":
    main()