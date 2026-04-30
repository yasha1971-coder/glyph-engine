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
    if len(data) < k:
        return
    for i in range(0, len(data) - k + 1):
        yield data[i:i + k]


def compute_minimizers(data: bytes, k: int, w: int):
    kmers = list(iter_kmers(data, k))
    if not kmers:
        return []

    if len(kmers) <= w:
        return [min(kmers)]

    mins = []
    for i in range(0, len(kmers) - w + 1):
        window = kmers[i:i + w]
        mins.append(min(window))
    return mins


# -----------------------------
# BUILD MAPS
# -----------------------------
def build_chunk_minimizer_maps(filter_obj):
    chunk_maps = {}
    for row in filter_obj["chunks"]:
        cmap = {}
        for tok_hex, cnt, df in row["minimizers"]:
            cmap[tok_hex] = (cnt, df)
        chunk_maps[row["chunk_id"]] = cmap
    return chunk_maps


# -----------------------------
# STAGE A: MINIMIZER SHORTLIST
# -----------------------------
def shortlist_minimizer(filter_obj, query_frags, top_k, query_max_df):
    N = len(filter_obj["chunks"])
    k = filter_obj["k"]
    w = filter_obj["w"]
    global_df = filter_obj["global_df"]
    chunk_maps = build_chunk_minimizer_maps(filter_obj)

    chunk_votes = defaultdict(lambda: [0] * len(query_frags))
    chunk_weighted = defaultdict(float)

    for fi, frag in enumerate(query_frags):
        seen = set()
        mins = compute_minimizers(frag, k, w)

        for tok in mins:
            tok_hex = tok.hex()
            if tok_hex in seen:
                continue
            seen.add(tok_hex)

            df = global_df.get(tok_hex)
            if df is None:
                continue
            if df > query_max_df:
                continue

            idf = math.log(1.0 + N / max(1, df))

            for chunk_id, cmap in chunk_maps.items():
                hit = cmap.get(tok_hex)
                if hit is None:
                    continue
                local_cnt, _local_df = hit
                chunk_votes[chunk_id][fi] += 1
                chunk_weighted[chunk_id] += local_cnt * idf

    scored = []
    for chunk_id, votes in chunk_votes.items():
        present_fragments = sum(v > 0 for v in votes)
        total_votes = sum(votes)
        idf_weighted_score = chunk_weighted[chunk_id]
        max_fragment_votes = max(votes) if votes else 0
        min_fragment_votes = min((v for v in votes if v > 0), default=0)

        scored.append((
            chunk_id,
            present_fragments,
            idf_weighted_score,
            total_votes,
            max_fragment_votes,
            min_fragment_votes,
            votes,
        ))

    scored.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4], -x[5], x[0]))
    return [row[0] for row in scored[:top_k]], scored[:top_k]


# -----------------------------
# STAGE B: EXACT RERANK
# -----------------------------
def exact_score(query_frags, candidate_chunk):
    present = 0
    total = 0

    for frag in query_frags:
        cnt = candidate_chunk.count(frag)
        if cnt > 0:
            present += 1
            total += cnt

    return present, total


def rerank_exact(chunks, query_frags, candidates):
    scored = []

    for cid in candidates:
        present, total = exact_score(query_frags, chunks[cid])
        scored.append((cid, present, total))

    scored.sort(key=lambda x: (-x[1], -x[2], x[0]))
    return scored


# -----------------------------
# MAIN BENCH
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--filter", required=True)
    ap.add_argument("--glyph", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--nfrag", type=int, default=5)
    ap.add_argument("--min-gap", type=int, default=128)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--trials", type=int, default=50)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--query-max-df", type=int, default=128)
    args = ap.parse_args()

    with open(args.filter, "rb") as f:
        filt = pickle.load(f)

    chunks = load_chunks(args.glyph)
    rng = random.Random(args.seed)

    indexed_ids = sorted(row["chunk_id"] for row in filt["chunks"])
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

    shortlist_hit = 0
    rerank_hit1 = 0
    rerank_hit4 = 0
    rerank_hit8 = 0
    rerank_hit16 = 0
    rerank_mrr = 0.0
    executed = 0

    examples = []

    print("=" * 60)
    print("  MINIMIZER PIPELINE + EXACT RERANK")
    print("=" * 60)
    print(f"  indexed_chunks={len(indexed_ids)}")
    print(f"  eligible_chunks={len(eligible_ids)}")
    print(f"  trials={len(bench_ids)}")
    print(f"  top_k={args.top_k}")
    print(f"  query_max_df={args.query_max_df}")

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

        shortlist, shortlist_scored = shortlist_minimizer(
            filt, frags, args.top_k, args.query_max_df
        )

        if qid in shortlist:
            shortlist_hit += 1

        rank = None
        reranked = None

        if qid in shortlist:
            reranked = rerank_exact(chunks, frags, shortlist)
            for i, (cid, _present, _total) in enumerate(reranked):
                if cid == qid:
                    rank = i + 1
                    break

            if rank is not None:
                if rank == 1:
                    rerank_hit1 += 1
                if rank <= 4:
                    rerank_hit4 += 1
                if rank <= 8:
                    rerank_hit8 += 1
                if rank <= 16:
                    rerank_hit16 += 1
                rerank_mrr += 1.0 / rank

        executed += 1

        if len(examples) < 5:
            examples.append({
                "qid": qid,
                "starts": starts,
                "shortlist": shortlist_scored[:5],
                "rerank": None if reranked is None else reranked[:5],
                "final_rank": rank,
            })

    if executed == 0:
        raise ValueError("no executed trials")

    print("\n  RESULTS:")
    print(f"    shortlist_hit@{args.top_k:<2} = {shortlist_hit / executed:.2%}")
    print(f"    rerank_hit@1      = {rerank_hit1 / executed:.2%}")
    print(f"    rerank_hit@4      = {rerank_hit4 / executed:.2%}")
    print(f"    rerank_hit@8      = {rerank_hit8 / executed:.2%}")
    print(f"    rerank_hit@16     = {rerank_hit16 / executed:.2%}")
    print(f"    rerank_MRR        = {rerank_mrr / executed:.4f}")

    print("\n  EXAMPLES:")
    for ex in examples:
        print(f"    query_chunk={ex['qid']} starts={ex['starts']} final_rank={ex['final_rank']}")
        print("      shortlist_top5:")
        for row in ex["shortlist"]:
            cid, present_fragments, idf_weighted_score, total_votes, max_fragment_votes, min_fragment_votes, votes = row
            marker = " <== TRUE" if cid == ex["qid"] else ""
            print(
                f"        chunk={cid} "
                f"present_fragments={present_fragments} "
                f"idf_weighted_score={idf_weighted_score:.3f} "
                f"total_votes={total_votes} "
                f"max_fragment_votes={max_fragment_votes} "
                f"min_fragment_votes={min_fragment_votes} "
                f"votes={votes}{marker}"
            )
        if ex["rerank"] is not None:
            print("      rerank_top5:")
            for cid, present, total in ex["rerank"]:
                marker = " <== TRUE" if cid == ex["qid"] else ""
                print(f"        chunk={cid} present={present} total={total}{marker}")


if __name__ == "__main__":
    main()