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
# HYBRID SHORTLIST
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

    # infer sig_len from filter metadata
    dense_sig_len = filter_obj["dense"].get("sig_len", 8)
    rare_sig_len = filter_obj["rare"].get("sig_len", 8)
    if dense_sig_len != rare_sig_len:
        raise ValueError(f"sig_len mismatch: dense={dense_sig_len} rare={rare_sig_len}")
    sig_len = dense_sig_len

    chunk_scores = defaultdict(lambda: {
        "dense_present": 0,
        "dense_total": 0,
        "rare_present": 0,
        "rare_matches": 0,
        "rare_score": 0.0
    })

    # ----- dense side -----
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

    # ----- rare side -----
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
# EXACT RERANK
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

    hit1 = hit4 = hit8 = hitK = 0
    mrr = 0.0
    shortlist_hitK = 0

    executed = 0
    examples = []

    print("=" * 60)
    print("  HYBRID PIPELINE + EXACT RERANK")
    print("=" * 60)
    print(f"  indexed_chunks={len(indexed_ids)}")
    print(f"  eligible_chunks={len(eligible_ids)}")
    print(f"  trials={len(bench_ids)}")
    print(f"  top_k={args.top_k}")

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

        shortlist, shortlist_scored = shortlist_hybrid(filt, frags, args.top_k)

        if qid in shortlist:
            shortlist_hitK += 1

        if qid not in shortlist:
            executed += 1
            if len(examples) < 5:
                examples.append({
                    "qid": qid,
                    "starts": starts,
                    "shortlist": shortlist_scored[:5],
                    "rerank": None,
                    "final_rank": None,
                })
            continue

        reranked = rerank_exact(chunks, frags, shortlist)

        rank = None
        for i, (cid, _present, _total) in enumerate(reranked):
            if cid == qid:
                rank = i + 1
                break

        if rank is not None:
            if rank == 1:
                hit1 += 1
            if rank <= 4:
                hit4 += 1
            if rank <= 8:
                hit8 += 1
            if rank <= args.top_k:
                hitK += 1
            mrr += 1.0 / rank

        executed += 1

        if len(examples) < 5:
            examples.append({
                "qid": qid,
                "starts": starts,
                "shortlist": shortlist_scored[:5],
                "rerank": reranked[:5],
                "final_rank": rank,
            })

    if executed == 0:
        raise ValueError("no executed trials")

    print("\n  RESULTS:")
    print(f"    shortlist_hit@{args.top_k} = {shortlist_hitK / executed:.2%}")
    print(f"    rerank_hit@1     = {hit1 / executed:.2%}")
    print(f"    rerank_hit@4     = {hit4 / executed:.2%}")
    print(f"    rerank_hit@8     = {hit8 / executed:.2%}")
    print(f"    rerank_hit@{args.top_k}    = {hitK / executed:.2%}")
    print(f"    rerank_MRR       = {mrr / executed:.4f}")

    print("\n  EXAMPLES:")
    for ex in examples:
        print(f"    query_chunk={ex['qid']} starts={ex['starts']} final_rank={ex['final_rank']}")
        print("      shortlist_top5:")
        for row in ex["shortlist"]:
            cid, rare_present, dense_present, rare_score, dense_total, rare_matches = row
            marker = " <== TRUE" if cid == ex["qid"] else ""
            print(
                f"        chunk={cid} "
                f"rare_present={rare_present} "
                f"dense_present={dense_present} "
                f"rare_score={rare_score:.4f} "
                f"dense_total={dense_total} "
                f"rare_matches={rare_matches}{marker}"
            )
        if ex["rerank"] is not None:
            print("      rerank_top5:")
            for cid, present, total in ex["rerank"]:
                marker = " <== TRUE" if cid == ex["qid"] else ""
                print(
                    f"        chunk={cid} present={present} total={total}{marker}"
                )


if __name__ == "__main__":
    main()