import argparse
import pickle
import numpy as np
from collections import Counter

def load_chunks(corpus_path, chunk_size=16384):
    chunks = []
    with open(corpus_path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            chunks.append(block)
    return chunks


def build_query(chunk, starts, frag_len=48):
    parts = [chunk[s:s+frag_len] for s in starts]
    return b"".join(parts)


# --- TOKENIZATION (очень простая, но быстрая) ---
def tokenize_bytes(x):
    return list(x)  # байтовые токены


# --- LATE INTERACTION APPROX ---
def late_interaction_score(query, candidate):
    q_tokens = tokenize_bytes(query)
    c_tokens = tokenize_bytes(candidate)

    c_count = Counter(c_tokens)

    score = 0.0
    for t in q_tokens:
        if t in c_count:
            score += 1.0  # binary match
    return score / max(1, len(q_tokens))


def shortlist_from_any(vals, top_k):
    out = []
    if not isinstance(vals, (list, tuple)):
        return out

    for item in vals:
        cid = None
        if isinstance(item, int):
            cid = item
        elif isinstance(item, (list, tuple)) and len(item) >= 1:
            cid = int(item[0])
        elif isinstance(item, dict):
            for k in ("cid", "chunk", "chunk_id", "id"):
                if k in item:
                    cid = int(item[k])
                    break

        if cid is not None:
            out.append(cid)

        if len(out) >= top_k:
            break

    return list(dict.fromkeys(out))[:top_k]


def get_shortlist(row, top_k):
    for key in ("shortlist", "ranked", "top", "topk", "pair_top", "fusion_top", "candidates"):
        if key in row:
            sl = shortlist_from_any(row[key], top_k)
            if sl:
                return sl
    return []


def get_qid(row):
    for k in ("qid", "query_chunk", "q_chunk", "chunk_id"):
        if k in row:
            return int(row[k])
    raise KeyError("no qid")


def get_starts(row):
    return [int(x) for x in row["starts"]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-results", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    with open(args.gap_results, "rb") as f:
        results = pickle.load(f)

    if isinstance(results, dict):
        results = results.get("results", list(results.values()))

    chunks = load_chunks(args.corpus)

    hit1 = hit4 = hit8 = hit16 = total = 0
    examples = []

    print("="*60)
    print(" LATE INTERACTION RERANK V1")
    print("="*60)

    for row in results:
        qid = get_qid(row)
        starts = get_starts(row)
        shortlist = get_shortlist(row, args.top_k)

        if not shortlist:
            continue

        query = build_query(chunks[qid], starts)

        scores = []
        for cid in shortlist:
            score = late_interaction_score(query, chunks[cid])
            scores.append((cid, score))

        scores.sort(key=lambda x: -x[1])
        ranked = [cid for cid, _ in scores]

        rank = None
        for i, cid in enumerate(ranked):
            if cid == qid:
                rank = i + 1
                break

        total += 1

        if rank == 1: hit1 += 1
        if rank and rank <= 4: hit4 += 1
        if rank and rank <= 8: hit8 += 1
        if rank and rank <= 16: hit16 += 1

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "rank": rank,
                "top5": scores[:5]
            })

    print("\nRESULTS:")
    print(f"  hit@1  = {hit1 / total:.2%}")
    print(f"  hit@4  = {hit4 / total:.2%}")
    print(f"  hit@8  = {hit8 / total:.2%}")
    print(f"  hit@16 = {hit16 / total:.2%}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(ex)


if __name__ == "__main__":
    main()