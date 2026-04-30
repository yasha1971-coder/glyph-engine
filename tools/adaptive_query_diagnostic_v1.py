import argparse
import pickle
from typing import Any, Dict, List

# ============================================================
# LOAD
# ============================================================
def normalize_rows(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict):
        if "results" in obj and isinstance(obj["results"], list):
            return obj["results"]
        if "queries" in obj and isinstance(obj["queries"], list):
            return obj["queries"]
        if all(isinstance(v, dict) for v in obj.values()):
            return list(obj.values())
    raise ValueError("unsupported format")


def shortlist_from_any(vals: Any, top_k: int) -> List[int]:
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

    # dedup
    seen = set()
    res = []
    for x in out:
        if x not in seen:
            seen.add(x)
            res.append(x)
    return res[:top_k]


def get_qid(row: Dict[str, Any]) -> int:
    for k in ("qid", "query_chunk", "q_chunk", "chunk_id"):
        if k in row:
            return int(row[k])
    raise KeyError("no qid")


# ============================================================
# ADAPTIVE MERGE
# ============================================================
def collect_views(row: Dict[str, Any], top_k: int) -> Dict[str, List[int]]:
    views = {}

    # базовый
    if "shortlist" in row:
        views["base"] = shortlist_from_any(row["shortlist"], top_k)

    # pair-based
    for k in row.keys():
        if k.startswith("pair") and isinstance(row[k], list):
            views[k] = shortlist_from_any(row[k], top_k)

    return views


def union_merge(views: Dict[str, List[int]], top_k: int) -> List[int]:
    seen = set()
    out = []
    for v in views.values():
        for cid in v:
            if cid not in seen:
                seen.add(cid)
                out.append(cid)
            if len(out) >= top_k:
                return out
    return out[:top_k]


def vote_merge(views: Dict[str, List[int]], top_k: int) -> List[int]:
    score = {}
    for v in views.values():
        for rank, cid in enumerate(v):
            score[cid] = score.get(cid, 0) + (top_k - rank)

    ranked = sorted(score.items(), key=lambda x: (-x[1], x[0]))
    return [cid for cid, _ in ranked[:top_k]]


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-results", required=True)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    print("=" * 60)
    print(" ADAPTIVE QUERY DIAGNOSTIC V1")
    print("=" * 60)

    with open(args.gap_results, "rb") as f:
        rows = normalize_rows(pickle.load(f))

    base_hit = 0
    union_hit = 0
    vote_hit = 0
    total = 0

    examples = []

    for row in rows[:args.limit]:
        try:
            qid = get_qid(row)
            views = collect_views(row, args.top_k)
            if not views:
                continue

            base = views.get("base", [])
            union = union_merge(views, args.top_k)
            vote = vote_merge(views, args.top_k)

            total += 1

            if qid in base:
                base_hit += 1
            if qid in union:
                union_hit += 1
            if qid in vote:
                vote_hit += 1

            if len(examples) < args.show:
                examples.append({
                    "qid": qid,
                    "base_hit": qid in base,
                    "union_hit": qid in union,
                    "vote_hit": qid in vote,
                    "views": {k: v[:5] for k, v in views.items()},
                })

        except Exception:
            continue

    if total == 0:
        raise ValueError("no usable rows")

    print("\nRESULTS:")
    print(f"  base_hit@{args.top_k}  = {base_hit / total:.2%}")
    print(f"  union_hit@{args.top_k} = {union_hit / total:.2%}")
    print(f"  vote_hit@{args.top_k}  = {vote_hit / total:.2%}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(ex)


if __name__ == "__main__":
    main()