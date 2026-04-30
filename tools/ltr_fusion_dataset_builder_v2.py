import argparse
import csv
import pickle
import struct
from typing import Any, Dict, List, Tuple, Optional


# ============================================================
# LOADERS
# ============================================================
def load_pickle(path: str):
    with open(path, "rb") as f:
        return pickle.load(f)


def load_chunk_map_header(chunk_map_bin: str) -> Tuple[int, int, int]:
    with open(chunk_map_bin, "rb") as f:
        magic = f.read(8)
        if magic != b"CHMAPV1\x00":
            raise ValueError(f"bad chunk_map magic: {magic!r}")
        sa_len = struct.unpack("<Q", f.read(8))[0]
        chunk_size = struct.unpack("<I", f.read(4))[0]
        num_chunks = struct.unpack("<I", f.read(4))[0]
    return sa_len, chunk_size, num_chunks


def load_raw_chunks(corpus_path: str, chunk_size: int = 16384, limit_chunks: Optional[int] = None) -> List[bytes]:
    chunks = []
    with open(corpus_path, "rb") as f:
        while True:
            if limit_chunks is not None and len(chunks) >= limit_chunks:
                break
            block = f.read(chunk_size)
            if not block:
                break
            chunks.append(block)
    return chunks


# ============================================================
# RESULTS HELPERS
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
    raise ValueError("unsupported results format")


def get_qid(row: Dict[str, Any]) -> int:
    for k in ("qid", "query_chunk", "q_chunk", "chunk_id"):
        if k in row:
            return int(row[k])
    raise KeyError("no qid/query_chunk field")


def get_starts(row: Dict[str, Any]) -> List[int]:
    if "starts" not in row:
        raise KeyError("no starts field")
    return [int(x) for x in row["starts"]]


def list_to_cids(vals: Any, limit: int) -> List[int]:
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
        if len(out) >= limit:
            break

    dedup = []
    seen = set()
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup[:limit]


def get_shortlist(row: Dict[str, Any], limit: int) -> List[int]:
    for key in ("shortlist", "ranked", "top", "topk", "pair_top", "fusion_top", "candidates"):
        if key in row:
            vals = list_to_cids(row[key], limit)
            if vals:
                return vals

    pair_keys = [k for k in row.keys() if k.startswith("pair") and isinstance(row[k], list)]
    if pair_keys:
        merged = []
        seen = set()
        for pk in sorted(pair_keys):
            for cid in list_to_cids(row[pk], limit):
                if cid not in seen:
                    seen.add(cid)
                    merged.append(cid)
                if len(merged) >= limit:
                    return merged
        return merged[:limit]

    return []


# ============================================================
# FEATURE HELPERS
# ============================================================
def build_query(chunk: bytes, starts: List[int], frag_len: int = 48) -> bytes:
    return b"".join(chunk[s:s + frag_len] for s in starts if s + frag_len <= len(chunk))


def rebuild_query_frags(chunk: bytes, starts: List[int], frag_len: int = 48) -> List[bytes]:
    return [chunk[s:s + frag_len] for s in starts if s + frag_len <= len(chunk)]


def exact_score(query_frags: List[bytes], cand: bytes) -> Tuple[int, int]:
    present = 0
    total = 0
    for frag in query_frags:
        cnt = cand.count(frag)
        if cnt > 0:
            present += 1
            total += cnt
    return present, total


def longest_exact_run(query: bytes, cand: bytes, min_run: int = 16) -> Tuple[int, int, int]:
    best = 0
    count_ge = 0
    sum_len = 0

    step = 8
    upper = max(1, len(query) - min_run + 1)
    for i in range(0, upper, step):
        if query[i:i + min_run] not in cand:
            continue

        count_ge += 1
        lo = min_run
        hi = len(query) - i
        best_here = min_run

        while lo <= hi:
            mid = (lo + hi) // 2
            if query[i:i + mid] in cand:
                best_here = mid
                lo = mid + 1
            else:
                hi = mid - 1

        best = max(best, best_here)
        sum_len += best_here

    return best, count_ge, sum_len


def phrase_span_score(query_chunk: bytes, starts: List[int], cand: bytes, span_len: int = 192) -> Tuple[int, int]:
    anchors = []
    if len(starts) >= 1:
        anchors.append(starts[0])
    if len(starts) >= 3:
        anchors.append(starts[2])

    full_hits = 0
    best_longest = 0

    for a in anchors:
        if a + span_len <= len(query_chunk):
            sp = query_chunk[a:a + span_len]
            if sp in cand:
                full_hits += 1
                best_longest = max(best_longest, len(sp))
            else:
                for L in (160, 128, 96, 64, 48, 32):
                    if L <= len(sp) and sp[:L] in cand:
                        best_longest = max(best_longest, L)
                        break

    return full_hits, best_longest


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-results", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--out-csv", required=True)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--span-len", type=int, default=192)
    args = ap.parse_args()

    raw = load_pickle(args.gap_results)
    rows = normalize_rows(raw)

    _sa_len, chunk_size, num_chunks = load_chunk_map_header(args.chunk_map)
    chunks = load_raw_chunks(args.corpus, chunk_size=chunk_size, limit_chunks=num_chunks)

    fieldnames = [
        "query_id",
        "candidate_id",
        "label",

        "shortlist_len",
        "gap_rank",
        "gap_rank_frac",
        "gap_in_topk",
        "is_gap_top1",
        "is_gap_top3",
        "is_gap_top5",
        "is_gap_top8",

        "exact_present",
        "exact_total",
        "exact_total_per_present",
        "exact_present_frac",
        "exact_total_norm",

        "run_longest",
        "run_count_ge16",
        "run_sum_len",
        "run_longest_norm",
        "run_count_ge16_norm",
        "run_sum_len_norm",
        "run_density",

        "span_full_hits",
        "span_best_longest",
        "span_full_hits_ratio",
        "span_best_longest_norm",

        "present_x_total",
        "run_sum_x_exact_total",
        "span_hits_x_exact_present",
        "run_sum_x_span_hits",

        "query_len",
        "candidate_len",
    ]

    n_rows = 0
    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames)
        wr.writeheader()

        for row in rows:
            qid = get_qid(row)
            starts = get_starts(row)
            shortlist = get_shortlist(row, args.top_k)
            if not shortlist:
                continue

            shortlist_len = len(shortlist)
            query_chunk = chunks[qid]
            query = build_query(query_chunk, starts, args.frag_len)
            query_frags = rebuild_query_frags(query_chunk, starts, args.frag_len)

            gap_rank_map = {cid: i + 1 for i, cid in enumerate(shortlist)}

            for cid in shortlist:
                cand = chunks[cid]

                exact_present, exact_total = exact_score(query_frags, cand)
                run_longest, run_count_ge16, run_sum_len = longest_exact_run(query, cand, min_run=16)
                span_full_hits, span_best_longest = phrase_span_score(query_chunk, starts, cand, args.span_len)

                gap_rank = gap_rank_map.get(cid, 999999)
                gap_rank_frac = safe_div(gap_rank, shortlist_len)

                exact_present_frac = safe_div(exact_present, len(query_frags))
                exact_total_per_present = safe_div(exact_total, exact_present)
                exact_total_norm = safe_div(exact_total, len(query))

                run_longest_norm = safe_div(run_longest, len(query))
                run_count_ge16_norm = safe_div(run_count_ge16, max(1, len(query) // 8))
                run_sum_len_norm = safe_div(run_sum_len, len(query))
                run_density = safe_div(run_sum_len, max(1, run_count_ge16))

                span_full_hits_ratio = safe_div(span_full_hits, 2.0)
                span_best_longest_norm = safe_div(span_best_longest, args.span_len)

                present_x_total = exact_present * exact_total
                run_sum_x_exact_total = run_sum_len * exact_total
                span_hits_x_exact_present = span_full_hits * exact_present
                run_sum_x_span_hits = run_sum_len * span_full_hits

                wr.writerow({
                    "query_id": qid,
                    "candidate_id": cid,
                    "label": 1 if cid == qid else 0,

                    "shortlist_len": shortlist_len,
                    "gap_rank": gap_rank,
                    "gap_rank_frac": gap_rank_frac,
                    "gap_in_topk": 1,
                    "is_gap_top1": 1 if gap_rank <= 1 else 0,
                    "is_gap_top3": 1 if gap_rank <= 3 else 0,
                    "is_gap_top5": 1 if gap_rank <= 5 else 0,
                    "is_gap_top8": 1 if gap_rank <= 8 else 0,

                    "exact_present": exact_present,
                    "exact_total": exact_total,
                    "exact_total_per_present": exact_total_per_present,
                    "exact_present_frac": exact_present_frac,
                    "exact_total_norm": exact_total_norm,

                    "run_longest": run_longest,
                    "run_count_ge16": run_count_ge16,
                    "run_sum_len": run_sum_len,
                    "run_longest_norm": run_longest_norm,
                    "run_count_ge16_norm": run_count_ge16_norm,
                    "run_sum_len_norm": run_sum_len_norm,
                    "run_density": run_density,

                    "span_full_hits": span_full_hits,
                    "span_best_longest": span_best_longest,
                    "span_full_hits_ratio": span_full_hits_ratio,
                    "span_best_longest_norm": span_best_longest_norm,

                    "present_x_total": present_x_total,
                    "run_sum_x_exact_total": run_sum_x_exact_total,
                    "span_hits_x_exact_present": span_hits_x_exact_present,
                    "run_sum_x_span_hits": run_sum_x_span_hits,

                    "query_len": len(query),
                    "candidate_len": len(cand),
                })
                n_rows += 1

    print("=" * 60)
    print(" LTR FUSION DATASET BUILDER V2")
    print("=" * 60)
    print(f"queries={len(rows)}")
    print(f"dataset_rows={n_rows}")
    print(f"saved={args.out_csv}")


if __name__ == "__main__":
    main()