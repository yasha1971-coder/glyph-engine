import argparse
import csv
import pickle
import struct
from collections import defaultdict
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
# GAP RESULTS HELPERS
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
    seen = set()
    dedup = []
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


def get_rank_in_list(cids: List[int], qid: int) -> int:
    for i, cid in enumerate(cids):
        if cid == qid:
            return i + 1
    return 999999


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
    """
    Cheap exact-run proxy:
      - longest exact substring hit among query substrings
      - number of query offsets whose >=min_run prefix appears in candidate
      - summed matched prefix lengths over sampled offsets
    """
    best = 0
    count_ge = 0
    sum_len = 0

    step = 8
    for i in range(0, max(1, len(query) - min_run + 1), step):
        max_len_here = 0

        lo = min_run
        hi = len(query) - i
        found_min = False

        if query[i:i + min_run] in cand:
            found_min = True
            count_ge += 1
        else:
            continue

        # binary-ish growth
        low = min_run
        high = hi
        while low <= high:
            mid = (low + high) // 2
            if query[i:i + mid] in cand:
                max_len_here = mid
                low = mid + 1
            else:
                high = mid - 1

        if max_len_here > best:
            best = max_len_here
        sum_len += max_len_here

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
                # cheap descending fallback
                for L in (160, 128, 96, 64, 48, 32):
                    if L <= len(sp) and sp[:L] in cand:
                        best_longest = max(best_longest, L)
                        break

    return full_hits, best_longest


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
        "gap_rank",
        "gap_in_topk",
        "exact_present",
        "exact_total",
        "exact_total_per_present",
        "run_longest",
        "run_count_ge16",
        "run_sum_len",
        "span_full_hits",
        "span_best_longest",
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

            query_chunk = chunks[qid]
            query = build_query(query_chunk, starts, args.frag_len)
            query_frags = rebuild_query_frags(query_chunk, starts, args.frag_len)

            gap_rank_map = {cid: i + 1 for i, cid in enumerate(shortlist)}

            for cid in shortlist:
                cand = chunks[cid]

                exact_present, exact_total = exact_score(query_frags, cand)
                run_longest, run_count_ge16, run_sum_len = longest_exact_run(query, cand, min_run=16)
                span_full_hits, span_best_longest = phrase_span_score(query_chunk, starts, cand, args.span_len)

                wr.writerow({
                    "query_id": qid,
                    "candidate_id": cid,
                    "label": 1 if cid == qid else 0,
                    "gap_rank": gap_rank_map.get(cid, 999999),
                    "gap_in_topk": 1,
                    "exact_present": exact_present,
                    "exact_total": exact_total,
                    "exact_total_per_present": (exact_total / exact_present) if exact_present > 0 else 0.0,
                    "run_longest": run_longest,
                    "run_count_ge16": run_count_ge16,
                    "run_sum_len": run_sum_len,
                    "span_full_hits": span_full_hits,
                    "span_best_longest": span_best_longest,
                    "query_len": len(query),
                    "candidate_len": len(cand),
                })
                n_rows += 1

    print("=" * 60)
    print(" LTR FUSION DATASET BUILDER V1")
    print("=" * 60)
    print(f" queries={len(rows)}")
    print(f" dataset_rows={n_rows}")
    print(f" saved={args.out_csv}")


if __name__ == "__main__":
    main()