import argparse
import difflib
import multiprocessing as mp
import pickle
import struct
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple


G = {}


# ============================================================
# LOADERS
# ============================================================
def load_raw_chunks(corpus_path: str, chunk_size: int = 16384, limit_chunks: Optional[int] = None) -> List[bytes]:
    chunks: List[bytes] = []
    with open(corpus_path, "rb") as f:
        while True:
            if limit_chunks is not None and len(chunks) >= limit_chunks:
                break
            block = f.read(chunk_size)
            if not block:
                break
            chunks.append(block)
    return chunks


def load_chunk_map_header(chunk_map_bin: str) -> Tuple[int, int, int]:
    with open(chunk_map_bin, "rb") as f:
        magic = f.read(8)
        if magic != b"CHMAPV1\x00":
            raise ValueError(f"bad chunk_map magic: {magic!r}")
        sa_len = struct.unpack("<Q", f.read(8))[0]
        chunk_size = struct.unpack("<I", f.read(4))[0]
        num_chunks = struct.unpack("<I", f.read(4))[0]
    return sa_len, chunk_size, num_chunks


def load_gap_results(path: str) -> List[Dict[str, Any]]:
    with open(path, "rb") as f:
        obj = pickle.load(f)

    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        if "results" in obj and isinstance(obj["results"], list):
            rows = obj["results"]
        elif "queries" in obj and isinstance(obj["queries"], list):
            rows = obj["queries"]
        elif all(isinstance(v, dict) for v in obj.values()):
            rows = list(obj.values())
        else:
            raise ValueError("unsupported gap_results.pkl format")
    else:
        raise ValueError("unsupported gap_results.pkl format")

    if not rows:
        raise ValueError("gap_results.pkl contains no query rows")
    return rows


def get_qid(row: Dict[str, Any]) -> int:
    for k in ("qid", "query_chunk", "q_chunk", "chunk_id"):
        if k in row:
            return int(row[k])
    raise KeyError("row has no qid/query_chunk")


def get_starts(row: Dict[str, Any]) -> List[int]:
    starts = row.get("starts")
    if starts is None:
        raise KeyError("row has no starts")
    return [int(x) for x in starts]


def shortlist_from_any(vals: Any, top_k: int) -> List[int]:
    out: List[int] = []
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

    dedup: List[int] = []
    seen = set()
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup[:top_k]


def shortlist_from_row(row: Dict[str, Any], top_k: int) -> List[int]:
    for key in ("shortlist", "ranked", "top", "topk", "pair_top", "fusion_top", "candidates"):
        if key in row:
            return shortlist_from_any(row[key], top_k)

    pair_keys = [k for k in row.keys() if k.startswith("pair") and isinstance(row[k], list)]
    if pair_keys:
        seen = []
        used = set()
        for pk in sorted(pair_keys):
            for cid in shortlist_from_any(row[pk], top_k):
                if cid not in used:
                    used.add(cid)
                    seen.append(cid)
                if len(seen) >= top_k:
                    return seen
        return seen[:top_k]

    raise KeyError("row has no recognizable shortlist field")


# ============================================================
# OLD EXACT SIGNAL
# ============================================================
def rebuild_query_frags_from_starts(true_chunk: bytes, starts: Sequence[int], frag_len_guess: int) -> List[bytes]:
    frags: List[bytes] = []
    for s in starts:
        s = int(s)
        if s + frag_len_guess <= len(true_chunk):
            frags.append(true_chunk[s:s + frag_len_guess])
    return frags


def exact_score(query_frags: Sequence[bytes], candidate_chunk: bytes) -> Tuple[int, int]:
    present = 0
    total = 0
    for frag in query_frags:
        cnt = candidate_chunk.count(frag)
        if cnt > 0:
            present += 1
            total += cnt
    return present, total


# ============================================================
# WINDOW BUILD
# ============================================================
def build_query_windows(
    true_chunk: bytes,
    starts: Sequence[int],
    window_lengths: Sequence[int],
) -> List[Tuple[int, bytes]]:
    anchors: List[int] = []
    if starts:
        anchors.append(int(starts[0]))
    if len(starts) >= 2:
        anchors.append(int(starts[1]))
    if len(starts) >= 3:
        anchors.append(int(starts[2]))

    uniq_anchors = []
    seen = set()
    for a in anchors:
        if a not in seen:
            seen.add(a)
            uniq_anchors.append(a)

    windows: List[Tuple[int, bytes]] = []
    for a in uniq_anchors:
        for L in window_lengths:
            if a + L <= len(true_chunk):
                windows.append((a, true_chunk[a:a + L]))

    return windows


# ============================================================
# SEQUENCE MATCHER METRICS
# ============================================================
def merge_intervals(intervals: List[Tuple[int, int]]) -> int:
    if not intervals:
        return 0
    intervals.sort()
    total = 0
    cur_l, cur_r = intervals[0]
    for l, r in intervals[1:]:
        if l <= cur_r:
            cur_r = max(cur_r, r)
        else:
            total += cur_r - cur_l
            cur_l, cur_r = l, r
    total += cur_r - cur_l
    return total


def sequence_metrics(a: bytes, b: bytes, min_block: int) -> Dict[str, int]:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    blocks = sm.get_matching_blocks()  # last one is sentinel size=0

    real_blocks = [blk for blk in blocks if blk.size > 0]
    if not real_blocks:
        return {
            "largest_block": 0,
            "sum_top3_blocks": 0,
            "sum_blocks_ge_32": 0,
            "sum_blocks_ge_64": 0,
            "count_blocks_ge_32": 0,
            "count_blocks_ge_64": 0,
            "coverage_union_ge_32": 0,
        }

    sizes = sorted((blk.size for blk in real_blocks), reverse=True)
    top3 = sizes[:3]

    ge32 = [blk for blk in real_blocks if blk.size >= 32]
    ge64 = [blk for blk in real_blocks if blk.size >= 64]

    union_intervals = [(blk.a, blk.a + blk.size) for blk in ge32]
    coverage_union_ge_32 = merge_intervals(union_intervals)

    return {
        "largest_block": sizes[0],
        "sum_top3_blocks": sum(top3),
        "sum_blocks_ge_32": sum(blk.size for blk in ge32),
        "sum_blocks_ge_64": sum(blk.size for blk in ge64),
        "count_blocks_ge_32": len(ge32),
        "count_blocks_ge_64": len(ge64),
        "coverage_union_ge_32": coverage_union_ge_32,
    }


# ============================================================
# WORKER
# ============================================================
def init_worker(corpus_path: str, chunk_map_path: str):
    _sa_len, chunk_size, num_chunks = load_chunk_map_header(chunk_map_path)
    chunks = load_raw_chunks(corpus_path, chunk_size=chunk_size, limit_chunks=num_chunks)
    G["chunks"] = chunks


def score_candidate(
    cand_chunk: bytes,
    windows: List[Tuple[int, bytes]],
    query_frags: List[bytes],
) -> Tuple[int, int, int, int, int, int, int, int, int]:
    agg_largest = 0
    agg_sum_top3 = 0
    agg_sum_ge32 = 0
    agg_sum_ge64 = 0
    agg_cnt_ge32 = 0
    agg_cnt_ge64 = 0
    agg_cov32 = 0
    full_window_hits = 0

    for _anchor, w in windows:
        m = sequence_metrics(w, cand_chunk, min_block=32)
        agg_largest = max(agg_largest, m["largest_block"])
        agg_sum_top3 += m["sum_top3_blocks"]
        agg_sum_ge32 += m["sum_blocks_ge_32"]
        agg_sum_ge64 += m["sum_blocks_ge_64"]
        agg_cnt_ge32 += m["count_blocks_ge_32"]
        agg_cnt_ge64 += m["count_blocks_ge_64"]
        agg_cov32 += m["coverage_union_ge_32"]
        if w in cand_chunk:
            full_window_hits += 1

    present, total = exact_score(query_frags, cand_chunk)

    return (
        full_window_hits,
        agg_sum_ge64,
        agg_sum_top3,
        agg_largest,
        agg_cov32,
        agg_cnt_ge64,
        agg_cnt_ge32,
        present,
        total,
    )


def process_one(row: Dict[str, Any], frag_len: int, top_k: int, window_lengths: Sequence[int]) -> Dict[str, Any]:
    chunks = G["chunks"]

    qid = get_qid(row)
    starts = get_starts(row)
    shortlist = shortlist_from_row(row, top_k)

    if qid < 0 or qid >= len(chunks):
        return {"skip": True}

    true_chunk = chunks[qid]
    query_frags = rebuild_query_frags_from_starts(true_chunk, starts, frag_len)
    windows = build_query_windows(true_chunk, starts, window_lengths)

    if not shortlist or not windows:
        return {"skip": True}

    reranked = []
    for cid in shortlist:
        if cid < 0 or cid >= len(chunks):
            continue
        cand = chunks[cid]
        score = score_candidate(cand, windows, query_frags)
        reranked.append((cid, *score))

    # sort by new signal first, then old exact signals
    reranked.sort(key=lambda x: (
        -x[1],  # full_window_hits
        -x[2],  # sum_ge64
        -x[3],  # sum_top3
        -x[4],  # largest
        -x[5],  # coverage
        -x[6],  # cnt_ge64
        -x[7],  # cnt_ge32
        -x[8],  # present
        -x[9],  # total
        x[0],
    ))

    rank = None
    for i, item in enumerate(reranked):
        if item[0] == qid:
            rank = i + 1
            break

    return {
        "skip": False,
        "qid": qid,
        "starts": starts,
        "rank": rank,
        "num_windows": len(windows),
        "window_anchors": [a for a, _ in windows],
        "top5": reranked[:5],
    }


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-results", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--top-k", type=int, default=32)
    ap.add_argument("--window-lengths", default="256,384")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    rows = load_gap_results(args.gap_results)
    window_lengths = [int(x) for x in args.window_lengths.split(",") if x.strip()]

    print("=" * 60)
    print(" SEQUENCE MATCHER DIAGNOSTIC V1")
    print("=" * 60)
    print(f" rows={len(rows)}")
    print(f" top_k={args.top_k}")
    print(f" window_lengths={window_lengths}")
    print(f" workers={args.workers}")

    with mp.Pool(processes=args.workers, initializer=init_worker, initargs=(args.corpus, args.chunk_map)) as pool:
        results = pool.starmap(
            process_one,
            [(row, args.frag_len, args.top_k, window_lengths) for row in rows]
        )

    executed = 0
    hit1 = 0
    hit4 = 0
    hit8 = 0
    hit16 = 0
    hit32 = 0
    mrr = 0.0
    examples = []

    for res in results:
        if res["skip"]:
            continue

        executed += 1
        rank = res["rank"]

        if rank == 1:
            hit1 += 1
        if rank is not None and rank <= 4:
            hit4 += 1
        if rank is not None and rank <= 8:
            hit8 += 1
        if rank is not None and rank <= 16:
            hit16 += 1
        if rank is not None and rank <= 32:
            hit32 += 1
        if rank is not None:
            mrr += 1.0 / rank

        if len(examples) < args.show:
            examples.append(res)

    if executed == 0:
        raise ValueError("no executed rows")

    print("\nRESULTS:")
    print(f"  seqmatch_hit@1  = {hit1 / executed:.2%}")
    print(f"  seqmatch_hit@4  = {hit4 / executed:.2%}")
    print(f"  seqmatch_hit@8  = {hit8 / executed:.2%}")
    print(f"  seqmatch_hit@16 = {hit16 / executed:.2%}")
    print(f"  seqmatch_hit@32 = {hit32 / executed:.2%}")
    print(f"  seqmatch_MRR    = {mrr / executed:.4f}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  query_chunk={ex['qid']} starts={ex['starts']} rank={ex['rank']} num_windows={ex['num_windows']}")
        print(f"    window_anchors={ex['window_anchors']}")
        for row in ex["top5"]:
            (
                cid,
                full_window_hits,
                sum_ge64,
                sum_top3,
                largest,
                cov32,
                cnt_ge64,
                cnt_ge32,
                present,
                total,
            ) = row
            marker = " <== TRUE" if cid == ex["qid"] else ""
            print(
                f"    chunk={cid} "
                f"full_window_hits={full_window_hits} "
                f"sum_ge64={sum_ge64} "
                f"sum_top3={sum_top3} "
                f"largest={largest} "
                f"cov32={cov32} "
                f"cnt_ge64={cnt_ge64} "
                f"cnt_ge32={cnt_ge32} "
                f"present={present} "
                f"total={total}{marker}"
            )


if __name__ == "__main__":
    main()