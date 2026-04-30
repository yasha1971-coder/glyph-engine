import argparse
import pickle
import struct
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


# ============================================================
# RAW CORPUS
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


# ============================================================
# PICKLE RESULTS LOADER
# Supports several shapes:
# - list[dict]
# - {"results": list[dict]}
# - {"queries": list[dict]}
# Each query dict should ideally contain:
#   qid / query_chunk
#   starts
#   shortlist / ranked / top / topk
# If shortlist rows are tuples like (cid, ...), first element is used.
# If shortlist is list[int], it's used directly.
# ============================================================
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
        else:
            # maybe dict[qid] -> row
            if all(isinstance(v, dict) for v in obj.values()):
                rows = list(obj.values())
            else:
                raise ValueError("unsupported gap_results.pkl format (dict without results/queries)")
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


def shortlist_from_row(row: Dict[str, Any], top_k: int) -> List[int]:
    # Try several fields in order
    for key in ("shortlist", "ranked", "top", "topk", "pair_top", "fusion_top", "candidates"):
        if key not in row:
            continue
        vals = row[key]
        return shortlist_from_any(vals, top_k)

    # Maybe pair lists only; merge them
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

    raise KeyError("row has no recognizable shortlist/ranked/top field")


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

    # dedup preserving order
    dedup: List[int] = []
    seen = set()
    for x in out:
        if x not in seen:
            seen.add(x)
            dedup.append(x)
    return dedup[:top_k]


# ============================================================
# SPAN SCORING
# ============================================================
def longest_common_prefix(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def longest_match_len(candidate: bytes, needle: bytes) -> int:
    """
    Fast-enough exact longest prefix match of 'needle' against any position in candidate.
    Strategy:
      - try exact full hit first
      - then try descending prefix lengths
    Since shortlist is tiny (<=32), this is acceptable for diagnostics.
    """
    if not needle or not candidate:
        return 0

    if needle in candidate:
        return len(needle)

    # descending checkpoints to avoid too much work
    # 192 -> 160 -> 128 -> 96 -> 64 -> 48 -> 32
    checkpoints = sorted(set([
        len(needle),
        min(len(needle), 160),
        min(len(needle), 128),
        min(len(needle), 96),
        min(len(needle), 64),
        min(len(needle), 48),
        min(len(needle), 32),
    ]), reverse=True)

    for L in checkpoints:
        if L <= 0:
            continue
        if needle[:L] in candidate:
            # binary refine upward between L and previous larger checkpoint if wanted
            # here we do local refinement linearly only in the remaining small tail
            best = L
            upper = len(needle)
            while best < upper:
                nxt = best + 1
                if needle[:nxt] in candidate:
                    best = nxt
                else:
                    break
            return best

    return 0


def build_query_spans(
    true_chunk: bytes,
    starts: Sequence[int],
    frag_len_guess: int,
    span_len: int,
    extra_offsets: Sequence[int],
) -> List[Tuple[int, bytes]]:
    """
    Make long spans anchored near the query geometry.
    Main anchor = starts[0].
    Additional anchors can be relative offsets from starts[0].
    """
    if not starts:
        return []

    anchors = [int(starts[0])]
    base = int(starts[0])

    for off in extra_offsets:
        pos = base + int(off)
        if 0 <= pos < len(true_chunk):
            anchors.append(pos)

    # also anchor at middle if possible
    if len(starts) >= 3:
        anchors.append(int(starts[2]))

    # dedup preserve order
    seen = set()
    uniq_anchors: List[int] = []
    for a in anchors:
        if a not in seen:
            seen.add(a)
            uniq_anchors.append(a)

    spans: List[Tuple[int, bytes]] = []
    for a in uniq_anchors:
        if a + span_len <= len(true_chunk):
            spans.append((a, true_chunk[a:a + span_len]))
        else:
            # allow clipped tail only if long enough
            tail = true_chunk[a:]
            if len(tail) >= max(32, frag_len_guess):
                spans.append((a, tail))

    return spans


def exact_score(query_frags: Sequence[bytes], candidate_chunk: bytes) -> Tuple[int, int]:
    present = 0
    total = 0
    for frag in query_frags:
        cnt = candidate_chunk.count(frag)
        if cnt > 0:
            present += 1
            total += cnt
    return present, total


def rebuild_query_frags_from_starts(true_chunk: bytes, starts: Sequence[int], frag_len_guess: int) -> List[bytes]:
    frags: List[bytes] = []
    for s in starts:
        s = int(s)
        if s + frag_len_guess <= len(true_chunk):
            frags.append(true_chunk[s:s + frag_len_guess])
    return frags


# ============================================================
# MAIN
# ============================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-results", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--chunk-map", required=False, default=None)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--span-len", type=int, default=192)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--show", type=int, default=5)
    ap.add_argument("--probe-offsets", default="64,128")
    args = ap.parse_args()

    rows = load_gap_results(args.gap_results)

    if args.chunk_map:
        _sa_len, chunk_size, num_chunks = load_chunk_map_header(args.chunk_map)
        chunks = load_raw_chunks(args.corpus, chunk_size=chunk_size, limit_chunks=num_chunks)
    else:
        chunks = load_raw_chunks(args.corpus, chunk_size=16384, limit_chunks=None)

    probe_offsets = []
    if args.probe_offsets.strip():
        probe_offsets = [int(x) for x in args.probe_offsets.split(",") if x.strip()]

    hit1 = 0
    hit4 = 0
    hit8 = 0
    hit16 = 0
    mrr = 0.0
    executed = 0
    examples: List[Dict[str, Any]] = []

    print("=" * 60)
    print(" PHRASE SPAN RERANK V1")
    print("=" * 60)
    print(f" span_len={args.span_len}")
    print(f" top_k={args.top_k}")
    print(f" rows={len(rows)}")
    print(f" probe_offsets={probe_offsets}")

    for row in rows:
        qid = get_qid(row)
        starts = get_starts(row)

        if qid < 0 or qid >= len(chunks):
            continue

        shortlist = shortlist_from_row(row, args.top_k)
        if not shortlist:
            continue

        true_chunk = chunks[qid]
        query_frags = rebuild_query_frags_from_starts(true_chunk, starts, args.frag_len)
        spans = build_query_spans(
            true_chunk=true_chunk,
            starts=starts,
            frag_len_guess=args.frag_len,
            span_len=args.span_len,
            extra_offsets=probe_offsets,
        )
        if not spans:
            continue

        reranked = []
        for cid in shortlist:
            if cid < 0 or cid >= len(chunks):
                continue
            cand = chunks[cid]

            # phrase signal
            span_hit_count = 0
            best_longest = 0
            exact_full_span_hits = 0

            for _anchor, sp in spans:
                lm = longest_match_len(cand, sp)
                if lm == len(sp):
                    exact_full_span_hits += 1
                if lm >= max(32, len(sp) // 2):
                    span_hit_count += 1
                if lm > best_longest:
                    best_longest = lm

            present, total = exact_score(query_frags, cand)

            reranked.append((
                cid,
                exact_full_span_hits,   # strongest new signal
                best_longest,           # next strongest
                span_hit_count,         # supporting signal
                present,                # old exact fragment signal
                total,                  # old frequency signal
            ))

        reranked.sort(key=lambda x: (-x[1], -x[2], -x[3], -x[4], -x[5], x[0]))

        rank = None
        for i, item in enumerate(reranked):
            if item[0] == qid:
                rank = i + 1
                break

        executed += 1

        if rank == 1:
            hit1 += 1
        if rank is not None and rank <= 4:
            hit4 += 1
        if rank is not None and rank <= 8:
            hit8 += 1
        if rank is not None and rank <= 16:
            hit16 += 1
        if rank is not None:
            mrr += 1.0 / rank

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "starts": starts,
                "rank": rank,
                "num_spans": len(spans),
                "span_anchors": [a for a, _ in spans],
                "top5": reranked[:5],
            })

    if executed == 0:
        raise ValueError("no executed rows")

    print("\nRESULTS:")
    print(f"  phrase_span_hit@1  = {hit1 / executed:.2%}")
    print(f"  phrase_span_hit@4  = {hit4 / executed:.2%}")
    print(f"  phrase_span_hit@8  = {hit8 / executed:.2%}")
    print(f"  phrase_span_hit@16 = {hit16 / executed:.2%}")
    print(f"  phrase_span_MRR    = {mrr / executed:.4f}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  query_chunk={ex['qid']} starts={ex['starts']} rank={ex['rank']} num_spans={ex['num_spans']}")
        print(f"    span_anchors={ex['span_anchors']}")
        for row in ex["top5"]:
            cid, full_hits, best_longest, span_hits, present, total = row
            marker = " <== TRUE" if cid == ex["qid"] else ""
            print(
                f"    chunk={cid} "
                f"full_span_hits={full_hits} "
                f"best_longest={best_longest} "
                f"span_hits={span_hits} "
                f"present={present} "
                f"total={total}{marker}"
            )


if __name__ == "__main__":
    main()