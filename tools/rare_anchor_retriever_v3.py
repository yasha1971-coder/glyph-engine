import argparse
import pickle
import struct
import subprocess
import time


def load_chunks(corpus_path: str, chunk_size: int = 16384):
    chunks = []
    with open(corpus_path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            chunks.append(block)
    return chunks


def load_u32(path: str):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not multiple of 4")
    return struct.unpack("<" + "I" * (len(data) // 4), data)


def normalize_rows(obj):
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


def find_row_by_qid(rows, qid: int):
    for row in rows:
        if int(row["qid"]) == int(qid):
            return row
    raise ValueError(f"qid={qid} not found in gap_results")


def candidate_starts(chunk_len, frag_len, step, max_windows):
    starts = list(range(0, max(0, chunk_len - frag_len + 1), step))
    return starts[:max_windows]


def interval_to_chunk_candidates(chunk_map, interval):
    l, r = interval
    if l >= r:
        return set()
    return set(chunk_map[l:r])


def run_batch_queries(batch_bin, fm, bwt, hex_patterns):
    """
    Runs one persistent batch process for all patterns.
    Returns list of (interval, count)
    """
    proc = subprocess.Popen(
        [str(batch_bin), str(fm), str(bwt)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    payload = "\n".join(hex_patterns) + "\n"
    stdout, stderr = proc.communicate(payload)

    if proc.returncode != 0:
        raise RuntimeError(stderr or stdout)

    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if len(lines) != len(hex_patterns):
        raise ValueError(
            f"batch output length mismatch: got {len(lines)}, expected {len(hex_patterns)}"
        )

    out = []
    for line in lines:
        parts = line.split()
        if len(parts) != 3:
            raise ValueError(f"bad batch output line: {line}")
        l, r, cnt = map(int, parts)
        out.append(((l, r), cnt))
    return out


def retrieve(args):
    t0_total = time.time()

    chunks = load_chunks(args.corpus)
    chunk_map = load_u32(args.chunk_map)

    with open(args.gap_results, "rb") as f:
        rows = normalize_rows(pickle.load(f))

    row = find_row_by_qid(rows, args.qid)
    chunk_id = int(row["qid"])
    query_chunk = chunks[chunk_id]

    starts = candidate_starts(len(query_chunk), args.frag_len, args.window_step, args.max_windows)

    hex_patterns = []
    frag_meta = []

    for start in starts:
        frag = query_chunk[start:start + args.frag_len]
        if len(frag) != args.frag_len:
            continue
        hex_patterns.append(frag.hex())
        frag_meta.append(start)

    t1 = time.time()
    batch_results = run_batch_queries(args.batch_bin, args.fm, args.bwt, hex_patterns)
    t2 = time.time()

    all_windows = []
    for start, ((l, r), count) in zip(frag_meta, batch_results):
        cand_chunks = interval_to_chunk_candidates(chunk_map, (l, r))
        all_windows.append({
            "start": start,
            "sa_hits": count,
            "interval": (l, r),
            "chunks": len(cand_chunks),
            "cand_chunks": cand_chunks,
        })

    all_windows.sort(key=lambda x: (x["sa_hits"], x["start"]))
    selected = all_windows[:args.pick_k]

    shortlist_set = set()
    for s in selected:
        shortlist_set |= s["cand_chunks"]

    shortlist = sorted(shortlist_set)
    total_time = time.time() - t0_total
    batch_time = t2 - t1

    print("============================================")
    print(" RARE ANCHOR RETRIEVER V3")
    print("============================================")
    print(f"qid: {chunk_id}")
    print(f"frag_len: {args.frag_len}")
    print(f"window_step: {args.window_step}")
    print(f"max_windows: {args.max_windows}")
    print(f"pick_k: {args.pick_k}")

    print("\nSELECTED ANCHORS:")
    for s in selected:
        print(
            f"  start={s['start']} "
            f"sa_hits={s['sa_hits']} "
            f"chunks={s['chunks']} "
            f"interval={s['interval']}"
        )

    print("\nRESULTS:")
    print(f"  shortlist_size: {len(shortlist)}")
    print(f"  shortlist[:16]: {shortlist[:16]}")
    print(f"  truth_in_shortlist: {chunk_id in shortlist_set}")

    print("\nTIMINGS:")
    print(f"  total_time_sec: {total_time:.6f}")
    print(f"  batch_query_time_sec: {batch_time:.6f}")
    print(f"  fm_calls_batched: {len(hex_patterns)}")
    if hex_patterns:
        print(f"  avg_fm_call_equiv_sec: {batch_time / len(hex_patterns):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gap-results", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--fm", required=True)
    parser.add_argument("--bwt", required=True)
    parser.add_argument("--chunk-map", required=True)
    parser.add_argument("--batch-bin", required=True)

    parser.add_argument("--qid", type=int, required=True)
    parser.add_argument("--frag-len", type=int, default=48)
    parser.add_argument("--window-step", type=int, default=64)
    parser.add_argument("--max-windows", type=int, default=64)
    parser.add_argument("--pick-k", type=int, default=3)

    args = parser.parse_args()
    retrieve(args)