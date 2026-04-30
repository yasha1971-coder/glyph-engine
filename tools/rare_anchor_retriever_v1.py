import argparse
import pickle
import struct
import subprocess
import time
from pathlib import Path


# =========================
# IO
# =========================

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


# =========================
# FM QUERY
# =========================

def run_fm_query(query_bin, fm, bwt, pattern_hex):
    res = subprocess.run(
        [str(query_bin), str(fm), str(bwt), pattern_hex],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(res.stderr or res.stdout)

    interval = None
    count = None

    for line in res.stdout.splitlines():
        line = line.strip()

        if line.startswith("interval:"):
            # expected:
            # interval: [88036406, 88067605)
            payload = line.split("interval:", 1)[1].strip()
            payload = payload.lstrip("[").rstrip(")")
            payload = payload.rstrip("]")
            parts = payload.split(",", 1)
            if len(parts) != 2:
                raise ValueError(f"bad interval line: {line}")

            left = parts[0].strip()
            right = parts[1].strip()

            # remove any stray bracket/paren chars
            left = left.replace("[", "").replace(")", "").replace("]", "")
            right = right.replace("[", "").replace(")", "").replace("]", "")

            l = int(left)
            r = int(right)
            interval = (l, r)

        elif line.startswith("count:"):
            count = int(line.split(":", 1)[1].strip())

    if interval is None or count is None:
        raise ValueError(f"failed to parse FM output:\n{res.stdout}")

    return interval, count


# =========================
# HELPERS
# =========================

def extract_fragment(chunk_bytes: bytes, start: int, length: int):
    frag = chunk_bytes[start:start + length]
    if len(frag) != length:
        raise ValueError(f"fragment too short at start={start}, len={len(frag)}")
    return frag


def find_row_by_qid(rows, qid: int):
    for row in rows:
        if int(row["qid"]) == int(qid):
            return row
    raise ValueError(f"qid={qid} not found in gap_results")


def interval_to_chunk_candidates(chunk_map, interval):
    l, r = interval
    if l >= r:
        return set()
    return set(chunk_map[l:r])


# =========================
# MAIN RETRIEVER
# =========================

def retrieve(args):
    t0_total = time.time()

    chunks = load_chunks(args.corpus)
    chunk_map = load_u32(args.chunk_map)

    with open(args.gap_results, "rb") as f:
        rows = pickle.load(f)

    row = find_row_by_qid(rows, args.qid)
    chunk_id = int(row["qid"])
    starts = [int(x) for x in row["starts"]]

    query_chunk = chunks[chunk_id]

    selected = []
    fm_timings = []

    for start in starts:
        frag = extract_fragment(query_chunk, start, args.frag_len)
        hex_pat = frag.hex()

        t1 = time.time()
        interval, count = run_fm_query(args.query_bin, args.fm, args.bwt, hex_pat)
        t2 = time.time()

        fm_timings.append(t2 - t1)

        cand_chunks = interval_to_chunk_candidates(chunk_map, interval)

        selected.append({
            "start": start,
            "sa_hits": count,
            "interval": interval,
            "chunks": len(cand_chunks),
            "cand_chunks": cand_chunks,
        })

    # sort by rarity (fewest SA hits first, then start)
    selected.sort(key=lambda x: (x["sa_hits"], x["start"]))
    selected = selected[:args.pick_k]

    shortlist_set = set()
    for s in selected:
        shortlist_set |= s["cand_chunks"]

    shortlist = sorted(shortlist_set)

    total_time = time.time() - t0_total

    print("============================================")
    print(" RARE ANCHOR RETRIEVER V1")
    print("============================================")
    print(f"qid: {chunk_id}")
    print(f"pick_k: {args.pick_k}")
    print(f"frag_len: {args.frag_len}")

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
    if fm_timings:
        print(f"  avg_fm_call_sec: {sum(fm_timings) / len(fm_timings):.6f}")
        print(f"  fm_calls: {len(fm_timings)}")


# =========================
# ENTRY
# =========================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gap-results", required=True)
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--fm", required=True)
    parser.add_argument("--bwt", required=True)
    parser.add_argument("--chunk-map", required=True)
    parser.add_argument("--query-bin", required=True)

    parser.add_argument("--qid", type=int, required=True)
    parser.add_argument("--frag-len", type=int, default=48)
    parser.add_argument("--pick-k", type=int, default=3)

    args = parser.parse_args()
    retrieve(args)