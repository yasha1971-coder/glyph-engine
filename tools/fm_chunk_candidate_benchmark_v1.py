import argparse
import pickle
import struct
import subprocess
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


def load_u32(path):
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


def query_interval(query_bin, fm, bwt, frag: bytes):
    cmd = [query_bin, fm, bwt, frag.hex()]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        raise RuntimeError(res.stderr or res.stdout)

    l = r = cnt = None
    for line in res.stdout.splitlines():
        line = line.strip()
        if line.startswith("interval:"):
            part = line.split("[", 1)[1].split(")", 1)[0]
            left, right = part.split(",", 1)
            l = int(left.strip())
            r = int(right.strip())
        elif line.startswith("count:"):
            cnt = int(line.split(":", 1)[1].strip())

    if l is None or r is None or cnt is None:
        raise RuntimeError("failed to parse query_fm_v1 output")
    return l, r, cnt


def get_unique_chunk_candidates(chunk_map, l, r):
    if l >= r:
        return []
    return sorted(set(chunk_map[l:r]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-results", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--query-bin", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    with open(args.gap_results, "rb") as f:
        rows = normalize_rows(pickle.load(f))

    chunks = load_chunks(args.corpus)
    chunk_map = load_u32(args.chunk_map)

    total_frags = 0
    total_sa_hits = 0
    total_unique_chunks = 0
    truth_present = 0
    examples = []

    print("=" * 60)
    print(" FM CHUNK CANDIDATE BENCHMARK V1")
    print("=" * 60)

    for row in rows[:args.limit]:
        qid = int(row["qid"])
        starts = [int(x) for x in row["starts"]]

        row_stats = []
        for s in starts:
            frag = chunks[qid][s:s + args.frag_len]
            if len(frag) != args.frag_len:
                continue

            l, r, cnt = query_interval(args.query_bin, args.fm, args.bwt, frag)
            cand_chunks = get_unique_chunk_candidates(chunk_map, l, r)

            total_frags += 1
            total_sa_hits += cnt
            total_unique_chunks += len(cand_chunks)
            if qid in cand_chunks:
                truth_present += 1

            row_stats.append({
                "start": s,
                "sa_hits": cnt,
                "unique_chunks": len(cand_chunks),
                "truth_in_chunks": qid in cand_chunks,
            })

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "stats": row_stats,
            })

    if total_frags == 0:
        raise ValueError("no fragments processed")

    print("\nRESULTS:")
    print(f"  fragments                  = {total_frags}")
    print(f"  avg_sa_hits_per_frag       = {total_sa_hits / total_frags:.2f}")
    print(f"  avg_unique_chunks_per_frag = {total_unique_chunks / total_frags:.2f}")
    print(f"  truth_present_rate         = {truth_present / total_frags:.2%}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  qid={ex['qid']}")
        for st in ex["stats"]:
            print(
                f"    start={st['start']} "
                f"sa_hits={st['sa_hits']} "
                f"unique_chunks={st['unique_chunks']} "
                f"truth_in_chunks={st['truth_in_chunks']}"
            )


if __name__ == "__main__":
    main()