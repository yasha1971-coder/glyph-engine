import argparse
import pickle
import struct
import subprocess


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


def get_chunk_set(chunk_map, l, r):
    if l >= r:
        return set()
    return set(chunk_map[l:r])


def vote_k_of_n(chunk_sets, k):
    from collections import Counter
    ctr = Counter()
    for s in chunk_sets:
        for cid in s:
            ctr[cid] += 1
    return {cid for cid, c in ctr.items() if c >= k}


def candidate_starts(chunk_len, frag_len, step, max_windows):
    starts = list(range(0, max(0, chunk_len - frag_len + 1), step))
    return starts[:max_windows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-results", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--query-bin", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--window-step", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--pick-k", type=int, default=5)
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    with open(args.gap_results, "rb") as f:
        rows = normalize_rows(pickle.load(f))

    chunks = load_chunks(args.corpus)
    chunk_map = load_u32(args.chunk_map)

    total_queries = 0
    total_selected_avg_df = 0.0
    total_union = 0
    total_inter = 0
    total_vote2 = 0
    total_vote3 = 0
    total_vote4 = 0
    total_vote5 = 0

    truth_union = 0
    truth_inter = 0
    truth_vote2 = 0
    truth_vote3 = 0
    truth_vote4 = 0
    truth_vote5 = 0

    examples = []

    print("=" * 60)
    print(" ADAPTIVE RARE FRAGMENT BENCHMARK V1")
    print("=" * 60)
    print(f"frag_len={args.frag_len}")
    print(f"window_step={args.window_step}")
    print(f"max_windows={args.max_windows}")
    print(f"pick_k={args.pick_k}")

    for row in rows[:args.limit]:
        qid = int(row["qid"])
        chunk = chunks[qid]

        starts = candidate_starts(len(chunk), args.frag_len, args.window_step, args.max_windows)
        scored = []

        for s in starts:
            frag = chunk[s:s + args.frag_len]
            if len(frag) != args.frag_len:
                continue
            l, r, cnt = query_interval(args.query_bin, args.fm, args.bwt, frag)
            scored.append((cnt, s, l, r))

        if len(scored) < args.pick_k:
            continue

        scored.sort(key=lambda x: (x[0], x[1]))
        selected = scored[:args.pick_k]

        chunk_sets = []
        selected_stats = []
        for cnt, s, l, r in selected:
            cset = get_chunk_set(chunk_map, l, r)
            chunk_sets.append(cset)
            selected_stats.append({
                "start": s,
                "sa_hits": cnt,
                "chunks": len(cset),
            })

        union_set = set().union(*chunk_sets)
        inter_set = set(chunk_sets[0])
        for s in chunk_sets[1:]:
            inter_set &= s

        vote2_set = vote_k_of_n(chunk_sets, 2)
        vote3_set = vote_k_of_n(chunk_sets, 3)
        vote4_set = vote_k_of_n(chunk_sets, 4) if args.pick_k >= 4 else set()
        vote5_set = vote_k_of_n(chunk_sets, 5) if args.pick_k >= 5 else set()

        total_queries += 1
        total_selected_avg_df += sum(x["sa_hits"] for x in selected_stats) / len(selected_stats)

        total_union += len(union_set)
        total_inter += len(inter_set)
        total_vote2 += len(vote2_set)
        total_vote3 += len(vote3_set)
        total_vote4 += len(vote4_set)
        total_vote5 += len(vote5_set)

        truth_union += int(qid in union_set)
        truth_inter += int(qid in inter_set)
        truth_vote2 += int(qid in vote2_set)
        truth_vote3 += int(qid in vote3_set)
        truth_vote4 += int(qid in vote4_set)
        truth_vote5 += int(qid in vote5_set)

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "selected": selected_stats,
                "union": len(union_set),
                "intersection": len(inter_set),
                "vote2": len(vote2_set),
                "vote3": len(vote3_set),
                "vote4": len(vote4_set),
                "vote5": len(vote5_set),
                "truth_union": qid in union_set,
                "truth_intersection": qid in inter_set,
                "truth_vote2": qid in vote2_set,
                "truth_vote3": qid in vote3_set,
                "truth_vote4": qid in vote4_set,
                "truth_vote5": qid in vote5_set,
            })

    if total_queries == 0:
        raise ValueError("no usable queries")

    print("\nRESULTS:")
    print(f"  queries                     = {total_queries}")
    print(f"  avg_selected_sa_hits        = {total_selected_avg_df / total_queries:.2f}")

    print(f"  avg_union_chunks            = {total_union / total_queries:.2f}")
    print(f"  avg_intersection_chunks     = {total_inter / total_queries:.2f}")
    print(f"  avg_vote2_chunks            = {total_vote2 / total_queries:.2f}")
    print(f"  avg_vote3_chunks            = {total_vote3 / total_queries:.2f}")
    print(f"  avg_vote4_chunks            = {total_vote4 / total_queries:.2f}")
    print(f"  avg_vote5_chunks            = {total_vote5 / total_queries:.2f}")

    print(f"  truth_union_rate            = {truth_union / total_queries:.2%}")
    print(f"  truth_intersection_rate     = {truth_inter / total_queries:.2%}")
    print(f"  truth_vote2_rate            = {truth_vote2 / total_queries:.2%}")
    print(f"  truth_vote3_rate            = {truth_vote3 / total_queries:.2%}")
    print(f"  truth_vote4_rate            = {truth_vote4 / total_queries:.2%}")
    print(f"  truth_vote5_rate            = {truth_vote5 / total_queries:.2%}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  qid={ex['qid']}")
        for st in ex["selected"]:
            print(f"    start={st['start']} sa_hits={st['sa_hits']} chunks={st['chunks']}")
        print(
            f"    union={ex['union']} "
            f"intersection={ex['intersection']} "
            f"vote2={ex['vote2']} "
            f"vote3={ex['vote3']} "
            f"vote4={ex['vote4']} "
            f"vote5={ex['vote5']}"
        )
        print(
            f"    truth_union={ex['truth_union']} "
            f"truth_intersection={ex['truth_intersection']} "
            f"truth_vote2={ex['truth_vote2']} "
            f"truth_vote3={ex['truth_vote3']} "
            f"truth_vote4={ex['truth_vote4']} "
            f"truth_vote5={ex['truth_vote5']}"
        )


if __name__ == "__main__":
    main()