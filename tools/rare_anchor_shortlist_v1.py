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
    ap.add_argument("--pick-k", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    with open(args.gap_results, "rb") as f:
        rows = normalize_rows(pickle.load(f))

    chunks = load_chunks(args.corpus)
    chunk_map = load_u32(args.chunk_map)

    total = 0
    hit = 0
    avg_selected_df = 0.0
    avg_shortlist_size = 0.0
    examples = []

    print("=" * 60)
    print(" RARE ANCHOR SHORTLIST V1")
    print("=" * 60)
    print(f"frag_len={args.frag_len}")
    print(f"window_step={args.window_step}")
    print(f"max_windows={args.max_windows}")
    print(f"pick_k={args.pick_k}")
    print(f"top_k={args.top_k}")

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

        # practically, union and intersection are identical in current successful regime,
        # but keep union as main shortlist source for robustness
        shortlist_set = set().union(*chunk_sets)
        shortlist = sorted(shortlist_set)[:args.top_k]

        total += 1
        hit += int(qid in shortlist)
        avg_selected_df += sum(x["sa_hits"] for x in selected_stats) / len(selected_stats)
        avg_shortlist_size += len(shortlist_set)

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "selected": selected_stats,
                "shortlist_size": len(shortlist_set),
                "shortlist_hit": qid in shortlist,
                "shortlist": shortlist[:8],
            })

    if total == 0:
        raise ValueError("no usable queries")

    print("\nRESULTS:")
    print(f"  queries               = {total}")
    print(f"  shortlist_hit@{args.top_k}      = {hit / total:.2%}")
    print(f"  avg_selected_sa_hits  = {avg_selected_df / total:.2f}")
    print(f"  avg_shortlist_size    = {avg_shortlist_size / total:.2f}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(f"  qid={ex['qid']} shortlist_size={ex['shortlist_size']} shortlist_hit={ex['shortlist_hit']}")
        for st in ex["selected"]:
            print(f"    start={st['start']} sa_hits={st['sa_hits']} chunks={st['chunks']}")
        print(f"    shortlist[:8]={ex['shortlist']}")


if __name__ == "__main__":
    main()