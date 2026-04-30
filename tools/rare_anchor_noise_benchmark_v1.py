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


def candidate_starts(chunk_len, frag_len, step, max_windows):
    starts = list(range(0, max(0, chunk_len - frag_len + 1), step))
    return starts[:max_windows]


def interval_to_chunk_candidates(chunk_map, interval):
    l, r = interval
    if l >= r:
        return set()
    return set(chunk_map[l:r])


def run_batch_queries(batch_bin, fm, bwt, hex_patterns):
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
        raise ValueError(f"batch output mismatch: got {len(lines)}, expected {len(hex_patterns)}")

    out = []
    for line in lines:
        l, r, cnt = map(int, line.split())
        out.append(((l, r), cnt))
    return out


def mutate_one_byte(hex_str: str):
    b = bytearray.fromhex(hex_str)
    if not b:
        return hex_str
    mid = len(b) // 2
    b[mid] ^= 1
    return b.hex()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gap-results", required=True)
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--batch-bin", required=True)
    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--window-step", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--pick-k", type=int, default=3)
    ap.add_argument("--top-k", type=int, default=16)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--show", type=int, default=5)
    args = ap.parse_args()

    chunks = load_chunks(args.corpus)
    chunk_map = load_u32(args.chunk_map)

    with open(args.gap_results, "rb") as f:
        rows = normalize_rows(pickle.load(f))

    total = 0
    clean_hit = 0
    noisy_hit = 0
    clean_short = 0
    noisy_short = 0
    examples = []

    print("=" * 60)
    print(" RARE ANCHOR NOISE BENCHMARK V1")
    print("=" * 60)

    for row in rows[:args.limit]:
        qid = int(row["qid"])
        query_chunk = chunks[qid]

        starts = candidate_starts(len(query_chunk), args.frag_len, args.window_step, args.max_windows)

        hex_patterns = []
        for start in starts:
            frag = query_chunk[start:start + args.frag_len]
            if len(frag) == args.frag_len:
                hex_patterns.append(frag.hex())

        batch_results = run_batch_queries(args.batch_bin, args.fm, args.bwt, hex_patterns)

        windows = []
        for start, hex_pat, ((l, r), count) in zip(starts, hex_patterns, batch_results):
            cand_chunks = interval_to_chunk_candidates(chunk_map, (l, r))
            windows.append({
                "start": start,
                "hex": hex_pat,
                "sa_hits": count,
                "cand_chunks": cand_chunks,
            })

        windows.sort(key=lambda x: (x["sa_hits"], x["start"]))
        selected = windows[:args.pick_k]

        clean_set = set()
        for w in selected:
            clean_set |= w["cand_chunks"]

        noisy_hex = [mutate_one_byte(w["hex"]) for w in selected]
        noisy_batch = run_batch_queries(args.batch_bin, args.fm, args.bwt, noisy_hex)

        noisy_set = set()
        noisy_stats = []
        for w, ((l, r), count) in zip(selected, noisy_batch):
            cand_chunks = interval_to_chunk_candidates(chunk_map, (l, r))
            noisy_set |= cand_chunks
            noisy_stats.append({
                "start": w["start"],
                "sa_hits": count,
                "chunks": len(cand_chunks),
            })

        total += 1
        clean_hit += int(qid in sorted(clean_set)[:args.top_k])
        noisy_hit += int(qid in sorted(noisy_set)[:args.top_k])
        clean_short += len(clean_set)
        noisy_short += len(noisy_set)

        if len(examples) < args.show:
            examples.append({
                "qid": qid,
                "clean_shortlist_size": len(clean_set),
                "noisy_shortlist_size": len(noisy_set),
                "clean_hit": qid in sorted(clean_set)[:args.top_k],
                "noisy_hit": qid in sorted(noisy_set)[:args.top_k],
                "selected": [{"start": w["start"], "sa_hits": w["sa_hits"], "chunks": len(w["cand_chunks"])} for w in selected],
                "noisy": noisy_stats,
            })

    print("\nRESULTS:")
    print(f"  queries                  = {total}")
    print(f"  clean_shortlist_hit@{args.top_k}   = {clean_hit / total:.2%}")
    print(f"  noisy_shortlist_hit@{args.top_k}   = {noisy_hit / total:.2%}")
    print(f"  avg_clean_shortlist_size = {clean_short / total:.2f}")
    print(f"  avg_noisy_shortlist_size = {noisy_short / total:.2f}")

    print("\nEXAMPLES:")
    for ex in examples:
        print(
            f"  qid={ex['qid']} "
            f"clean_size={ex['clean_shortlist_size']} "
            f"noisy_size={ex['noisy_shortlist_size']} "
            f"clean_hit={ex['clean_hit']} "
            f"noisy_hit={ex['noisy_hit']}"
        )
        for s in ex["selected"]:
            print(f"    clean start={s['start']} sa_hits={s['sa_hits']} chunks={s['chunks']}")
        for s in ex["noisy"]:
            print(f"    noisy start={s['start']} sa_hits={s['sa_hits']} chunks={s['chunks']}")


if __name__ == "__main__":
    main()