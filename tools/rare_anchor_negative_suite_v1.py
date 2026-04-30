import argparse
import random
import struct
import subprocess
import time
from collections import Counter, defaultdict


def load_chunks(corpus_path: str, chunk_size: int = 16384):
    chunks = []
    with open(corpus_path, "rb") as f:
        while True:
            block = f.read(chunk_size)
            if not block:
                break
            chunks.append(block)
    return chunks


def load_corpus_bytes(corpus_path: str):
    with open(corpus_path, "rb") as f:
        return f.read()


def load_u32(path: str):
    with open(path, "rb") as f:
        data = f.read()
    if len(data) % 4 != 0:
        raise ValueError(f"{path} size is not multiple of 4")
    return struct.unpack("<" + "I" * (len(data) // 4), data)


def candidate_starts(query_len, frag_len, step, max_windows):
    starts = list(range(0, max(0, query_len - frag_len + 1), step))
    return starts[:max_windows]


def interval_to_chunks(chunk_map, l, r):
    if l >= r:
        return set()
    return set(chunk_map[l:r])


def mutate_one_byte(b: bytes):
    if not b:
        return b
    x = bytearray(b)
    mid = len(x) // 2
    x[mid] ^= 1
    return bytes(x)


def shannon_like_complexity(b: bytes):
    if not b:
        return 0.0
    c = Counter(b)
    n = len(b)
    return sum((v / n) ** 2 for v in c.values())  # higher = lower complexity


class FMServer:
    def __init__(self, server_bin, fm, bwt):
        self.proc = subprocess.Popen(
            [str(server_bin), str(fm), str(bwt)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        ready = self.proc.stderr.readline().strip()
        if ready != "READY":
            err = self.proc.stderr.read()
            raise RuntimeError(f"FM server failed to start: {ready}\n{err}")

    def query_many(self, hex_patterns):
        for pat in hex_patterns:
            self.proc.stdin.write(pat + "\n")
        self.proc.stdin.flush()

        out = []
        for _ in hex_patterns:
            line = self.proc.stdout.readline().strip()
            if not line:
                raise RuntimeError("empty line from FM server")
            l, r, cnt = map(int, line.split())
            out.append((l, r, cnt))
        return out

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self.proc.stdin.write("__EXIT__\n")
                self.proc.stdin.flush()
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except Exception:
                self.proc.kill()


def build_queries(chunks, corpus_bytes, frag_len, chunk_size, seed, count_from_corpus=5):
    random.seed(seed)
    queries = []

    # 1. too_short
    queries.append({
        "name": "too_short_ascii",
        "kind": "too_short",
        "query": b"short query",
        "expected": "too_short",
        "truth_qid": None,
    })
    queries.append({
        "name": "too_short_binary",
        "kind": "too_short",
        "query": bytes.fromhex("414243"),
        "expected": "too_short",
        "truth_qid": None,
    })

    # 2. low complexity / repeats
    queries.append({
        "name": "low_complexity_A",
        "kind": "low_complexity",
        "query": b"A" * max(frag_len * 2, 128),
        "expected": "non_selective_or_no_hit",
        "truth_qid": None,
    })
    queries.append({
        "name": "low_complexity_AT",
        "kind": "low_complexity",
        "query": (b"AT" * max(frag_len, 64))[: max(frag_len * 2, 128)],
        "expected": "non_selective_or_no_hit",
        "truth_qid": None,
    })
    queries.append({
        "name": "xmlish_repeat",
        "kind": "repeat",
        "query": (b"<title>" * 32)[: max(frag_len * 2, 128)],
        "expected": "multi_or_no_hit",
        "truth_qid": None,
    })

    # 3. definitely absent-ish
    queries.append({
        "name": "absent_XYZQ",
        "kind": "absent",
        "query": (b"XYZQ" * 40)[: max(frag_len * 2, 128)],
        "expected": "no_hit_or_multi",
        "truth_qid": None,
    })
    queries.append({
        "name": "absent_binary_like",
        "kind": "absent",
        "query": bytes([250, 251, 252, 253, 254, 255]) * 40,
        "expected": "no_hit",
        "truth_qid": None,
    })

    # 4. exact chunks from corpus
    valid_qids = [i for i, ch in enumerate(chunks) if len(ch) >= max(frag_len * 2, 128)]
    sampled = random.sample(valid_qids, min(count_from_corpus, len(valid_qids)))
    for qid in sampled:
        q = chunks[qid][: max(frag_len * 2, 128)]
        queries.append({
            "name": f"exact_chunk_qid_{qid}",
            "kind": "exact",
            "query": q,
            "expected": "unique_hit",
            "truth_qid": qid,
        })

        queries.append({
            "name": f"mutated_chunk_qid_{qid}",
            "kind": "mutated",
            "query": mutate_one_byte(q),
            "expected": "no_hit_or_multi",
            "truth_qid": qid,
        })

    # 5. cross-chunk query
    # Take end of one chunk + start of next chunk
    if len(chunks) >= 2:
        qid = sampled[0] if sampled else 0
        if qid + 1 < len(chunks):
            left = chunks[qid][-64:]
            right = chunks[qid + 1][:64]
            cross = left + right
            queries.append({
                "name": f"cross_chunk_qid_{qid}_{qid+1}",
                "kind": "cross_chunk",
                "query": cross,
                "expected": "maybe_unique_or_multi",
                "truth_qid": None,  # current chunk_map is chunk-based, boundary semantics to inspect
            })

    # 6. raw interior fragment from corpus (not full chunk)
    # choose a 256-byte exact substring from inside corpus
    if len(corpus_bytes) > 100000:
        start = 50000
        q = corpus_bytes[start:start + max(frag_len * 2, 128)]
        queries.append({
            "name": "exact_interior_substring",
            "kind": "exact_substring",
            "query": q,
            "expected": "unique_or_multi_hit",
            "truth_qid": None,
        })

    return queries


def classify_outcome(query_len, frag_len, shortlist_size, truth_hit, min_sa_hits, complexity_score,
                     non_selective_threshold):
    if query_len < frag_len:
        return "too_short"

    if shortlist_size == 0:
        return "no_hit"

    if shortlist_size == 1:
        return "unique_hit"

    if shortlist_size > non_selective_threshold:
        return "non_selective"

    return "multi_hit"


def run_one_query(server, chunk_map, query_bytes, frag_len, window_step, max_windows, pick_k,
                  non_selective_threshold):
    starts = candidate_starts(len(query_bytes), frag_len, window_step, max_windows)

    hex_patterns = []
    valid_starts = []
    for s in starts:
        frag = query_bytes[s:s + frag_len]
        if len(frag) != frag_len:
            continue
        hex_patterns.append(frag.hex())
        valid_starts.append(s)

    if not hex_patterns:
        return {
            "selected": [],
            "shortlist": [],
            "shortlist_size": 0,
            "min_sa_hits": None,
            "server_time_sec": 0.0,
            "complexity_score": shannon_like_complexity(query_bytes),
        }

    t1 = time.time()
    results = server.query_many(hex_patterns)
    t2 = time.time()

    windows = []
    for start, (l, r, cnt) in zip(valid_starts, results):
        chunks = interval_to_chunks(chunk_map, l, r)
        windows.append({
            "start": start,
            "sa_hits": cnt,
            "interval": (l, r),
            "chunk_count": len(chunks),
            "chunks": chunks,
        })

    windows.sort(key=lambda x: (x["sa_hits"], x["start"]))
    selected = windows[:pick_k]

    shortlist = set()
    for w in selected:
        shortlist |= w["chunks"]
    shortlist = sorted(shortlist)

    min_sa_hits = min((w["sa_hits"] for w in selected), default=None)

    return {
        "selected": selected,
        "shortlist": shortlist,
        "shortlist_size": len(shortlist),
        "min_sa_hits": min_sa_hits,
        "server_time_sec": t2 - t1,
        "complexity_score": shannon_like_complexity(query_bytes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--server-bin", required=True)

    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--window-step", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--pick-k", type=int, default=3)
    ap.add_argument("--chunk-size", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--show", type=int, default=50)
    ap.add_argument("--non-selective-threshold", type=int, default=16)

    args = ap.parse_args()

    chunks = load_chunks(args.corpus, args.chunk_size)
    chunk_map = load_u32(args.chunk_map)
    corpus_bytes = load_corpus_bytes(args.corpus)

    queries = build_queries(
        chunks=chunks,
        corpus_bytes=corpus_bytes,
        frag_len=args.frag_len,
        chunk_size=args.chunk_size,
        seed=args.seed,
        count_from_corpus=5,
    )

    server = FMServer(args.server_bin, args.fm, args.bwt)

    rows = []
    summary = Counter()

    try:
        for q in queries:
            out = run_one_query(
                server=server,
                chunk_map=chunk_map,
                query_bytes=q["query"],
                frag_len=args.frag_len,
                window_step=args.window_step,
                max_windows=args.max_windows,
                pick_k=args.pick_k,
                non_selective_threshold=args.non_selective_threshold,
            )

            truth_hit = False
            if q["truth_qid"] is not None:
                truth_hit = q["truth_qid"] in out["shortlist"][:16]

            outcome = classify_outcome(
                query_len=len(q["query"]),
                frag_len=args.frag_len,
                shortlist_size=out["shortlist_size"],
                truth_hit=truth_hit,
                min_sa_hits=out["min_sa_hits"],
                complexity_score=out["complexity_score"],
                non_selective_threshold=args.non_selective_threshold,
            )

            row = {
                "name": q["name"],
                "kind": q["kind"],
                "expected": q["expected"],
                "outcome": outcome,
                "query_len": len(q["query"]),
                "truth_qid": q["truth_qid"],
                "truth_hit": truth_hit,
                "shortlist_size": out["shortlist_size"],
                "shortlist_top16": out["shortlist"][:16],
                "min_sa_hits": out["min_sa_hits"],
                "server_time_sec": out["server_time_sec"],
                "complexity_score": out["complexity_score"],
                "selected": [
                    {
                        "start": w["start"],
                        "sa_hits": w["sa_hits"],
                        "chunk_count": w["chunk_count"],
                        "interval": w["interval"],
                    }
                    for w in out["selected"]
                ],
            }
            rows.append(row)
            summary[outcome] += 1

    finally:
        server.close()

    print("=" * 72)
    print(" RARE ANCHOR NEGATIVE SUITE V1")
    print("=" * 72)

    print("\nSUMMARY:")
    total = len(rows)
    print(f"  total_queries          = {total}")
    for k in ["unique_hit", "multi_hit", "non_selective", "no_hit", "too_short"]:
        print(f"  {k:22s}= {summary.get(k, 0)}")

    print("\nDETAILS:")
    for row in rows[:args.show]:
        print(
            f"  name={row['name']}"
            f" | kind={row['kind']}"
            f" | expected={row['expected']}"
            f" | outcome={row['outcome']}"
            f" | len={row['query_len']}"
            f" | shortlist_size={row['shortlist_size']}"
            f" | min_sa_hits={row['min_sa_hits']}"
            f" | truth_hit={row['truth_hit']}"
            f" | server_time={row['server_time_sec']:.6f}"
        )
        for s in row["selected"][:3]:
            print(
                f"    start={s['start']} "
                f"sa_hits={s['sa_hits']} "
                f"chunks={s['chunk_count']} "
                f"interval={s['interval']}"
            )

    print("\nNOTES:")
    print("  - unique_hit     = shortlist_size == 1")
    print("  - multi_hit      = 1 < shortlist_size <= non_selective_threshold")
    print("  - non_selective  = shortlist_size > non_selective_threshold")
    print("  - no_hit         = shortlist_size == 0")
    print("  - too_short      = query_len < frag_len")
    print("  - expected field is advisory, not enforced")


if __name__ == "__main__":
    main()