#!/usr/bin/env python3
import argparse
import json
import random
import statistics
import time

from glyph_live_retrieve_v1 import FMServer, load_u32, retrieve_bytes


def percentile(values, p):
    if not values:
        return 0.0
    vals = sorted(values)
    k = (len(vals) - 1) * p
    f = int(k)
    c = min(f + 1, len(vals) - 1)
    if f == c:
        return vals[f]
    return vals[f] * (c - k) + vals[c] * (k - f)


def load_query_from_corpus(corpus, qid, chunk_size):
    with open(corpus, "rb") as f:
        f.seek(qid * chunk_size)
        return f.read(chunk_size)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--corpus", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--chunk-map", required=True)
    ap.add_argument("--server-bin", required=True)

    ap.add_argument("--limit-queries", type=int, default=100)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk-size", type=int, default=16384)

    ap.add_argument("--frag-len", type=int, default=48)
    ap.add_argument("--window-step", type=int, default=64)
    ap.add_argument("--max-windows", type=int, default=64)
    ap.add_argument("--pick-k", type=int, default=3)
    ap.add_argument("--entropy-min", type=float, default=2.0)
    ap.add_argument("--non-selective-threshold", type=int, default=16)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max-query-bytes", type=int, default=1048576)

    args = ap.parse_args()

    random.seed(args.seed)

    print("=" * 72)
    print(" GLYPH LIVE BENCHMARK V1")
    print("=" * 72)

    t_load0 = time.time()
    chunk_map = load_u32(args.chunk_map)
    server = FMServer(args.server_bin, args.fm, args.bwt)
    t_load1 = time.time()

    corpus_size = 0
    with open(args.corpus, "rb") as f:
        f.seek(0, 2)
        corpus_size = f.tell()

    num_chunks = corpus_size // args.chunk_size
    qids = random.sample(range(num_chunks - 1), min(args.limit_queries, num_chunks - 1))

    rows = []
    counts = {}

    try:
        for qid in qids:
            query = load_query_from_corpus(args.corpus, qid, args.chunk_size)
            out = retrieve_bytes(query, server, chunk_map, args)
            out["qid"] = qid
            rows.append(out)
            counts[out["outcome"]] = counts.get(out["outcome"], 0) + 1
    finally:
        server.close()

    total_times = [r["timings"]["total_time_sec"] for r in rows]
    server_times = [r["timings"]["server_time_sec"] for r in rows]

    summary = {
        "queries": len(rows),
        "startup_load_time_sec": t_load1 - t_load0,
        "outcomes": counts,
        "total_time": {
            "avg": sum(total_times) / len(total_times),
            "p50": statistics.median(total_times),
            "p95": percentile(total_times, 0.95),
            "p99": percentile(total_times, 0.99),
            "max": max(total_times),
        },
        "server_time": {
            "avg": sum(server_times) / len(server_times),
            "p50": statistics.median(server_times),
            "p95": percentile(server_times, 0.95),
            "p99": percentile(server_times, 0.99),
            "max": max(server_times),
        },
        "slowest": [
            {
                "qid": r["qid"],
                "outcome": r["outcome"],
                "total_time_sec": r["timings"]["total_time_sec"],
                "server_time_sec": r["timings"]["server_time_sec"],
                "shortlist_size": r["shortlist_size"],
            }
            for r in sorted(rows, key=lambda x: x["timings"]["total_time_sec"], reverse=True)[:10]
        ],
    }

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
