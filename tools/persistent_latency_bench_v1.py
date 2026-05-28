#!/usr/bin/env python3

import argparse
import statistics
import subprocess
import time
from pathlib import Path


def percentile(values, p):
    if not values:
        return 0.0

    values = sorted(values)
    idx = int(len(values) * p)

    if idx >= len(values):
        idx = len(values) - 1

    return values[idx]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--server", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)

    ap.add_argument(
        "--queries",
        required=True,
        help="text file with one hex query per line",
    )

    ap.add_argument("--warmup", type=int, default=10)

    args = ap.parse_args()

    query_lines = [
        x.strip()
        for x in Path(args.queries).read_text().splitlines()
        if x.strip()
    ]

    proc = subprocess.Popen(
        [
            args.server,
            args.fm,
            args.bwt,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    ready = proc.stdout.readline().strip()

    if ready != "READY":
        raise RuntimeError(f"unexpected startup line: {ready}")

    for i in range(min(args.warmup, len(query_lines))):
        q = query_lines[i]
        proc.stdin.write(q + "\n")
        proc.stdin.flush()
        proc.stdout.readline()

    latencies = []

    t_global0 = time.perf_counter()

    for q in query_lines:
        t0 = time.perf_counter()

        proc.stdin.write(q + "\n")
        proc.stdin.flush()

        line = proc.stdout.readline()

        dt_ms = (time.perf_counter() - t0) * 1000.0

        if not line.strip():
            raise RuntimeError("empty response")

        latencies.append(dt_ms)

    t_global1 = time.perf_counter()

    proc.kill()

    total_ms = (t_global1 - t_global0) * 1000.0

    qps = len(query_lines) / (total_ms / 1000.0)

    print("PERSISTENT_LATENCY_BENCH_V1")
    print()

    print("queries:", len(query_lines))
    print("warmup:", args.warmup)

    print()

    print("avg_ms:", round(statistics.mean(latencies), 6))
    print("p50_ms:", round(percentile(latencies, 0.50), 6))
    print("p95_ms:", round(percentile(latencies, 0.95), 6))
    print("p99_ms:", round(percentile(latencies, 0.99), 6))

    print()

    print("min_ms:", round(min(latencies), 6))
    print("max_ms:", round(max(latencies), 6))

    print()

    print("total_ms:", round(total_ms, 6))
    print("qps:", round(qps, 3))


if __name__ == "__main__":
    main()
