#!/usr/bin/env python3

import argparse
import statistics
import subprocess
import time
from pathlib import Path


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return 0.0
    i = int((len(xs) - 1) * p)
    return xs[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--server", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--warmup", type=int, default=20)
    args = ap.parse_args()

    queries = [
        x.strip()
        for x in Path(args.queries).read_text().splitlines()
        if x.strip()
    ]

    proc = subprocess.Popen(
        [args.server, args.fm, args.bwt],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    # query_fm_server_v1 prints READY to stderr on Linux.
    ready = proc.stderr.readline().strip()
    if ready != "READY":
        raise RuntimeError(f"unexpected ready line: {ready!r}")

    for i in range(args.warmup):
        q = queries[i % len(queries)]
        proc.stdin.write(q + "\n")
        proc.stdin.flush()
        proc.stdout.readline()

    lat = []

    for q in queries:
        t0 = time.perf_counter()
        proc.stdin.write(q + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline().strip()
        dt = (time.perf_counter() - t0) * 1000.0

        if not line:
            raise RuntimeError("empty response")

        lat.append(dt)

    proc.kill()

    print("STEADY_STATE_LATENCY_BENCH_V1")
    print()
    print("queries:", len(queries))
    print("warmup:", args.warmup)
    print()
    print("avg_ms:", round(statistics.mean(lat), 6))
    print("p50_ms:", round(pct(lat, 0.50), 6))
    print("p95_ms:", round(pct(lat, 0.95), 6))
    print("p99_ms:", round(pct(lat, 0.99), 6))
    print("min_ms:", round(min(lat), 6))
    print("max_ms:", round(max(lat), 6))
    print("qps:", round(len(lat) / (sum(lat) / 1000.0), 3))


if __name__ == "__main__":
    main()
