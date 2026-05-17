#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def percentile(values, p):
    if not values:
        return 0.0

    vals = sorted(values)

    k = (len(vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)

    if lo == hi:
        return vals[lo]

    frac = k - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def stats(values):
    return {
        "min": round(min(values), 6),
        "p50": round(percentile(values, 50), 6),
        "p95": round(percentile(values, 95), 6),
        "p99": round(percentile(values, 99), 6),
        "max": round(max(values), 6),
        "mean": round(statistics.mean(values), 6),
    }


def load_queries(path: Path | None, fallback_pattern: str):
    if path is None:
        return [fallback_pattern]

    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--pattern", default="6572726f72")
    ap.add_argument("--queries-file", default=None)
    ap.add_argument("--warm-runs", type=int, default=100)

    args = ap.parse_args()

    queries_file = None

    if args.queries_file:
        queries_file = Path(args.queries_file)

        if not queries_file.is_absolute():
            queries_file = ROOT / queries_file

    text_queries = load_queries(queries_file, "")
    queries = [
        q.encode("utf-8").hex()
        for q in text_queries
    ] if queries_file else [args.pattern]

    t0 = time.perf_counter()

    proc = subprocess.Popen(
        [
            str(ROOT / "build" / "query_fm_server_v1"),
            args.fm,
            args.bwt,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    ready = proc.stderr.readline().strip()
    startup_ms = (time.perf_counter() - t0) * 1000.0

    if ready != "READY":
        raise RuntimeError(f"backend not ready: {ready}")

    def query_once(hex_pattern: str):
        qs = time.perf_counter()

        proc.stdin.write(hex_pattern + "\n")
        proc.stdin.flush()

        line = proc.stdout.readline()

        qe = time.perf_counter()

        if not line:
            raise RuntimeError("empty backend response")

        return (qe - qs) * 1000.0, line.strip()

    cold = []
    sample_response = None

    for q in queries:
        dt, resp = query_once(q)
        cold.append(dt)

        if sample_response is None:
            sample_response = resp

    warm = []

    for _ in range(args.warm_runs):
        for q in queries:
            dt, _ = query_once(q)
            warm.append(dt)

    proc.kill()

    result = {
        "benchmark": "persistent_fm_v1",
        "mode": "persistent_cpp_backend",
        "startup_ms": round(startup_ms, 6),
        "queries_file": (
            str(queries_file) if queries_file else None
        ),
        "query_count": len(queries),
        "queries": text_queries if queries_file else [args.pattern],
        "cold_query_ms": stats(cold),
        "warm_runs": args.warm_runs,
        "warm_query_total": len(warm),
        "warm_ms": stats(warm),
        "sample_response": sample_response,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
