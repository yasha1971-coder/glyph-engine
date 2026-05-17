#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUERIES = ROOT / "tests" / "fixtures" / "benchmark_queries.txt"


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


def load_queries(path: Path | None, fallback_pattern: str):
    if path is None:
        return [fallback_pattern]

    return [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def run_query(index_dir: Path, pattern: str):
    t0 = time.perf_counter()

    r = subprocess.run(
        [
            "python3",
            str(ROOT / "tools" / "query_verified_v1.py"),
            str(index_dir),
            pattern,
        ],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    dt = (time.perf_counter() - t0) * 1000.0

    if r.returncode != 0:
        raise RuntimeError(
            f"query failed\nstdout={r.stdout}\nstderr={r.stderr}"
        )

    return dt, r.stdout


def stats(values):
    return {
        "min": round(min(values), 6),
        "p50": round(percentile(values, 50), 6),
        "p95": round(percentile(values, 95), 6),
        "p99": round(percentile(values, 99), 6),
        "max": round(max(values), 6),
        "mean": round(statistics.mean(values), 6),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--index-dir", default="examples/mini/out")
    ap.add_argument("--pattern", default="error")
    ap.add_argument("--queries-file", default=None)
    ap.add_argument("--warm-runs", type=int, default=25)
    ap.add_argument("--build-mini", action="store_true")

    args = ap.parse_args()

    index_dir = ROOT / args.index_dir

    queries_file = None

    if args.queries_file:
        queries_file = Path(args.queries_file)

        if not queries_file.is_absolute():
            queries_file = ROOT / queries_file

    queries = load_queries(queries_file, args.pattern)

    if args.build_mini:
        subprocess.run(
            [str(ROOT / "examples" / "mini" / "build_mini.sh")],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    cold = []

    for q in queries:
        dt, _ = run_query(index_dir, q)
        cold.append(dt)

    warm = []

    for _ in range(args.warm_runs):
        for q in queries:
            dt, _ = run_query(index_dir, q)
            warm.append(dt)

    result = {
        "mode": "verified_query_wrapper_subprocess",
        "note": (
            "Measures end-to-end verified query wrapper cost: "
            "Python startup + manifest verification + "
            "query_fm_v1 subprocess."
        ),
        "benchmark": "cold_warm_v1",
        "index_dir": str(index_dir),
        "queries_file": (
            str(queries_file) if queries_file else None
        ),
        "query_count": len(queries),
        "queries": queries,
        "cold_ms": stats(cold),
        "warm_runs": args.warm_runs,
        "warm_query_total": len(warm),
        "warm_ms": stats(warm),
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
