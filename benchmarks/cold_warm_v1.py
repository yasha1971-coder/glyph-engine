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


def run_query(index_dir: Path, pattern: str):
    t0 = time.perf_counter()
    r = subprocess.run(
        ["python3", str(ROOT / "tools" / "query_verified_v1.py"), str(index_dir), pattern],
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    dt = (time.perf_counter() - t0) * 1000.0
    if r.returncode != 0:
        raise RuntimeError(f"query failed\nstdout={r.stdout}\nstderr={r.stderr}")
    return dt, r.stdout


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--index-dir", default="examples/mini/out")
    ap.add_argument("--pattern", default="error")
    ap.add_argument("--warm-runs", type=int, default=25)
    ap.add_argument("--build-mini", action="store_true")
    args = ap.parse_args()

    index_dir = ROOT / args.index_dir

    if args.build_mini:
        subprocess.run(
            [str(ROOT / "examples" / "mini" / "build_mini.sh")],
            cwd=str(ROOT),
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    cold_ms, cold_out = run_query(index_dir, args.pattern)

    warm = []
    for _ in range(args.warm_runs):
        dt, _ = run_query(index_dir, args.pattern)
        warm.append(dt)

    result = {
        "mode": "verified_query_wrapper_subprocess",
        "note": "Measures end-to-end verified query wrapper cost: Python startup + manifest verification + query_fm_v1 subprocess. Not raw persistent FM latency.",
        "benchmark": "cold_warm_v1",
        "index_dir": str(index_dir),
        "pattern": args.pattern,
        "cold_ms": round(cold_ms, 6),
        "warm_runs": len(warm),
        "warm_ms": {
            "min": round(min(warm), 6),
            "p50": round(percentile(warm, 50), 6),
            "p95": round(percentile(warm, 95), 6),
            "p99": round(percentile(warm, 99), 6),
            "max": round(max(warm), 6),
            "mean": round(statistics.mean(warm), 6),
        },
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
