#!/usr/bin/env python3
import argparse
import json
import statistics
import subprocess
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def percentile(values, p):
    vals = sorted(values)
    if not vals:
        return 0.0

    k = (len(vals) - 1) * (p / 100.0)
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)

    if lo == hi:
        return vals[lo]

    frac = k - lo
    return vals[lo] * (1 - frac) + vals[hi] * frac


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--pattern", default="6572726f72")
    ap.add_argument("--warm-runs", type=int, default=100)
    args = ap.parse_args()

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

    def query_once():
        qs = time.perf_counter()

        proc.stdin.write(args.pattern + "\n")
        proc.stdin.flush()

        line = proc.stdout.readline()

        qe = time.perf_counter()

        if not line:
            raise RuntimeError("empty backend response")

        return (qe - qs) * 1000.0, line.strip()

    cold_ms, cold_resp = query_once()

    warm = []
    for _ in range(args.warm_runs):
        dt, _ = query_once()
        warm.append(dt)

    proc.kill()

    result = {
        "benchmark": "persistent_fm_v1",
        "mode": "persistent_cpp_backend",
        "startup_ms": round(startup_ms, 6),
        "cold_query_ms": round(cold_ms, 6),
        "warm_runs": len(warm),
        "warm_ms": {
            "min": round(min(warm), 6),
            "p50": round(percentile(warm, 50), 6),
            "p95": round(percentile(warm, 95), 6),
            "p99": round(percentile(warm, 99), 6),
            "max": round(max(warm), 6),
            "mean": round(statistics.mean(warm), 6),
        },
        "sample_response": cold_resp,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
