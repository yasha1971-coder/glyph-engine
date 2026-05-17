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


def load_queries(path: Path):
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def start_backend(fm: Path, bwt: Path):
    t0 = time.perf_counter()

    proc = subprocess.Popen(
        [
            str(ROOT / "build" / "query_fm_server_v1"),
            str(fm),
            str(bwt),
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

    return proc, startup_ms


def query_backend(proc, hex_pattern: str):
    proc.stdin.write(hex_pattern + "\n")
    proc.stdin.flush()

    line = proc.stdout.readline()
    if not line:
        raise RuntimeError("empty backend response")

    parts = line.strip().split()
    if len(parts) != 3:
        raise RuntimeError(f"bad backend response: {line!r}")

    return {
        "l": int(parts[0]),
        "r": int(parts[1]),
        "count": int(parts[2]),
        "raw": line.strip(),
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--queries-file", required=True)
    ap.add_argument("--warm-runs", type=int, default=10)
    ap.add_argument(
        "--shard",
        action="append",
        required=True,
        help="Shard spec: name,fm_path,bwt_path",
    )

    args = ap.parse_args()

    queries_path = Path(args.queries_file)
    if not queries_path.is_absolute():
        queries_path = ROOT / queries_path

    text_queries = load_queries(queries_path)
    hex_queries = [q.encode("utf-8").hex() for q in text_queries]

    shards = []
    for spec in args.shard:
        parts = spec.split(",")
        if len(parts) != 3:
            raise RuntimeError(f"bad shard spec: {spec!r}")

        name, fm, bwt = parts
        shards.append({
            "name": name,
            "fm": ROOT / fm if not Path(fm).is_absolute() else Path(fm),
            "bwt": ROOT / bwt if not Path(bwt).is_absolute() else Path(bwt),
        })

    procs = []
    startup = []

    try:
        for shard in shards:
            proc, startup_ms = start_backend(shard["fm"], shard["bwt"])
            procs.append((shard, proc))
            startup.append(startup_ms)

        def fanout_once(hex_pattern: str):
            t0 = time.perf_counter()

            responses = []
            total_count = 0

            for shard, proc in procs:
                resp = query_backend(proc, hex_pattern)
                responses.append({
                    "shard": shard["name"],
                    "count": resp["count"],
                    "raw": resp["raw"],
                })
                total_count += resp["count"]

            dt = (time.perf_counter() - t0) * 1000.0
            return dt, total_count, responses

        cold = []
        sample = None

        for q in hex_queries:
            dt, total_count, responses = fanout_once(q)
            cold.append(dt)

            if sample is None:
                sample = {
                    "query": text_queries[0],
                    "total_count": total_count,
                    "responses": responses,
                }

        warm = []
        for _ in range(args.warm_runs):
            for q in hex_queries:
                dt, _total_count, _responses = fanout_once(q)
                warm.append(dt)

        result = {
            "benchmark": "segmented_fanout_v1",
            "mode": "sequential_fanout_over_persistent_cpp_backends",
            "shard_count": len(shards),
            "shards": [s["name"] for s in shards],
            "startup_ms": stats(startup),
            "queries_file": str(queries_path),
            "query_count": len(text_queries),
            "warm_runs": args.warm_runs,
            "warm_query_total": len(warm),
            "cold_ms": stats(cold),
            "warm_ms": stats(warm),
            "sample": sample,
        }

        print(json.dumps(result, indent=2))

    finally:
        for _shard, proc in procs:
            proc.kill()


if __name__ == "__main__":
    main()
