import argparse
import statistics
import subprocess
import time
from pathlib import Path


def percentile(values, p):
    if not values:
        return 0.0
    values = sorted(values)
    idx = int((len(values) - 1) * p)
    return values[idx]


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--server", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--warmup", type=int, default=0)

    args = ap.parse_args()

    query_lines = [
        x.strip()
        for x in Path(args.queries).read_text().splitlines()
        if x.strip()
    ]

    warmup_lines = query_lines[: args.warmup]
    timed_lines = query_lines

    payload = ""
    if warmup_lines:
        payload += "\n".join(warmup_lines) + "\n"
    payload += "\n".join(timed_lines) + "\n"

    t0 = time.perf_counter()

    proc = subprocess.run(
        [
            args.server,
            args.fm,
            args.bwt,
        ],
        input=payload,
        text=True,
        capture_output=True,
    )

    dt_ms = (time.perf_counter() - t0) * 1000.0

    stdout_lines = [
        x.strip()
        for x in proc.stdout.splitlines()
        if x.strip()
    ]

    # READY may appear on stderr on Linux builds.
    stderr_lines = [
        x.strip()
        for x in proc.stderr.splitlines()
        if x.strip()
    ]

    result_lines = [
        x for x in stdout_lines
        if x != "READY"
    ]

    timed_result_lines = result_lines[len(warmup_lines):]

    if proc.returncode != 0:
        raise RuntimeError(
            f"server failed: returncode={proc.returncode}, stderr={proc.stderr[:500]}"
        )

    if len(timed_result_lines) != len(timed_lines):
        raise RuntimeError(
            f"result count mismatch: expected={len(timed_lines)}, got={len(timed_result_lines)}"
        )

    avg_ms = dt_ms / max(1, len(timed_lines))

    # Batch mode cannot measure per-query steady latency precisely.
    # It measures process+load+batch total divided by query count.
    synthetic = [avg_ms for _ in timed_lines]

    qps = len(timed_lines) / (dt_ms / 1000.0)

    print("PERSISTENT_LATENCY_BENCH_V1")
    print()
    print("mode: batch-stdin")
    print("returncode:", proc.returncode)
    print("queries:", len(timed_lines))
    print("warmup:", args.warmup)
    print("stdout_lines:", len(stdout_lines))
    print("stderr_lines:", len(stderr_lines))
    print()
    print("avg_ms_including_startup_load:", round(avg_ms, 6))
    print("p50_ms_synthetic:", round(percentile(synthetic, 0.50), 6))
    print("p95_ms_synthetic:", round(percentile(synthetic, 0.95), 6))
    print("p99_ms_synthetic:", round(percentile(synthetic, 0.99), 6))
    print()
    print("total_ms:", round(dt_ms, 6))
    print("qps_including_startup_load:", round(qps, 3))
    print()
    print("first_results:")
    for line in timed_result_lines[:5]:
        print(line)
    print()
    print("stderr_head:")
    for line in stderr_lines[:5]:
        print(line)


if __name__ == "__main__":
    main()