#!/usr/bin/env python3

import argparse
import re
import subprocess
import statistics


FIELDS = [
    "avg_ms",
    "p50_ms",
    "p95_ms",
    "p99_ms",
    "min_ms",
    "max_ms",
    "qps",
]


def parse_output(text):
    out = {}

    for field in FIELDS:
        m = re.search(rf"^{field}:\s+([0-9.]+)", text, re.MULTILINE)
        if not m:
            raise RuntimeError(f"missing field: {field}\n{text}")
        out[field] = float(m.group(1))

    return out


def verdict(ratio):
    if ratio < 1.5:
        return "LAW_CONFIRMED"
    if ratio < 2.0:
        return "LIKELY_STABLE"
    return "UNSTABLE_INVESTIGATE"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--bench", required=True)
    ap.add_argument("--server", required=True)
    ap.add_argument("--fm", required=True)
    ap.add_argument("--bwt", required=True)
    ap.add_argument("--queries", required=True)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--runs", type=int, default=10)

    args = ap.parse_args()

    rows = []

    for i in range(args.runs):
        cmd = [
            "python3",
            args.bench,
            "--server",
            args.server,
            "--fm",
            args.fm,
            "--bwt",
            args.bwt,
            "--queries",
            args.queries,
            "--warmup",
            str(args.warmup),
        ]

        p = subprocess.run(
            cmd,
            text=True,
            capture_output=True,
        )

        if p.returncode != 0:
            raise RuntimeError(
                f"run {i+1} failed\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}"
            )

        row = parse_output(p.stdout)
        rows.append(row)

        print(
            f"run={i+1}",
            f"p50={row['p50_ms']:.6f}",
            f"p95={row['p95_ms']:.6f}",
            f"p99={row['p99_ms']:.6f}",
            f"max={row['max_ms']:.6f}",
            f"qps={row['qps']:.3f}",
        )

    print()
    print("REPEATABILITY_SUMMARY_V1")
    print()

    for field in ["avg_ms", "p50_ms", "p95_ms", "p99_ms", "max_ms", "qps"]:
        values = [x[field] for x in rows]
        mn = min(values)
        mx = max(values)
        mean = statistics.mean(values)
        ratio = mx / mn if mn else 0.0

        print(
            field,
            "min=", round(mn, 6),
            "max=", round(mx, 6),
            "mean=", round(mean, 6),
            "ratio=", round(ratio, 3),
        )

    p95_values = [x["p95_ms"] for x in rows]
    p95_ratio = max(p95_values) / min(p95_values)

    print()
    print("verdict_basis: p95_ms max/min")
    print("ratio:", round(p95_ratio, 3))
    print("verdict:", verdict(p95_ratio))


if __name__ == "__main__":
    main()
