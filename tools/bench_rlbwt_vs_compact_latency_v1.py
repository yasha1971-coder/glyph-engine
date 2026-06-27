#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATASETS = [
    {
        "label": "pizza50",
        "source": ROOT / "examples/public-evidence-demo/work/pizza_english_50mb",
        "query": "Ten Days that Shook the World",
        "expected_l": 12587658,
        "expected_r": 12587659,
        "expected_count": 1,
        "expected_offset": 53,
        "corpus_bytes": 50_000_000,
    },
    {
        "label": "xz_cve",
        "source": ROOT / "examples/xz-cve-2024-3094-demo/work/phase1_corpus",
        "query": "CVE-2024-3094",
        "expected_l": 11030,
        "expected_r": 11068,
        "expected_count": 38,
        "expected_offset": 274,
        "corpus_bytes": 38_927,
    },
    {
        "label": "synthetic_logs50",
        "source": Path("/tmp/glyph_synthetic_logs_50mb_v1/index"),
        "query": "GLYPH_UNIQUE_EVENT_424242",
        "expected_l": 20912727,
        "expected_r": 20912728,
        "expected_count": 1,
        "expected_offset": 25000227,
        "corpus_bytes": 50_000_000,
    },
]


def run(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def time_cmd(cmd, repeats):
    vals = []
    last_stdout = ""
    for _ in range(repeats):
        t0 = time.perf_counter()
        p = run(cmd)
        t1 = time.perf_counter()
        vals.append(t1 - t0)
        last_stdout = p.stdout
    return {
        "avg": statistics.mean(vals),
        "min": min(vals),
        "max": max(vals),
        "stdout": last_stdout,
    }


def runtime_size(path: Path):
    total = 0
    files = []
    for p in sorted(path.iterdir()):
        if p.is_file():
            sz = p.stat().st_size
            total += sz
            files.append((p.name, sz))
    return total, files


def main():
    ap = argparse.ArgumentParser(description="Benchmark RLBWT full runtime vs Compact Runtime V1.")
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--rank-step", type=int, default=8192)
    ap.add_argument("--sample-step", type=int, default=128)
    ap.add_argument("--checkpoint-step", type=int, default=8192)
    args = ap.parse_args()

    out_root = Path(args.out_root).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    rows = []

    for ds in DATASETS:
        label = ds["label"]
        source = ds["source"]

        if not source.exists():
            raise SystemExit(f"missing source index for {label}: {source}")

        compact_dir = out_root / label / "compact"
        rlbwt_dir = out_root / label / "rlbwt_full"
        compact_dir.parent.mkdir(parents=True, exist_ok=True)

        print(f"\n=== {label}: export compact ===")
        subprocess.run([
            sys.executable,
            str(ROOT / "tools/export_compact_runtime_v1.py"),
            "--source-index-dir", str(source),
            "--out-dir", str(compact_dir),
            "--checkpoint-step", str(args.checkpoint_step),
            "--sample-step", str(args.sample_step),
            "--force",
        ], check=True)

        print(f"\n=== {label}: export rlbwt full ===")
        subprocess.run([
            sys.executable,
            str(ROOT / "tools/export_rlbwt_full_runtime_v1.py"),
            "--source-index-dir", str(source),
            "--out-dir", str(rlbwt_dir),
            "--rank-step", str(args.rank_step),
            "--sample-step", str(args.sample_step),
            "--force",
        ], check=True)

        query_hex = ds["query"].encode("utf-8").hex()

        compact_query_cmd = [
            str(ROOT / "build/query_fm_core_v1"),
            str(compact_dir / "fm_core.bin"),
            str(compact_dir / "bwt.bin"),
            query_hex,
        ]

        compact_locate_cmd = [
            sys.executable,
            str(ROOT / "tools/glyph_locate_offsets_v0.py"),
            "--index-dir", str(compact_dir),
            "--l", str(ds["expected_l"]),
            "--r", str(ds["expected_r"]),
            "--sample-step", str(args.sample_step),
        ]

        rlbwt_query_cmd = [
            sys.executable,
            str(ROOT / "tools/rlbwt_fm_query_v1.py"),
            "--rlbwt", str(rlbwt_dir / "bwt.rlbwt"),
            "--rank-index", str(rlbwt_dir / "bwt.rlbwt.rank"),
            "--query", ds["query"],
            "--expected-l", str(ds["expected_l"]),
            "--expected-r", str(ds["expected_r"]),
            "--expected-count", str(ds["expected_count"]),
        ]

        rlbwt_locate_cmd = [
            sys.executable,
            str(ROOT / "tools/rlbwt_locate_offsets_v1.py"),
            "--rlbwt", str(rlbwt_dir / "bwt.rlbwt"),
            "--rank-index", str(rlbwt_dir / "bwt.rlbwt.rank"),
            "--locate-core", str(rlbwt_dir / f"locate_core_s{args.sample_step}.bin"),
            "--l", str(ds["expected_l"]),
            "--r", str(ds["expected_r"]),
            "--expected-offset", str(ds["expected_offset"]),
        ]

        print(f"\n=== {label}: timing ===")

        compact_query = time_cmd(compact_query_cmd, args.repeats)
        compact_locate = time_cmd(compact_locate_cmd, args.repeats)
        rlbwt_query = time_cmd(rlbwt_query_cmd, args.repeats)
        rlbwt_locate = time_cmd(rlbwt_locate_cmd, args.repeats)

        compact_total, _ = runtime_size(compact_dir)
        rlbwt_total, _ = runtime_size(rlbwt_dir)

        row = {
            "label": label,
            "corpus_bytes": ds["corpus_bytes"],
            "compact_runtime_bytes": compact_total,
            "compact_ratio": compact_total / ds["corpus_bytes"],
            "rlbwt_runtime_bytes": rlbwt_total,
            "rlbwt_ratio": rlbwt_total / ds["corpus_bytes"],
            "compact_query_avg_sec": compact_query["avg"],
            "compact_locate_avg_sec": compact_locate["avg"],
            "compact_query_locate_avg_sec": compact_query["avg"] + compact_locate["avg"],
            "rlbwt_query_avg_sec": rlbwt_query["avg"],
            "rlbwt_locate_avg_sec": rlbwt_locate["avg"],
            "rlbwt_query_locate_avg_sec": rlbwt_query["avg"] + rlbwt_locate["avg"],
        }

        rows.append(row)

        print(json.dumps(row, indent=2))

    csv_path = out_root / "rlbwt_vs_compact_latency_v1.csv"
    md_path = out_root / "rlbwt_vs_compact_latency_v1.md"

    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with md_path.open("w", encoding="utf-8") as f:
        f.write("# RLBWT vs Compact Runtime Latency V1\n\n")
        f.write(f"repeats: {args.repeats}\n\n")
        f.write("| corpus | compact_ratio | rlbwt_ratio | compact_query_avg_sec | compact_locate_avg_sec | compact_total_avg_sec | rlbwt_query_avg_sec | rlbwt_locate_avg_sec | rlbwt_total_avg_sec |\n")
        f.write("|---|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['label']} "
                f"| {r['compact_ratio']:.3f}x "
                f"| {r['rlbwt_ratio']:.3f}x "
                f"| {r['compact_query_avg_sec']:.6f} "
                f"| {r['compact_locate_avg_sec']:.6f} "
                f"| {r['compact_query_locate_avg_sec']:.6f} "
                f"| {r['rlbwt_query_avg_sec']:.6f} "
                f"| {r['rlbwt_locate_avg_sec']:.6f} "
                f"| {r['rlbwt_query_locate_avg_sec']:.6f} |\n"
            )

    print("\n[latency] wrote:")
    print(csv_path)
    print(md_path)
    print(md_path.read_text())


if __name__ == "__main__":
    main()
