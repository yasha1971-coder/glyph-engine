#!/usr/bin/env python3
import argparse
import csv
import json
import statistics
import subprocess
import time
from pathlib import Path


def parse_int_list(s: str):
    return [int(x.strip()) for x in s.split(",") if x.strip()]


def run(cmd):
    return subprocess.run(cmd, check=True, text=True, capture_output=True)


def main():
    ap = argparse.ArgumentParser(description="Benchmark GLYPH Compact Runtime Profile V1 matrix.")
    ap.add_argument("--source-index-dir", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--expected-count", type=int, required=True)
    ap.add_argument("--expected-l", type=int, required=True)
    ap.add_argument("--expected-r", type=int, required=True)
    ap.add_argument("--expected-offset", type=int, action="append", default=[])
    ap.add_argument("--corpus-bytes", type=int, required=True)
    ap.add_argument("--checkpoint-steps", default="1024,2048,4096,8192")
    ap.add_argument("--sample-steps", default="16,32,64,128")
    ap.add_argument("--repeats", type=int, default=7)
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = Path(args.source_index_dir)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    checkpoint_steps = parse_int_list(args.checkpoint_steps)
    sample_steps = parse_int_list(args.sample_steps)
    query_hex = args.query.encode("utf-8").hex()

    rows = []

    for cstep in checkpoint_steps:
        for sstep in sample_steps:
            out_dir = out_root / f"c{cstep}_s{sstep}"
            print(f"\n=== checkpoint_step={cstep} sample_step={sstep} ===")

            subprocess.run([
                "python3", str(root / "tools/export_compact_runtime_v1.py"),
                "--source-index-dir", str(src),
                "--out-dir", str(out_dir),
                "--checkpoint-step", str(cstep),
                "--sample-step", str(sstep),
                "--force",
            ], check=True)

            verify_cmd = [
                "python3", str(root / "tools/verify_compact_runtime_v1.py"),
                "--runtime-dir", str(out_dir),
                "--sample-step", str(sstep),
                "--query", args.query,
                "--expected-count", str(args.expected_count),
                "--expected-l", str(args.expected_l),
                "--expected-r", str(args.expected_r),
            ]

            for off in args.expected_offset:
                verify_cmd += ["--expected-offset", str(off)]

            verify = run(verify_cmd)
            if "[compact-runtime] VERIFY OK" not in verify.stdout:
                raise SystemExit(f"verify failed for checkpoint_step={cstep}, sample_step={sstep}")

            files = {p.name: p.stat().st_size for p in out_dir.iterdir() if p.is_file()}
            loc_name = f"locate_core_s{sstep}.bin"
            runtime_total = sum(files.values())

            query_cmd = [
                str(root / "build/query_fm_core_v1"),
                str(out_dir / "fm_core.bin"),
                str(out_dir / "bwt.bin"),
                query_hex,
            ]

            locate_cmd = [
                "python3", str(root / "tools/glyph_locate_offsets_v0.py"),
                "--index-dir", str(out_dir),
                "--l", str(args.expected_l),
                "--r", str(args.expected_r),
                "--sample-step", str(sstep),
            ]

            run(query_cmd)
            run(locate_cmd)

            query_times = []
            locate_times = []
            locate_stdout_last = ""

            for _ in range(args.repeats):
                t0 = time.perf_counter()
                q = run(query_cmd)
                t1 = time.perf_counter()
                if f"match_count: {args.expected_count}" not in q.stdout:
                    raise SystemExit("bad query result")
                query_times.append(t1 - t0)

            for _ in range(args.repeats):
                t0 = time.perf_counter()
                loc = run(locate_cmd)
                t1 = time.perf_counter()
                locate_stdout_last = loc.stdout
                locate_json = json.loads(loc.stdout)
                if locate_json.get("ok") is not True:
                    raise SystemExit("bad locate result")
                for off in args.expected_offset:
                    if off not in locate_json.get("offsets", []):
                        raise SystemExit(f"expected offset {off} missing")
                locate_times.append(t1 - t0)

            locate_json = json.loads(locate_stdout_last)

            row = {
                "checkpoint_step": cstep,
                "sample_step": sstep,
                "bwt_bytes": files.get("bwt.bin", 0),
                "fm_core_bytes": files.get("fm_core.bin", 0),
                "locate_core_bytes": files.get(loc_name, 0),
                "manifest_bytes": files.get("manifest.json", 0),
                "compact_manifest_bytes": files.get("compact_runtime_manifest_v1.json", 0),
                "runtime_total_bytes": runtime_total,
                "ratio_vs_corpus": runtime_total / args.corpus_bytes,
                "query_cli_avg_sec": statistics.mean(query_times),
                "query_cli_min_sec": min(query_times),
                "locate_cli_avg_sec": statistics.mean(locate_times),
                "locate_cli_min_sec": min(locate_times),
                "locate_total_steps": locate_json.get("total_steps"),
                "locate_max_steps": locate_json.get("max_steps"),
                "verify_ok": True,
            }
            rows.append(row)

            print(f"runtime_total_bytes: {runtime_total}")
            print(f"ratio_vs_corpus: {row['ratio_vs_corpus']:.3f}x")
            print(f"query_cli_avg_sec: {row['query_cli_avg_sec']:.6f}")
            print(f"locate_cli_avg_sec: {row['locate_cli_avg_sec']:.6f}")
            print(f"locate_total_steps: {row['locate_total_steps']}")

    csv_path = out_root / "compact_runtime_matrix_v1.csv"
    md_path = out_root / "compact_runtime_matrix_v1.md"

    with csv_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    with md_path.open("w") as f:
        f.write("# Compact Runtime Matrix V1\n\n")
        f.write("| checkpoint_step | sample_step | runtime_total_bytes | ratio_vs_corpus | fm_core_bytes | locate_core_bytes | query_cli_avg_sec | locate_cli_avg_sec | locate_total_steps | locate_max_steps |\n")
        f.write("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                f"| {r['checkpoint_step']} "
                f"| {r['sample_step']} "
                f"| {r['runtime_total_bytes']} "
                f"| {r['ratio_vs_corpus']:.3f}x "
                f"| {r['fm_core_bytes']} "
                f"| {r['locate_core_bytes']} "
                f"| {r['query_cli_avg_sec']:.6f} "
                f"| {r['locate_cli_avg_sec']:.6f} "
                f"| {r['locate_total_steps']} "
                f"| {r['locate_max_steps']} |\n"
            )

    print("\n[compact-runtime-matrix] wrote:")
    print(csv_path)
    print(md_path)
    print()
    print(md_path.read_text())


if __name__ == "__main__":
    main()
