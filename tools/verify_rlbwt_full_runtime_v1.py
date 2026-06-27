#!/usr/bin/env python3
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN = {
    "bwt.bin",
    "fm.bin",
    "fm_core.bin",
    "sa.bin",
    "corpus.bin",
    "corpus.sentinel.bin",
    "chunk_map.bin",
}


def main():
    ap = argparse.ArgumentParser(description="Verify RLBWT Full Runtime Profile V1.")
    ap.add_argument("--runtime-dir", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--expected-l", type=int, required=True)
    ap.add_argument("--expected-r", type=int, required=True)
    ap.add_argument("--expected-count", type=int, required=True)
    ap.add_argument("--expected-offset", type=int, action="append", default=[])
    ap.add_argument("--corpus-bytes", type=int, required=True)
    ap.add_argument("--sample-step", type=int, default=128)
    args = ap.parse_args()

    runtime = Path(args.runtime_dir).resolve()

    rlbwt = runtime / "bwt.rlbwt"
    rank = runtime / "bwt.rlbwt.rank"
    locate = runtime / f"locate_core_s{args.sample_step}.bin"
    manifest = runtime / "rlbwt_full_runtime_manifest_v1.json"

    required = [rlbwt, rank, locate, manifest]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit("missing required runtime files: " + ", ".join(missing))

    files = [p for p in sorted(runtime.iterdir()) if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    forbidden_present = sorted([p.name for p in files if p.name in FORBIDDEN])
    if forbidden_present:
        raise SystemExit("forbidden runtime files present: " + ", ".join(forbidden_present))

    q = subprocess.run([
        sys.executable,
        str(ROOT / "tools" / "rlbwt_fm_query_v1.py"),
        "--rlbwt", str(rlbwt),
        "--rank-index", str(rank),
        "--query", args.query,
        "--expected-l", str(args.expected_l),
        "--expected-r", str(args.expected_r),
        "--expected-count", str(args.expected_count),
    ], check=True, text=True, capture_output=True)

    query_result = json.loads(q.stdout)

    loc_cmd = [
        sys.executable,
        str(ROOT / "tools" / "rlbwt_locate_offsets_v1.py"),
        "--rlbwt", str(rlbwt),
        "--rank-index", str(rank),
        "--locate-core", str(locate),
        "--l", str(args.expected_l),
        "--r", str(args.expected_r),
    ]

    for off in args.expected_offset:
        loc_cmd += ["--expected-offset", str(off)]

    l = subprocess.run(loc_cmd, check=True, text=True, capture_output=True)
    locate_result = json.loads(l.stdout)

    result = {
        "profile": "RLBWT_FULL_RUNTIME_PROFILE_V1",
        "runtime_dir": str(runtime),
        "runtime_total_bytes": total,
        "ratio_vs_corpus": total / args.corpus_bytes,
        "forbidden_runtime_files_absent": True,
        "query_result": query_result,
        "locate_result": locate_result,
        "ok": query_result.get("ok") is True and locate_result.get("ok") is True,
        "files": [{"name": p.name, "bytes": p.stat().st_size} for p in files],
    }

    print(json.dumps(result, indent=2))

    if not result["ok"]:
        raise SystemExit(1)

    print("[rlbwt-full-runtime] VERIFY OK")


if __name__ == "__main__":
    main()
