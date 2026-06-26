#!/usr/bin/env python3
import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


FORBIDDEN = [
    "fm.bin",
    "sa.bin",
    "corpus.sentinel.bin",
    "corpus.bin",
    "chunk_map.bin",
]


def parse_query_output(text: str):
    m_interval = re.search(r"fm_interval:\s*\[(\d+),\s*(\d+)\]", text)
    m_count = re.search(r"match_count:\s*(\d+)", text)
    m_step = re.search(r"checkpoint_step:\s*(\d+)", text)
    m_bwt = re.search(r"bwt_bytes:\s*(\d+)", text)

    if not m_interval or not m_count:
        raise ValueError("could not parse query_fm_core_v1 output")

    return {
        "bwt_bytes": int(m_bwt.group(1)) if m_bwt else None,
        "checkpoint_step": int(m_step.group(1)) if m_step else None,
        "fm_interval": [int(m_interval.group(1)), int(m_interval.group(2))],
        "match_count": int(m_count.group(1)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runtime-dir", required=True)
    ap.add_argument("--sample-step", type=int, default=16)
    group = ap.add_mutually_exclusive_group(required=True)
    group.add_argument("--query")
    group.add_argument("--query-hex")
    ap.add_argument("--expected-count", type=int)
    ap.add_argument("--expected-l", type=int)
    ap.add_argument("--expected-r", type=int)
    ap.add_argument("--expected-offset", type=int, action="append", default=[])
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    runtime = Path(args.runtime_dir)

    if not runtime.is_dir():
        raise SystemExit(f"runtime dir not found: {runtime}")

    required = [
        runtime / "bwt.bin",
        runtime / "fm_core.bin",
        runtime / f"locate_core_s{args.sample_step}.bin",
    ]

    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise SystemExit(f"missing compact runtime files: {missing}")

    forbidden_present = [name for name in FORBIDDEN if (runtime / name).exists()]
    if forbidden_present:
        raise SystemExit(f"forbidden runtime files present: {forbidden_present}")

    if args.query_hex:
        query_hex = args.query_hex
    else:
        query_hex = args.query.encode("utf-8").hex()

    query_bin = root / "build" / "query_fm_core_v1"
    if not query_bin.exists():
        raise SystemExit(f"missing binary: {query_bin}; run cmake --build build -j2")

    q = subprocess.run(
        [
            str(query_bin),
            str(runtime / "fm_core.bin"),
            str(runtime / "bwt.bin"),
            query_hex,
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    query_result = parse_query_output(q.stdout)
    l, r = query_result["fm_interval"]

    if args.expected_count is not None and query_result["match_count"] != args.expected_count:
        raise SystemExit(f"expected count {args.expected_count}, got {query_result['match_count']}")
    if args.expected_l is not None and l != args.expected_l:
        raise SystemExit(f"expected l {args.expected_l}, got {l}")
    if args.expected_r is not None and r != args.expected_r:
        raise SystemExit(f"expected r {args.expected_r}, got {r}")

    loc = subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "glyph_locate_offsets_v0.py"),
            "--index-dir",
            str(runtime),
            "--l",
            str(l),
            "--r",
            str(r),
            "--sample-step",
            str(args.sample_step),
        ],
        check=True,
        text=True,
        capture_output=True,
    )

    locate_result = json.loads(loc.stdout)

    expected_offsets = args.expected_offset
    if expected_offsets:
        got = locate_result.get("offsets", [])
        for x in expected_offsets:
            if x not in got:
                raise SystemExit(f"expected offset {x} not found in {got}")

    files = []
    total = 0
    for p in sorted(runtime.iterdir()):
        if p.is_file():
            sz = p.stat().st_size
            total += sz
            files.append({"name": p.name, "bytes": sz})

    result = {
        "profile": "COMPACT_RUNTIME_PROFILE_V1",
        "runtime_dir": str(runtime),
        "forbidden_runtime_files_absent": True,
        "runtime_total_bytes": total,
        "files": files,
        "query_result": query_result,
        "locate_result": locate_result,
        "ok": True,
    }

    print(json.dumps(result, indent=2))
    print("[compact-runtime] VERIFY OK")


if __name__ == "__main__":
    main()
