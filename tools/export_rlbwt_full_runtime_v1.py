#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
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


def file_info(path: Path):
    return {"name": path.name, "bytes": path.stat().st_size}


def main():
    ap = argparse.ArgumentParser(description="Export RLBWT Full Runtime Profile V1.")
    ap.add_argument("--source-index-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rank-step", type=int, default=8192)
    ap.add_argument("--sample-step", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = Path(args.source_index_dir).resolve()
    out = Path(args.out_dir).resolve()

    bwt = src / "bwt.bin"
    sa = src / "sa.bin"
    manifest = src / "manifest.json"

    missing = [str(p) for p in [bwt, sa, manifest] if not p.exists()]
    if missing:
        raise SystemExit("missing required source files: " + ", ".join(missing))

    if out.exists():
        if not args.force:
            raise SystemExit(f"out dir exists: {out}; use --force")
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)

    rlbwt = out / "bwt.rlbwt"
    rank = out / "bwt.rlbwt.rank"
    locate_core = out / f"locate_core_s{args.sample_step}.bin"

    subprocess.run([
        sys.executable,
        str(ROOT / "tools" / "rlbwt_container_v1.py"),
        "encode",
        "--bwt", str(bwt),
        "--out", str(rlbwt),
    ], check=True)

    subprocess.run([
        sys.executable,
        str(ROOT / "tools" / "rlbwt_rank_blocks_v1.py"),
        "build",
        "--rlbwt", str(rlbwt),
        "--out", str(rank),
        "--rank-step", str(args.rank_step),
    ], check=True)

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        subprocess.run([
            sys.executable,
            str(ROOT / "tools" / "build_locate_fixture_v1.py"),
            "--bwt", str(bwt),
            "--sa", str(sa),
            "--out-dir", str(tmp),
            "--checkpoint-step", str(args.rank_step),
            "--sample-step", str(args.sample_step),
        ], check=True)

        built_loc = tmp / f"locate_core_s{args.sample_step}.bin"
        if not built_loc.exists():
            raise SystemExit(f"locate core was not built: {built_loc}")
        shutil.copy2(built_loc, locate_core)

    shutil.copy2(manifest, out / "manifest.json")

    files = [p for p in sorted(out.iterdir()) if p.is_file()]
    forbidden_present = sorted([p.name for p in files if p.name in FORBIDDEN])

    profile = {
        "profile": "RLBWT_FULL_RUNTIME_PROFILE_V1",
        "source_index_dir": str(src),
        "out_dir": str(out),
        "rank_step": args.rank_step,
        "sample_step": args.sample_step,
        "query_supported": True,
        "fm_interval_supported": True,
        "match_count_supported": True,
        "locate_offsets_supported": True,
        "forbidden_runtime_files_present": forbidden_present,
        "runtime_total_bytes": 0,
        "files": [],
    }

    manifest_path = out / "rlbwt_full_runtime_manifest_v1.json"
    manifest_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    files = [p for p in sorted(out.iterdir()) if p.is_file()]
    total = sum(p.stat().st_size for p in files)

    profile["runtime_total_bytes"] = total
    profile["files"] = [file_info(p) for p in files]
    manifest_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print("[rlbwt-full-runtime] exported")
    print(f"[rlbwt-full-runtime] out_dir={out}")
    print(f"[rlbwt-full-runtime] rank_step={args.rank_step}")
    print(f"[rlbwt-full-runtime] sample_step={args.sample_step}")
    print(f"[rlbwt-full-runtime] runtime_total_bytes={total}")
    print(f"[rlbwt-full-runtime] locate_offsets_supported=true")


if __name__ == "__main__":
    main()
