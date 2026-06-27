#!/usr/bin/env python3
import argparse
import json
import shutil
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


def file_info(path: Path):
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
    }


def main():
    ap = argparse.ArgumentParser(description="Export RLBWT Query Runtime Profile V1.")
    ap.add_argument("--source-index-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--rank-step", type=int, default=8192)
    ap.add_argument("--sample-step", type=int, default=128)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    src = Path(args.source_index_dir).resolve()
    out = Path(args.out_dir).resolve()

    bwt = src / "bwt.bin"
    manifest = src / "manifest.json"
    locate_src = src / f"locate_core_s{args.sample_step}.bin"

    missing = []
    for p in [bwt, manifest]:
        if not p.exists():
            missing.append(str(p))

    if missing:
        raise SystemExit("missing required files: " + ", ".join(missing))

    if out.exists():
        if not args.force:
            raise SystemExit(f"out dir exists: {out}; use --force")
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)

    rlbwt = out / "bwt.rlbwt"
    rank = out / "bwt.rlbwt.rank"

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "rlbwt_container_v1.py"),
            "encode",
            "--bwt",
            str(bwt),
            "--out",
            str(rlbwt),
        ],
        check=True,
    )

    subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "rlbwt_rank_blocks_v1.py"),
            "build",
            "--rlbwt",
            str(rlbwt),
            "--out",
            str(rank),
            "--rank-step",
            str(args.rank_step),
        ],
        check=True,
    )

    shutil.copy2(manifest, out / "manifest.json")

    locate_present = False
    if locate_src.exists():
        shutil.copy2(locate_src, out / locate_src.name)
        locate_present = True

    files = [p for p in sorted(out.iterdir()) if p.is_file()]
    total = sum(p.stat().st_size for p in files)

    forbidden_present = sorted([p.name for p in files if p.name in FORBIDDEN])

    profile = {
        "profile": "RLBWT_QUERY_RUNTIME_PROFILE_V1",
        "source_index_dir": str(src),
        "out_dir": str(out),
        "rank_step": args.rank_step,
        "sample_step": args.sample_step,
        "runtime_total_bytes": total,
        "files": [file_info(p) for p in files],
        "forbidden_runtime_files_present": forbidden_present,
        "locate_core_present": locate_present,
        "query_supported": True,
        "fm_interval_supported": True,
        "match_count_supported": True,
        "locate_offsets_supported": False,
        "locate_note": "Current locate_backend_v2 still requires raw bwt.bin. RLBWT runtime profile supports query/count/FM interval only.",
    }

    (out / "rlbwt_query_runtime_manifest_v1.json").write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )

    # Recompute after manifest write.
    files = [p for p in sorted(out.iterdir()) if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    profile["runtime_total_bytes"] = total
    profile["files"] = [file_info(p) for p in files]
    (out / "rlbwt_query_runtime_manifest_v1.json").write_text(
        json.dumps(profile, indent=2),
        encoding="utf-8",
    )

    print("[rlbwt-runtime] exported")
    print(f"[rlbwt-runtime] out_dir={out}")
    print(f"[rlbwt-runtime] rank_step={args.rank_step}")
    print(f"[rlbwt-runtime] sample_step={args.sample_step}")
    print(f"[rlbwt-runtime] runtime_total_bytes={total}")
    print(f"[rlbwt-runtime] locate_offsets_supported=false")


if __name__ == "__main__":
    main()
