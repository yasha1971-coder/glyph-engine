#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
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


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path):
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-index-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--checkpoint-step", type=int, default=2048)
    ap.add_argument("--sample-step", type=int, default=16)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    src = Path(args.source_index_dir)
    out = Path(args.out_dir)

    if not src.is_dir():
        raise SystemExit(f"source index dir not found: {src}")

    bwt_src = src / "bwt.bin"
    sa_src = src / "sa.bin"
    manifest_src = src / "manifest.json"

    if not bwt_src.exists():
        raise SystemExit(f"missing source bwt.bin: {bwt_src}")
    if not sa_src.exists():
        raise SystemExit(f"missing source sa.bin: {sa_src}")

    if out.exists():
        if not args.force:
            raise SystemExit(f"out dir exists; use --force: {out}")
        shutil.rmtree(out)

    out.mkdir(parents=True, exist_ok=True)

    shutil.copy2(bwt_src, out / "bwt.bin")
    if manifest_src.exists():
        shutil.copy2(manifest_src, out / "manifest.json")

    subprocess.run(
        [
            sys.executable,
            str(root / "tools" / "build_locate_fixture_v1.py"),
            "--bwt",
            str(bwt_src),
            "--sa",
            str(sa_src),
            "--out-dir",
            str(out),
            "--checkpoint-step",
            str(args.checkpoint_step),
            "--sample-step",
            str(args.sample_step),
        ],
        check=True,
    )

    loc_name = f"locate_core_s{args.sample_step}.bin"

    runtime_files = ["bwt.bin", "fm_core.bin", loc_name]
    if (out / "manifest.json").exists():
        runtime_files.append("manifest.json")

    missing = [name for name in runtime_files if not (out / name).exists()]
    if missing:
        raise SystemExit(f"missing runtime files after export: {missing}")

    forbidden_present = [name for name in FORBIDDEN if (out / name).exists()]
    if forbidden_present:
        raise SystemExit(f"forbidden runtime files present: {forbidden_present}")

    records = [file_record(out / name) for name in runtime_files]
    runtime_total = sum(r["bytes"] for r in records)

    profile = {
        "profile": "COMPACT_RUNTIME_PROFILE_V1",
        "checkpoint_step": args.checkpoint_step,
        "sample_step": args.sample_step,
        "source_index_dir": str(src),
        "runtime_dir": str(out),
        "runtime_total_bytes": runtime_total,
        "files": records,
        "forbidden_runtime_files_absent": True,
        "forbidden_runtime_files": FORBIDDEN,
    }

    profile_path = out / "compact_runtime_manifest_v1.json"
    profile_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print("[compact-runtime] exported")
    print(f"[compact-runtime] out_dir={out}")
    print(f"[compact-runtime] checkpoint_step={args.checkpoint_step}")
    print(f"[compact-runtime] sample_step={args.sample_step}")
    print(f"[compact-runtime] runtime_total_bytes={runtime_total}")
    print(f"[compact-runtime] manifest={profile_path}")


if __name__ == "__main__":
    main()
