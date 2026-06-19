#!/usr/bin/env python3
import argparse
import datetime as dt
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def run(cmd, cwd=None):
    return subprocess.run(
        cmd,
        cwd=str(cwd or ROOT),
        text=True,
        capture_output=True,
    )


def parse_query_output(stdout: str):
    interval = None
    count = None

    m = re.search(r"interval:\s*\[(\d+),\s*(\d+)\)", stdout)
    if m:
        interval = [int(m.group(1)), int(m.group(2))]

    m = re.search(r"count:\s*(\d+)", stdout)
    if m:
        count = int(m.group(1))

    return interval, count


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Create GLYPH_AUDIT_ARTIFACT_V0 for exact retrieval."
    )
    ap.add_argument("--index-dir", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--hex", action="store_true")
    args = ap.parse_args()

    index_dir = Path(args.index_dir).resolve()
    out_path = Path(args.output).resolve()

    raw_corpus = index_dir / "corpus.bin"
    manifest = index_dir / "manifest.json"
    fm = index_dir / "fm.bin"
    bwt = index_dir / "bwt.bin"

    required = [raw_corpus, manifest, fm, bwt]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("ERROR missing required files:", file=sys.stderr)
        for p in missing:
            print(f"  {p}", file=sys.stderr)
        return 2

    query_bytes = bytes.fromhex(args.query) if args.hex else args.query.encode("utf-8")
    query_hex = query_bytes.hex()

    verify_cmd = [
        "python3",
        "tools/query_verified_v1.py",
        str(index_dir.relative_to(ROOT) if index_dir.is_relative_to(ROOT) else index_dir),
        query_hex,
        "--hex",
    ]

    verified = run(verify_cmd, cwd=ROOT)
    interval, count = parse_query_output(verified.stdout)

    reproduce_status = "PASS" if verified.returncode == 0 and count is not None else "FAIL"

    offsets = []
    offset_mode = "not_available_in_query_fm_v1"

    if interval is not None:
        locate_cmd = [
            "python3",
            "tools/glyph_locate_offsets_v0.py",
            "--index-dir",
            str(index_dir.relative_to(ROOT) if index_dir.is_relative_to(ROOT) else index_dir),
            "--l",
            str(interval[0]),
            "--r",
            str(interval[1]),
        ]
        locate_run = run(locate_cmd, cwd=ROOT)
        if locate_run.returncode == 0:
            try:
                locate_data = json.loads(locate_run.stdout)
                if locate_data.get("ok") is True:
                    offsets = locate_data.get("offsets", [])
                    offset_mode = locate_data.get("offset_mode", "locate_backend_v2")
            except json.JSONDecodeError:
                pass

    artifact = {
        "artifact_version": "GLYPH_AUDIT_ARTIFACT_V0",
        "created_at_utc": dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat(),
        "glyph_version": "unknown",
        "corpus": {
            "path": str(raw_corpus.relative_to(ROOT) if raw_corpus.is_relative_to(ROOT) else raw_corpus),
            "sha256": sha256_file(raw_corpus),
            "size_bytes": raw_corpus.stat().st_size,
        },
        "index_manifest": {
            "path": str(manifest.relative_to(ROOT) if manifest.is_relative_to(ROOT) else manifest),
            "sha256": sha256_file(manifest),
        },
        "query": {
            "encoding": "hex",
            "hex": query_hex,
            "sha256": sha256_bytes(query_bytes),
            "length_bytes": len(query_bytes),
        },
        "result": {
            "match_count": count if count is not None else 0,
            "fm_interval": interval,
            "offsets": offsets,
            "offset_mode": offset_mode,
        },
        "verification": {
            "command": " ".join(verify_cmd),
            "reproduce_status": reproduce_status,
            "returncode": verified.returncode,
        },
    }

    if verified.stderr:
        artifact["verification"]["stderr"] = verified.stderr.strip()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"[audit-v0] wrote {out_path}")
    print(f"[audit-v0] reproduce_status={reproduce_status}")
    print(f"[audit-v0] match_count={artifact['result']['match_count']}")
    print(f"[audit-v0] offset_mode={artifact['result']['offset_mode']}")
    print(f"[audit-v0] offsets={artifact['result']['offsets']}")
    return 0 if reproduce_status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
