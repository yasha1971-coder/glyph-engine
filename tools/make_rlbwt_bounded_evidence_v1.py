#!/usr/bin/env python3
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_bytes_at(path: Path, off: int, n: int) -> bytes:
    with path.open("rb") as f:
        f.seek(off)
        return f.read(n)


def parse_server_line(line: str):
    parts = line.rstrip("\n").split("\t")
    if len(parts) < 9:
        raise RuntimeError(f"bad server line: {line!r}")
    if parts[0] != "OK":
        raise RuntimeError(f"server error: {line!r}")

    offsets = []
    if parts[8]:
        offsets = [int(x) for x in parts[8].split(",") if x]

    return {
        "l": int(parts[1]),
        "r": int(parts[2]),
        "match_count": int(parts[3]),
        "located_count": int(parts[4]),
        "bounded": parts[5] == "true",
        "total_steps": int(parts[6]),
        "max_steps": int(parts[7]),
        "offsets": offsets,
    }


def run_server_once(runtime_dir: Path, query_hex: str, max_offsets: int):
    server = ROOT / "build" / "rlbwt_full_query_locate_server_v1"
    if not server.exists():
        raise FileNotFoundError(f"missing server binary: {server}")

    proc = subprocess.run(
        [
            str(server),
            str(runtime_dir / "bwt.rlbwt"),
            str(runtime_dir / "bwt.rlbwt.rank"),
            str(runtime_dir / "locate_core_s128.bin"),
        ],
        input=f"{query_hex}\t{max_offsets}\nQUIT\n",
        text=True,
        capture_output=True,
        check=True,
    )

    lines = [x for x in proc.stdout.splitlines() if x.strip()]
    if not lines:
        raise RuntimeError(f"server produced no stdout, stderr={proc.stderr}")

    return parse_server_line(lines[0])


def main() -> int:
    ap = argparse.ArgumentParser(description="Create RLBWT bounded evidence artifact V1.")
    ap.add_argument("--runtime-dir", required=True)
    ap.add_argument("--source-corpus", required=True)
    ap.add_argument("--query", required=True)
    ap.add_argument("--max-offsets", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    runtime = Path(args.runtime_dir).resolve()
    corpus = Path(args.source_corpus).resolve()
    out = Path(args.out).resolve()

    query_bytes = args.query.encode("utf-8")
    query_hex = query_bytes.hex()
    query_sha256 = hashlib.sha256(query_bytes).hexdigest()

    required_runtime = [
        "bwt.rlbwt",
        "bwt.rlbwt.rank",
        "locate_core_s128.bin",
        "manifest.json",
        "rlbwt_full_runtime_manifest_v1.json",
    ]

    runtime_files = []
    for name in required_runtime:
        p = runtime / name
        runtime_files.append({
            "name": name,
            "path": str(p),
            "bytes": p.stat().st_size,
            "sha256": sha256_file(p),
        })

    result = run_server_once(runtime, query_hex, args.max_offsets)

    byte_checks = []
    all_ok = True

    for off in result["offsets"]:
        got = read_bytes_at(corpus, off, len(query_bytes))
        ok = got == query_bytes
        all_ok = all_ok and ok
        byte_checks.append({
            "offset": off,
            "expected_hex": query_hex,
            "observed_hex": got.hex(),
            "ok": ok,
        })

    artifact = {
        "artifact_version": "RLBWT_BOUNDED_EVIDENCE_V1",
        "profile": "RLBWT_FULL_RUNTIME_PROFILE_V1",
        "server_protocol": "query_hex<TAB>max_offsets",
        "engine": {
            "repo": "glyph-engine",
            "tool": "tools/make_rlbwt_bounded_evidence_v1.py",
            "server_binary": "rlbwt_full_query_locate_server_v1",
        },
        "query": {
            "text": args.query,
            "hex": query_hex,
            "sha256": query_sha256,
            "bytes": len(query_bytes),
        },
        "source_corpus": {
            "path": str(corpus),
            "bytes": corpus.stat().st_size,
            "sha256": sha256_file(corpus),
        },
        "runtime_dir": str(runtime),
        "runtime_files": runtime_files,
        "retrieval": {
            "fm_interval": [result["l"], result["r"]],
            "match_count": result["match_count"],
            "total_possible_count": result["match_count"],
            "max_offsets": args.max_offsets,
            "returned_count": result["located_count"],
            "bounded": result["bounded"],
            "offsets": result["offsets"],
            "total_steps_returned": result["total_steps"],
            "max_steps_returned": result["max_steps"],
        },
        "byte_check": {
            "all_returned_offsets_match_query": all_ok,
            "checks": byte_checks,
        },
        "ok": all_ok,
    }

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({
        "ok": all_ok,
        "out": str(out),
        "match_count": result["match_count"],
        "returned_count": result["located_count"],
        "bounded": result["bounded"],
        "offsets": result["offsets"],
        "byte_check": all_ok,
    }, indent=2, sort_keys=True))

    if not all_ok:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
