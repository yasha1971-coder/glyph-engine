#!/usr/bin/env python3

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit():
    try:
        p = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            text=True,
            capture_output=True,
            check=True,
        )
        return p.stdout.strip()
    except Exception:
        return "unknown"


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--evidence", required=True)
    ap.add_argument("--out-dir", required=True)

    args = ap.parse_args()

    evidence_path = Path(args.evidence)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ev = json.loads(evidence_path.read_text(encoding="utf-8"))

    manifest = {
        "bundle_version": "EVIDENCE_BUNDLE_V1",
        "evidence_path": str(evidence_path),
        "evidence_sha256": sha256_file(evidence_path),

        "corpus_path": ev["corpus_path"],
        "corpus_sha256": sha256_file(ev["corpus_path"]),

        "fm_path": ev["fm_path"],
        "fm_sha256": sha256_file(ev["fm_path"]),

        "bwt_path": ev["bwt_path"],
        "bwt_sha256": sha256_file(ev["bwt_path"]),

        "sa_path": ev["sa_path"],
        "sa_sha256": sha256_file(ev["sa_path"]),

        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "generated_by_tool": "tools/build_evidence_bundle_v1.py",
        "generated_by_commit": git_commit(),
        "method": ev.get("method"),
        "index_tag": ev.get("index_tag"),
        "replay_command": (
            "tools/replay_evidence_v1.py "
            f"--evidence {str(evidence_path)}"
        ),
    }

    out_path = out_dir / "bundle_manifest.json"
    out_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(str(out_path))


if __name__ == "__main__":
    main()
