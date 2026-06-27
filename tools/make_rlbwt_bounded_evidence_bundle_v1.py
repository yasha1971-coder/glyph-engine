#!/usr/bin/env python3
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(src)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description="Create portable RLBWT bounded evidence bundle V1.")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    artifact_path = Path(args.artifact).resolve()
    out_dir = Path(args.out_dir).resolve()

    if out_dir.exists():
        if not args.force:
            raise SystemExit(f"out dir exists: {out_dir}; use --force")
        shutil.rmtree(out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)

    artifact = json.loads(artifact_path.read_text())

    if artifact.get("artifact_version") != "RLBWT_BOUNDED_EVIDENCE_V1":
        raise SystemExit("artifact_version is not RLBWT_BOUNDED_EVIDENCE_V1")

    corpus_src = Path(artifact["source_corpus"]["path"]).resolve()
    runtime_src = Path(artifact["runtime_dir"]).resolve()

    corpus_dst = out_dir / "corpus.bin"
    evidence_dst = out_dir / "evidence.json"
    runtime_dst = out_dir / "runtime"

    copy_file(corpus_src, corpus_dst)

    runtime_files = [
        "bwt.rlbwt",
        "bwt.rlbwt.rank",
        "locate_core_s128.bin",
        "manifest.json",
        "rlbwt_full_runtime_manifest_v1.json",
    ]

    for name in runtime_files:
        copy_file(runtime_src / name, runtime_dst / name)

    bundled_artifact = json.loads(json.dumps(artifact))
    bundled_artifact["source_corpus"]["path"] = "corpus.bin"
    bundled_artifact["runtime_dir"] = "runtime"

    by_name = {x["name"]: x for x in bundled_artifact["runtime_files"]}
    for name in runtime_files:
        by_name[name]["path"] = f"runtime/{name}"

    evidence_dst.write_text(
        json.dumps(bundled_artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    readme = """# RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1

This bundle contains a replayable bounded evidence artifact over GLYPH RLBWT compressed runtime.

## Contents

- corpus.bin
- evidence.json
- runtime/bwt.rlbwt
- runtime/bwt.rlbwt.rank
- runtime/locate_core_s128.bin
- runtime/manifest.json
- runtime/rlbwt_full_runtime_manifest_v1.json
- bundle_manifest_v1.json

## Replay

From the GLYPH repository root, run:

    python3 tools/verify_rlbwt_bounded_evidence_bundle_v1.py --bundle <bundle_dir>

## Non-claim

This is bounded evidence. If bounded=true, the bundle does not claim exhaustive offset enumeration.
"""
    (out_dir / "README_REPLAY.md").write_text(readme, encoding="utf-8")

    files = []
    for p in sorted(out_dir.rglob("*")):
        if p.is_file():
            rel = p.relative_to(out_dir).as_posix()
            files.append({
                "path": rel,
                "bytes": p.stat().st_size,
                "sha256": sha256_file(p),
            })

    manifest = {
        "bundle_version": "RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1",
        "artifact_version": "RLBWT_BOUNDED_EVIDENCE_V1",
        "profile": "RLBWT_FULL_RUNTIME_PROFILE_V1",
        "evidence": "evidence.json",
        "source_corpus": "corpus.bin",
        "runtime_dir": "runtime",
        "files": files,
        "replay_command": "python3 tools/verify_rlbwt_bounded_evidence_bundle_v1.py --bundle <bundle_dir>",
    }

    (out_dir / "bundle_manifest_v1.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps({
        "ok": True,
        "bundle": str(out_dir),
        "bundle_version": "RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1",
        "evidence": str(evidence_dst),
        "file_count": len(files),
    }, indent=2, sort_keys=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
