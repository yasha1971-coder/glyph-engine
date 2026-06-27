#!/usr/bin/env python3
import argparse
import gzip
import hashlib
import json
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def add_path_to_tar(tar: tarfile.TarFile, path: Path, arcname: str) -> None:
    ti = tar.gettarinfo(str(path), arcname=arcname)

    # Deterministic tar metadata.
    ti.mtime = 0
    ti.uid = 0
    ti.gid = 0
    ti.uname = ""
    ti.gname = ""

    if path.is_dir():
        ti.mode = 0o755
        tar.addfile(ti)
    else:
        ti.mode = 0o644
        with path.open("rb") as f:
            tar.addfile(ti, fileobj=f)


def make_deterministic_tar_gz(src_dir: Path, out_tar: Path, bundle_name: str) -> None:
    out_tar.parent.mkdir(parents=True, exist_ok=True)

    with out_tar.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz:
            with tarfile.open(fileobj=gz, mode="w") as tar:
                add_path_to_tar(tar, src_dir, bundle_name)

                for p in sorted(src_dir.rglob("*")):
                    rel = p.relative_to(src_dir).as_posix()
                    add_path_to_tar(tar, p, f"{bundle_name}/{rel}")


def main() -> int:
    ap = argparse.ArgumentParser(description="Create portable tar.gz RLBWT bounded evidence bundle V1.")
    ap.add_argument("--artifact", required=True)
    ap.add_argument("--out-tar", required=True)
    ap.add_argument("--bundle-name", default="rlbwt_bounded_evidence_bundle_v1")
    ap.add_argument("--work-dir", default="")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    artifact = Path(args.artifact).resolve()
    out_tar = Path(args.out_tar).resolve()
    bundle_name = args.bundle_name

    if "/" in bundle_name or bundle_name in ("", ".", ".."):
        raise SystemExit("bundle-name must be a simple directory name")

    if out_tar.exists() and not args.force:
        raise SystemExit(f"out tar exists: {out_tar}; use --force")

    if args.work_dir:
        work = Path(args.work_dir).resolve()
        if work.exists():
            if not args.force:
                raise SystemExit(f"work dir exists: {work}; use --force")
            shutil.rmtree(work)
        work.mkdir(parents=True, exist_ok=True)
        cleanup = False
    else:
        tmp = tempfile.TemporaryDirectory()
        work = Path(tmp.name)
        cleanup = True

    try:
        bundle_dir = work / bundle_name

        subprocess.run(
            [
                "python3",
                str(ROOT / "tools" / "make_rlbwt_bounded_evidence_bundle_v1.py"),
                "--artifact",
                str(artifact),
                "--out-dir",
                str(bundle_dir),
                "--force",
            ],
            check=True,
            text=True,
        )

        make_deterministic_tar_gz(bundle_dir, out_tar, bundle_name)

        result = {
            "ok": True,
            "tar_version": "RLBWT_BOUNDED_EVIDENCE_BUNDLE_TAR_V1",
            "bundle_version": "RLBWT_BOUNDED_EVIDENCE_BUNDLE_V1",
            "artifact_version": "RLBWT_BOUNDED_EVIDENCE_V1",
            "out_tar": str(out_tar),
            "tar_bytes": out_tar.stat().st_size,
            "tar_sha256": sha256_file(out_tar),
            "bundle_name": bundle_name,
        }

        print(json.dumps(result, indent=2, sort_keys=True))
    finally:
        if cleanup:
            tmp.cleanup()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
