#!/usr/bin/env python3
"""Measure reversible precompression without weakening byte identity.

This experimental archive extends the whole-file V1 router with an external
precompressor.  An external candidate is eligible only after an immediate
byte-for-byte round trip.  The pinned executable hash is part of the receipt
and is checked again by restore.
"""

from __future__ import annotations

import argparse
import json
import lzma
import os
import shutil
import sqlite3
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

import verified_hybrid_archive as base


FORMAT = "GLYPH_VERIFIED_REVERSIBLE_PRECOMPRESSION_ARCHIVE_V1"
RECEIPT = f"{FORMAT}.json"
UNTRUSTED = 2
ELIGIBLE_SUFFIXES = frozenset({".jpeg", ".jpg", ".png", ".pdf", ".zip"})
PRECOMP_TIMEOUT_SECONDS = 900
PRECOMP_REFERENCE_SOURCE_COMMIT = "31b693d843e378e7d30190736f95c095769868b0"


ArchiveError = base.ArchiveError
canonical_hash = base.canonical_hash
canonical_json = base.canonical_json
load_inventory = base.load_inventory
read_stable_file = base.read_stable_file
safe_relative = base.safe_relative
sha256_bytes = base.sha256_bytes


def regular_executable(path: Path) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ArchiveError(f"missing precompressor: {path}") from exc
    if not stat.S_ISREG(mode) or not os.access(path, os.X_OK):
        raise ArchiveError(f"precompressor is not an executable regular file: {path}")


def verify_precompressor(path: Path, expected_sha256: str) -> str:
    regular_executable(path)
    if len(expected_sha256) != 64 or any(c not in "0123456789abcdef" for c in expected_sha256):
        raise ArchiveError("invalid precompressor SHA-256")
    actual = sha256_bytes(path.read_bytes())
    if actual != expected_sha256:
        raise ArchiveError(f"precompressor SHA-256 mismatch: expected {expected_sha256}, got {actual}")
    return actual


def run_checked(command: list[str], working_directory: Path) -> subprocess.CompletedProcess[bytes]:
    try:
        return subprocess.run(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=working_directory,
            timeout=PRECOMP_TIMEOUT_SECONDS,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ArchiveError("precompressor timed out") from exc
    except OSError as exc:
        raise ArchiveError("cannot execute precompressor") from exc


def precompress_roundtrip(precompressor: Path, data: bytes, suffix: str) -> tuple[bytes | None, str]:
    with tempfile.TemporaryDirectory(prefix="glyph-precomp-") as raw_directory:
        directory = Path(raw_directory)
        source = directory / f"input{suffix}"
        encoded = directory / "candidate.pcf"
        restored = directory / f"restored{suffix}"
        source.write_bytes(data)
        encode = run_checked([
            str(precompressor), "-cn", "-d0", f"-o{encoded}", str(source),
        ], directory)
        if encode.returncode == 2:
            return None, "no-supported-stream"
        if encode.returncode != 0 or not encoded.is_file():
            return None, f"encode-exit-{encode.returncode}"
        decode = run_checked([
            str(precompressor), "-r", f"-o{restored}", str(encoded),
        ], directory)
        if decode.returncode != 0 or not restored.is_file():
            return None, f"restore-exit-{decode.returncode}"
        if restored.read_bytes() != data:
            return None, "roundtrip-mismatch"
        return encoded.read_bytes(), "verified"


def encode_best(
    data: bytes,
    suffix: str,
    precompressor: Path,
) -> tuple[str, bytes, str]:
    codec, payload = base.encode_best(data)
    candidates = [(codec, payload)]
    attempted = suffix.casefold() in ELIGIBLE_SUFFIXES
    status = "not-eligible"
    if attempted:
        transformed, status = precompress_roundtrip(precompressor, data, suffix)
        if transformed is not None:
            candidates.extend([
                ("precomp-cn", transformed),
                ("precomp-cn+xz-9", lzma.compress(transformed, format=lzma.FORMAT_XZ, preset=9)),
            ])
    order = {name: index for index, (name, _) in enumerate(candidates)}
    selected, selected_payload = min(candidates, key=lambda item: (len(item[1]), order[item[0]]))
    return selected, selected_payload, status


def restore_precompressed(precompressor: Path, transformed: bytes, suffix: str) -> bytes:
    if suffix not in ELIGIBLE_SUFFIXES:
        raise ArchiveError(f"invalid precompressed object suffix: {suffix!r}")
    with tempfile.TemporaryDirectory(prefix="glyph-precomp-restore-") as raw_directory:
        directory = Path(raw_directory)
        encoded = directory / "object.pcf"
        restored = directory / f"restored{suffix}"
        encoded.write_bytes(transformed)
        result = run_checked([
            str(precompressor), "-r", f"-o{restored}", str(encoded),
        ], directory)
        if result.returncode != 0 or not restored.is_file():
            raise ArchiveError(f"precompressor restore failed with exit {result.returncode}")
        return restored.read_bytes()


def decode(codec: str, payload: bytes, suffix: str, precompressor: Path) -> bytes:
    if codec == "precomp-cn":
        return restore_precompressed(precompressor, payload, suffix)
    if codec == "precomp-cn+xz-9":
        try:
            transformed = lzma.decompress(payload, format=lzma.FORMAT_XZ)
        except lzma.LZMAError as exc:
            raise ArchiveError("invalid precompressed XZ payload") from exc
        return restore_precompressed(precompressor, transformed, suffix)
    return base.decode(codec, payload)


def object_path(root: Path, digest: str) -> Path:
    return root / "objects" / digest[:2] / digest[2:]


def finalize_manifest(manifest: dict, payload_bytes: int) -> tuple[bytes, bytes]:
    checksum_bytes = 64 + 2 + len(RECEIPT.encode("utf-8")) + 1
    for _ in range(100):
        raw = canonical_json(manifest)
        total = payload_bytes + len(raw) + checksum_bytes
        ratio = total / manifest["source_logical_bytes"] if manifest["source_logical_bytes"] else None
        saving = 1.0 - ratio if ratio is not None else None
        values = {
            "manifest_bytes": len(raw),
            "checksum_sidecar_bytes": checksum_bytes,
            "container_logical_bytes": total,
            # Fixed-width decimal strings avoid a JSON-length oscillation at a
            # floating-point representation boundary while preserving exact
            # container byte accounting.
            "storage_ratio": f"{ratio:.18f}" if ratio is not None else None,
            "saving_fraction": f"{saving:.18f}" if saving is not None else None,
            "target_met": bool(saving is not None and saving >= manifest["required_saving_fraction"]),
        }
        if all(manifest.get(key) == value for key, value in values.items()):
            break
        manifest.update(values)
    else:
        raise ArchiveError("manifest size did not converge")
    raw = canonical_json(manifest)
    checksum = f"{sha256_bytes(raw)}  {RECEIPT}\n".encode("ascii")
    if payload_bytes + len(raw) + len(checksum) != manifest["container_logical_bytes"]:
        raise ArchiveError("manifest size did not converge")
    return raw, checksum


def build(
    inventory_state: Path,
    output: Path,
    required_percent: float,
    precompressor: Path,
    precompressor_sha256: str,
) -> dict:
    if output.exists():
        raise ArchiveError(f"output already exists: {output}")
    verified_binary_sha = verify_precompressor(precompressor, precompressor_sha256)
    source, inventory_sha, rows = load_inventory(inventory_state)
    output = Path(os.path.realpath(output))
    if output == source or source in output.parents or output in source.parents:
        raise ArchiveError("archive and source must be disjoint")
    temporary = output.with_name(f".{output.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ArchiveError(f"temporary path already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        directories: list[str] = []
        files: list[dict] = []
        objects: dict[str, dict] = {}
        codec_counts: dict[str, int] = {}
        attempt_counts: dict[str, int] = {}
        logical_total = 0
        files_expected = sum(1 for row in rows if row[1] == "file")
        files_processed = 0
        for row in rows:
            raw_path, kind, *_ = row
            relative = safe_relative(raw_path)
            if kind == "directory":
                directories.append(raw_path)
                continue
            if kind != "file":
                raise ArchiveError(f"unsupported inventory entry kind {kind!r}: {raw_path!r}")
            data = read_stable_file(source, row)
            digest = sha256_bytes(data)
            logical_total += len(data)
            if digest not in objects:
                suffix = relative.suffix.casefold()
                codec, encoded, attempt_status = encode_best(data, suffix, precompressor)
                attempt_counts[attempt_status] = attempt_counts.get(attempt_status, 0) + 1
                if decode(codec, encoded, suffix, precompressor) != data:
                    raise ArchiveError(f"selected codec roundtrip failed: {raw_path!r}")
                destination = object_path(temporary, digest)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(encoded)
                objects[digest] = {
                    "sha256": digest,
                    "logical_bytes": len(data),
                    "stored_bytes": len(encoded),
                    "stored_sha256": sha256_bytes(encoded),
                    "codec": codec,
                    "suffix": suffix,
                    "path": destination.relative_to(temporary).as_posix(),
                }
                codec_counts[codec] = codec_counts.get(codec, 0) + 1
            elif objects[digest]["logical_bytes"] != len(data):
                raise ArchiveError("SHA-256 identity collision")
            files.append({"path": relative.as_posix(), "bytes": len(data), "sha256": digest})
            files_processed += 1
            if files_processed % 10 == 0 or files_processed == files_expected:
                print(json.dumps({
                    "format": FORMAT,
                    "progress": "compressing",
                    "files_processed": files_processed,
                    "files_total": files_expected,
                    "logical_bytes_processed": logical_total,
                    "unique_objects": len(objects),
                }, sort_keys=True), file=sys.stderr, flush=True)
        object_list = [objects[key] for key in sorted(objects)]
        payload_bytes = sum(item["stored_bytes"] for item in object_list)
        records = [{"path": item["path"], "bytes": item["bytes"], "sha256": item["sha256"]} for item in files]
        manifest = {
            "format": FORMAT,
            "schema_version": 1,
            "operation": "BUILD_WHOLE_FILE_CONTENT_ADDRESSED_REVERSIBLE_PRECOMPRESSION_ARCHIVE",
            "inventory_receipt_sha256": inventory_sha,
            "source_logical_bytes": logical_total,
            "source_manifest_root_sha256": canonical_hash(records),
            "files_total": len(files),
            "directories_total": len(directories),
            "unique_objects": len(object_list),
            "duplicate_file_references": len(files) - len(object_list),
            "compressed_object_payload_bytes": payload_bytes,
            "codec_object_counts": codec_counts,
            "precompression_attempt_counts": attempt_counts,
            "precompressor_sha256": verified_binary_sha,
            "precompressor_contract": "external-precomp-v0.4.7-compatible-cli",
            "precompressor_reference_source_commit": PRECOMP_REFERENCE_SOURCE_COMMIT,
            "eligible_suffixes": sorted(ELIGIBLE_SUFFIXES),
            "required_saving_fraction": required_percent / 100.0,
            "ratio_encoding": "fixed-18-decimal-string",
            "files": files,
            "directories": directories,
            "objects": object_list,
            "byte_perfect_object_roundtrip_verified": True,
            "complete": True,
            "limitations": [
                "Experimental external precompressor; not a trusted Personal Memory dependency.",
                "Every selected precompressed object passed an immediate exact-byte roundtrip.",
                "Whole-file routing only; no CDC, delta, cross-file dictionary or self-index.",
                "File contents and relative paths are preserved; filesystem metadata is not.",
                "At most one source object is processed at a time, but a single object is held in memory.",
            ],
        }
        raw, checksum = finalize_manifest(manifest, payload_bytes)
        (temporary / RECEIPT).write_bytes(raw)
        (temporary / f"{RECEIPT}.sha256").write_bytes(checksum)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_archive(archive: Path) -> dict:
    receipt = archive / RECEIPT
    checksum = archive / f"{RECEIPT}.sha256"
    base.regular(receipt, "archive receipt")
    base.regular(checksum, "archive receipt checksum")
    raw = receipt.read_bytes()
    if checksum.read_text(encoding="ascii") != f"{sha256_bytes(raw)}  {RECEIPT}\n":
        raise ArchiveError("archive receipt checksum mismatch")
    try:
        manifest = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveError("invalid archive receipt") from exc
    if raw != canonical_json(manifest) or manifest.get("format") != FORMAT or manifest.get("complete") is not True:
        raise ArchiveError("non-canonical or incompatible archive receipt")
    return manifest


def restore(archive: Path, destination: Path, precompressor: Path) -> dict:
    if destination.exists():
        raise ArchiveError(f"restore destination already exists: {destination}")
    manifest = load_archive(archive)
    verify_precompressor(precompressor, manifest.get("precompressor_sha256", ""))
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ArchiveError(f"temporary path already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        objects = {}
        for item in manifest.get("objects", []):
            digest = item.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or digest in objects:
                raise ArchiveError("invalid or duplicate object identity")
            objects[digest] = item
        files = manifest.get("files")
        directories = manifest.get("directories")
        if not isinstance(files, list) or not isinstance(directories, list):
            raise ArchiveError("invalid archive manifest collections")
        if len(objects) != manifest.get("unique_objects") or len(files) != manifest.get("files_total"):
            raise ArchiveError("archive manifest counters mismatch")
        for digest, item in objects.items():
            path = archive / safe_relative(item["path"])
            base.regular(path, "archive object")
            payload = path.read_bytes()
            if len(payload) != item["stored_bytes"] or sha256_bytes(payload) != item["stored_sha256"]:
                raise ArchiveError(f"stored object mismatch: {digest}")
            data = decode(item["codec"], payload, item.get("suffix", ""), precompressor)
            if len(data) != item["logical_bytes"] or sha256_bytes(data) != digest:
                raise ArchiveError(f"decoded object mismatch: {digest}")
        seen_directories = set()
        for raw in directories:
            if raw in seen_directories:
                raise ArchiveError("duplicate directory path")
            seen_directories.add(raw)
            (temporary / safe_relative(raw)).mkdir(parents=True, exist_ok=True)
        records = []
        seen_files = set()
        for item in files:
            relative = safe_relative(item["path"])
            if item["path"] in seen_files:
                raise ArchiveError("duplicate file path")
            seen_files.add(item["path"])
            try:
                obj = objects[item["sha256"]]
            except KeyError as exc:
                raise ArchiveError(f"missing referenced object: {item['sha256']}") from exc
            payload = (archive / safe_relative(obj["path"])).read_bytes()
            data = decode(obj["codec"], payload, obj.get("suffix", ""), precompressor)
            if len(data) != item["bytes"] or sha256_bytes(data) != item["sha256"]:
                raise ArchiveError(f"file/object mismatch: {item['path']!r}")
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            records.append({"path": item["path"], "bytes": len(data), "sha256": item["sha256"]})
        root = canonical_hash(records)
        if root != manifest["source_manifest_root_sha256"]:
            raise ArchiveError("restored manifest root mismatch")
        os.replace(temporary, destination)
        restored_bytes = sum(item["bytes"] for item in records)
        if restored_bytes != manifest.get("source_logical_bytes"):
            raise ArchiveError("restored logical byte count mismatch")
        return {
            "format": FORMAT,
            "complete": True,
            "files_restored": len(records),
            "restored_logical_bytes": restored_bytes,
            "restored_manifest_root_sha256": root,
            "byte_perfect_restore": True,
        }
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    build_parser = sub.add_parser("build")
    build_parser.add_argument("--inventory-state", type=Path, required=True)
    build_parser.add_argument("--output", type=Path, required=True)
    build_parser.add_argument("--required-saving-percent", type=float, default=30.0)
    build_parser.add_argument("--precomp", type=Path, required=True)
    build_parser.add_argument("--precomp-sha256", required=True)
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--archive", type=Path, required=True)
    restore_parser.add_argument("--destination", type=Path, required=True)
    restore_parser.add_argument("--precomp", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build(
                args.inventory_state.resolve(), args.output, args.required_saving_percent,
                args.precomp.resolve(), args.precomp_sha256,
            )
            shown = {key: result[key] for key in (
                "format", "complete", "source_logical_bytes", "container_logical_bytes",
                "storage_ratio", "saving_fraction", "target_met", "codec_object_counts",
                "precompression_attempt_counts",
            )}
            shown["storage_ratio"] = float(shown["storage_ratio"])
            shown["saving_fraction"] = float(shown["saving_fraction"])
        else:
            result = restore(args.archive.resolve(), args.destination, args.precomp.resolve())
            shown = result
        print(json.dumps(shown, sort_keys=True))
        return 0
    except (ArchiveError, OSError, sqlite3.DatabaseError, ValueError, TypeError, KeyError) as exc:
        print(f"STOP: untrusted reversible precompression operation: {exc}", file=sys.stderr)
        return UNTRUSTED


if __name__ == "__main__":
    raise SystemExit(main())
