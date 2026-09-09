#!/usr/bin/env python3
"""Build and independently restore a small, evidence-bearing lossless archive.

This V1 pilot deliberately uses whole-file content addressing and an exhaustive
per-object router over raw, DEFLATE, bzip2 and XZ.  It is a compression truth
probe, not the final Personal Memory container format.
"""

from __future__ import annotations

import argparse
import bz2
import hashlib
import json
import lzma
import os
import shutil
import sqlite3
import stat
import sys
import zlib
from pathlib import Path


FORMAT = "GLYPH_VERIFIED_HYBRID_ARCHIVE_V1"
INVENTORY_FORMAT = "GLYPH_1TB_CORPUS_TRUTH_GATE_V1"
RECEIPT = "GLYPH_VERIFIED_HYBRID_ARCHIVE_V1.json"
UNTRUSTED = 2


class ArchiveError(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")


def canonical_hash(records: list[dict]) -> str:
    digest = hashlib.sha256()
    for record in records:
        raw = json.dumps(record, ensure_ascii=True, sort_keys=True, separators=(",", ":")).encode("ascii")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def safe_relative(raw: str) -> Path:
    path = Path(raw)
    if not raw or path.is_absolute() or ".." in path.parts or "." in path.parts:
        raise ArchiveError(f"unsafe inventory path: {raw!r}")
    return path


def regular(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise ArchiveError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(mode):
        raise ArchiveError(f"{label} is not a regular file: {path}")


def load_inventory(state: Path) -> tuple[Path, str, list[tuple]]:
    receipt_path = state / "GLYPH_1TB_CORPUS_TRUTH_RECEIPT_V1.json"
    checksum_path = receipt_path.with_suffix(".json.sha256")
    database_path = state / "inventory.sqlite3"
    regular(receipt_path, "inventory receipt")
    regular(checksum_path, "inventory receipt checksum")
    regular(database_path, "inventory database")
    raw = receipt_path.read_bytes()
    try:
        receipt = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArchiveError("invalid inventory receipt") from exc
    digest = sha256_bytes(raw)
    if checksum_path.read_text(encoding="ascii") != f"{digest}  {receipt_path.name}\n":
        raise ArchiveError("inventory receipt checksum mismatch")
    if receipt.get("format") != INVENTORY_FORMAT or receipt.get("complete") is not True:
        raise ArchiveError("completed inventory is required")
    if receipt.get("inventory", {}).get("errors") != 0:
        raise ArchiveError("inventory contains errors")
    source = Path(os.path.realpath(receipt["source"]))
    if not source.is_dir():
        raise ArchiveError("inventory source directory is missing")
    db = sqlite3.connect(f"{database_path.as_uri()}?mode=ro", uri=True)
    try:
        db.execute("PRAGMA query_only=ON")
        if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
            raise ArchiveError("inventory database failed quick_check")
        rows = list(db.execute(
            "SELECT path,kind,logical_bytes,mtime_ns,device,inode FROM entries ORDER BY path"
        ))
    finally:
        db.close()
    return source, digest, rows


def read_stable_file(source: Path, row: tuple) -> bytes:
    raw_path, kind, size, mtime_ns, device, inode = row
    if kind != "file":
        raise ArchiveError(f"not a file row: {raw_path!r}")
    relative = safe_relative(raw_path)
    path = source / relative
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(path, flags)
    except OSError as exc:
        raise ArchiveError(f"cannot open source file: {raw_path!r}") from exc
    try:
        before = os.fstat(fd)
        expected = (int(size), int(mtime_ns), int(device), int(inode))
        actual = (before.st_size, before.st_mtime_ns, before.st_dev, before.st_ino)
        if actual != expected or not stat.S_ISREG(before.st_mode):
            raise ArchiveError(f"source identity changed: {raw_path!r}")
        with os.fdopen(fd, "rb", closefd=False) as stream:
            data = stream.read()
        after = os.fstat(fd)
        actual_after = (after.st_size, after.st_mtime_ns, after.st_dev, after.st_ino)
        if actual_after != expected or len(data) != size:
            raise ArchiveError(f"source changed while reading: {raw_path!r}")
        return data
    finally:
        os.close(fd)


def encode_best(data: bytes) -> tuple[str, bytes]:
    candidates = [
        ("raw", data),
        ("deflate9", zlib.compress(data, 9)),
        ("bzip2-9", bz2.compress(data, 9)),
        ("xz-9", lzma.compress(data, format=lzma.FORMAT_XZ, preset=9)),
    ]
    order = {name: index for index, (name, _) in enumerate(candidates)}
    return min(candidates, key=lambda item: (len(item[1]), order[item[0]]))


def decode(codec: str, payload: bytes) -> bytes:
    if codec == "raw":
        return payload
    if codec == "deflate9":
        return zlib.decompress(payload)
    if codec == "bzip2-9":
        return bz2.decompress(payload)
    if codec == "xz-9":
        return lzma.decompress(payload, format=lzma.FORMAT_XZ)
    raise ArchiveError(f"unsupported codec: {codec!r}")


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
            "storage_ratio": ratio,
            "saving_fraction": saving,
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


def build(inventory_state: Path, output: Path, required_percent: float) -> dict:
    if output.exists():
        raise ArchiveError(f"output already exists: {output}")
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
        logical_total = 0
        files_expected = sum(1 for row in rows if row[1] == "file")
        files_processed = 0
        for row in rows:
            raw_path, kind, size, *_ = row
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
                codec, encoded = encode_best(data)
                if decode(codec, encoded) != data:
                    raise ArchiveError(f"codec roundtrip failed: {raw_path!r}")
                destination = object_path(temporary, digest)
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(encoded)
                objects[digest] = {
                    "sha256": digest,
                    "logical_bytes": len(data),
                    "stored_bytes": len(encoded),
                    "stored_sha256": sha256_bytes(encoded),
                    "codec": codec,
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
            "operation": "BUILD_WHOLE_FILE_CONTENT_ADDRESSED_HYBRID_ARCHIVE",
            "inventory_receipt_sha256": inventory_sha,
            "source_logical_bytes": logical_total,
            "source_manifest_root_sha256": canonical_hash(records),
            "files_total": len(files),
            "directories_total": len(directories),
            "unique_objects": len(object_list),
            "duplicate_file_references": len(files) - len(object_list),
            "compressed_object_payload_bytes": payload_bytes,
            "codec_object_counts": codec_counts,
            "required_saving_fraction": required_percent / 100.0,
            "files": files,
            "directories": directories,
            "objects": object_list,
            "byte_perfect_object_roundtrip_verified": True,
            "complete": True,
            "limitations": [
                "Whole-file codec routing only; no cross-file dictionary, CDC, delta or self-index.",
                "File contents and relative paths are preserved; timestamps, permissions and other filesystem metadata are not.",
                "Logical container bytes include object payloads, manifest and checksum sidecar; directory-entry bytes are excluded.",
                "XZ encoder byte determinism across library versions is not claimed.",
            ],
        }
        raw, checksum = finalize_manifest(manifest, payload_bytes)
        (temporary / RECEIPT).write_bytes(raw)
        (temporary / f"{RECEIPT}.sha256").write_bytes(checksum)
        os.replace(temporary, output)
        allocated = sum(p.stat().st_blocks * 512 for p in output.rglob("*") if p.is_file())
        manifest["container_allocated_file_bytes_observed"] = allocated
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_archive(archive: Path) -> dict:
    receipt = archive / RECEIPT
    checksum = archive / f"{RECEIPT}.sha256"
    regular(receipt, "archive receipt")
    regular(checksum, "archive receipt checksum")
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


def restore(archive: Path, destination: Path) -> dict:
    if destination.exists():
        raise ArchiveError(f"restore destination already exists: {destination}")
    manifest = load_archive(archive)
    temporary = destination.with_name(f".{destination.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise ArchiveError(f"temporary path already exists: {temporary}")
    temporary.mkdir(parents=True)
    try:
        object_items = manifest.get("objects")
        file_items = manifest.get("files")
        directory_items = manifest.get("directories")
        if not isinstance(object_items, list) or not isinstance(file_items, list) or not isinstance(directory_items, list):
            raise ArchiveError("invalid archive manifest collections")
        objects = {}
        for item in object_items:
            digest = item.get("sha256")
            if not isinstance(digest, str) or len(digest) != 64 or digest in objects:
                raise ArchiveError("invalid or duplicate object identity")
            objects[digest] = item
        if len(objects) != manifest.get("unique_objects") or len(file_items) != manifest.get("files_total"):
            raise ArchiveError("archive manifest counters mismatch")
        # Verify every stored object once while retaining at most one decoded file.
        for digest, item in objects.items():
            path = archive / safe_relative(item["path"])
            regular(path, "archive object")
            payload = path.read_bytes()
            if len(payload) != item["stored_bytes"] or sha256_bytes(payload) != item["stored_sha256"]:
                raise ArchiveError(f"stored object mismatch: {digest}")
            data = decode(item["codec"], payload)
            if len(data) != item["logical_bytes"] or sha256_bytes(data) != digest:
                raise ArchiveError(f"decoded object mismatch: {digest}")
        seen_directories = set()
        for raw in directory_items:
            if raw in seen_directories:
                raise ArchiveError("duplicate directory path")
            seen_directories.add(raw)
            (temporary / safe_relative(raw)).mkdir(parents=True, exist_ok=True)
        records = []
        seen_files = set()
        for item in file_items:
            relative = safe_relative(item["path"])
            if item["path"] in seen_files:
                raise ArchiveError("duplicate file path")
            seen_files.add(item["path"])
            try:
                obj = objects[item["sha256"]]
            except KeyError as exc:
                raise ArchiveError(f"missing referenced object: {item['sha256']}") from exc
            payload = (archive / safe_relative(obj["path"])).read_bytes()
            data = decode(obj["codec"], payload)
            if len(data) != item["bytes"]:
                raise ArchiveError(f"file/object size mismatch: {item['path']!r}")
            path = temporary / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            records.append({"path": item["path"], "bytes": len(data), "sha256": sha256_bytes(data)})
        root = canonical_hash(records)
        if root != manifest["source_manifest_root_sha256"]:
            raise ArchiveError("restored manifest root mismatch")
        os.replace(temporary, destination)
        restored_bytes = sum(x["bytes"] for x in records)
        if restored_bytes != manifest.get("source_logical_bytes"):
            raise ArchiveError("restored logical byte count mismatch")
        return {"format": FORMAT, "complete": True, "files_restored": len(records), "restored_logical_bytes": restored_bytes, "restored_manifest_root_sha256": root, "byte_perfect_restore": True}
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
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--archive", type=Path, required=True)
    restore_parser.add_argument("--destination", type=Path, required=True)
    args = parser.parse_args()
    try:
        if args.command == "build":
            result = build(args.inventory_state.resolve(), args.output, args.required_saving_percent)
            shown = {key: result[key] for key in ("format", "complete", "source_logical_bytes", "container_logical_bytes", "storage_ratio", "saving_fraction", "target_met", "codec_object_counts")}
        else:
            shown = restore(args.archive.resolve(), args.destination)
        print(json.dumps(shown, sort_keys=True))
        return 0
    except (ArchiveError, OSError, sqlite3.DatabaseError, ValueError, TypeError, KeyError) as exc:
        print(f"STOP: untrusted hybrid archive operation: {exc}", file=sys.stderr)
        return UNTRUSTED


if __name__ == "__main__":
    raise SystemExit(main())
