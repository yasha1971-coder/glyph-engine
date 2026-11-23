#!/usr/bin/env python3
"""Measure fixed, aligned deduplication from a completed chunk-truth state."""

from __future__ import annotations

import argparse
from contextlib import closing, contextmanager
import fcntl
import hashlib
import json
import os
import sqlite3
import stat
import sys
from pathlib import Path


FORMAT = "GLYPH_DEDUP_BASELINE_V1"
SCHEMA_VERSION = 1
CHUNK_FORMAT = "GLYPH_RESTARTABLE_CHUNK_TRUTH_MANIFEST_V1"
CHUNK_RECEIPT = "GLYPH_CHUNK_TRUTH_RECEIPT_V1.json"
DATABASE = "chunk-truth.sqlite3"
LOCK = ".chunk-truth.lock"
INCOMPLETE = 75
UNTRUSTED = 2


class BaselineError(Exception):
    pass


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical_hash(records) -> str:
    digest = hashlib.sha256()
    for record in records:
        update_canonical_hash(digest, record)
    return digest.hexdigest()


def update_canonical_hash(digest, record) -> None:
    raw = json.dumps(
        record, ensure_ascii=True, sort_keys=True, separators=(",", ":")
    ).encode("ascii")
    digest.update(len(raw).to_bytes(8, "big"))
    digest.update(raw)


def reject_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BaselineError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def regular_file(path: Path, label: str) -> None:
    try:
        mode = path.lstat().st_mode
    except OSError as exc:
        raise BaselineError(f"missing {label}: {path}") from exc
    if not stat.S_ISREG(mode):
        raise BaselineError(f"{label} is not a regular file: {path}")


def parse_receipt(state: Path) -> tuple[dict, str]:
    path = state / CHUNK_RECEIPT
    checksum_path = path.with_suffix(".json.sha256")
    regular_file(path, "chunk receipt")
    regular_file(checksum_path, "chunk receipt checksum")
    raw = path.read_bytes()
    try:
        receipt = json.loads(raw, object_pairs_hook=reject_duplicate_keys)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BaselineError("invalid chunk receipt JSON") from exc
    canonical = (
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise BaselineError("chunk receipt is not in its canonical writer form")
    digest = sha256_bytes(raw)
    try:
        checksum = checksum_path.read_text(encoding="ascii")
    except UnicodeDecodeError as exc:
        raise BaselineError("invalid chunk receipt checksum encoding") from exc
    if checksum != f"{digest}  {CHUNK_RECEIPT}\n":
        raise BaselineError("chunk receipt checksum mismatch")
    if receipt.get("format") != CHUNK_FORMAT:
        raise BaselineError("chunk receipt format mismatch")
    if receipt.get("complete") is not True or receipt.get("errors") != 0:
        raise BaselineError("completed error-free chunk state is required")
    return receipt, digest


@contextmanager
def state_lock(state: Path):
    lock_path = state / LOCK
    if lock_path.exists() and lock_path.is_symlink():
        raise BaselineError("chunk lock must not be a symbolic link")
    with lock_path.open("a+b") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            print("STOP: chunk state is busy; another writer holds the lock", file=sys.stderr)
            raise SystemExit(INCOMPLETE)
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


def connect_read_only(path: Path) -> sqlite3.Connection:
    regular_file(path, "chunk database")
    db = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True)
    db.execute("PRAGMA query_only=ON")
    if db.execute("PRAGMA quick_check").fetchone() != ("ok",):
        db.close()
        raise BaselineError("chunk database failed quick_check")
    return db


def decode_meta(db: sqlite3.Connection) -> dict:
    try:
        return {
            key: json.loads(value, object_pairs_hook=reject_duplicate_keys)
            for key, value in db.execute("SELECT key,value FROM meta")
        }
    except (json.JSONDecodeError, TypeError) as exc:
        raise BaselineError("invalid chunk database metadata") from exc


def validate_chain(
    db: sqlite3.Connection, path: str, logical_bytes: int, chunk_bytes: int
) -> tuple[int, int, str]:
    offset = 0
    count = 0
    records = [{"logical_bytes": logical_bytes}]
    for ordinal, actual_offset, size, digest in db.execute(
        "SELECT ordinal,offset,bytes,sha256 FROM chunks "
        "WHERE path=? ORDER BY ordinal", (path,)
    ):
        if ordinal != count or actual_offset != offset:
            raise BaselineError(f"non-contiguous chunks for {path!r}")
        wanted = min(chunk_bytes, logical_bytes - offset)
        if size != wanted or size <= 0:
            raise BaselineError(f"invalid chunk size for {path!r}")
        if not isinstance(digest, str) or len(digest) != 64 or any(
            byte not in "0123456789abcdef" for byte in digest
        ):
            raise BaselineError(f"invalid chunk digest for {path!r}")
        records.append({"ordinal": ordinal, "bytes": size, "sha256": digest})
        offset += size
        count += 1
    return offset, count, canonical_hash(records)


def require_equal(receipt: dict, key: str, actual) -> None:
    if receipt.get(key) != actual:
        raise BaselineError(f"chunk receipt/database mismatch: {key}")


def measure(db: sqlite3.Connection, receipt: dict, receipt_sha256: str) -> dict:
    meta = decode_meta(db)
    if meta.get("format") != CHUNK_FORMAT or meta.get("schema_version") != 1:
        raise BaselineError("chunk database identity mismatch")
    if receipt.get("schema_version") != 1:
        raise BaselineError("chunk receipt schema mismatch")
    chunk_bytes = meta.get("chunk_bytes")
    if not isinstance(chunk_bytes, int) or chunk_bytes < 4096:
        raise BaselineError("invalid chunk size in database identity")
    for key in ("source", "inventory_receipt_sha256", "chunk_algorithm", "chunk_bytes"):
        require_equal(receipt, key, meta.get(key))

    manifest_digest = hashlib.sha256()
    files_total = 0
    logical_total = 0
    for row in db.execute(
        "SELECT path,logical_bytes,state,next_offset,chunk_count,"
        "content_root_sha256,error FROM files ORDER BY path"
    ):
        path, logical_bytes, state, next_offset, chunk_count, root, error = row
        if state != "done" or error is not None:
            raise BaselineError(f"incomplete file row: {path!r}")
        if not isinstance(logical_bytes, int) or logical_bytes < 0:
            raise BaselineError(f"invalid logical size: {path!r}")
        offset, count, calculated_root = validate_chain(
            db, path, logical_bytes, chunk_bytes
        )
        if offset != logical_bytes or next_offset != offset or chunk_count != count:
            raise BaselineError(f"file counters mismatch: {path!r}")
        if root != calculated_root:
            raise BaselineError(f"content root mismatch: {path!r}")
        update_canonical_hash(manifest_digest, {
            "path": path,
            "logical_bytes": logical_bytes,
            "content_root_sha256": root,
        })
        files_total += 1
        logical_total += logical_bytes

    chunk_references, referenced_bytes = db.execute(
        "SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM chunks"
    ).fetchone()
    unique_objects, unique_bytes = db.execute(
        "SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM "
        "(SELECT sha256,bytes FROM chunks GROUP BY sha256,bytes)"
    ).fetchone()
    if referenced_bytes != logical_total:
        raise BaselineError("chunk coverage does not equal logical bytes")

    calculated_manifest_root = manifest_digest.hexdigest()
    require_equal(receipt, "files_total", files_total)
    require_equal(receipt, "files_done", files_total)
    require_equal(receipt, "file_states", {"done": files_total})
    require_equal(receipt, "logical_bytes_total", logical_total)
    require_equal(receipt, "logical_bytes_hashed", logical_total)
    require_equal(receipt, "total_chunks", chunk_references)
    require_equal(receipt, "unique_chunks", unique_objects)
    require_equal(receipt, "hashed_chunk_bytes", referenced_bytes)
    require_equal(receipt, "unique_chunk_bytes", unique_bytes)
    require_equal(
        receipt, "proven_aligned_duplicate_bytes", referenced_bytes - unique_bytes
    )
    require_equal(receipt, "content_manifest_root_sha256", calculated_manifest_root)

    ratio = unique_bytes / referenced_bytes if referenced_bytes else None
    require_equal(receipt, "aligned_chunk_storage_ratio", ratio)
    for key in ("source_write_attempted", "source_payload_copied", "claim_1tb_closed"):
        require_equal(receipt, key, False)

    duplicate_groups = list(db.execute(
        "SELECT logical_bytes,COUNT(*) FROM files "
        "GROUP BY logical_bytes,content_root_sha256 HAVING COUNT(*)>1"
    ))
    duplicate_file_references = sum(count - 1 for _, count in duplicate_groups)
    duplicate_file_bytes = sum(size * (count - 1) for size, count in duplicate_groups)
    saved_bytes = referenced_bytes - unique_bytes

    return {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "operation": "MEASURE_COMPLETED_FIXED_CHUNK_STATE",
        "source": receipt["source"],
        "source_write_attempted": False,
        "payload_store_created": False,
        "compression_performed": False,
        "bounded_python_memory": True,
        "chunk_truth_receipt_sha256": receipt_sha256,
        "inventory_receipt_sha256": receipt["inventory_receipt_sha256"],
        "content_manifest_root_sha256": calculated_manifest_root,
        "chunk_algorithm": receipt["chunk_algorithm"],
        "chunk_bytes": chunk_bytes,
        "files_total": files_total,
        "logical_bytes_total": logical_total,
        "exact_duplicate_file_groups": len(duplicate_groups),
        "exact_duplicate_file_references": duplicate_file_references,
        "exact_duplicate_file_bytes": duplicate_file_bytes,
        "chunk_references": chunk_references,
        "unique_chunk_objects": unique_objects,
        "reused_chunk_references": chunk_references - unique_objects,
        "referenced_chunk_bytes": referenced_bytes,
        "unique_chunk_bytes": unique_bytes,
        "proven_aligned_duplicate_bytes": saved_bytes,
        "aligned_chunk_storage_ratio": ratio,
        "aligned_chunk_saving_fraction": 1.0 - ratio if ratio is not None else None,
        "complete": True,
        "claim_compression": False,
        "claim_shifted_content_dedup": False,
        "limitations": [
            "Measures exact fixed, file-aligned SHA-256 chunk reuse only.",
            "Does not detect general shifted-content redundancy, delta similarity or compressibility.",
            "Does not create or independently restore a payload object store.",
            "Trust inherits the completed chunk-truth run and its stable-source assumptions.",
            "Receipt checksums are not signatures or externally authenticated roots.",
            "The cooperating-writer lock is advisory and is not hostile-writer protection.",
        ],
    }


def write_result(state: Path, result: dict) -> Path:
    path = state / "GLYPH_DEDUP_BASELINE_V1.json"
    checksum_path = path.with_suffix(".json.sha256")
    raw = (
        json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    temporary = path.with_suffix(".json.tmp")
    temporary_checksum = checksum_path.with_suffix(".sha256.tmp")
    temporary.write_bytes(raw)
    temporary_checksum.write_text(
        f"{sha256_bytes(raw)}  {path.name}\n", encoding="ascii"
    )
    os.replace(temporary, path)
    os.replace(temporary_checksum, checksum_path)
    return path


def run(state: Path) -> Path:
    receipt, receipt_sha256 = parse_receipt(state)
    with closing(connect_read_only(state / DATABASE)) as db:
        result = measure(db, receipt, receipt_sha256)
    return write_result(state, result)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    state = Path(os.path.realpath(os.fspath(args.state)))
    if not state.is_dir():
        raise SystemExit("STOP: existing chunk state directory is required")
    try:
        with state_lock(state):
            result_path = run(state)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        print(json.dumps({
            "format": FORMAT,
            "complete": result["complete"],
            "logical_bytes_total": result["logical_bytes_total"],
            "proven_aligned_duplicate_bytes": result["proven_aligned_duplicate_bytes"],
            "aligned_chunk_storage_ratio": result["aligned_chunk_storage_ratio"],
            "receipt": os.fspath(result_path),
        }, sort_keys=True))
        return 0
    except (BaselineError, OSError, sqlite3.DatabaseError, ValueError, TypeError) as exc:
        print(f"STOP: untrusted dedup baseline input: {exc}", file=sys.stderr)
        return UNTRUSTED


if __name__ == "__main__":
    raise SystemExit(main())
