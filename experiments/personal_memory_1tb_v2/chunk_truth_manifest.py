#!/usr/bin/env python3
"""Restartable, read-only fixed-chunk truth manifest for GLYPH V2."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import itertools
import json
import os
import sqlite3
from pathlib import Path

FORMAT = "GLYPH_RESTARTABLE_CHUNK_TRUTH_MANIFEST_V1"
SCHEMA_VERSION = 1
DEFAULT_CHUNK_BYTES = 8 * 1024 * 1024
INCOMPLETE = 75
UNTRUSTED = 2


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical(path: Path) -> Path:
    return Path(os.path.realpath(os.fspath(path)))


def is_within(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(records) -> str:
    digest = hashlib.sha256()
    for record in records:
        raw = json.dumps(
            record, ensure_ascii=True, sort_keys=True, separators=(",", ":")
        ).encode("ascii")
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
    return digest.hexdigest()


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    check = db.execute("PRAGMA quick_check").fetchone()
    if check != ("ok",):
        raise SystemExit("STOP: chunk state database failed quick_check")
    return db


def load_inventory(inventory_state: Path) -> tuple[Path, Path, str]:
    receipt_path = inventory_state / "GLYPH_1TB_CORPUS_TRUTH_RECEIPT_V1.json"
    database_path = inventory_state / "inventory.sqlite3"
    if not receipt_path.is_file() or not database_path.is_file():
        raise SystemExit("STOP: complete metadata inventory is required")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("format") != "GLYPH_1TB_CORPUS_TRUTH_GATE_V1":
        raise SystemExit("STOP: inventory format mismatch")
    if receipt.get("complete") is not True or receipt.get("traversal_finished") is not True:
        raise SystemExit("STOP: inventory is incomplete")
    if receipt.get("inventory", {}).get("errors") != 0:
        raise SystemExit("STOP: inventory contains errors")
    return canonical(Path(receipt["source"])), database_path, sha256_file(receipt_path)


def initialize(db, source: Path, state: Path, inventory_sha: str, chunk_bytes: int) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY,value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS files(
          path TEXT PRIMARY KEY,
          logical_bytes INTEGER NOT NULL,
          mtime_ns INTEGER NOT NULL,
          device INTEGER NOT NULL,
          inode INTEGER NOT NULL,
          state TEXT NOT NULL CHECK(state IN ('pending','reading','done','error')),
          next_offset INTEGER NOT NULL DEFAULT 0,
          chunk_count INTEGER NOT NULL DEFAULT 0,
          content_root_sha256 TEXT,
          error TEXT
        );
        CREATE TABLE IF NOT EXISTS chunks(
          path TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
          ordinal INTEGER NOT NULL,
          offset INTEGER NOT NULL,
          bytes INTEGER NOT NULL,
          sha256 TEXT NOT NULL,
          PRIMARY KEY(path,ordinal)
        );
        CREATE INDEX IF NOT EXISTS chunks_identity ON chunks(sha256,bytes);
        """
    )
    identity = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "source": os.fspath(source),
        "state": os.fspath(state),
        "inventory_receipt_sha256": inventory_sha,
        "chunk_algorithm": "FIXED_SHA256_V1",
        "chunk_bytes": chunk_bytes,
    }
    existing = dict(db.execute("SELECT key,value FROM meta"))
    if existing:
        for key, value in identity.items():
            if existing.get(key) != json.dumps(value, sort_keys=True):
                raise SystemExit(f"STOP: state identity mismatch: {key}")
    else:
        values = {**identity, "created_utc": utc_now()}
        db.executemany(
            "INSERT INTO meta(key,value) VALUES(?,?)",
            [(key, json.dumps(value, sort_keys=True)) for key, value in values.items()],
        )
    db.execute("UPDATE files SET state='pending',error=NULL WHERE state='reading'")
    db.commit()


def import_inventory(db, inventory_database: Path) -> int:
    source_db = sqlite3.connect(f"file:{inventory_database}?mode=ro", uri=True)
    source_db.execute("PRAGMA query_only=ON")
    if source_db.execute("PRAGMA quick_check").fetchone() != ("ok",):
        raise SystemExit("STOP: inventory database failed quick_check")
    inserted = 0
    with db:
        for row in source_db.execute(
            "SELECT path,logical_bytes,mtime_ns,device,inode FROM entries "
            "WHERE kind='file' ORDER BY path"
        ):
            inserted += db.execute(
                "INSERT OR IGNORE INTO files "
                "(path,logical_bytes,mtime_ns,device,inode,state) "
                "VALUES(?,?,?,?,?,'pending')",
                row,
            ).rowcount
    source_db.close()
    return inserted


def stat_identity(info) -> tuple[int, int, int, int]:
    return info.st_size, info.st_mtime_ns, info.st_dev, info.st_ino


def expected_identity(row) -> tuple[int, int, int, int]:
    return tuple(map(int, row[1:5]))


def validate_chain(db, path: str, logical_bytes: int, chunk_bytes: int) -> tuple[int, int]:
    next_offset = 0
    count = 0
    for ordinal, offset, size, digest in db.execute(
        "SELECT ordinal,offset,bytes,sha256 FROM chunks WHERE path=? ORDER BY ordinal",
        (path,),
    ):
        if ordinal != count or offset != next_offset:
            raise ValueError("non-contiguous chunk checkpoint")
        wanted = min(chunk_bytes, logical_bytes - next_offset)
        if size != wanted or size <= 0:
            raise ValueError("invalid chunk size")
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise ValueError("invalid chunk digest")
        next_offset += size
        count += 1
    if next_offset > logical_bytes:
        raise ValueError("chunk checkpoint beyond file")
    return next_offset, count


def content_root(db, path: str, logical_bytes: int) -> str:
    head = ({"logical_bytes": logical_bytes},)
    chunks = (
        {"ordinal": ordinal, "bytes": size, "sha256": digest}
        for ordinal, size, digest in db.execute(
            "SELECT ordinal,bytes,sha256 FROM chunks WHERE path=? ORDER BY ordinal",
            (path,),
        )
    )
    return canonical_hash(itertools.chain(head, chunks))


def process_file(db, source: Path, row, chunk_bytes: int, budget: int | None) -> int:
    path = row[0]
    absolute = source / path
    used = 0
    try:
        before = absolute.stat(follow_symlinks=False)
        if expected_identity(row) != stat_identity(before):
            raise ValueError("source identity changed since metadata inventory")
        offset, ordinal = validate_chain(db, path, int(row[1]), chunk_bytes)
        with db:
            db.execute(
                "UPDATE files SET state='reading',next_offset=?,chunk_count=?,error=NULL WHERE path=?",
                (offset, ordinal, path),
            )
        with absolute.open("rb", buffering=0) as stream:
            stream.seek(offset)
            while offset < int(row[1]):
                if budget is not None and used >= budget:
                    with db:
                        db.execute("UPDATE files SET state='pending' WHERE path=?", (path,))
                    return used
                wanted = min(chunk_bytes, int(row[1]) - offset)
                data = stream.read(wanted)
                if len(data) != wanted:
                    raise ValueError(f"short read: {len(data)} != {wanted}")
                digest = hashlib.sha256(data).hexdigest()
                with db:
                    db.execute(
                        "INSERT INTO chunks(path,ordinal,offset,bytes,sha256) VALUES(?,?,?,?,?)",
                        (path, ordinal, offset, wanted, digest),
                    )
                    offset += wanted
                    ordinal += 1
                    used += 1
                    db.execute(
                        "UPDATE files SET next_offset=?,chunk_count=? WHERE path=?",
                        (offset, ordinal, path),
                    )
            after = os.fstat(stream.fileno())
        if expected_identity(row) != stat_identity(after):
            raise ValueError("source identity changed while reading")
        root = content_root(db, path, int(row[1]))
        with db:
            db.execute(
                "UPDATE files SET state='done',next_offset=logical_bytes," 
                "chunk_count=?,content_root_sha256=?,error=NULL WHERE path=?",
                (ordinal, root, path),
            )
        return used
    except (OSError, ValueError, sqlite3.DatabaseError) as exc:
        with db:
            db.execute(
                "UPDATE files SET state='error',error=? WHERE path=?",
                (f"{type(exc).__name__}: {exc}", path),
            )
        return used


def manifest_root(db) -> str:
    return canonical_hash(
        {"path": path, "logical_bytes": size, "content_root_sha256": root}
        for path, size, root in db.execute(
            "SELECT path,logical_bytes,content_root_sha256 FROM files "
            "WHERE state='done' ORDER BY path"
        )
    )


def write_receipt(db, source: Path, state: Path, inventory_sha: str, chunk_bytes: int) -> Path:
    states = dict(db.execute("SELECT state,COUNT(*) FROM files GROUP BY state"))
    files_total, logical_total = db.execute(
        "SELECT COUNT(*),COALESCE(SUM(logical_bytes),0) FROM files"
    ).fetchone()
    files_done, logical_done = db.execute(
        "SELECT COUNT(*),COALESCE(SUM(logical_bytes),0) FROM files WHERE state='done'"
    ).fetchone()
    total_chunks, hashed_bytes = db.execute(
        "SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM chunks"
    ).fetchone()
    unique_chunks, unique_bytes = db.execute(
        "SELECT COUNT(*),COALESCE(SUM(bytes),0) FROM "
        "(SELECT sha256,bytes FROM chunks GROUP BY sha256,bytes)"
    ).fetchone()
    errors = states.get("error", 0)
    complete = files_done == files_total and errors == 0
    receipt = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "operation": "READ_ONLY_CONTENT_HASHING",
        "source": os.fspath(source),
        "inventory_receipt_sha256": inventory_sha,
        "chunk_algorithm": "FIXED_SHA256_V1",
        "chunk_bytes": chunk_bytes,
        "source_write_attempted": False,
        "source_payload_copied": False,
        "restartable_inside_files": True,
        "bounded_memory": True,
        "files_total": files_total,
        "files_done": files_done,
        "logical_bytes_total": logical_total,
        "logical_bytes_hashed": logical_done,
        "file_states": states,
        "total_chunks": total_chunks,
        "unique_chunks": unique_chunks,
        "hashed_chunk_bytes": hashed_bytes,
        "unique_chunk_bytes": unique_bytes,
        "proven_aligned_duplicate_bytes": hashed_bytes - unique_bytes,
        "aligned_chunk_storage_ratio": unique_bytes / hashed_bytes if hashed_bytes else None,
        "content_manifest_root_sha256": manifest_root(db),
        "errors": errors,
        "complete": complete,
        "claim_1tb_closed": False,
        "limitations": [
            "Fixed chunks prove aligned duplicates, not shifted-content redundancy.",
            "Hashes only: no compressed payload repository is created by this gate.",
            "Source stability is checked using size, mtime, device and inode.",
        ],
    }
    path = state / "GLYPH_CHUNK_TRUTH_RECEIPT_V1.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)
    path.with_suffix(".json.sha256").write_text(
        f"{sha256_file(path)}  {path.name}\n", encoding="ascii"
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory-state", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK_BYTES)
    parser.add_argument("--max-chunks", type=int)
    args = parser.parse_args()
    if args.chunk_bytes < 4096:
        raise SystemExit("STOP: chunk size must be at least 4096 bytes")
    if args.max_chunks is not None and args.max_chunks < 1:
        raise SystemExit("STOP: max chunks must be positive")

    inventory_state = canonical(args.inventory_state)
    state = canonical(args.state)
    source, inventory_db, inventory_sha = load_inventory(inventory_state)
    if is_within(state, source) or is_within(source, state):
        raise SystemExit("STOP: source and state must be disjoint")
    if is_within(state, inventory_state) or is_within(inventory_state, state):
        raise SystemExit("STOP: inventory and chunk state must be disjoint")
    state.mkdir(parents=True, exist_ok=True)
    db = connect(state / "chunk-truth.sqlite3")
    initialize(db, source, state, inventory_sha, args.chunk_bytes)
    inserted = import_inventory(db, inventory_db)
    processed = 0
    while args.max_chunks is None or processed < args.max_chunks:
        row = db.execute(
            "SELECT path,logical_bytes,mtime_ns,device,inode FROM files "
            "WHERE state='pending' ORDER BY path LIMIT 1"
        ).fetchone()
        if row is None:
            break
        remaining = None if args.max_chunks is None else args.max_chunks - processed
        used = process_file(db, source, row, args.chunk_bytes, remaining)
        processed += used
        if remaining == 0:
            break
    receipt_path = write_receipt(db, source, state, inventory_sha, args.chunk_bytes)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    print(json.dumps({
        "format": FORMAT,
        "complete": receipt["complete"],
        "errors": receipt["errors"],
        "files_imported": inserted,
        "chunks_processed_this_run": processed,
        "receipt": os.fspath(receipt_path),
    }, sort_keys=True))
    if receipt["errors"]:
        return UNTRUSTED
    return 0 if receipt["complete"] else INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main())
