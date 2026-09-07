#!/usr/bin/env python3
"""Read-only, restartable metadata inventory for the GLYPH 1 TB gate."""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
from pathlib import Path


FORMAT = "GLYPH_1TB_CORPUS_TRUTH_GATE_V1"
SCHEMA_VERSION = 1
INCOMPLETE = 75

CLASSES = {
    "archive": {".7z", ".bz2", ".gz", ".rar", ".tar", ".tgz", ".xz", ".zip", ".zst"},
    "document": {".doc", ".docx", ".epub", ".html", ".md", ".odt", ".pdf", ".ppt", ".pptx", ".rtf", ".txt", ".xls", ".xlsx"},
    "source_text": {".c", ".cc", ".cpp", ".css", ".go", ".h", ".hpp", ".java", ".js", ".json", ".py", ".rs", ".sh", ".ts", ".xml", ".yaml", ".yml"},
    "image": {".avif", ".bmp", ".gif", ".heic", ".jpeg", ".jpg", ".png", ".raw", ".svg", ".tif", ".tiff", ".webp"},
    "audio": {".aac", ".flac", ".m4a", ".mp3", ".ogg", ".wav", ".wma"},
    "video": {".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv"},
    "database": {".db", ".mdb", ".sqlite", ".sqlite3"},
    "model": {".bin", ".gguf", ".onnx", ".safetensors"},
    "executable": {".dll", ".dylib", ".exe", ".msi", ".so"},
}


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


def classify(name: str) -> tuple[str, str]:
    suffix = Path(name).suffix.casefold()
    for group, suffixes in CLASSES.items():
        if suffix in suffixes:
            return group, suffix
    return "other", suffix


def run_json(command: list[str]) -> object | None:
    try:
        completed = subprocess.run(command, check=True, text=True, capture_output=True)
        return json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def disk_facts(source: Path) -> dict[str, object]:
    fs = os.statvfs(source)
    facts: dict[str, object] = {
        "block_size": fs.f_frsize,
        "capacity_bytes": fs.f_blocks * fs.f_frsize,
        "free_bytes": fs.f_bfree * fs.f_frsize,
        "available_bytes": fs.f_bavail * fs.f_frsize,
    }
    mount = run_json(["findmnt", "-J", "-T", os.fspath(source), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"])
    if isinstance(mount, dict):
        filesystems = mount.get("filesystems")
        if isinstance(filesystems, list) and filesystems:
            facts["mount"] = filesystems[0]
    return facts


def connect(path: Path) -> sqlite3.Connection:
    db = sqlite3.connect(path)
    db.execute("PRAGMA journal_mode=WAL")
    db.execute("PRAGMA synchronous=FULL")
    db.execute("PRAGMA foreign_keys=ON")
    return db


def initialize(db: sqlite3.Connection, source: Path, state: Path) -> None:
    db.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        CREATE TABLE IF NOT EXISTS directories(
          path TEXT PRIMARY KEY,
          state TEXT NOT NULL CHECK(state IN ('pending','scanning','done','error')),
          error TEXT
        );
        CREATE TABLE IF NOT EXISTS entries(
          path TEXT PRIMARY KEY,
          kind TEXT NOT NULL,
          logical_bytes INTEGER NOT NULL,
          allocated_bytes INTEGER,
          mtime_ns INTEGER,
          device INTEGER,
          inode INTEGER,
          links INTEGER,
          suffix TEXT,
          class TEXT
        );
        CREATE TABLE IF NOT EXISTS errors(
          path TEXT PRIMARY KEY,
          operation TEXT NOT NULL,
          error TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS entries_class ON entries(class);
        CREATE INDEX IF NOT EXISTS entries_inode ON entries(device,inode);
        """
    )
    identity = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "source": os.fspath(source),
        "source_device": source.stat().st_dev,
        "source_inode": source.stat().st_ino,
        "state": os.fspath(state),
        "created_utc": utc_now(),
    }
    existing = dict(db.execute("SELECT key,value FROM meta"))
    if existing:
        for key in ("format", "schema_version", "source", "source_device", "source_inode"):
            if existing.get(key) != json.dumps(identity[key], sort_keys=True):
                raise SystemExit(f"STOP: state identity mismatch: {key}")
    else:
        db.executemany(
            "INSERT INTO meta(key,value) VALUES(?,?)",
            [(key, json.dumps(value, sort_keys=True)) for key, value in identity.items()],
        )
        db.execute("INSERT INTO directories(path,state) VALUES('','pending')")
    db.execute("UPDATE directories SET state='pending' WHERE state='scanning'")
    db.commit()


def entry_row(relative: str, item: os.DirEntry[str]) -> tuple[object, ...]:
    info = item.stat(follow_symlinks=False)
    mode = info.st_mode
    if stat.S_ISREG(mode):
        kind = "file"
        group, suffix = classify(item.name)
        logical = info.st_size
    elif stat.S_ISDIR(mode):
        kind, group, suffix, logical = "directory", None, None, 0
    elif stat.S_ISLNK(mode):
        kind, group, suffix, logical = "symlink", None, None, info.st_size
    else:
        kind, group, suffix, logical = "special", None, None, info.st_size
    blocks = getattr(info, "st_blocks", None)
    allocated = blocks * 512 if isinstance(blocks, int) else None
    return (
        relative, kind, logical, allocated, info.st_mtime_ns,
        info.st_dev, info.st_ino, info.st_nlink, suffix, group,
    )


def scan_one(db: sqlite3.Connection, source: Path, relative_dir: str) -> None:
    absolute = source / relative_dir
    db.execute("UPDATE directories SET state='scanning',error=NULL WHERE path=?", (relative_dir,))
    db.commit()
    try:
        with os.scandir(absolute) as iterator, db:
            # Stream entries. Never collect a whole large directory in memory.
            for item in iterator:
                relative = f"{relative_dir}/{item.name}" if relative_dir else item.name
                try:
                    row = entry_row(relative, item)
                    db.execute(
                        """INSERT OR REPLACE INTO entries
                        (path,kind,logical_bytes,allocated_bytes,mtime_ns,device,inode,links,suffix,class)
                        VALUES(?,?,?,?,?,?,?,?,?,?)""",
                        row,
                    )
                    if row[1] == "directory":
                        db.execute(
                            "INSERT OR IGNORE INTO directories(path,state) VALUES(?,'pending')",
                            (relative,),
                        )
                except OSError as exc:
                    db.execute(
                        "INSERT OR REPLACE INTO errors(path,operation,error) VALUES(?,?,?)",
                        (relative, "lstat", f"{type(exc).__name__}: {exc}"),
                    )
            db.execute("UPDATE directories SET state='done' WHERE path=?", (relative_dir,))
    except OSError as exc:
        with db:
            db.execute(
                "UPDATE directories SET state='error',error=? WHERE path=?",
                (f"{type(exc).__name__}: {exc}", relative_dir),
            )
            db.execute(
                "INSERT OR REPLACE INTO errors(path,operation,error) VALUES(?,?,?)",
                (relative_dir, "scandir", f"{type(exc).__name__}: {exc}"),
            )


def aggregate(db: sqlite3.Connection) -> dict[str, object]:
    by_kind = {
        kind: {"entries": count, "logical_bytes": size}
        for kind, count, size in db.execute(
            "SELECT kind,COUNT(*),SUM(logical_bytes) FROM entries GROUP BY kind"
        )
    }
    by_class = [
        {"class": group or "unclassified", "files": count, "logical_bytes": size}
        for group, count, size in db.execute(
            "SELECT class,COUNT(*),SUM(logical_bytes) FROM entries WHERE kind='file' GROUP BY class ORDER BY SUM(logical_bytes) DESC"
        )
    ]
    extensions = [
        {"suffix": suffix or "", "files": count, "logical_bytes": size}
        for suffix, count, size in db.execute(
            "SELECT suffix,COUNT(*),SUM(logical_bytes) FROM entries WHERE kind='file' GROUP BY suffix ORDER BY SUM(logical_bytes) DESC LIMIT 100"
        )
    ]
    directory_states = dict(db.execute("SELECT state,COUNT(*) FROM directories GROUP BY state"))
    return {
        "by_kind": by_kind,
        "by_class": by_class,
        "top_extensions": extensions,
        "directory_states": directory_states,
        "errors": db.execute("SELECT COUNT(*) FROM errors").fetchone()[0],
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_receipt(db: sqlite3.Connection, source: Path, state: Path) -> Path:
    summary = aggregate(db)
    unfinished = sum(summary["directory_states"].get(x, 0) for x in ("pending", "scanning"))
    receipt = {
        "format": FORMAT,
        "schema_version": SCHEMA_VERSION,
        "timestamp_utc": utc_now(),
        "operation": "READ_ONLY_METADATA_INVENTORY",
        "source": os.fspath(source),
        "source_content_read": False,
        "symlinks_followed": False,
        "source_write_attempted": False,
        "restartable": True,
        "bounded_memory": True,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "disk": disk_facts(source),
        "inventory": summary,
        "complete": unfinished == 0,
        "claim_1tb_closed": False,
        "note": "Metadata inventory does not establish duplicate bytes, compression ratio, content identity or 1 TB acceptance.",
    }
    path = state / "GLYPH_1TB_CORPUS_TRUTH_RECEIPT_V1.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    (state / "GLYPH_1TB_CORPUS_TRUTH_RECEIPT_V1.json.sha256").write_text(
        f"{sha256(path)}  {path.name}\n", encoding="ascii"
    )
    return path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--max-directories", type=int)
    args = parser.parse_args()

    source = canonical(args.source)
    state = canonical(args.state)
    if not source.is_dir():
        raise SystemExit(f"STOP: source directory missing: {source}")
    if is_within(state, source) or is_within(source, state):
        raise SystemExit("STOP: source and state must be disjoint")
    state.mkdir(parents=True, exist_ok=True)

    db = connect(state / "inventory.sqlite3")
    initialize(db, source, state)
    processed = 0
    while args.max_directories is None or processed < args.max_directories:
        row = db.execute("SELECT path FROM directories WHERE state='pending' ORDER BY path LIMIT 1").fetchone()
        if row is None:
            break
        scan_one(db, source, row[0])
        processed += 1
    receipt = write_receipt(db, source, state)
    complete = json.loads(receipt.read_text(encoding="utf-8"))["complete"]
    print(json.dumps({"format": FORMAT, "complete": complete, "directories_processed_this_run": processed, "receipt": os.fspath(receipt)}, sort_keys=True))
    return 0 if complete else INCOMPLETE


if __name__ == "__main__":
    raise SystemExit(main())
