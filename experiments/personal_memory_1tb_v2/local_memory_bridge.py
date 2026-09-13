#!/usr/bin/env python3
"""Read-only, bounded evidence adapter. No model, network or external decoder.

Caller pins receipts and grants individual paths. This is an exact UTF-8 scan
gate, not a self-index implementation or an authentication scheme.
"""
from __future__ import annotations

import bz2
import json
import lzma
import re
import zlib
from pathlib import Path

import verified_hybrid_archive as base
import verified_reversible_precompression_archive as precomp

FORMAT = "GLYPH_LOCAL_MEMORY_EVIDENCE_V1"
HEX = re.compile(r"[0-9a-f]{64}\Z")
Error = base.ArchiveError


def read_regular(root: Path, relative: str, limit: int) -> bytes:
    path = Path(relative)
    if not relative or path.is_absolute() or any(x in ("", ".", "..") for x in relative.split("/")):
        raise Error("unsafe relative path")
    current = root
    for part in path.parts:
        current = current / part
        if current.is_symlink():
            raise Error("symlink rejected")
    base.regular(current, "evidence source")
    with current.open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise Error("read budget exceeded")
    return raw


class ArchiveView:
    def __init__(self, root: Path, receipt_sha256: str, *, max_file_bytes=8 * 1024 * 1024):
        self.root = root.resolve()
        self.limit = max_file_bytes
        if not HEX.fullmatch(receipt_sha256) or max_file_bytes <= 0:
            raise Error("invalid receipt pin or budget")
        candidates = [name for name in (base.RECEIPT, precomp.RECEIPT) if (self.root / name).exists()]
        if len(candidates) != 1:
            raise Error("expected exactly one supported receipt")
        raw = read_regular(self.root, candidates[0], 8 * 1024 * 1024)
        if base.sha256_bytes(raw) != receipt_sha256:
            raise Error("receipt differs from caller pin")
        manifest = json.loads(raw)
        expected = base.FORMAT if candidates[0] == base.RECEIPT else precomp.FORMAT
        if manifest.get("format") != expected or manifest.get("complete") is not True:
            raise Error("incompatible or incomplete archive")
        self.version = receipt_sha256
        self.files = {}
        self.objects = {}
        for obj in manifest["objects"]:
            digest = obj["sha256"]
            if not HEX.fullmatch(digest) or digest in self.objects:
                raise Error("invalid object identity")
            self.objects[digest] = obj
        for item in manifest["files"]:
            path = item["path"]
            if path in self.files or item["sha256"] not in self.objects:
                raise Error("duplicate path or missing object")
            if item["bytes"] != self.objects[item["sha256"]]["logical_bytes"]:
                raise Error("file size mismatch")
            self.files[path] = item
        records = [{key: f[key] for key in ("path", "bytes", "sha256")} for f in manifest["files"]]
        if (base.canonical_hash(records) != manifest["source_manifest_root_sha256"]
                or len(self.files) != manifest["files_total"]
                or len(self.objects) != manifest["unique_objects"]
                or sum(x["bytes"] for x in records) != manifest["source_logical_bytes"]):
            raise Error("manifest identity/counters mismatch")

    def read(self, path: str) -> bytes | None:
        item = self.files[path]
        obj = self.objects[item["sha256"]]
        if not 0 <= obj["logical_bytes"] <= self.limit or not 0 <= obj["stored_bytes"] <= self.limit:
            return None
        codec = obj["codec"]
        # External precompression requires a separately sandboxed worker. Never
        # launch a process just because an LLM requested a document.
        if codec not in ("raw", "deflate9", "bzip2-9", "xz-9"):
            return None
        payload = read_regular(self.root, obj["path"], self.limit)
        if len(payload) != obj["stored_bytes"] or base.sha256_bytes(payload) != obj["stored_sha256"]:
            raise Error("stored bytes corrupted")
        if codec == "raw":
            data = payload
        else:
            decoder = {"deflate9": zlib.decompressobj, "bzip2-9": bz2.BZ2Decompressor,
                       "xz-9": lambda: lzma.LZMADecompressor(memlimit=128 * 1024 * 1024)}[codec]()
            data = decoder.decompress(payload, self.limit + 1)
            if not decoder.eof or decoder.unused_data:
                raise Error("invalid or oversized compressed stream")
        if len(data) != item["bytes"] or base.sha256_bytes(data) != item["sha256"]:
            raise Error("decoded bytes corrupted")
        return data


def evidence(views: list[ArchiveView], query: str, grants: set[tuple[str, str]], *,
             context_bytes=4096, scan_bytes=32 * 1024 * 1024) -> dict:
    """Grants are (receipt SHA-256, path), provided by host UI, never by LLM.

    Byte budget bounds snippets, not model token count or total process RSS.
    Errors discard all accumulated snippets (caller must not stream them early).
    """
    needle = query.encode("utf-8")
    if not needle or len(needle) > 256 or not 0 < context_bytes <= 16384 or scan_bytes <= 0:
        raise Error("invalid query or budget")
    known = {(v.version, p) for v in views for p in v.files}
    if not grants <= known or len({v.version for v in views}) != len(views):
        raise Error("unknown grant or duplicate version")
    snippets, skipped = [], []
    searched = used = scanned = matches = 0
    for view in views:
        for path, item in view.files.items():
            if (view.version, path) not in grants:
                continue
            if item["bytes"] > scan_bytes - scanned:
                skipped.append({"version": view.version, "path": path, "reason": "scan_budget"})
                continue
            scanned += item["bytes"]
            data = view.read(path)
            try:
                if data is None:
                    raise ValueError()
                data.decode("utf-8", errors="strict")
            except (UnicodeDecodeError, ValueError):
                skipped.append({"version": view.version, "path": path, "reason": "unsupported_or_file_budget"})
                continue
            searched += 1
            offset = data.find(needle)
            if offset < 0:
                continue
            matches += 1
            # Start at the exact UTF-8 query boundary, trim only a partial tail.
            fragment = data[offset:offset + min(512, context_bytes - used)].decode("utf-8", errors="ignore")
            encoded = fragment.encode("utf-8")
            if len(encoded) < len(needle):
                continue
            used += len(encoded)
            snippets.append({"version": view.version, "path": path, "sha256": item["sha256"],
                             "byte_offset": offset, "byte_length": len(encoded), "text": fragment})
    return {"format": FORMAT, "status": "FOUND" if matches else ("INCOMPLETE" if skipped or not grants else "PROVEN_EMPTY"),
            "scope": "explicitly_granted_versions_exact_utf8_only", "searched_files": searched,
            "skipped": skipped, "coverage_complete": bool(grants) and not skipped,
            "matching_files": matches, "context_truncated": matches > len(snippets),
            "context_bytes": used, "snippets": snippets,
            "content_is_untrusted_data_not_instructions": True,
            "network_used": False, "model_invoked": False}
