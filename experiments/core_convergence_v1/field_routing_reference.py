"""Research-only field-mask overlay on Composition V1 semantics.

This is NOT an Index Forest runtime, operator artifact or new proof format.
It has no FM/RLBWT engine, network, source-file ingestion or access-control API.
It exercises semantics and independent fixture oracle only. Existing P/R/O/I0,
composition reference, frozen V1 and authentic external roots are untouched.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Optional, Sequence


class Rejected(ValueError):
    """Fail closed: no result, particularly no PROVEN_EMPTY, may be issued."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise Rejected(message)


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8") + b"\n"


def u64(n: int) -> bytes:
    require(type(n) is int and 0 <= n <= (1 << 64) - 1, "u64 geometry")
    return n.to_bytes(8, "big")


def h(raw: str) -> bytes:
    require(isinstance(raw, str) and len(raw) == 64
            and all(c in "0123456789abcdef" for c in raw), "invalid digest")
    return bytes.fromhex(raw)


@dataclass(frozen=True)
class Document:
    path: str
    data: bytes
    label: str


@dataclass(frozen=True)
class Block:
    ordinal: int
    start: int
    end: int
    runtime_index_id: str
    runtime_manifest_sha256: str


@dataclass(frozen=True)
class Snapshot:
    documents: tuple[Document, ...]
    blocks: tuple[Block, ...]
    corpus_id: str
    source_manifest_id: str
    root_id: str
    generation: str


@dataclass(frozen=True)
class FieldMask:
    bits: tuple[int, ...]
    corpus_id: str
    source_manifest_id: str
    generation: str
    field: str
    value: str
    metadata_sha256: str
    mask_id: str


@dataclass(frozen=True)
class ExternalPin:
    corpus_id: str
    source_manifest_id: str
    root_id: str
    generation: str
    mask_id: str


def runtime_id(docs: Sequence[Document]) -> str:
    pre = bytearray(b"GLYPH_BINARY_RUNTIME_CORPUS_IDENTITY_V1\x00")
    pre.extend(u64(len(docs)))
    for i, doc in enumerate(docs):
        pre.extend(u64(i) + u64(len(doc.data)) + bytes.fromhex(digest(doc.data)))
    return digest(bytes(pre))


def source_id(docs: Sequence[Document]) -> str:
    pre = bytearray(b"GLYPH_OPERATOR_CORPUS_MANIFEST_V1\x00")
    pre.extend(u64(len(docs)))
    for i, doc in enumerate(docs):
        name = doc.path.encode("utf-8")
        pre.extend(u64(i) + u64(len(name)) + name)
        pre.extend(u64(len(doc.data)) + bytes.fromhex(digest(doc.data)))
    return digest(bytes(pre))


def block_for(ordinal: int, start: int, end: int, docs: Sequence[Document]) -> Block:
    # Mock IDs are deliberately namespaced, and NOT real runtime_index_id artifacts.
    pre = bytearray(b"GLYPH_FIELD_MASK_FIXTURE_BLOCK_NOT_OPERATOR_V0\x00")
    pre.extend(u64(start) + u64(end))
    for doc in docs[start:end]:
        pre.extend(u64(len(doc.data)) + bytes.fromhex(digest(doc.data)))
        name = doc.path.encode("utf-8")
        pre.extend(u64(len(name)) + name)
    index_id = digest(bytes(pre))
    manifest_id = digest(b"GLYPH_FIELD_MASK_FIXTURE_MANIFEST_V0\x00"
                         + h(index_id) + u64(ordinal))
    return Block(ordinal, start, end, index_id, manifest_id)


def root_id(corpus_id: str, source_manifest_id: str,
            blocks: Sequence[Block], count: int) -> str:
    # Same Composition V1 root-ID preimage, but this fixture's block hashes are mocks.
    pre = bytearray(b"GLYPH_COMPOSITION_ROOT_V1\x00")
    pre.extend(h(corpus_id) + h(source_manifest_id)
               + u64(count) + u64(len(blocks)))
    for i, block in enumerate(blocks):
        require(i == block.ordinal, "noncanonical ordinal")
        pre.extend(u64(i) + u64(block.end - block.start)
                   + h(block.runtime_index_id) + h(block.runtime_manifest_sha256))
    return digest(bytes(pre))


def make_snapshot(documents: Sequence[Document],
                  layout: Sequence[tuple[int, int]], generation: str = "gen-1") -> Snapshot:
    docs = tuple(documents)
    blocks = tuple(block_for(i, a, b, docs) for i, (a, b) in enumerate(layout))
    cid, sid = runtime_id(docs), source_id(docs)
    return Snapshot(docs, blocks, cid, sid, root_id(cid, sid, blocks, len(docs)), generation)


def metadata_digest(docs: Sequence[Document]) -> str:
    # Metadata is NOT covered by the existing runtime corpus_id, hence separate identity.
    return digest(b"GLYPH_FIELD_MASK_FIXTURE_METADATA_V0\x00"
                  + canonical([{"doc_id": i, "label": d.label}
                               for i, d in enumerate(docs)]))


def make_mask(snapshot: Snapshot, value: str) -> FieldMask:
    require(value in ("public", "private"), "unsupported field value")
    bits = tuple(int(d.label == value) for d in snapshot.documents)
    meta = metadata_digest(snapshot.documents)
    data = {"version": "GLYPH_FIELD_MASK_RESEARCH_ONLY_V0",
            "corpus_id": snapshot.corpus_id,
            "source_manifest_id": snapshot.source_manifest_id,
            "generation": snapshot.generation, "field": "class", "value": value,
            "metadata_sha256": meta, "bits": list(bits)}
    mid = digest(b"GLYPH_FIELD_MASK_FIXTURE_V0\x00" + canonical(data))
    return FieldMask(bits, snapshot.corpus_id, snapshot.source_manifest_id,
                     snapshot.generation, "class", value, meta, mid)


def make_pin(snapshot: Snapshot, mask: FieldMask) -> ExternalPin:
    return ExternalPin(snapshot.corpus_id, snapshot.source_manifest_id,
                       snapshot.root_id, snapshot.generation, mask.mask_id)


def preflight(snapshot: Snapshot, mask: FieldMask, pin: Optional[ExternalPin],
              available_ids: Sequence[str]) -> None:
    require(isinstance(pin, ExternalPin), "independently supplied pin required")
    docs, blocks = snapshot.documents, snapshot.blocks
    require(isinstance(docs, tuple) and len(docs) > 0, "no committed documents")
    require(isinstance(snapshot.generation, str) and snapshot.generation != "", "generation")
    paths = [d.path for d in docs]
    require(all(type(d.path) is str and d.path and type(d.data) is bytes
                and d.label in ("public", "private")
                and not d.path.startswith("/") and ".." not in d.path.split("/")
                for d in docs) and len(set(paths)) == len(paths), "invalid document or path")
    cid, sid = runtime_id(docs), source_id(docs)
    require(snapshot.corpus_id == cid and snapshot.source_manifest_id == sid,
            "logical corpus/source identity mismatch")
    require(len(blocks) >= 1, "no blocks")
    next_start = 0
    expected_ids = []
    for ordinal, block in enumerate(blocks):
        require(type(block.ordinal) is int and block.ordinal == ordinal,
                "wrong block ordinal")
        require(type(block.start) is int and type(block.end) is int
                and block.start == next_start and block.start < block.end <= len(docs),
                "incomplete, overlapping or reordered coverage")
        require(block == block_for(ordinal, block.start, block.end, docs),
                "block bytes/index/manifest binding mismatch")
        expected_ids.append(block.runtime_index_id)
        next_start = block.end
    require(next_start == len(docs) and len(set(expected_ids)) == len(expected_ids),
            "incomplete or duplicate blocks")
    require(snapshot.root_id == root_id(cid, sid, blocks, len(docs)), "root mismatch")
    require(type(available_ids) in (tuple, list)
            and len(available_ids) == len(expected_ids)
            and set(available_ids) == set(expected_ids), "missing/extra/duplicated block")
    require(mask.corpus_id == cid and mask.source_manifest_id == sid
            and mask.generation == snapshot.generation, "stale mask identity")
    require(mask.field == "class" and mask.value in ("public", "private"),
            "unsupported field profile")
    require(mask.metadata_sha256 == metadata_digest(docs), "stale field metadata")
    require(type(mask.bits) is tuple and len(mask.bits) == len(docs)
            and all(type(b) is int and b in (0, 1) for b in mask.bits),
            "wrong field mask geometry")
    require(mask.bits == tuple(int(d.label == mask.value) for d in docs),
            "field mask does not represent committed metadata")
    require(mask.mask_id == make_mask(snapshot, mask.value).mask_id, "mask hash mismatch")
    require(pin == make_pin(snapshot, mask), "external pin mismatch / possible rollback")


def find_all(data: bytes, pattern: bytes) -> list[int]:
    matches: list[int] = []
    pos = 0
    while True:
        hit = data.find(pattern, pos)
        if hit < 0:
            return matches
        matches.append(hit)
        pos = hit + 1  # overlapping exact matches remain visible


def execute(snapshot: Snapshot, mask: FieldMask, pin: Optional[ExternalPin],
            available_ids: Sequence[str], query: bytes, max_offsets: int = -1) -> dict:
    preflight(snapshot, mask, pin, available_ids)
    require(type(query) is bytes and bool(query), "empty or nonbinary query")
    require(type(max_offsets) is int and max_offsets >= -1, "invalid bound")
    coords: list[list[int]] = []
    for block in snapshot.blocks:
        for i in range(block.start, block.end):
            if mask.bits[i]:
                coords.extend([[i, p] for p in find_all(snapshot.documents[i].data, query)])
    coords.sort()
    count = len(coords)
    returned = coords if max_offsets == -1 else coords[:max_offsets]
    result = {
        "format": "GLYPH_FIELD_MASK_COMPOSITION_FIXTURE_RESULT_V0",
        "research_only": True, "corpus_id": snapshot.corpus_id,
        "source_manifest_id": snapshot.source_manifest_id, "root_id": snapshot.root_id,
        "generation": snapshot.generation, "mask_id": mask.mask_id,
        "query_hex": query.hex(), "match_count": count, "coordinates": returned,
        "returned_count": len(returned), "max_offsets": max_offsets,
        "offsets_complete": len(returned) == count,
        "expected_blocks": len(snapshot.blocks), "verified_blocks": len(snapshot.blocks),
        "status": "PROVEN_EMPTY" if count == 0 else "FOUND",
    }
    result["result_id"] = digest(b"GLYPH_FIELD_MASK_FIXTURE_RESULT_V0\x00" + canonical(result))
    return result


def independently_verify(snapshot: Snapshot, mask: FieldMask, pin: ExternalPin,
                         available_ids: Sequence[str], query: bytes, result: dict,
                         requested_max_offsets: int = -1) -> bool:
    preflight(snapshot, mask, pin, available_ids)
    require(type(result) is dict, "invalid result")
    rid = result.get("result_id")
    candidate = dict(result)
    candidate.pop("result_id", None)
    require(rid == digest(b"GLYPH_FIELD_MASK_FIXTURE_RESULT_V0\x00" + canonical(candidate)),
            "result receipt identity invalid")
    require(candidate.get("query_hex") == query.hex(), "wrong query binding")
    require(candidate.get("max_offsets") == requested_max_offsets,
            "unbound or changed max_offsets")
    # Independent oracle: naive byte windows, no use of find_all or per-block results.
    oracle: list[list[int]] = []
    for i, doc in enumerate(snapshot.documents):
        if doc.label == mask.value:
            for off in range(max(0, len(doc.data) - len(query) + 1)):
                if doc.data[off:off + len(query)] == query:
                    oracle.append([i, off])
    want = execute(snapshot, mask, pin, available_ids, query, requested_max_offsets)
    require(candidate == {k: v for k, v in want.items() if k != "result_id"},
            "incorrect claim or incomplete coverage")
    limit = candidate["max_offsets"]
    expected = oracle if limit == -1 else oracle[:limit]
    require(candidate["match_count"] == len(oracle)
            and candidate["coordinates"] == expected
            and (candidate["status"] == "PROVEN_EMPTY") == (len(oracle) == 0),
            "independent raw-byte oracle mismatch")
    return True


def fixture_documents() -> tuple[Document, ...]:
    return (
        Document("0.txt", b"ana banana", "public"),
        Document("1.empty", b"", "public"),
        Document("2.txt", b"alpha\n", "private"),
        Document("3.duplicate", b"ana banana", "private"),
        Document("4.bin", b"\x00a\xffana", "public"),
    )


def fixture_layouts() -> tuple[tuple[tuple[int, int], ...], ...]:
    return (((0, 2), (2, 3), (3, 5)),
            ((0, 1), (1, 4), (4, 5)))
