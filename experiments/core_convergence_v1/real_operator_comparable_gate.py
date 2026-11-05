#!/usr/bin/env python3
"""Real GLYPH operator Composition V1 vs external CRoaring + raw-byte oracle.

Research-only cross-check. Loads an UNMODIFIED pinned Composition Reference checkout,
builds REAL C++ SA/BWT/FM operator indexes, independently verifies their root,
original full result and filtered view in a fresh subprocess. No production field
API, Index Forest, cryptographic nonmembership or root-currentness claim.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

REFERENCE_SHA = "81628f843dc8aa633c65594a66d2386bfc3cb5c2"
PREFIX = b"GLYPH_RESEARCH_FIELD_FILTER_V0\x00"
PIN_PREFIX = b"GLYPH_RESEARCH_FIELD_PIN_V0\x00"
QUERIES = [b"shared", b"dup", b"\x00\xff", b"alpha LEFT",
           b"not-present", b"\x00\xffshared", b"a", b"EDGE-A"]
LAYOUTS = {"a": ((0, 3), (3, 6), (6, 9)),
           "b": ((0, 2), (2, 6), (6, 9))}


def sha(b):
    return hashlib.sha256(b).hexdigest()


def canonical(obj):
    return (json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n").encode()


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical(obj))


def read(path):
    raw = path.read_bytes()
    obj = json.loads(raw.decode())
    require(canonical(obj) == raw, "non-canonical JSON")
    return obj


def require(ok, msg):
    if not ok:
        raise ValueError(msg)


def load_module(filename, name):
    spec = importlib.util.spec_from_file_location(name, str(filename))
    require(spec and spec.loader, "missing module loader")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def libraries(checkout):
    tools = checkout / "tools"
    require(tools.is_dir(), "missing composition checkout")
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    return (load_module(tools / "check_composition_reference_v1.py", "glyph_composition_pinned"),
            load_module(tools / "replay_composition_reference_v1.py", "glyph_replay_pinned"))


def verify_checkout(checkout):
    head = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    require(head == REFERENCE_SHA, "wrong composition reference commit")
    return head


def metadata(documents):
    return {
        "format": "GLYPH_RESEARCH_FIELD_METADATA_V0",
        "generation": "synthetic-gen-1",
        "documents": [
            {"doc_id": i, "relative_path_hex": d.path.encode().hex(),
             "sha256": sha(d.data), "field_class": "public" if i % 2 == 0 else "private"}
            for i, d in enumerate(documents)
        ]
    }


def make_mask(meta, root):
    ids = [d["doc_id"] for d in meta["documents"] if d["field_class"] == "public"]
    core = {"format": "GLYPH_RESEARCH_FIELD_MASK_V0",
            "generation": meta["generation"],
            "runtime_corpus_id": root.corpus_id,
            "source_manifest_id": root.source_manifest_id,
            "metadata_sha256": sha(canonical(meta)),
            "field": "class", "value": "public", "document_ids": ids}
    return dict(core, mask_id=sha(PREFIX + canonical(core)))


def make_pin(root, mask):
    core = {"format": "GLYPH_RESEARCH_FIELD_PIN_V0",
            "composition_root_id": root.composition_root_id,
            "runtime_corpus_id": root.corpus_id,
            "source_manifest_id": root.source_manifest_id,
            "generation": mask["generation"], "mask_id": mask["mask_id"]}
    return dict(core, pin_id=sha(PIN_PREFIX + canonical(core)))


def filtered_receipt(core_result, mask, limit):
    # Count is based on FULL coordinate coverage, never on bounded input.
    eligible = set(mask["document_ids"])
    coord = [p for p in core_result["coordinates"] if p[0] in eligible]
    returned = coord if limit is None else coord[:limit]
    fields = {"format": "GLYPH_RESEARCH_FILTERED_RESULT_V0",
              "composition_result_id": core_result["composition_result_id"],
              "composition_root_id": core_result["composition_root_id"],
              "runtime_corpus_id": core_result["runtime_corpus_id"],
              "source_manifest_id": core_result["source_manifest_id"],
              "mask_id": mask["mask_id"], "query_hex": core_result["query_hex"],
              "max_offsets": limit, "match_count": len(coord),
              "returned_count": len(returned), "coordinates": returned,
              "offsets_complete": len(coord) == len(returned),
              "status": "PROVEN_EMPTY" if len(coord) == 0 else "FOUND"}
    return dict(fields, receipt_id=sha(PREFIX + canonical(fields)))


def build(ref, work):
    mod, _ = libraries(ref)
    mod.ensure_binaries() if hasattr(mod, "ensure_binaries") else None
    docs = mod.fixture_documents()
    cid = mod.runtime_corpus_id(docs)
    sid = mod.source_manifest_id(docs)
    meta = metadata(docs)
    save(work / "metadata.json", meta)
    results = {}
    for label, layout in LAYOUTS.items():
        root = mod.build_root(work, "real-field-" + label, layout, docs, cid, sid)
        mask = make_mask(meta, root)
        pin = make_pin(root, mask)
        area = work / label
        area.mkdir()
        save(area / "mask.json", mask)
        save(area / "pin.json", pin)
        blocks = [str(b.corpus) for b in root.blocks]
        queries = []
        for i, pattern in enumerate(QUERIES):
            original = mod.run_full_results(root, pattern, list(range(len(root.blocks))))
            unfiltered = mod.compose_full(root, pattern, original, None)
            mod.validate_oracle(root, docs, pattern, None, unfiltered)
            # The independent verifier, not the builder, is the authority for acceptance.
            full_path = area / f"core-{i:02d}.json"
            save(full_path, unfiltered)
            limits = (None, 0, 2) if i in (0, 1) else (None,)
            for limit in limits:
                result = filtered_receipt(unfiltered, mask, limit)
                save(area / f"field-{i:02d}-{'all' if limit is None else limit}.json", result)
            queries.append(pattern.hex())
        save(area / "manifest.json", {
            "root_path": str(work / f"real-field-{label}-composition-root-v1.json"),
            "blocks": blocks, "queries": queries, "pin_sha256": sha(canonical(pin))
        })
        results[label] = {"root": root.composition_root_id, "mask": mask["mask_id"],
                          "pin_sha256": sha(canonical(pin))}
    require(results["a"]["root"] != results["b"]["root"], "partition root must differ")
    require(results["a"]["mask"] == results["b"]["mask"], "logical mask should be layout invariant")
    return results


def independent_verify(ref, work, label, expected_pin_sha, drop_block=False):
    _, replay = libraries(ref)
    area = work / label
    manifest = read(area / "manifest.json")
    mask = read(area / "mask.json")
    pin = read(area / "pin.json")
    meta = read(work / "metadata.json")
    require(sha(canonical(pin)) == expected_pin_sha, "out-of-band pin changed")
    pin_data = dict(pin); pin_data.pop("pin_id", None)
    require(pin["pin_id"] == sha(PIN_PREFIX + canonical(pin_data)), "pin digest mismatch")
    mask_data = dict(mask); mask_data.pop("mask_id", None)
    require(mask["mask_id"] == sha(PREFIX + canonical(mask_data)), "mask digest mismatch")
    require(mask["metadata_sha256"] == sha(canonical(meta)), "field metadata commitment mismatch")
    require(mask["field"] == "class" and mask["value"] == "public", "unsupported predicate")
    require(meta["generation"] == mask["generation"] == pin["generation"], "stale generation")
    roots = [Path(p) for p in manifest["blocks"]]
    if drop_block:
        roots = roots[:-1]
    root = replay.verify_root(Path(manifest["root_path"]), roots)
    require(root.composition_root_id == pin["composition_root_id"], "untrusted composition root")
    require(root.runtime_corpus_id == pin["runtime_corpus_id"] == mask["runtime_corpus_id"],
            "untrusted corpus identity")
    require(root.source_manifest_id == pin["source_manifest_id"] == mask["source_manifest_id"],
            "untrusted source identity")
    require(mask["mask_id"] == pin["mask_id"], "untrusted mask identity")
    require(len(meta["documents"]) == root.document_count, "metadata geometry mismatch")
    recomputed = []
    for i, (d, md) in enumerate(zip(root.documents, meta["documents"])):
        require(md["doc_id"] == i and md["relative_path_hex"] == d.path_bytes.hex()
                and md["sha256"] == sha(d.payload), "metadata not bound to verified bytes")
        require(md["field_class"] in ("public", "private"), "bad field label")
        if md["field_class"] == mask["value"]:
            recomputed.append(i)
    require(mask["document_ids"] == recomputed, "wrong bitmap membership")
    from pyroaring import BitMap
    roaring = BitMap(recomputed)
    require(list(roaring) == recomputed, "CRoaring bitset mismatch")
    semantic = []
    for i, pattern_hex in enumerate(manifest["queries"]):
        pattern = bytes.fromhex(pattern_hex)
        unfiltered_path = area / f"core-{i:02d}.json"
        verified = replay.verify_result(root, unfiltered_path)
        require(verified["ok"] is True
                and verified["independent_source_oracle_replayed"] is True,
                "core verifier did not independently replay byte oracle")
        core = read(unfiltered_path)
        coords = replay.oracle_coordinates(root.documents, pattern)
        require(core["coordinates"] == coords, "raw oracle differs from C++ operator")
        present_doc_ids = BitMap([p[0] for p in coords])
        allowed_docs = present_doc_ids & roaring
        require(set(allowed_docs) == {p[0] for p in coords if p[0] in recomputed},
                "external CRoaring intersection disagreement")
        expected_full = [p for p in coords if p[0] in allowed_docs]
        for limit in ((None, 0, 2) if i in (0, 1) else (None,)):
            suffix = "all" if limit is None else str(limit)
            field = read(area / f"field-{i:02d}-{suffix}.json")
            pre = dict(field); claimed = pre.pop("receipt_id", None)
            require(claimed == sha(PREFIX + canonical(pre)), "filtered proof hash mismatch")
            expected = expected_full if limit is None else expected_full[:limit]
            require(field["composition_result_id"] == core["composition_result_id"]
                    and field["composition_root_id"] == root.composition_root_id
                    and field["runtime_corpus_id"] == root.runtime_corpus_id
                    and field["source_manifest_id"] == root.source_manifest_id
                    and field["mask_id"] == mask["mask_id"]
                    and field["query_hex"] == pattern_hex
                    and field["max_offsets"] == limit, "filtered proof binding mismatch")
            require(field["match_count"] == len(expected_full)
                    and field["coordinates"] == expected
                    and field["returned_count"] == len(expected)
                    and field["offsets_complete"] == (len(expected) == len(expected_full))
                    and field["status"] == ("PROVEN_EMPTY" if not expected_full else "FOUND"),
                    "filtered claim fails independent byte/CRoaring oracle")
            semantic.append((pattern_hex, limit, len(expected_full), expected))
    return {"verified_queries": len(manifest["queries"]),
            "filtered_claims": len(semantic),
            "semantic_sha256": sha(canonical(semantic)),
            "root": root.composition_root_id,
            "corpus_id": root.runtime_corpus_id,
            "mask_sha256": mask["mask_id"],
            "external_bitmap_implementation": "pyroaring CRoaring"}


def python_verify(ref, work, label, pin_sha, *, drop=False):
    cmd = [sys.executable, "-I", str(Path(__file__).resolve()),
           "--reference", str(ref), "--workdir", str(work), "--verify-only",
           "--partition", label, "--trusted-pin-sha256", pin_sha]
    if drop:
        cmd.append("--drop-block")
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(cmd, text=True, capture_output=True, timeout=300, env=env)


def expect_rejected(ref, work, label, pin, target=None, edit=None, *, drop=False):
    if target:
        original = target.read_bytes()
        target.write_bytes(edit(original))
    try:
        run = python_verify(ref, work, label, pin, drop=drop)
        require(run.returncode != 0, "mutated proof accepted")
    finally:
        if target:
            target.write_bytes(original)
    return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--workdir", type=Path, required=True)
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--partition", choices=("a", "b"))
    p.add_argument("--trusted-pin-sha256")
    p.add_argument("--drop-block", action="store_true")
    args = p.parse_args()
    ref = args.reference.resolve()
    verify_checkout(ref)
    if args.verify_only:
        require(args.partition and args.trusted_pin_sha256, "missing verification pin")
        result = independent_verify(ref, args.workdir.resolve(), args.partition,
                                    args.trusted_pin_sha256, args.drop_block)
        print(json.dumps({"status": "INDEPENDENT_REAL_OPERATOR_FIELD_REPLAY_OK", **result},
                         sort_keys=True))
        return
    work = args.workdir.resolve()
    require(not work.exists(), "refuse to overwrite existing output")
    work.mkdir(parents=True)
    built = build(ref, work)
    replayed = {}
    for layout in LAYOUTS:
        r = python_verify(ref, work, layout, built[layout]["pin_sha256"])
        require(r.returncode == 0, "independent process failed: " + r.stderr[-1500:])
        replayed[layout] = json.loads(r.stdout)
    require(replayed["a"]["semantic_sha256"] == replayed["b"]["semantic_sha256"],
            "repartition changed count/locate/field result")
    hostile = {}
    pin = built["a"]["pin_sha256"]
    hostile["missing_block"] = expect_rejected(ref, work, "a", pin, drop=True)
    hostile["stale_pin"] = expect_rejected(ref, work, "a", "0" * 64)
    path = work / "a" / "mask.json"
    hostile["stale_mask"] = expect_rejected(
        ref, work, "a", pin, path, lambda b: b.replace(b'"generation":"synthetic-gen-1"',
                                                       b'"generation":"synthetic-gen-0"'))
    path = work / "real-field-a-composition-root-v1.json"
    hostile["root_tamper"] = expect_rejected(ref, work, "a", pin, path, lambda b: b + b" ")
    path = work / "a" / "field-00-all.json"
    hostile["false_empty_rehashed"] = expect_rejected(
        ref, work, "a", pin, path,
        lambda b: forge_false_empty(b))
    summary = {
        "schema": "GLYPH_REAL_OPERATOR_FIELD_REFERENCE_V0",
        "status": "SCOPED_REAL_OPERATOR_COMPARABLE_PASS_NOT_PRODUCTION_FOREST",
        "composition_ref_commit": REFERENCE_SHA,
        "reference": built, "replay": replayed, "mutations_rejected": hostile,
        "limits": ["Field mask is additive research adapter, NOT operator schema",
                   "External pin supplied by experimental driver, not a trusted remote witness",
                   "CRoaring is comparison oracle, not an authenticated field index",
                   "No Index Forest runtime or O1-O6 standalone portable field proof",
                   "No external r-index/SDSL performance comparison yet",
                   "No currentness, production ACL, or global cryptographic absence proof"]
    }
    save(work / "result.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


def forge_false_empty(raw):
    result = json.loads(raw.decode())
    result.update(match_count=0, returned_count=0, coordinates=[],
                  offsets_complete=True, status="PROVEN_EMPTY")
    result.pop("receipt_id")
    result["receipt_id"] = sha(PREFIX + canonical(result))
    return canonical(result)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print("CORE_COMPARABLE_REJECTED: " + str(error), file=sys.stderr)
        sys.exit(2)
