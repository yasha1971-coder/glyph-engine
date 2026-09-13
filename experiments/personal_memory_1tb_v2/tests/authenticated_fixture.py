"""Tiny real on-disk RLB3X/LOC2/TRUST2 fixture, not a production builder.

Suffix oracle uses sentinel -1 ordered before all bytes. No host functions are
mocked. One block/page keeps geometry inspectable; max input 4096 bytes.
"""
import hashlib
import json
import lzma
import struct
from pathlib import Path

H = lambda b: hashlib.sha256(b).digest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")
    return path.read_bytes()


def make_fixture(root, frozen, files):
    vault, trust = root / "vault", root / "trust"
    seg, side = vault / "segments/01", trust / "segments/01"
    seg.mkdir(parents=True)
    side.mkdir(parents=True)
    corpus = b"".join(files.values())
    assert 0 < len(corpus) <= 4096
    symbols = list(corpus) + [-1]
    sa = sorted(range(len(symbols)), key=lambda i: symbols[i:])
    bwt = [symbols[i-1] if i else -1 for i in sa]
    runs = []
    for symbol in bwt:
        symbol = 256 if symbol == -1 else symbol
        if runs and runs[-1][0] == symbol:
            runs[-1][1] += 1
        else:
            runs.append([symbol, 1])
    encoded = bytearray()
    for s, n in runs:
        encoded.extend(bytes((0, 1)) if s == 256 else bytes((0, 0)) if s == 0 else bytes((s,)))
        while n >= 128:
            encoded.append((n & 127) | 128)
            n >>= 7
        encoded.append(n)
    compressed = lzma.compress(encoded)
    rh = struct.pack("<8sIQQIIQ", b"RLB3X001", 1, len(symbols), len(runs), len(runs), 0, 1)
    directory = struct.pack("<QQQQ", 0, 0, len(rh) + 32, len(compressed))
    rlb = rh + directory + compressed
    (seg / "bwt.rlb3x").write_bytes(rlb)
    lh = struct.pack("<4sIQIIQ", b"LOC2", 1, len(sa), 1, 4, len(sa))
    samples = struct.pack(f"<{len(sa)}I", *sa)
    loc = lh + samples
    (seg / "locate.loc2").write_bytes(loc)
    authloc = struct.pack("<8sIIQQII32s", b"ALOC0001", 1, len(sa), len(sa), len(sa), 1, 1, H(lh))
    authloc += struct.pack("<II32s", 0, len(sa), H(samples))
    (side / "loc.auth").write_bytes(authloc)
    counts = [bwt.count(s) for s in range(256)] + [1]
    rank = struct.pack("<8sIQQQIIIIQ32s32s", b"RAUTHS01", 1, len(sa), len(runs), 1,
                       len(runs), 0, 4, 1, 1, H(directory), H(authloc))
    rank += struct.pack("<257Q", *counts) + bytes(257 * 4) + H(compressed)
    (side / "rank.s1.auth").write_bytes(rank)
    objects, offset = [], 0
    for i, (path, data) in enumerate(files.items()):
        objects.append({"id": i, "path": path, "bytes": len(data), "offset": offset, "sha256": H(data).hex()})
        offset += len(data)
    raw_objects = write_json(seg / "objects.json", {"format": "GLYPH_PERSONAL_VAULT_V1_OBJECT_MAP", "objects": objects})
    manifest = {"segment_id": "01", "files": {
        name: {"name": filename, "bytes": len(data), "sha256": H(data).hex()}
        for name, filename, data in (("objects", "objects.json", raw_objects), ("rlb3x", "bwt.rlb3x", rlb), ("loc2", "locate.loc2", loc))}}
    mh = H(write_json(seg / "segment-manifest.json", manifest)).hex()
    root_record = {"format": "GLYPH_VAULT_ROOT_MANIFEST_V0", "segments": ["01"], "parent_root_name": None,
                   "segment_entries": [{"segment_id": "01", "segment_manifest_sha256": mh}]}
    root_hash = H(write_json(vault / "manifests/roots/1.json", root_record)).hex()
    sp = {"segment_id": "01", "segment_manifest_sha256": mh, "objects_bytes": len(raw_objects),
          "objects_sha256": H(raw_objects).hex(), "canonical_rlb_sha256": H(rlb).hex(), "rlb_bytes": len(rlb),
          "canonical_loc_sha256": H(loc).hex(), "loc_bytes": len(loc), "rank_sha256": H(rank).hex(),
          "authloc_sha256": H(authloc).hex(), "rlb_path": str(seg / "bwt.rlb3x"),
          "loc_path": str(seg / "locate.loc2"), "rank_path": str(side / "rank.s1.auth"),
          "authloc_path": str(side / "loc.auth")}
    pin = {"format": "GLYPH_LAPTOP_AUTH_READPATH_PIN_V2_MULTI", "root_name": "1.json", "root_sha256": root_hash,
           "segment_ids": ["01"], "segments": [sp], "query_reader": str(frozen / "query_authloc_all.py"),
           "query_reader_sha256": H((frozen / "query_authloc_all.py").read_bytes()).hex()}
    pin_path = root / "pin.json"
    write_json(pin_path, pin)
    return vault, trust, pin_path
