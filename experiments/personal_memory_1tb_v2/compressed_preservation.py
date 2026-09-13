"""Experimental post-selection backend for the existing V1 host.

Does not change frozen discovery or replace full Vault search data. Precomp
execution retains the earlier experimental decoder's resource/security limits.
Not approved for hostile archives, phones, or production deployment.
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import local_memory_bridge as bridge
import verified_hybrid_archive as base
import verified_reversible_precompression_archive as precomp


class CompressedPreservation:
    def __init__(self, root, receipt_sha256, precompressor=None, binary_sha256=None):
        self.view = bridge.ArchiveView(Path(root), receipt_sha256, max_file_bytes=64 * 1024 * 1024)
        self.root = self.view.root
        self.precompressor = Path(precompressor).resolve() if precompressor else None
        self.binary_sha256 = binary_sha256
        # Only the host may supply the binary and its independently retained pin.
        if self.precompressor:
            precomp.verify_precompressor(self.precompressor, binary_sha256 or "")

    def identity(self):
        return {"format": "GLYPH_COMPRESSED_PRESERVATION_BACKEND_V1",
                "receipt_sha256": self.view.version,
                "precompressor_sha256": self.binary_sha256,
                "experimental": True}

    def read_verified(self, path, size, expected):
        item = self.view.files.get(path)
        if item is None or item["bytes"] != size or item["sha256"] != expected:
            raise base.ArchiveError("selected V1 identity not present in compressed archive")
        obj = self.view.objects[expected]
        if obj["codec"] in ("precomp-cn", "precomp-cn+xz-9"):
            if not self.precompressor:
                raise base.ArchiveError("selected object needs pinned Precomp")
            if size > self.view.limit or obj["stored_bytes"] > self.view.limit:
                raise base.ArchiveError("experimental per-file budget exceeded")
            precomp.verify_precompressor(self.precompressor, self.binary_sha256)
            payload = bridge.read_regular(self.root, obj["path"], self.view.limit)
            if len(payload) != obj["stored_bytes"] or base.sha256_bytes(payload) != obj["stored_sha256"]:
                raise base.ArchiveError("compressed object corruption")
            data = precomp.decode(obj["codec"], payload, obj.get("suffix", ""), self.precompressor)
        else:
            data = self.view.read(path)
        if data is None or len(data) != size or base.sha256_bytes(data) != expected:
            raise base.ArchiveError("selected object decode/hash mismatch")
        return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("vault", "trust-root", "phase-a", "selection", "output", "archive", "archive-sha256"):
        parser.add_argument("--" + name, required=True)
    for name in ("pin", "pin-sha256", "precompressor", "precompressor-sha256"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    runtime_path = Path(__file__).resolve().parents[2] / "deploy/glyph-v1/portable_runtime.py"
    spec = importlib.util.spec_from_file_location("glyph_portable_v1", runtime_path)
    runtime = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runtime)
    backend = CompressedPreservation(args.archive, args.archive_sha256, args.precompressor, args.precompressor_sha256)
    runtime.run_materialization(args, preservation=backend)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(json.dumps({"status": "FAIL_CLOSED", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)
