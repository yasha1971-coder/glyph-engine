#!/usr/bin/env python3
"""GLYPH CORE-L8: digest-pinned source materialization research gate.

This verifier stages private copies of *all* mandatory RLB2/RLR2/EF data,
validates them against a separately trusted root SHA-256, checks the native
decoder executable digest, then executes candidate restoration using only
the staged retained state. It independently hashes the reconstructed bytes
BEFORE publishing the output under a new name with os.link (no overwrite).

Trust scope: an externally transported root hash is assumed authentic.
The caller-controlled --root-sha256 is NOT itself a signature, a witness,
a freshness proof, or authoritative P/R/O evidence. This implementation
does not claim multi-document identity, non-membership proof, ACLs, or
durable atomic publication of file+receipt.
"""
import argparse
import hashlib
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path

SCHEMA = "GLYPH_CORE_L8_PINNED_SOURCE_MATERIALIZATION_V0"
ORIGINAL = "c33cb493bc75f13ee4a904fabb190a29a196c94b"
HEX = frozenset("0123456789abcdef")
MAX_ROOT = 8192
MAX_FILE = 1 << 34  # 16 GiB research cap, not an appropriate 1TB model
COPY_CHUNK = 1024 * 1024

class Untrusted(ValueError):
    pass

def demand(condition, message):
    if not condition:
        raise Untrusted(message)

def digest(data):
    return hashlib.sha256(data).hexdigest()

def digest_file(path):
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for part in iter(lambda: handle.read(COPY_CHUNK), b""):
            h.update(part)
    return h.hexdigest()

def valid_sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in HEX for c in value)

def positive_size(value):
    return type(value) is int and 0 < value <= MAX_FILE

def pairs_no_duplicate(pairs):
    found = {}
    for key, value in pairs:
        demand(key not in found, "duplicate JSON field")
        found[key] = value
    return found

def parse_manifest(raw, external_root_sha):
    demand(valid_sha(external_root_sha), "root pin must be lowercase sha256")
    demand(0 < len(raw) <= MAX_ROOT, "manifest size admission")
    demand(digest(raw) == external_root_sha, "external root identity mismatch")
    try:
        d = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs_no_duplicate,
                       parse_constant=lambda x: (_ for _ in ()).throw(Untrusted("nonfinite JSON")))
    except (ValueError, UnicodeError) as exc:
        raise Untrusted("manifest JSON invalid") from exc
    demand(type(d) is dict and set(d) ==
           {"schema", "frozen_commit", "source", "artifacts", "decoder"},
           "manifest top-level schema mismatch")
    demand(d["schema"] == SCHEMA and d["frozen_commit"] == ORIGINAL,
           "manifest frozen source/profile version mismatch")
    src = d["source"]
    demand(type(src) is dict and set(src) == {"bytes", "sha256"} and
           positive_size(src["bytes"]) and valid_sha(src["sha256"]),
           "source identity invalid")
    comps = d["artifacts"]
    demand(type(comps) is dict and set(comps) == {"rlb2", "rlr2", "ef"},
           "required search state incomplete")
    for key, value in comps.items():
        demand(type(value) is dict and set(value) == {"bytes", "sha256"}
               and positive_size(value["bytes"]) and valid_sha(value["sha256"]),
               "component "+key+" identity invalid")
    dec = d["decoder"]
    demand(type(dec) is dict and set(dec) == {"sha256"} and
           valid_sha(dec["sha256"]), "decoder identity invalid")
    return d

def pinned_input(source, destination, record):
    """No symlinks; verify a PRIVATE copied snapshot used by the decoder."""
    source = Path(source)
    info = source.lstat()
    demand(stat.S_ISREG(info.st_mode), "not regular pinned input: "+str(source))
    demand(info.st_size == record["bytes"], "input source size mismatch")
    # O_NOFOLLOW protects the original leaf; a privileged hostile ancestor is
    # outside the threat model. Copy to a private mode-0700 dir before verifying.
    fd = os.open(source, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    try:
        actual = os.fstat(fd)
        demand(stat.S_ISREG(actual.st_mode) and actual.st_size == record["bytes"],
               "source changed between lstat/open")
        h = hashlib.sha256()
        count = 0
        with destination.open("xb") as dest:
            while True:
                data = os.read(fd, COPY_CHUNK)
                if not data:
                    break
                count += len(data)
                demand(count <= record["bytes"], "source grew during snapshot")
                dest.write(data)
                h.update(data)
        demand(count == record["bytes"] and h.hexdigest() == record["sha256"],
               "mandatory snapshot size/hash mismatch")
    finally:
        os.close(fd)

def verify_candidate(path, record):
    info = path.lstat()
    demand(stat.S_ISREG(info.st_mode) and info.st_size == record["bytes"],
           "decoded candidate geometry mismatch")
    demand(digest_file(path) == record["sha256"], "decoded source SHA mismatch")

def materialize(args):
    output = Path(args.output).absolute()
    demand(output.parent.is_dir(), "output parent directory missing")
    demand(not os.path.lexists(output), "output exists; no overwrite allowed")
    root_path = Path(args.manifest)
    root_stat = root_path.lstat()
    demand(stat.S_ISREG(root_stat.st_mode) and root_stat.st_size <= MAX_ROOT,
           "root manifest must be bounded regular file")
    raw = root_path.read_bytes()
    manifest = parse_manifest(raw, args.root_sha256)
    decoder = Path(args.decoder)
    dec_info = decoder.lstat()
    demand(stat.S_ISREG(dec_info.st_mode) and os.access(decoder, os.X_OK),
           "decoder must be executable regular file")
    demand(1 <= args.threads <= 16 and args.mode in ("rank", "direct"),
           "unsupported native decoder parameters")
    # Snapshot into private storage before launching. Also snapshot and
    # validate executable BYTES to eliminate the hash-then-exec path race.
    # Temporary copies are verification work and require separate disk budget.
    with tempfile.TemporaryDirectory(prefix=".glyph-l8-private-",
                                     dir=str(output.parent)) as private:
        stage = Path(private)
        os.chmod(stage, 0o700)
        for key, src in [("rlb2", args.rlb2), ("rlr2", args.rlr2), ("ef", args.ef)]:
            pinned_input(src, stage / (key+".snapshot"), manifest["artifacts"][key])
        executable_copy = stage / "decoder.snapshot"
        decoder_record = {
            "bytes": dec_info.st_size,
            "sha256": manifest["decoder"]["sha256"],
        }
        pinned_input(decoder, executable_copy, decoder_record)
        os.chmod(executable_copy, 0o700)
        candidate = stage / "candidate.source"
        command = [str(executable_copy), str(stage / "rlb2.snapshot"),
                   str(stage / "ef.snapshot"), str(candidate),
                   str(args.threads), args.mode]
        proc = subprocess.run(command, stdin=subprocess.DEVNULL,
                              capture_output=True, text=True, timeout=1200,
                              check=False)
        demand(proc.returncode == 0,
               "decoder failed closed (exit="+str(proc.returncode)+"): "+proc.stderr[-500:])
        verify_candidate(candidate, manifest["source"])
        # Publisher is still not a P/R/O signer/currentness witness.
        # Hardlink fails when output already exists; no replace/overwrite.
        os.link(candidate, output)
        try:
            fd = os.open(output.parent, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
        except OSError:
            # De-publicize if directory sync failed. Crash consistency/atomic
            # root+receipt publication is not guaranteed by this prototype.
            output.unlink(missing_ok=True)
            raise
        return {
            "status": "CORE_L8_PINNED_SHA256_MATERIALIZATION_PASS_RESEARCH",
            "manifest_sha256": args.root_sha256,
            "source_bytes": manifest["source"]["bytes"],
            "source_sha256": manifest["source"]["sha256"],
            "rlb2_bytes": manifest["artifacts"]["rlb2"]["bytes"],
            "rlr2_bytes": manifest["artifacts"]["rlr2"]["bytes"],
            "ef_bytes": manifest["artifacts"]["ef"]["bytes"],
            "snapshot_inputs_verified": ["rlb2", "rlr2", "ef"],
            "verified_source_before_publication": True,
            "decoder_sha256": manifest["decoder"]["sha256"],
            "private_decoder_snapshot_verified": True,
            "frozen_commit": ORIGINAL,
            "authority": "caller-provided external root SHA, not signature or TUF witness",
            "limitation": "no P/R/O root authentication, no anti-rollback, no proof of nonmembership"
        }

def main():
    parser = argparse.ArgumentParser()
    for name in ("manifest", "root-sha256", "rlb2", "rlr2", "ef",
                 "decoder", "output"):
        parser.add_argument("--"+name, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--mode", choices=["rank", "direct"], default="rank")
    args = parser.parse_args()
    try:
        print(json.dumps(materialize(args), sort_keys=True), flush=True)
        return 0
    except Exception as exc:
        # Untrusted is always a nonzero exit. Avoid returning unverified bytes.
        print(json.dumps({"status": "UNTRUSTED",
                          "error_class": type(exc).__name__,
                          "error": str(exc)[:300]}, sort_keys=True),
              file=sys.stderr, flush=True)
        return 3

if __name__ == "__main__":
    sys.exit(main())
