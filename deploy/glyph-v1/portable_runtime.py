#!/usr/bin/env python3
"""Portable host adapter for the frozen GLYPH V1 acceptance path.

The adapter does not edit the frozen resolver, reader, pin, release manifest, or
Vault.  Host-specific physical paths are expressed by a temporary derived pin.
The authenticated identities inside the canonical pin remain unchanged.
"""

import argparse
import datetime as dt
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path, PurePosixPath


HERE = Path(__file__).resolve().parent
PROFILE_PATH = HERE / "PROFILE.json"
FROZEN = HERE / "frozen"
REPRODUCTION = HERE / "reproduction"
HEX = frozenset("0123456789abcdef")


class Refusal(RuntimeError):
    """Fail-closed refusal caused by an unmet trust precondition."""


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open("r", encoding="utf-8") as source:
        return json.load(source)


def write_json_exclusive(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        json.dump(value, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")


def utc_now():
    return dt.datetime.now(dt.timezone.utc)


def stamp():
    return utc_now().strftime("%Y%m%d-%H%M%S-%f")


def verify_declared_files(section):
    profile = read_json(PROFILE_PATH)
    failures = []
    for relative, expected in profile[section].items():
        path = HERE / relative
        if not path.is_file():
            failures.append({"path": relative, "failure": "missing"})
            continue
        actual = sha256_file(path)
        if actual != expected:
            failures.append(
                {"path": relative, "failure": "sha256", "expected": expected, "actual": actual}
            )
    if failures:
        raise Refusal(json.dumps(failures, ensure_ascii=False, sort_keys=True))
    return len(profile[section])


def verify_profile(include_evidence=True):
    frozen_count = verify_declared_files("frozen_files")
    evidence_count = verify_declared_files("evidence_files") if include_evidence else 0
    return {
        "format": "GLYPH_V1_PORTABLE_PROFILE_CHECK_V1",
        "status": "GREEN",
        "frozen_files_verified": frozen_count,
        "evidence_files_verified": evidence_count,
        "frozen_files_mutated": False,
    }


def require_directory(path, label):
    path = Path(path).expanduser().resolve()
    if not path.is_dir():
        raise Refusal(f"{label} missing: {path}")
    return path


def canonical_pin(path=None, expected_sha256=None):
    pin_path = Path(path).expanduser().resolve() if path else FROZEN / "readpath-pin-v2.json"
    if not pin_path.is_file():
        raise Refusal(f"pin missing: {pin_path}")
    bundled_expected = read_json(PROFILE_PATH)["frozen_files"]["frozen/readpath-pin-v2.json"]
    if path and not expected_sha256:
        raise Refusal("a custom pin requires --pin-sha256")
    expected = expected_sha256 if path else bundled_expected
    if len(expected) != 64 or set(expected.lower()) > HEX:
        raise Refusal("invalid expected pin SHA-256")
    actual = sha256_file(pin_path)
    if actual != expected.lower():
        raise Refusal(f"pin SHA-256 mismatch: expected={expected.lower()} actual={actual}")
    pin = read_json(pin_path)
    if pin.get("format") != "GLYPH_LAPTOP_AUTH_READPATH_PIN_V2_MULTI":
        raise Refusal("unsupported pin format")
    return pin_path, pin


def is_within(path, root):
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def require_separate_output(output, protected):
    output = Path(output).expanduser().resolve()
    for root, label in protected:
        if is_within(output, root) or is_within(root, output):
            raise Refusal(f"output must remain separate from {label}: {root}")
    return output


def relocated_pin(pin, vault, trust_root, query_reader):
    """Return a derived locator pin while retaining all authenticated hashes."""
    derived = json.loads(json.dumps(pin))
    derived["query_reader"] = str(Path(query_reader).resolve())
    for segment in derived.get("segments", []):
        sid = str(segment.get("segment_id", ""))
        if not sid.isdigit() or sid not in derived.get("segment_ids", []):
            raise Refusal(f"invalid segment id in pin: {sid!r}")
        segment_dir = Path(vault) / "segments" / sid
        trust_dir = Path(trust_root) / "segments" / sid
        segment["rlb_path"] = str(segment_dir / "bwt.rlb3x")
        segment["loc_path"] = str(segment_dir / "locate.loc2")
        segment["rank_path"] = str(trust_dir / "rank.s1.auth")
        segment["authloc_path"] = str(trust_dir / "loc.auth")
    return derived


def external_paths(pin, vault, trust_root):
    missing = []
    roots = Path(vault) / "manifests" / "roots"
    required = [roots / pin["root_name"]]
    for sid in pin["segment_ids"]:
        segment = Path(vault) / "segments" / sid
        trust = Path(trust_root) / "segments" / sid
        required.extend(
            [
                segment / "segment-manifest.json",
                segment / "objects.json",
                segment / "bwt.rlb3x",
                segment / "locate.loc2",
                trust / "rank.s1.auth",
                trust / "loc.auth",
            ]
        )
    for path in required:
        if not path.is_file():
            missing.append(str(path))
    return required, missing


def prepare_compatibility_home(home, vault, trust_root, source_pin):
    """Build an ephemeral legacy-shaped tree without changing canonical state."""
    pilot = home / "GlyphPilot"
    release = pilot / "LI-V0-LAPTOP" / "release" / "GLYPH-LI-LAPTOP-RC1-FINAL"
    components = release / "components"
    trust = pilot / "VIKA-trust2-five"
    reader_dir = pilot / "VIKA-laptop-trust2"
    components.mkdir(parents=True)
    trust.mkdir(parents=True)
    reader_dir.mkdir(parents=True)

    shutil.copyfile(FROZEN / "RELEASE_MANIFEST.json", release / "RELEASE_MANIFEST.json")
    shutil.copyfile(FROZEN / "hybrid_resolver_v613.py", components / "hybrid_resolver_v613.py")
    shutil.copyfile(FROZEN / "glyph-trust2-multi.py", trust / "glyph-trust2-multi.py")
    shutil.copyfile(FROZEN / "query_authloc_all.py", reader_dir / "query_authloc_all.py")

    pin = relocated_pin(
        source_pin,
        vault,
        trust_root,
        reader_dir / "query_authloc_all.py",
    )
    with (trust / "readpath-pin-v2.json").open("x", encoding="utf-8") as output:
        json.dump(pin, output, ensure_ascii=False, indent=2, sort_keys=True)
        output.write("\n")

    vault_link = pilot / "VIKA_proekt-vault-patched-test"
    vault_link.symlink_to(Path(vault).resolve(), target_is_directory=True)
    return pilot, release


def run_phase_a(args):
    verify_profile(include_evidence=False)
    vault = require_directory(args.vault, "Vault")
    trust_root = require_directory(args.trust_root, "trust root")
    output = require_separate_output(
        args.output, ((vault, "canonical Vault"), (trust_root, "trust sidecars"))
    )
    output.mkdir(parents=True, exist_ok=True)
    _, pin = canonical_pin(args.pin, args.pin_sha256)
    _, missing = external_paths(pin, vault, trust_root)
    if missing:
        raise Refusal("required runtime paths missing: " + json.dumps(missing))

    before = set(output.glob("GLYPH_V1_LIVE_ACCEPTANCE_PHASE_A_*.json"))
    with tempfile.TemporaryDirectory(prefix="glyph-v1-home-") as temporary:
        home = Path(temporary)
        pilot, release = prepare_compatibility_home(home, vault, trust_root, pin)
        base = pilot / "LI-V0-LAPTOP"
        resolver = release / "components" / "hybrid_resolver_v613.py"
        environment = os.environ.copy()
        environment.update(
            {
                "HOME": str(home),
                "BASE": str(base),
                "REL": str(release),
                "RES": str(resolver),
                "RUN": str(output),
                "QUERY": args.query,
            }
        )
        completed = subprocess.run(
            [sys.executable, str(REPRODUCTION / "phase_a_history.py")], env=environment
        )
        if completed.returncode:
            raise Refusal(f"Phase A failed with status {completed.returncode}")

    created = set(output.glob("GLYPH_V1_LIVE_ACCEPTANCE_PHASE_A_*.json")) - before
    if len(created) != 1:
        raise Refusal(f"Phase A receipt cardinality: {len(created)}")
    receipt = created.pop()
    print(
        json.dumps(
            {
                "format": "GLYPH_V1_PORTABLE_PHASE_A_RESULT_V1",
                "receipt": str(receipt),
                "sha256": sha256_file(receipt),
                "next": "human must inspect candidates and run select",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def validate_phase_a(phase_a):
    required = {
        "human_selection_required": True,
        "payload_touched": False,
        "materialized": False,
    }
    for field, expected in required.items():
        if phase_a.get(field) is not expected:
            raise Refusal(f"Phase A boundary failed: {field}")
    profile = read_json(PROFILE_PATH)
    frozen = profile["frozen_files"]
    if phase_a.get("release_manifest_sha256") != frozen["frozen/RELEASE_MANIFEST.json"]:
        raise Refusal("Phase A release manifest identity")
    if phase_a.get("frozen_resolver_sha256") != frozen["frozen/hybrid_resolver_v613.py"]:
        raise Refusal("Phase A resolver identity")
    if phase_a.get("target_oracle_present") is not False:
        raise Refusal("Phase A oracle boundary")
    if phase_a.get("query_specific_bridge_added") is not False:
        raise Refusal("Phase A query-specific bridge boundary")
    shortlist = phase_a.get("shortlist")
    if not isinstance(shortlist, list) or not shortlist:
        raise Refusal("Phase A has no selectable candidates")
    return shortlist


def create_selection(phase_a_path, candidate_number, output_path):
    phase_a_path = Path(phase_a_path).expanduser().resolve()
    phase_a = read_json(phase_a_path)
    shortlist = validate_phase_a(phase_a)
    if candidate_number < 1 or candidate_number > len(shortlist):
        raise Refusal(f"candidate must be in 1..{len(shortlist)}")
    output_path = Path(output_path).expanduser().resolve()
    if output_path.parent != phase_a_path.parent:
        raise Refusal("selection receipt must be written beside its Phase A receipt")
    selected = shortlist[candidate_number - 1]
    receipt = {
        "format": "GLYPH_V1_HUMAN_SELECTION_V1",
        "timestamp_utc": utc_now().isoformat(),
        "authority": "explicit human choice",
        "phase_a_receipt": str(phase_a_path),
        "phase_a_sha256": sha256_file(phase_a_path),
        "candidate_number": candidate_number,
        "selected": selected,
        "payload_touched": False,
    }
    write_json_exclusive(output_path, receipt)
    return receipt


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise Refusal(f"cannot load module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def object_expected_sha(obj):
    for field in ("sha256", "sha256_hex", "content_sha256", "object_sha256", "hash"):
        value = obj.get(field)
        if isinstance(value, str):
            digest = value.lower().strip()
            if len(digest) == 64 and set(digest) <= HEX:
                return digest, field
    raise Refusal("authenticated object has no explicit SHA-256")


def verified_slice(corpus, offset, size, expected):
    with Path(corpus).open("rb") as source:
        source.seek(offset)
        data = source.read(size)
    if len(data) != size:
        raise Refusal(f"short preservation read: {len(data)} != {size}")
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise Refusal(f"object SHA-256 mismatch: expected={expected} actual={actual}")
    return data


def run_materialization(args):
    verify_profile(include_evidence=False)
    vault = require_directory(args.vault, "Vault")
    trust_root = require_directory(args.trust_root, "trust root")
    cache = require_directory(args.cache, "preservation cache")
    output = require_separate_output(
        args.output,
        ((vault, "canonical Vault"), (trust_root, "trust sidecars"), (cache, "derived cache")),
    )
    output.mkdir(parents=True, exist_ok=True)

    phase_a_path = Path(args.phase_a).expanduser().resolve()
    selection_path = Path(args.selection).expanduser().resolve()
    phase_a = read_json(phase_a_path)
    shortlist = validate_phase_a(phase_a)
    selection = read_json(selection_path)
    phase_a_sha = sha256_file(phase_a_path)
    if selection.get("format") != "GLYPH_V1_HUMAN_SELECTION_V1":
        raise Refusal("selection receipt format")
    if selection.get("phase_a_sha256") != phase_a_sha:
        raise Refusal("selection is not bound to this Phase A receipt")
    number = selection.get("candidate_number")
    if not isinstance(number, int) or not 1 <= number <= len(shortlist):
        raise Refusal("selection candidate number")
    selected = shortlist[number - 1]
    if selection.get("selected") != selected:
        raise Refusal("selected candidate content mismatch")
    if phase_a.get("payload_touched") is not False or phase_a.get("materialized") is not False:
        raise Refusal("Phase A payload boundary")

    _, source_pin = canonical_pin(args.pin, args.pin_sha256)
    pin = relocated_pin(source_pin, vault, trust_root, FROZEN / "query_authloc_all.py")
    hard = load_module("glyph_v1_hardened_reader", FROZEN / "glyph-trust2-multi.py")
    if hard.sha(FROZEN / "query_authloc_all.py") != pin["query_reader_sha256"]:
        raise Refusal("query reader authentication failed")
    root, _ = hard.authenticate_root_chain(vault, pin)
    by_sid = {item["segment_id"]: item for item in pin["segments"]}
    selected_objects = []
    selected_parent = selected["parent"]
    for sid in pin["segment_ids"]:
        objects, rlb, _, _, _ = hard.authenticate_segment(
            vault, root, by_sid[sid], pin["query_reader_sha256"]
        )
        for index, obj in enumerate(objects["objects"]):
            parent = str(PurePosixPath(str(obj["path"])).parent)
            if parent == selected_parent:
                selected_objects.append((sid, index, obj, rlb))
    if not selected_objects:
        raise Refusal("selected parent absent from authenticated object maps")

    needed = sorted({sid for sid, _, _, _ in selected_objects})
    missing_cache = [sid for sid in needed if not (cache / sid / "corpus.bin").is_file()]
    if missing_cache:
        print(
            json.dumps(
                {
                    "format": "GLYPH_V1_MATERIALIZATION_REFUSAL_V1",
                    "status": "FAIL_CLOSED",
                    "reason": "preservation cache missing",
                    "segments": missing_cache,
                    "payload_materialized": False,
                },
                sort_keys=True,
            )
        )
        raise SystemExit(20)

    stage = Path(tempfile.mkdtemp(prefix=".glyph-v1-materialize-", dir=output))
    verified = []
    try:
        for sid, index, obj, _ in selected_objects:
            expected, sha_field = object_expected_sha(obj)
            size = int(obj["bytes"])
            offset = int(obj["offset"])
            data = verified_slice(cache / sid / "corpus.bin", offset, size, expected)
            basename = PurePosixPath(str(obj["path"])).name or f"object-{index}"
            destination = stage / f"{sid}-{index:04d}-{basename}"
            with destination.open("xb") as materialized:
                materialized.write(data)
            verified.append(
                {
                    "segment_id": sid,
                    "object_index": index,
                    "canonical_path": obj["path"],
                    "bytes": size,
                    "offset": offset,
                    "sha_field": sha_field,
                    "expected_sha256": expected,
                    "materialized_sha256": sha256_file(destination),
                    "materialized_file": destination.name,
                }
            )
        final = output / f"materialized-{stamp()}"
        stage.rename(final)
    except BaseException:
        shutil.rmtree(stage, ignore_errors=True)
        raise

    for item in verified:
        item["materialized_path"] = str(final / item.pop("materialized_file"))
    receipt = {
        "format": "GLYPH_V1_PORTABLE_PHASE_B_V1",
        "timestamp_utc": utc_now().isoformat(),
        "phase_a_receipt": str(phase_a_path),
        "phase_a_sha256": phase_a_sha,
        "selection_receipt": str(selection_path),
        "selection_sha256": sha256_file(selection_path),
        "human_selection": {"candidate_number": number, "parent": selected_parent},
        "authenticated_object_count": len(selected_objects),
        "needed_segments": needed,
        "objects": verified,
        "all_object_hashes_verified": True,
        "payload_touched_after_human_selection": True,
        "qwen_used": False,
        "canonical_mutated": False,
        "frozen_files_mutated": False,
        "status": "GREEN_AUTHENTICATED_MATERIALIZATION",
    }
    receipt_path = output / f"GLYPH_V1_PORTABLE_PHASE_B_{stamp()}.json"
    write_json_exclusive(receipt_path, receipt)
    print(
        json.dumps(
            {
                "format": "GLYPH_V1_PORTABLE_PHASE_B_RESULT_V1",
                "status": receipt["status"],
                "objects": len(verified),
                "receipt": str(receipt_path),
                "sha256": sha256_file(receipt_path),
                "materialized_directory": str(final),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def doctor(args):
    profile = verify_profile(include_evidence=True)
    _, pin = canonical_pin(args.pin, args.pin_sha256)
    vault = Path(args.vault).expanduser().resolve() if args.vault else None
    trust = Path(args.trust_root).expanduser().resolve() if args.trust_root else None
    missing = []
    present = 0
    if vault and trust:
        required, missing = external_paths(pin, vault, trust)
        present = len(required) - len(missing)
    result = {
        "format": "GLYPH_V1_PORTABLE_DOCTOR_V1",
        "ok": sys.version_info >= (3, 10) and not missing,
        "python": sys.version.split()[0],
        "platform": sys.platform,
        "profile": profile,
        "runtime_paths_checked": bool(vault and trust),
        "runtime_paths_present": present,
        "runtime_paths_missing": missing,
        "canonical_write_attempted": False,
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    if not result["ok"]:
        raise SystemExit(2)


def parser():
    ap = argparse.ArgumentParser(prog="glyph-v1")
    commands = ap.add_subparsers(dest="command", required=True)
    commands.add_parser("verify-profile")

    p = commands.add_parser("doctor")
    p.add_argument("--vault")
    p.add_argument("--trust-root")
    p.add_argument("--pin")
    p.add_argument("--pin-sha256")

    p = commands.add_parser("query")
    p.add_argument("--vault", required=True)
    p.add_argument("--trust-root", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--query", required=True)
    p.add_argument("--pin")
    p.add_argument("--pin-sha256")

    p = commands.add_parser("select")
    p.add_argument("--phase-a", required=True)
    p.add_argument("--candidate", required=True, type=int)
    p.add_argument("--out", required=True)

    p = commands.add_parser("materialize")
    p.add_argument("--vault", required=True)
    p.add_argument("--trust-root", required=True)
    p.add_argument("--cache", required=True)
    p.add_argument("--phase-a", required=True)
    p.add_argument("--selection", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--pin")
    p.add_argument("--pin-sha256")
    return ap


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "verify-profile":
            print(json.dumps(verify_profile(), ensure_ascii=False, sort_keys=True))
        elif args.command == "doctor":
            doctor(args)
        elif args.command == "query":
            run_phase_a(args)
        elif args.command == "select":
            receipt = create_selection(args.phase_a, args.candidate, args.out)
            print(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        elif args.command == "materialize":
            run_materialization(args)
    except Refusal as error:
        print(f"GLYPH V1 REFUSAL: {error}", file=sys.stderr)
        raise SystemExit(3)


if __name__ == "__main__":
    main()
