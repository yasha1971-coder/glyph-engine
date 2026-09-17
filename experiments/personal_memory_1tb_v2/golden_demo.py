#!/usr/bin/env python3
"""Golden Demo RC1: full locally supplied Silesia, isolated reproducible workflow.

No download, source writes, original archive deletion, or benchmark data upload.
"""
import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zlib
import incremental_memory as inc
import verified_hybrid_archive as archive
import permanent_delete

# Primary: https://sun.aei.polsl.pl/~sdeor/index.php?page=silesia
SILESIA = {
    'dickens': (10192446, '88334708559f6db57d79096bc0aca07e'),
    'mozilla': (51220480, 'c7789a2097f1ff944b0c737430a339b3'),
    'mr': (9970564, '38e623e3093b7bf2003ca4b1bbc19927'),
    'nci': (33553445, '31f85bc8706f3c921104e7c169e2e2e1'),
    'ooffice': (6152192, '573c4ae915e36631d8f2dcffb9b9b66d'),
    'osdb': (10085684, 'e734b0c48e6a982adfb5802da3032ecd'),
    'reymont': (6627202, 'd8f54d78105079775f32d76dc55fc671'),
    'samba': (21606400, '154eaea7ea70e89f6339ff0abf4112ca'),
    'sao': (7251944, '79e95a22e18cd82b7e42bf91b380d30b'),
    'webster': (41458703, '474931ad907ac27bf962c75ded46c069'),
    'xml': (5345280, '9b09c0c80104adb8aae910b7d7db003e'),
    'x-ray': (8474240, '9baec32ad14ec3eff487d254382cb91c'),
}
ALIASES = {'dickens': 'dickens.txt', 'reymont': 'reymont.pdf', 'xml': 'xml.xml'}
REPORT = 'GLYPH_GOLDEN_DEMO_RC1.json'


def event(stage, **values):
    print(json.dumps(dict(stage=stage, **values), ensure_ascii=False), flush=True)


def stable_read(path, limit):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, 'rb') as f:
        a = os.fstat(f.fileno())
        if not stat.S_ISREG(a.st_mode) or a.st_size > limit:
            raise inc.Error('unsupported or oversized benchmark file')
        data = f.read(limit + 1); b = os.fstat(f.fileno())
    def identity(s):
        return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns
    if identity(a) != identity(b) or len(data) != a.st_size:
        raise inc.Error('benchmark source changed during reading')
    return data


def discover(golden):
    """Enumerate names only inside explicit GOLDEN; no filesystem-wide inventory."""
    golden = Path(golden)
    if golden.is_symlink() or not golden.is_dir():
        raise inc.Error('GOLDEN directory missing or symlink')
    candidates = {name: [] for name in SILESIA}
    accepted_names = {name: name for name in SILESIA}
    accepted_names.update({alias.casefold(): name for name, alias in ALIASES.items() if name in SILESIA})
    count = 0
    for folder, dirs, files in os.walk(golden, followlinks=False):
        relative = Path(folder).relative_to(golden)
        if len(relative.parts) >= 6:
            dirs[:] = []
        dirs[:] = sorted(d for d in dirs if not (Path(folder) / d).is_symlink())
        count += len(files) + len(dirs)
        if count > 20000:
            raise inc.Error('GOLDEN name discovery budget exceeded')
        for name in sorted(files):
            key = accepted_names.get(name.casefold())
            if key in candidates and not (Path(folder) / name).is_symlink():
                candidates[key].append(Path(folder) / name)
    chosen = {}
    for name, (size, expected) in SILESIA.items():
        for path in sorted(candidates[name]):
            if path.stat().st_size != size:
                continue
            raw = stable_read(path, size)
            if hashlib.md5(raw).hexdigest() == expected:
                chosen[name] = path
                event('source-verified', file=name, bytes=size, sha256=inc.digest(raw))
                break
        if name not in chosen:
            raise inc.Error('Missing or MD5/size mismatch: ' + name + '; unpacked full Silesia is required')
    return chosen


def sizes(root):
    files = [p for p in Path(root).rglob('*') if p.is_file()]
    return {'logical': sum(p.stat().st_size for p in files),
            'allocated': sum(p.stat().st_blocks * 512 for p in files)}


def run(command):
    subprocess.run([sys.executable, str(Path(__file__).with_name(command[0]))] + command[1:], check=True)


def build(golden, output, revisions=100, *, codec_policy='legacy-best', workers=1):
    golden, output = Path(golden).resolve(), Path(output).resolve()
    if codec_policy not in ('legacy-best', 'sample-bz-xz6') or type(workers) is not int or workers not in (1, 2):
        raise inc.Error('unsupported codec policy or workers')
    if not 1 <= revisions <= 100:
        raise inc.Error('revision budget is 1..100')
    if output == golden or golden in output.parents or output in golden.parents:
        raise inc.Error('demo output must be separate from GOLDEN')
    if output.exists():
        raise inc.Error('demo output already exists; nothing overwritten')
    if not output.parent.is_dir() or shutil.disk_usage(output.parent).free < 2 * 1024**3:
        raise inc.Error('existing output parent with at least 2 GiB free required')
    chosen = discover(golden)
    started = time.monotonic()
    output.mkdir(mode=0o700)
    source = output / 'input'; source.mkdir()
    inventory, container, memory_dir = (output / x for x in ('inventory', 'archive', 'memory'))
    identities = []
    for name, path in chosen.items():
        size, md5 = SILESIA[name]
        data = stable_read(path, size)
        if len(data) != size or hashlib.md5(data).hexdigest() != md5:
            raise inc.Error('source identity changed after preflight')
        alias = ALIASES.get(name, name)
        (source / alias).write_bytes(data)
        identities.append({'name': name, 'archive_path': alias, 'bytes': size, 'md5': md5, 'sha256': inc.digest(data)})
    event('build-base-archive', files=len(identities), note='Full Silesia; compression can take several minutes')
    run(['corpus_truth_gate.py', '--source', str(source), '--state', str(inventory)])
    base_started = time.monotonic()
    base_report = archive.build(inventory, container, 0, codec_policy=codec_policy, workers=workers)
    base_seconds = time.monotonic() - base_started
    receipt_pin = inc.digest((container / archive.RECEIPT).read_bytes())
    m = inc.Memory(memory_dir, container, receipt_pin)
    first = m.initialize(); permanent_delete.set_head(m, first)
    # Independently compare decoder output to staged source, not just a success flag.
    with tempfile.TemporaryDirectory(prefix='verify-', dir=output) as temp:
        for ident in identities:
            path = ident['archive_path']
            dest = Path(temp) / 'restored'
            m.restore(first, path, dest)
            if dest.read_bytes() != (source / path).read_bytes():
                raise inc.Error('base restore mismatch')
            dest.unlink()
    event('base-byte-perfect', files=len(identities))
    base_storage = sizes(container)
    # A separate generated gzip control is NOT included in the Silesia ratio.
    xml_path = ALIASES.get('xml', 'xml')
    original = (source / xml_path).read_bytes()
    versions = [{'snapshot': first, 'sha256': inc.digest(original), 'bytes': len(original), 'revision': 0}]
    full_payload_baseline = 0
    baseline_seen = {inc.digest(original)}
    with tempfile.TemporaryDirectory(prefix='incoming-', dir=output) as temp:
        incoming = Path(temp)
        for i in range(1, revisions + 1):
            # Synthetic byte edits, explicitly not naturally observed document history.
            # Include reversion every 25th step; never modify benchmark originals.
            if i % 25 == 0:
                data = original
            else:
                marker = ('\n<!-- GLYPH synthetic revision %03d -->\n' % i).encode()
                position = (i * 7919) % max(1, len(original))
                data = marker + original[:position] + original[min(position + i % 17, len(original)):]
            (incoming / xml_path).write_bytes(data)
            parent = permanent_delete.head(m)
            pin = m.add(parent, incoming, version_note='Synthetic revision %03d' % i)
            permanent_delete.set_head(m, pin)
            if m.read_item(m.snapshot(pin)['files'][xml_path]) != data:
                raise inc.Error('version restore mismatch')
            if inc.digest(data) not in baseline_seen:
                full_payload_baseline += min(len(data), len(zlib.compress(data, 6)))
                baseline_seen.add(inc.digest(data))
            versions.append({'snapshot': pin, 'sha256': inc.digest(data), 'bytes': len(data), 'revision': i})
            if i % 10 == 0 or i == revisions:
                event('versions-verified', versions=i + 1)
        version_storage = sizes(memory_dir)
        (incoming / xml_path).unlink()
        control = gzip.compress(original, compresslevel=9, mtime=0)
        (incoming / 'control-generated.xml.gz').write_bytes(control)
        pin = m.add(permanent_delete.head(m), incoming, version_note='Generated gzip control; excluded from Silesia score')
        permanent_delete.set_head(m, pin)
        if m.read_item(m.snapshot(pin)['files']['control-generated.xml.gz']) != control:
            raise inc.Error('gzip control mismatch')
    # A cold reopen validates every retained version identity after all additions.
    cold = inc.Memory(memory_dir, container, receipt_pin)
    for version in versions:
        data = cold.read_item(cold.snapshot(version['snapshot'])['files'][xml_path])
        if len(data) != version['bytes'] or inc.digest(data) != version['sha256']:
            raise inc.Error('cold history restore mismatch')
    # Run destructive/tamper checks only on a disposable COPY of the demo overlay.
    with tempfile.TemporaryDirectory(prefix='destructive-check-', dir=output) as temp:
        clone = Path(temp) / 'memory'; shutil.copytree(memory_dir, clone)
        check = inc.Memory(clone, container, receipt_pin)
        payload = clone / 'objects' / inc.digest(control)
        if not payload.is_file():
            raise inc.Error('expected independent control payload missing')
        payload.write_bytes(b'INTENTIONALLY CORRUPTED TEST COPY')
        try:
            check.read_item(check.snapshot(permanent_delete.head(check))['files']['control-generated.xml.gz'])
        except Exception:
            pass
        else:
            raise inc.Error('corruption was accepted')
        permanent_delete.delete(check, permanent_delete.head(check), 'control-generated.xml.gz')
        for _, doc in permanent_delete.timeline(check, permanent_delete.head(check)):
            if xml_path in doc['files']:
                data = check.read_item(doc['files'][xml_path])
                if inc.digest(data) != doc['files'][xml_path]['sha256']:
                    raise inc.Error('shared history damaged by deletion')
    # Text retrieval proof is explicitly an API byte scan, not an indexed UI feature.
    probe = original[:min(32, len(original))]
    located = cold.read_item(cold.snapshot(first)['files'][xml_path]).find(probe)
    if located != 0:
        raise inc.Error('exact byte retrieval failed')
    per_file = []
    objects = {x['sha256']: x for x in base_report['objects']}
    for row in identities:
        obj = objects[row['sha256']]
        per_file.append(dict(row, codec=obj['codec'], stored_payload_bytes=obj['stored_bytes']))
    try:
        commit = subprocess.check_output(['git', '-C', str(Path(__file__).parent), 'rev-parse', 'HEAD'], text=True).strip()
    except Exception:
        commit = 'unknown'
    try:
        clean = not subprocess.check_output(['git', '-C', str(Path(__file__).parent), 'status', '--porcelain'], text=True).strip()
    except Exception:
        clean = False
    report = {'format': 'GLYPH_GOLDEN_DEMO_RC1', 'status': 'LOCAL_GATES_PASSED',
        'code_commit': commit, 'code_worktree_clean': clean,
        'runtime_source_sha256': {p.name: inc.digest(p.read_bytes()) for p in sorted(Path(__file__).parent.glob('*.py'))}, 'python': sys.version, 'platform': platform.platform(),
        'corpus': 'full Silesia; official sizes and MD5 checked, local SHA-256 recorded',
        'source_files': per_file, 'source_bytes': sum(x['bytes'] for x in identities),
        'base_receipt_sha256': receipt_pin, 'base_container_bytes': base_storage,
        'base_storage_ratio': base_report['storage_ratio'], 'base_build_seconds': base_seconds,
        'versions': versions, 'version_edit_model': 'synthetic prefix insertion/deletion/reversion on Silesia XML',
        'version_overlay_bytes_before_control': version_storage,
        'whole_file_deflate6_deduplicated_new_versions_payload_baseline': full_payload_baseline,
        'current_snapshot': permanent_delete.head(m), 'final_overlay_bytes': sizes(memory_dir),
        'checks': {'base_independent_byte_compare': True, 'all_versions_cold_sha256': True,
                   'generated_gzip_roundtrip': True, 'corruption_rejected_on_copy': True,
                   'corrupt_control_deleted_on_copy': True, 'surviving_history_after_delete': True,
                   'selected_file_exact_byte_scan': True},
        'elapsed_seconds': time.monotonic() - started,
        'not_claimed': ['industrial readiness', 'LLM integration', 'indexed full-text UI search',
                        'representative average personal-data ratio', 'graphical PDF acceptance',
                        'one-terabyte performance', 'cryptographic publisher authentication'],
        'base_codec_profile': ('existing raw/deflate9/bzip2-9/xz-9 hybrid; no Precomp in this self-contained demo'
                               if codec_policy == 'legacy-best' else 'sample-bz-xz6; raw fallback; no Precomp'),
        'base_codec_policy': codec_policy, 'base_codec_workers': workers,
        'cdc_backend': {'mode': 'native' if os.environ.get('GLYPH_CDC_NATIVE') else 'python',
                        'library_sha256': inc.digest(Path(os.environ['GLYPH_CDC_NATIVE']).read_bytes())
                                          if os.environ.get('GLYPH_CDC_NATIVE') else None}}
    raw = archive.canonical_json(report)
    (output / REPORT).write_bytes(raw)
    (output / (REPORT + '.sha256')).write_text(inc.digest(raw) + '\n')
    event('demo-ready', report_sha256=inc.digest(raw), storage_ratio=report['base_storage_ratio'], versions=len(versions))
    return report


def serve(output):
    output = Path(output).resolve()
    raw = inc.read_regular(output, REPORT, inc.META_LIMIT)
    expected = inc.read_regular(output, REPORT + '.sha256', 65).decode().strip()
    if inc.digest(raw) != expected:
        raise inc.Error('demo report checksum mismatch')
    report = json.loads(raw)
    if report['format'] != 'GLYPH_GOLDEN_DEMO_RC1' or report['status'] != 'LOCAL_GATES_PASSED':
        raise inc.Error('demo build not complete')
    os.execv(sys.executable, [sys.executable, str(Path(__file__).with_name('memory_browser.py')),
        '--memory', str(output / 'memory'), '--archive', str(output / 'archive'),
        '--archive-sha256', report['base_receipt_sha256'], '--port', '8767'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('build', 'serve'))
    p.add_argument('--golden', type=Path)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--revisions', type=int, default=100)
    p.add_argument('--codec-policy', choices=('legacy-best', 'sample-bz-xz6'), default='legacy-best')
    p.add_argument('--workers', type=int, choices=(1, 2), default=1)
    a = p.parse_args()
    if a.action == 'build':
        if a.golden is None: p.error('--golden required')
        build(a.golden, a.output, a.revisions, codec_policy=a.codec_policy, workers=a.workers)
    else:
        serve(a.output)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('STOP: ' + str(exc), file=sys.stderr)
        raise SystemExit(2)
