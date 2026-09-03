#!/usr/bin/env python3
import argparse, hashlib, json, os, platform, shutil, subprocess, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CORE = ROOT / 'experiments' / 'personal_vault_v0'
CLI = CORE / 'glyph_vault_cli_v1.py'
QUERY = CORE / 'query_rlb3x_object.py'
BUNDLED_BIN = HERE / 'bin'
RELEASE_JSON = HERE / 'RELEASE.json'
VERSION = '0.1'


def sha256_path(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require_repo_layout():
    for p in (CLI, QUERY, ROOT / 'CMakeLists.txt'):
        if not p.exists():
            raise SystemExit(f'GLYPH release layout incomplete: missing {p}')


def bundled_binaries_usable():
    required = [BUNDLED_BIN / 'build_sa_binary_v1', BUNDLED_BIN / 'build_bwt_binary_v1']
    return platform.system() == 'Linux' and platform.machine() in ('x86_64', 'amd64') and all(p.is_file() and os.access(p, os.X_OK) for p in required)


def seed_build_cache(vault: Path):
    if not bundled_binaries_usable():
        return False
    build = vault.resolve() / 'cache' / 'build'
    build.mkdir(parents=True, exist_ok=True)
    for name in ('build_sa_binary_v1', 'build_bwt_binary_v1'):
        src = BUNDLED_BIN / name
        dst = build / name
        if not dst.exists() or sha256_path(dst) != sha256_path(src):
            shutil.copy2(src, dst)
            dst.chmod(0o755)
    return True


def passthrough(args):
    require_repo_layout()
    raise SystemExit(subprocess.call([sys.executable, str(CLI), *args]))


def latest_root(vault: Path):
    roots = sorted((vault / 'manifests' / 'roots').glob('*.json'))
    if not roots:
        return None, None
    p = roots[-1]
    return p, json.loads(p.read_text())


def release_info():
    if RELEASE_JSON.is_file():
        try:
            return json.loads(RELEASE_JSON.read_text())
        except Exception:
            pass
    git_sha = None
    try:
        git_sha = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        pass
    return {'format': 'GLYPH_RELEASE_INFO_V0_1', 'version': VERSION, 'git_sha': git_sha, 'packaged': False}


def doctor():
    require_repo_layout()
    tools = {name: shutil.which(name) for name in ('cmake', 'c++', 'git', 'sha256sum')}
    optional = {'rsync': shutil.which('rsync'), 'ssh': shutil.which('ssh')}
    supported_os = platform.system() == 'Linux'
    supported_arch = platform.machine() in ('x86_64', 'amd64', 'aarch64', 'arm64')
    python_ok = sys.version_info >= (3, 10)
    bundled = bundled_binaries_usable()
    source_build_ok = bool(tools['cmake'] and tools['c++'])
    runtime_ok = bundled or source_build_ok
    out = {
        'format': 'GLYPH_PERSONAL_VAULT_DOCTOR_V0_1',
        'version': VERSION,
        'ok': supported_os and supported_arch and python_ok and runtime_ok,
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python': platform.python_version(),
        'python_ok': python_ok,
        'supported_os': supported_os,
        'supported_arch': supported_arch,
        'bundled_build_binaries_usable': bundled,
        'source_build_fallback_available': source_build_ok,
        'build_tools': tools,
        'optional_transfer_tools': optional,
        'release': release_info(),
        'repo_root': str(ROOT),
        'source_deletion_policy': 'disabled',
    }
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))
    if not out['ok']:
        raise SystemExit(2)


def status(vault: Path):
    vault = vault.resolve()
    meta_path = vault / 'repo.meta'
    if not meta_path.is_file():
        raise SystemExit(f'not a GLYPH Vault: {vault}')
    meta = json.loads(meta_path.read_text())
    root_path, root = latest_root(vault)
    object_count = 0
    source_bytes = 0
    segments = []
    if root:
        for sid in root.get('segments', []):
            manifest_path = vault / 'segments' / sid / 'segment-manifest.json'
            if not manifest_path.is_file():
                raise SystemExit(f'committed segment missing manifest: {sid}')
            m = json.loads(manifest_path.read_text())
            object_count += int(m.get('object_count', 0))
            source_bytes += int(m.get('source_bytes', 0))
            segments.append(sid)
    verify_path = vault / 'manifests' / 'last-verify.json'
    verify = json.loads(verify_path.read_text()) if verify_path.is_file() else None
    out = {
        'format': 'GLYPH_PERSONAL_VAULT_STATUS_V0_1',
        'version': VERSION,
        'vault': str(vault),
        'repo_format': meta.get('format'),
        'segments': segments,
        'objects': object_count,
        'recoverable_source_bytes': source_bytes,
        'latest_root': None if root_path is None else root_path.name,
        'latest_root_sha256': None if root_path is None else sha256_path(root_path),
        'last_verify': verify,
        'source_deletion_enabled': bool(meta.get('source_deletion_enabled', False)),
    }
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


def search(vault: Path, pattern: str):
    require_repo_layout()
    vault = vault.resolve()
    _, root = latest_root(vault)
    if root is None:
        print(json.dumps({'format': 'GLYPH_PERSONAL_VAULT_SEARCH_V0_1', 'pattern': pattern, 'hits': [], 'valid_count': 0}, ensure_ascii=False, sort_keys=True))
        return
    pattern_bytes = pattern.encode('utf-8')
    if not pattern_bytes:
        raise SystemExit('empty search pattern')
    hits = []
    raw_count = 0
    rejected = 0
    for sid in root.get('segments', []):
        seg = vault / 'segments' / sid
        cmd = [
            sys.executable, str(QUERY),
            '--rlb3x', str(seg / 'bwt.rlb3x'),
            '--locate-core', str(seg / 'locate.loc2'),
            '--objects', str(seg / 'objects.json'),
            '--pattern-hex', pattern_bytes.hex(),
        ]
        got = json.loads(subprocess.check_output(cmd, text=True))
        raw_count += int(got.get('raw_count', 0))
        rejected += int(got.get('rejected_cross_object_count', 0))
        for h in got.get('valid_hits', []):
            h = dict(h)
            h['segment_id'] = sid
            hits.append(h)
    out = {
        'format': 'GLYPH_PERSONAL_VAULT_SEARCH_V0_1',
        'version': VERSION,
        'pattern': pattern,
        'pattern_hex': pattern_bytes.hex(),
        'raw_count': raw_count,
        'valid_count': len(hits),
        'rejected_cross_object_count': rejected,
        'hits': hits,
        'object_boundary_filter_applied': True,
    }
    print(json.dumps(out, ensure_ascii=False, sort_keys=True))


def main():
    ap = argparse.ArgumentParser(prog='glyph-v0.1')
    sp = ap.add_subparsers(dest='cmd', required=True)
    sp.add_parser('doctor')
    sp.add_parser('version')
    p = sp.add_parser('init'); p.add_argument('vault')
    p = sp.add_parser('add'); p.add_argument('vault'); p.add_argument('source')
    p = sp.add_parser('verify'); p.add_argument('vault')
    p = sp.add_parser('status'); p.add_argument('vault')
    p = sp.add_parser('list'); p.add_argument('vault')
    p = sp.add_parser('search'); p.add_argument('vault'); p.add_argument('pattern')
    p = sp.add_parser('restore'); p.add_argument('vault'); p.add_argument('selector'); p.add_argument('out')
    p = sp.add_parser('free-space'); p.add_argument('vault'); p.add_argument('--dry-run', action='store_true', required=True)
    a = ap.parse_args()
    if a.cmd == 'doctor': doctor()
    elif a.cmd == 'version': print(json.dumps(release_info(), ensure_ascii=False, sort_keys=True))
    elif a.cmd == 'status': status(Path(a.vault))
    elif a.cmd == 'search': search(Path(a.vault), a.pattern)
    elif a.cmd == 'init': passthrough(['init', a.vault])
    elif a.cmd == 'add':
        seed_build_cache(Path(a.vault))
        passthrough(['add', a.vault, a.source])
    elif a.cmd == 'verify': passthrough(['verify', a.vault])
    elif a.cmd == 'list': passthrough(['list', a.vault])
    elif a.cmd == 'restore': passthrough(['restore', a.vault, a.selector, a.out])
    elif a.cmd == 'free-space': passthrough(['free-space', a.vault, '--dry-run'])


if __name__ == '__main__':
    main()
