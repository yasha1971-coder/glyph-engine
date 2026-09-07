#!/usr/bin/env python3
import json,subprocess,tempfile,hashlib
from pathlib import Path

CLI='experiments/personal_vault_v0/glyph_vault_cli_v0.py'
FETCH='experiments/personal_vault_v0/fetch_corpora.sh'

def run(*args):
    return subprocess.check_output([str(x) for x in args],text=True)

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

with tempfile.TemporaryDirectory(prefix='glyph-vault-cli-v0-') as td:
    t=Path(td); src=t/'src'; vault=t/'vault'; restored=t/'restored.bin'
    subprocess.check_call(['bash',FETCH,str(src)])
    init=json.loads(run('python3',CLI,'init',vault))
    assert init['ok'] and init['source_deletion_enabled'] is False
    added=json.loads(run('python3',CLI,'add',vault,src))
    assert added['ok'] and added['objects']==30 and added['source_deleted'] is False
    seg=vault/'segments'/'00000001'
    assert seg.is_dir()
    for req in ('bwt.rlb3x','locate.loc2','objects.json','source-state.json','segment-manifest.json'):
        assert (seg/req).is_file(),req
    for forbidden in ('corpus.bin','sa.bin','bwt.bin','restore-test.bin'):
        assert not (seg/forbidden).exists(),forbidden
    ver=json.loads(run('python3',CLI,'verify',vault))
    assert ver['ok'] and ver['segments']==1 and ver['objects']==30 and ver['full_restore_hash_check'] is True
    om=json.loads((seg/'objects.json').read_text())
    target=om['objects'][0]
    out=json.loads(run('python3',CLI,'restore',vault,target['path'],restored))
    assert out['ok'] and out['sha256']==target['sha256'] and sha(restored)==target['sha256']
    fs=json.loads(run('python3',CLI,'free-space',vault,'--dry-run'))
    assert fs['ok'] and fs['dry_run'] is True and fs['source_deletion_performed'] is False
    assert fs['eligible_objects']==30 and fs['safe_to_free_bytes']==added['source_bytes']
    assert all(p.exists() for p in src.rglob('*') if p.is_file())
    report={
      'format':'GLYPH_VAULT_CLI_GATE_V0','all_checks_passed':True,'segments':1,'objects':30,
      'source_bytes':added['source_bytes'],'safe_to_free_bytes':fs['safe_to_free_bytes'],
      'source_deletion_performed':False,'construction_artifacts_not_published':True,
      'restore_object_sha256_verified':True,'full_verify_passed':True,
      'important_non_claim':'This validates a local repository lifecycle in CI. It does not yet provide a mounted virtual filesystem, crash-injection testing, multi-segment global search, or permanent source deletion.'
    }
    print(json.dumps(report,sort_keys=True))
