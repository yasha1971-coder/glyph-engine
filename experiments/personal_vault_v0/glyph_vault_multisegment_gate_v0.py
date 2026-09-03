#!/usr/bin/env python3
import hashlib,json,shutil,subprocess,tempfile
from pathlib import Path

CLI='experiments/personal_vault_v0/glyph_vault_cli_v0.py'
FETCH='experiments/personal_vault_v0/fetch_corpora.sh'
QUERY='experiments/personal_vault_v0/query_rlb3x_object.py'

def run(*args):
    return subprocess.check_output([str(x) for x in args],text=True)

def sha_file(p):
    h=hashlib.sha256();
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def tree_fingerprint(root):
    rows=[]
    for p in sorted(x for x in root.rglob('*') if x.is_file()):
        rows.append((str(p.relative_to(root)),p.stat().st_size,sha_file(p)))
    return rows

def search_all_segments(vault,pattern):
    hits=[]
    for seg in sorted(p for p in (vault/'segments').iterdir() if p.is_dir()):
        out=subprocess.check_output([
            'python3',QUERY,
            '--rlb3x',str(seg/'bwt.rlb3x'),
            '--locate-core',str(seg/'locate.loc2'),
            '--objects',str(seg/'objects.json'),
            '--pattern-hex',pattern.encode('utf-8').hex(),
        ],text=True)
        got=json.loads(out)
        assert got['rlb2_not_used'] is True
        for hit in got['valid_hits']:
            hits.append({'segment':seg.name,'path':hit['path'],'object_offset':hit['object_offset'],'corpus_offset':hit['corpus_offset']})
    return hits

with tempfile.TemporaryDirectory(prefix='glyph-vault-multisegment-v0-') as td:
    t=Path(td); raw=t/'raw'; src1=t/'src1'; src2=t/'src2'; vault=t/'vault'
    subprocess.check_call(['bash',FETCH,str(raw)])
    src1.mkdir(); src2.mkdir()
    shutil.copytree(raw/'canterbury',src1/'canterbury')
    for name in ('calgary','artificial','miscellaneous'):
        shutil.copytree(raw/name,src2/name)

    init=json.loads(run('python3',CLI,'init',vault))
    assert init['ok']

    add1=json.loads(run('python3',CLI,'add',vault,src1))
    assert add1['segment_id']=='00000001' and add1['objects']==11
    seg1=vault/'segments'/'00000001'
    first_before=tree_fingerprint(seg1)
    roots_after_first=sorted((vault/'manifests'/'roots').glob('*.json'))
    assert len(roots_after_first)==1
    root1=json.loads(roots_after_first[0].read_text())
    assert root1['segments']==['00000001']

    add2=json.loads(run('python3',CLI,'add',vault,src2))
    assert add2['segment_id']=='00000002' and add2['objects']==19
    seg2=vault/'segments'/'00000002'
    assert seg2.is_dir()
    first_after=tree_fingerprint(seg1)
    assert first_after==first_before, 'first immutable segment changed while adding second segment'

    roots=sorted((vault/'manifests'/'roots').glob('*.json'))
    assert len(roots)==2
    root2=json.loads(roots[-1].read_text())
    assert root2['segments']==['00000001','00000002']

    ver=json.loads(run('python3',CLI,'verify',vault))
    assert ver['ok'] and ver['segments']==2 and ver['objects']==30
    assert ver['recoverable_bytes']==7230581

    out1=t/'alice-restored.txt'; out2=t/'book1-restored'
    r1=json.loads(run('python3',CLI,'restore',vault,'canterbury/alice29.txt',out1))
    r2=json.loads(run('python3',CLI,'restore',vault,'calgary/book1',out2))
    assert sha_file(out1)==r1['sha256']==sha_file(src1/'canterbury'/'alice29.txt')
    assert sha_file(out2)==r2['sha256']==sha_file(src2/'calgary'/'book1')

    alice_hits=search_all_segments(vault,'Cheshire')
    hardy_hits=search_all_segments(vault,'Bathsheba')
    assert any(h['segment']=='00000001' and h['path']=='canterbury/alice29.txt' for h in alice_hits),alice_hits
    assert not any(h['segment']=='00000002' and h['path']=='canterbury/alice29.txt' for h in alice_hits),alice_hits
    assert any(h['segment']=='00000002' and h['path']=='calgary/book1' for h in hardy_hits),hardy_hits

    fs=json.loads(run('python3',CLI,'free-space',vault,'--dry-run'))
    assert fs['source_deletion_performed'] is False
    assert fs['eligible_objects']==30 and fs['safe_to_free_bytes']==7230581
    assert all(p.exists() for p in src1.rglob('*') if p.is_file())
    assert all(p.exists() for p in src2.rglob('*') if p.is_file())

    report={
      'format':'GLYPH_VAULT_MULTISEGMENT_GATE_V0',
      'all_checks_passed':True,
      'segments':2,
      'objects':30,
      'recoverable_bytes':7230581,
      'first_segment_bit_identical_after_second_add':True,
      'root_history_generations':2,
      'latest_root_segments':['00000001','00000002'],
      'restore_from_each_segment_sha_verified':True,
      'exact_search_fanout_across_segments_verified':True,
      'search_examples':{'Cheshire':'canterbury/alice29.txt','Bathsheba':'calgary/book1'},
      'safe_to_free_bytes':fs['safe_to_free_bytes'],
      'source_deletion_performed':False,
      'important_non_claim':'This proves additive two-segment repository growth and fan-out exact search in CI. It does not yet implement global AI ranking across segments, deduplication across segments, compaction, crash injection, or permanent source deletion.'
    }
    (vault/'manifests'/'multisegment-gate-v0.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps(report,sort_keys=True))
