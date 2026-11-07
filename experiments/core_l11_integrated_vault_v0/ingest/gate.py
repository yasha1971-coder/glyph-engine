"""Synthetic file integration and process-crash gate; no private inputs."""
import json, os, shutil, subprocess, sys, tempfile
from pathlib import Path
from vault import Vault, sha
HERE=Path(__file__).resolve().parent
ref=HERE.parents[2].parent/'glyph-reference'
if len(sys.argv)>1:
    v=Vault(sys.argv[1],ref)
    v.import_files(Path(sys.argv[2]),[{'name':'new.bin'}],sys.argv[3],'/tmp/glyph-ingest-decoder')
    raise SystemExit
with tempfile.TemporaryDirectory(prefix='glyph-ingest-synthetic-') as tmp:
    base=Path(tmp); src=base/'source'; src.mkdir()
    (src/'z.bin').write_bytes(b'alpha\x00\xff shared'); (src/'empty').write_bytes(b'')
    v=Vault(base/'vault',ref)
    a=v.import_files(src,[{'name':'z.bin'},{'name':'empty'}],None,'/tmp/glyph-ingest-decoder')
    assert v.query(a,b'shared')['coordinates']==[[0,0,8]]
    (src/'a.bin').write_bytes((src/'z.bin').read_bytes())
    b=v.import_files(src,[{'name':'a.bin'}],a,'/tmp/glyph-ingest-decoder')
    assert v.query(b,b'shared')['count']==2
    assert v.query(a,b'shared')['count']==1
    assert len(list((v.path/'objects').iterdir()))==2
    (src/'rename.bin').write_bytes(b'revised shared')
    c=v.import_files(src,[{'name':'rename.bin','doc_id':0}],b,'/tmp/glyph-ingest-decoder')
    assert v.query(c,b'revised')['coordinates']==[[0,1,0]]
    assert v.query(b,b'revised')['status']=='PROVEN_EMPTY'
    (src/'new.bin').write_bytes(b'new shared')
    kills=0
    for point in ('object-ready','snapshot-ready','before-current','after-current','published'):
        case=base/point; shutil.copytree(v.path,case)
        env=dict(os.environ,GLYPH_TEST_CRASH=point)
        p=subprocess.run([sys.executable,str(HERE/'gate.py'),str(case),str(src),c],env=env)
        assert p.returncode==-9,(point,p.returncode)
        resumed=Vault(case,ref); pin=resumed.current()
        assert resumed.query(pin,b'shared')['count'] in (2,3)
        assert resumed.query(c,b'revised')['count']==1
        if pin==c:
            pin=resumed.import_files(src,[{'name':'new.bin'}],c,'/tmp/glyph-ingest-decoder')
        assert resumed.query(pin,b'new')['count']==1
        kills+=1
    doc=v.snapshot(c)['documents'][0]
    v.restore_object(v.path/'objects'/doc['sha256'],doc['sha256'],doc['bytes'],base/'restored')
    assert (base/'restored').read_bytes()==(src/'rename.bin').read_bytes()
    obj=v.path/'objects'/doc['sha256']; saved=(obj/'ef').read_bytes(); (obj/'ef').write_bytes(b'bad')
    try: v.query(c,b'absent')
    except ValueError: pass
    else: raise AssertionError('false empty on corrupted leaf')
    (obj/'ef').write_bytes(saved)
    report=dict(status='LOCAL_BOUNDED_INGEST_SMOKE_PASS',snapshots=3,sigkill_boundaries=kills,independent_selected_restore=True,corrupt_leaf_absence_rejected=True,scope='256-byte files; local process crashes only; not full acceptance')
    print(json.dumps(report,sort_keys=True))
