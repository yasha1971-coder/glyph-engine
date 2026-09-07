#!/usr/bin/env python3
import hashlib,json,os,sqlite3,stat,subprocess,tempfile,zipfile
from pathlib import Path

CLI='experiments/personal_vault_v0/glyph_vault_cli_v1.py'

def run(*args):
    return subprocess.check_output([str(x) for x in args],text=True)

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()

def write_fixture(root):
    (root/'Documents'/'Nested').mkdir(parents=True)
    (root/'Media').mkdir()
    (root/'Archives').mkdir()
    (root/'Empty Folder').mkdir()
    (root/'Documents'/'note.txt').write_text('old personal note\nremember Italy and Treviso\n',encoding='utf-8')
    (root/'Documents'/'копия заметки.txt').write_text('old personal note\nremember Italy and Treviso\n',encoding='utf-8')
    (root/'Documents'/'page.html').write_text('<html><body><h1>Saved page</h1><p>Rental archive</p></body></html>',encoding='utf-8')
    (root/'Documents'/'Nested'/'empty.bin').write_bytes(b'')
    (root/'Documents'/'report.pdf').write_bytes(b'%PDF-1.4\n1 0 obj<</Type/Catalog>>endobj\ntrailer<</Root 1 0 R>>\n%%EOF\n')
    with zipfile.ZipFile(root/'Documents'/'draft.docx','w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('[Content_Types].xml','<?xml version="1.0"?><Types></Types>')
        z.writestr('word/document.xml','<document><body>GLYPH DOCX MEMORY</body></document>')
    db=root/'Documents'/'memory.sqlite'
    con=sqlite3.connect(db); con.execute('create table notes(id integer primary key, body text)'); con.execute('insert into notes(body) values (?)',('SQLite personal memory',)); con.commit(); con.close()
    (root/'Media'/'photo.jpg').write_bytes(bytes.fromhex('ffd8ffe000104a46494600010100000100010000ffdb00040000ffd9'))
    (root/'Media'/'clip.mp4').write_bytes(b'\x00\x00\x00\x18ftypmp42\x00\x00\x00\x00mp42isom')
    with zipfile.ZipFile(root/'Archives'/'old.zip','w',compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr('inside.txt','archived exact bytes')
    (root/'Documents'/'данные-2020.bin').write_bytes(bytes(range(256)))
    # Stable old mtime on one file to verify temporal metadata is captured, not replaced by ingest time.
    old_ns=946684800_123456789  # 2000-01-01 UTC plus fractional ns
    os.utime(root/'Documents'/'note.txt',ns=(old_ns,old_ns))
    os.chmod(root/'Documents'/'note.txt',0o640)
    return old_ns

with tempfile.TemporaryDirectory(prefix='glyph-real-intake-v1-') as td:
    t=Path(td); src=t/'Real Folder'; vault=t/'vault'; restored=t/'restored'
    src.mkdir(); old_ns=write_fixture(src)
    original={p.relative_to(src).as_posix():(p.stat().st_size,sha(p)) for p in src.rglob('*') if p.is_file()}

    init=json.loads(run('python3',CLI,'init',vault)); assert init['ok']
    add=json.loads(run('python3',CLI,'add',vault,src))
    assert add['ok'] and add['intake_format']=='GLYPH_REAL_INTAKE_V1' and add['source_deleted'] is False
    seg=vault/'segments'/'00000001'
    intake=json.loads((seg/'intake.json').read_text())
    assert intake['format']=='GLYPH_REAL_INTAKE_V1'
    assert intake['file_count']==len(original)==11
    assert intake['directory_count']>=5
    assert intake['payload_bytes']==sum(x[0] for x in original.values())
    assert any(d['path']=='Empty Folder' for d in intake['directories'])

    by_path={x['path']:x for x in intake['files']}
    note=by_path['Documents/note.txt']; copy=by_path['Documents/копия заметки.txt']
    assert note['mtime_ns']==old_ns
    assert note['mode']==0o640
    assert note['mime_guess']=='text/plain'
    assert by_path['Documents/report.pdf']['mime_guess']=='application/pdf'
    assert by_path['Documents/draft.docx']['mime_guess']=='application/vnd.openxmlformats-officedocument.wordprocessingml.document'
    assert by_path['Documents/memory.sqlite']['mime_guess'] in ('application/vnd.sqlite3','application/x-sqlite3',None)
    assert by_path['Media/photo.jpg']['mime_guess']=='image/jpeg'
    assert by_path['Media/clip.mp4']['mime_guess']=='video/mp4'
    assert by_path['Archives/old.zip']['mime_guess']=='application/zip'
    assert note['content_id']==copy['content_id'], 'same bytes should have same content identity'
    assert note['logical_version_id']!=copy['logical_version_id'], 'different logical paths must remain distinct versions'
    assert by_path['Documents/Nested/empty.bin']['bytes']==0

    manifest=json.loads((seg/'segment-manifest.json').read_text())
    assert manifest['intake_format']=='GLYPH_REAL_INTAKE_V1'
    assert 'intake' in manifest['files']
    assert manifest['filesystem_metadata_captured'] is True

    ver=json.loads(run('python3',CLI,'verify',vault))
    assert ver['ok'] and ver['objects']==11 and ver['root_binding_verified'] is True

    restored.mkdir()
    om=json.loads((seg/'objects.json').read_text())
    for o in om['objects']:
        out=restored/o['path']; out.parent.mkdir(parents=True,exist_ok=True)
        rr=json.loads(run('python3',CLI,'restore',vault,o['path'],out))
        assert rr['sha256']==o['sha256']==sha(out)==original[o['path']][1]
        assert out.stat().st_size==original[o['path']][0]

    fs=json.loads(run('python3',CLI,'free-space',vault,'--dry-run'))
    assert fs['source_deletion_performed'] is False and fs['eligible_objects']==11
    assert fs['safe_to_free_bytes']==sum(x[0] for x in original.values())
    assert all(p.exists() for p in src.rglob('*') if p.is_file())

    # Fail closed on symlinks: no committed segment may appear.
    bad_src=t/'bad-source'; bad_src.mkdir(); (bad_src/'real.txt').write_text('real')
    os.symlink('real.txt',bad_src/'link.txt')
    bad_vault=t/'bad-vault'; json.loads(run('python3',CLI,'init',bad_vault))
    bad=subprocess.run(['python3',CLI,'add',str(bad_vault),str(bad_src)],text=True,capture_output=True)
    assert bad.returncode!=0
    assert 'symlink_not_supported_v1' in (bad.stdout+bad.stderr)
    assert not list((bad_vault/'segments').iterdir())
    assert not list((bad_vault/'manifests'/'roots').glob('*.json'))
    assert (bad_src/'real.txt').read_text()=='real' and (bad_src/'link.txt').is_symlink()

    report={
        'format':'GLYPH_REAL_INTAKE_GATE_V1','all_checks_passed':True,
        'files':11,'heterogeneous_types':['TXT','HTML','PDF','DOCX','SQLite','JPEG','MP4','ZIP','BIN','empty-file','Unicode-path'],
        'empty_directories_preserved_in_intake_metadata':True,
        'historical_mtime_preserved':True,'unix_mode_preserved':True,
        'content_identity_separated_from_logical_version_identity':True,
        'all_objects_bit_exact_restored':True,'root_and_manifest_hash_bindings_verified':True,
        'symlink_fail_closed_before_commit':True,'source_deletion_performed':False,
        'important_non_claim':'This tests heterogeneous byte preservation and filesystem metadata capture. It does not yet extract semantic text from PDF/DOCX/SQLite/media, preserve ACLs/xattrs/resource forks, ingest symlinks, or prove cross-platform metadata equivalence.'
    }
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
