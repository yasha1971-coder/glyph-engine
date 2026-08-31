#!/usr/bin/env python3
import argparse,hashlib,json,mimetypes,os,stat,time
from pathlib import Path


def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()


def logical_version_id(path,content_sha256):
    raw=("GLYPH_LOGICAL_VERSION_V1\0"+path+"\0"+content_sha256).encode('utf-8')
    return hashlib.sha256(raw).hexdigest()


def classify(root):
    files=[]; dirs=[]; unsupported=[]
    for p in sorted(root.rglob('*'),key=lambda x:x.relative_to(root).as_posix()):
        rel=p.relative_to(root).as_posix()
        try: st=p.lstat()
        except FileNotFoundError:
            unsupported.append({'path':rel,'reason':'vanished_during_scan'}); continue
        mode=st.st_mode
        if stat.S_ISLNK(mode):
            unsupported.append({'path':rel,'reason':'symlink_not_supported_v1'}); continue
        if stat.S_ISDIR(mode):
            dirs.append((p,rel,st)); continue
        if stat.S_ISREG(mode):
            files.append((p,rel,st)); continue
        unsupported.append({'path':rel,'reason':'special_file_not_supported_v1','mode':mode})
    return files,dirs,unsupported


def metadata_for_file(p,rel,st,data):
    content_sha=sha_bytes(data)
    mime,encoding=mimetypes.guess_type(rel,strict=False)
    return {
        'path':rel,
        'bytes':len(data),
        'content_sha256':content_sha,
        'content_id':'sha256:'+content_sha,
        'logical_version_id':'sha256:'+logical_version_id(rel,content_sha),
        'extension':Path(rel).suffix.lower(),
        'mime_guess':mime,
        'content_encoding_guess':encoding,
        'mode':stat.S_IMODE(st.st_mode),
        'executable':bool(st.st_mode & (stat.S_IXUSR|stat.S_IXGRP|stat.S_IXOTH)),
        'mtime_ns':st.st_mtime_ns,
        'ctime_ns':st.st_ctime_ns,
        'birthtime_ns':int(st.st_birthtime*1_000_000_000) if hasattr(st,'st_birthtime') else None,
        'timestamp_semantics':{
            'mtime':'filesystem modification time',
            'ctime':'filesystem metadata-change time on Unix; not treated as creation time',
            'birthtime':'captured only when host filesystem exposes st_birthtime',
        },
    }


def metadata_for_dir(rel,st):
    return {
        'path':rel,
        'mode':stat.S_IMODE(st.st_mode),
        'mtime_ns':st.st_mtime_ns,
        'ctime_ns':st.st_ctime_ns,
        'birthtime_ns':int(st.st_birthtime*1_000_000_000) if hasattr(st,'st_birthtime') else None,
    }


def pack(root,corpus_path,objects_path,intake_path):
    root=root.resolve()
    if not root.is_dir(): raise SystemExit('source must be a directory')
    files,dirs,unsupported=classify(root)
    if unsupported:
        raise SystemExit('unsupported filesystem entries: '+json.dumps(unsupported,ensure_ascii=False,separators=(',',':')))
    objects=[]; metadata=[]; pos=0
    with corpus_path.open('wb') as w:
        for i,(p,rel,st0) in enumerate(files):
            data=p.read_bytes()
            st1=p.stat()
            # Fail closed if the file changed materially while being captured.
            if st1.st_size!=st0.st_size or st1.st_mtime_ns!=st0.st_mtime_ns:
                raise SystemExit('file changed during intake: '+rel)
            content_sha=sha_bytes(data)
            w.write(data)
            objects.append({'id':i,'path':rel,'offset':pos,'bytes':len(data),'sha256':content_sha})
            metadata.append(metadata_for_file(p,rel,st1,data))
            pos+=len(data)
    intake={
        'format':'GLYPH_REAL_INTAKE_V1',
        'captured_unix_ns':time.time_ns(),
        'source_root':str(root),
        'file_count':len(objects),
        'directory_count':len(dirs),
        'payload_bytes':pos,
        'files':metadata,
        'directories':[metadata_for_dir(rel,st) for _,rel,st in dirs],
        'unsupported_entries':[],
        'canonical_vs_derived':{
            'canonical':['original bytes','content_sha256','logical path','filesystem metadata observed at intake'],
            'derived':['mime_guess','content_encoding_guess'],
        },
        'important_non_claim':'MIME values are filename-based guesses. ctime is not interpreted as creation time on Unix. V1 does not follow symlinks or ingest special files.',
    }
    objects_doc={'format':'GLYPH_PERSONAL_VAULT_V1_OBJECT_MAP','corpus_bytes':pos,'objects':objects}
    objects_path.write_text(json.dumps(objects_doc,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    intake_path.write_text(json.dumps(intake,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    return intake


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('corpus',type=Path); ap.add_argument('objects',type=Path); ap.add_argument('intake',type=Path)
    a=ap.parse_args(); report=pack(a.root,a.corpus,a.objects,a.intake)
    print(json.dumps({'ok':True,'format':report['format'],'files':report['file_count'],'directories':report['directory_count'],'payload_bytes':report['payload_bytes']},sort_keys=True))

if __name__=='__main__': main()
