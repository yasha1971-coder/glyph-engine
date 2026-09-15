#!/usr/bin/env python3
"""Resumable synthetic storage endurance lab, NOT a Personal Memory vault.

Uses GLYPH hybrid decoder for raw/deflate9 objects. Local seed samples may contain
private data. Nothing is uploaded. Existing source files are never changed.
"""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import resource
import shutil
import sqlite3
import stat
import sys
import time
import zlib
import verified_hybrid_archive as hybrid

BLOCK = 16 * 1024**2
SEED_LIMIT = 1024**2
MAX_SEEDS = 64
EXCLUDED = {'тут все.txt', 'тут все.txt\\'}
SUFFIXES = {'.txt','.md','.pdf','.jpg','.jpeg','.png','.mp4','.webm','.mp3','.wav','.json','.csv','.docx','.xlsx'}

def sha(b): return hashlib.sha256(b).hexdigest()
def log(stage, **kw): print(json.dumps(dict(stage=stage, **kw)), flush=True)

def syncdir(p):
    fd = os.open(p, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(fd)
    finally: os.close(fd)

def atomic(p, data):
    temp = p.with_name(p.name+'.partial')
    with temp.open('wb') as f:
        f.write(data); f.flush(); os.fsync(f.fileno())
    os.replace(temp,p); syncdir(p.parent)

def free_gate(root, reserve, extra=0):
    if shutil.disk_usage(root).free < reserve + extra:
        raise RuntimeError('free-space reserve reached; resume later with same command')

def prepare(source, root, target, reserve):
    if root == source or root in source.parents or source in root.parents:
        raise ValueError('source and output must be separate')
    if root.exists():
        raise ValueError('output already exists; use --resume only for this lab')
    free_gate(root.parent,reserve,128*1024**2)
    root.mkdir(); (root/'seeds').mkdir(); (root/'objects').mkdir()
    seeds=[]; entries=0
    for folder, dirs, files in os.walk(source, followlinks=False):
        dirs[:] = sorted(d for d in dirs if not (Path(folder)/d).is_symlink())
        if len(Path(folder).relative_to(source).parts)>=6: dirs[:]=[]
        entries += len(dirs)+len(files)
        if entries>10000: raise ValueError('source discovery bound exceeded')
        for name in sorted(files):
            p=Path(folder)/name
            if name.casefold() in EXCLUDED or p.suffix.casefold() not in SUFFIXES: continue
            if p.is_symlink(): continue
            fd=os.open(p,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
            with os.fdopen(fd,'rb') as f:
                before=os.fstat(f.fileno())
                if not stat.S_ISREG(before.st_mode): continue
                b=f.read(SEED_LIMIT)
                after=os.fstat(f.fileno())
                if (before.st_size,before.st_mtime_ns,before.st_ctime_ns)!=(after.st_size,after.st_mtime_ns,after.st_ctime_ns):
                    raise ValueError('seed source changed')
            if not b: continue
            h=sha(b)
            if h in seeds: continue
            atomic(root/'seeds'/h,b); seeds.append(h)
            if len(seeds)>=MAX_SEEDS: break
        if len(seeds)>=MAX_SEEDS: break
    if not seeds: raise ValueError('no usable seed samples')
    c=dict(format='GLYPH_SCALE_LAB_V1',target_bytes=target,block_bytes=BLOCK,
           reserve_bytes=reserve,seeds=seeds,recipe='v1: half unique SHAKE256; quarter repeated seed; quarter seed with synthetic replacements')
    atomic(root/'config.json',json.dumps(c,sort_keys=True).encode())
    return c

def source_block(c, root, i):
    n=min(c['block_bytes'],c['target_bytes']-i*c['block_bytes'])
    if n<=0: raise ValueError('index beyond target')
    if i%4 in (0,1):
        return hashlib.shake_256(b'GLYPH_SCALE_V1_UNIQUE'+str(i).encode()).digest(n),'unique-synthetic'
    h=c['seeds'][(i//4)%len(c['seeds'])]
    seed=(root/'seeds'/h).read_bytes()
    if sha(seed)!=h: raise ValueError('seed checksum mismatch')
    b=(seed*((n+len(seed)-1)//len(seed)))[:n]
    if i%4==2: return b,'repeated-seed'
    b=bytearray(b)
    for offset in range(0,n,64*1024):
        count=min(4096,n-offset)
        b[offset:offset+count]=hashlib.shake_256(b'GLYPH_SCALE_V1_EDIT'+str(i).encode()+b':'+str(offset).encode()).digest(count)
    return bytes(b),'modified-seed'

def database(root):
    db=sqlite3.connect(root/'index.sqlite3')
    db.execute('PRAGMA journal_mode=DELETE'); db.execute('PRAGMA synchronous=FULL')
    db.execute('CREATE TABLE IF NOT EXISTS blocks (i INTEGER PRIMARY KEY, kind TEXT, sha TEXT, bytes INTEGER, codec TEXT, stored INTEGER, encode_s REAL, write_verify_s REAL)')
    db.commit(); return db

def check_row(root,c,row):
    i,kind,h,n,codec,stored,_,_=row
    expected,expected_kind=source_block(c,root,i)
    if n!=len(expected) or h!=sha(expected) or kind!=expected_kind: raise ValueError('index identity mismatch')
    p=root/'objects'/h
    if p.is_symlink() or p.stat().st_size!=stored or stored>c['block_bytes']: raise ValueError('payload size mismatch')
    data=hybrid.decode(codec,p.read_bytes())
    if data!=expected: raise ValueError('restored bytes mismatch')
    return n

def run(root,c, max_blocks=None):
    db=database(root)
    try:
        rows=db.execute('SELECT * FROM blocks ORDER BY i')
        count=0; verified=0
        # Resume never trusts previous progress without revalidating it.
        for row in rows:
            if row[0]!=count: raise ValueError('non-contiguous index')
            verified+=check_row(root,c,row); count+=1
            if count%64==0: log('resume-verified',bytes=verified)
        started=time.monotonic(); added=0
        total=math.ceil(c['target_bytes']/c['block_bytes'])
        for i in range(count,total):
            if max_blocks is not None and added>=max_blocks: break
            free_gate(root,c['reserve_bytes'],2*c['block_bytes']+16*1024**2)
            block_started=time.perf_counter()
            b,kind=source_block(c,root,i); h=sha(b)
            t=time.perf_counter()
            if kind=='unique-synthetic': codec,payload='raw',b
            else:
                encoded=zlib.compress(b,9)
                codec,payload=('deflate9',encoded) if len(encoded)<len(b) else ('raw',b)
            encode_s=time.perf_counter()-t
            path=root/'objects'/h
            if path.exists():
                if path.is_symlink() or path.read_bytes()!=payload: raise ValueError('existing object differs')
            else: atomic(path,payload)
            # Read back from disk before durable progress acknowledgement.
            if hybrid.decode(codec,path.read_bytes())!=b: raise ValueError('write readback mismatch')
            with db:
                db.execute('INSERT INTO blocks VALUES (?,?,?,?,?,?,?,?)',(i,kind,h,len(b),codec,len(payload),encode_s,time.perf_counter()-block_started))
            added+=1
            if added%16==0 or i+1==total:
                done=min((i+1)*c['block_bytes'],c['target_bytes'])
                log('written-and-verified',logical_bytes=done,target_bytes=c['target_bytes'],
                    elapsed_this_session_s=round(time.monotonic()-started,2),free_bytes=shutil.disk_usage(root).free)
        return db.execute('SELECT COUNT(*) FROM blocks').fetchone()[0]==total
    finally: db.close()

def verify(root,c):
    db=database(root); start=time.monotonic(); total=0; count=0
    try:
        if db.execute('PRAGMA integrity_check').fetchone()[0]!='ok': raise ValueError('SQLite integrity failure')
        for row in db.execute('SELECT * FROM blocks ORDER BY i'):
            if row[0]!=count: raise ValueError('index gap')
            total+=check_row(root,c,row); count+=1
            if count%64==0: log('final-verification',logical_bytes=total)
        if total!=c['target_bytes']: raise ValueError('target not complete')
        unique=db.execute('SELECT SUM(stored) FROM (SELECT sha,MAX(stored) AS stored FROM blocks GROUP BY sha)').fetchone()[0]
        encode_seconds,write_seconds=db.execute('SELECT SUM(encode_s),SUM(write_verify_s) FROM blocks').fetchone()
        kinds=dict(db.execute('SELECT kind,SUM(bytes) FROM blocks GROUP BY kind'))
    finally: db.close()
    physical=logical=0
    for folder,_,names in os.walk(root):
        for name in names:
            st=(Path(folder)/name).stat();logical+=st.st_size;physical+=st.st_blocks*512
    report=dict(format='GLYPH_SCALE_LAB_RESULT_V1',status='FULL_TARGET_BYTE_VERIFIED',
        logical_input_bytes=total,blocks=count,unique_payload_bytes=unique,
        directory_bytes_before_report=logical,allocated_bytes_before_report=physical,
        workload_bytes=kinds,encode_seconds=encode_seconds,
        generated_encoded_fsynced_readback_seconds_excluding_index_commit=write_seconds,
        final_verify_s=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        config_sha256=sha((root/'config.json').read_bytes()),
        runner_sha256=sha(Path(__file__).read_bytes()),
        decoder_sha256=sha(Path(hybrid.__file__).read_bytes()),
        limitations=['synthetic sharded storage lab; not a unified Personal Memory vault',
                      'only GLYPH raw/deflate9 decoder; no xz/bzip2 codec selection',
                      'half of logical bytes intentionally unique pseudorandom',
                      'remaining data derived from local samples, not representative personal corpus',
                      'readback may be served by OS cache; no power-loss claim',
                      'fsync on WSL/DrvFS is not proof of physical media persistence'])
    raw=json.dumps(report,indent=2).encode();atomic(root/'RESULT.json',raw)
    atomic(root/'RESULT.json.sha256',(sha(raw)+'\n').encode());log('scale-complete',**report)
    return report

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--target-tb',type=float,default=1);p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not math.isfinite(a.target_tb) or not .001<=a.target_tb<=2: p.error('target must be 0.001..2 decimal TB')
    source,root=a.source.resolve(),a.output.resolve()
    if not source.is_dir(): p.error('source directory missing')
    target=int(a.target_tb*10**12);reserve=300*1024**3
    resource.setrlimit(resource.RLIMIT_AS,(1024**3,1024**3))
    if a.resume:
        c=json.loads((root/'config.json').read_text())
        if c['format']!='GLYPH_SCALE_LAB_V1' or c['target_bytes']!=target or c['block_bytes']!=BLOCK or c['reserve_bytes']!=reserve:
            raise ValueError('resume configuration mismatch')
    else: c=prepare(source,root,target,reserve)
    with (root/'lab.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if run(root,c): verify(root,c)

if __name__=='__main__':
    try: main()
    except KeyboardInterrupt:
        print('STOP: interrupted; resume with --resume',file=sys.stderr);sys.exit(130)
    except Exception as e:
        print('STOP: '+str(e),file=sys.stderr);sys.exit(2)
