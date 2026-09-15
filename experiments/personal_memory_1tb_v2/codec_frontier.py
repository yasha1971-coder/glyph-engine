#!/usr/bin/env python3
"""Independent bounded codec frontier experiment; no app format changes."""
import argparse
import bz2
from concurrent.futures import ProcessPoolExecutor
import ctypes as C
import ctypes.util
import hashlib
import json
import lzma
import os
from pathlib import Path
import platform
import resource
import stat
import sys
import time
import zlib

LIMIT=64*1024**2
POLICIES=('legacy-best','bzip2-9','xz-3','xz-6','xz-9','sample-bz-xz6')
EXCLUDED={'тут все.txt','тут все.txt\\'}

class Zstd:
    def __init__(self):
        name=C.util.find_library('zstd')
        if not name: raise RuntimeError('system libzstd unavailable')
        self.lib=C.CDLL(name)
        signatures={'ZSTD_compressBound':([C.c_size_t],C.c_size_t),
                    'ZSTD_compress':([C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t,C.c_int],C.c_size_t),
                    'ZSTD_decompress':([C.c_void_p,C.c_size_t,C.c_void_p,C.c_size_t],C.c_size_t),
                    'ZSTD_isError':([C.c_size_t],C.c_uint),
                    'ZSTD_versionString':([],C.c_char_p)}
        for key,(args,ret) in signatures.items():
            fn=getattr(self.lib,key);fn.argtypes=args;fn.restype=ret
        self.version=self.lib.ZSTD_versionString().decode()
    def compress(self,b,level):
        capacity=self.lib.ZSTD_compressBound(len(b));out=C.create_string_buffer(capacity)
        n=self.lib.ZSTD_compress(out,capacity,b,len(b),level)
        if self.lib.ZSTD_isError(n):raise ValueError('zstd compression error')
        return out.raw[:n]
    def decompress(self,b,size):
        out=C.create_string_buffer(max(1,size))
        n=self.lib.ZSTD_decompress(out,size,b,len(b))
        if self.lib.ZSTD_isError(n) or n!=size:raise ValueError('zstd restore error')
        return out.raw[:n]


def sha(b):return hashlib.sha256(b).hexdigest()

def stable(path):
    fd=os.open(path,os.O_RDONLY|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,'rb') as f:
        a=os.fstat(f.fileno())
        if not stat.S_ISREG(a.st_mode) or a.st_size>LIMIT:raise ValueError('unsupported input size/type')
        b=f.read(LIMIT+1);s=os.fstat(f.fileno())
        if (a.st_size,a.st_mtime_ns,a.st_ctime_ns)!=(s.st_size,s.st_mtime_ns,s.st_ctime_ns) or len(b)!=a.st_size:
            raise ValueError('source changed during reading')
        return b


def encode(codec,b,zstd=None):
    if codec=='raw':return b
    if codec=='deflate9':return zlib.compress(b,9)
    if codec=='bzip2-9':return bz2.compress(b,9)
    if codec.startswith('xz-'):return lzma.compress(b,preset=int(codec[3:]))
    if codec.startswith('zstd-'):return zstd.compress(b,int(codec[5:]))
    raise ValueError('unknown codec')

def decode(codec,b,size,zstd=None):
    if codec=='raw':return b
    if codec=='deflate9':return zlib.decompress(b)
    if codec=='bzip2-9':return bz2.decompress(b)
    if codec.startswith('xz-'):return lzma.decompress(b)
    if codec.startswith('zstd-'):return zstd.decompress(b,size)
    raise ValueError('unknown codec')

def probe(b):
    # Fixed before external Silesia/VIKA evaluation; no names, hashes, or old results.
    # Small files get no costly probe. Three disjoint 64KiB windows otherwise.
    width=64*1024
    if len(b)<3*width:return 'bzip2-9',dict(reason='small-input',sample_bytes=0)
    starts=[0,(len(b)-width)//2,len(b)-width]
    sizes={'bzip2-9':0,'xz-6':0}
    for start in starts:
        sample=b[start:start+width]
        for codec in sizes:sizes[codec]+=len(encode(codec,sample))
    gain=sizes['bzip2-9']-sizes['xz-6']
    estimate=gain*len(b)/(3*width)
    choice='xz-6' if gain>=.10*sizes['bzip2-9'] and estimate>=256*1024 else 'bzip2-9'
    return choice,dict(reason='sample',sample_bytes=3*width,sample_sizes=sizes,estimated_saved_bytes=estimate)

def select(policy,b,zstd=None):
    info={}
    if policy=='legacy-best':
        candidates=[('raw',b)]
        for name in ('deflate9','bzip2-9','xz-9'):candidates.append((name,encode(name,b)))
        codec,payload=min(candidates,key=lambda x:len(x[1]))
    else:
        if policy=='sample-bz-xz6':codec,info=probe(b)
        else:codec=policy
        payload=encode(codec,b,zstd)
        if len(payload)>=len(b):codec,payload='raw',b
    return codec,payload,info

def job(args):
    source,expected,policy,temp=args
    zstd=Zstd() if policy.startswith('zstd-') else None
    total_start=time.perf_counter();b=stable(Path(source))
    if sha(b)!=expected:raise ValueError('input identity changed')
    t=time.perf_counter();codec,payload,info=select(policy,b,zstd);encode_s=time.perf_counter()-t
    t=time.perf_counter()
    # Fresh regular temporary file; scope excludes durable directory/catalogue commit.
    path=Path(temp)
    with path.open('xb') as f:f.write(payload);f.flush();os.fsync(f.fileno())
    disk_payload=path.read_bytes();io_s=time.perf_counter()-t
    t=time.perf_counter();restored=decode(codec,disk_payload,len(b),zstd)
    if restored!=b:raise ValueError('full byte restoration mismatch')
    verify_s=time.perf_counter()-t
    elapsed=time.perf_counter()-total_start
    path.unlink()
    return dict(source_sha256=expected,source_bytes=len(b),policy=policy,codec=codec,
                payload_bytes=len(payload),encode_including_probe_s=encode_s,
                write_fsync_read_s=io_s,decode_compare_s=verify_s,pipeline_s=elapsed,
                peak_worker_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                probe=info,exact_restore=True)


def inventory(source):
    paths=[];seen=0
    for folder,dirs,files in os.walk(source,followlinks=False):
        dirs[:]=sorted(d for d in dirs if not (Path(folder)/d).is_symlink())
        if len(Path(folder).relative_to(source).parts)>=6:dirs[:]=[]
        seen+=len(dirs)+len(files)
        if seen>10000:raise ValueError('input discovery budget exceeded')
        for name in sorted(files):
            p=Path(folder)/name
            if name.casefold() in EXCLUDED or p.is_symlink():continue
            if p.is_file():paths.append(p)
    if not paths or len(paths)>500:raise ValueError('requires 1..500 files')
    return [(str(p),sha(stable(p))) for p in paths]

def run(source,out,repeats=3,workers=2,policies=None):
    source,out=Path(source).resolve(),Path(out).resolve()
    if out==source or out in source.parents or source in out.parents:raise ValueError('output must be separate')
    if out.exists():raise ValueError('output exists; nothing overwritten')
    if not 1<=repeats<=3 or workers not in (1,2):raise ValueError('run limits')
    try:zversion=Zstd().version
    except (OSError,RuntimeError):zversion=None
    policies=list(policies or POLICIES)+(['zstd-3','zstd-9'] if policies is None and zversion else [])
    if any(p not in (*POLICIES,'zstd-3','zstd-9') for p in policies):raise ValueError('invalid policy')
    inputs=inventory(source)
    out.mkdir(); scratch=out/'temporary';scratch.mkdir()
    start=time.perf_counter();runs=[]
    # Each process executes one complete policy batch. RSS is a worker lifetime peak.
    for trial in range(repeats):
        order=policies[trial:]+policies[:trial]
        if trial%2:order=list(reversed(order))
        for policy in order:
            jobs=[(p,h,policy,str(scratch/f'{trial}-{i}.tmp')) for i,(p,h) in enumerate(inputs)]
            t=time.perf_counter()
            with ProcessPoolExecutor(max_workers=workers) as pool:rows=list(pool.map(job,jobs))
            r=dict(trial=trial+1,policy=policy,workers=workers,wall_s=time.perf_counter()-t,
                   payload_bytes=sum(x['payload_bytes'] for x in rows),files=rows)
            runs.append(r)
            with (out/f'{trial}-{policy}.json').open('x') as f:json.dump(r,f,indent=2)
            print(json.dumps(dict(stage='policy-measured',trial=trial+1,policy=policy,seconds=r['wall_s'],bytes=r['payload_bytes'])),flush=True)
    report=dict(format='GLYPH_CODEC_FRONTIER_V1',status='ALL_SELECTED_POLICIES_EXACT',
                source_files=len(inputs),source_logical_bytes=sum(x['source_bytes'] for x in runs[0]['files']),
                zstd_version=zversion,zstd_skipped=zversion is None,python=sys.version,platform=platform.platform(),
                runner_sha256=sha(Path(__file__).read_bytes()),runs=runs,total_wall_s=time.perf_counter()-start,
                methodology=['fixed probe rule, never reads old benchmark results',
                             'same inputs and worker count for each policy; rotating order',
                             'each payload written, fsynced, reread and restored exactly',
                             'reported sizes are summed payload, no metadata or whole-file dedup',
                             'wall includes processes, source read/hash, probe, codec, I/O and validation',
                             'no cache flushing; temporary-file fsync is not full archive commit',
                             'RSS values are per-worker high-water marks, not sum of peaks at same instant'],
                not_claimed=['app format integration','Precomp-equivalent VIKA density','universal best codec','directory/catalogue durability','representative user-corpus statistics'])
    raw=json.dumps(report,indent=2).encode();(out/'FRONTIER.json').write_bytes(raw)
    (out/'FRONTIER.json.sha256').write_text(sha(raw)+'\n')
    scratch.rmdir();print('COMPLETE: '+str(out/'FRONTIER.json'))
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--repeats',type=int,default=3);p.add_argument('--workers',type=int,default=2)
    a=p.parse_args()
    try:run(a.source,a.output,a.repeats,a.workers)
    except Exception as e:print('STOP: '+str(e),file=sys.stderr);sys.exit(2)
