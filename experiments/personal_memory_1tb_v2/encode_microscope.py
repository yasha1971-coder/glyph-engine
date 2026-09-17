#!/usr/bin/env python3
"""Independent experiments: exact-boundary CDC and whole-file codec routing."""
import argparse
import bz2
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import lzma
from pathlib import Path
import random
import statistics
import sys
import time
import zlib
import chunk_versions as c
import native_cdc as native


def codec_job(job):
    path,expected=job
    b=Path(path).read_bytes()
    if hashlib.sha256(b).hexdigest()!=expected: raise ValueError('input changed')
    rows=[]
    for codec,enc,dec in [('deflate9',lambda x:zlib.compress(x,9),zlib.decompress),('bzip2-9',lambda x:bz2.compress(x,9),bz2.decompress),('xz-9',lambda x:lzma.compress(x,preset=9),lzma.decompress)]:
        t=time.perf_counter();packed=enc(b);seconds=time.perf_counter()-t
        if dec(packed)!=b: raise ValueError('codec mismatch')
        rows.append(dict(codec=codec,bytes=len(packed),encode_seconds=seconds))
    best=min([dict(codec='raw',bytes=len(b),encode_seconds=0)]+rows,key=lambda x:x['bytes'])
    return dict(sha256=expected,source_bytes=len(b),winner=best['codec'],best_bytes=best['bytes'],codecs=rows)


def study(paths,lib,repeats):
    rows=[];jobs=[]
    for path in paths:
        b=path.read_bytes()
        h=hashlib.sha256(b).hexdigest();jobs.append((str(path),h))
        times={'python':[],'native':[]}
        for trial in range(repeats):
            result={}
            # Alternate ordering; timings include Python slicing in both implementations.
            for mode in (['python','native'] if trial%2==0 else ['native','python']):
                t=time.perf_counter()
                chunks=list(c.split_python(b)) if mode=='python' else list(native.split(b,c.GEAR,c.MIN,c.TARGET,c.MAX,lib))
                times[mode].append(time.perf_counter()-t)
                result[mode]=[(start,len(chunk),hashlib.sha256(chunk).hexdigest()) for start,chunk in chunks]
                del chunks
            if result['native']!=result['python']: raise ValueError('CDC boundaries differ')
        rows.append(dict(sha256=h,bytes=len(b),boundaries=len(result['native']),seconds=times))
        print(json.dumps(dict(stage='cdc-exact',bytes=len(b))),flush=True)
    return rows,jobs


def main():
    p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True)
    p.add_argument('--native',required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--codec-study',action='store_true');p.add_argument('--repeats',type=int,default=3)
    a=p.parse_args()
    if not 1<=a.repeats<=5:p.error('repeats 1..5')
    if a.output.exists():p.error('output exists')
    # Explicit single directory, no recursive scan or old archives.
    paths=sorted(x for x in a.source.iterdir() if x.is_file() and not x.is_symlink() and x.name.casefold() not in ('тут все.txt','тут все.txt\\'))
    if not paths or len(paths)>500 or any(x.stat().st_size>64*1024**2 for x in paths):p.error('1..500 files, each <=64MiB')
    rows,jobs=study(paths,a.native,a.repeats)
    result=dict(format='GLYPH_ENCODE_MICROSCOPE_V1',cdc=rows,boundaries_exact=True,
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                native_source_sha256=hashlib.sha256(Path(__file__).with_name('gear_native.cpp').read_bytes()).hexdigest(),
                native_library_sha256=hashlib.sha256(Path(a.native).read_bytes()).hexdigest(),
                cdc_scope='split plus Python byte slicing, no compression or vault metadata')
    if a.codec_study:
        experiments=[]
        for trial in range(a.repeats):
            trial_rows={}
            for workers in ([1,2] if trial%2==0 else [2,1]):
                t=time.perf_counter()
                if workers==1: values=list(map(codec_job,jobs))
                else:
                    with ProcessPoolExecutor(max_workers=workers) as pool:values=list(pool.map(codec_job,jobs))
                trial_rows[str(workers)]=dict(wall_seconds=time.perf_counter()-t,files=values)
                print(json.dumps(dict(stage='codecs',trial=trial+1,workers=workers)),flush=True)
            if [(x['sha256'],x['best_bytes']) for x in trial_rows['1']['files']]!=[(x['sha256'],x['best_bytes']) for x in trial_rows['2']['files']]:raise ValueError('parallel output sizes differ')
            experiments.append(trial_rows)
        result['codec_trials']=experiments
        result['codec_scope']='all three codecs; wall includes reading, compression, roundtrip and pool startup; no archive writing'
    result['limitations']=['local workload only','no one-terabyte or end-to-end Vault speed claim','xz-only size is candidate measurement, not automatically deployed']
    with a.output.open('x') as f:json.dump(result,f,indent=2)
    print('COMPLETE: '+str(a.output))
if __name__=='__main__':main()
