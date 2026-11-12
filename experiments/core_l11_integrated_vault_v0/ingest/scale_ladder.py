"""Resource-bounded synthetic integration ladder. Does not establish hardware max."""
import argparse,hashlib,json,os,random,resource,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from vault_scale import VaultScale
p=argparse.ArgumentParser()
for n in ('reference','decoder','output'):p.add_argument('--'+n,type=Path,required=True)
p.add_argument('--sizes',default='1048576,8388608,100000000');a=p.parse_args()
assert not a.output.exists();a.output.mkdir(parents=True)
rng=random.Random(100000000);block=bytes(rng.randrange(256) for _ in range(65536));marker=b'GLYPH_SCALE_UNIQUE_20261108'
results=[]
for size in map(int,a.sizes.split(',')):
    started=time.monotonic();folder=a.output/str(size);folder.mkdir();src=folder/'source';src.mkdir()
    raw=(marker+block*((size+65535)//65536))[:size]
    (src/'mixed.bin').write_bytes(raw)
    v=VaultScale(folder/'vault',a.reference)
    try:
        pin=v.import_files(src,[{'name':'mixed.bin'}],None,a.decoder)
        # New logical document sharing one existing physical object; old snapshot retained.
        (src/'a-copy.bin').write_bytes(raw)
        second=v.import_files(src,[{'name':'a-copy.bin'}],pin,a.decoder)
        cases=[]
        for index,(snap,needle) in enumerate(((pin,marker),(second,b'GLYPH_SCALE_ABSENT_20261108'),(second,b'\0'))):
            claim=v.query(snap,needle,256);file=folder/(str(index)+'.json');file.write_text(json.dumps(claim,sort_keys=True,separators=(',',':'))+'\n')
            r=subprocess.run([sys.executable,'-I',str(HERE/'independent_scale.py'),'--vault',str(v.path),'--pin',snap,'--query-hex',needle.hex(),'--claim',str(file)],capture_output=True,text=True,timeout=600)
            (folder/(str(index)+'.log')).write_text(r.stdout+r.stderr)
            assert r.returncode==0,(r.stdout,r.stderr)
            cases.append(claim)
        assert cases[0]['count']==1 and cases[1]['status']=='PROVEN_EMPTY'
        assert cases[2]['count']==raw.count(b'\0')*2
        retained=sum(x.stat().st_size for x in v.path.rglob('*') if x.is_file())
        u=resource.getrusage(resource.RUSAGE_SELF);c=resource.getrusage(resource.RUSAGE_CHILDREN)
        row=dict(status='SCALE_LADDER_STAGE_PASS',source_file_bytes=size,logical_snapshot_bytes=size*2,snapshots=2,physical_content_objects=1,queries=3,wall_seconds=time.monotonic()-started,retained_vault_bytes=retained,self_peak_rss_kib=u.ru_maxrss,max_child_rss_kib=c.ru_maxrss,cumulative_self_cpu=u.ru_utime+u.ru_stime,cumulative_child_cpu=c.ru_utime+c.ru_stime,independent_index_source_semantics=False)
    except Exception as e:
        row=dict(status='SCALE_LADDER_STAGE_FAILED',source_file_bytes=size,error=str(e)[:1000],wall_seconds=time.monotonic()-started)
        results.append(row);(a.output/'result.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True);raise
    results.append(row);(a.output/'result.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(row),flush=True)
print('SCALE_LADDER_COMPLETE_PROTOCOL_CAP_NOT_HARDWARE_MAX',flush=True)
