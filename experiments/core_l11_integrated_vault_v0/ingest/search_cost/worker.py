"""stdio-only measurement worker; not a service or a default runtime change."""
import argparse,cProfile,json,pstats,resource,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(1,str(HERE.parent))
from vault_packed import VaultPacked
from session import Session

def emit(x):print(json.dumps(x,sort_keys=True),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('baseline','session'),required=True)
    p.add_argument('--vault',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--pin',required=True)
    p.add_argument('--profile',action='store_true');a=p.parse_args();s=None;opened=time.monotonic()
    try:
        if a.mode=='session':s=Session(a.vault,a.pin,a.reference)
        else:v=VaultPacked(a.vault,a.reference)
        emit(dict(status='READY',open_seconds=time.monotonic()-opened,session=None if s is None else s.metrics))
        for line in sys.stdin:
            q=json.loads(line)
            if q['op']=='close':break
            if q.get('expected_pin')!=a.pin:raise ValueError('caller snapshot pin mismatch')
            start=time.monotonic();profile=cProfile.Profile() if a.profile else None
            fn=lambda:s.query(a.pin,bytes.fromhex(q['query_hex']),q['limit']) if s is not None else v.query(a.pin,bytes.fromhex(q['query_hex']),q['limit'])
            claim=profile.runcall(fn) if profile else fn()
            result=dict(claim=claim,worker_query_seconds=time.monotonic()-start,metrics=s.last_metrics if s else None)
            if profile:
                stats=pstats.Stats(profile)
                rows=[dict(file=k[0],line=k[1],function=k[2],calls=x[1],self_seconds=x[2],cumulative_seconds=x[3]) for k,x in stats.stats.items()]
                result['profile']=dict(total_seconds=stats.total_tt,rows=sorted(rows,key=lambda x:-x['cumulative_seconds'])[:80])
            emit(result)
        if s:s.close()
        own=resource.getrusage(resource.RUSAGE_SELF);child=resource.getrusage(resource.RUSAGE_CHILDREN)
        emit(dict(status='DONE',self_cpu_seconds=own.ru_utime+own.ru_stime,reaped_children_cpu_seconds=child.ru_utime+child.ru_stime,
            self_peak_rss_kib=own.ru_maxrss,max_child_peak_rss_kib=child.ru_maxrss))
        return 0
    except Exception as e:emit(dict(status='UNTRUSTED',reason=str(e)));return 3
    finally:
        if s:s.close()
if __name__=='__main__':sys.exit(main())
