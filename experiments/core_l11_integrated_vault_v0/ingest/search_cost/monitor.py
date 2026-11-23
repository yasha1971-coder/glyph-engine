"""Own process-group sampler; RSS sums double-count shared pages, not PSS.

10ms sampling can miss transient peaks. RLIMIT_AS is a separate per-process
virtual-address cap, not an aggregate physical-memory claim.
"""
import json,os,resource,selectors,signal,stat,subprocess,sys,threading,time
from pathlib import Path
HERE=Path(__file__).resolve().parent

def run(mode,vault,pin,reference,cases,output,limits,profile=False):
    output.mkdir();temporary=output/'temporary';temporary.mkdir()
    err=(output/'stderr.log').open('w');start=time.monotonic();client_cpu=time.process_time();stop=threading.Event();observed=dict(peak_sampled_tree_rss_bytes=0,peak_sampled_temp_file_lengths=0,samples=0,budget_violation=None)
    def address_limit():resource.setrlimit(resource.RLIMIT_AS,(limits['per_process_address_space_limit_bytes'],)*2)
    cmd=[sys.executable,'-I',str(HERE/'worker.py'),'--mode',mode,'--vault',str(vault),'--pin',pin,'--reference',str(reference)]
    if profile:cmd+=['--profile']
    p=subprocess.Popen(cmd,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1,start_new_session=True,
        preexec_fn=address_limit,env={**os.environ,'TMPDIR':str(temporary),'PYTHONDONTWRITEBYTECODE':'1'})
    def sample():
        while not stop.is_set():
            rss=0
            for proc in Path('/proc').iterdir():
                if not proc.name.isdigit():continue
                try:
                    s=(proc/'stat').read_text();parts=s[s.rindex(')')+2:].split()
                    if int(parts[2])==p.pid:rss+=int(parts[21])*os.sysconf('SC_PAGE_SIZE')
                except (FileNotFoundError,ProcessLookupError,PermissionError,ValueError,IndexError):pass
            observed['samples']+=1;observed['peak_sampled_tree_rss_bytes']=max(observed['peak_sampled_tree_rss_bytes'],rss)
            try:temp=sum(f.lstat().st_size for f in temporary.rglob('*') if stat.S_ISREG(f.lstat().st_mode))
            except FileNotFoundError:temp=0 # concurrently deleted private temporary files
            observed['peak_sampled_temp_file_lengths']=max(observed['peak_sampled_temp_file_lengths'],temp)
            reason=('sampled_process_tree_rss_budget' if rss>limits['sampled_tree_rss_budget_bytes'] else
                    'wall_timeout' if time.monotonic()-start>limits['wall_timeout_seconds'] else None)
            if reason and p.poll() is None:
                observed['budget_violation']=reason
                try:os.killpg(p.pid,signal.SIGKILL)
                except ProcessLookupError:pass
            stop.wait(limits['sample_interval_seconds'])
    thread=threading.Thread(target=sample,daemon=True);thread.start()
    def receive():
        line=p.stdout.readline()
        if not line:raise RuntimeError('worker ended: '+str(p.poll())+' '+str(observed['budget_violation']))
        data=json.loads(line)
        if data.get('status')=='UNTRUSTED':raise ValueError(data['reason'])
        return data
    try:
        ready=receive();assert ready['status']=='READY';ready_seconds=time.monotonic()-start;answers=[]
        for case in cases:
            at=time.monotonic();p.stdin.write(json.dumps(dict(op='query',expected_pin=pin,query_hex=case['query_hex'],limit=case.get('limit',256)))+'\n');p.stdin.flush()
            answer=receive();answer['round_trip_seconds']=time.monotonic()-at;answers.append(answer)
        p.stdin.write('{"op":"close"}\n');p.stdin.flush();done=receive();assert done['status']=='DONE'
        p.wait(timeout=10);stop.set();thread.join(timeout=2);assert p.returncode==0
        assert observed['budget_violation'] is None,observed
        assert not list(temporary.iterdir()),'worker private temporary state leaked'
        result=dict(mode=mode,command=cmd,exit_code=p.returncode,wall_seconds=time.monotonic()-start,
            ready_seconds=ready_seconds,first_answer_seconds=ready_seconds+answers[0]['round_trip_seconds'],
            client_cpu_seconds=time.process_time()-client_cpu,client_self_peak_rss_kib_cumulative=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            ready=ready,answers=answers,usage=done,**observed)
        (output/'result.json').write_text(json.dumps(result,sort_keys=True,indent=2)+'\n');return result
    finally:
        if p.poll() is None:
            os.killpg(p.pid,signal.SIGKILL);p.wait()
        stop.set();thread.join(timeout=2);err.close();p.stdin.close();p.stdout.close()
