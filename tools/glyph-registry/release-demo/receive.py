#!/usr/bin/env python3
"""Transport receiver: independent hashes, empty folder, no original source input."""
from pathlib import Path
import argparse,hashlib,json,subprocess,sys,tarfile,time,os
def require(x,reason):
 if not x:raise ValueError(reason)
def copy_checked(src,dst,pin,maximum):
 h=hashlib.sha256();total=0
 with Path(src).open('rb') as f,Path(dst).open('xb') as g:
  while b:=f.read(1<<20):total+=len(b);require(total<=maximum,'transport too large');h.update(b);g.write(b)
 require(h.hexdigest()==pin,'independent transport SHA mismatch');return total
def run(a):
 start=time.perf_counter();root=Path(a.receiver);root.mkdir(parents=True,exist_ok=False);t=time.perf_counter();copy_checked(a.trusted,root/'trusted.json',a.expected_trusted_sha,1<<22);copy_checked(a.archive,root/'input.tar.gz',a.expected_sha,2<<30);phases={'copy_and_hash_s':time.perf_counter()-t};t=time.perf_counter();seen=set();total=0
 with tarfile.open(root/'input.tar.gz','r:gz') as z:
  for m in z:
   parts=Path(m.name).parts;require(parts and parts[0]=='kit' and not m.name.startswith('/') and '..' not in parts and (m.isfile() or m.isdir()) and m.name not in seen,'unsafe or duplicate member');seen.add(m.name);total+=m.size;require(total<=4<<30,'expanded archive too large');z.extract(m,root,filter='data')
 phases['unpack_s']=time.perf_counter()-t;t=time.perf_counter();cmd=['python3','kit/chain.py','verify','--trusted','trusted.json','--state','state']
 if a.isolate:
  cmd=['bwrap','--unshare-all','--die-with-parent','--ro-bind','/usr','/usr','--ro-bind','/bin','/bin','--ro-bind','/lib','/lib','--ro-bind','/lib64','/lib64','--proc','/proc','--dev','/dev','--tmpfs','/tmp','--dir','/etc','--ro-bind','/etc/ld.so.cache','/etc/ld.so.cache','--bind',str(root),'/work','--ro-bind',str(root/'kit'),'/work/kit','--ro-bind',str(root/'trusted.json'),'/work/trusted.json','--chdir','/work',*cmd]
 p=subprocess.run(cmd,cwd=root,capture_output=True,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',OMP_NUM_THREADS='1',GOMAXPROCS='1'));(root/'verify.stdout').write_bytes(p.stdout);(root/'verify.stderr').write_bytes(p.stderr);require(p.returncode==0,'full verifier failed: '+p.stderr.decode(errors='replace')[-1800:]);v=json.loads(p.stdout);require(v['ok'] and v['byte_check'] and v['native_replay'],'missing full verifier PASS');phases['verify_s']=time.perf_counter()-t
 return {'ok':True,'full_PASS_s':time.perf_counter()-start,'phases':phases,'verification':v,'command':cmd,'archive_sha256':a.expected_sha,'trusted_sha256':a.expected_trusted_sha,'source_files_transferred':0}
def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ['archive','trusted','expected-sha','expected-trusted-sha','receiver']:p.add_argument('--'+name,required=True)
 p.add_argument('--isolate',action='store_true');a=p.parse_args()
 try:print(json.dumps(run(a),sort_keys=True),flush=True);return 0
 except Exception as e:print(json.dumps({'ok':False,'error':str(e),'type':type(e).__name__}),file=sys.stderr);return 1
if __name__=='__main__':sys.exit(main())
