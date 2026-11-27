#!/usr/bin/env python3
"""Local experimental source registry. External pins are required at receiver.

Stock GLYPH codecs and delivered engine are unchanged. Compact is an explicit
experimental schema, not a release .gidx. Zoekt's overlap adapter is disclosed.
"""
import argparse,array,bisect,contextlib,hashlib,io,json,os,shutil,subprocess,sys,tempfile,time
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'glyph/tools'));sys.path.insert(0,str(ROOT/'research'))
from convert import convert,recover_compact,digest
PARTS=('bwt.rlbwt','bwt.rlbwt.rank','locate_core_s128.bin')
def require(ok,msg):
 if not ok:raise ValueError(msg)
def encode(v):return (json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2)+'\n').encode()
def relpath(name):
 p=PurePosixPath(name)
 require(not p.is_absolute() and '..' not in p.parts and str(p)==name and name!='.','unsafe relative path')
 return p
def pin(s):require(isinstance(s,str) and len(s)==64 and all(x in '0123456789abcdef' for x in s),'invalid external SHA-256')
def manifest(raw):
 m=json.loads(raw);files=m['files'];require(len(files)==m['file_count'] and files and type(m['corpus_bytes']) is int and 1<=m['corpus_bytes']<=128*1024*1024,'invalid manifest count/size')
 paths=[];offset=0
 for d in files:
  relpath(d['path']);paths.append(d['path']);pin(d['sha256'])
  require(type(d['size']) is int and d['size']>=0 and d['concat_start']==offset and d['concat_end_exclusive']==offset+d['size'] and d['delimiter_offset']==offset+d['size'],'invalid document boundaries')
  offset+=d['size']+1
 require(paths==sorted(set(paths),key=lambda p:p.encode()) and offset==m['corpus_bytes'] and sum(d['size'] for d in files)==m['source_bytes'],'invalid manifest ordering/total')
 pin(m['corpus_sha256']);return m
def build(args):
 source=Path(args.source);out=Path(args.out);require(not out.exists(),'output exists');out.mkdir(parents=True)
 raw=(source/'MANIFEST.json').read_bytes();m=manifest(raw)
 require((source/'corpus.bin').stat().st_size==m['corpus_bytes'] and digest(source/'corpus.bin')==m['corpus_sha256'],'frozen corpus differs')
 (out/'documents.json').write_bytes(raw)
 core=out/'core';conversion=None
 if args.engine in ('stock','compact'):
  subprocess.run([str(ROOT/'glyph/glyph-search'),'build',str(source/'corpus.bin'),'--out',str(core)],check=True)
  if args.engine=='compact':conversion=convert(core,ROOT/'glyph/tools',ROOT/'research')
 else:
  core.mkdir()
  subprocess.run([str(ROOT/'zoekt-registry'),'build',str(source/'source'),str(source/'MANIFEST.json'),str(core/'index.zoekt')],check=True)
 files={str(p.relative_to(out)).replace('\\','/'):{'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(out.rglob('*')) if p.is_file()}
 reg={'schema':'glyph03-fixed-source-registry-v0','engine':args.engine,'documents_sha256':hashlib.sha256(raw).hexdigest(),'corpus_sha256':m['corpus_sha256'],'corpus_bytes':m['corpus_bytes'],'source_bytes':m['source_bytes'],'file_count':m['file_count'],'files':files}
 (out/'registry.json').write_bytes(encode(reg))
 print(json.dumps({'ok':True,'index_sha256':digest(out/'registry.json'),'documents_sha256':reg['documents_sha256'],'conversion':conversion,'working_bytes':sum(p.stat().st_size for p in out.rglob('*') if p.is_file())}),flush=True)
def snapshot(index,expected_index,expected_docs,scratch):
 pin(expected_index);pin(expected_docs);index=Path(index)
 require(not (index/'registry.json').is_symlink(),'symlink registry')
 raw=(index/'registry.json').read_bytes();require(len(raw)<1<<20 and hashlib.sha256(raw).hexdigest()==expected_index,'external index SHA-256 mismatch')
 reg=json.loads(raw);require(reg['schema']=='glyph03-fixed-source-registry-v0' and reg['engine'] in ('stock','compact','zoekt'),'unknown registry schema')
 require(reg['documents_sha256']==expected_docs,'external documents SHA-256 mismatch')
 dest=Path(scratch)/'index';dest.mkdir()
 (dest/'registry.json').write_bytes(raw)
 for name,meta in reg['files'].items():
  relpath(name);p=index/name;q=dest/name
  require(not p.is_symlink() and p.is_file() and p.stat().st_size==meta['bytes'],'index part length/type differs: '+name)
  q.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
  with p.open('rb') as f,q.open('xb') as g:
   while b:=f.read(1<<20):h.update(b);g.write(b)
  require(q.stat().st_size==meta['bytes'] and h.hexdigest()==meta['sha256'],'index part SHA-256 mismatch: '+name)
 require(digest(dest/'documents.json')==expected_docs,'document manifest SHA-256 mismatch')
 m=manifest((dest/'documents.json').read_bytes())
 require(m['corpus_sha256']==reg['corpus_sha256'] and m['corpus_bytes']==reg['corpus_bytes'],'registry corpus identity mismatch')
 if reg['engine']!='zoekt':
  cm=json.loads((dest/'core/manifest.json').read_bytes())
  fmt='glyph-search-index-v2' if reg['engine']=='stock' else 'glyph-search-index-experimental-rlr2-loc2-v0'
  require(cm['format']==fmt and cm['corpus_sha256']==m['corpus_sha256'] and cm['corpus_bytes']==m['corpus_bytes'],'core format/corpus differs')
  for name in PARTS:
   require(cm['files'][name]==reg['files']['core/'+name],'core part identity mismatch')
 return dest,reg,m
def recover_glyph(index,engine,m,dest,scratch):
 data_path=Path(scratch)/'corpus.bin';core=index/'core'
 if engine=='stock':
  from recover_search_index_v1 import recover
  with contextlib.redirect_stdout(io.StringIO()):recover(core/PARTS[0],m['corpus_bytes'],m['corpus_sha256'],data_path,ROOT/'glyph/build/recover_bwt_libsais',scratch)
 else:
  import rlbwt_rank_v2 as rank2
  # Unmodified research inspector checks header/FNV/SHA/counter geometry.
  check=rank2.inspect_rank(core/PARTS[1])
  require(check['source_rlb2_sha256']==digest(core/PARTS[0]) and check['source_rlb2_bytes']==(core/PARTS[0]).stat().st_size,'RLR2 source RLB2 SHA/size mismatch')
  import rlbwt_container_v2 as rlb2
  with (core/PARTS[0]).open('rb') as f:header=rlb2.parse_rlb2_header(f,(core/PARTS[0]).stat().st_size)
  require(header['corpus_bytes']==m['corpus_bytes'] and header['row_count']==m['corpus_bytes']+1,'RLB2 dimensions differ from pinned source manifest')
  recover_compact(core/PARTS[0],data_path,ROOT/'glyph/build/recover_bwt_libsais',scratch)
 data=data_path.read_bytes()
 require(len(data)==m['corpus_bytes'] and hashlib.sha256(data).hexdigest()==m['corpus_sha256'],'recovered corpus length/SHA-256 mismatch')
 dest=Path(dest);dest.mkdir(parents=True,exist_ok=True)
 for d in m['files']:
  start=d['concat_start'];body=data[start:start+d['size']]
  require(hashlib.sha256(body).hexdigest()==d['sha256'] and data[start+d['size']]==1,'recovered document SHA/delimiter mismatch')
  p=dest/d['path'];p.parent.mkdir(parents=True,exist_ok=True)
  with p.open('xb') as f:f.write(body)
 return data
def glyph_result(proc,q,limit,m,starts,data):
 proc.stdin.write(q.hex()+'\t'+str(limit)+'\n');proc.stdin.flush()
 line=proc.stdout.readline();fields=line.rstrip('\n').split('\t')
 require(len(fields)==9 and fields[0]=='OK','native GLYPH response invalid: '+line)
 count=int(fields[3]);n=int(fields[4]);offsets=[int(x) for x in fields[8].split(',')] if fields[8] else []
 require(0<=count<=m['corpus_bytes'] and n==len(offsets)==min(count,limit) and offsets==sorted(set(offsets)) and fields[5]==str(n<count).lower(),'native GLYPH cardinality/bounds invalid')
 matches=[]
 for off in offsets:
  j=bisect.bisect_right(starts,off)-1;require(j>=0,'coordinate before first document');d=m['files'][j];end=d['concat_start']+d['size']
  require(d['concat_start']<=off and off+len(q)<=end and data[off:off+len(q)]==q,'coordinate crosses file boundary or byte mismatch')
  matches.append({'path':d['path'],'offset':off-d['concat_start']})
 return {'ok':True,'count':count,'matches':matches,'bounded':n<count,'byte_check':True}
def serve(args):
 phase={};start=time.perf_counter();proc=None
 with tempfile.TemporaryDirectory(prefix='glyph03-receiver-') as t:
  index,reg,m=snapshot(args.index,args.expected_index,args.expected_docs,t);phase['hash_snapshot_seconds']=time.perf_counter()-start
  t0=time.perf_counter();dest=Path(args.recovered);require(not dest.exists(),'recovery output exists')
  if reg['engine']=='zoekt':
   proc=subprocess.Popen([str(ROOT/'zoekt-registry'),'serve',str(index/'core/index.zoekt'),str(index/'documents.json'),str(dest)],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
   ready=proc.stdout.readline();require(bool(ready),'Zoekt failed before ready');ready=json.loads(ready);require(ready.get('ready'),'Zoekt not ready')
   # Independently validate the reconstructed paths and bytes before user search.
   for d in m['files']:
    p=dest/d['path'];require(p.is_file() and p.stat().st_size==d['size'] and digest(p)==d['sha256'],'Zoekt recovered document differs')
   data=None
  else:
   data=recover_glyph(index,reg['engine'],m,dest,t)
   binary=ROOT/('glyph/build/rlbwt_full_query_locate_server_v1' if reg['engine']=='stock' else 'compact-server')
   proc=subprocess.Popen([str(binary),*[str(index/'core'/p) for p in PARTS]],stdin=subprocess.PIPE,stdout=subprocess.PIPE,text=True,bufsize=1)
   proc.stdin.write('01\t0\n');proc.stdin.flush();p=proc.stdout.readline().split('\t')
   require(len(p)==9 and p[0]=='OK' and int(p[3])==m['file_count'],'native delimiter/startup probe differs')
  phase['recovery_load_seconds']=time.perf_counter()-t0
  phase['total_ready_seconds']=time.perf_counter()-start
  print(json.dumps({'ready':True,'engine':reg['engine'],'files_verified':len(m['files']),'manifest_sha256':args.expected_docs,'index_sha256':args.expected_index,'phases':phase}),flush=True)
  starts=[d['concat_start'] for d in m['files']]
  try:
   for line in sys.stdin:
    req=json.loads(line)
    if req.get('op')=='quit':break
    q=bytes.fromhex(req['hex']);limit=req.get('limit',65536)
    require(q and 0 not in q and 1 not in q and type(limit) is int and 0<=limit<=65536,'unsupported exact query/limit');q.decode('utf-8')
    if reg['engine']=='zoekt':
     proc.stdin.write(json.dumps(req)+'\n');proc.stdin.flush();line=proc.stdout.readline();require(bool(line),'Zoekt response missing');result=json.loads(line)
     require(result.get('ok') and result['byte_check'] and len(result['matches'])==min(limit,result['count']),'Zoekt cardinality differs')
    else:result=glyph_result(proc,q,limit,m,starts,data)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True),flush=True)
  finally:
   if proc and proc.poll() is None:
    try:proc.stdin.write('QUIT\n' if reg['engine']!='zoekt' else '{"op":"quit"}\n');proc.stdin.flush()
    except BrokenPipeError:pass
    proc.stdin.close();require(proc.wait()==0,'engine exit code differs')
def main():
 ap=argparse.ArgumentParser(description=__doc__);s=ap.add_subparsers(dest='cmd',required=True)
 b=s.add_parser('build');b.add_argument('--engine',choices=['stock','compact','zoekt'],required=True);b.add_argument('--source',required=True);b.add_argument('--out',required=True)
 q=s.add_parser('serve');q.add_argument('--index',required=True);q.add_argument('--expected-index',required=True);q.add_argument('--expected-docs',required=True);q.add_argument('--recovered',required=True)
 a=ap.parse_args();build(a) if a.cmd=='build' else serve(a)
if __name__=='__main__':
 try:main()
 except Exception as e:print(json.dumps({'ok':False,'reason':str(e),'type':type(e).__name__}),file=sys.stderr);sys.exit(1)
