#!/usr/bin/env python3
"""Pinned-index №1 result + actual native №2 replay + immutable registry №3 map.
Not glyph-evidence canonical rebuild. Zoekt support is an experimental frontend.
"""
from pathlib import Path
import argparse,bisect,hashlib,importlib.util,json,os,subprocess,sys,tempfile,time
import registry as registry
ROOT=Path(__file__).resolve().parent
def require(x,reason):
 if not x:raise ValueError(reason)
def sha(p):return registry.digest(p)
def save(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,sort_keys=True,ensure_ascii=False,indent=2)+'\n')
def execute(cmd,label,state):
 start=time.perf_counter();p=subprocess.run(list(map(str,cmd)),capture_output=True,env=dict(os.environ,OMP_NUM_THREADS='1',GOMAXPROCS='1',PYTHONDONTWRITEBYTECODE='1'));log=Path(state)/'logs';log.mkdir(parents=True,exist_ok=True);(log/(label+'.stdout')).write_bytes(p.stdout);(log/(label+'.stderr')).write_bytes(p.stderr);save(log/(label+'.json'),{'command':list(map(str,cmd)),'seconds':time.perf_counter()-start,'exit_code':p.returncode});require(p.returncode==0,label+' failed: '+p.stderr.decode(errors='replace')[-1200:]);return json.loads(p.stdout)
def mapped(offsets,q,m):
 starts=[d['concat_start'] for d in m['files']];result=[]
 for off in offsets:
  require(type(off)is int,'noninteger offset');i=bisect.bisect_right(starts,off)-1;require(i>=0,'offset before document');d=m['files'][i];require(d['concat_start']<=off and off+len(q)<=d['concat_end_exclusive'],'offset crosses file boundary');result.append({'path':d['path'],'offset':off-d['concat_start']})
 return result
def original_compact_replay(index,request,query,state):
 cmd=[str(ROOT/'compact-server'),*[str(index/'core'/x) for x in registry.PARTS]];p=subprocess.run(cmd,input=(query.hex()+'\t'+str(request['limit'])+'\nQUIT\n').encode(),capture_output=True);save(Path(state)/'logs/original-RLB2-native.json',{'command':cmd,'exit_code':p.returncode});(Path(state)/'logs/original-RLB2-native.stdout').write_bytes(p.stdout);(Path(state)/'logs/original-RLB2-native.stderr').write_bytes(p.stderr);require(p.returncode==0,'original RLB2 reader rejected');f=p.stdout.decode().rstrip('\n').split('\t');require(len(f)==9 and f[0]=='OK','bad original RLB2 response');return int(f[3]),list(map(int,f[8].split(','))) if f[8] else []
def prepare(t,state,scratch):
 kind=t['engine'];query=(ROOT/'query.bin').read_bytes();require(hashlib.sha256(query).hexdigest()==t['query_sha256'] and query.hex()==t['query_hex'],'independent query SHA mismatch');require(type(t['limit'])is int and 1<=t['limit']<=65536,'invalid independent limit');(Path(state)/'logs').mkdir(parents=True,exist_ok=True)
 if kind in ['mover','sdsl']:
  import registry_native
  index,reg,m=registry_native.snapshot(ROOT/'index',t['registry_sha256'],t['documents_sha256'],scratch)
 else:index,reg,m=registry.snapshot(ROOT/'index',t['registry_sha256'],t['documents_sha256'],scratch)
 require(reg['engine']==kind and m['corpus_sha256']==t['corpus_sha256'] and m['corpus_bytes']==t['corpus_bytes'],'independent corpus identity mismatch');binding={x:t[x] for x in ['engine','registry_sha256','documents_sha256','corpus_sha256','corpus_bytes','core_sha256']};cache=Path(state)/'cache';body=cache/'number1/corpus.bin'
 if cache.exists():
  require((cache/'source_binding.json').is_file() and json.loads((cache/'source_binding.json').read_bytes())==binding,'original-index/cache binding mismatch');require(body.is_file() and body.stat().st_size==m['corpus_bytes'] and sha(body)==t['corpus_sha256'],'cached corpus SHA/length mismatch before search')
 os.environ['GLYPH_REGISTRY_MANIFEST']=str(index/'documents.json');return index,m,query,binding
def n1(kind,mode,index,t,queryfile,result,state):
 engine='glyph' if kind in ['stock','compact'] else kind;tool=ROOT/'glyph' if engine=='glyph' else ROOT/'native';args=['--engine',engine,'--tool',str(tool),'--index',str(index),'--expect-index-sha256',t['core_sha256'],'--expect-corpus-sha256',t['corpus_sha256'],'--corpus-bytes',str(t['corpus_bytes']),'--query-file',str(queryfile),'--limit',str(t['limit'])]
 if kind=='zoekt':
  # Same unchanged pinned-profile implementation; published argparse choices
  # do not include Zoekt. This frontend extension is disclosed in every receipt.
  sys.path.insert(0,str(ROOT/'glyph/tools'));import index_result_v1 as core
  a=argparse.Namespace(engine='zoekt',tool=tool,index=index,expect_index_sha256=t['core_sha256'],expect_corpus_sha256=t['corpus_sha256'],corpus_bytes=t['corpus_bytes'],query_file=queryfile,limit=t['limit'],out=result,result=result,cache=Path(state)/'cache/number1')
  with tempfile.TemporaryDirectory(prefix='zoekt-index-result-') as temp:r=(core.create if mode=='create' else core.verify)(a,Path(temp))
  save(Path(state)/'logs'/('number1-'+mode+'.json'),{'ok':r['ok'],'implementation_sha256':sha(ROOT/'glyph/tools/index_result_v1.py'),'frontend_extension':'engine=zoekt; unchanged create/verify implementation','native_replay':r.get('native_replay')});return r
 cmd=[sys.executable,str(ROOT/'glyph/glyph-evidence'),'create-index' if mode=='create' else 'verify-index'] if engine=='glyph' else [sys.executable,str(ROOT/'glyph/tools/index_result_v1.py'),mode]
 if mode=='create':args+=['--out',str(result)]
 else:args+=['--result',str(result),'--cache',str(Path(state)/'cache/number1')]
 return execute(cmd+args,'number1-'+mode,state)
def run(a):
 t=json.loads(Path(a.trusted).read_bytes());state=Path(a.state);state.mkdir(parents=True,exist_ok=True);phases={};start=time.perf_counter();kind=t['engine']
 with tempfile.TemporaryDirectory(prefix='registry-result-') as temp:
  temp=Path(temp);t0=time.perf_counter();index,m,q,binding=prepare(t,state,temp);phases['original_snapshot_s']=time.perf_counter()-t0
  core_result=ROOT/'results/index-result.json';claim_path=ROOT/'results/registry-result.json'
  if a.mode=='verify':
   core=json.loads(core_result.read_bytes());claim=json.loads(claim_path.read_bytes());expected={'profile':'glyph-registry-index-result-v1','engine':kind,'registry_sha256':t['registry_sha256'],'documents_sha256':t['documents_sha256'],'corpus_sha256':t['corpus_sha256'],'query_sha256':t['query_sha256'],'limit':t['limit'],'index_result_sha256':sha(core_result),'count':core['count'],'matches':mapped(core['offsets'],q,m)};require(claim==expected,'registry result/original index/path binding mismatch')
  t0=time.perf_counter();original=None
  if kind=='compact':
   original=original_compact_replay(index,t,q,state);from export_release import export
   target=temp/'release-index';e=export(index/'core',target,ROOT/'glyph',ROOT/'research',t['export_manifest_raw'],t['core_sha256']);save(state/'logs/export.json',e);target_index=target
  else:target_index=index/'core' if kind=='stock' else index/'core'/('index.zoekt' if kind=='zoekt' else 'index.native')
  phases['compatibility_export_and_original_replay_s']=time.perf_counter()-t0;t0=time.perf_counter();result=n1(kind,'create' if a.mode=='create' else 'verify',target_index,t,ROOT/'query.bin',core_result,state);phases['number1_s']=time.perf_counter()-t0;core=json.loads(core_result.read_bytes());matches=mapped(core['offsets'],q,m)
  if original is not None:require(original==(core['count'],core['offsets']),'original RLB2/result exported native replay differs')
  if a.mode=='create':
   claim={'profile':'glyph-registry-index-result-v1','engine':kind,'registry_sha256':t['registry_sha256'],'documents_sha256':t['documents_sha256'],'corpus_sha256':t['corpus_sha256'],'query_sha256':t['query_sha256'],'limit':t['limit'],'index_result_sha256':sha(core_result),'count':core['count'],'matches':matches};save(claim_path,claim)
   if a.search_result:
    searched=json.loads(Path(a.search_result).read_bytes());require(searched['count']==core['count'] and searched['matches']==matches and searched['byte_check'],'№2 search and №1 artifact/path mapping differ')
   return {'ok':True,'engine':kind,'mode':'create','count':core['count'],'matches':matches,'phases':phases,'profile':'glyph-index-result-v1','registry_binding':True}
  t0=time.perf_counter();body=state/'cache/number1/corpus.bin';data=body.read_bytes();require(len(data)==m['corpus_bytes'] and hashlib.sha256(data).hexdigest()==t['corpus_sha256'],'final corpus SHA mismatch');dest=state/'restored';require(not dest.exists(),'restored tree already exists');dest.mkdir(parents=True)
  for d in m['files']:
   chunk=data[d['concat_start']:d['concat_end_exclusive']];require(len(chunk)==d['size'] and hashlib.sha256(chunk).hexdigest()==d['sha256'] and data[d['delimiter_offset']]==1,'file SHA/boundary mismatch');p=dest/d['path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(chunk);require(p.stat().st_size==d['size'] and sha(p)==d['sha256'],'written file SHA/length mismatch')
  for match in matches:
   p=dest/match['path'];require(p.is_file(),'mapped path absent')
   with p.open('rb') as f:f.seek(match['offset']);require(f.read(len(q))==q,'mapped byte mismatch')
  save(state/'cache/source_binding.json',binding);phases['restore_tree_and_all_SHA_s']=time.perf_counter()-t0
  require(result['ok'] and result['native_replay'] and result['byte_check'],'№1 verifier did not pass')
  return {'ok':True,'engine':kind,'count':core['count'],'matches':matches,'files_verified':m['file_count'],'byte_check':True,'native_replay':True,'canonical_runtime_rebuilt':False,'original_registry_sha256':t['registry_sha256'],'documents_sha256':t['documents_sha256'],'corpus_sha256':t['corpus_sha256'],'verified_core_sha256':t['core_sha256'],'number1_profile':'glyph-index-result-v1','zoekt_frontend_extension':kind=='zoekt','RLB2_compatibility_export':kind=='compact','phases':phases,'verify_total_s':time.perf_counter()-start}
def main():
 ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('mode',choices=['create','verify']);ap.add_argument('--trusted',required=True);ap.add_argument('--state',required=True);ap.add_argument('--search-result');a=ap.parse_args()
 try:print(json.dumps(run(a),ensure_ascii=False,sort_keys=True),flush=True);return 0
 except Exception as e:print(json.dumps({'ok':False,'error':str(e),'type':type(e).__name__}),file=sys.stderr);return 1
if __name__=='__main__':sys.exit(main())
