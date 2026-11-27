"""One-way experimental codec conversion; stock index and LOC2 remain supported.

Use the unmodified research RLB2/RLR2 writers rather than inventing a similarly
named format. All intermediate work is charged to candidate construction.
"""
from pathlib import Path
import array,hashlib,json,shutil,struct,sys,tempfile

def digest(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def convert(core,stock_tools,research_tools):
 core=Path(core)
 sys.path.insert(0,str(stock_tools));sys.path.insert(0,str(research_tools))
 from rlbwt_container_v1 import decode_rlbwt
 import rlbwt_container_v2 as rlb2
 import rlbwt_rank_v2 as rank2
 manifest=json.loads((core/'manifest.json').read_bytes())
 assert manifest['format']=='glyph-search-index-v2'
 before=digest(core/'locate_core_s128.bin')
 with tempfile.TemporaryDirectory() as t:
  t=Path(t);raw=t/'legacy.bwt';canon=t/'canonical.bwt'
  decode_rlbwt(core/'bwt.rlbwt',raw)
  fnv=rlb2.Fnv1a64();rows=0;sentinels=0
  with canon.open('wb') as dest,raw.open('rb') as src:
   dest.write(b'\0'*rlb2.BWT_HEADER.size)
   while block:=src.read(1<<20):
    a=array.array('H',(x if x else 256 for x in block));sentinels+=block.count(0);rows+=len(block)
    if sys.byteorder!='little':a.byteswap()
    data=a.tobytes();fnv.update(data);dest.write(data)
   if rows!=manifest['corpus_bytes']+1 or sentinels!=1:raise ValueError('legacy BWT rows/sentinel mismatch')
   dest.seek(0);dest.write(rlb2.BWT_HEADER.pack(rlb2.BWT_MAGIC,1,rows-1,rows,257,256,16,rows*2,fnv.value))
  rlb2.encode_file(canon,t/'bwt.rlbwt')
  rank2.build_rank(t/'bwt.rlbwt',t/'bwt.rlbwt.rank',8192)
  for name in ('bwt.rlbwt','bwt.rlbwt.rank'):shutil.copyfile(t/name,core/name)
 manifest['format']='glyph-search-index-experimental-rlr2-loc2-v0'
 manifest['bridge']='NUL-free physical0=logical256 sentinel-first; LOC2 unchanged'
 for name in manifest['files']:
  p=core/name;manifest['files'][name]={'bytes':p.stat().st_size,'sha256':digest(p)}
 (core/'manifest.json').write_text(json.dumps(manifest,sort_keys=True,indent=2)+'\n')
 if digest(core/'locate_core_s128.bin')!=before:raise ValueError('LOC2 changed')
 return {'locate_sha256':before,'locate_unchanged':True,'format':manifest['format']}

def recover_compact(rlb,out,helper,scratch):
 import rlbwt_container_v2 as rlb2
 import subprocess
 rlb=Path(rlb);scratch=Path(scratch)
 with tempfile.TemporaryDirectory(dir=scratch) as t:
  t=Path(t);canonical=t/'canonical';raw=t/'raw'
  rlb2.decode_file(rlb,canonical)
  with canonical.open('rb') as src,raw.open('wb') as dest:
   src.seek(rlb2.BWT_HEADER.size)
   while block:=src.read(1<<20):
    a=array.array('H');a.frombytes(block)
    if sys.byteorder!='little':a.byteswap()
    if any(x==0 or x>256 for x in a):raise ValueError('RLB2 bridge rejects source zero/invalid symbol')
    dest.write(bytes(x if x<256 else 0 for x in a))
  subprocess.run([str(helper),str(raw),str(out)],check=True)
