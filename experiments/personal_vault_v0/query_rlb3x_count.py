#!/usr/bin/env python3
import argparse,bisect,json,lzma,mmap,struct,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
import rlbwt_query_v2 as q
MAGIC=b"RLB3X001"; HDR=struct.Struct("<8sIQQIIQ"); REC=struct.Struct("<QQQQ")
class Rank3X:
 def __init__(self,path):
  self.path=path;self.stream=path.open("rb");self.mm=mmap.mmap(self.stream.fileno(),0,access=mmap.ACCESS_READ)
  magic,ver,self.raw_length,self.run_count,self.block_runs,self.escape_symbol,self.blocks=HDR.unpack_from(self.mm,0)
  q.require(magic==MAGIC and ver==1,"bad RLB3X header");q.require(self.blocks>0,"zero blocks")
  self.dir=[REC.unpack_from(self.mm,HDR.size+i*REC.size) for i in range(self.blocks)]
  self.row_starts=[x[1] for x in self.dir];self.cache={};self.decoded_blocks=0;self.decoded_runs=0;self.scanned_symbols=0;self.rank_calls=0
  self.block_prefix=[]; counts=[0]*257
  for i in range(self.blocks):
   self.block_prefix.append(tuple(counts))
   for s,l in self._decode(i): counts[s]+=l
  q.require(sum(counts)==self.raw_length and counts[256]==1,"RLB3X histogram")
  self.frequencies=counts;self.C=[0]*257;self.C[256]=0;running=1
  for s in range(256):self.C[s]=running;running+=counts[s]
  q.require(running==self.raw_length,"C total")
  self.cache.clear();self.decoded_blocks=0;self.decoded_runs=0
 def close(self):self.mm.close();self.stream.close()
 def _uleb(self,b,p):
  v=0;s=0
  for _ in range(10):
   q.require(p<len(b),"truncated uleb");x=b[p];p+=1;v|=(x&127)<<s
   if not x&128:q.require(v>0,"zero run");return v,p
   s+=7
  raise q.QueryError("uleb too long")
 def _decode(self,i):
  if i in self.cache:return self.cache[i]
  rs,row,off,cl=self.dir[i];q.require(off+cl<=len(self.mm),"block outside file")
  b=lzma.decompress(self.mm[off:off+cl],format=lzma.FORMAT_XZ);p=0;r=[]
  while p<len(b):
   h=b[p];p+=1
   if h==self.escape_symbol:
    q.require(p<len(b),"escape");tag=b[p];p+=1;q.require(tag in (0,1),"tag");h=self.escape_symbol if tag==0 else 256
   l,p=self._uleb(b,p);r.append((h,l))
  expected=min(self.block_runs,self.run_count-rs);q.require(len(r)==expected,"run geometry")
  self.cache[i]=r;self.decoded_blocks+=1;self.decoded_runs+=len(r);return r
 def rank(self,symbol,position):
  q.require(0<=symbol<257 and 0<=position<=self.raw_length,"rank range");self.rank_calls+=1
  if position==self.raw_length:return self.frequencies[symbol]
  i=max(0,bisect.bisect_right(self.row_starts,position)-1);cur=self.row_starts[i];ans=self.block_prefix[i][symbol]
  for h,l in self._decode(i):
   take=min(l,position-cur)
   if h==symbol:ans+=take
   self.scanned_symbols+=take;cur+=take
   if cur>=position:break
  q.require(cur==position,"rank scan");return ans
 def backward_search(self,pattern):
  q.require(pattern,"empty pattern");l=0;r=self.raw_length
  for s in reversed(pattern):
   l=self.C[s]+self.rank(s,l);r=self.C[s]+self.rank(s,r)
   if l>=r:return l,l
  return l,r
def main():
 p=argparse.ArgumentParser();p.add_argument("--rlb3x",type=Path,required=True);p.add_argument("--pattern-hex",required=True);a=p.parse_args()
 rt=Rank3X(a.rlb3x)
 try:
  pat=bytes.fromhex(a.pattern_hex);t=time.perf_counter_ns();l,r=rt.backward_search(pat);elapsed=time.perf_counter_ns()-t
  print(json.dumps({"format":"GLYPH_RLB3X_COUNT_EXPERIMENT_V0","count":r-l,"fm_interval":[l,r],"pattern_hex":pat.hex(),"query_elapsed_ns":elapsed,"rank_calls":rt.rank_calls,"decoded_blocks":rt.decoded_blocks,"decoded_runs":rt.decoded_runs,"scanned_symbols":rt.scanned_symbols},sort_keys=True,separators=(",",":")))
 finally:rt.close()
if __name__=="__main__":main()
