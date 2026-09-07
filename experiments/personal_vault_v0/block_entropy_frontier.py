#!/usr/bin/env python3
import argparse,bz2,json,lzma,mmap,struct,zlib
from pathlib import Path
BWT_HEADER=struct.Struct("<8sIQQIIIQQ")
def uleb(v):
 o=bytearray()
 while True:
  b=v&127; v>>=7
  if v: o.append(b|128)
  else: o.append(b); return bytes(o)
def emit_head(x,esc=0):
 if x==256:return bytes((esc,1))
 if x==esc:return bytes((esc,0))
 return bytes((x,))
def main():
 p=argparse.ArgumentParser();p.add_argument("corpus",type=Path);p.add_argument("bwt",type=Path);p.add_argument("out",type=Path);a=p.parse_args()
 n=a.corpus.stat().st_size; runs=[]; prev=None; ln=0
 with a.bwt.open("rb") as f:
  mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ);v=memoryview(mm)[BWT_HEADER.size:].cast("H")
  for y0 in v:
   y=int(y0)
   if prev is None:prev=y;ln=1
   elif y==prev:ln+=1
   else:runs.append((prev,ln));prev=y;ln=1
  if prev is not None:runs.append((prev,ln))
  v.release();mm.close()
 # Choose globally cheapest escape by run-head occurrence.
 freq=[0]*256
 for h,_ in runs:
  if h<256:freq[h]+=1
 esc=min(range(256),key=lambda x:(freq[x],x))
 def raw_chunk(chunk):
  o=bytearray()
  for h,l in chunk:o+=emit_head(h,esc)+uleb(l)
  return bytes(o)
 models=[]
 for br in (1024,2048,4096,8192,16384,32768,65536):
  sums={"raw":0,"deflate9":0,"bz2_9":0,"xz6":0}; blocks=0
  for i in range(0,len(runs),br):
   raw=raw_chunk(runs[i:i+br]);blocks+=1;sums["raw"]+=len(raw)
   sums["deflate9"]+=len(zlib.compress(raw,9))
   sums["bz2_9"]+=len(bz2.compress(raw,9))
   sums["xz6"]+=len(lzma.compress(raw,preset=6))
  # 24 bytes/block model: compressed offset, BWT row start, run start.
  frame=blocks*24+160
  models.append({"runs_per_block":br,"blocks":blocks,**{k+"_bytes":v+frame for k,v in sums.items()},
    **{k+"_ratio":(v+frame)/n for k,v in sums.items()}})
 report={"format":"GLYPH_PERSONAL_VAULT_BLOCK_ENTROPY_FRONTIER_V0","status":"MEASURED_REAL_CODEC_PAYLOAD_MODEL_NOT_QUERY_IMPLEMENTATION",
 "source_bytes":n,"runs":len(runs),"escape_symbol":esc,"models":models,
 "non_claims":["Codec blocks are measured real compressed bytes plus an explicit 24-byte-per-block navigation budget.","No rank/locate implementation uses these blocks yet.","Codec decoder latency and canonical cross-version encoding are not yet accepted contracts."]}
 a.out.write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
 print(json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
