#!/usr/bin/env python3
import argparse,hashlib,json,lzma,mmap,struct
from pathlib import Path
BWT_HEADER=struct.Struct("<8sIQQIIIQQ")
MAGIC=b"RLB3X001"; VERSION=1
HDR=struct.Struct("<8sIQQIIQ") # magic,ver,rows,runs,block_runs,escape,blocks
REC=struct.Struct("<QQQQ") # run_start,row_start,file_off,comp_len

def req(c,m):
 if not c: raise SystemExit(m)
def uleb(v):
 o=bytearray()
 while 1:
  b=v&127;v>>=7
  if v:o.append(b|128)
  else:o.append(b);return bytes(o)
def unuleb(b,p):
 v=0;s=0
 while 1:
  req(p<len(b),"truncated uleb");x=b[p];p+=1;v|=(x&127)<<s
  if not x&128:return v,p
  s+=7;req(s<=63,"uleb overflow")
def main():
 ap=argparse.ArgumentParser();ap.add_argument("bwt",type=Path);ap.add_argument("out",type=Path);ap.add_argument("report",type=Path);ap.add_argument("--block-runs",type=int,default=8192);a=ap.parse_args()
 raw=a.bwt.read_bytes(); payload=raw[BWT_HEADER.size:]; req(len(payload)%2==0,"bwt payload")
 vals=memoryview(payload).cast("H"); rows=len(vals)
 runs=[];prev=None;ln=0
 for z0 in vals:
  z=int(z0)
  if prev is None:prev=z;ln=1
  elif z==prev:ln+=1
  else:runs.append((prev,ln));prev=z;ln=1
 if prev is not None:runs.append((prev,ln))
 vals.release()
 freq=[0]*256
 for h,l in runs:
  if h<256:freq[h]+=1
 esc=min(range(256),key=lambda x:(freq[x],x))
 blocks=[];row=0
 for i in range(0,len(runs),a.block_runs):
  rr=runs[i:i+a.block_runs]; enc=bytearray()
  for h,l in rr:
   if h==256:enc+=bytes((esc,1))
   elif h==esc:enc+=bytes((esc,0))
   else:enc.append(h)
   enc+=uleb(l)
  comp=lzma.compress(bytes(enc),format=lzma.FORMAT_XZ,preset=6)
  blocks.append((i,row,comp,rr));row+=sum(l for _,l in rr)
 header=HDR.pack(MAGIC,VERSION,rows,len(runs),a.block_runs,esc,len(blocks))
 directory_bytes=REC.size*len(blocks); off=len(header)+directory_bytes
 directory=bytearray();data=bytearray()
 for run_start,row_start,comp,rr in blocks:
  directory+=REC.pack(run_start,row_start,off,len(comp));data+=comp;off+=len(comp)
 a.out.write_bytes(header+directory+data)
 # independent decode from file bytes
 blob=a.out.read_bytes();magic,ver,drows,druns,br,desc,nb=HDR.unpack_from(blob,0)
 req(magic==MAGIC and ver==VERSION and drows==rows and druns==len(runs),"header")
 decoded=bytearray(); total_runs=0
 for bi in range(nb):
  rs,ro,fo,cl=REC.unpack_from(blob,HDR.size+bi*REC.size)
  enc=lzma.decompress(blob[fo:fo+cl],format=lzma.FORMAT_XZ);p=0
  while p<len(enc):
   h=enc[p];p+=1
   if h==desc:
    req(p<len(enc),"escape");tag=enc[p];p+=1;req(tag in (0,1),"tag");h=desc if tag==0 else 256
   l,p=unuleb(enc,p);req(l>0,"zero run");decoded+=struct.pack("<H",h)*l;total_runs+=1
 req(bytes(decoded)==payload,"roundtrip mismatch")
 report={"format":"GLYPH_RLB3X_FIXED_BLOCK_EXPERIMENT_V0","status":"MEASURED_ROUNDTRIP_EXPERIMENT",
 "rows":rows,"runs":len(runs),"block_runs":a.block_runs,"blocks":len(blocks),"escape_symbol":esc,
 "file_bytes":a.out.stat().st_size,"ratio_vs_bwt_rows":a.out.stat().st_size() if False else a.out.stat().st_size/ max(1,rows-1),
 "directory_bytes":directory_bytes,"roundtrip_bit_identical":True,
 "decoded_sha256":hashlib.sha256(decoded).hexdigest(),"canonical_bwt_payload_sha256":hashlib.sha256(payload).hexdigest(),
 "non_claims":["Experimental sidecar format; Runtime V2 unchanged.","No rank/count/locate uses RLB3X yet.","XZ canonical cross-version determinism is not claimed."]}
 a.report.write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n");print(json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
