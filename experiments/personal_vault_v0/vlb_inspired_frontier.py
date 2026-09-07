#!/usr/bin/env python3
import argparse,bz2,json,lzma,mmap,struct,zlib
from pathlib import Path

BWT_HEADER=struct.Struct("<8sIQQIIIQQ")

def uleb(v):
    out=bytearray()
    while True:
        b=v&127; v>>=7
        if v: out.append(b|128)
        else: out.append(b); return bytes(out)

def choose_escape(runs):
    freq=[0]*256
    for h,l in runs:
        if h<256: freq[h]+=1
    return min(range(256),key=lambda x:(freq[x],x))

def encode_runs(runs,esc):
    out=bytearray()
    for h,l in runs:
        if h==256: out+=bytes((esc,1))
        elif h==esc: out+=bytes((esc,0))
        else: out.append(h)
        out+=uleb(l)
    return bytes(out)

def slice_runs(runs,start,end):
    # runs: (symbol,length,row_start,row_end)
    out=[]
    for h,l,a,b in runs:
        if b<=start: continue
        if a>=end: break
        x=max(a,start); y=min(b,end)
        if x<y: out.append((h,y-x))
    return out

def partition(runs,n,ell,w):
    leaves=[]
    stack=[(0,n)]
    while stack:
        a,b=stack.pop()
        rr=slice_runs(runs,a,b)
        rc=len(rr)
        if (b-a)<=ell and rc<=w:
            leaves.append((a,b,rr)); continue
        if (b-a)<=1:
            leaves.append((a,b,rr)); continue
        mid=(a+b)//2
        stack.append((mid,b)); stack.append((a,mid))
    leaves.sort()
    return leaves

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("corpus",type=Path); ap.add_argument("bwt",type=Path); ap.add_argument("out",type=Path)
    a=ap.parse_args(); n=a.corpus.stat().st_size; rows=n+1
    runs=[]; prev=None; ln=0; start=0
    with a.bwt.open("rb") as f:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        vals=memoryview(mm)[BWT_HEADER.size:].cast("H")
        assert len(vals)==rows
        pos=0
        for y0 in vals:
            y=int(y0)
            if prev is None: prev=y;ln=1;start=0
            elif y==prev: ln+=1
            else:
                runs.append((prev,ln,start,pos)); prev=y;ln=1;start=pos
            pos+=1
        if prev is not None:runs.append((prev,ln,start,pos))
        vals.release(); mm.close()
    esc=choose_escape([(h,l) for h,l,_,_ in runs])
    configs=[]
    for ell in (65536,262144,1048576):
      for w in (512,1024,2048,4096,8192):
        leaves=partition(runs,rows,ell,w)
        sums={"raw":0,"deflate9":0,"bz2_9":0,"xz6":0}
        max_runs=0; max_symbols=0; total_runs=0
        for x,y,rr4 in leaves:
            rr=[(h,l) for h,l in rr4]
            raw=encode_runs(rr,esc)
            sums["raw"]+=len(raw)
            sums["deflate9"]+=len(zlib.compress(raw,9))
            sums["bz2_9"]+=len(bz2.compress(raw,9))
            sums["xz6"]+=len(lzma.compress(raw,preset=6))
            max_runs=max(max_runs,len(rr)); max_symbols=max(max_symbols,y-x); total_runs+=len(rr)
        # 32 bytes/leaf: row start/end, compressed offset/length. Conservative.
        frame=160+32*len(leaves)
        rec={"ell_symbols":ell,"w_runs":w,"leaves":len(leaves),
             "avg_runs_per_leaf":total_runs/len(leaves),"max_runs_per_leaf":max_runs,
             "avg_symbols_per_leaf":rows/len(leaves),"max_symbols_per_leaf":max_symbols}
        for k,v in sums.items():
            rec[k+"_bytes"]=v+frame; rec[k+"_ratio"]=(v+frame)/n
        configs.append(rec)
    best={}
    for codec in ("deflate9","bz2_9","xz6"):
        best[codec]=min(configs,key=lambda x:x[codec+"_bytes"])
    report={"format":"GLYPH_PERSONAL_VAULT_VLB_INSPIRED_FRONTIER_V0",
            "status":"MEASURED_REAL_CODEC_ADAPTIVE_PARTITION_MODEL_NOT_QUERY_IMPLEMENTATION",
            "source_bytes":n,"rows":rows,"runs":len(runs),"escape_symbol":esc,
            "configs":configs,"best":best,
            "non_claims":[
              "This is a VLB-inspired adaptive partition measurement, not an implementation of the 2026 VLB data structure.",
              "Leaf payloads are independently compressed and include a conservative 32-byte navigation budget per leaf.",
              "No rank/locate path uses these leaves yet.",
              "No latency or canonical-format claim is made."
            ]}
    a.out.write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps(report,sort_keys=True))
if __name__=="__main__":main()
