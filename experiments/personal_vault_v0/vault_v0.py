#!/usr/bin/env python3
import argparse, hashlib, json, struct
from array import array
from pathlib import Path

BWT_HEADER = struct.Struct("<8sIQQIIIQQ")
BWT_MAGIC=b"GLYBWT1\0"
SA_HEADER_BYTES=64

def sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()

def files(root):
    return sorted(p for p in root.rglob("*") if p.is_file() and p.name!="SHA256SUMS")

def pack(root,out):
    objs=[]; pos=0
    with out.open("wb") as w:
        for i,p in enumerate(files(root)):
            data=p.read_bytes(); w.write(data)
            objs.append({"id":i,"path":str(p.relative_to(root)),"offset":pos,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
            pos+=len(data)
    return {"format":"GLYPH_PERSONAL_VAULT_V0_OBJECT_MAP","corpus_bytes":pos,"objects":objs}

def build_loc(sa_path,row_count,step,out):
    raw=sa_path.read_bytes()
    if len(raw)!=SA_HEADER_BYTES+row_count*4: raise SystemExit("SA geometry mismatch")
    vals=memoryview(raw)[SA_HEADER_BYTES:].cast("I")
    with out.open("wb") as f:
        samples=list(range(0,row_count,step))
        f.write(b"LOC1"); f.write(struct.pack("<QIQ",row_count,step,len(samples)))
        for row in samples: f.write(struct.pack("<QQ",row,int(vals[row])))

def restore_bwt(bwt_path,out):
    raw=bwt_path.read_bytes()
    vals=memoryview(raw)[BWT_HEADER.size:].cast("H")
    n=len(vals); counts=[0]*257; occ=array("I",[0])*n
    for i,s in enumerate(vals):
        occ[i]=counts[s]; counts[s]+=1
    if counts[256]!=1: raise SystemExit("sentinel cardinality")
    C=[0]*257; C[256]=0; run=1
    for s in range(256): C[s]=run; run+=counts[s]
    row=next(i for i,s in enumerate(vals) if s==256)
    restored=bytearray(n-1)
    for j in range(n-2,-1,-1):
        row=C[vals[row]]+occ[row]
        s=vals[row]
        if s==256: raise SystemExit("early sentinel")
        restored[j]=s
    out.write_bytes(restored)

def verify_objects(corpus,map_path,root):
    m=json.loads(map_path.read_text())
    data=corpus.read_bytes()
    for o in m["objects"]:
        got=data[o["offset"]:o["offset"]+o["bytes"]]
        src=root/o["path"]
        if hashlib.sha256(got).hexdigest()!=o["sha256"] or got!=src.read_bytes(): raise SystemExit("object restore mismatch "+o["path"])

def make_queries(corpus,map_path,out):
    data=corpus.read_bytes(); m=json.loads(map_path.read_text()); qs=[]
    for o in m["objects"]:
        b=data[o["offset"]:o["offset"]+o["bytes"]]
        if not b: continue
        for size in (16,8,4):
            if len(b)>=size:
                candidates=[b[:size],b[len(b)//2:len(b)//2+size],b[-size:]]
                unique=[x for x in candidates if x and data.count(x)==1]
                if unique:
                    q=unique[0]; qs.append({"object_id":o["id"],"path":o["path"],"pattern_hex":q.hex(),"expected_offset":data.find(q),"bytes":len(q)}); break
    out.write_text(json.dumps({"format":"GLYPH_PERSONAL_VAULT_V0_QUERY_SET","queries":qs},sort_keys=True,separators=(",",":"))+"\n")

def verify_query(result_path,query_path):
    r=json.loads(result_path.read_text()); q=json.loads(query_path.read_text())
    if r["count"]!=1 or r["offsets"]!=[q["expected_offset"]]: raise SystemExit("query mismatch")

def boundary_tests(corpus,map_path,out):
    data=corpus.read_bytes(); m=json.loads(map_path.read_text()); cases=[]
    for a,b in zip(m["objects"],m["objects"][1:]):
        boundary=a["offset"]+a["bytes"]
        for k in range(1,9):
            if boundary>=k and boundary+k<=len(data):
                pat=data[boundary-k:boundary+k]
                starts=[]; pos=0
                while True:
                    p=data.find(pat,pos)
                    if p<0: break
                    starts.append(p); pos=p+1
                if starts==[boundary-k]:
                    cases.append({"pattern_hex":pat.hex(),"forbidden_offset":boundary-k,"left_object":a["id"],"right_object":b["id"]}); break
    out.write_text(json.dumps({"format":"GLYPH_PERSONAL_VAULT_V0_BOUNDARY_SET","cases":cases},sort_keys=True,separators=(",",":"))+"\n")

def main():
    ap=argparse.ArgumentParser(); sp=ap.add_subparsers(dest="cmd",required=True)
    p=sp.add_parser("pack"); p.add_argument("root",type=Path); p.add_argument("corpus",type=Path); p.add_argument("map",type=Path)
    p=sp.add_parser("locate"); p.add_argument("sa",type=Path); p.add_argument("row_count",type=int); p.add_argument("step",type=int); p.add_argument("out",type=Path)
    p=sp.add_parser("restore-bwt"); p.add_argument("bwt",type=Path); p.add_argument("out",type=Path)
    p=sp.add_parser("verify-objects"); p.add_argument("corpus",type=Path); p.add_argument("map",type=Path); p.add_argument("root",type=Path)
    p=sp.add_parser("queries"); p.add_argument("corpus",type=Path); p.add_argument("map",type=Path); p.add_argument("out",type=Path)
    p=sp.add_parser("verify-query"); p.add_argument("result",type=Path); p.add_argument("query",type=Path)
    p=sp.add_parser("boundaries"); p.add_argument("corpus",type=Path); p.add_argument("map",type=Path); p.add_argument("out",type=Path)
    a=ap.parse_args()
    if a.cmd=="pack":
        m=pack(a.root,a.corpus); a.map.write_text(json.dumps(m,sort_keys=True,separators=(",",":"))+"\n")
    elif a.cmd=="locate": build_loc(a.sa,a.row_count,a.step,a.out)
    elif a.cmd=="restore-bwt": restore_bwt(a.bwt,a.out)
    elif a.cmd=="verify-objects": verify_objects(a.corpus,a.map,a.root)
    elif a.cmd=="queries": make_queries(a.corpus,a.map,a.out)
    elif a.cmd=="verify-query": verify_query(a.result,a.query)
    elif a.cmd=="boundaries": boundary_tests(a.corpus,a.map,a.out)
if __name__=="__main__": main()
