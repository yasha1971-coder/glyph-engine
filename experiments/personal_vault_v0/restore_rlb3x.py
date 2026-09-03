#!/usr/bin/env python3
import argparse,bisect,hashlib,json,lzma,struct
from array import array
from pathlib import Path

MAGIC=b"RLB3X001"
HDR=struct.Struct("<8sIQQIIQ")
REC=struct.Struct("<QQQQ")

def req(c,m):
    if not c: raise SystemExit(m)

def unuleb(buf,p):
    v=0; shift=0
    for _ in range(10):
        req(p<len(buf),"truncated uleb")
        x=buf[p]; p+=1; v|=(x&127)<<shift
        if not x&128:
            req(v>0,"zero run")
            return v,p
        shift+=7
    raise SystemExit("uleb too long")

def decode_bwt(path):
    blob=path.read_bytes()
    req(len(blob)>=HDR.size,"RLB3X too small")
    magic,ver,rows,runs,block_runs,esc,blocks=HDR.unpack_from(blob,0)
    req(magic==MAGIC and ver==1,"bad RLB3X header")
    req(blocks>0 and block_runs>0,"bad RLB3X geometry")
    values=array('H')
    decoded_runs=0
    for i in range(blocks):
        run_start,row_start,off,cl=REC.unpack_from(blob,HDR.size+i*REC.size)
        req(off+cl<=len(blob),"RLB3X block outside file")
        raw=lzma.decompress(blob[off:off+cl],format=lzma.FORMAT_XZ)
        p=0
        while p<len(raw):
            h=raw[p]; p+=1
            if h==esc:
                req(p<len(raw),"truncated escape")
                tag=raw[p]; p+=1
                req(tag in (0,1),"bad escape tag")
                h=esc if tag==0 else 256
            ln,p=unuleb(raw,p)
            values.extend([h]*ln)
            decoded_runs+=1
    req(len(values)==rows,"decoded BWT row mismatch")
    req(decoded_runs==runs,"decoded run count mismatch")
    return values

def inverse_bwt(vals):
    n=len(vals)
    counts=[0]*257
    occ=array('I',[0])*n
    sentinel_row=None
    for i,s in enumerate(vals):
        if s==256:
            req(sentinel_row is None,"multiple sentinels")
            sentinel_row=i
        occ[i]=counts[s]
        counts[s]+=1
    req(counts[256]==1 and sentinel_row is not None,"sentinel cardinality")
    C=[0]*257; C[256]=0; running=1
    for s in range(256):
        C[s]=running; running+=counts[s]
    req(running==n,"C total mismatch")
    row=sentinel_row
    out=bytearray(n-1)
    for j in range(n-2,-1,-1):
        row=C[vals[row]]+occ[row]
        s=vals[row]
        req(s!=256,"early sentinel")
        out[j]=s
    return bytes(out)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("rlb3x",type=Path)
    ap.add_argument("out",type=Path)
    ap.add_argument("report",type=Path)
    a=ap.parse_args()
    vals=decode_bwt(a.rlb3x)
    restored=inverse_bwt(vals)
    a.out.write_bytes(restored)
    report={
      "format":"GLYPH_RLB3X_DIRECT_RESTORE_V0",
      "restored_bytes":len(restored),
      "restored_sha256":hashlib.sha256(restored).hexdigest(),
      "rlb2_not_used":True,
      "rlr2_not_used":True,
      "locate_not_required_for_restore":True
    }
    a.report.write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
