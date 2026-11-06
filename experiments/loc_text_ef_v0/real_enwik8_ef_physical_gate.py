#!/usr/bin/env python3
"""Physical text-sampled EF prototype against the pinned original GLYPH SA.

Experimental format only. Neither this file nor its manifest is a released
authenticated runtime, complete locate verifier, or a portable proof bundle.
"""
import argparse
import array
import hashlib
import json
import mmap
import random
import struct
import sys
import urllib.request
import zipfile
import subprocess
from pathlib import Path

MAGIC=b"GL3EF001"
HEADER=struct.Struct("<8sQQQQBBHQQQQI") # 80 bytes
STRIDE=256
SOURCE_SHA="2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"
SA_SHA="a70951cba4f51915f9e4ef7d7e819c11c09dcaf57677cd573a9ab746e9aed727"
REF="c33cb493bc75f13ee4a904fabb190a29a196c94b"
URL="https://mattmahoney.net/dc/enwik8.zip"


def must(ok,s):
    if not ok: raise RuntimeError(s)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()


def ceildiv(a,b): return (a+b-1)//b


def bitset(buf,pos,width,value):
    if not width:return
    must(0<=value<(1<<width),"bitpack overflow")
    offset=pos>>3
    shift=pos&7
    size=ceildiv(shift+width,8)
    old=int.from_bytes(buf[offset:offset+size],"little")
    buf[offset:offset+size]=(old|(value<<shift)).to_bytes(size,"little")


def bitget(buf,pos,width):
    if not width:return 0
    off=pos>>3
    shift=pos&7
    size=ceildiv(shift+width,8)
    return (int.from_bytes(buf[off:off+size],"little")>>shift)&((1<<width)-1)


def build(sa_path,n,step,out):
    must(0<n<2**32 and step>0 and not out.exists(),"invalid geometry/output")
    m=(n-1)//step+1
    low=(n//m).bit_length()-1
    qw=(m-1).bit_length()
    hb=((n-1)>>low)+m+1
    lows=bytearray(ceildiv(m*low,8))
    high=bytearray(ceildiv(hb,8))
    q=bytearray(ceildiv(m*qw,8))
    checkpoints=array.array("I")
    seen=bytearray(ceildiv(n,8))
    with sa_path.open("rb") as file:
        mm=mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)
        try:
            must(len(mm)==60+4*n,"canonical SA file size")
            sample=0
            for row,(value,) in enumerate(struct.iter_unpack("<I",memoryview(mm)[60:])):
                must(value<n,"SA out of range")
                byte=value>>3; mask=1<<(value&7)
                must(not(seen[byte]&mask),"duplicate SA offset")
                seen[byte]|=mask
                if value%step==0:
                    must(sample<m,"sample overflow")
                    highpos=(row>>low)+sample
                    high[highpos>>3]|=1<<(highpos&7)
                    bitset(lows,sample*low,low,row&((1<<low)-1))
                    bitset(q,sample*qw,qw,value//step)
                    if sample%STRIDE==0:
                        must(highpos<2**32,"u32 checkpoint overflow")
                        checkpoints.append(highpos)
                    sample+=1
            must(sample==m,"wrong number of samples")
        finally:
            mm.close()
    must(all(b==255 for b in seen[:-1]),"SA is not a permutation")
    rem=n%8
    must(seen[-1]==((1<<rem)-1 if rem else 255),"SA permutation tail")
    must(len(checkpoints)==ceildiv(m,STRIDE),"missing select checkpoints")
    if sys.byteorder!="little": checkpoints.byteswap()
    cp=checkpoints.tobytes()
    header=HEADER.pack(MAGIC,n,step,m,hb,low,qw,STRIDE,
                       len(lows),len(high),len(q),len(cp),0)
    with out.open("xb") as f:
        for part in (header,lows,high,q,cp): f.write(part)
    return {"n":n,"step":step,"m":m,"low":low,"q_width":qw,"high_bits":hb,
            "low_bytes":len(lows),"high_bytes":len(high),"quotient_bytes":len(q),
            "checkpoint_bytes":len(cp),"header_bytes":len(header)}


def verify(sa_path,out,expect):
    data=out.read_bytes()
    must(len(data)>=HEADER.size,"LOC3 truncated")
    (magic,n,step,m,hb,low,qw,stride,lbs,hbs,qbs,cbs,flags)=HEADER.unpack_from(data)
    must(magic==MAGIC and flags==0 and stride==STRIDE,"invalid LOC3 header")
    must(n==expect["n"] and m==expect["m"] and step==expect["step"],
         "incorrect file identity geometry")
    must(low==expect["low"] and qw==expect["q_width"] and hb==expect["high_bits"],
         "noncanonical sampling widths")
    must([lbs,hbs,qbs,cbs]==[expect["low_bytes"],expect["high_bytes"],
                            expect["quotient_bytes"],expect["checkpoint_bytes"]],
         "unexpected section lengths")
    must(len(data)==HEADER.size+lbs+hbs+qbs+cbs,"trailing/truncated LOC3 data")
    at=HEADER.size
    lows=data[at:at+lbs];at+=lbs
    high=data[at:at+hbs];at+=hbs
    q=data[at:at+qbs];at+=qbs
    cp=data[at:at+cbs]
    must(sum(int(x).bit_count() for x in high)==m,"invalid high cardinality")
    # Full independent Elias--Fano select using a streaming high-bit scan.
    # No unaccounted select table, and every saved quotient is checked vs SA.
    with sa_path.open("rb") as file:
        mm=mmap.mmap(file.fileno(),0,access=mmap.ACCESS_READ)
        try:
            sample=0;prev=-1
            for hi_pos in range(hb):
                if (high[hi_pos>>3]>>(hi_pos&7))&1:
                    row=((hi_pos-sample)<<low)|bitget(lows,sample*low,low)
                    pos=bitget(q,sample*qw,qw)*step
                    must(prev<row<n,"EF row order/range")
                    must(struct.unpack_from("<I",mm,60+4*row)[0]==pos,
                         "EF sampled SA mismatch")
                    if sample%STRIDE==0:
                        stored=struct.unpack_from("<I",cp,4*(sample//STRIDE))[0]
                        must(stored==hi_pos,"invalid select checkpoint")
                    prev=row;sample+=1
            must(sample==m,"wrong sample count on independent verifier")
        finally:mm.close()
    # Exercise candidate header/geometry corruption without altering source.
    bad=bytearray(data)
    bad[0]^=1
    must(HEADER.unpack_from(bad)[0]!=MAGIC,"magic mutation not detectable")
    bad=bytearray(data)
    bad[HEADER.size-4]^=1
    must(HEADER.unpack_from(bad)[-1]!=0,"flag mutation not detectable")
    return {"sampled_sa_records_verified":m,"corruption_geometry_cases":2,
            "actual_locate_file_bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()}


def main():
    a=argparse.ArgumentParser()
    a.add_argument("--reference",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    opt=a.parse_args()
    ref=opt.reference.resolve(); out=opt.output.resolve()
    must(not out.exists(),"refuse existing output directory")
    must(subprocess.check_output(["git","-C",str(ref),"rev-parse","HEAD"],text=True).strip()==REF,
         "wrong frozen GLYPH source")
    out.mkdir(parents=True)
    record={"status":"BLOCKED","schema":"LOC_TEXT_EF_ENWIK8_SOURCE_PINNED_RESEARCH_V0"}
    try:
        z=out/"enwik8.zip"
        with urllib.request.urlopen(URL,timeout=180) as src,z.open("wb") as dst:
            while True:
                part=src.read(1<<20)
                if not part:break
                dst.write(part)
        with zipfile.ZipFile(z) as archive:
            must(archive.namelist()==["enwik8"],"unexpected ZIP entry")
            with archive.open("enwik8") as src,(out/"enwik8").open("wb") as dst:
                while True:
                    part=src.read(1<<20)
                    if not part:break
                    dst.write(part)
        raw=out/"enwik8";z.unlink()
        must(raw.stat().st_size==100000000 and sha(raw)==SOURCE_SHA,"source SHA mismatch")
        sa=out/"original.sa"
        subprocess.run([str(ref/"build/build_sa_binary_v1"),str(raw),str(sa)],
                       check=True,capture_output=True,text=True,timeout=180)
        must(sa.stat().st_size==400000064 and sha(sa)==SA_SHA,"original SA SHA mismatch")
        n=100000001;step=128
        artifact=out/"candidate.loc-text-ef"
        geometry=build(sa,n,step,artifact)
        checked=verify(sa,artifact,geometry)
        # Never count 400MB construction SA or 100MB input in locate-only size.
        record={"status":"REAL_ENWIK8_LOC_TEXT_EF_SAMPLE_PARITY_PASS_RESEARCH",
                "reference_commit":REF,"source_sha256":SOURCE_SHA,"sa_sha256":SA_SHA,
                "geometry":geometry,"verification":checked,
                "historical_loc2_file_bytes":3125036,
                "observed_locate_file_saving_bytes":3125036-artifact.stat().st_size,
                "disclaimers":[
                    "No actual RLB2/RLR2 query reader or native speed comparison in this gate.",
                    "No GLYPH P/R/O proof bundle, currentness witness or source-extract proof.",
                    "Bounded-LF proof was separately checked in exhaustive small binary experiment.",
                    "Research file is NOT a canonical frozen LOC2/LOC3 format.",
                    "Build-time original SA and source are not retained runtime and not uploaded.",
                ]}
    except Exception as e:
        record["error"]=repr(e)
    (out/"result.json").write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:record.get(k) for k in
                      ("status","error","verification","observed_locate_file_saving_bytes")},
                     sort_keys=True))
    return 0 if record["status"].startswith("REAL_ENWIK8_") else 2


if __name__=="__main__":
    sys.exit(main())
