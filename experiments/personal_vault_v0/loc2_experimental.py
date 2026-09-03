#!/usr/bin/env python3
import argparse, mmap, struct, sys
from pathlib import Path

SA_HEADER_BYTES=60
MAGIC=b"LOC2"
VERSION=1
HEADER=struct.Struct("<4sIQIIQ")  # 32 bytes

def require(c,m):
    if not c: raise SystemExit(m)

def build(sa_path,row_count,step,out):
    raw=sa_path.read_bytes()
    require(len(raw)==SA_HEADER_BYTES+row_count*4,"SA geometry mismatch")
    vals=memoryview(raw)[SA_HEADER_BYTES:].cast("I")
    width=4 if row_count-1 < 2**32 else 8
    count=(row_count+step-1)//step
    with out.open("wb") as f:
        f.write(HEADER.pack(MAGIC,VERSION,row_count,step,width,count))
        for row in range(0,row_count,step):
            v=int(vals[row])
            f.write(struct.pack("<I" if width==4 else "<Q",v))
    print(f"LOC2_BUILD_OK bytes={out.stat().st_size} width={width} samples={count}")

def verify(sa_path,loc):
    raw=sa_path.read_bytes()
    with loc.open("rb") as f:
        h=f.read(HEADER.size)
        magic,ver,rows,step,width,count=HEADER.unpack(h)
        require(magic==MAGIC and ver==VERSION,"LOC2 header")
        require(width in (4,8),"LOC2 width")
        require(loc.stat().st_size==HEADER.size+count*width,"LOC2 geometry")
        sa=memoryview(raw)[SA_HEADER_BYTES:].cast("I")
        require(len(sa)==rows,"LOC2/SA rows")
        for i,row in enumerate(range(0,rows,step)):
            b=f.read(width); v=struct.unpack("<I" if width==4 else "<Q",b)[0]
            require(v==int(sa[row]),f"LOC2 sample mismatch {i}")
        require(f.read(1)==b"","LOC2 trailing")
    print(f"LOC2_VERIFY_OK samples={count}")

def main():
    p=argparse.ArgumentParser(); s=p.add_subparsers(dest="cmd",required=True)
    b=s.add_parser("build"); b.add_argument("sa",type=Path); b.add_argument("rows",type=int); b.add_argument("step",type=int); b.add_argument("out",type=Path)
    v=s.add_parser("verify"); v.add_argument("sa",type=Path); v.add_argument("loc",type=Path)
    a=p.parse_args()
    build(a.sa,a.rows,a.step,a.out) if a.cmd=="build" else verify(a.sa,a.loc)
if __name__=="__main__": main()
