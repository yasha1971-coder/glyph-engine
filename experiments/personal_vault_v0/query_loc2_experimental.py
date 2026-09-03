#!/usr/bin/env python3
import argparse,json,mmap,struct,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
import rlbwt_query_v2 as q

MAGIC=b"LOC2"; VERSION=1; HEADER=struct.Struct("<4sIQIIQ")

class LocateCore2:
    def __init__(self,path):
        self.path=path; self.stream=path.open("rb")
        self.map=mmap.mmap(self.stream.fileno(),0,access=mmap.ACCESS_READ)
        magic,ver,self.sa_size,self.sample_step,self.width,self.sampled_count=HEADER.unpack(self.map[:HEADER.size])
        q.require(magic==MAGIC and ver==VERSION,"bad LOC2 header")
        q.require(self.sample_step>0 and self.width in (4,8),"bad LOC2 geometry")
        q.require(path.stat().st_size==HEADER.size+self.sampled_count*self.width,"LOC2 size mismatch")
    def close(self):
        if getattr(self,"map",None) is not None: self.map.close(); self.map=None
        if getattr(self,"stream",None) is not None: self.stream.close(); self.stream=None
    def sampled_sa(self,row):
        if row%self.sample_step: return None
        i=row//self.sample_step
        if i>=self.sampled_count: return None
        off=HEADER.size+i*self.width
        v=struct.unpack_from("<I" if self.width==4 else "<Q",self.map,off)[0]
        q.require(v<self.sa_size,"LOC2 suffix outside SA")
        return v

class Runtime2(q.QueryRuntime):
    def __init__(self,rlb2,rlr2,loc2):
        self.rank=q.RankV2(rlb2,rlr2); self.locate=LocateCore2(loc2)
        q.require(self.locate.sa_size==self.rank.raw_length,"LOC2/RLR2 rows")
        self.corpus_bytes=self.rank.raw_length-1

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rlb2",type=Path,required=True); p.add_argument("--rank-index",type=Path,required=True)
    p.add_argument("--locate-core",type=Path,required=True); p.add_argument("--pattern-hex",required=True)
    p.add_argument("--max-offsets",type=int,default=-1); a=p.parse_args()
    rt=Runtime2(a.rlb2,a.rank_index,a.locate_core)
    try: r=rt.query(bytes.fromhex(a.pattern_hex),a.max_offsets)
    finally: rt.close()
    r["locate_format"]="LOC2_EXPERIMENTAL_V1"
    print(json.dumps(r,sort_keys=True,separators=(",",":")))
if __name__=="__main__": main()
