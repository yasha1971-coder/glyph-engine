#!/usr/bin/env python3
"""Research-only replacement LOC reader for the original RankV2/QueryRuntime.

Uses physically accounted Elias–Fano + quotient values from LOC-TEXT-EF.
Does not bind this format into an approved GLYPH manifest or evidence bundle.
"""
import bisect
import mmap
import struct
from pathlib import Path

MAGIC=b"GL3EF001"
HEADER=struct.Struct("<8sQQQQBBHQQQQI")
CHECKPOINT_STRIDE=256


class EFError(ValueError):
    pass


def require(ok,msg):
    if not ok:
        raise EFError(msg)


def ceildiv(n,d):
    return (n+d-1)//d


def getbits(data,pos,width):
    if width==0:
        return 0
    off=pos>>3
    shift=pos&7
    chunk=ceildiv(shift+width,8)
    return (int.from_bytes(data[off:off+chunk],"little")>>shift)&((1<<width)-1)


class LocateEF:
    def __init__(self,path):
        self.path=Path(path)
        self.file=self.path.open("rb")
        self.mem=mmap.mmap(self.file.fileno(),0,access=mmap.ACCESS_READ)
        try:
            require(len(self.mem)>=HEADER.size,"LOC-TEXT-EF truncated header")
            (magic,self.sa_size,self.sample_step,self.m,self.highbits,
             self.low,self.width,self.stride,lbs,hbs,qbs,cbs,flags)=HEADER.unpack_from(self.mem)
            require(magic==MAGIC and flags==0,"invalid LOC-TEXT-EF header")
            require(self.sa_size>0 and self.sample_step>0,"zero dimensions")
            require(self.m==(self.sa_size-1)//self.sample_step+1,"wrong sample count")
            require(self.low==(self.sa_size//self.m).bit_length()-1,"noncanonical low")
            require(self.width==(self.m-1).bit_length(),"noncanonical quotient")
            require(self.highbits==((self.sa_size-1)>>self.low)+self.m+1,
                    "noncanonical high bits")
            require(self.stride==CHECKPOINT_STRIDE,"unsupported stride")
            require(lbs==ceildiv(self.m*self.low,8)
                    and hbs==ceildiv(self.highbits,8)
                    and qbs==ceildiv(self.m*self.width,8)
                    and cbs==4*ceildiv(self.m,self.stride),"invalid section geometry")
            require(len(self.mem)==HEADER.size+lbs+hbs+qbs+cbs,
                    "LOC-TEXT-EF trailing/truncated bytes")
            pos=HEADER.size
            self.lows=memoryview(self.mem)[pos:pos+lbs];pos+=lbs
            self.high=memoryview(self.mem)[pos:pos+hbs];pos+=hbs
            self.values=memoryview(self.mem)[pos:pos+qbs];pos+=qbs
            self.cp=memoryview(self.mem)[pos:pos+cbs]
            require(sum(int(b).bit_count() for b in self.high)==self.m,
                    "high bit cardinality mismatch")
        except Exception:
            self.close()
            raise

    def close(self):
        for name in ("lows","high","values","cp"):
            view=getattr(self,name,None)
            if view is not None:
                view.release()
                setattr(self,name,None)
        if getattr(self,"mem",None) is not None:
            self.mem.close();self.mem=None
        if getattr(self,"file",None) is not None:
            self.file.close();self.file=None

    def select(self,index):
        require(0<=index<self.m,"sample rank out of range")
        group=index//self.stride
        pos=struct.unpack_from("<I",self.cp,4*group)[0]
        require(pos<self.highbits and (self.high[pos>>3]>>(pos&7))&1,
                "invalid select checkpoint")
        remaining=index%self.stride
        if remaining:
            pos+=1
            while True:
                require(pos<self.highbits,"high bitmap exhausted")
                off=pos>>3
                data=self.high[off:off+8]
                bits=int.from_bytes(data,"little")>>(pos&7)
                span=len(data)*8-(pos&7)
                count=bits.bit_count()
                if count >= remaining:
                    for _ in range(remaining):
                        bit=bits&-bits
                        require(bit!=0,"select bit disappeared")
                        delta=bit.bit_length()-1
                        bits ^= bit
                    pos+=delta
                    break
                remaining-=count
                pos+=span
        row=((pos-index)<<self.low)|getbits(self.lows,index*self.low,self.low)
        require(row<self.sa_size,"sample row overflow")
        return row

    def sampled_sa(self,row):
        require(0<=row<self.sa_size,"row outside corpus")
        lo,hi=0,self.m
        while lo<hi:
            mid=(lo+hi)//2
            if self.select(mid)<row:
                lo=mid+1
            else:
                hi=mid
        if lo>=self.m or self.select(lo)!=row:
            return None
        value=getbits(self.values,lo*self.width,self.width)*self.sample_step
        require(0<=value<self.sa_size,"sample SA overflow")
        return value


def make_runtime(reference_tools,rlb2,rlr2,ef_path):
    import sys
    sys.path.insert(0,str(Path(reference_tools)))
    import rlbwt_query_v2 as q
    class ExperimentalRuntime(q.QueryRuntime):
        def __init__(self):
            self.rank=q.RankV2(rlb2,rlr2)
            self.locate=LocateEF(ef_path)
            q.require(self.rank.raw_length==self.locate.sa_size,
                      "RLB2/RLR2/LOC-TEXT-EF geometry mismatch")
            self.corpus_bytes=self.rank.raw_length-1
    return ExperimentalRuntime()
