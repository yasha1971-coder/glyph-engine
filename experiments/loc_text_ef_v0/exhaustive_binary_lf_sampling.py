#!/usr/bin/env python3
"""Exhaustive binary-safe test of text-sampled bounded LF and Elias–Fano.

Experimental small-model only: no production LOC2/P-R-O claim.
"""
import itertools
import json
import math
import argparse


def require(value, reason):
    if not value:
        raise AssertionError(reason)


def pack_bits(values, bits):
    buf = bytearray((len(values)*bits+7)//8)
    for index, value in enumerate(values):
        require(0 <= value < (1 << bits), "invalid packed value")
        shift = index*bits
        off = shift >> 3
        count = (bits+(shift&7)+7)//8
        if count:
            existing = int.from_bytes(buf[off:off+count], "little")
            buf[off:off+count] = (existing | (value << (shift&7))).to_bytes(count,"little")
    return bytes(buf)


def unpack(data, pos, bits):
    if bits==0:
        return 0
    off=pos>>3
    length=(bits+(pos&7)+7)//8
    return (int.from_bytes(data[off:off+length],"little")>>(pos&7)) & ((1<<bits)-1)


class SampleEF:
    def __init__(self, sa, step):
        self.n=len(sa)
        self.step=step
        require(step>0 and sorted(sa)==list(range(self.n)), "invalid SA or step")
        samples=[(row,x//step) for row,x in enumerate(sa) if x%step==0]
        self.m=len(samples)
        require(self.m==(self.n-1)//step+1, "invalid sample count")
        self.low=(self.n//self.m).bit_length()-1
        self.width=(self.m-1).bit_length()
        self.lows=pack_bits([row & ((1<<self.low)-1) for row,q in samples],self.low)
        self.quotients=pack_bits([q for row,q in samples],self.width)
        high_bits=((self.n-1)>>self.low)+self.m+1
        high=bytearray((high_bits+7)//8)
        self.select_positions=[]
        for index,(row,q) in enumerate(samples):
            pos=(row>>self.low)+index
            high[pos>>3]|=1<<(pos&7)
            self.select_positions.append(pos)
        self.high=bytes(high)
        # The in-memory list is NOT included in the experimental size estimate;
        # production must implement a separately accounted succinct select index.
        require(sum(byte.bit_count() for byte in self.high)==self.m,"invalid EF high bits")

    def select(self,i):
        hi=self.select_positions[i]-i
        return (hi<<self.low)|unpack(self.lows,i*self.low,self.low)

    def lookup(self,row):
        left,right=0,self.m
        while left<right:
            middle=(left+right)//2
            if self.select(middle)<row:
                left=middle+1
            else:
                right=middle
        if left==self.m or self.select(left)!=row:
            return None
        return unpack(self.quotients,left*self.width,self.width)*self.step


def check(raw):
    # Match pinned GLYPH binary-safe construction: source bytes shifted by +1;
    # unique terminal sentinel=0, lexicographically smallest, at corpus end.
    chars=[x+1 for x in raw]+[0]
    N=len(chars)
    sa=sorted(range(N),key=lambda i:tuple(chars[i:]))
    require(sa[0]==N-1,"terminal suffix not first")
    inverse=[0]*N
    for row,pos in enumerate(sa):
        inverse[pos]=row
    bwt=[chars[(pos-1)%N] for pos in sa]
    lf=[inverse[(pos-1)%N] for pos in sa]
    # Independent LF oracle from BWT C+rank, not only from SA inverse.
    for row,symbol in enumerate(bwt):
        C=sum(c<symbol for c in bwt)
        rank=sum(c==symbol for c in bwt[:row])
        require(lf[row]==C+rank,"BWT rank LF disagrees with inverse SA")
    old_bound_failures=0
    examples=[]
    profile_count=0
    for step in range(1,min(len(raw)+2,9)):
        codec=SampleEF(sa,step)
        row_samples=set(range(0,N,step))
        profile_count+=1
        new_max=old_max=0
        for row,value in enumerate(sa):
            state=row
            hops=0
            while codec.lookup(state) is None:
                state=lf[state]
                hops+=1
                require(hops<N, "LF cycle did not hit text position sample")
            actual=(codec.lookup(state)+hops)%N
            require(actual==value, "wrong recovered SA offset")
            require(hops<=step-1, "text-sampled LF upper bound violated")
            new_max=max(new_max,hops)
            old_state=row
            old_hops=0
            while old_state not in row_samples:
                old_state=lf[old_state]
                old_hops+=1
                require(old_hops<N, "historical row-sampled LF cycle")
            old_max=max(old_max,old_hops)
        if N>step:
            require(new_max==step-1, "non-tight upper-bound test")
        if old_max>step-1:
            old_bound_failures+=1
            if len(examples)<2:
                examples.append({"source_hex":raw.hex(),"step":step,
                                 "old_max":old_max,"new_max":new_max})
    return profile_count,old_bound_failures,examples


def exact_size_model(N=100000001,step=128):
    m=(N-1)//step+1
    low=(N//m).bit_length()-1
    width=(m-1).bit_length()
    highbits=((N-1)>>low)+m+1
    components={
        "ef_low_bytes":(low*m+7)//8,
        "ef_high_bytes":(highbits+7)//8,
        "sa_over_step_quotient_bytes":(m*width+7)//8,
        "select_checkpoint_bytes":((m+255)//256)*4,
        "research_header_bytes":80
    }
    estimate=sum(components.values())
    return {"N":N,"step":step,"sample_count":m,
            "low_bits_per_value":low,"quotient_bits_per_value":width,
            "total_estimated_physical_bytes":estimate,
            "historical_loc2_bytes":3125036,
            "expected_saving_before_full_runtime_and_proof":3125036-estimate,
            "components":components,
            "status":"DERIVED_NOT_MEASURED",
            "disclaimer":"Select checkpoint included, but no source hash, manifest, runtime code or evidence."}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--max-len",type=int,default=8)
    parser.add_argument("--out",type=str)
    a=parser.parse_args()
    require(0<=a.max_len<=9, "exhaustive CI length cap")
    corpora=profiles=failures=0
    counterexamples=[]
    for length in range(a.max_len+1):
        for data in itertools.product((0,1,255),repeat=length):
            p,b,e=check(bytes(data))
            corpora+=1
            profiles+=p
            failures+=b
            if len(counterexamples)<5:
                counterexamples.extend(e[:5-len(counterexamples)])
    require(failures>0 and profiles>0, "no historical sampling counterexample")
    result={
        "status":"EXHAUSTIVE_BINARY_TEXT_SAMPLED_EF_PARITY_PASS",
        "sample_corpora":corpora,
        "corpus_step_profiles":profiles,
        "historical_row_sampled_bound_failures":failures,
        "counterexamples":counterexamples,
        "enwik8_physical_prediction":exact_size_model(),
        "source":"sentinel-first lexicographic binary byte shifted by +1",
        "limitations":["No actual enwik8 LOC3 file or speed measured by this gate",
                       "No cryptographic evidence bundle; no authenticated source restore",
                       "Select_positions reference list is not retained: estimated physical format budgets checkpoints"]
    }
    if a.out:
        with open(a.out,"x",encoding="utf8") as file:
            json.dump(result,file,sort_keys=True,indent=2)
            file.write("\n")
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
