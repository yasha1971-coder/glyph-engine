#!/usr/bin/env python3
import argparse,json,sys,time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"tools"))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import rlbwt_query_v2 as q
import query_rlb3x_count as r3
import query_loc2_experimental as l2

class Runtime3X:
    def __init__(self,rlb3x,loc2):
        self.rank=r3.Rank3X(rlb3x)
        self.locate=l2.LocateCore2(loc2)
        q.require(self.locate.sa_size==self.rank.raw_length,"LOC2/RLB3X rows")
        self.corpus_bytes=self.rank.raw_length-1
    def close(self):
        self.locate.close(); self.rank.close()
    def locate_row(self,row):
        q.require(0<=row<self.rank.raw_length,"FM row outside range")
        current=row; steps=0
        while True:
            sampled=self.locate.sampled_sa(current)
            if sampled is not None:
                suffix=(sampled+steps)%self.locate.sa_size
                return suffix,steps
            current=self.rank.lf(current); steps+=1
            q.require(steps<=self.locate.sa_size,"locate LF walk exceeded SA size")
    def query(self,pattern,max_offsets=-1):
        started=time.perf_counter_ns()
        left,right=self.rank.backward_search(pattern); count=right-left
        locate_count=count if max_offsets<0 else min(count,max_offsets)
        offsets=[]; total=0; mx=0
        for row in range(left,left+locate_count):
            off,steps=self.locate_row(row)
            q.require(off<self.corpus_bytes,"ordinary pattern resolved to terminal suffix")
            offsets.append(off); total+=steps; mx=max(mx,steps)
        offsets.sort()
        return {
          "format":"GLYPH_RLB3X_LOC2_QUERY_EXPERIMENT_V0",
          "count":count,"fm_interval":[left,right],"locate_offsets":offsets,
          "locate_offsets_complete":locate_count==count,"located_count":locate_count,
          "maximum_lf_steps":mx,"total_lf_steps":total,
          "rank_calls":self.rank.rank_calls,"lf_calls":self.rank.lf_calls,
          "decoded_blocks":self.rank.decoded_blocks,"decoded_runs":self.rank.decoded_runs,
          "scanned_symbols":self.rank.scanned_symbols,
          "query_elapsed_ns":time.perf_counter_ns()-started,
          "rlb2_not_used":True
        }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--rlb3x",type=Path,required=True);p.add_argument("--locate-core",type=Path,required=True)
    p.add_argument("--pattern-hex",required=True);p.add_argument("--max-offsets",type=int,default=-1);a=p.parse_args()
    rt=Runtime3X(a.rlb3x,a.locate_core)
    try:r=rt.query(bytes.fromhex(a.pattern_hex),a.max_offsets)
    finally:rt.close()
    print(json.dumps(r,sort_keys=True,separators=(",",":")))
if __name__=="__main__":main()
