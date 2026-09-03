#!/usr/bin/env python3
import argparse, json, math, mmap, struct
from collections import Counter
from pathlib import Path

BWT_HEADER=struct.Struct("<8sIQQIIIQQ")

def uleb(v):
    n=1
    while v>=128: v>>=7; n+=1
    return n

def h0_bytes(counts):
    total=sum(counts.values())
    if not total: return 0.0
    return sum(c*math.log2(total/c) for c in counts.values())/8.0

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("corpus",type=Path); ap.add_argument("bwt",type=Path)
    ap.add_argument("rlb2",type=Path); ap.add_argument("rlr2",type=Path)
    ap.add_argument("loc1",type=Path); ap.add_argument("out",type=Path)
    a=ap.parse_args()
    n=a.corpus.stat().st_size; rows=n+1
    heads=Counter(); lens=Counter(); r=0; prev=None; ln=0
    with a.bwt.open("rb") as f:
        mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ)
        vals=memoryview(mm)[BWT_HEADER.size:].cast("H")
        assert len(vals)==rows
        for x0 in vals:
            x=int(x0)
            if prev is None: prev=x; ln=1
            elif x==prev: ln+=1
            else:
                heads[prev]+=1; lens[ln]+=1; r+=1; prev=x; ln=1
        if prev is not None: heads[prev]+=1; lens[ln]+=1; r+=1
        vals.release(); mm.close()
    current={"rlb2":a.rlb2.stat().st_size,"rlr2":a.rlr2.stat().st_size,"loc1":a.loc1.stat().st_size}
    current["total"]=sum(current.values())
    samples=(rows+127)//128
    loc_u64=24+samples*8
    loc_u32=24+samples*4 if n < 2**32 else None
    head_h=h0_bytes(heads); len_h=h0_bytes(lens)
    entropy_payload=math.ceil(head_h+len_h)
    # This is a Shannon H0 lower-bound model for independent run-head/run-length streams,
    # NOT a realizable random-access format and NOT a claim.
    models={
      "current_ratio":current["total"]/n,
      "loc_implicit_u64_ratio":(current["rlb2"]+current["rlr2"]+loc_u64)/n,
      "loc_implicit_u32_ratio":None if loc_u32 is None else (current["rlb2"]+current["rlr2"]+loc_u32)/n,
      "entropy_runs_plus_current_aux_ratio":(entropy_payload+current["rlr2"]+(loc_u32 or loc_u64))/n,
      "entropy_runs_only_lower_bound_ratio":entropy_payload/n,
    }
    targets={str(x):{"target_bytes":math.floor(n*x),"gap_from_current":current["total"]-math.floor(n*x),
                     "max_aux_if_rlb2_unchanged":math.floor(n*x)-current["rlb2"]}
             for x in (1.0,0.8,0.7,0.6)}
    report={"format":"GLYPH_PERSONAL_VAULT_SPACE_FRONTIER_PROBE_V0",
      "status":"MEASURED_PLUS_DERIVED_MODELS_NOT_IMPLEMENTATION",
      "source_bytes":n,"rows":rows,"runs":r,"r_over_n":r/n,"n_over_r":n/r,
      "current":current,"locate_models":{"samples":samples,"implicit_u64_bytes":loc_u64,"implicit_u32_bytes":loc_u32},
      "run_entropy_model":{"head_h0_bytes":head_h,"length_h0_bytes":len_h,"joint_separate_h0_bytes_ceil":entropy_payload,
        "warning":"Shannon H0 model ignores random-access metadata, coding overhead, dependencies, and canonical block framing."},
      "models":models,"targets":targets,
      "non_claims":["No runtime format changed.","No query path changed.","Entropy model is not an implemented codec.","No latency preservation is claimed."]}
    a.out.write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
