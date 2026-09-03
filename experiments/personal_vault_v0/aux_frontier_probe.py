#!/usr/bin/env python3
import argparse,json,math,mmap,struct
from collections import Counter
from pathlib import Path
BWT_HEADER=struct.Struct("<8sIQQIIIQQ")
def bits_for(v): return max(1,int(v).bit_length())
def main():
 p=argparse.ArgumentParser(); p.add_argument("corpus",type=Path); p.add_argument("bwt",type=Path); p.add_argument("out",type=Path); a=p.parse_args()
 n=a.corpus.stat().st_size; rows=n+1
 steps=[512,1024,2048,4096,8192,16384,32768,65536]
 with a.bwt.open("rb") as f:
  mm=mmap.mmap(f.fileno(),0,access=mmap.ACCESS_READ); vals=memoryview(mm)[BWT_HEADER.size:].cast("H")
  assert len(vals)==rows
  # Active-symbol delta checkpoint model: for each interval store only non-zero per-symbol deltas.
  models=[]
  for step in steps:
   pairs=0; checkpoints=0; maxdelta=0
   for start in range(0,rows,step):
    c=Counter(int(x) for x in vals[start:min(start+step,rows)])
    pairs+=len(c); checkpoints+=1
    if c: maxdelta=max(maxdelta,max(c.values()))
   # Conservative byte models, not implementations:
   # symbol u16 + delta u16/u32 selected by step; plus u64 cumulative run position/checkpoint.
   dw=2 if step<=65535 else 4
   sparse_bytes=160+checkpoints*8+pairs*(2+dw)
   # dense bitpacked delta vector: 257 counters, each ceil(log2(step+1)) bits.
   b=bits_for(step)
   dense_bp=160+checkpoints*8+math.ceil(checkpoints*257*b/8)
   models.append({"step":step,"checkpoints":checkpoints,"active_symbol_pairs":pairs,"avg_active_symbols":pairs/checkpoints,
                  "sparse_delta_bytes":sparse_bytes,"dense_bitpacked_delta_bytes":dense_bp,"delta_bits":b})
  vals.release(); mm.close()
 loc_samples=(rows+127)//128; loc_u32=24+loc_samples*4
 report={"format":"GLYPH_PERSONAL_VAULT_AUX_FRONTIER_PROBE_V0","status":"DERIVED_SIZE_MODELS_NOT_IMPLEMENTATION",
 "source_bytes":n,"loc_implicit_u32_bytes":loc_u32,"rank_models":models,
 "combined_with_current_rlb2_6058925":[{"step":m["step"],"sparse_plus_loc_ratio":(6058925+m["sparse_delta_bytes"]+loc_u32)/n,
 "dense_bitpacked_plus_loc_ratio":(6058925+m["dense_bitpacked_delta_bytes"]+loc_u32)/n} for m in models],
 "non_claims":["Models do not establish rank latency.","Sparse delta model still requires a navigation design to recover cumulative ranks.","No runtime format changed."]}
 a.out.write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n"); print(json.dumps(report,sort_keys=True))
if __name__=="__main__": main()
