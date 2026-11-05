#!/usr/bin/env python3
"""CORE-L5 independent windows smoke using original binary-safe corpus/SA/RLB2."""
import argparse
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

REF="c33cb493bc75f13ee4a904fabb190a29a196c94b"

def must(x,msg):
    if not x: raise RuntimeError(msg)

def load(path):
    spec=importlib.util.spec_from_file_location("pinned_fixture",path)
    must(spec and spec.loader,"missing original fixture")
    mod=importlib.util.module_from_spec(spec);sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

def run(bin,rlb2,ef,target,threads,mode):
    return subprocess.run([str(bin),str(rlb2),str(ef),str(target),
                           str(threads),mode],
        stdin=subprocess.DEVNULL,capture_output=True,text=True,timeout=120)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    ref=a.reference.resolve()
    must(subprocess.check_output(["git","-C",str(ref),"rev-parse","HEAD"],
        text=True).strip()==REF,"incorrect frozen checkout")
    root=a.output.resolve()
    must(not root.exists(),"refuse existing output")
    root.mkdir(parents=True)
    tools=ref/"tools"
    sys.path.insert(0,str(Path(__file__).parent))
    from real_enwik8_ef_physical_gate import build,verify
    fixture=load(tools/"check_rlbwt_query_v2.py").build_fixture(
        root,tools/"rlbwt_container_v2.py",tools/"rlbwt_rank_v2.py")
    src=fixture["source"]
    sa=fixture["suffix_array"]
    safile=root/"construction.sa"
    safile.write_bytes(b"\0"*60+struct.pack("<"+"I"*len(sa),*sa))
    rows=[]
    count=0
    for step in (2,3,5):
        ef=root/f"native-ef-{step}"
        geo=build(safile,len(sa),step,ef)
        checked=verify(safile,ef,geo)
        must(checked["sampled_sa_records_verified"]==(len(sa)-1)//step+1,
             "wrong EF sample count")
        for mode in ("rank","direct"):
            for threads in (1,2,4):
                target=root/f"result-{step}-{mode}-{threads}.raw"
                r=run(a.binary,fixture["rlb2"],ef,target,threads,mode)
                must(r.returncode==0,f"native failed {step}/{mode}/{threads}: {r.stderr}")
                metrics=json.loads(r.stdout)
                must(target.read_bytes()==src,"exact native bytes differ")
                must(metrics["step"]==step and metrics["threads"]==threads
                     and metrics["symbol_mode"]==mode and metrics["source_bytes"]==len(src),
                     "wrong claimed geometry")
                count+=1
                rows.append({"step":step,"mode":mode,"threads":threads,
                             "source_sha256":hashlib.sha256(target.read_bytes()).hexdigest(),
                             "metric":metrics})
    safile.unlink()
    # No read from the original source, canonical SA or BWT is possible in CLI.
    rlb2=fixture["rlb2"]
    ef=root/"native-ef-5"
    corrupted=root/"corrupted.rlb2"
    b=bytearray(rlb2.read_bytes())
    b[-1]^=1
    corrupted.write_bytes(b)
    result=run(a.binary,corrupted,ef,root/"hostile-rlb2.raw",2,"rank")
    must(result.returncode!=0 and not (root/"hostile-rlb2.raw").exists(),
         "corrupted RLB2 accepted")
    mutated=root/"mutated-ef"
    b=bytearray(ef.read_bytes())
    b[0]^=1
    mutated.write_bytes(b)
    result=run(a.binary,rlb2,mutated,root/"hostile-ef.raw",2,"rank")
    must(result.returncode!=0 and not (root/"hostile-ef.raw").exists(),
         "corrupted EF header accepted")
    truncated=root/"truncated-ef"
    truncated.write_bytes(ef.read_bytes()[:-1])
    result=run(a.binary,rlb2,truncated,root/"hostile-truncated-ef.raw",2,"rank")
    must(result.returncode!=0 and not (root/"hostile-truncated-ef.raw").exists(),
         "truncated EF accepted")
    record={
        "status":"SMALL_PARALLEL_BWT_WINDOWS_ALL_EXACT_PASS",
        "original_source_sha256":hashlib.sha256(src).hexdigest(),
        "original_frozen_commit":REF,
        "source_bytes":len(src),
        "tested_cases":count,
        "window_sizes":[2,3,5],
        "threads":[1,2,4],
        "symbol_modes":["rank","direct"],
        "full_byte_exact_cases":rows,
        "hostile_cases_rejected":["RLB2 bitflip","EF magic","EF truncated"],
        "limitations":["Small frozen 24-byte binary fixture, not enwik8",
                       "Uses O(n) transient LF table, and direct mode optionally an O(n) BWT symbol cache",
                       "No claim of speed improvement on microbenchmarks",
                       "No independent P/R/O signed proof or version currentness"]
    }
    (root/"result.json").write_text(json.dumps(record,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:record[k] for k in
        ("status","tested_cases","source_bytes","hostile_cases_rejected")},sort_keys=True))
if __name__=="__main__":
    main()
