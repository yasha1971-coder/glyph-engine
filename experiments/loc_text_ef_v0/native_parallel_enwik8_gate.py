#!/usr/bin/env python3
"""CORE-L5 full enwik8 same-run, same-archive native parallel inverse benchmark.

Rebuilds all original source/SA/BWT/RLB2/RLR2 hashes and the physical EF hash,
deletes raw/SA/BWT, and compares independent native sequential inversion with
parallel 1/2/4 thread implementations using exactly the retained files.
All 100,000,000 reconstructed bytes are SHA256 verified per result.
No P/R/O signed evidence or TB memory claim is implied.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

from native_inverse_enwik8_gate import (
    REF, SHA, BYTES, check, sha, run, download_source,
)
from real_enwik8_ef_physical_gate import build, verify

EF_SHA="c1cd29d7bb40805d23d4aa891cd08808a1d7e1e3734d6a13799163dc23d469c7"
PASS="PARALLEL_100MB_RLB2_EF_BYTE_PERFECT_COMPARISON_PASS_RESEARCH"

def record_case(binary,archive,ef,out,threads,mode,log):
    command=[binary,archive,out] if mode=="sequential" else [
        binary,archive,ef,out,str(threads),mode]
    p,meta=run(command,log,timeout=900)
    check(out.is_file() and out.stat().st_size==BYTES["corpus"],
          "native decode output size mismatch")
    recovered_sha=sha(out)
    check(recovered_sha==SHA["corpus"],
          "CRITICAL: native decode returned noncanonical enwik8")
    data=json.loads(p.stdout)
    check(data["source_bytes"]==BYTES["corpus"],"native metric mismatches source")
    if mode!="sequential":
        check(data["threads"]==threads and data["symbol_mode"]==mode,
              "native parallel metric mismatches requested configuration")
        check(data["windows"]==781250 and data["step"]==128,
              "native parallel sample geometry incorrect")
    row={
        "mode":mode,"threads":threads,"runner_wall_seconds":meta["wall_seconds"],
        "peak_rss_kb":meta["peak_rss_kb"],
        "source_sha256":recovered_sha,"native_metrics":data,
    }
    out.unlink()
    return row

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--reference",type=Path,required=True)
    a.add_argument("--sequential",type=Path,required=True)
    a.add_argument("--parallel",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    v=a.parse_args()
    ref=v.reference.resolve()
    seq=v.sequential.resolve()
    par=v.parallel.resolve()
    target=v.output.resolve()
    check(subprocess.check_output(["git","-C",str(ref),"rev-parse","HEAD"],
          text=True).strip()==REF,"wrong pinned frozen source")
    check(seq.is_file() and par.is_file() and not target.exists(),
          "missing binary or preexisting output")
    target.mkdir(parents=True)
    work=target/"work"
    work.mkdir()
    report={
        "schema":"GLYPH_CORE_L5_PARALLEL_ENWIK8_V0",
        "status":"BLOCKED",
        "frozen_source_commit":REF,
        "source_sha256":SHA["corpus"],
        "parallel_decoder_sha256":sha(par),
        "sequential_decoder_sha256":sha(seq)
    }
    logs=[]
    try:
        raw=download_source(work,logs)
        sa=work/"construction.sa"
        bwt=work/"construction.bwt"
        rlb2=work/"retained.rlb2"
        rlr2=work/"retained.rlr2"
        ef=work/"retained.loc-text-ef"
        run([ref/"build/build_sa_binary_v1",raw,sa],logs,timeout=900)
        run([ref/"build/build_bwt_binary_v1",raw,sa,bwt],logs,timeout=900)
        for label,path in (("sa",sa),("bwt",bwt)):
            check(path.stat().st_size==BYTES[label] and sha(path)==SHA[label],
                  f"historical {label} hash differs")
        run([sys.executable,ref/"tools/rlbwt_container_v2.py",
             "encode",bwt,rlb2],logs,timeout=1800)
        run([sys.executable,ref/"tools/rlbwt_rank_v2.py",
             "build",rlb2,rlr2,"--rank-step","8192"],logs,timeout=1800)
        for label,path in (("rlb2",rlb2),("rlr2",rlr2)):
            check(path.stat().st_size==BYTES[label] and sha(path)==SHA[label],
                  f"historical {label} hash differs")
        geometry=build(sa,BYTES["corpus"]+1,128,ef)
        sample_verify=verify(sa,ef,geometry)
        check(ef.stat().st_size==2844324 and sha(ef)==EF_SHA,
              "EF physically encoded bytes not reproducible")
        check(sample_verify["sampled_sa_records_verified"]==781251,
              "not all EF samples verified against SA")
        # Build-only source and canonical SA/BWT no longer exist during ANY
        # benchmark. All results derive from the committed compressed state.
        for path in (raw,sa,bwt): path.unlink()
        check(all(not x.exists() for x in (raw,sa,bwt)),
              "construction data remained available to runtime")
        rows=[]
        cases=[
            ("sequential",1,seq),
            ("rank",1,par),("rank",2,par),("rank",4,par),
            ("direct",1,par),("direct",2,par),("direct",4,par),
        ]
        for index,(mode,threads,binary) in enumerate(cases):
            row=record_case(binary,rlb2,ef,work/f"restore-{index}.raw",
                            threads,mode,logs)
            rows.append(row)
            print(json.dumps({"case":index+1,"of":len(cases),
                              "mode":mode,"threads":threads,
                              "wall_seconds":row["runner_wall_seconds"],
                              "peak_rss_kb":row["peak_rss_kb"],
                              "source_sha256":row["source_sha256"]}),flush=True)
        report.update({
            "status":PASS,
            "source_bytes":BYTES["corpus"],
            "original_historical_file_hashes_reproduced":SHA,
            "ef_sha256":EF_SHA,
            "ef_sample_records_verified":781251,
            "rlb2_bytes":rlb2.stat().st_size,
            "rlr2_bytes":rlr2.stat().st_size,
            "ef_bytes":ef.stat().st_size,
            "retained_single_index_bytes":sum(x.stat().st_size for x in (rlb2,rlr2,ef)),
            "no_original_raw_sa_bwt_at_decode":True,
            "cases":[{k:v for k,v in x.items()} for x in rows],
            "limits":[
                "Each case executed once; timing is descriptive, not a statistical p95.",
                "Same GitHub runner, binary class, source and corpus, but frequency and contention uncontrolled.",
                "Full 32-bit LF table and in-memory reconstructed output scale linearly with N.",
                "Optional direct-symbol mode allocates another O(N) byte cache; report peak RSS.",
                "Native parallel decode needs EF in addition to RLB2; no duplicate source payload.",
                "No independently trusted O3/O4 answer, proof of global absence, freshness root or full package ratio.",
                "No claim of 1TB performance, single-file packaging or optimized state-of-the-art compression."
            ]
        })
    except Exception as e:
        report.update(status="BLOCKED",error=repr(e))
    finally:
        (target/"result.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
        (target/"commands.json").write_text(json.dumps(logs,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report.get(k) for k in
          ("status","error","source_bytes","retained_single_index_bytes",
           "ef_sample_records_verified")},sort_keys=True),flush=True)
    return 0 if report["status"]==PASS else 2

if __name__=="__main__":
    sys.exit(main())
