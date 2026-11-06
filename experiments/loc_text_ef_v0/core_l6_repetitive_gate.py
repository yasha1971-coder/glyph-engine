#!/usr/bin/env python3
"""CORE-L6 deterministic highly repetitive 8MiB corpus: dense vs run-LF.

This fixture deliberately tests an r << n structural regime, unlike enwik8.
The corpus is construction-only and removed before decoding. It is not
a substitute for an independently collected real repetitive corpus.
No result is a production or external Move-r/Orbit performance claim.
"""
import argparse
import hashlib
import json
import statistics
import subprocess
import sys
from pathlib import Path

from native_inverse_enwik8_gate import REF,check,sha,run
from real_enwik8_ef_physical_gate import build,verify

PASS="CORE_L6_REPETITIVE_SINGLE_STATE_COMPARE_PASS_RESEARCH"

def generate(path):
    base=bytes(((i*73+(i>>3)*19+(i>>7)*11) & 255) for i in range(65536))
    h=hashlib.sha256()
    with path.open("xb") as out:
        for generation in range(128):
            block=bytearray(base)
            for k in range(32):
                at=(generation*197+ k*2017) % len(block)
                block[at]=((generation*13 + k*59) ^ (k>>2)) & 255
            data=bytes(block)
            out.write(data)
            h.update(data)
    return {"source_sha256":h.hexdigest(),
            "source_bytes":path.stat().st_size,
            "generations":128,
            "base_snapshot_bytes":65536,
            "changed_positions_per_generation":32,
            "includes_nul_and_ff":True}

def case(binary,rlb2,ef,out,mode,threads,expected_bytes,expected_sha,logs):
    command=[binary,rlb2,out] if mode=="sequential" else [
        binary,rlb2,ef,out,str(threads),mode.replace("run-","").replace("dense-","").replace("stream-","")]
    p,meta=run(command,logs,timeout=240)
    check(out.stat().st_size==expected_bytes and sha(out)==expected_sha,
          f"source restore mismatch {mode} {threads}")
    metric=json.loads(p.stdout)
    out.unlink()
    return {"mode":mode,"threads":threads,
            "wall_seconds":meta["wall_seconds"],
            "peak_rss_kb":meta["peak_rss_kb"],
            "native_metrics":metric,
            "sha256_verified":True}

def main():
    arg=argparse.ArgumentParser()
    arg.add_argument("--reference",type=Path,required=True)
    arg.add_argument("--sequential",type=Path,required=True)
    arg.add_argument("--dense",type=Path,required=True)
    arg.add_argument("--run-index",type=Path,required=True)
    arg.add_argument("--stream-index",type=Path,default=None)
    arg.add_argument("--output",type=Path,required=True)
    arg.add_argument("--repeats",type=int,default=1)
    a=arg.parse_args()
    check(1<=a.repeats<=5,"repeat cap 1..5")
    ref=a.reference.resolve()
    target=a.output.resolve()
    check(subprocess.check_output(["git","-C",str(ref),"rev-parse","HEAD"],
          text=True).strip()==REF,"wrong frozen reference")
    check(not target.exists(),"refuse preexisting output")
    target.mkdir(parents=True)
    work=target/"work"
    work.mkdir()
    logs=[]
    result={"schema":"GLYPH_CORE_L6_REPETITIVE_RLB2_RUN_INTERVAL_V0",
            "status":"BLOCKED","historical_commit":REF}
    try:
        raw=work/"generated-source.bin"
        gen=generate(raw)
        check(gen["source_bytes"]==8388608,"generation size mismatch")
        sa=work/"construction.sa"
        bwt=work/"construction.bwt"
        rlb2=work/"retained.rlb2"
        rlr2=work/"retained.rlr2"
        ef=work/"retained.ef"
        run([ref/"build/build_sa_binary_v1",raw,sa],logs,timeout=300)
        run([ref/"build/build_bwt_binary_v1",raw,sa,bwt],logs,timeout=300)
        run([sys.executable,ref/"tools/rlbwt_container_v2.py",
             "encode",bwt,rlb2],logs,timeout=600)
        run([sys.executable,ref/"tools/rlbwt_rank_v2.py",
             "build",rlb2,rlr2,"--rank-step","8192"],logs,timeout=600)
        geometry=build(sa,gen["source_bytes"]+1,128,ef)
        checked=verify(sa,ef,geometry)
        check(checked["sampled_sa_records_verified"]==(gen["source_bytes"]//128)+1,
              "EF sample verification incomplete")
        # Geometry from original canonical RLB2 header v2, little-endian.
        import struct
        with rlb2.open("rb") as fd: header=fd.read(160)
        rows=struct.unpack_from("<Q",header,24)[0]
        runs=struct.unpack_from("<Q",header,32)[0]
        check(rows==gen["source_bytes"]+1 and 0<runs<=rows,
              "RLB2 invalid rows/runs")
        for x in (raw,sa,bwt):x.unlink()
        check(all(not x.exists() for x in (raw,sa,bwt)),
              "construction-only data remained during benchmark")
        cases=[
            ("sequential",1,a.sequential),
            ("dense-rank",1,a.dense),
            ("dense-rank",4,a.dense),
            ("run-rank",1,a.run_index),
            ("run-rank",2,a.run_index),
            ("run-rank",4,a.run_index),
            ("run-direct",4,a.run_index),
        ]
        if a.stream_index:
            cases.extend([
                ("stream-rank",1,a.stream_index),
                ("stream-rank",4,a.stream_index),
                ("stream-direct",4,a.stream_index),
            ])
        measured=[]
        for rep in range(a.repeats):
            # Alternate case order to reveal warm-cache/frequency sensitivity.
            ordered=cases if rep%2==0 else list(reversed(cases))
            for i,(mode,threads,binary) in enumerate(ordered):
                row=case(binary,rlb2,ef,work/f"decoded-{rep}-{i}.raw",
                         mode,threads,gen["source_bytes"],gen["source_sha256"],logs)
                row["repeat"]=rep+1
                measured.append(row)
                print(json.dumps({"case":len(measured),
                                  "mode":mode,"threads":threads,
                                  "repeat":rep+1,
                                  "wall_seconds":row["wall_seconds"],
                                  "peak_rss_kb":row["peak_rss_kb"],
                                  "sha256_verified":True}),flush=True)
        medians=[]
        for mode,threads,_ in cases:
            matches=[x for x in measured
                     if x["mode"]==mode and x["threads"]==threads]
            check(len(matches)==a.repeats,"missing repeated profile")
            medians.append({
                "mode":mode,"threads":threads,
                "repetitions":len(matches),
                "wall_median_seconds":statistics.median(x["wall_seconds"] for x in matches),
                "peak_rss_max_kb":max(x["peak_rss_kb"] for x in matches),
                "wall_min_seconds":min(x["wall_seconds"] for x in matches),
                "wall_max_seconds":max(x["wall_seconds"] for x in matches)})
        result.update({
            "repeats":a.repeats,"per_profile_summary":medians,
            "status":PASS,
            "generated_source":gen,
            "bwt_rows":rows,"bwt_runs":runs,"r_over_n":runs/rows,
            "retained_rlb2_bytes":rlb2.stat().st_size,
            "retained_rlr2_bytes":rlr2.stat().st_size,
            "retained_ef_bytes":ef.stat().st_size,
            "persistent_subtotal_bytes":sum(x.stat().st_size for x in (rlb2,rlr2,ef)),
            "source_sa_bwt_removed_before_decoder":True,
            "cases":measured,
            "limitations":[
                "Synthetic deterministic 128 lightly edited snapshots: not personal customer files.",
                "All physical source-restored SHA checked but no external timestamp/authority.",
                "No P/R/O signed evidence, portable proof, multi-document provenance or 1TB proof.",
                "Short repeated profiles, no cold-process p95/p99 or rigorous variance controls.",
                "Run-LF index is temporary and employs bucketed binary predecessor, not Orbit/Move-r.",
                "Benchmark excludes build-only SA/BWT source and code/proof deliverable overhead from ratio."
            ]
        })
    except Exception as e:
        result.update(status="BLOCKED",error=repr(e))
    finally:
        (target/"result.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
        (target/"commands.json").write_text(json.dumps(logs,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:result.get(k) for k in
        ("status","error","bwt_rows","bwt_runs","r_over_n","persistent_subtotal_bytes")},sort_keys=True))
    return 0 if result["status"]==PASS else 2

if __name__=="__main__":
    sys.exit(main())
