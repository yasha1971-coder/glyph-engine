#!/usr/bin/env python3
"""Integrated 100MB enwik8 GLYPH research gate: search + byte ranges + restore.

Rebuilds frozen RLB2/RLR2, builds physically pinned EF locate, destroys the
original raw source/SA/BWT temp, recovers all bytes with independent native
inverse from RLB2 only, and checks range extraction and source-byte queries
through a different RankV2+EF path. Not an O3/O4 signed evidence release.
"""
import argparse
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

from native_inverse_enwik8_gate import (
    REF,SHA,BYTES,check,sha,run,download_source,
)
from real_enwik8_ef_physical_gate import build, verify

EF_SHA="c1cd29d7bb40805d23d4aa891cd08808a1d7e1e3734d6a13799163dc23d469c7"
PASS="INTEGRATED_ENWIK8_SELF_INDEX_SEARCH_RANGE_RESTORE_PASS_RESEARCH"

CHALLENGES=[
    (0,1),(0,24),(127,32),(128,24),(4095,55),
    (1000000,24),(13579113,24),(27890000,24),
    (55432100,24),(83567543,24),(99999999,1),(99999951,49),
    (98765432,48),(76543210,31),(12345678,36),(3456789,29),
]
QUERIES=[1200,1000000,13579113,27890000,55432100,83567543]

def direct_hits(data,pat):
    offsets=[]
    pos=0
    while True:
        at=data.find(pat,pos)
        if at<0:return offsets
        offsets.append(at)
        pos=at+1

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--native",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    ref=a.reference.resolve()
    native=a.native.resolve()
    output=a.output.resolve()
    check(subprocess.check_output(["git","-C",str(ref),"rev-parse","HEAD"],
          text=True).strip()==REF,"wrong frozen GLYPH checkout")
    check(native.is_file() and not output.exists(),"invalid compiled decoder or output")
    output.mkdir(parents=True)
    work=output/"work"
    work.mkdir()
    logs=[]
    evidence={"schema":"GLYPH_ENWIK8_SINGLE_RETENTION_SELF_INDEX_E2E_V0",
              "status":"BLOCKED","frozen_reference_commit":REF,
              "source_sha256":SHA["corpus"],"candidate_ef_sha256":EF_SHA}
    try:
        source=download_source(work,logs)
        sa=work/"source-only.sa"
        bwt=work/"source-only.bwt"
        rlb2=work/"kept.rlb2"
        rlr2=work/"kept.rlr2"
        ef=work/"kept.loc-text-ef"
        run([ref/"build/build_sa_binary_v1",source,sa],logs,timeout=900)
        run([ref/"build/build_bwt_binary_v1",source,sa,bwt],logs,timeout=900)
        for key,path in (("sa",sa),("bwt",bwt)):
            check(path.stat().st_size==BYTES[key] and sha(path)==SHA[key],
                  f"frozen build identity changed: {key}")
        run([sys.executable,ref/"tools/rlbwt_container_v2.py","encode",
             bwt,rlb2],logs,timeout=1800)
        run([sys.executable,ref/"tools/rlbwt_rank_v2.py","build",
             rlb2,rlr2,"--rank-step","8192"],logs,timeout=1800)
        for key,path in (("rlb2",rlb2),("rlr2",rlr2)):
            check(path.stat().st_size==BYTES[key] and sha(path)==SHA[key],
                  f"frozen compressed state identity changed: {key}")
        geom=build(sa,100000001,128,ef)
        sample_replay=verify(sa,ef,geom)
        check(ef.stat().st_size==2844324 and sha(ef)==EF_SHA,
              "physical EF identity mismatch")
        raw=source.read_bytes()
        canonical_challenges=[
            {"start":i,"length":L,
             "sha256":hashlib.sha256(raw[i:i+L]).hexdigest()}
            for i,L in CHALLENGES
        ]
        del raw
        for item in (source,sa,bwt):
            item.unlink()
        check(all(not item.exists() for item in (source,sa,bwt)),
              "original source, SA or canonical BWT retained at runtime")
        restored=work/"recovered-from-RLB2-only"
        stdout,metric=run([native,rlb2,restored],logs,timeout=1800)
        native_evidence=json.loads(stdout.stdout)
        check(native_evidence["status"]=="NATIVE_RLB2_SELF_INDEX_RESTORE_RESEARCH_SUCCESS",
              "native inverse was not successful")
        check(restored.stat().st_size==BYTES["corpus"]
              and sha(restored)==SHA["corpus"],
              "fatal: reconstructed complete corpus hash mismatch")
        # The Python search/extractor path below only receives these three
        # retained index artifacts. Independent source comparison reads the
        # separately restored, externally SHA-pinned output.
        sys.path[:0]=[str(Path(__file__).parent),str(ref/"tools")]
        from query_ef_research import make_runtime
        from self_index_exact_range_extractor import EFByteExtractor
        rt=make_runtime(ref/"tools",rlb2,rlr2,ef)
        try:
            start=time.perf_counter()
            extractor=EFByteExtractor(rt)
            setup_sec=time.perf_counter()-start
            recovered=restored.read_bytes()
            check(len(recovered)==BYTES["corpus"],"source oracle missing bytes")
            actual_ranges=[]
            for c in canonical_challenges:
                at,L=c["start"],c["length"]
                t0=time.perf_counter()
                out,steps=extractor.extract(at,L)
                elapsed=time.perf_counter()-t0
                check(out==recovered[at:at+L]
                      and hashlib.sha256(out).hexdigest()==c["sha256"],
                      f"range extraction mismatch at {at} len={L}")
                check(steps<=L+127,"range bound exceeded")
                actual_ranges.append({
                    "start":at,"length":L,
                    "verified_sha256":c["sha256"],
                    "lf_steps":steps,"extract_seconds":round(elapsed,7)})
            query_cases=[]
            for index in QUERIES:
                pattern=recovered[index:index+24]
                oracle=direct_hits(recovered,pattern)
                max_offsets=-1 if len(oracle)<64 else 2
                data=rt.query(pattern,max_offsets)
                check(data["count"]==len(oracle),"full corpus query count mismatch")
                offsets=data["locate_offsets"]
                if max_offsets==-1:
                    check(offsets==oracle,"full query offsets incorrect")
                else:
                    check(len(offsets)==min(2,len(oracle))
                          and set(offsets).issubset(set(oracle)),
                          "bounded query offsets not exact")
                for offset in offsets:
                    excerpt,steps=extractor.extract(offset,len(pattern))
                    check(excerpt==pattern,"search result byte-evidence mismatch")
                    check(steps<=len(pattern)+127,"query byte extract bound broken")
                query_cases.append({
                    "pattern_sha256":hashlib.sha256(pattern).hexdigest(),
                    "count":len(oracle),"returned_offsets":len(offsets),
                    "bytes_verified_per_hit":len(pattern),
                    "query_elapsed_ns":data["query_elapsed_ns"]})
        finally:
            rt.close()
        stored=rlb2.stat().st_size+rlr2.stat().st_size+ef.stat().st_size
        evidence.update({
            "status":PASS,
            "original_hashes_reproduced":SHA,
            "canonical_ef_hash_verified":EF_SHA,
            "source_restored_bytes":restored.stat().st_size,
            "source_recovered_sha256":sha(restored),
            "native_restore_seconds":metric["wall_seconds"],
            "native_restore_peak_rss_kb":metric["peak_rss_kb"],
            "ef_sample_records_verified":sample_replay["sampled_sa_records_verified"],
            "range_cases":len(actual_ranges),"range_results":actual_ranges,
            "query_cases":len(query_cases),"query_results":query_cases,
            "ephemeral_q_to_row_bytes":extractor.sample_rows_ram_bytes,
            "ephemeral_q_to_row_build_seconds":round(setup_sec,6),
            "python_process_peak_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "only_retained_persistent_index_bytes":stored,
            "only_retained_index_components":{"rlb2":rlb2.stat().st_size,
                                              "rlr2":rlr2.stat().st_size,
                                              "ef":ef.stat().st_size},
            "retained_index_ratio_to_source":stored/BYTES["corpus"],
            "construction_only_original_source_sa_bwt_removed_before_queries":True,
            "no_separately_retained_compressed_source_copy":True,
            "limitations":[
                "Actual 100MB identical-binary-source search/range/full restore, but research scope only.",
                "EF source extract currently uses Python RankV2 and may be slow; no native performance win.",
                "Source recovered first through native self-index; oracle is SHA-pinned to the known source.",
                "No independent signed O3/O4 evidence receipt, proof of global absence or freshness witness.",
                "Physical storage subtotal excludes current research verifier/code and new proof manifest.",
                "Not a multi-document canonical provenance/corpus-composition result.",
                "No 1TB scaling or bounded full-recovery RAM: native inverse requires O(n) temporary LF.",
                "Original row-prefix max_offsets semantics are not the O3 canonical-order bounded API.",
            ]
        })
    except Exception as e:
        evidence.update(status="BLOCKED",error=repr(e))
    finally:
        (output/"result.json").write_text(json.dumps(evidence,indent=2,sort_keys=True)+"\n")
        (output/"commands.json").write_text(json.dumps(logs,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:evidence.get(k) for k in
        ("status","error","source_restored_bytes","source_recovered_sha256",
         "range_cases","query_cases","ephemeral_q_to_row_build_seconds",
         "only_retained_persistent_index_bytes","retained_index_ratio_to_source")},
        sort_keys=True))
    return 0 if evidence["status"]==PASS else 2

if __name__=="__main__":
    sys.exit(main())
