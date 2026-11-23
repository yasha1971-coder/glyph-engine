#!/usr/bin/env python3
"""P-PERF-01: strict candidate-vs-unchanged-L9 golden safety and 100MB benchmark.

Reproducer: caller pins the golden L9-S source file from the same commit.
Candidate and golden operate on identical source bytes and exact patterns.
A fail-open candidate aborts; speed predictions may FAIL without CI laundering.
"""
import argparse
import hashlib
import importlib.util
import itertools
import json
from pathlib import Path
import resource
import shutil
import statistics
import sys
import time
import urllib.request
import zipfile

SOURCE_SHA="2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"
SOURCE_BYTES=100_000_000
SOURCE_ZIP_BYTES=36_445_475
URL="https://mattmahoney.net/dc/enwik8.zip"

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    if not spec or not spec.loader:
        raise RuntimeError("missing reference "+str(path))
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def check(ok,message):
    if not ok:
        raise AssertionError(message)

def digest_file(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):
            h.update(block)
    return h.hexdigest()

def oracle_offsets(data,pattern):
    result=[]
    pos=0
    while True:
        at=data.find(pattern,pos)
        if at<0:break
        result.append(at)
        pos=at+1
    return result

def compare(path,pattern,golden,candidate,chunk,expected=None):
    record={"bytes":path.stat().st_size,"sha256":digest_file(path)}
    w=golden.Work()
    old=golden.scan_authenticated(path,record,pattern,chunk,w)
    new=candidate.scan_authenticated_once(path,record,pattern,chunk)
    for key in ("count","offsets","excerpts","scan_bytes","eof","truncated"):
        check(old[key]==new[key],key+" differs for pattern "+repr(pattern))
    check(w.source_read_bytes==2*record["bytes"] and
          new["read_bytes"]==record["bytes"],"misstated source I/O")
    check(new["source_hash_passes"]==1 and new["sha256"]==record["sha256"],
          "source identity not proven")
    if expected is not None:
        check(new["count"]==len(expected) and new["offsets"]==expected[:256],
              "failed independent byte oracle")
    return new

def reject(label,callback,negative):
    try:
        callback()
    except (OSError,ValueError,TypeError,OverflowError):
        negative.append(label)
        return
    raise AssertionError("FAIL-OPEN: "+label)

def safety(root,golden,candidate):
    root.mkdir(parents=True)
    count=0
    source=root/"test"
    for data in (b"abc",b"aaaaaa",b"ababababa",b"\0\xffbanana\0",
                 b"fox",b"ABCabc",b"xy"):
        source.write_bytes(data)
        patterns={b"z",b"\0",b"\xff",b"a",b"aa",b"aba",b"abc",b"bc",b"abab",b"babab"}
        patterns.update(data[i:j] for i in range(len(data)) for j in range(i+1,len(data)+1))
        for p in sorted(patterns):
            compare(source,p,golden,candidate,1<<20,oracle_offsets(data,p))
            count+=1
    patterns=[bytes(t) for n in range(1,5) for t in itertools.product((0,255),repeat=n)]
    for n in range(1,9):
        for t in itertools.product((0,255),repeat=n):
            data=bytes(t)
            source.write_bytes(data)
            for p in patterns:
                compare(source,p,golden,candidate,1<<20,oracle_offsets(data,p))
                count+=1
    chunk=1<<20
    for p in (b"aba",b"aaaaa",b"\0\xff\0",b"xyxy",bytes(range(17))):
        data=bytearray(b"_"*(2*chunk+87))
        for pos in (0,chunk-len(p)+1,chunk-1,chunk,chunk+1,len(data)-len(p)):
            data[pos:pos+len(p)]=p
        source.write_bytes(data)
        compare(source,p,golden,candidate,chunk,oracle_offsets(data,p))
        count+=1
    source.write_bytes(b"a"*1024)
    record={"bytes":1024,"sha256":digest_file(source)}
    out=candidate.scan_authenticated_once(source,record,b"a")
    check(out["count"]==1024 and len(out["offsets"])==256 and out["truncated"] and
          out["status"]=="UNTRUSTED" and out["reason"]=="LIMITED_COMPLETENESS",
          "257+ hits claimed complete coverage")
    count+=1
    negatives=[]
    reject("wrong-source-sha",lambda:candidate.scan_authenticated_once(
        source,{"bytes":1024,"sha256":"0"*64},b"a"),negatives)
    reject("truncated-source-size",lambda:candidate.scan_authenticated_once(
        source,{"bytes":1025,"sha256":record["sha256"]},b"a"),negatives)
    reject("source-size-too-small",lambda:candidate.scan_authenticated_once(
        source,{"bytes":1023,"sha256":record["sha256"]},b"a"),negatives)
    reject("empty-pattern",lambda:candidate.scan_authenticated_once(
        source,record,b""),negatives)
    reject("oversize-pattern",lambda:candidate.scan_authenticated_once(
        source,record,b"a"*4098),negatives)
    reject("unicode-not-bytes",lambda:candidate.scan_authenticated_once(
        source,record,"a"),negatives)
    reject("subminimum-chunk",lambda:candidate.scan_authenticated_once(
        source,record,b"a",3),negatives)
    reject("oversize-chunk",lambda:candidate.scan_authenticated_once(
        source,record,b"a",5<<20),negatives)
    reject("boolean-source-size",lambda:candidate.scan_authenticated_once(
        source,{"bytes":True,"sha256":record["sha256"]},b"a"),negatives)
    reject("unknown-record-field",lambda:candidate.scan_authenticated_once(
        source,{**record,"isTrusted":True},b"a"),negatives)
    link=root/"symlink"
    link.symlink_to(source)
    reject("symlink-source",lambda:candidate.scan_authenticated_once(
        link,record,b"a"),negatives)
    source.unlink()
    reject("missing-source",lambda:candidate.scan_authenticated_once(
        source,record,b"a"),negatives)
    source.write_bytes(b"!"*1024)
    reject("changed-source-body",lambda:candidate.scan_authenticated_once(
        source,record,b"a"),negatives)
    return {"status":"P01_SAFETY_PASS","exact_oracle_cases":count,
            "hostile_count":len(negatives),"hostile_cases_rejected":negatives,
            "scope":"candidate versus unchanged CORE-L9-S golden and independent byte oracle"}

def enwik8(root,golden,candidate):
    root.mkdir(parents=True)
    source=root/"enwik8"
    archive=root/"enwik8.zip"
    with urllib.request.urlopen(URL,timeout=180) as online,archive.open("wb") as dest:
        shutil.copyfileobj(online,dest,length=1<<20)
    check(archive.stat().st_size==SOURCE_ZIP_BYTES,"official ZIP size drift")
    with zipfile.ZipFile(archive) as z:
        check(z.namelist()==["enwik8"] and
              z.getinfo("enwik8").file_size==SOURCE_BYTES,"official archive geometry")
        with z.open("enwik8") as payload,source.open("wb") as dest:
            shutil.copyfileobj(payload,dest,length=1<<20)
    archive.unlink()
    check(source.stat().st_size==SOURCE_BYTES and digest_file(source)==SOURCE_SHA,
          "canonical source SHA mismatch")
    record={"bytes":SOURCE_BYTES,"sha256":SOURCE_SHA}
    queries={
        "frequent_one_byte":(b"e",8001205),
        "two_hits":(b"Constitution of the United States of America",2),
        "absent_binary":(b"\x00GLYPH_CORE_L9S_ABSENT\x00",0),
        "binary_nul":(b"\x00",None),
    }
    trials=[]
    for label,(p,expected) in queries.items():
        for i in range(3):
            modes=("baseline_2pass","candidate_1pass") if i%2==0 else (
                "candidate_1pass","baseline_2pass")
            for mode in modes:
                start=time.perf_counter()
                cpu=time.process_time()
                if mode=="baseline_2pass":
                    w=golden.Work()
                    data=golden.scan_authenticated(source,record,p,1<<20,w)
                    io=w.source_read_bytes
                    passes=2
                else:
                    data=candidate.scan_authenticated_once(source,record,p,1<<20)
                    io=data["read_bytes"]
                    passes=1
                wall=round(time.perf_counter()-start,6)
                used_cpu=round(time.process_time()-cpu,6)
                check(data["scan_bytes"]==SOURCE_BYTES and data["eof"] and
                      io==passes*SOURCE_BYTES,"source scan incomplete")
                if expected is not None:
                    check(data["count"]==expected,
                          f"wrong {label} source occurrence count {data['count']}")
                trials.append({"query":label,"mode":mode,"round":i,
                               "wall_seconds":wall,"cpu_seconds":used_cpu,
                               "source_read_bytes":io,"sha_passes":passes,
                               "count":data["count"],"offset_prefix":data["offsets"][:10],
                               "truncated":data["truncated"]})
        old=[x for x in trials if x["query"]==label and x["mode"]=="baseline_2pass"]
        new=[x for x in trials if x["query"]==label and x["mode"]=="candidate_1pass"]
        for old_trial,new_trial in zip(old,new):
            check(old_trial["count"]==new_trial["count"] and
                  old_trial["offset_prefix"]==new_trial["offset_prefix"] and
                  old_trial["truncated"]==new_trial["truncated"],
                  "cross-mode semantics mismatch")
    summary={}
    for label in queries:
        old=statistics.median(x["wall_seconds"] for x in trials if x["query"]==label and
                              x["mode"]=="baseline_2pass")
        new=statistics.median(x["wall_seconds"] for x in trials if x["query"]==label and
                              x["mode"]=="candidate_1pass")
        summary[label]={"median_2pass_seconds":old,
                        "median_1pass_seconds":new,
                        "observed_speedup_x":round(old/new,3) if new else None,
                        "candidate_faster":new<old,
                        "source_read_reduction_fraction":0.5}
    source.unlink()
    return {"status":"P01_100MB_EXACT_BENCH_PASS",
            "source_bytes":SOURCE_BYTES,"source_sha256":SOURCE_SHA,
            "trials":trials,"summary":summary,"repetitions_per_mode":3,
            "process_peak_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "scope":"independent source byte scan only; NOT the full P/R/O verifier",
            "currentness":"unsigned historically pinned SHA, not an external witness"}

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--golden",type=Path,required=True)
    parser.add_argument("--candidate",type=Path,required=True)
    parser.add_argument("--mode",choices=("safety","enwik8"),required=True)
    parser.add_argument("--output",type=Path,required=True)
    a=parser.parse_args()
    a.output.mkdir(parents=True,exist_ok=False)
    result={"status":"BLOCKED"}
    try:
        golden=load(a.golden,"unchanged_core_l9_golden")
        candidate=load(a.candidate,"core_perf_p01_onepass")
        result=safety(a.output/"fixtures",golden,candidate) if a.mode=="safety" else (
            enwik8(a.output/"source",golden,candidate))
        code=0
    except Exception as e:
        result={"status":"BLOCKED","error":repr(e)}
        code=3
    (a.output/"result.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:result.get(k) for k in
          ("status","error","exact_oracle_cases","hostile_count","source_bytes","summary")},
          sort_keys=True),flush=True)
    return code

if __name__=="__main__":
    sys.exit(main())
