#!/usr/bin/env python3
"""P02 independent exact-oracle fixtures + paired P01 vs P02 100MB measurements."""
import argparse
import hashlib
import importlib.util
import itertools
import json
import random
import resource
import shutil
import statistics
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

N=100_000_000
HASH="2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"
ZIP_BYTES=36_445_475
URL="https://mattmahoney.net/dc/enwik8.zip"
CHUNK=1<<20

def check(ok,msg):
    if not ok:raise AssertionError(msg)

def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    check(spec is not None and spec.loader is not None,"missing module")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def file_sha(p):
    h=hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()

def oracle(b,p):
    coords=[]
    i=b.find(p)
    while i>=0:
        coords.append(i)
        i=b.find(p,i+1)
    return coords

def compare(p,patterns,single,batch,expected=None):
    record={"bytes":p.stat().st_size,"sha256":file_sha(p)}
    result=batch.scan_batch_once(p,record,patterns,CHUNK)
    check(result["source_reads_bytes"]==record["bytes"] and
          result["source_hash_passes"]==1 and result["sha256"]==record["sha256"] and
          result["eof"],"batch source incomplete")
    for pat,r in zip(patterns,result["results"]):
        old=single.scan_authenticated_once(p,record,pat,CHUNK)
        check(r["count"]==old["count"] and r["offsets"]==old["offsets"] and
              r["excerpts"]==old["excerpts"] and
              r["truncated"]==old["truncated"] and r["status"]==old["status"],
              "different exact result for "+repr(pat))
        if expected is not None:
            coords=oracle(expected,pat)
            check(r["count"]==len(coords) and
                  r["offsets"]==coords[:256],"source byte oracle mismatch")
    return result

def reject(label,f,negative):
    try:f()
    except (OSError,ValueError,TypeError,OverflowError):
        negative.append(label)
        return
    raise AssertionError("FAIL-OPEN: "+label)

def safety(root,single,batch):
    root.mkdir(parents=True,exist_ok=False)
    source=root/"source.bin"
    cases=0
    rng=random.Random(20261106)
    for n in range(1,129):
        data=bytes(rng.choice((0,0,0,1,65,66,255)) for _ in range(n))
        source.write_bytes(data)
        pats=[b"\x00",b"\xff",b"A",b"B",b"\x00\x00",b"\x00\x01",
              b"AB",b"\xff\x00",b"BA",b"\x00\xff\x00",b"AAB",b"ABAB"]
        pats.extend([data[i:i+min(n-i,4+j)] for j,i in
                     enumerate(range(min(n,8))) if n-i>=1])
        pats=list(dict.fromkeys(pats))[:16]
        compare(source,pats,single,batch,data)
        cases+=len(pats)
    # Mixed lengths test: tail from 1024-byte pattern must NOT cause a
    # short two-byte query completed in the carry to be counted twice.
    longpat=b"\x00"+bytes(range(1,256))*3+b"\xff"*34
    check(len(longpat)<4096,"bad long test pattern")
    marker=b"ab"
    for k in range(8):
        data=bytearray(b"_"*(2*CHUNK+3500))
        data[CHUNK-900:CHUNK-900+len(longpat)]=longpat
        for off in (CHUNK-1000+k,CHUNK-20+k,CHUNK-1,
                    CHUNK+5+k,2*CHUNK-1,2*CHUNK+100+k):
            data[off:off+2]=marker
        data=bytes(data)
        source.write_bytes(data)
        pats=[marker,longpat,b"__ab",b"\x00",b"\xff",b"not-here",
              b"ab__a",b"_",b"abab",b"\x00\x01"]
        compare(source,pats,single,batch,data)
        cases+=len(pats)
    source.write_bytes(b"a"*1024)
    pinned={"bytes":1024,"sha256":file_sha(source)}
    saturated=batch.scan_batch_once(source,pinned,[b"a",b"aa",b"aaa",b"zz"])
    check([r["count"] for r in saturated["results"]]==[1024,1023,1022,0] and
          [r["status"] for r in saturated["results"]]==[
              "UNTRUSTED","UNTRUSTED","UNTRUSTED","PROVEN_EMPTY"],
          "incomplete results were mislabelled as authoritative")
    cases+=4
    negative=[]
    reject("wrong-hash",lambda:batch.scan_batch_once(
        source,{"bytes":1024,"sha256":"0"*64},[b"a"]),negative)
    reject("wrong-size",lambda:batch.scan_batch_once(
        source,{"bytes":1025,"sha256":pinned["sha256"]},[b"a"]),negative)
    reject("short-size",lambda:batch.scan_batch_once(
        source,{"bytes":1023,"sha256":pinned["sha256"]},[b"a"]),negative)
    reject("empty-pattern",lambda:batch.scan_batch_once(source,pinned,[b""]),negative)
    reject("empty-list",lambda:batch.scan_batch_once(source,pinned,[]),negative)
    reject("duplicate-pattern",lambda:batch.scan_batch_once(
        source,pinned,[b"a",b"a"]),negative)
    reject("pattern-str",lambda:batch.scan_batch_once(source,pinned,["a"]),negative)
    reject("query-count-over-cap",lambda:batch.scan_batch_once(
        source,pinned,[bytes([i]) for i in range(17)]),negative)
    reject("pattern-total-over-cap",lambda:batch.scan_batch_once(
        source,pinned,[b"a"*4096,b"b"]),negative)
    reject("oversize-individual",lambda:batch.scan_batch_once(
        source,pinned,[b"a"*4098]),negative)
    reject("chunk-too-small",lambda:batch.scan_batch_once(
        source,pinned,[b"a"],chunk_bytes=3),negative)
    reject("chunk-too-large",lambda:batch.scan_batch_once(
        source,pinned,[b"a"],chunk_bytes=10<<20),negative)
    reject("invalid-record",lambda:batch.scan_batch_once(
        source,{**pinned,"isTrusted":True},[b"a"]),negative)
    link=root/"alias"
    link.symlink_to(source)
    reject("symlink-source",lambda:batch.scan_batch_once(link,pinned,[b"a"]),negative)
    source.write_bytes(b"b"*1024)
    reject("source-mutation",lambda:batch.scan_batch_once(
        source,pinned,[b"a",b"b"]),negative)
    source.unlink()
    reject("missing-source",lambda:batch.scan_batch_once(
        source,pinned,[b"a"]),negative)
    check(len(negative)==16,"missing hostile tests")
    return {"status":"P02_EXACT_SAFETY_PASS","oracle_cases":cases,
            "hostiles_rejected":negative,"hostile_count":len(negative),
            "scope":"independent source byte oracle + unchanged P01 scanner"}

def source_patterns(source):
    choices=[b"e",b"Constitution of the United States of America",
             b"\x00GLYPH_CORE_L9S_ABSENT\x00",b"\x00",
             b"American Declaration of Independence",
             source[-48:],source[1048560:1048608],
             source[1049416:1049464]]
    for at in (55,112344,857709,999999,9002022,19700000,68334441,99999140):
        choices.append(source[at:at+32])
    check(len(choices)==16 and len(set(choices))==16 and
          all(p for p in choices),"not 16 independent patterns")
    return choices

def enwik8(root,single,batch):
    root.mkdir(parents=True,exist_ok=False)
    source=root/"enwik8"
    zip_path=root/"enwik8.zip"
    with urllib.request.urlopen(URL,timeout=180) as fd,zip_path.open("wb") as out:
        shutil.copyfileobj(fd,out,length=CHUNK)
    check(zip_path.stat().st_size==ZIP_BYTES,"official archive size drift")
    with zipfile.ZipFile(zip_path) as archive:
        check(archive.namelist()==["enwik8"] and
              archive.getinfo("enwik8").file_size==N,"official enwik8 drift")
        with archive.open("enwik8") as fd,source.open("wb") as out:
            shutil.copyfileobj(fd,out,length=CHUNK)
    zip_path.unlink()
    check(source.stat().st_size==N and file_sha(source)==HASH,
          "wrong canonical enwik8 SHA")
    # Derive static batch query bytes from a once-authenticated canonical input.
    raw=source.read_bytes()
    patterns=source_patterns(raw)
    del raw
    pinned={"bytes":N,"sha256":HASH}
    trials=[]
    for q in (1,2,4,8,16):
        these=patterns[:q]
        for trial in range(3):
            modes=("sequential","batch") if trial%2==0 else ("batch","sequential")
            paired={}
            for mode in modes:
                begin=time.perf_counter()
                cpu=time.process_time()
                if mode=="batch":
                    data=batch.scan_batch_once(source,pinned,these)
                    produced=data["results"]
                    read=data["source_reads_bytes"]
                else:
                    produced=[]
                    for pattern in these:
                        src=single.scan_authenticated_once(source,pinned,pattern)
                        produced.append({
                            "pattern_hex":pattern.hex(),"count":src["count"],
                            "offsets":src["offsets"],"excerpts":src["excerpts"],
                            "truncated":src["truncated"],"status":src["status"]})
                    read=q*N
                wall=round(time.perf_counter()-begin,6)
                used_cpu=round(time.process_time()-cpu,6)
                check(read==(N if mode=="batch" else q*N),
                      "incorrect physical source reads")
                paired[mode]=produced
                trials.append({"pattern_count":q,"mode":mode,"round":trial,
                    "wall_seconds":wall,"cpu_seconds":used_cpu,
                    "source_reads_bytes":read,
                    "counts":[r["count"] for r in produced],
                    "statuses":[r["status"] for r in produced]})
            for old,new in zip(paired["sequential"],paired["batch"]):
                for key in ("count","offsets","excerpts","truncated","status"):
                    check(old[key]==new[key],"batch semantics differ: "+key)
    summary={}
    for q in (1,2,4,8,16):
        old=statistics.median(x["wall_seconds"] for x in trials if
               x["pattern_count"]==q and x["mode"]=="sequential")
        new=statistics.median(x["wall_seconds"] for x in trials if
               x["pattern_count"]==q and x["mode"]=="batch")
        summary[str(q)]={
            "sequential_median_seconds":old,
            "batch_median_seconds":new,
            "speedup_x":round(old/new,3) if new else None,
            "batch_faster":new<old,
            "source_read_reduction_fraction":1-1/q,
            "note":"Physical source reads only; in-memory native scanning still scales with pattern count"}
    source.unlink()
    return {"status":"P02_100MB_EXACT_BENCH_PASS",
        "source_bytes":N,"source_sha256":HASH,
        "query_lengths":[len(p) for p in patterns],
        "pattern_hex":[p.hex() for p in patterns],
        "repetitions":3,"trials":trials,"summary":summary,
        "peak_process_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "scope":"isolated exact source-byte authentication and batch search; NO indexed query/locate claim"}

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--one-pass",type=Path,required=True)
    a.add_argument("--batch",type=Path,required=True)
    a.add_argument("--mode",choices=("safety","enwik8"),required=True)
    a.add_argument("--output",type=Path,required=True)
    args=a.parse_args()
    check(not args.output.exists(),"refuse existing result tree")
    args.output.mkdir(parents=True)
    outcome={"status":"BLOCKED"}
    try:
        single=load(args.one_pass,"original_p01_scanner")
        # import-time dependency under this directory
        sys.path.insert(0,str(args.batch.parent))
        candidate=load(args.batch,"candidate_p02_batch")
        outcome=(safety(args.output/"small",single,candidate) if args.mode=="safety"
                 else enwik8(args.output/"canonical",single,candidate))
        code=0
    except Exception as e:
        outcome={"status":"BLOCKED","error":repr(e)}
        code=3
    (args.output/"result.json").write_text(json.dumps(outcome,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:outcome.get(k) for k in
        ("status","error","oracle_cases","hostile_count","source_bytes","summary")},
        sort_keys=True),flush=True)
    return code

if __name__=="__main__":
    sys.exit(main())
