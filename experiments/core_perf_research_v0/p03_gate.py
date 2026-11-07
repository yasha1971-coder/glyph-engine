#!/usr/bin/env python3
"""P03: independent binary oracle, controlled P02-vs-C++ AC 100MB experiment."""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import random
import resource
import shutil
import statistics
import subprocess
import sys
import time
import urllib.request
import zipfile

SRC_SHA="2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8"
SRC_LEN=100_000_000
ZIP_LEN=36_445_475
URL="https://mattmahoney.net/dc/enwik8.zip"

def check(x,msg):
    if not x:raise AssertionError(msg)
def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    check(spec and spec.loader,"module missing")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod
def sha(x):return hashlib.sha256(x).hexdigest()
def read_sha(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1<<20),b""):h.update(block)
    return h.hexdigest()
def oracle(data,p):
    xs=[];at=data.find(p)
    while at>=0:
        xs.append(at)
        at=data.find(p,at+1)
    return xs
def native(binary,source,pin,patterns):
    cmd=[str(binary),"--source",str(source),"--source-bytes",
         str(source.stat().st_size),"--sha256",pin]
    for p in patterns:cmd.extend(["--pattern-hex",p.hex()])
    begin=time.perf_counter()
    before=resource.getrusage(resource.RUSAGE_CHILDREN)
    run=subprocess.run(cmd,capture_output=True,text=True,timeout=90)
    end=resource.getrusage(resource.RUSAGE_CHILDREN)
    wall=time.perf_counter()-begin
    cpu=(end.ru_utime+end.ru_stime)-(before.ru_utime+before.ru_stime)
    if run.returncode==0:
        out=json.loads(run.stdout)
        check(out["source_hash_passes"]==1 and
              out["source_reads_bytes"]==source.stat().st_size and
              out["eof"] is True and out["signed_root"] is False,
              "missing native full authentication")
        return {"rc":0,"output":out,"wall":wall,"cpu":cpu,
                "child_rss_kib":end.ru_maxrss}
    return {"rc":run.returncode,"stderr":run.stderr,"wall":wall,
            "cpu":cpu,"child_rss_kib":end.ru_maxrss}
def positive(binary,source,data,patterns,p02):
    pin=sha(data)
    native_out=native(binary,source,pin,patterns)
    check(native_out["rc"]==0,"native positive rejected: "+str(native_out.get("stderr","")))
    out=native_out["output"]
    check(out["automaton_nodes"]<=4097 and out["automaton_bytes"]<=5*(1<<20),
          "automaton exceeded promised admission")
    reference=p02.scan_batch_once(source,{"bytes":len(data),"sha256":pin},patterns)
    for p,got,ref in zip(patterns,out["results"],reference["results"]):
        offsets=oracle(data,p)
        check(got["count"]==len(offsets) and got["offsets"]==offsets[:256] and
              got["status"]==ref["status"] and
              got["count"]==ref["count"] and got["offsets"]==ref["offsets"],
              "native/P02/independent oracle diverged for "+p.hex())
    return native_out
def reject(label,cmd,negative):
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=10)
    check(p.returncode==3 and "UNTRUSTED" in p.stderr and not p.stdout,
          "FAIL OPEN "+label+" "+p.stdout[:200]+" "+p.stderr[:200])
    negative.append(label)
def safety(root,p02,binary):
    root.mkdir(parents=True)
    source=root/"source"
    rng=random.Random(20261107)
    tested=0
    for n in range(1,181):
        # Biased repetitive bytes plus some high-entropy fixtures.
        data=(bytes(rng.randrange(256) for _ in range(n)) if n%3==0
              else bytes(rng.choice([0,0,0,1,1,65,66,255]) for _ in range(n)))
        source.write_bytes(data)
        pats=[b"\0",b"\xff",b"A",b"aa",b"\0\0",b"\0\xff",
              b"AB",b"BA",b"not-here",b"\x00\x01\x00"]
        for i in range(6):
            if i<n:pats.append(data[i:i+min(n-i,1+i)])
        pats=list(dict.fromkeys(pats))[:16]
        positive(binary,source,data,pats,p02)
        tested+=len(pats)
    # Multi-MiB repetitive worst case with very frequent overlapping matches.
    data=b"a"*(4<<20)+b"\0\xff\0\xff"+b"a"*(1<<20)
    source.write_bytes(data)
    pats=[b"a",b"aa",b"aaa",b"aaaa",b"aaaaa",b"a"*100,
          b"\0",b"\xff",b"\0\xff",b"\xff\0",b"never",b"a"*4096]
    check(sum(map(len,pats))>4096,"bad total-budget stress fixture")
    # The oversized set must be rejected, then bounded valid subset accepted.
    negative=[]
    bad_args=[str(binary),"--source",str(source),"--source-bytes",str(len(data)),
              "--sha256",sha(data)]
    for p in pats:bad_args.extend(["--pattern-hex",p.hex()])
    reject("pattern-sum-over-memory-budget",bad_args,negative)
    pats=[b"a",b"aa",b"aaa",b"aaaa",b"aaaaa",b"a"*100,
          b"\0",b"\xff",b"\0\xff",b"\xff\0",b"never"]
    positive(binary,source,data,pats,p02)
    tested+=len(pats)
    # Hard crossing at 1MiB boundary, mixed lengths and overlapping queries.
    for k in range(5):
        payload=bytearray(b"_"*(2*(1<<20)+1100))
        for pos in ((1<<20)-2+k,(1<<20)+7+k,2*(1<<20)-1):
            payload[pos:pos+5]=b"ababa"
        payload[-90:-86]=b"\0\xff\0\xff"
        data=bytes(payload);source.write_bytes(data)
        pats=[b"a",b"aba",b"ababa",b"bab",b"ba",b"_",
              b"\0",b"\xff",b"\0\xff",b"\xff\0",b"no"]
        positive(binary,source,data,pats,p02)
        tested+=len(pats)
    source.write_bytes(b"a"*1024)
    good=[str(binary),"--source",str(source),"--source-bytes","1024",
          "--sha256",read_sha(source),"--pattern-hex","61"]
    variants=[
        ("bad-pin",good[:6]+["0"*64]+good[7:]),
        ("bad-length",good[:4]+["1025"]+good[5:]),
        ("missing-pattern",good[:8]),
        ("empty-pattern",good[:-1]+[""]),
        ("odd-hex",good[:-1]+["6"]),
        ("uppercase-hex",good[:-1]+["FF"]),
        ("unparseable-bytes",good[:4]+["-1"]+good[5:]),
        ("unknown-flag",good+["--bogus","1"]),
        ("no-source",good[2:]),
        ("too-many-patterns",good+sum((["--pattern-hex",f"{i:02x}"] for i in range(16)),[])),
        ("duplicate-pattern",good+["--pattern-hex","61"]),
        ("nonregular-input",good[:2]+[str(root)]+good[3:]),
    ]
    link=root/"symlink";link.symlink_to(source)
    variants.append(("symlink",good[:2]+[str(link)]+good[3:]))
    source.write_bytes(b"b"*1024)
    variants.append(("mutated-source",good))
    for name,args in variants:reject(name,args,negative)
    source.unlink()
    reject("missing-source",good,negative)
    check(len(negative)==15,"hostile coverage drift")
    return {"status":"P03_NATIVE_AC_SAFETY_PASS","exact_oracle_checks":tested,
            "hostile_count":len(negative),"rejected":negative}
def enwik8(root,p02,binary):
    root.mkdir(parents=True)
    archive=root/"enwik8.zip";source=root/"enwik8"
    with urllib.request.urlopen(URL,timeout=180) as src,archive.open("wb") as out:
        shutil.copyfileobj(src,out,length=1<<20)
    check(archive.stat().st_size==ZIP_LEN,"official zip drift")
    with zipfile.ZipFile(archive) as z:
        check(z.namelist()==["enwik8"] and z.getinfo("enwik8").file_size==SRC_LEN,
              "official source geometry")
        with z.open("enwik8") as inp,source.open("wb") as out:
            shutil.copyfileobj(inp,out,length=1<<20)
    archive.unlink()
    check(source.stat().st_size==SRC_LEN and read_sha(source)==SRC_SHA,
          "canonical SHA mismatch")
    with source.open("rb") as f:
        f.seek(SRC_LEN-80);last=f.read(48)
        f.seek(1_048_560);cross=f.read(48)
        f.seek(10_494_416);nonascii=f.read(48)
        fixed=[]
        for pos in [55,112344,857709,999999,9002022,19700000,68334441,99999140]:
            f.seek(pos);fixed.append(f.read(32))
    choices=[b"e",b"Constitution of the United States of America",
             b"\x00GLYPH_CORE_L9S_ABSENT\x00",b"\x00",
             b"American Declaration of Independence",
             last,cross,nonascii,*fixed]
    check(len(choices)==16 and len(set(choices))==16,"invalid pattern set")
    record={"bytes":SRC_LEN,"sha256":SRC_SHA}
    samples=[]
    for q in [1,2,4,8,16]:
        pat=choices[:q]
        for trial in range(3):
            modes=("batch","native") if trial%2==0 else ("native","batch")
            responses={}
            for mode in modes:
                if mode=="native":
                    got=native(binary,source,SRC_SHA,pat)
                    check(got["rc"]==0,"native canonical error")
                    out=got["output"]
                    check(out["automaton_nodes"]<=4097 and
                          out["automaton_bytes"]<=5*(1<<20),"native RSS bound")
                    results=out["results"]
                    wall,cpu=got["wall"],got["cpu"]
                    rss=got["child_rss_kib"]
                    nodes=out["automaton_nodes"]
                else:
                    start=time.perf_counter();begin=time.process_time()
                    out=p02.scan_batch_once(source,record,pat)
                    results=out["results"]
                    wall=time.perf_counter()-start
                    cpu=time.process_time()-begin
                    rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                    nodes=None
                responses[mode]=results
                samples.append({"patterns":q,"round":trial,"mode":mode,
                                "wall_seconds":round(wall,6),
                                "cpu_seconds":round(cpu,6),
                                "process_peak_rss_kib":rss,
                                "automaton_nodes":nodes,
                                "read_bytes":SRC_LEN,
                                "counts":[r["count"] for r in results]})
            for a,b in zip(responses["native"],responses["batch"]):
                for key in ["pattern_hex","count","offsets","status",
                            "truncated","reason"]:
                    check(a[key]==b[key],"real canonical corpus mismatch "+key)
    report={}
    for q in [1,2,4,8,16]:
        old=statistics.median(s["wall_seconds"] for s in samples
                              if s["patterns"]==q and s["mode"]=="batch")
        new=statistics.median(s["wall_seconds"] for s in samples
                              if s["patterns"]==q and s["mode"]=="native")
        report[str(q)]={"p02_batch_median_seconds":old,
                        "p03_native_ac_median_seconds":new,
                        "speedup_x":round(old/new,3),"native_faster":new<old}
    return {"status":"P03_100MB_NATIVE_AC_EXACT_BENCH_PASS",
            "source_sha256":SRC_SHA,"source_bytes":SRC_LEN,
            "comparison":"C++ AC vs pinned P02 Python batch, both one-pass SHA",
            "trial_data":samples,"summary":report,
            "rounds_per_profile":3,
            "P03_C_prediction_Q16_faster":report["16"]["native_faster"],
            "python_process_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss}
def main():
    p=argparse.ArgumentParser()
    p.add_argument("--native",type=Path,required=True)
    p.add_argument("--p02",type=Path,required=True)
    p.add_argument("--mode",choices=["safety","enwik8"],required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    check(not args.output.exists(),"output exists")
    args.output.mkdir(parents=True)
    result={"status":"BLOCKED"}
    try:
        p02=load(args.p02,"pinned_python_p02")
        result=(safety(args.output/"fixtures",p02,args.native) if args.mode=="safety"
                else enwik8(args.output/"canonical",p02,args.native))
        code=0
    except Exception as exc:
        result={"status":"BLOCKED","error":repr(exc)}
        code=3
    (args.output/"result.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:result.get(k) for k in
        ["status","error","exact_oracle_checks","hostile_count","summary",
         "P03_C_prediction_Q16_faster"]},sort_keys=True),flush=True)
    return code
if __name__=="__main__":
    sys.exit(main())
