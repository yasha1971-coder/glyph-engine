#!/usr/bin/env python3
"""P-PERF-02: batch independent exact byte patterns in ONE authenticated source pass.

Research-only source oracle. Not FM/EF index proof, currentness, or a signed
absence proof. No answer is made available before full SHA+EOF+fd signature.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from one_pass_scan import (MAX_SOURCE,MAX_HITS,MIN_CHUNK,MAX_CHUNK,
                           MAX_PATTERN,read_source,source_record,source_signature,demand)

MAX_QUERIES=16
MAX_TOTAL_PATTERN_BYTES=4096

def scan_batch_once(path,record,patterns,chunk_bytes=MIN_CHUNK):
    demand(type(record) is dict and set(record)=={"bytes","sha256"},
           "invalid source record shape")
    source_record(record["bytes"],record["sha256"])
    demand(type(patterns) is list and 1<=len(patterns)<=MAX_QUERIES,
           "query count outside admission")
    demand(all(type(p) is bytes and 0<len(p)<=MAX_PATTERN for p in patterns),
           "invalid binary pattern")
    demand(sum(map(len,patterns))<=MAX_TOTAL_PATTERN_BYTES and
           len(set(patterns))==len(patterns),
           "duplicate patterns or unbounded automaton budget")
    demand(type(chunk_bytes) is int and MIN_CHUNK<=chunk_bytes<=MAX_CHUNK,
           "invalid chunk size")
    carry_bytes=max(len(p) for p in patterns)-1
    counts=[0]*len(patterns)
    offsets=[[] for _ in patterns]
    phashes=[hashlib.sha256(p).hexdigest() for p in patterns]
    sha=hashlib.sha256()
    total=0
    chunks=0
    tail=b""
    with read_source(path,MAX_SOURCE) as stream:
        signature=source_signature(stream)
        demand(signature[2]==record["bytes"],"wrong pinned size")
        while True:
            block=stream.read(chunk_bytes)
            if not block:
                break
            demand(total+len(block)<=record["bytes"],
                   "source grew during batch scan")
            sha.update(block)
            window=tail+block
            start_window=total-len(tail)
            for i,p in enumerate(patterns):
                # All searches share the same input read. The native bytes
                # primitives scan the in-memory block; CPU work may remain O(Q*n).
                if len(p)==1:
                    counts[i]+=block.count(p)
                    pos=0
                    while len(offsets[i])<MAX_HITS:
                        at=block.find(p,pos)
                        if at<0:
                            break
                        offsets[i].append(total+at)
                        pos=at+1
                else:
                    pos=0
                    while True:
                        at=window.find(p,pos)
                        if at<0:
                            break
                        absolute=start_window+at
                        # A large carry may contain an entire SHORTER query.
                        # Such matches were already counted in a prior chunk.
                        if absolute+len(p)>total:
                            counts[i]+=1
                            if len(offsets[i])<MAX_HITS:
                                offsets[i].append(absolute)
                        pos=at+1
            tail=window[-carry_bytes:] if carry_bytes else b""
            total+=len(block)
            chunks+=1
        demand(total==record["bytes"] and source_signature(stream)==signature,
               "unstable source or missing EOF")
    digest=sha.hexdigest()
    demand(digest==record["sha256"],"source SHA changed during batch scan")
    result=[]
    for i,p in enumerate(patterns):
        n=counts[i]
        out=offsets[i]
        demand(out==sorted(out) and len(out)==min(n,MAX_HITS),
               "incomplete/noncanonical offsets")
        truncated=n>MAX_HITS
        result.append({
            "pattern_hex":p.hex(),
            "count":n,
            "offsets":out,
            "excerpts":[{"offset":off,"bytes":len(p),
                         "sha256":phashes[i],"bytes_included":False}
                        for off in out],
            "truncated":truncated,
            "status":"UNTRUSTED" if truncated else ("FOUND" if n else "PROVEN_EMPTY"),
            "reason":"LIMITED_COMPLETENESS" if truncated else "AUTHENTICATED_FULL_SCAN",
        })
    return {"profile":"CORE_PERF_P02_SOURCE_SCAN_RESEARCH",
            "results":result,"sha256":digest,"source_bytes":total,
            "source_reads_bytes":total,"source_hash_passes":1,
            "eof":True,"chunks":chunks,"pattern_count":len(patterns),
            "index_semantics_proven":False,"signed_root":False,
            "complete_scope":"single externally SHA-pinned source bytes"}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--source-bytes",type=int,required=True)
    p.add_argument("--sha256",required=True)
    p.add_argument("--patterns-hex-json",required=True,
                   help="JSON array of lowercase nonempty byte-pattern hex")
    p.add_argument("--chunk-bytes",type=int,default=MIN_CHUNK)
    args=p.parse_args()
    try:
        raw=json.loads(args.patterns_hex_json)
        demand(type(raw) is list and all(type(x) is str and
               len(x)%2==0 and all(ch in "0123456789abcdef" for ch in x)
               for x in raw),"noncanonical hex patterns")
        out=scan_batch_once(args.source,source_record(args.source_bytes,args.sha256),
                            [bytes.fromhex(x) for x in raw],args.chunk_bytes)
        print(json.dumps(out,sort_keys=True))
        return 3 if any(x["truncated"] for x in out["results"]) else 0
    except (OSError,ValueError,TypeError,OverflowError) as exc:
        print(json.dumps({"status":"UNTRUSTED","error":str(exc)[:240]},
                         sort_keys=True),file=sys.stderr)
        return 3

if __name__=="__main__":
    sys.exit(main())
