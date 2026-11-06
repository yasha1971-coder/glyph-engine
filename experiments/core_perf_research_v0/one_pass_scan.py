#!/usr/bin/env python3
"""CORE-PERF-P01: one-pass authenticated exact-byte source scan, RESEARCH ONLY.

No result is returned/published before EOF, SHA-256 of ALL input bytes and
stable fd signature match the externally pinned source identity. One-byte
patterns use bytes.count for exact cardinality; output positions are capped
at 256, but a truncated enumeration is always UNTRUSTED. Multibyte patterns
can overlap, including at chunk boundaries, with carry <= pattern_length-1.

Does NOT authenticate the root's owner, currentness, FM/rank/EF semantics,
or a full index dependency set. Baseline L9 verifier is not modified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys

MAX_SOURCE=100_000_000
MAX_HITS=256
MAX_PATTERN=4097
MIN_CHUNK=1<<20
MAX_CHUNK=4<<20

def demand(ok,message):
    if not ok:
        raise ValueError(message)

def source_signature(stream):
    s=os.fstat(stream.fileno())
    demand(stat.S_ISREG(s.st_mode),"not a regular source file")
    return s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns

def read_source(path,cap):
    obj=os.lstat(path)
    demand(stat.S_ISREG(obj.st_mode) and obj.st_size<=cap,
           "source missing, symlinked or larger than admission cap")
    fd=os.open(path,os.O_RDONLY|getattr(os,"O_NOFOLLOW",0)|
               getattr(os,"O_NONBLOCK",0))
    stream=os.fdopen(fd,"rb")
    try:
        demand(source_signature(stream)[2]<=cap,"source grew before scan")
        return stream
    except BaseException:
        stream.close()
        raise

def source_record(size,sha256):
    demand(type(size) is int and 0<size<=MAX_SOURCE,
           "unsupported source size or type")
    demand(type(sha256) is str and len(sha256)==64 and
           all(c in "0123456789abcdef" for c in sha256),
           "invalid externally pinned source sha256")
    return {"bytes":size,"sha256":sha256}

def scan_authenticated_once(path,record,pattern,chunk_bytes=MIN_CHUNK):
    """Exact complete count, <=256 prefix offsets; no result before full auth."""
    demand(type(pattern) is bytes and 0<len(pattern)<=MAX_PATTERN,
           "invalid nonempty binary query")
    demand(type(chunk_bytes) is int and MIN_CHUNK<=chunk_bytes<=MAX_CHUNK,
           "invalid chunk-size admission")
    demand(type(record) is dict and set(record)=={"bytes","sha256"},
           "invalid source record shape")
    source_record(record["bytes"],record["sha256"])
    fullhash=hashlib.sha256()
    count=total=0
    offsets=[]
    excerpts=[]
    carry=b""
    is_one_byte=len(pattern)==1
    with read_source(path,MAX_SOURCE) as f:
        sig=source_signature(f)
        demand(sig[2]==record["bytes"],"source size does not match pinned identity")
        while True:
            block=f.read(chunk_bytes)
            if not block:
                break
            demand(total+len(block)<=record["bytes"],"source grew while scanning")
            fullhash.update(block)
            if is_one_byte:
                # O(n) count in the native bytes primitive; do not iterate
                # millions of single-character offsets in Python.
                count+=block.count(pattern)
                start=0
                while len(offsets)<MAX_HITS:
                    at=block.find(pattern,start)
                    if at<0:
                        break
                    offsets.append(total+at)
                    excerpts.append({"offset":total+at,"bytes":1,
                        "sha256":hashlib.sha256(pattern).hexdigest(),
                        "bytes_included":False})
                    start=at+1
            else:
                window=carry+block
                start=0
                while True:
                    at=window.find(pattern,start)
                    if at<0:
                        break
                    count+=1
                    if len(offsets)<MAX_HITS:
                        offset=total-len(carry)+at
                        offsets.append(offset)
                        excerpts.append({"offset":offset,"bytes":len(pattern),
                            "sha256":hashlib.sha256(window[at:at+len(pattern)]).hexdigest(),
                            "bytes_included":False})
                    start=at+1
                carry=window[-(len(pattern)-1):]
            total+=len(block)
        demand(total==record["bytes"] and source_signature(f)==sig,
               "source changed or incomplete at EOF")
    digest=fullhash.hexdigest()
    demand(digest==record["sha256"],"full source hash mismatch; result UNTRUSTED")
    demand(len(offsets)==min(count,MAX_HITS) and offsets==sorted(offsets),
           "internal result cardinality/order mismatch")
    return {
        "count":count,"offsets":offsets,"excerpts":excerpts,
        "scan_bytes":total,"read_bytes":total,"source_hash_passes":1,
        "sha256":digest,"eof":True,"truncated":count>MAX_HITS,
        "status":"UNTRUSTED" if count>MAX_HITS else ("FOUND" if count else "PROVEN_EMPTY"),
        "reason":"LIMITED_COMPLETENESS" if count>MAX_HITS else "AUTHENTICATED_FULL_SCAN",
        "index_semantics_proven":False,"signed_root":False,
    }

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--source",type=Path,required=True)
    p.add_argument("--source-bytes",type=int,required=True)
    p.add_argument("--sha256",required=True)
    p.add_argument("--pattern-hex",required=True)
    p.add_argument("--chunk-bytes",type=int,default=MIN_CHUNK)
    args=p.parse_args()
    try:
        demand(len(args.pattern_hex)%2==0 and
               all(c in "0123456789abcdef" for c in args.pattern_hex),
               "noncanonical pattern hex")
        result=scan_authenticated_once(args.source,
            source_record(args.source_bytes,args.sha256),
            bytes.fromhex(args.pattern_hex),args.chunk_bytes)
        print(json.dumps(result,sort_keys=True),flush=True)
        return 3 if result["truncated"] else 0
    except (OSError,ValueError,TypeError,OverflowError) as e:
        print(json.dumps({"status":"UNTRUSTED","error":str(e)[:230]},sort_keys=True),
              file=sys.stderr,flush=True)
        return 3

if __name__=="__main__":
    sys.exit(main())
