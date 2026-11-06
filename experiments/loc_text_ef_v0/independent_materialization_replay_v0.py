#!/usr/bin/env python3
"""GLYPH CORE-L8 independent offline digest replay, no encoder or decoder.

Inputs: external pinned root SHA, exact manifest, physical RLB2/RLR2/EF and
published source bytes, optional recorded status receipt. Recomputes every
digest independently. This is a simple BYTE IDENTITY REPLAY, not a
cryptographic non-membership proof or currentness/author-signature protocol.
No imports from trusted_materialize_v0.py by design.
"""
import argparse
import hashlib
import json
import os
import stat
import sys
from pathlib import Path

SCHEMA="GLYPH_CORE_L8_PINNED_SOURCE_MATERIALIZATION_V0"
SOURCE_COMMIT="c33cb493bc75f13ee4a904fabb190a29a196c94b"
HEX=frozenset("0123456789abcdef")

def require(ok,why):
    if not ok:raise ValueError(why)

def hbytes(data):
    return hashlib.sha256(data).hexdigest()

def hfile(path):
    d=hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda:f.read(1024*1024),b""):
            d.update(block)
    return d.hexdigest()

def noduplicates(pairs):
    d={}
    for k,v in pairs:
        require(k not in d,"duplicate JSON property")
        d[k]=v
    return d

def read_json(path,cap):
    st=path.lstat()
    require(stat.S_ISREG(st.st_mode) and st.st_size<=cap and st.st_size>0,
            "unbounded or nonregular JSON artifact")
    raw=path.read_bytes()
    data=json.loads(raw.decode("utf-8"),object_pairs_hook=noduplicates)
    require(isinstance(data,dict),"invalid JSON object")
    return data,raw

def valid_sha(value):
    return type(value) is str and len(value)==64 and all(c in HEX for c in value)

def verify_bytes(path,record):
    require(type(record) is dict and set(record)=={"bytes","sha256"},
            "invalid object identity schema")
    require(type(record["bytes"]) is int and 0<=record["bytes"]<(1<<36)
            and valid_sha(record["sha256"]),"invalid size/hash metadata")
    st=path.lstat()
    require(stat.S_ISREG(st.st_mode),"non-regular referenced source")
    require(st.st_size==record["bytes"],"source size not identical")
    require(hfile(path)==record["sha256"],"source digest not identical")

def independently_verify(args):
    require(valid_sha(args.root_sha256),"invalid external root")
    manifest,raw=read_json(Path(args.manifest),8192)
    require(hbytes(raw)==args.root_sha256,"root digest mismatch")
    require(set(manifest)=={"schema","frozen_commit","source","artifacts","decoder"}
            and manifest["schema"]==SCHEMA
            and manifest["frozen_commit"]==SOURCE_COMMIT,
            "manifest source/version/schema mismatch")
    comps=manifest["artifacts"]
    require(type(comps) is dict and set(comps)=={"rlb2","rlr2","ef"},
            "required search components not present")
    for name,path in [("rlb2",args.rlb2),("rlr2",args.rlr2),("ef",args.ef)]:
        verify_bytes(Path(path),comps[name])
    verify_bytes(Path(args.materialized),manifest["source"])
    backend=manifest["decoder"]
    require(type(backend) is dict and set(backend)=={"sha256"}
            and valid_sha(backend["sha256"]),"invalid decoder identity")
    if args.decoder:
        require(hfile(Path(args.decoder))==backend["sha256"],
                "recorded decoder code digest mismatch")
    if args.receipt:
        receipt,_=read_json(Path(args.receipt),32768)
        require(receipt.get("status")==
                "CORE_L8_PINNED_SHA256_MATERIALIZATION_PASS_RESEARCH",
                "producer receipt has no expected success status")
        require(receipt.get("manifest_sha256")==args.root_sha256
                and receipt.get("source_sha256")==manifest["source"]["sha256"]
                and type(receipt.get("source_bytes")) is int
                and receipt["source_bytes"]==manifest["source"]["bytes"],
                "producer claimed source/root identity mismatch")
        require(receipt.get("decoder_sha256")==backend["sha256"],
                "producer claimed decoder digest mismatch")
        for key in ("rlb2","rlr2","ef"):
            require(receipt.get(key+"_bytes")==comps[key]["bytes"],
                    "producer claimed index bytes mismatch")
    return {
        "status":"CORE_L8_OFFLINE_SOURCE_IDENTITY_REPLAY_PASS",
        "root_sha256":args.root_sha256,
        "source_sha256":manifest["source"]["sha256"],
        "source_bytes":manifest["source"]["bytes"],
        "pinned_artifact_bytes":sum(comps[key]["bytes"] for key in ("rlb2","rlr2","ef")),
        "artifacts_independently_hashed":["rlb2","rlr2","ef","materialized-source"],
        "decoder_code_sha256_checked":bool(args.decoder),
        "receipt_claims_matched":bool(args.receipt),
        "formal_limit":"byte-digest replay, no proof of publish order, completeness, signature or currentness",
    }

def main():
    p=argparse.ArgumentParser()
    for n in ("manifest","root-sha256","rlb2","rlr2","ef","materialized"):
        p.add_argument("--"+n,required=True)
    p.add_argument("--decoder")
    p.add_argument("--receipt")
    a=p.parse_args()
    try:
        print(json.dumps(independently_verify(a),sort_keys=True))
        return 0
    except Exception as e:
        print(json.dumps({"status":"UNTRUSTED",
                          "error_class":type(e).__name__,
                          "reason":str(e)[:250]},sort_keys=True),file=sys.stderr)
        return 3

if __name__=="__main__":
    sys.exit(main())
