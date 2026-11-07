#!/usr/bin/env python3
"""CORE-L11: non-normative bridge from stable doc revisions to real L10 indexed leaves.

Uses existing actual frozen RLB2/RLR2/EF search, not fake plaintext index.
Outputs canonical overlay roots+physical receipts without retaining raw source.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
sys.path.insert(0,str(REPO/"experiments/core_l10_composition_v0"))
from producer import RootView

FORMAT="GLYPH_CORE_L11_INDEXED_IDENTITY_BRIDGE_RESEARCH_V0"
DOMAIN=b"GLYPH_L11_BRIDGE_ROOT_RESEARCH_V0\x00"
RECEIPT=FORMAT+"_RECEIPT"
PATTERNS=(b"alpha LEFT",b"ONLY_IN_V1",b"dup",b"not-present",b"shared",b"\x00\xff")

def demand(test,msg):
    if not test:raise ValueError(msg)

def canonical(x):
    return (json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False)+"\n").encode()

def digest(x):return hashlib.sha256(x).hexdigest()

def rootpin(root):return digest(DOMAIN+canonical(root))

def pathbytes(s):
    b=s.encode("utf-8")
    demand(b and not b.startswith(b"/") and b"\0" not in b and b"\\" not in b
           and all(p not in (b"",b".",b"..") for p in b.split(b"/")),"unsafe path")
    return b.hex()

def record(doc_id,revision,source_leaf,root,display_name,visible=True):
    d=root["documents"][source_leaf]
    return {"doc_id":doc_id,"revision":revision,"physical_leaf":source_leaf,
            "path_hex":pathbytes(display_name),"visible":visible,
            "byte_length":d["byte_length"],"sha256":d["sha256"],
            "leaf_binding_sha256":root["leaves"][source_leaf]["binding"]["sha256"]}

def snapshots(physical0,physical1,pins):
    v0=[record(i,0,i,physical0,"vault/%08d.bin"%i) for i in range(9)]
    v1=[dict(d) for d in v0]
    v1[1]=record(1,1,9,physical1,"renamed/alpha.bin")
    v1.extend((record(9,0,10,physical1,"a-early.bin"),
               record(10,0,11,physical1,"z-append.bin")))
    v2=[dict(d) for d in v1]
    v2[6]["visible"]=False
    arr=[]
    for i,(docs,phys,pin) in enumerate((
        (v0,physical0,pins["A0"]),(v1,physical1,pins["A1"]),
        (v2,physical1,pins["A1"]))):
        root={"format":FORMAT,"generation":i,"vault_id":"8f"*16,
              "parent_root_sha256":None if i==0 else rootpin(arr[-1]),
              "physical_root_sha256":pin,
              "physical_leaf_count":len(phys["leaves"]),
              "expected_physical_leaves":[x["binding"]["sha256"] for x in phys["leaves"]],
              "documents":docs,
              "search_scope":"VISIBLE_REVISION_IN_PINNED_SNAPSHOT"}
        arr.append(root)
    return arr

def main():
    p=argparse.ArgumentParser()
    for x in ("bundle","reference","output","expected-physical-pins-json"):
        p.add_argument("--"+x,required=True)
    a=p.parse_args()
    out=Path(a.output)
    demand(not out.exists(),"refuse existing result directory")
    out.mkdir(parents=True)
    pins=json.loads(a.expected_physical_pins_json)
    demand(type(pins) is dict and set(pins)=={"A0","B0","A1","B1"},"pinned layouts absent")
    bundle=Path(a.bundle);ref=Path(a.reference)
    path0=bundle/"roots/A0.json";path1=bundle/"roots/A1.json"
    raw0=path0.read_bytes();raw1=path1.read_bytes()
    demand(digest(raw0)==pins["A0"] and digest(raw1)==pins["A1"],
           "untrusted physical roots")
    physical0=json.loads(raw0);physical1=json.loads(raw1)
    roots=snapshots(physical0,physical1,pins)
    for i,root in enumerate(roots):
        (out/("V%d.json"%i)).write_bytes(canonical(root))
    (out/"receipts").mkdir()
    produced=[]
    views={}
    try:
        for generation in range(3):
            root=roots[generation]
            phys_index=0 if generation==0 else 1
            if phys_index not in views:
                kwargs={}
                if phys_index==1:
                    kwargs={"previous_path":path0,"previous_pin":pins["A0"]}
                views[phys_index]=RootView(bundle,path0 if phys_index==0 else path1,
                                          pins["A0" if phys_index==0 else "A1"],
                                          phys_index,ref,**kwargs)
            view=views[phys_index]
            for j,pattern in enumerate(PATTERNS):
                limit=256 if j!=2 else 2
                full=view.query(pattern,256)
                demand(full["coverage_complete"] and len(full["leaf_results"])==
                       root["physical_leaf_count"],"incomplete physical fanout")
                by_leaf={r["doc_id"]:r for r in full["leaf_results"]}
                coords=[]
                for d in root["documents"]:
                    if not d["visible"]:continue
                    leaf=by_leaf[d["physical_leaf"]]
                    for off in leaf["receipt"]["offsets"] if leaf["receipt"] is not None else []:
                        coords.append([d["doc_id"],d["revision"],off])
                coords.sort()
                limited=len(coords)>limit
                result={"format":RECEIPT,"bridge_root_sha256":rootpin(root),
                        "physical_root_sha256":root["physical_root_sha256"],
                        "query_hex":pattern.hex(),"max_offsets":limit,
                        "match_count":len(coords),"coordinates":coords[:limit],
                        "expected_physical_leaves":root["expected_physical_leaves"],
                        "verified_physical_leaves":root["expected_physical_leaves"],
                        "status":"UNTRUSTED" if limited else "FOUND" if coords else "PROVEN_EMPTY",
                        "reason":"LIMITED_COMPLETENESS" if limited else "FULL_INDEXED_SNAPSHOT_REPLAY"}
                key="V%d-Q%d"%(generation,j)
                file=out/"receipts"/(key+".json")
                file.write_bytes(canonical(result))
                (out/"receipts"/(key+".physical.json")).write_bytes(canonical(full))
                produced.append({"key":key,"count":len(coords),"status":result["status"]})
    finally:
        for v in views.values():v.close()
    report={"status":"CORE_L11_BRIDGE_PRODUCED_UNVERIFIED","pinned_bridge_roots":[rootpin(x) for x in roots],
            "physical_roots":[pins["A0"],pins["A1"]],"test_count":len(produced),
            "scenarios":produced,"retained_raw_source_bytes":0}
    (out/"result.json").write_bytes(canonical(report))
    print(json.dumps(report,sort_keys=True))
if __name__=="__main__":
    try:main()
    except Exception as e:
        print(json.dumps({"status":"UNTRUSTED","reason":str(e)[:400]}),file=sys.stderr)
        sys.exit(3)
