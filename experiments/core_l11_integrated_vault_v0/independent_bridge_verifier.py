#!/usr/bin/env python3
"""CORE-L11 independent replay of logical V2 docs over real verified L10 leaves.

NO bridge producer imports. Freshly restores every leaf only from RLB2/RLR2/EF
through SHA-pinned L8 materialization. Independently executes unchanged
composition verifier on every unique physical receipt, then scans restored
sources for stable doc/revision scoped query answers. Full source scan, not a
compact absence proof. Unsigned pins only.
"""
import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import tempfile
import time

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]
FORMAT="GLYPH_CORE_L11_INDEXED_IDENTITY_BRIDGE_RESEARCH_V0"
DOMAIN=b"GLYPH_L11_BRIDGE_ROOT_RESEARCH_V0\x00"
RECEIPT=FORMAT+"_RECEIPT"
FIELDS=("format","generation","vault_id","parent_root_sha256","physical_root_sha256",
        "physical_leaf_count","expected_physical_leaves","documents","search_scope")
DOC_FIELDS=("doc_id","revision","physical_leaf","path_hex","visible","byte_length",
            "sha256","leaf_binding_sha256")
PATTERNS=(b"alpha LEFT",b"ONLY_IN_V1",b"dup",b"not-present",b"shared",b"\x00\xff")

def check(ok,why):
    if not ok:raise ValueError(why)
def canonical(x):
    return (json.dumps(x,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False)+"\n").encode()
def sha(x):return hashlib.sha256(x).hexdigest()
def pairs(x):
    out={}
    for a,b in x:
        check(a not in out,"duplicate JSON field")
        out[a]=b
    return out
def parse(raw):
    v=json.loads(raw.decode("ascii"),object_pairs_hook=pairs,
                 parse_constant=lambda _:(_ for _ in ()).throw(ValueError("nonfinite")))
    check(type(v) is dict and canonical(v)==raw,"noncanonical JSON")
    return v
def read(path,maxsize=1<<20):
    path=Path(path)
    check(path.is_file() and not path.is_symlink() and
          0<path.stat().st_size<=maxsize,"missing or nonregular required file")
    return path.read_bytes()
def hashfile(path):
    h=hashlib.sha256()
    with Path(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<18),b""):h.update(b)
    return h.hexdigest()
def roots_validate(bridge,pins,physical):
    check(type(pins) is list and len(pins)==3,"expected pin set")
    for i,root in enumerate(bridge):
        check(set(root)==set(FIELDS) and root["format"]==FORMAT
              and root["generation"]==i and root["vault_id"]=="8f"*16,
              "wrong bridge profile/generation/vault")
        check(sha(DOMAIN+canonical(root))==pins[i],"untrusted bridge root pin")
        check(root["parent_root_sha256"]==(None if i==0 else pins[i-1]),"broken root lineage")
        phys=physical[0] if i==0 else physical[1]
        check(root["physical_root_sha256"]==phys["pin"] and
              root["physical_leaf_count"]==len(phys["root"]["leaves"]) and
              root["expected_physical_leaves"]==
              [x["binding"]["sha256"] for x in phys["root"]["leaves"]],
              "physical incomplete coverage/identity mismatch")
        docs=root["documents"]
        check(type(docs) is list and len(docs)==(9 if i==0 else 11)
              and root["search_scope"]=="VISIBLE_REVISION_IN_PINNED_SNAPSHOT",
              "wrong logical snapshot shape/scope")
        names=[]
        for j,d in enumerate(docs):
            check(type(d) is dict and set(d)==set(DOC_FIELDS)
                  and type(d["doc_id"]) is int and d["doc_id"]==j
                  and type(d["revision"]) is int and d["revision"] in (0,1)
                  and type(d["visible"]) is bool,"document identity/revision")
            p=bytes.fromhex(d["path_hex"])
            check(p and p.hex()==d["path_hex"] and len(p)<1024 and
                  not p.startswith(b"/") and b"\0" not in p and b"\\" not in p
                  and all(c not in (b"",b".",b"..") for c in p.split(b"/")),
                  "unsafe/path alias")
            if d["visible"]:names.append(p)
            leaf=d["physical_leaf"]
            check(type(leaf) is int and 0<=leaf<len(phys["root"]["leaves"]),
                  "unbound physical leaf")
            source=phys["root"]["documents"][leaf]
            check(d["sha256"]==source["sha256"] and d["byte_length"]==source["byte_length"]
                  and d["leaf_binding_sha256"]==phys["root"]["leaves"][leaf]["binding"]["sha256"],
                  "cross-layer byte/index identity mismatch")
        check(len(names)==len(set(names)),"duplicate visible path")
    v0,v1,v2=[x["documents"] for x in bridge]
    check(all(v0[i]["doc_id"]==v1[i]["doc_id"] for i in range(9))
          and v0[1]["physical_leaf"]==1 and v0[1]["revision"]==0
          and v1[1]["physical_leaf"]==9 and v1[1]["revision"]==1
          and v0[1]["sha256"]!=v1[1]["sha256"]
          and v0[1]["path_hex"]!=v1[1]["path_hex"],
          "revision/rename missing or old document renumbered")
    check(all(v0[i]==v1[i] for i in range(9) if i!=1),
          "previous logical documents silently changed")
    check(v1[9]["physical_leaf"]==10 and v1[10]["physical_leaf"]==11
          and v1[9]["doc_id"]==9 and v1[10]["doc_id"]==10
          and bytes.fromhex(v1[9]["path_hex"])==b"a-early.bin",
          "out-of-order path append broke stable IDs")
    check(all(v1[i]==v2[i] for i in range(11) if i!=6)
          and v1[6]["visible"] and not v2[6]["visible"]
          and bridge[1]["physical_root_sha256"]==bridge[2]["physical_root_sha256"],
          "metadata-only tombstone rebuilt or rewrote source")
    check([x["binding"]["sha256"] for x in physical[1]["root"]["leaves"][:9]]==
          [x["binding"]["sha256"] for x in physical[0]["root"]["leaves"]],
          "historical physical leaves changed on append")
def scan(source,needle):
    pos=source.find(needle)
    result=[]
    while pos>=0:
        result.append(pos)
        pos=source.find(needle,pos+1)
    return result
def run(cmd,timeout=90):
    x=subprocess.run([str(q) for q in cmd],capture_output=True,text=True,timeout=timeout)
    check(x.returncode==0,"independent materializer/leaf replay refused: "+
          x.stderr[-650:])
    return x
def materialize(bundle,physical,work):
    fromroot=physical[1]["root"]
    found={}
    stages=0
    for leaf,d in enumerate(fromroot["documents"]):
        target=work/(str(leaf)+".raw")
        if d["byte_length"]==0:
            target.write_bytes(b"")
        else:
            directory=bundle/"leaves"/str(leaf)
            lr=directory/"root.json"
            # Root SHA from the independently checked leaf binding in physical root.
            binding=parse(read(directory/"binding.json"))
            assert binding["document"]==d
            run([sys.executable,REPO/"experiments/loc_text_ef_v0/trusted_materialize_v0.py",
                 "--manifest",lr,"--root-sha256",binding["index_root"]["sha256"],
                 "--rlb2",directory/"rlb2","--rlr2",directory/"rlr2",
                 "--ef",directory/"ef","--decoder",bundle/"decoder",
                 "--output",target,"--threads","2","--mode","rank"],180)
            stages+=1
        raw=target.read_bytes()
        check(len(raw)==d["byte_length"] and sha(raw)==d["sha256"],
              "reconstructed physical leaf incorrect")
        found[leaf]=raw
    check(len(found)==12 and stages==11,"missing independent restored leaf")
    return found
def verify_query(bridge_root,claim,physical_receipt,needle,limit,sources):
    check(set(claim)=={"format","bridge_root_sha256","physical_root_sha256",
                      "query_hex","max_offsets","match_count","coordinates",
                      "expected_physical_leaves","verified_physical_leaves",
                      "status","reason"},"invalid logical receipt")
    check(claim["format"]==RECEIPT
          and claim["bridge_root_sha256"]==sha(DOMAIN+canonical(bridge_root))
          and claim["physical_root_sha256"]==bridge_root["physical_root_sha256"]
          and claim["query_hex"]==needle.hex()
          and type(limit) is int and claim["max_offsets"]==limit
          and 0<=limit<=256,"query pin/policy mismatch")
    expected=bridge_root["expected_physical_leaves"]
    check(claim["expected_physical_leaves"]==expected
          and claim["verified_physical_leaves"]==expected,
          "false full index coverage")
    check(physical_receipt["coverage_complete"] is True
          and physical_receipt["coverage"]["expected_leaves"]==expected
          and physical_receipt["coverage"]["verified_leaves"]==expected
          and physical_receipt["coverage"]["queried_leaves"]==expected,
          "physical skipped expected index leaf")
    coords=[]
    for doc in bridge_root["documents"]:
        if not doc["visible"]:continue
        for off in scan(sources[doc["physical_leaf"]],needle):
            coords.append([doc["doc_id"],doc["revision"],off])
    coords.sort()
    limited=len(coords)>limit
    status="UNTRUSTED" if limited else "FOUND" if coords else "PROVEN_EMPTY"
    check(type(claim["match_count"]) is int and claim["match_count"]==len(coords)
          and claim["coordinates"]==coords[:limit] and claim["status"]==status
          and claim["reason"]==("LIMITED_COMPLETENESS" if limited else
                                "FULL_INDEXED_SNAPSHOT_REPLAY"),
          "false count / old-revision leak / false empty / incomplete prefix")
    return status,len(coords)
def main():
    a=argparse.ArgumentParser()
    for key in ("bundle","overlay","reference","expected-bridge-pins-json",
                "expected-physical-pins-json","output"):
        a.add_argument("--"+key,required=True)
    p=a.parse_args()
    output=Path(p.output)
    check(not output.exists(),"refuse existing evidence directory")
    output.mkdir()
    result={"status":"BLOCKED"}
    started=time.monotonic()
    try:
        bundle=Path(p.bundle);overlay=Path(p.overlay)
        pins=json.loads(p.expected_bridge_pins_json)
        physpins=json.loads(p.expected_physical_pins_json)
        check(type(physpins) is dict and set(physpins)=={"A0","B0","A1","B1"},
              "physical external pin set")
        phy=[]
        for i in (0,1):
            path=bundle/"roots"/("A%d.json"%i)
            raw=read(path)
            pin=physpins["A%d"%i]
            check(sha(raw)==pin,"externally pinned physical root changed")
            phy.append({"root":parse(raw),"pin":pin})
        bridge=[parse(read(overlay/("V%d.json"%i))) for i in range(3)]
        roots_validate(bridge,pins,phy)
        count=0
        verified_physical={}
        outcomes=[]
        with tempfile.TemporaryDirectory(prefix="glyph-l11-indexed-restore-") as d:
            staging=Path(d)
            sources=materialize(bundle,phy,staging)
            for i,root in enumerate(bridge):
                for j,needle in enumerate(PATTERNS):
                    key="V%d-Q%d"%(i,j)
                    claim=parse(read(overlay/"receipts"/(key+".json")))
                    full_path=overlay/"receipts"/(key+".physical.json")
                    physical_receipt=parse(read(full_path,1<<20))
                    physical_key=(root["physical_root_sha256"],needle.hex())
                    fingerprint=sha(canonical(physical_receipt))
                    if physical_key in verified_physical:
                        check(verified_physical[physical_key]==fingerprint,
                              "same physical root and query returned changing receipt")
                    if physical_key not in verified_physical:
                        # Invoke unchanged standalone Composition V1-aware research
                        # verifier, which independently replays each original L9
                        # leaf receipt and verifies physical rank/EF search.
                        rootpath=bundle/"roots"/("A0.json" if i==0 else "A1.json")
                        command=[sys.executable,"-I",
                            REPO/"experiments/core_l10_composition_v0/verifier.py",
                            "--bundle",bundle,"--root",rootpath,
                            "--pin",root["physical_root_sha256"],
                            "--generation","0" if i==0 else "1",
                            "--oracles",staging,"--receipt",full_path,
                            "--query-hex",needle.hex(),"--limit","256",
                            "--research",REPO,"--reference",p.reference]
                        if i:
                            old=bundle/"roots/A0.json"
                            command+=["--previous-root",old,
                                      "--previous-pin",physpins["A0"]]
                        info=run(command,240)
                        checked=json.loads(info.stdout)
                        check(checked["full_leaf_block_coverage"] is True
                              and checked["count_verified"] is True,
                              "physical independent search not checked")
                        verified_physical[physical_key]=fingerprint
                    status,num=verify_query(root,claim,physical_receipt,
                                            needle,2 if j==2 else 256,sources)
                    count+=1
                    outcomes.append({"generation":i,"query_hex":needle.hex(),
                                     "count":num,"status":status})
            # Required cross-version semantics on the reconstructed actual bytes.
            find=lambda g,needle:next(x for x in outcomes if
                x["generation"]==g and x["query_hex"]==needle.hex())
            check(find(0,b"alpha LEFT")["status"]=="FOUND"
                  and find(1,b"alpha LEFT")["status"]=="PROVEN_EMPTY"
                  and find(0,b"ONLY_IN_V1")["status"]=="PROVEN_EMPTY"
                  and find(1,b"ONLY_IN_V1")["status"]=="FOUND"
                  and find(2,b"ONLY_IN_V1")["status"]=="FOUND",
                  "new revision or old version search was not isolated")
            check(find(0,b"dup")["count"]==2 and
                  find(1,b"dup")["count"]==3 and
                  find(2,b"dup")["count"]==2,
                  "duplicated document/tombstone semantics violated")
            # Negative tests run against the very same independent replay
            # inputs in memory, without reopening a source-backed producer.
            rejected=[]
            def should_reject(name,func):
                try:func()
                except (ValueError,TypeError,KeyError,IndexError,AssertionError):
                    rejected.append(name)
                    return
                raise AssertionError("FAIL OPEN negative test "+name)
            logical=parse(read(overlay/"receipts/V1-Q1.json"))
            physical_receipt=parse(read(overlay/"receipts/V1-Q1.physical.json"))
            claimroot=bridge[1]
            def changed(fn):
                x=copy.deepcopy(logical)
                fn(x)
                return verify_query(claimroot,x,physical_receipt,
                                    b"ONLY_IN_V1",256,sources)
            should_reject("false-proven-empty",lambda:changed(
                lambda x:x.update(status="PROVEN_EMPTY",match_count=0,coordinates=[])))
            should_reject("wrong-count",lambda:changed(
                lambda x:x.update(match_count=999)))
            should_reject("missing-offset",lambda:changed(
                lambda x:x["coordinates"].clear()))
            should_reject("fake-revision",lambda:changed(
                lambda x:x["coordinates"][0].__setitem__(1,0)))
            should_reject("fake-doc-id",lambda:changed(
                lambda x:x["coordinates"][0].__setitem__(0,9)))
            should_reject("fake-offset",lambda:changed(
                lambda x:x["coordinates"][0].__setitem__(2,11)))
            should_reject("missing-physical-leaf",lambda:changed(
                lambda x:x["verified_physical_leaves"].pop()))
            should_reject("forged-physical-root",lambda:changed(
                lambda x:x.update(physical_root_sha256="0"*64)))
            should_reject("forged-overlay-root",lambda:changed(
                lambda x:x.update(bridge_root_sha256="0"*64)))
            should_reject("false-limited-policy",lambda:changed(
                lambda x:x.update(status="UNTRUSTED",reason="LIMITED_COMPLETENESS")))
            should_reject("unknown-receipt-field",lambda:changed(
                lambda x:x.update(secret="bad")))
            # Re-pin the adversarial bridge root: a new digest must not
            # legitimize an unbound/renumbered/rewritten version history.
            def invalid_bridge(name,mutate):
                xs=copy.deepcopy(bridge)
                nextpins=list(pins)
                mutate(xs[2])
                nextpins[2]=sha(DOMAIN+canonical(xs[2]))
                should_reject(name,lambda:roots_validate(xs,nextpins,phy))
            invalid_bridge("repinned-unbound-leaf",
                           lambda x:x["documents"][0].update(physical_leaf=999))
            invalid_bridge("repinned-old-doc-renumber",
                           lambda x:x["documents"][0].update(doc_id=42))
            invalid_bridge("repinned-physical-coverage-short",
                           lambda x:x["expected_physical_leaves"].pop())
            invalid_bridge("repinned-invalid-parent",
                           lambda x:x.update(parent_root_sha256="0"*64))
            invalid_bridge("repinned-mutated-old-document",
                           lambda x:x["documents"][1].update(revision=0))
            invalid_bridge("repinned-path-traversal",
                           lambda x:x["documents"][1].update(path_hex=b"../escape".hex()))
            invalid_bridge("repinned-undeclared-tombstone",
                           lambda x:x["documents"][5].update(visible=False))
            check(len(rejected)==18,"negative-case count drift")
            saved=[{"physical_leaf":n,"source_bytes":len(v),"sha256":sha(v)}
                   for n,v in sources.items()]
        # Do not retain reconstructed full sources in the receipt/capsule.
        result={"status":"CORE_L11_INDEXED_IDENTITY_BRIDGE_INDEPENDENT_PASS",
                "bridge_snapshot_count":3,"logical_replay_count":count,
                "adversarial_cases_rejected":len(rejected),
                "hostile_labels":rejected,
                "independent_physical_query_replays":len(verified_physical),
                "physical_indexes_restored":len(saved),
                "recovered_source_sha256":saved,
                "v0_v1_parent_integrity":True,"old_physical_leaves_unchanged":True,
                "renames_do_not_reindex":True,"revision_adds_new_leaf":True,
                "tombstone_metadata_only":True,
                "all_physical_leaves_covered_even_when_historical":True,
                "no_original_raw_source_retained":True,
                "source_scan_is_full_not_succinct":True,
                "external_roots_are_unsigned":True,
                "wall_seconds":round(time.monotonic()-started,5),
                "peak_python_rss_kib":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                "outcomes":outcomes}
    except Exception as exc:
        result={"status":"BLOCKED","error":type(exc).__name__+": "+str(exc)[:400]}
    (output/"result.json").write_bytes(canonical(result))
    print(json.dumps({k:result.get(k) for k in
        ("status","error","bridge_snapshot_count","logical_replay_count",
         "independent_physical_query_replays","physical_indexes_restored")},sort_keys=True))
    return 0 if result["status"]=="CORE_L11_INDEXED_IDENTITY_BRIDGE_INDEPENDENT_PASS" else 3

if __name__=="__main__":
    sys.exit(main())
