#!/usr/bin/env python3
"""CORE-L10: append-only multi-document RLB2/EF research evidence and hostiles.

This gate does NOT advertise historical GLYPH_COMPOSITION_ROOT_V1 compatibility:
repartition refers to grouping leaf references, not true V1 operator units.
"""
import argparse
import copy
import hashlib
import json
import resource
import subprocess
import sys
import time
from pathlib import Path

import producer as model

PASS="CORE_L10_APPEND_RESEARCH_EXACT_AND_INDEPENDENT_REPLAY_PASS"

def must(x,msg):
    if not x:raise AssertionError(msg)

def launch_verify(verifier,frozen,tree,root_meta,result_path,parent=None):
    cmd=[sys.executable,str(verifier),"--tree",str(tree),
         "--root",str(root_meta["path"]),"--root-sha256",root_meta["sha256"],
         "--result",str(result_path),"--frozen",str(frozen)]
    if parent:
        cmd.extend(["--parent",str(parent["path"]),
                    "--parent-sha256",parent["sha256"]])
    return subprocess.run(cmd,capture_output=True,text=True,timeout=110)

def saved(tree,name,obj):
    path=tree/(name+".json")
    path.write_bytes(model.canonical(obj))
    return path

def main():
    a=argparse.ArgumentParser()
    a.add_argument("--frozen",type=Path,required=True)
    a.add_argument("--research",type=Path,required=True)
    a.add_argument("--output",type=Path,required=True)
    args=a.parse_args()
    frozen=args.frozen.resolve()
    research=args.research.resolve()
    output=args.output.resolve()
    verifier=research/"experiments/core_l10_composition_v0/verifier.py"
    must(verifier.is_file(),"missing independent verifier")
    report={"schema":"GLYPH_CORE_L10_EXECUTED_TEST_V0","status":"BLOCKED",
            "composition_reference_commit":"81628f843dc8aa633c65594a66d2386bfc3cb5c2",
            "frozen_ref":model.FROZEN,
            "l9s_starting_commit":"e7f0e3b16381519c8b0269f6fea0ff11feb0cc80"}
    root_meta={}
    start=time.perf_counter()
    try:
        old=model.create_all(output,frozen)
        must(len(old)==9,"wrong V0 document count")
        v0a=model.make_root(old,"V0",model.PARTITIONS[0],None)
        v0b=model.make_root(old,"V0",model.PARTITIONS[1],None)
        root_meta["V0_A"]=model.store_root(output,v0a,"V0_A")
        root_meta["V0_B"]=model.store_root(output,v0b,"V0_B")
        must(v0a["runtime_corpus_id"]==v0b["runtime_corpus_id"] and
             v0a["source_manifest_id"]==v0b["source_manifest_id"] and
             v0a["research_snapshot_id"]==v0b["research_snapshot_id"] and
             v0a["research_layout_id"]!=v0b["research_layout_id"],
             "V0 repartition semantics violated")
        # Capture initial immutable bytes *before* appending new leaves.
        pinned_old={
            path.relative_to(output).as_posix():(path.stat().st_size,model.hfile(path))
            for path in (output/"leaves").rglob("*") if path.is_file()
        }
        # Execute and verify V0's COMPLETE and ABSENT results before V1 exists.
        cases=[]
        case_count=0
        for name in ("V0_A","V0_B"):
            root={"V0_A":v0a,"V0_B":v0b}[name]
            meta=root_meta[name]
            for pattern,cap in ((b"banana",None),(b"V1_NEEDLE",None),
                                (b"XY",None),(b"banana",0),(b"banana",1),
                                (b"banana",2),(b"\xff\x00",None),(b"\x00",None)):
                t0=time.perf_counter()
                receipt=model.produce_query(output,root,meta["sha256"],pattern,cap,frozen)
                fname="%s_%d"%(name,len(cases))
                path=saved(output,"result-"+fname,receipt)
                checked=launch_verify(verifier,frozen,output,meta,path)
                must(checked.returncode==0,
                     "independent V0 replay mismatch: "+checked.stderr[-600:])
                cases.append({"root":name,"pattern_hex":pattern.hex(),"max_offsets":cap,
                              "status":receipt["status"],"match_count":receipt["match_count"],
                              "coordinates":receipt["coordinates"],
                              "producer_plus_replay_wall_seconds":round(time.perf_counter()-t0,6),
                              "receipt_bytes":path.stat().st_size})
                case_count+=1
        # Second chronological phase: construct only three NEW leaf indexes.
        docs,original_inside=model.append_three(output,frozen,old)
        must(original_inside==pinned_old,"V0 leaf bytes changed during append")
        must(len(docs)==12 and docs[:9]==old,"old document identities not stable")
        v1a=model.make_root(docs,"V1",model.NEW_PARTITIONS[0],
                            root_meta["V0_A"]["sha256"])
        v1b=model.make_root(docs,"V1",model.NEW_PARTITIONS[1],
                            root_meta["V0_B"]["sha256"])
        root_meta["V1_A"]=model.store_root(output,v1a,"V1_A")
        root_meta["V1_B"]=model.store_root(output,v1b,"V1_B")
        must(v1a["runtime_corpus_id"]==v1b["runtime_corpus_id"] and
             v1a["source_manifest_id"]==v1b["source_manifest_id"] and
             v1a["research_snapshot_id"]==v1b["research_snapshot_id"] and
             v1a["research_layout_id"]!=v1b["research_layout_id"] and
             v1a["runtime_corpus_id"]!=v0a["runtime_corpus_id"] and
             v1a["source_manifest_id"]!=v0a["source_manifest_id"],
             "append/repartition identity mismatch")
        for name in ("V1_A","V1_B"):
            root={"V1_A":v1a,"V1_B":v1b}[name]
            meta=root_meta[name]
            parent=root_meta["V0_A" if name=="V1_A" else "V0_B"]
            for pattern,cap in ((b"banana",None),(b"V1_NEEDLE",None),
                                (b"XY",None),(b"banana",0),(b"banana",1),
                                (b"banana",2),(b"\xff\x00",None),(b"\x00",None),
                                (b"Banana",None),(b"not-present",None)):
                t0=time.perf_counter()
                receipt=model.produce_query(output,root,meta["sha256"],pattern,cap,frozen)
                path=saved(output,"result-%s_%d"%(name,len(cases)),receipt)
                checked=launch_verify(verifier,frozen,output,meta,path,parent)
                must(checked.returncode==0,
                     "independent V1 replay mismatch: "+checked.stderr[-600:])
                cases.append({"root":name,"pattern_hex":pattern.hex(),"max_offsets":cap,
                              "status":receipt["status"],"match_count":receipt["match_count"],
                              "coordinates":receipt["coordinates"],
                              "producer_plus_replay_wall_seconds":round(time.perf_counter()-t0,6),
                              "receipt_bytes":path.stat().st_size})
                case_count+=1
        # Check all paired semantic outcomes across physical groupings.
        for version in ("V0","V1"):
            for prefix in (b"banana",b"V1_NEEDLE",b"XY"):
                lhs=[x for x in cases if x["root"]==version+"_A" and
                     x["pattern_hex"]==prefix.hex() and x["max_offsets"] is None][0]
                rhs=[x for x in cases if x["root"]==version+"_B" and
                     x["pattern_hex"]==prefix.hex() and x["max_offsets"] is None][0]
                must((lhs["status"],lhs["match_count"],lhs["coordinates"])==
                     (rhs["status"],rhs["match_count"],rhs["coordinates"]),
                     "layout changed canonical query semantics")
        absent_v0=next(x for x in cases if x["root"]=="V0_A" and
                       x["pattern_hex"]==b"V1_NEEDLE".hex())
        found_v1=next(x for x in cases if x["root"]=="V1_A" and
                      x["pattern_hex"]==b"V1_NEEDLE".hex())
        must(absent_v0["status"]=="PROVEN_EMPTY" and absent_v0["match_count"]==0
             and found_v1["status"]=="FOUND" and found_v1["match_count"]==1
             and found_v1["coordinates"]==[[9,0]],
             "append version-specific absent/present guarantee violated")
        old_positive=next(x for x in cases if x["root"]=="V0_A" and
                          x["pattern_hex"]==b"banana".hex() and
                          x["max_offsets"] is None)
        new_positive=next(x for x in cases if x["root"]=="V1_A" and
                          x["pattern_hex"]==b"banana".hex() and
                          x["max_offsets"] is None)
        must(old_positive["coordinates"]==new_positive["coordinates"][:len(old_positive["coordinates"])],
             "appending changed prior canonical doc coordinates")
        # Independent negative testing with dedicated verifier subprocess.
        baseline=model.produce_query(output,v1a,root_meta["V1_A"]["sha256"],
                                     b"banana",None,frozen)
        parent=root_meta["V0_A"]
        negatives=[]
        def reject(label,rootmeta=root_meta["V1_A"],receipt=None,par=parent):
            candidate=copy.deepcopy(baseline if receipt is None else receipt)
            path=saved(output,"hostile-%02d"%len(negatives),candidate)
            process=launch_verify(verifier,frozen,output,rootmeta,path,par)
            must(process.returncode==3 and "UNTRUSTED" in process.stderr,
                 "fail-open hostile "+label+" stdout="+process.stdout[-250:]+
                 " stderr="+process.stderr[-350:])
            negatives.append(label)
        # Each mutation is named and independently executed; never just
        # asserts that the object differs without invoking verifier.
        def mutate(label,fn):
            changed=copy.deepcopy(baseline)
            fn(changed)
            reject(label,receipt=changed)
        mutate("false-proven-empty",lambda x:x.update(
            status="PROVEN_EMPTY",match_count=0,returned_count=0,
            coordinates=[],excerpts=[],offsets_complete=True))
        mutate("count-off-by-one",lambda x:x.update(match_count=x["match_count"]+1))
        mutate("remove-first-coordinate",lambda x:x["coordinates"].pop(0))
        mutate("duplicate-coordinate",lambda x:x["coordinates"].append(x["coordinates"][0]))
        mutate("reverse-coordinates",lambda x:x["coordinates"].reverse())
        mutate("fabricated-global-doc-id",lambda x:x["coordinates"][0].__setitem__(0,99))
        mutate("forged-doc-offset",lambda x:x["coordinates"][0].__setitem__(1,999))
        mutate("wrong-query-pattern",lambda x:x.update(pattern_hex=b"bandana".hex()))
        mutate("wrong-query-length",lambda x:x.update(pattern_bytes=999))
        mutate("missing-required-group",lambda x:x["queried_blocks"].pop())
        mutate("forged-verified-group",lambda x:x["verified_blocks"].__setitem__(0,7))
        mutate("missing-expected-group",lambda x:x["expected_blocks"].pop())
        mutate("bad-root-pin-in-receipt",lambda x:x.update(root_sha256="0"*64))
        mutate("wrong-snapshot-identity",lambda x:x.update(research_snapshot_id="0"*64))
        mutate("wrong-research-layout",lambda x:x.update(research_layout_id="0"*64))
        mutate("wrong-corpus-identity",lambda x:x.update(runtime_corpus_id="0"*64))
        mutate("wrong-source-manifest",lambda x:x.update(source_manifest_id="0"*64))
        mutate("wrong-version-claim",lambda x:x.update(version="V0"))
        mutate("bad-excerpt-sha",lambda x:x["excerpts"][0].update(sha256="0"*64))
        mutate("bad-excerpt-coordinate",lambda x:x["excerpts"][0].update(doc_id=55))
        mutate("bad-excerpt-byte-length",lambda x:x["excerpts"][0].update(bytes=999))
        mutate("forged-full-completeness",lambda x:x.update(offsets_complete=False))
        mutate("unknown-result-field",lambda x:x.update(secret="unexpected"))
        mutate("wrong-limited-claim",lambda x:x.update(status="UNTRUSTED",reason="LIMITED_COMPLETENESS"))
        mutate("wrong-result-count",lambda x:x.update(returned_count=0))
        # Root physical mutation, with and without new physical root pins.
        def root_reject(label,fn,version="V1"):
            m=copy.deepcopy(v1a)
            fn(m)
            meta=model.store_root(output,m,"hostile-root-%02d"%len(negatives))
            reject(label,rootmeta=meta)
        root_reject("missing-leaf-in-root",lambda x:x["documents"].pop())
        root_reject("missing-group-from-root",lambda x:x["groups"].pop())
        root_reject("duplicate-group-doc-id",lambda x:x["groups"][0].append(0))
        root_reject("reordered-groups",lambda x:x["groups"].reverse())
        root_reject("empty-document-removed",lambda x:x["documents"].pop(1))
        root_reject("duplicate-doc-id",lambda x:x["documents"][2].update(doc_id=1))
        root_reject("reordered-path",lambda x:x["documents"][0].update(
            path_hex=b"vault/00000010.bin".hex()))
        root_reject("forged-old-root-link",lambda x:x.update(parent_root_sha256="0"*64))
        root_reject("forged-new-version",lambda x:x.update(version="V0"))
        root_reject("altered-old-leaf-commitment",lambda x:x["documents"][0].update(sha256="0"*64))
        # Missing physical leaf is UNTRUSTED, not an absence claim.
        target=output/v1a["documents"][0]["artifacts"]["rlr2"]["path"]
        saved_bytes=target.read_bytes()
        target.unlink()
        try:
            reject("mandatory-rank-index-missing")
        finally:
            target.write_bytes(saved_bytes)
        must(model.hfile(target)==v1a["documents"][0]["artifacts"]["rlr2"]["sha256"],
             "negative test failed to restore rank data")
        # Re-pinned, physically invalid EF checkpoint: PROVEN_EMPTY must be
        # rejected by producer before any source-empty success.
        rec=v1a["documents"][0]
        ef=output/rec["artifacts"]["ef"]["path"]
        raw=ef.read_bytes()
        corrupted=bytearray(raw)
        corrupted[-1]^=1
        ef.write_bytes(corrupted)
        try:
            root_bad=copy.deepcopy(v1a)
            root_bad["documents"][0]["artifacts"]["ef"]["sha256"]=model.hfile(ef)
            root_bad["research_layout_id"]=model.h(
                model.LAYOUT_DOMAIN+model.canonical({
                    "runtime_corpus_id":root_bad["runtime_corpus_id"],
                    "source_manifest_id":root_bad["source_manifest_id"],
                    "groups":root_bad["groups"],
                    "leaf_artifacts":[d["artifacts"] for d in root_bad["documents"]]}))
            pin=model.h(model.canonical(root_bad))
            failed=False
            try:
                model.produce_query(output,root_bad,pin,b"not-present",None,frozen)
            except Exception:
                failed=True
            must(failed,"re-pinned malformed EF checkpoint accepted as PROVEN_EMPTY")
            negatives.append("repinned-EF-checkpoint-producer-rejected")
        finally:
            ef.write_bytes(raw)
        must(model.hfile(ef)==rec["artifacts"]["ef"]["sha256"],
             "failed to recover canonical EF after hostile test")
        # Archive all evidence without hiding negative invocation failures.
        storage=sum(p.stat().st_size for p in (output/"leaves").rglob("*") if p.is_file())
        old_storage=sum(x[0] for x in pinned_old.values())
        report.update({
            "status":PASS,
            "v0_document_count":9,"v1_document_count":12,
            "v0_nonempty_leaves":8,"v1_nonempty_leaves":11,
            "physical_old_leaf_files_untouched":len(pinned_old),
            "physical_old_leaf_bytes_untouched":old_storage,
            "retained_index_bytes_across_12_documents":storage,
            "roots":{k:{j:str(v) for j,v in meta.items()
                        if j not in ("path",)}
                     for k,meta in root_meta.items()},
            "normative_runtime_corpus_id_repartition_equal":True,
            "normative_source_manifest_id_repartition_equal":True,
            "research_layout_id_repartition_differs":True,
            "global_document_ids_stable_on_append":True,
            "version_scoped_proven_empty_transitions_to_found":True,
            "independent_full_coverage_queries_replayed":case_count,
            "hostile_cases_rejected":negatives,
            "hostile_case_count":len(negatives),
            "total_wall_seconds":round(time.perf_counter()-start,6),
            "python_process_peak_rss_kb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "query_cases":cases,
            "limitations":[
                "Historic Composition V1 operator runtime is not byte compatible with new single-doc RLB2/EF leaves.",
                "research_layout_id is NOT GLYPH_COMPOSITION_ROOT_V1 or actual Index Forest.",
                "One nonempty document per physical RLB2 leaf; empty doc uses explicit SHA-verified manifest record.",
                "Source-order path IDs are stable only for monotonic canonical append namespace.",
                "Unsigned root pins assumed externally authentic; no rollback/fork/currentness witness.",
                "Independent verifier uses small O(n^2)-constructed suffix array and full RLB2 BWT inversion for each doc.",
                "No full 100MB multi-document index/composition, no 1TB/64-bit claim, no signed O3/O4 proof.",
                "Receipt/caps are fail-closed, not a complete independent succinct nonmembership proof."
            ]
        })
    except Exception as e:
        report.update(status="BLOCKED",error=str(e))
    finally:
        # Result file survives failed gates; do not emit false PASS.
        if output.exists():
            (output/"result.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report.get(k) for k in (
        "status","error","v0_document_count","v1_document_count",
        "retained_index_bytes_across_12_documents",
        "independent_full_coverage_queries_replayed","hostile_case_count",
        "physical_old_leaf_files_untouched")},sort_keys=True))
    return 0 if report["status"]==PASS else 3

if __name__=="__main__":
    sys.exit(main())
