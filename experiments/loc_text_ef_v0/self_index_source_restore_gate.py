#!/usr/bin/env python3
"""Can the actual frozen RLB2/RLR2 recover source bytes without source/SA?

A scoped SOURCE-RESTORE research probe, not an efficient 100MB inverse-BWT
implementation, independently signed root, or portable end-user proof bundle.
"""
import argparse
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


def must(ok,message):
    if not ok: raise AssertionError(message)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    must(spec and spec.loader,"cannot load immutable fixture generator")
    obj=importlib.util.module_from_spec(spec);sys.modules[name]=obj
    spec.loader.exec_module(obj)
    return obj


def restore_from_rlb2_rank(rlb2,rlr2,query_module):
    # This function does not receive the corpus, SA, BWT build artifact, LOC1,
    # LOC2 or LOC-TEXT-EF. The indexed BWT and rank suffice for full inversion.
    rank=query_module.RankV2(rlb2,rlr2)
    try:
        n=rank.raw_length-1
        out=bytearray(n)
        row=0 # canonical suffix row for terminal suffix, verified at build gate
        for i in range(n-1,-1,-1):
            symbol,rank_before=rank.symbol_and_rank(row)
            must(0<=symbol<256,"premature terminal symbol in inverse BWT")
            out[i]=symbol
            row=rank.C[symbol]+rank_before
        terminal,rank_before=rank.symbol_and_rank(row)
        must(terminal==256 and rank.C[terminal]+rank_before==0,
             "inversion did not close on sole logical sentinel")
        return bytes(out),rank.lf_calls,rank.scanned_symbols
    finally:rank.close()


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--out",type=Path,required=True)
    opt=p.parse_args()
    reference=opt.reference.resolve()
    must(subprocess.check_output(["git","-C",str(reference),"rev-parse","HEAD"],
                                 text=True).strip()=="c33cb493bc75f13ee4a904fabb190a29a196c94b",
         "wrong original source")
    root=opt.out.resolve()
    must(not root.exists(),"refuse overwrite")
    root.mkdir(parents=True)
    tools=reference/"tools"
    sys.path.insert(0,str(tools))
    import rlbwt_query_v2 as q
    fixture_module=load(tools/"check_rlbwt_query_v2.py","original_binary_safe_fixture")
    fixture=fixture_module.build_fixture(root,
        tools/"rlbwt_container_v2.py",tools/"rlbwt_rank_v2.py")
    expected=fixture["source"]
    restored,lf_calls,scanned=restore_from_rlb2_rank(
        fixture["rlb2"],fixture["rlr2"],q)
    must(restored==expected,"inverse BWT did not restore original bytes")
    original_sha=hashlib.sha256(expected).hexdigest()
    must(hashlib.sha256(restored).hexdigest()==original_sha,"source SHA mismatch")
    bad_rlb2=root/"bad.rlb2"
    bad_rlb2.write_bytes(fixture["rlb2"].read_bytes())
    with bad_rlb2.open("r+b") as f:
        f.seek(-1,2);orig=f.read(1);f.seek(-1,2);f.write(bytes([orig[0]^1]))
    denied=False
    try:
        restore_from_rlb2_rank(bad_rlb2,fixture["rlr2"],q)
    except Exception:
        denied=True
    must(denied,"modified compressed payload accepted with original RLR2")
    result={
        "schema":"GLYPH_SOURCE_RESTORE_FROM_RLB2_RLR2_SMALL_V0",
        "status":"BINARY_SAFE_SELF_INDEX_SOURCE_RESTORE_PASS_SMALL_ONLY",
        "source_bytes":len(expected),
        "source_sha256":original_sha,
        "bytes_verified":len(restored),
        "no_original_source_or_construction_sa_given_to_restore":True,
        "uses_only_rlb2_rlr2":True,
        "corrupted_rlb2_with_old_rank_rejected":True,
        "scope_limits":[
            "Only frozen 24-byte binary-safe fixture, not 100MB or all corpora",
            "Python full inversion repeatedly queries rank, no acceptable large-corpus latency proved",
            "No new source extraction API, query evidence O3/O4, currentness or committed root",
            "No proof that LOC-TEXT-EF can independently restore source (restoration uses RLB2/RLR2)"
        ]
    }
    (root/"result.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
