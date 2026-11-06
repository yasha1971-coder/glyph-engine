#!/usr/bin/env python3
"""Native RLB2 inverse-BWT smoke and hostile checks on original frozen fixture."""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

REF="c33cb493bc75f13ee4a904fabb190a29a196c94b"

def check(ok, msg):
    if not ok: raise AssertionError(msg)

def module(path):
    spec=importlib.util.spec_from_file_location("pinned_binary_fixture", path)
    check(spec and spec.loader,"cannot load fixture tool")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=mod
    spec.loader.exec_module(mod)
    return mod

def process(binary,rlb2,target):
    return subprocess.run([str(binary),str(rlb2),str(target)],capture_output=True,text=True,timeout=120)

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--binary",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    ref=a.reference.resolve()
    check(subprocess.check_output(["git","-C",str(ref),"rev-parse","HEAD"],
        text=True).strip()==REF,"wrong pinned historical source")
    root=a.output.resolve()
    check(not root.exists(),"test output must be new")
    root.mkdir(parents=True)
    tools=ref/"tools"
    fixture=module(tools/"check_rlbwt_query_v2.py").build_fixture(
        root,tools/"rlbwt_container_v2.py",tools/"rlbwt_rank_v2.py")
    r=process(a.binary,fixture["rlb2"],root/"restored.bin")
    check(r.returncode==0,f"native decoder failed: {r.stderr}")
    actual=(root/"restored.bin").read_bytes()
    expected=fixture["source"]
    check(actual==expected,"native inversion byte mismatch")
    metrics=json.loads(r.stdout)
    check(metrics["source_bytes"]==len(expected),"native size not the same")
    mutations={}
    source=fixture["rlb2"].read_bytes()
    for name,data in (
        ("payload_bitflip",source[:-1]+bytes([source[-1]^1])),
        ("header_magic",bytes([source[0]^1])+source[1:]),
        ("truncated",source[:-1]),
    ):
        bad=root/f"bad-{name}.rlb2"
        output=root/f"bad-{name}.raw"
        bad.write_bytes(data)
        failed=process(a.binary,bad,output)
        check(failed.returncode!=0 and not output.exists(),
              f"hostile case incorrectly accepted: {name}")
        mutations[name]="REJECTED"
    result={
        "status":"NATIVE_SELF_INDEX_INVERSE_SMALL_EXACT_AND_HOSTILE_PASS",
        "historical_commit":REF,
        "source_bytes":len(expected),
        "source_sha256":hashlib.sha256(actual).hexdigest(),
        "native_decoder_metrics":metrics,
        "hostile":mutations,
        "limitations":["Small 24B binary fixture only, not 100MB enwik8",
                       "Original RLR2 is built but actual native inverse needs RLB2 only",
                       "Decoder has transient O(n) 32-bit LF map and output RAM",
                       "No production P/R/O/currentness/root pin or multi-document coordinates"]
    }
    (root/"result.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))
if __name__=="__main__":
    main()
