#!/usr/bin/env python3
"""CORE-L8: exact 24-byte pinned-root materialization + negative trust tests.

The source comes only from the frozen fixture. The decoder is passed private
RLB2+EF snapshots, not raw source/SA/BWT, and its candidate bytes are hashed
before public materialization. The root digest is test supplied externally;
this is NOT a signed trusted root or production/currentness proof.
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
from pathlib import Path

REF="c33cb493bc75f13ee4a904fabb190a29a196c94b"

def require(ok,why):
    if not ok:raise AssertionError(why)

def sha(data):
    return hashlib.sha256(data).hexdigest()

def sha_file(path):
    return sha(path.read_bytes())

def source_tool(p):
    spec=importlib.util.spec_from_file_location("original_frozen_fixture",p)
    require(spec and spec.loader,"missing frozen fixture")
    m=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=m
    spec.loader.exec_module(m)
    return m

def create_root(folder,manifest,seq):
    path=folder/f"root-{seq}.json"
    raw=(json.dumps(manifest,sort_keys=True,separators=(",",":"))+"\n").encode()
    path.write_bytes(raw)
    return path,sha(raw)

def invoke(tool,root,pin,fixture,ef,decoder,out):
    command=[sys.executable,str(tool),
             "--manifest",str(root),
             "--root-sha256",pin,
             "--rlb2",str(fixture["rlb2"]),
             "--rlr2",str(fixture["rlr2"]),
             "--ef",str(ef),
             "--decoder",str(decoder),
             "--output",str(out),
             "--threads","2","--mode","rank"]
    return subprocess.run(command,capture_output=True,text=True,timeout=90)

def main():
    p=argparse.ArgumentParser()
    for key in ("reference","decoder","verifier","output"):
        p.add_argument("--"+key,type=Path,required=True)
    a=p.parse_args()
    ref=a.reference.resolve()
    target=a.output.resolve()
    require(subprocess.check_output(["git","-C",str(ref),"rev-parse","HEAD"],
            text=True).strip()==REF,"wrong frozen source checkout")
    require(not target.exists(),"output directory must be new")
    target.mkdir()
    tools=ref/"tools"
    sys.path.insert(0,str(Path(__file__).parent))
    from real_enwik8_ef_physical_gate import build,verify
    fixture=source_tool(tools/"check_rlbwt_query_v2.py").build_fixture(
        target,tools/"rlbwt_container_v2.py",tools/"rlbwt_rank_v2.py")
    sa=target/"temporary-construction.sa"
    originals=fixture["suffix_array"]
    sa.write_bytes(b"\0"*60+struct.pack("<"+"I"*len(originals),*originals))
    ef=target/"physical.ef"
    geo=build(sa,len(originals),fixture["sample_step"],ef)
    observed=verify(sa,ef,geo)
    require(observed["sampled_sa_records_verified"]>0,"EF verify failed")
    sa.unlink()
    manifest={
        "schema":"GLYPH_CORE_L8_PINNED_SOURCE_MATERIALIZATION_V0",
        "frozen_commit":REF,
        "source":{"bytes":len(fixture["source"]),
                  "sha256":sha(fixture["source"])},
        "artifacts":{name:{"bytes":path.stat().st_size,
                           "sha256":sha_file(path)} for name,path in [
                  ("rlb2",fixture["rlb2"]),("rlr2",fixture["rlr2"]),("ef",ef)]},
        "decoder":{"sha256":sha_file(a.decoder)}
    }
    root,pin=create_root(target,manifest,0)
    output=target/"verified.materialized"
    success=invoke(a.verifier,root,pin,fixture,ef,a.decoder,output)
    require(success.returncode==0,f"correct root was rejected: {success.stderr}")
    receipt=json.loads(success.stdout)
    require(receipt["status"]=="CORE_L8_PINNED_SHA256_MATERIALIZATION_PASS_RESEARCH"
            and receipt["verified_source_before_publication"]
            and output.read_bytes()==fixture["source"],
            "published bytes were not independently verified")
    positive_sha=sha_file(output)

    negatives=[]
    def reject(label,new_root,new_pin,new_output,original_paths=None):
        result=invoke(a.verifier,new_root,new_pin,fixture,ef,a.decoder,new_output)
        require(result.returncode!=0 and not new_output.exists(),
                "negative case accepted: "+label+" "+result.stdout+" "+result.stderr)
        negatives.append(label)

    reject("wrong-external-root-pin",root,"0"*64,target/"bad-pin.out")
    edited=bytearray(root.read_bytes())
    edited[-2]^=1
    tampered=target/"tampered-root.json"
    tampered.write_bytes(edited)
    reject("manifest-modified-without-pin-update",tampered,pin,target/"bad-root.out")

    def corrupt_file(path,label):
        raw=path.read_bytes()
        try:
            bad=bytearray(raw)
            bad[-1]^=1
            path.write_bytes(bad)
            reject(label,root,pin,target/(label+".out"))
        finally:
            path.write_bytes(raw)
        require(sha_file(path)==manifest["artifacts"][
            "ef" if path==ef else "rlb2" if path==fixture["rlb2"] else "rlr2"
        ]["sha256"],"failed to restore test fixture")
    corrupt_file(fixture["rlb2"],"rlb2-bitflip")
    corrupt_file(fixture["rlr2"],"mandatory-rlr2-bitflip")
    corrupt_file(ef,"ef-bitflip")

    absent=fixture["rlr2"]
    renamed=target/"rlr2.saved"
    absent.rename(renamed)
    try:
        reject("mandatory-rlr2-missing",root,pin,target/"missing.out")
    finally:
        renamed.rename(absent)

    wrong_src=copy.deepcopy(manifest)
    wrong_src["source"]["sha256"]="0"*64
    root_wrong,pin_wrong=create_root(target,wrong_src,1)
    reject("decoded-source-sha-mismatch-after-native-run",root_wrong,pin_wrong,
           target/"wrong-source.out")
    wrong_decoder=copy.deepcopy(manifest)
    wrong_decoder["decoder"]["sha256"]="0"*64
    root_decoder,pin_decoder=create_root(target,wrong_decoder,2)
    reject("decoder-not-pinned",root_decoder,pin_decoder,
           target/"wrong-decoder.out")

    duplicate=target/"duplicate-root.json"
    duplicate.write_text(
        '{"schema":"GLYPH_CORE_L8_PINNED_SOURCE_MATERIALIZATION_V0",'
        '"schema":"GLYPH_CORE_L8_PINNED_SOURCE_MATERIALIZATION_V0"}\n')
    reject("duplicate-json-field",duplicate,sha_file(duplicate),
           target/"duplicate.out")

    original_content=output.read_bytes()
    repeat=invoke(a.verifier,root,pin,fixture,ef,a.decoder,output)
    require(repeat.returncode!=0 and output.read_bytes()==original_content,
            "existing output was overwritten")
    negatives.append("no-overwrite-existing-output")

    symlink=target/"symlink.out"
    symlink.symlink_to(output)
    tamper=invoke(a.verifier,root,pin,fixture,ef,a.decoder,symlink)
    require(tamper.returncode!=0 and symlink.is_symlink()
            and output.read_bytes()==original_content,
            "existing symlink was followed or overwritten")
    negatives.append("no-follow-existing-output-symlink")

    report={
        "status":"CORE_L8_PINNED_SOURCE_MATERIALIZATION_POSITIVE_AND_HOSTILES_PASS",
        "source_bytes":len(fixture["source"]),
        "original_source_sha256":sha(fixture["source"]),
        "actual_published_sha256":positive_sha,
        "frozen_commit":REF,
        "root_manifest_sha256":pin,
        "backend_sha256":sha_file(a.decoder),
        "positive":{"source_sha256_verified_before_publish":True,
                    "inputs_snapshotted_cryptographically":True,
                    "published_source_bytes_verified":len(fixture["source"])},
        "hostile_cases_rejected":negatives,
        "hostile_case_count":len(negatives),
        "limitations":[
            "Caller-supplied root digest simulated external authority; no cryptographic signature.",
            "No root freshness/rollback/fork protection or fully portable O3/O4 proof.",
            "Native backend must be independently trusted for resource/environment safety.",
            "Snapshotted compressed inputs use extra temporary disk that is not retained storage.",
            "All output exactness is for a 24-byte binary-safe fixture.",
            "No source/SA passed to verifier/decoder; fixture source exists only as test oracle."
        ]
    }
    (target/"result.json").write_text(json.dumps(report,indent=2,sort_keys=True)+"\n")
    print(json.dumps({k:report.get(k) for k in
        ("status","source_bytes","root_manifest_sha256",
         "hostile_case_count","hostile_cases_rejected")},sort_keys=True))
if __name__=="__main__":
    main()
