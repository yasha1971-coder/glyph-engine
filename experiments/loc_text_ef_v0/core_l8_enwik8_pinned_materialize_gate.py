#!/usr/bin/env python3
"""CORE-L8 canonical enwik8 exact authenticated-identity materialization gate.

Research scope: manifest root SHA transported as test-only known external pin,
source/RLB2/RLR2/EF externally pinned to historical hashes BEFORE generating
a new manifest. External pinned SHA != signature/currentness witness.

Rebuild source and all original retained artifacts, remove source/SA/BWT.
Verifier snapshots RLB2+RLR2+EF+decoder into a private directory, validates
each digest, recovers output using compressed index only, then hashes output
against frozen source identity before making it visible in a final path.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

from native_inverse_enwik8_gate import REF,SHA,BYTES,download_source,run,sha,check
from real_enwik8_ef_physical_gate import build,verify

PASS="CORE_L8_PINNED_ENWIK8_MATERIALIZATION_RESEARCH_PASS"
EF_SHA="c1cd29d7bb40805d23d4aa891cd08808a1d7e1e3734d6a13799163dc23d469c7"

def sha_bytes(data):
    return hashlib.sha256(data).hexdigest()

def pinned_json(target,body):
    raw=(json.dumps(body,sort_keys=True,separators=(",",":"))+"\n").encode()
    target.write_bytes(raw)
    return sha_bytes(raw)

def gated_command(verifier,manifest,root,rlb2,rlr2,ef,decoder,out):
    return [sys.executable,verifier,
            "--manifest",manifest,"--root-sha256",root,
            "--rlb2",rlb2,"--rlr2",rlr2,"--ef",ef,
            "--decoder",decoder,"--output",out,"--threads","4",
            "--mode","rank"]

def main():
    p=argparse.ArgumentParser()
    for name in ("reference","decoder","verifier","output"):
        p.add_argument("--"+name,type=Path,required=True)
    a=p.parse_args()
    ref=a.reference.resolve()
    decoder=a.decoder.resolve()
    verifier=a.verifier.resolve()
    target=a.output.resolve()
    check(subprocess.check_output(["git","-C",str(ref),"rev-parse","HEAD"],
          text=True).strip()==REF,"frozen source commit mismatch")
    check(decoder.is_file() and verifier.is_file() and not target.exists(),
          "required verifier/decoder missing or output dir already exists")
    target.mkdir(parents=True)
    work=target/"work"
    work.mkdir()
    logs=[]
    report={"schema":"GLYPH_CORE_L8_100MB_RESTORE_PINNED_ROOT_V0",
            "status":"BLOCKED","original_frozen_commit":REF}
    try:
        source=download_source(work,logs)
        sa=work/"temporary-source.sa"
        bwt=work/"temporary-source.bwt"
        rlb2=work/"retained.rlb2"
        rlr2=work/"retained.rlr2"
        ef=work/"retained.loc-ef"
        run([ref/"build/build_sa_binary_v1",source,sa],logs,timeout=900)
        run([ref/"build/build_bwt_binary_v1",source,sa,bwt],logs,timeout=900)
        for key,path in [("sa",sa),("bwt",bwt)]:
            check(path.stat().st_size==BYTES[key] and sha(path)==SHA[key],
                  "historical construction SHA mismatch: "+key)
        run([sys.executable,ref/"tools/rlbwt_container_v2.py",
             "encode",bwt,rlb2],logs,timeout=1800)
        run([sys.executable,ref/"tools/rlbwt_rank_v2.py",
             "build",rlb2,rlr2,"--rank-step","8192"],logs,timeout=1800)
        for key,path in [("rlb2",rlb2),("rlr2",rlr2)]:
            check(path.stat().st_size==BYTES[key] and sha(path)==SHA[key],
                  "historical compressed source SHA mismatch: "+key)
        geom=build(sa,100000001,128,ef)
        inspected=verify(sa,ef,geom)
        check(inspected["sampled_sa_records_verified"]==781251,
              "noncanonical EF sample count")
        check(ef.stat().st_size==2844324 and sha(ef)==EF_SHA,
              "experimental EF SHA mismatch")
        manifest={
            "schema":"GLYPH_CORE_L8_PINNED_SOURCE_MATERIALIZATION_V0",
            "frozen_commit":REF,
            "source":{"bytes":BYTES["corpus"],"sha256":SHA["corpus"]},
            "artifacts":{name:{"bytes":path.stat().st_size,"sha256":sha(path)}
                         for name,path in [("rlb2",rlb2),("rlr2",rlr2),("ef",ef)]},
            "decoder":{"sha256":sha(decoder)}
        }
        manifest_path=work/"l8-manifest.json"
        root_pin=pinned_json(manifest_path,manifest)
        report["root_pin_sha256"]=root_pin
        report["manifest_canonical_json_bytes"]=manifest_path.stat().st_size
        report["verified_original_artifact_sha256"]={**SHA,"ef":EF_SHA}
        # Verify the source source SHA was pinned BEFORE removing original.
        for old in (source,sa,bwt):
            old.unlink()
        check(all(not x.exists() for x in (source,sa,bwt)),
              "build-only source or construction index still present")
        destination=work/"final-verified-source-enwik8"
        success,metric=run(gated_command(
            verifier,manifest_path,root_pin,rlb2,rlr2,ef,decoder,destination),
            logs,timeout=1800)
        attestation=json.loads(success.stdout)
        check(attestation["status"]==
              "CORE_L8_PINNED_SHA256_MATERIALIZATION_PASS_RESEARCH",
              "source materialization not authenticated against pinned manifest")
        check(attestation["verified_source_before_publication"]
              and attestation["private_decoder_snapshot_verified"],
              "verified-then-publish contract unsatisfied")
        check(destination.stat().st_size==BYTES["corpus"]
              and sha(destination)==SHA["corpus"],
              "source re-read after publication disagrees with pinned identity")
        check(attestation["manifest_sha256"]==root_pin
              and attestation["source_sha256"]==SHA["corpus"],
              "wrong effective root or source identity")

        def negative(label,path,expected_root,output):
            proc,info=run(gated_command(verifier,path,expected_root,rlb2,
                rlr2,ef,decoder,output),logs,timeout=1800,
                expected_success=False)
            check(proc.returncode!=0 and not output.exists(),
                  "fail-open materialization for "+label)
            return {"name":label,"rejected":True,"wall_seconds":info["wall_seconds"]}

        hostiles=[]
        hostiles.append(negative("wrong-root-pin",manifest_path,"0"*64,
                                 work/"untrusted-root-out"))
        # A mandatory RLR2 is not read by native inverse but is still
        # required to remain cryptographically intact for search state.
        with rlr2.open("r+b") as fd:
            fd.seek(-1,2); prior=fd.read(1)
            fd.seek(-1,2); fd.write(bytes([prior[0]^1]))
        try:
            hostiles.append(negative("mandatory-rlr2-bitflip",manifest_path,root_pin,
                                     work/"untrusted-rank-out"))
        finally:
            with rlr2.open("r+b") as fd:
                fd.seek(-1,2);fd.write(prior)
        check(sha(rlr2)==SHA["rlr2"],"failed to restore immutable test fixture")
        # Deliberately pinned wrong expected source SHA (separate new root)
        # must be rejected AFTER complete decoding, without visible output.
        false_manifest=json.loads(manifest_path.read_text())
        false_manifest["source"]["sha256"]="0"*64
        false_root=work/"false-source-root.json"
        false_sha=pinned_json(false_root,false_manifest)
        hostiles.append(negative("wrong-output-hash-after-decoder",
                                 false_root,false_sha,
                                 work/"untrusted-source-out"))
        report.update({
            "status":PASS,
            "canonical_source_sha256":SHA["corpus"],
            "canonical_source_bytes":BYTES["corpus"],
            "source_verified_after_publication":True,
            "source_verified_before_publication":True,
            "decoder_snapshot_verified":True,
            "verified_components":["rlb2","rlr2","ef","decoder"],
            "positive_receipt":attestation,
            "materialization_wall_seconds":metric["wall_seconds"],
            "materialization_peak_rss_kb":metric["peak_rss_kb"],
            "retained_index_bytes":sum(p.stat().st_size for p in (rlb2,rlr2,ef)),
            "retained_index_ratio":sum(p.stat().st_size for p in (rlb2,rlr2,ef))
                                   / BYTES["corpus"],
            "negative_scenarios":hostiles,
            "rejected_cases":len(hostiles),
            "no_source_sa_bwt_files_used_in_restore":True,
            "limitations":[
                "CI root SHA generated inside trusted test after matching all historically pinned component digests.",
                "A caller-supplied root SHA is not an authenticated signature or a rollback/currentness witness.",
                "Verifier snapshots three retained inputs and decoder to extra temporary disk.",
                "No P/R/O signed portable evidence or canonical multi-document query result proof.",
                "Not a production hostile-host security proof; privileged local attacker is out of scope.",
                "This is a single-corpus 100MB test, not full 1TB admission."
            ]
        })
    except Exception as exc:
        report.update(status="BLOCKED",error=repr(exc))
    finally:
        (target/"result.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        (target/"commands.json").write_text(json.dumps(logs,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:report.get(k) for k in
        ("status","error","root_pin_sha256","canonical_source_bytes",
         "materialization_wall_seconds","materialization_peak_rss_kb",
         "retained_index_bytes","rejected_cases")},sort_keys=True))
    return 0 if report["status"]==PASS else 2

if __name__=="__main__":
    sys.exit(main())
