#!/usr/bin/env python3
"""Full 100MB enwik8 native self-index restoration and negative gates.

Authenticates frozen source and generated RLB2/RLR2 against previously pinned
hashes, removes all build-only inputs, then invokes an independent C++ inverse
BWT implementation using RLB2 alone. Complete result SHA checked afterward.
No GLYPH O3/O4 provenance service, currentness root or production hardening.
"""
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

REF="c33cb493bc75f13ee4a904fabb190a29a196c94b"
SOURCE_URL="https://mattmahoney.net/dc/enwik8.zip"
SHA={
    "corpus":"2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8",
    "sa":"a70951cba4f51915f9e4ef7d7e819c11c09dcaf57677cd573a9ab746e9aed727",
    "bwt":"d00b44f165ba78dc11bdc048f8578021003f17fa19587fea944e499fb6514195",
    "rlb2":"4ca1218cc84b38539b6d12b0aacdfa30b21e387b24e4f8abaa828f1a6fffc0d3",
    "rlr2":"e76a49db554860bf907aa307a503240d1e71ef0400282c34548a9726a3f22719",
}
BYTES={"corpus":100000000,"sa":400000064,"bwt":200000058,
       "rlb2":73365105,"rlr2":12844028}
STATUS_PASS="NATIVE_ENWIK8_RLB2_SELF_INDEX_BYTE_RESTORE_PASS_EXPERIMENTAL"

def check(ok,why):
    if not ok: raise RuntimeError(why)

def sha(path):
    digest=hashlib.sha256()
    with path.open("rb") as fd:
        for part in iter(lambda:fd.read(4*1024*1024),b""):
            digest.update(part)
    return digest.hexdigest()

def run(command,logs,*,timeout=900,expected_success=True):
    command=[str(c) for c in command]
    recorded_command=command[:]
    if Path("/usr/bin/time").is_file():
        command=["/usr/bin/time","-f","GLYPH_NATIVE_PEAK_RSS_KB=%M"]+command
    started=time.perf_counter()
    p=subprocess.run(command,capture_output=True,text=True,
                     stdin=subprocess.DEVNULL,timeout=timeout)
    wall=round(time.perf_counter()-started,6)
    rss=re.search(r"GLYPH_NATIVE_PEAK_RSS_KB=(\d+)",p.stderr)
    rec={"command":recorded_command,"returncode":p.returncode,
         "wall_seconds":wall,
         "peak_rss_kb":int(rss.group(1)) if rss else None,
         "stdout_tail":p.stdout[-1200:],
         "stderr_tail":p.stderr[-900:]}
    logs.append(rec)
    if expected_success: check(p.returncode==0,f"command rejected: {rec}")
    return p,rec

def download_source(work,log):
    archive=work/"enwik8.zip"
    started=time.perf_counter()
    with urllib.request.urlopen(SOURCE_URL,timeout=180) as src,archive.open("wb") as out:
        shutil.copyfileobj(src,out,length=1024*1024)
    check(archive.stat().st_size==36445475,"unexpected official ZIP size")
    with zipfile.ZipFile(archive) as z:
        check(z.namelist()==["enwik8"],"unexpected zip members")
        check(z.getinfo("enwik8").file_size==BYTES["corpus"],"untrusted unzip size")
        with z.open("enwik8") as src,(work/"enwik8").open("wb") as out:
            shutil.copyfileobj(src,out,length=1024*1024)
    archive.unlink()
    src=work/"enwik8"
    check(src.stat().st_size==BYTES["corpus"] and sha(src)==SHA["corpus"],
          "wrong official source identity")
    log.append({"step":"official_enwik8_fetch_and_hash","source_bytes":BYTES["corpus"],
                "source_sha256":SHA["corpus"],
                "elapsed_seconds":round(time.perf_counter()-started,6)})
    return src

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--native",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    args=p.parse_args()
    reference=args.reference.resolve()
    native=args.native.resolve()
    output=args.output.resolve()
    check(subprocess.check_output(["git","-C",str(reference),"rev-parse","HEAD"],
          text=True).strip()==REF,"wrong original reference checkout")
    check(native.is_file(),"compiled independent native decoder missing")
    check(not output.exists(),"refuse existing output directory")
    output.mkdir(parents=True)
    work=output/"work"
    work.mkdir()
    logs=[]
    report={"schema":"GLYPH_NATIVE_ENWIK8_SELF_INDEX_RESTORE_V0",
            "status":"BLOCKED","original_commit":REF,
            "corpus_sha256":SHA["corpus"],"corpus_bytes":BYTES["corpus"],
            "native_decoder_sha256":sha(native)}
    try:
        source=download_source(work,logs)
        sa=work/"canonical.sa.binary_v1"
        bwt=work/"canonical.bwt.binary_v1"
        rlb2=work/"canonical.rlb2"
        rlr2=work/"canonical.rlr2"
        run([reference/"build/build_sa_binary_v1",source,sa],logs,timeout=900)
        run([reference/"build/build_bwt_binary_v1",source,sa,bwt],logs,timeout=900)
        for key,artifact in [("sa",sa),("bwt",bwt)]:
            check(artifact.stat().st_size==BYTES[key] and sha(artifact)==SHA[key],
                  f"cannot reproduce historical {key}")
        run([sys.executable,reference/"tools/rlbwt_container_v2.py",
             "encode",bwt,rlb2],logs,timeout=1800)
        run([sys.executable,reference/"tools/rlbwt_rank_v2.py",
             "build",rlb2,rlr2,"--rank-step","8192"],logs,timeout=1800)
        for key,artifact in [("rlb2",rlb2),("rlr2",rlr2)]:
            check(artifact.stat().st_size==BYTES[key] and sha(artifact)==SHA[key],
                  f"cannot reproduce historical {key}")
        # Prove that no original source, 400MB SA, 200MB canonical BWT or
        # uncompressed source payload is available to the next decoder call.
        for generated in (source,sa,bwt):
            generated.unlink()
        check(all(not x.exists() for x in (source,sa,bwt)),
              "a construction-only source is still available")
        # The two retained original file hashes are verified against pinned
        # external source evidence; native inverse independently consumes RLB2.
        restored=work/"restored-from-self-index-enwik8"
        inverse,metric=run([native,rlb2,restored],logs,timeout=1800)
        native_metrics=json.loads(inverse.stdout)
        check(native_metrics["status"]=="NATIVE_RLB2_SELF_INDEX_RESTORE_RESEARCH_SUCCESS",
              "native decoder did not affirm completion")
        check(restored.stat().st_size==BYTES["corpus"] and sha(restored)==SHA["corpus"],
              "FATAL: native full 100MB source SHA/length mismatch")
        # Corruption of compressed source must fail before any forged byte
        # stream can be materialized as trusted restoration.
        corrupted=work/"corrupted.rlb2"
        shutil.copyfile(rlb2,corrupted)
        with corrupted.open("r+b") as out:
            out.seek(-1,os.SEEK_END)
            old=out.read(1)
            out.seek(-1,os.SEEK_END)
            out.write(bytes([old[0]^1]))
        rejected=work/"untrusted-source-output"
        hostile,hostile_info=run([native,corrupted,rejected],logs,
                                timeout=900,expected_success=False)
        check(hostile.returncode!=0 and not rejected.exists(),
              "corrupt mandatory compressed payload was trusted")
        report.update({
            "status":STATUS_PASS,
            "original_hashes_reproduced":SHA,
            "native_inverse":native_metrics,
            "native_wall_seconds":metric["wall_seconds"],
            "native_peak_rss_kb":metric["peak_rss_kb"],
            "recovery_sha256":sha(restored),
            "recovery_bytes":restored.stat().st_size,
            "retained_rlb2_bytes":rlb2.stat().st_size,
            "retained_rlr2_bytes":rlr2.stat().st_size,
            "retained_exact_search_rank_bytes":rlb2.stat().st_size+rlr2.stat().st_size,
            "build_only_source_sa_bwt_removed_before_inverse":True,
            "hostile_corrupt_rlb2_rejected":True,
            "cannot_treat_fnv_as_crypto_trust":True,
            "limitations":[
                "Native RLB2 inversion is full sequential source recovery, not fast bounded range extract.",
                "Transient O(N) u32 LF table plus raw output and visited bits are required; record peak RSS.",
                "RLR2 is hash-bound source search material, but native restoration uses only RLB2.",
                "No sample index or query proof passed this workflow, so not a combined proof-carrying release.",
                "Pinned external SHA after restoration is required; RLB2 header FNV is not cryptographic authentication.",
                "This source has no embedded NUL and does not establish universal compression ratio.",
                "No signed witness/rollback proof, O3/O4 canonical doc coordinates or production ACLs.",
            ],
        })
    except Exception as e:
        report["status"]="BLOCKED"
        report["error"]=repr(e)
    finally:
        (output/"result.json").write_text(json.dumps(report,sort_keys=True,indent=2)+"\n")
        (output/"commands.json").write_text(json.dumps(logs,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:report.get(k) for k in ("status","error","recovery_bytes",
                       "recovery_sha256","native_wall_seconds","native_peak_rss_kb",
                       "retained_exact_search_rank_bytes","hostile_corrupt_rlb2_rejected")},
                     sort_keys=True))
    return 0 if report["status"]==STATUS_PASS else 2

if __name__=="__main__":
    sys.exit(main())
