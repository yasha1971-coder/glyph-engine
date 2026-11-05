#!/usr/bin/env python3
"""Independent, short enwik8 LOC2 physical geometry gate against canonical SA.

Does not reconstruct RLB2/RLR2; intentionally makes NO full query/restore claim.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from enwik8_loc1_loc2_full_replay import (
    ENWIK8, REFERENCE_SHA, PIN, EXPECTED_BYTES, SAMPLES,
    read_official_enwik8, run, emit_loc1, file_sha,
)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args=p.parse_args()
    ref=args.reference.resolve()
    require(subprocess.check_output(["git","-C",str(ref),"rev-parse","HEAD"],
                                    text=True).strip()==REFERENCE_SHA,"wrong source")
    out=args.output.resolve()
    require(not out.exists(),"refuse existing output")
    out.mkdir(parents=True)
    work=out/"work";work.mkdir()
    logs=[];r={"format":"GLYPH_ENWIK8_LOC2_SA_GEOMETRY_V0","status":"NOT_EXECUTED"}
    try:
        src=read_official_enwik8(work,logs)
        sa=work/"canonical.sa"
        run([ref/"build/build_sa_binary_v1",src,sa],logs)
        require(sa.stat().st_size==EXPECTED_BYTES["sa"] and file_sha(sa)==PIN["sa"],
                "SA exact historical SHA failed")
        loc1=work/"canonical.loc1"
        emit_loc1(sa,loc1,ENWIK8["bytes"]+1,SAMPLES)
        require(loc1.stat().st_size==EXPECTED_BYTES["loc1"]
                and file_sha(loc1)==PIN["loc1"],"LOC1 historical hash mismatch")
        loc2=work/"candidate.loc2"
        tool=ref/"experiments/personal_vault_v0/loc2_experimental.py"
        run([sys.executable,tool,"build",sa,str(ENWIK8["bytes"]+1),
             str(SAMPLES),loc2],logs)
        run([sys.executable,tool,"verify",sa,loc2],logs)
        require(loc2.stat().st_size==EXPECTED_BYTES["loc2"],
                "LOC2 expected sample geometry mismatch")
        rows=ENWIK8["bytes"]+1
        samples=(rows+SAMPLES-1)//SAMPLES
        require(samples==781251,"LOC2 sample count mismatch")
        r.update({
            "status":"ENWIK8_LOC2_PHYSICAL_AND_CANONICAL_SA_PARITY_PASS",
            "source_sha256":ENWIK8["sha256"],
            "sa_sha256":file_sha(sa),
            "loc1_sha256":file_sha(loc1),
            "loc2_sha256":file_sha(loc2),
            "loc1_bytes":loc1.stat().st_size,
            "loc2_bytes":loc2.stat().st_size,
            "saved_locate_bytes":loc1.stat().st_size-loc2.stat().st_size,
            "sampled_sa_records_checked":samples,
            "source_bytes":ENWIK8["bytes"],
            "hypothetical_runtime_before_loc2_manifest_bytes":(
                73365105+12844028+loc2.stat().st_size),
            "limits":[
                "Only SA sample geometry, checksum and exact canonical SA ancestry proven.",
                "No physical RLB2/RLR2 combined runtime or query execution in this gate.",
                "The 89,334,169-byte figure excludes a new LOC2 research manifest.",
                "LOC2 has no approved canonical V2 production binding/portable proof contract.",
                "No speed claim, independent freshness witness or general 1TB result.",
            ],
        })
    except Exception as exc:
        r.update(status="BLOCKED",error=str(exc))
    finally:
        (out/"result.json").write_text(json.dumps(r,sort_keys=True,indent=2)+"\n")
        (out/"commands.json").write_text(json.dumps(logs,sort_keys=True,indent=2)+"\n")
    print(json.dumps({k:r.get(k) for k in
                       ("status","error","loc1_bytes","loc2_bytes","saved_locate_bytes")},
                     sort_keys=True))
    return 0 if r["status"].startswith("ENWIK8_LOC2_PHYSICAL_") else 2


if __name__=="__main__":
    raise SystemExit(main())
