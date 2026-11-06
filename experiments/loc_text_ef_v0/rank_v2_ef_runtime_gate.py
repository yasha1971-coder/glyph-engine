#!/usr/bin/env python3
"""Real RankV2/RLB2/RLR2 queries with new physical text-sampled EF locate.

Reference C++/Python binary-safe runtime is kept at exact immutable historical
commit. Small binary fixture; full canonical enwik8 query integration is NOT
covered here. Full raw-byte oracle and every LF row are checked.
"""
import argparse
import hashlib
import importlib.util
import json
import struct
import subprocess
import sys
from pathlib import Path


def must(ok,msg):
    if not ok: raise AssertionError(msg)


def load(path,name):
    spec=importlib.util.spec_from_file_location(name,path)
    must(spec is not None and spec.loader is not None,"cannot import ref fixture")
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod


def oracle(data,p):
    found=[]
    pos=0
    while True:
        at=data.find(p,pos)
        if at<0:return found
        found.append(at)
        pos=at+1


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    reference=a.reference.resolve()
    source_sha="c33cb493bc75f13ee4a904fabb190a29a196c94b"
    must(subprocess.check_output(
        ["git","-C",str(reference),"rev-parse","HEAD"],
        text=True).strip()==source_sha,"wrong historical checkout")
    output=a.output.resolve()
    must(not output.exists(),"disposable output must not exist")
    output.mkdir(parents=True)
    tools=reference/"tools"
    vault=reference/"experiments/personal_vault_v0"
    sys.path[:0]=[str(tools),str(vault),str(Path(__file__).parent)]
    import rlbwt_query_v2 as query
    import loc2_experimental as loc2tool
    from real_enwik8_ef_physical_gate import build,verify
    from query_ef_research import make_runtime, LocateEF

    gate=load(tools/"check_rlbwt_query_v2.py","reference_fixture")
    fixture=gate.build_fixture(output,tools/"rlbwt_container_v2.py",
                               tools/"rlbwt_rank_v2.py")
    sa_values=fixture["suffix_array"]
    sa=output/"canonical_frozen_synthetic.sa"
    sa.write_bytes(b"\x00"*60+struct.pack("<"+"I"*len(sa_values),*sa_values))
    step=fixture["sample_step"]
    artifact=output/"text_ef.research"
    geometry=build(sa,len(sa_values),step,artifact)
    verified=verify(sa,artifact,geometry)
    expected_sha=verified["sha256"]
    loc2=output/"baseline.loc2"
    loc2tool.build(sa,len(sa_values),step,loc2)
    loc2tool.verify(sa,loc2)
    original=query.QueryRuntime(fixture["rlb2"],fixture["rlr2"],fixture["locate"])
    new=make_runtime(tools,fixture["rlb2"],fixture["rlr2"],artifact)
    row_step_count=[]
    test_patterns=[
        b"\x00",b"\xff",b"banana",b"bandana",
        b"na",b"an",b"banana_bandana",b"_",
        b"\xff\x00",b"na!",b"\x00banana",b"absent"
    ]
    count=0
    maximum=0
    max_old=0
    try:
        for row in range(len(sa_values)):
            new_offset,steps=new.locate_row(row)
            original_offset,oldsteps=original.locate_row(row)
            must(new_offset==original_offset==sa_values[row],
                 "full SA row locate oracle failed")
            must(steps<=step-1,
                 f"LF strict bound broken at row {row}: {steps}")
            maximum=max(maximum,steps)
            max_old=max(max_old,oldsteps)
            row_step_count.append(steps)
        for pattern in test_patterns:
            raw_hits=oracle(fixture["source"],pattern)
            for limit in (-1,0,1,2):
                for repetition in range(5):
                    a=original.query(pattern,limit)
                    b=new.query(pattern,limit)
                    must(a["count"]==b["count"]==len(raw_hits),
                         "count mismatched raw oracle")
                    must(a["fm_interval"]==b["fm_interval"],
                         "backward-search FM interval changed")
                    if limit<0:
                        must(a["locate_offsets"]==b["locate_offsets"]==raw_hits,
                             "full locate mismatch source")
                    else:
                        selected=b["locate_offsets"]
                        must(len(selected)==min(limit,len(raw_hits))
                             and len(set(selected))==len(selected)
                             and set(selected).issubset(set(raw_hits)),
                             "bounded locate not exact subset")
                    must(b["maximum_lf_steps"]<=step-1,
                         "unbounded LF path in new query")
                    count+=1
    finally:
        original.close()
        new.close()
    # Explicit independent SHA pin must reject payload mutation. This is
    # NOT an authoritative signed root/currentness mechanism.
    data=bytearray(artifact.read_bytes())
    data[-1]^=1
    mutated=output/"mutated.text_ef.research"
    mutated.write_bytes(data)
    must(hashlib.sha256(mutated.read_bytes()).hexdigest()!=expected_sha,
         "mutated artifact matched pinned sha")
    # Wrong header must fail inside actual parser, not just a hash.
    data=bytearray(artifact.read_bytes())
    data[0]^=1
    bad=output/"bad_header.text_ef.research"
    bad.write_bytes(data)
    rejected=False
    try:
        o=LocateEF(bad)
        o.close()
    except Exception:
        rejected=True
    must(rejected,"broken magic accepted")
    result={
        "schema":"GLYPH_LOC_TEXT_EF_REAL_RANKV2_QUERY_PARITY_RESEARCH_V0",
        "status":"REAL_RLB2_RLR2_EF_LOCATE_PARITY_PASS_NOT_PROOF",
        "pinned_reference_sha":source_sha,
        "source_sha256":hashlib.sha256(fixture["source"]).hexdigest(),
        "original_sample_step":step,
        "loc1_bytes":fixture["locate"].stat().st_size,
        "loc2_bytes":loc2.stat().st_size,
        "ef_file_bytes":artifact.stat().st_size,
        "ef_sha256":expected_sha,
        "all_sa_rows_located":len(sa_values),
        "max_lf_steps_new":maximum,
        "max_lf_steps_row_sampled":max_old,
        "test_cases":count,
        "patterns":len(test_patterns),
        "hostile_cases":2,
        "verified_ef_samples":verified["sampled_sa_records_verified"],
        "scope":["Only original small binary-safe fixture, not full enwik8 RankV2 query replay",
                 "No production O3 canonical bounded output contract",
                 "No O4 portable proof/currentness/source restore or released manifest",
                 "EF sample lookup adds select/binary-search cost; no speed win proven"]
    }
    (output/"result.json").write_text(json.dumps(result,sort_keys=True,indent=2)+"\n")
    print(json.dumps(result,sort_keys=True))
    return 0


if __name__=="__main__":
    raise SystemExit(main())
