#!/usr/bin/env python3
"""Experimental byte-range extraction from *only* RLB2/RLR2 + EF locate.

No separately stored original corpus or second compressed data payload is used
by the extraction function. The q -> SA row table is reconstructed in RAM from
the already accounted physical EF file, not persisted in a hidden sidecar.
"""
import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

def insist(ok, message):
    if not ok:
        raise ValueError(message)

def original_module(path):
    s=importlib.util.spec_from_file_location("canonical_fixture_generator",path)
    insist(s and s.loader, "missing pinned fixture module")
    obj=importlib.util.module_from_spec(s)
    sys.modules[s.name]=obj
    s.loader.exec_module(obj)
    return obj

class EFByteExtractor:
    """Reads exact source bytes backwards from text-order sampled suffix rows."""
    def __init__(self,query_runtime):
        from query_ef_research import getbits
        self.rt=query_runtime
        self.rank=query_runtime.rank
        self.loc=query_runtime.locate
        self.n=query_runtime.corpus_bytes
        self.step=self.loc.sample_step
        self.rows_by_text_quotient=[None]*self.loc.m
        for i in range(self.loc.m):
            q=getbits(self.loc.values,i*self.loc.width,self.loc.width)
            insist(q<len(self.rows_by_text_quotient)
                   and self.rows_by_text_quotient[q] is None,
                   "duplicate, missing or out-of-range source text sample")
            self.rows_by_text_quotient[q]=self.loc.select(i)
        insist(all(r is not None for r in self.rows_by_text_quotient),
               "incomplete source sampled suffix coverage")
        self.sample_rows_ram_bytes=self.loc.m*4

    def row_for_suffix_offset(self,at):
        insist(0<=at<=self.n,"sample suffix offset outside source")
        if at==self.n and (at%self.step):
            # Pinned GLYPH binary-safe SA includes terminal suffix, row 0.
            return 0
        insist(at%self.step==0,"unaligned suffix offset")
        return self.rows_by_text_quotient[at//self.step]

    def extract(self,start,length):
        insist(start>=0 and length>=0 and start<=self.n
               and length<=self.n-start,"out-of-range materialization")
        if length==0:
            return b"",0
        answer=bytearray(length)
        target_end=start+length
        steps=0
        cur=start
        while cur<target_end:
            right=min((cur//self.step+1)*self.step,self.n)
            row=self.row_for_suffix_offset(right)
            insist(0<=row<self.loc.sa_size,"invalid sample suffix row")
            # Reconstruct [cur,right) BACKWARDS; discard the unrequested
            # trailing tail [target_end,right) when the last block is partial.
            for pos in range(right-1,cur-1,-1):
                symbol,rank_before=self.rank.symbol_and_rank(row)
                insist(0<=symbol<256,"unexpected logical sentinel inside text")
                if pos<target_end:
                    answer[pos-start]=symbol
                row=self.rank.C[symbol]+rank_before
                steps+=1
            if cur%self.step==0:
                insist(row==self.row_for_suffix_offset(cur),
                       "LF block failed to reach committed adjacent sample")
            cur=right
        insist(steps<=length+self.step-1,"bounded range decode amplification")
        return bytes(answer),steps

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--reference",type=Path,required=True)
    p.add_argument("--output",type=Path,required=True)
    a=p.parse_args()
    ref=a.reference.resolve()
    root=a.output.resolve()
    insist(not root.exists(),"refuse preexisting experiment directory")
    insist(subprocess.check_output(["git","-C",str(ref),"rev-parse","HEAD"],
           text=True).strip()=="c33cb493bc75f13ee4a904fabb190a29a196c94b",
           "wrong original immutable GLYPH source")
    root.mkdir(parents=True)
    tools=ref/"tools"
    sys.path[:0]=[str(tools),str(Path(__file__).parent)]
    from query_ef_research import make_runtime
    from real_enwik8_ef_physical_gate import build,verify
    gate=original_module(tools/"check_rlbwt_query_v2.py")
    fixture=gate.build_fixture(root,tools/"rlbwt_container_v2.py",
                               tools/"rlbwt_rank_v2.py")
    import struct
    sa=root/"source-construction-only.sa"
    arr=fixture["suffix_array"]
    sa.write_bytes(b"\x00"*60+struct.pack("<"+"I"*len(arr),*arr))
    artifact=root/"physical-text-ef.research"
    metadata=build(sa,len(arr),fixture["sample_step"],artifact)
    checked=verify(sa,artifact,metadata)
    runtime=make_runtime(tools,fixture["rlb2"],fixture["rlr2"],artifact)
    try:
        extractor=EFByteExtractor(runtime)
        source=fixture["source"]
        total=0
        worst_amp=0
        for begin in range(len(source)+1):
            for length in range(len(source)-begin+1):
                produced,lf=extractor.extract(begin,length)
                insist(produced==source[begin:begin+length],
                       f"source byte-range mismatch {begin},{length}")
                insist(lf<=length+extractor.step-1,
                       "invalid LF-work guarantee")
                total+=1
                worst_amp=max(worst_amp,lf-length)
        patterns=[b"\x00",b"\xff",b"banana",b"bandana",b"na",
                  b"banana_bandana",b"\xff\x00",b"absent"]
        verified_matches=0
        for pat in patterns:
            result=runtime.query(pat,-1)
            for offset in result["locate_offsets"]:
                recovered,lf=extractor.extract(offset,len(pat))
                insist(recovered==pat,"query evidence bytes do not equal pattern")
                verified_matches+=1
        outside=0
        for start,length in [(-1,1),(len(source),1),(0,len(source)+1),
                             (len(source)+1,0)]:
            try:
                extractor.extract(start,length)
            except ValueError:
                outside+=1
        insist(outside==4,"out-of-bound query accepted")
        record={
            "status":"PHYSICAL_SELF_INDEX_BOUNDED_BYTE_RANGE_RETRIEVAL_PASS_SMALL",
            "pinned_reference":"c33cb493bc75f13ee4a904fabb190a29a196c94b",
            "source_bytes":len(source),
            "source_sha256":hashlib.sha256(source).hexdigest(),
            "sample_step":extractor.step,
            "physical_ef_bytes":artifact.stat().st_size,
            "physical_ef_sha256":checked["sha256"],
            "exhaustive_byte_ranges_verified":total,
            "query_matches_byte_verified":verified_matches,
            "out_of_range_cases_rejected":outside,
            "max_extra_decoded_bytes":worst_amp,
            "expected_max_extra_decoded_bytes":extractor.step-1,
            "ephemeral_quotient_to_row_ram_bytes":extractor.sample_rows_ram_bytes,
            "persistent_extra_source_payload_bytes":0,
            "falsifiable_scope_limits":[
                "Only 24 byte binary-safe source, not 100 MB enwik8.",
                "EF q-to-row mapping is constructed in RAM; startup/RSS not yet measured.",
                "Python RankV2 may be too slow for production range extraction.",
                "No external O3/O4 committed evidence, currentness, signatures or false-empty proof.",
                "Sample authenticity depends on separately pinned index hash and correct rank source."
            ]
        }
    finally:
        runtime.close()
    (root/"result.json").write_text(json.dumps(record,sort_keys=True,indent=2)+"\n")
    print(json.dumps(record,sort_keys=True))

if __name__=="__main__":
    main()
