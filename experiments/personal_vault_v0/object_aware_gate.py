#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path


def run_query(root,pattern_hex):
    out=subprocess.check_output([
        "python3","experiments/personal_vault_v0/query_rlb3x_object.py",
        "--rlb3x",str(root/"bwt.rlb3x"),
        "--locate-core",str(root/"locate.loc2"),
        "--objects",str(root/"objects.json"),
        "--pattern-hex",pattern_hex,
    ],text=True)
    return json.loads(out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root",type=Path)
    a=ap.parse_args(); r=a.root

    om=json.loads((r/"objects.json").read_text())
    by_id={o["id"]:o for o in om["objects"]}
    queries=json.loads((r/"queries.json").read_text())["queries"]
    boundaries=json.loads((r/"boundaries.json").read_text())["cases"]

    positive=[]
    max_steps=0; total_steps=0
    for q in queries:
        got=run_query(r,q["pattern_hex"])
        obj=by_id[q["object_id"]]
        expected_object_offset=q["expected_offset"]-obj["offset"]
        assert got["raw_count"]==1,(q,got)
        assert got["valid_count"]==1,(q,got)
        assert got["rejected_cross_object_count"]==0,(q,got)
        assert got["rlb2_not_used"] is True,(q,got)
        hit=got["valid_hits"][0]
        assert hit["object_id"]==q["object_id"],(q,hit)
        assert hit["path"]==q["path"],(q,hit)
        assert hit["corpus_offset"]==q["expected_offset"],(q,hit)
        assert hit["object_offset"]==expected_object_offset,(q,hit)
        positive.append({"object_id":hit["object_id"],"object_offset":hit["object_offset"],"corpus_offset":hit["corpus_offset"]})
        max_steps=max(max_steps,got["maximum_lf_steps"]); total_steps+=got["total_lf_steps"]

    rejected=0
    for case in boundaries:
        got=run_query(r,case["pattern_hex"])
        assert got["raw_count"]==1,(case,got)
        assert got["raw_locate_offsets"]==[case["forbidden_offset"]],(case,got)
        assert got["valid_count"]==0,(case,got)
        assert got["rejected_cross_object_count"]==1,(case,got)
        assert got["rejected_cross_object_offsets"]==[case["forbidden_offset"]],(case,got)
        assert got["rlb2_not_used"] is True,(case,got)
        rejected+=1
        max_steps=max(max_steps,got["maximum_lf_steps"]); total_steps+=got["total_lf_steps"]

    report={
      "format":"GLYPH_PERSONAL_VAULT_OBJECT_AWARE_GATE_V0",
      "positive_queries":len(queries),
      "positive_queries_passed":len(positive),
      "cross_object_cases":len(boundaries),
      "cross_object_cases_rejected":rejected,
      "all_valid_hits_object_mapped":len(positive)==len(queries),
      "all_cross_object_hits_rejected":rejected==len(boundaries),
      "result_coordinates":["object_id","path","object_offset","corpus_offset"],
      "maximum_lf_steps_observed":max_steps,
      "total_lf_steps":total_steps,
      "rlb2_not_used_by_object_aware_path":True,
    }
    (r/"object-aware-gate.json").write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps(report,sort_keys=True))

if __name__=="__main__": main()
