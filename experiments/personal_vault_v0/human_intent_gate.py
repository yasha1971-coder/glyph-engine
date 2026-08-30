#!/usr/bin/env python3
import argparse,collections,json,subprocess
from pathlib import Path


def glyph_query(root, probe):
    out=subprocess.check_output([
        "python3","experiments/personal_vault_v0/query_rlb3x_object.py",
        "--rlb3x",str(root/"bwt.rlb3x"),
        "--locate-core",str(root/"locate.loc2"),
        "--objects",str(root/"objects.json"),
        "--pattern-hex",probe.encode("utf-8").hex(),
    ],text=True)
    return json.loads(out)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("root",type=Path)
    ap.add_argument("--cases",type=Path,default=Path("experiments/personal_vault_v0/human_intent_cases_v0.json"))
    a=ap.parse_args(); root=a.root

    cases_doc=json.loads(a.cases.read_text())
    object_map=json.loads((root/"objects.json").read_text())
    known_paths={o["path"] for o in object_map["objects"]}

    results=[]; passed=0; total_probes=0; glyph_calls=0
    for case in cases_doc["cases"]:
        expected=case.get("expected_path")
        if expected is not None:
            assert expected in known_paths,(case["id"],expected)
        scores=collections.Counter()
        evidence=collections.defaultdict(list)
        probe_reports=[]

        for probe in case["probes"]:
            got=glyph_query(root,probe); glyph_calls+=1; total_probes+=1
            assert got["rlb2_not_used"] is True,(case,probe,got)
            object_ids=set()
            for hit in got["valid_hits"]:
                key=(hit["object_id"],hit["path"])
                object_ids.add(key)
                evidence[key].append({
                    "probe":probe,
                    "object_offset":hit["object_offset"],
                    "corpus_offset":hit["corpus_offset"],
                })
            for key in object_ids:
                scores[key]+=1
            probe_reports.append({
                "probe":probe,
                "raw_count":got["raw_count"],
                "valid_count":got["valid_count"],
                "matched_objects":sorted({p for _,p in object_ids}),
            })

        ranked=sorted(scores.items(),key=lambda kv:(-kv[1],kv[0][1],kv[0][0]))
        top_score=ranked[0][1] if ranked else 0
        top=[key for key,score in ranked if score==top_score]
        selected_path=top[0][1] if len(top)==1 else None
        action="found" if selected_path is not None else "not_found" if not ranked else "ambiguous"
        matched_fraction=(top_score/len(case["probes"])) if case["probes"] else 0.0

        if case.get("expected_action")=="not_found":
            ok=(not ranked and action=="not_found")
        else:
            ok=(action=="found" and selected_path==expected and top_score==len(case["probes"]))
        assert ok,(case,{"action":action,"selected_path":selected_path,"top_score":top_score,"ranked":ranked,"probes":probe_reports})
        passed+=1

        ranked_out=[]
        for (oid,path),score in ranked[:5]:
            ranked_out.append({
                "object_id":oid,"path":path,"matched_probes":score,
                "probe_count":len(case["probes"]),
                "score_fraction":score/len(case["probes"]),
                "evidence":evidence[(oid,path)],
            })
        results.append({
            "id":case["id"],
            "human_query":case["human_query"],
            "ai_search_plan":{"probes":case["probes"],"source":"fixed assistant-authored plan"},
            "expected_path":expected,
            "action":action,
            "selected_path":selected_path,
            "top_score_fraction":matched_fraction,
            "ranked_candidates":ranked_out,
            "passed":ok,
        })

    report={
        "format":"GLYPH_HUMAN_INTENT_GATE_V0",
        "planner_status":cases_doc["planner_status"],
        "cases":len(cases_doc["cases"]),
        "cases_passed":passed,
        "all_cases_passed":passed==len(cases_doc["cases"]),
        "total_exact_probes":total_probes,
        "glyph_calls":glyph_calls,
        "retrieval_substrate":"RLB3X+LOC2+object-boundary-filter",
        "runtime_llm_used":False,
        "important_non_claim":"This gate validates AI-authored search plans against GLYPH exact evidence; it does not yet validate an on-device LLM generating those plans autonomously.",
        "results":results,
    }
    (root/"human-intent-gate.json").write_text(json.dumps(report,sort_keys=True,separators=(",",":"))+"\n")
    print(json.dumps({k:v for k,v in report.items() if k!="results"},sort_keys=True))

if __name__=="__main__":
    main()
