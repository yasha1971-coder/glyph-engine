#!/usr/bin/env python3
import argparse, collections, json
from pathlib import Path


def load_objects(root):
    items=[]
    for p in sorted(x for x in root.rglob('*') if x.is_file() and x.name!='SHA256SUMS'):
        items.append((str(p.relative_to(root)),p.read_bytes()))
    return items


def evaluate_case(case,objects):
    scores=collections.Counter()
    probes=[]
    for probe in case['probes']:
        needle=probe.encode('utf-8')
        matched=[]
        for path,data in objects:
            if needle in data:
                matched.append(path)
                scores[path]+=1
        probes.append({'probe':probe,'matched_objects':matched,'matched_count':len(matched)})
    ranked=sorted(scores.items(),key=lambda kv:(-kv[1],kv[0]))
    top_score=ranked[0][1] if ranked else 0
    top=[path for path,score in ranked if score==top_score]
    selected=top[0] if len(top)==1 else None
    action='found' if selected is not None else 'not_found' if not ranked else 'ambiguous'
    expected_action=case.get('expected_action','found')
    expected_path=case.get('expected_path')
    if expected_action=='found':
        ok=(action=='found' and selected==expected_path and top_score==len(case['probes']))
    elif expected_action=='ambiguous':
        ok=(action=='ambiguous' and len(top)>1)
    elif expected_action=='not_found':
        ok=(action=='not_found' and not ranked)
    else:
        ok=False
    return {
        'id':case['id'],'human_query':case['human_query'],'expected_action':expected_action,
        'expected_path':expected_path,'action':action,'selected_path':selected,
        'top_score':top_score,'top_candidates':top,'ranked_candidates':ranked[:10],
        'probes':probes,'passed':ok,
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('input_root',type=Path)
    ap.add_argument('--cases',type=Path,default=Path('experiments/personal_vault_v0/human_intent_cases_v0.json'))
    ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    doc=json.loads(a.cases.read_text())
    objects=load_objects(a.input_root)
    results=[evaluate_case(case,objects) for case in doc['cases']]
    failed=[r for r in results if not r['passed']]
    report={
        'format':'GLYPH_HUMAN_INTENT_PREFLIGHT_V0',
        'planner_status':doc['planner_status'],
        'objects':len(objects),'cases':len(results),'cases_passed':len(results)-len(failed),
        'cases_failed':len(failed),'all_cases_passed':not failed,
        'method':'direct-byte-scan-fixture-preflight-not-GLYPH-runtime',
        'results':results,
    }
    if a.out:
        a.out.write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},sort_keys=True))
    if failed:
        for r in failed:
            print(json.dumps(r,ensure_ascii=False,sort_keys=True))
        raise SystemExit(1)

if __name__=='__main__':
    main()
