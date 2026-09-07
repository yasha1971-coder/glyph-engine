#!/usr/bin/env python3
import argparse,json,subprocess,tempfile
from pathlib import Path


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--plans',type=Path,default=Path('experiments/personal_vault_v0/llm_planner_replay_v1.json'))
    ap.add_argument('--oracle',type=Path,default=Path('experiments/personal_vault_v0/human_intent_cases_v0.json'))
    a=ap.parse_args()

    plans=json.loads(a.plans.read_text())['plans']
    cases={x['id']:x for x in json.loads(a.oracle.read_text())['cases']}
    results=[]
    for plan in plans:
        with tempfile.NamedTemporaryFile('w',encoding='utf-8',suffix='.json',delete=False) as f:
            json.dump(plan,f,ensure_ascii=False)
            p=Path(f.name)
        try:
            out=subprocess.check_output([
                'python3','experiments/personal_vault_v0/ai_glyph_bridge_v0.py',str(a.root),'--plan',str(p)
            ],text=True)
        finally:
            p.unlink(missing_ok=True)
        got=json.loads(out)
        case=cases[plan['id']]
        expected_action=case.get('expected_action','found')
        expected_path=case.get('expected_path')
        assert got['contract']['oracle_not_used'] is True,(plan['id'],got)
        assert got['action']==expected_action,(plan['id'],expected_action,got['action'],got['ranked_candidates'])
        if expected_action=='found':
            assert got['selected_object'] is not None,(plan['id'],got)
            assert got['selected_object']['path']==expected_path,(plan['id'],expected_path,got['selected_object'])
        else:
            assert got['selected_object'] is None,(plan['id'],got)
        if expected_action=='ambiguous':
            assert got['clarification'] and got['clarification']['needed'] is True,(plan['id'],got)
        if expected_action=='not_found':
            assert got['ranked_candidates']==[],(plan['id'],got)
        results.append({'id':plan['id'],'action':got['action'],'selected_object':got['selected_object'],'passed':True})

    report={
        'format':'GLYPH_AI_GLYPH_BRIDGE_GATE_V0',
        'cases':len(results),'cases_passed':len(results),'all_cases_passed':True,
        'bridge_oracle_free':True,
        'planner_source':'external LLM replay artifact without expected answers',
        'retrieval_substrate':'RLB3X+LOC2+object-boundary-filter',
        'results':results,
    }
    (a.root/'ai-glyph-bridge-gate.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},sort_keys=True))

if __name__=='__main__': main()
