#!/usr/bin/env python3
import json,subprocess,tempfile
from pathlib import Path

ROOT=Path('/tmp/pv0')
PLANS=Path('experiments/personal_vault_v0/blind_memory_resolution_v1.json')
ORACLE=Path('experiments/personal_vault_v0/blind_memory_oracle_v1.json')
BRIDGE='experiments/personal_vault_v0/ai_glyph_bridge_v1.py'


def run_bridge(plan):
    with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False,encoding='utf-8') as f:
        json.dump(plan,f,ensure_ascii=False,separators=(',',':'))
        p=f.name
    out=subprocess.check_output(['python3',BRIDGE,str(ROOT),'--plan',p],text=True)
    return json.loads(out)


def main():
    plans=json.loads(PLANS.read_text())['plans']
    oracle_doc=json.loads(ORACLE.read_text())
    oracle={x['id']:x for x in oracle_doc['cases']}
    results=[]; resolved=0; false_confident=0
    for p in plans:
        got=run_bridge({'id':p['id'],'human_query':p['human_clarification'],'probes':p['probes']})
        exp=oracle[p['id']]
        expected_action=exp['expected_action']
        expected_path=exp.get('expected_path')
        ok=False
        if expected_action=='found':
            ok=got['action']=='found' and got['selected_object'] and got['selected_object']['path']==expected_path
        elif expected_action=='not_found':
            ok=got['action']=='not_found'
        elif expected_action=='ambiguous':
            ok=got['action']=='ambiguous'
        if got['action']=='found' and not (expected_action=='found' and got['selected_object'] and got['selected_object']['path']==expected_path):
            false_confident+=1
        if ok: resolved+=1
        results.append({'id':p['id'],'first_pass':p['first_pass'],'second_action':got['action'],'selected_path':None if not got['selected_object'] else got['selected_object']['path'],'expected_action':expected_action,'expected_path':expected_path,'passed':ok})
    report={
      'format':'GLYPH_BLIND_MEMORY_RESOLUTION_GATE_V1',
      'frozen_first_pass_baseline':{'passed':43,'total':50,'accuracy':0.86,'false_confident_answer':0,'unresolved':7},
      'second_pass_cases':len(plans),'second_pass_resolved':resolved,
      'combined_resolved':43+resolved,'combined_total':50,'combined_resolution_rate':(43+resolved)/50,
      'false_confident_answer_second_pass':false_confident,
      'first_pass_artifacts_modified':False,
      'all_second_pass_cases_resolved':resolved==len(plans),
      'results':results,
      'important_non_claim':'Second-pass clarifications are externally authored. This measures whether dialogue refinement can resolve the frozen seven cases through GLYPH, not autonomous question generation by a local model.'
    }
    (ROOT/'blind-memory-resolution-gate.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False,sort_keys=True))
    assert false_confident==0,report
    assert resolved==len(plans),report

if __name__=='__main__':main()
