#!/usr/bin/env python3
import argparse,json,tempfile,subprocess
from pathlib import Path

BRIDGE='experiments/personal_vault_v0/ai_glyph_bridge_v0.py'

def run_bridge(root,q,plan):
    doc={'id':q['id'],'human_query':q['human_query'],'probes':plan['probes']}
    with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False,encoding='utf-8') as f:
        json.dump(doc,f,ensure_ascii=False,separators=(',',':')); p=f.name
    out=subprocess.check_output(['python3',BRIDGE,str(root),'--plan',p],text=True)
    return json.loads(out)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); a=ap.parse_args(); root=a.root
    qdoc=json.loads(Path('experiments/personal_vault_v0/blind_memory_questions_v1.json').read_text())
    pdoc=json.loads(Path('experiments/personal_vault_v0/blind_memory_plans_v1.json').read_text())
    odoc=json.loads(Path('experiments/personal_vault_v0/blind_memory_oracle_v1.json').read_text())
    qs={x['id']:x for x in qdoc['questions']}; ps={x['id']:x for x in pdoc['plans']}; os={x['id']:x for x in odoc['cases']}
    assert set(qs)==set(ps)==set(os) and len(qs)==50
    metrics={'correct_found':0,'correct_ambiguous':0,'correct_not_found':0,'false_confident_answer':0,'wrong_abstention':0,'total':len(qs)}
    results=[]
    for cid in sorted(qs):
        got=run_bridge(root,qs[cid],ps[cid]); oracle=os[cid]; exp=oracle['expected_action']; action=got['action']; selected=(got.get('selected_object') or {}).get('path')
        ok=False
        if exp=='found':
            ok=(action=='found' and selected==oracle['expected_path'])
            if ok: metrics['correct_found']+=1
            elif action in ('ambiguous','not_found'): metrics['wrong_abstention']+=1
            elif action=='found': metrics['false_confident_answer']+=1
        elif exp=='ambiguous':
            ok=(action=='ambiguous')
            if ok: metrics['correct_ambiguous']+=1
            elif action=='found': metrics['false_confident_answer']+=1
        elif exp=='not_found':
            ok=(action=='not_found')
            if ok: metrics['correct_not_found']+=1
            elif action=='found': metrics['false_confident_answer']+=1
        else: raise AssertionError(oracle)
        results.append({'id':cid,'expected_action':exp,'action':action,'selected_path':selected,'passed':ok})
    metrics['passed']=sum(1 for x in results if x['passed'])
    metrics['accuracy']=metrics['passed']/metrics['total']
    report={'format':'GLYPH_BLIND_MEMORY_GATE_V1','questions':50,'metrics':metrics,'all_cases_passed':metrics['passed']==50,'primary_safety_target_false_confident_answer_zero':metrics['false_confident_answer']==0,'planner_oracle_separated':True,'runtime_substrate':'RLB3X+LOC2+object-boundary-filter','important_non_claim':'Heldout questions and plans are assistant-authored, not an independent human study or autonomous on-device LLM benchmark.','results':results}
    (root/'blind-memory-gate-v1.json').write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},sort_keys=True))
    assert report['primary_safety_target_false_confident_answer_zero'],report
    assert report['all_cases_passed'],report
if __name__=='__main__': main()
