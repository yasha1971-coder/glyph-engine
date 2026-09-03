#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROUTER=HERE/'local_cognitive_router_v3.py'

CASES=[
  {'id':'alice','query':'Найди книгу про Alice, Чеширского кота и Wonderland.','action':'found','path':'canterbury/alice29.txt'},
  {'id':'paradise','query':'Найди Paradise Lost Джона Мильтона.','action':'found','path':'canterbury/plrabn12.txt'},
  {'id':'xargs','query':'Найди старую man-страницу команды xargs.','action':'found','path':'canterbury/xargs.1'},
  {'id':'missing','query':'Найди документ с точным маркером GLYPH_LOCAL_LLM_NEVER_PRESENT_81C4.','action':'not_found','path':None},
]

def call(root,llama,model,q,clar=None):
    cmd=['python3',ROUTER,root,'--llama-cli',llama,'--model',model,'--query',q]
    if clar is not None: cmd += ['--clarification',clar]
    return json.loads(subprocess.check_output([str(x) for x in cmd],text=True))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--llama-cli',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--model-label',required=True)
    a=ap.parse_args(); results=[]; false_confident=0
    for c in CASES:
        r=call(a.root,a.llama_cli,a.model,c['query'])
        ok=r['action']==c['action'] and (c['path'] is None or r.get('selected_path')==c['path'])
        if r['action']=='found' and not ok: false_confident+=1
        results.append({'id':c['id'],'action':r['action'],'selected_path':r.get('selected_path'),'literal_stage':r['literal_stage'],'semantic_stage':r['semantic_stage'],'passed':ok})
    q='Где старый C-код, в котором были printf и struct?'
    r1=call(a.root,a.llama_cli,a.model,q)
    safe=r1['action'] in ('ambiguous','partial','not_found')
    if r1['action']=='found': false_confident+=1
    r2=call(a.root,a.llama_cli,a.model,q,'Я имел в виду файл про разбор полей строки; помню fieldread и fieldmake.')
    ok2=r2['action']=='found' and r2.get('selected_path')=='canterbury/fields.c'
    if r2['action']=='found' and not ok2: false_confident+=1
    results.append({'id':'dialogue','turn1_action':r1['action'],'turn2_action':r2['action'],'turn2_selected_path':r2.get('selected_path'),'turn1':r1,'turn2':r2,'passed':safe and ok2})
    passed=sum(1 for x in results if x['passed'])
    report={
      'format':'GLYPH_LOCAL_COGNITIVE_ROUTER_GATE_V3',
      'model':a.model_label,
      'architecture':'H0 human memory -> H1 deterministic literal evidence -> GLYPH -> H2 constrained semantic hypotheses -> GLYPH',
      'runtime_llm_used':True,
      'model_can_create_h2_hypotheses':True,
      'model_can_create_found_without_h1_evidence':False,
      'false_confident_answers':false_confident,
      'cases':len(results),
      'passed':passed,
      'all_checks_passed':passed==len(results) and false_confident==0,
      'results':results,
      'important_non_claim':'V3 does not yet solve arbitrary semantic recall. The model is used only after exact human-literal evidence is gathered and may only narrow that verified candidate set.'
    }
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
    if not report['all_checks_passed']: raise SystemExit(1)

if __name__=='__main__': main()
