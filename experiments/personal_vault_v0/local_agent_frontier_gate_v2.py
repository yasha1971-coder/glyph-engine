#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
AGENT=HERE/'local_agent_frontier_v2.py'

CASES=[
  {'id':'alice','query':'Найди книгу про Alice, Чеширского кота и Wonderland.','action':'found','path':'canterbury/alice29.txt'},
  {'id':'paradise','query':'Найди Paradise Lost Джона Мильтона.','action':'found','path':'canterbury/plrabn12.txt'},
  {'id':'xargs','query':'Найди старую man-страницу команды xargs.','action':'found','path':'canterbury/xargs.1'},
  {'id':'missing','query':'Найди документ с точным маркером GLYPH_LOCAL_LLM_NEVER_PRESENT_81C4.','action':'not_found','path':None},
]


def call(root,llama,model,query,clarification=None):
    cmd=['python3',AGENT,root,'--llama-cli',llama,'--model',model,'--query',query]
    if clarification is not None: cmd += ['--clarification',clarification]
    return json.loads(subprocess.check_output([str(x) for x in cmd],text=True))


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--llama-cli',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--model-label',required=True)
    a=ap.parse_args(); results=[]; false_confident=0
    for c in CASES:
        got=call(a.root,a.llama_cli,a.model,c['query'])
        ok=got['action']==c['action'] and (c['path'] is None or got.get('selected_path')==c['path'])
        if got['action']=='found' and not ok: false_confident+=1
        results.append({'id':c['id'],'action':got['action'],'selected_path':got.get('selected_path'),'history':got['history'],'passed':ok})

    # The first vague C query should remain unresolved or ambiguous without invented probes.
    q='Где старый C-код, в котором были printf и struct?'
    first=call(a.root,a.llama_cli,a.model,q)
    if first['action']=='found':
        false_confident+=1
    turn1_safe=first['action'] in ('ambiguous','partial','not_found')
    second=call(a.root,a.llama_cli,a.model,q,'Я имел в виду файл про разбор полей строки; помню fieldread и fieldmake.')
    turn2_ok=second['action']=='found' and second.get('selected_path')=='canterbury/fields.c'
    if second['action']=='found' and not turn2_ok: false_confident+=1
    results.append({'id':'dialogue','turn1_action':first['action'],'turn1_history':first['history'],'turn2_action':second['action'],'turn2_selected_path':second.get('selected_path'),'turn2_history':second['history'],'passed':turn1_safe and turn2_ok})

    passed=sum(1 for x in results if x['passed'])
    report={
      'format':'GLYPH_LOCAL_AGENT_FRONTIER_GATE_V2',
      'model':a.model_label,
      'architecture':'constrained_json + human-extractive candidate set + iterative exact GLYPH tool calls',
      'runtime_llm_used':True,
      'assistant_authored_runtime_probes':False,
      'model_can_invent_probe':False,
      'oracle_visible_to_model':False,
      'cases':len(results),
      'passed':passed,
      'false_confident_answers':false_confident,
      'all_checks_passed':passed==len(results) and false_confident==0,
      'results':results,
      'important_non_claim':'This tests an edge-agent architecture, not unrestricted semantic expansion. Model choices are constrained to literal candidates extracted from the human query/clarification; GLYPH remains the exact evidence authority.'
    }
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
    if not report['all_checks_passed']: raise SystemExit(1)

if __name__=='__main__': main()
