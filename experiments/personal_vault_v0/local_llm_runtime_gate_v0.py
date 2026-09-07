#!/usr/bin/env python3
import argparse,json,subprocess,tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
PLANNER=HERE/'local_llm_planner_v0.py'
PROTOCOL=HERE/'planner_protocol_v0.py'
BRIDGE=HERE/'ai_glyph_bridge_v1.py'

CASES=[
  {'id':'alice','human_query':'Найди книгу про Алису, Чеширского кота и Wonderland.','expected_action':'found','expected_path':'canterbury/alice29.txt'},
  {'id':'paradise','human_query':'Найди Paradise Lost Джона Мильтона.','expected_action':'found','expected_path':'canterbury/plrabn12.txt'},
  {'id':'xargs','human_query':'Найди старую man-страницу команды xargs.','expected_action':'found','expected_path':'canterbury/xargs.1'},
  {'id':'missing','human_query':'Найди документ с точным маркером GLYPH_LOCAL_LLM_NEVER_PRESENT_81C4.','expected_action':'not_found','expected_path':None},
]


def run_json(cmd):
    return json.loads(subprocess.check_output([str(x) for x in cmd],text=True))


def make_input(path,human_query,previous_state=None,human_clarification=None):
    doc={'format':'GLYPH_PLANNER_INPUT_V0','human_query':human_query,'previous_state':previous_state,'human_clarification':human_clarification}
    path.write_text(json.dumps(doc,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')


def local_plan(args,inp,out):
    cmd=['python3',PLANNER,'--llama-cli',args.llama_cli,'--model',args.model,'--input',inp,'--output',out]
    if args.hf: cmd.append('--hf')
    subprocess.check_call([str(x) for x in cmd])
    subprocess.check_call(['python3',PROTOCOL,'validate-output',out])
    return json.loads(out.read_text())


def bridge(root,plan_path):
    return run_json(['python3',BRIDGE,root,'--plan',plan_path])


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--llama-cli',type=Path,required=True)
    ap.add_argument('--model',required=True)
    ap.add_argument('--hf',action='store_true')
    a=ap.parse_args()

    results=[]; false_confident=0
    with tempfile.TemporaryDirectory(prefix='glyph-local-llm-gate-') as td:
        t=Path(td)
        for case in CASES:
            inp=t/(case['id']+'-input.json'); out=t/(case['id']+'-plan.json')
            make_input(inp,case['human_query'])
            plan=local_plan(a,inp,out)
            got=bridge(a.root,out)
            passed=(got['action']==case['expected_action'] and (case['expected_path'] is None or (got['selected_object'] or {}).get('path')==case['expected_path']))
            if got['action']=='found' and not passed: false_confident+=1
            results.append({'id':case['id'],'probes':plan['probes'],'action':got['action'],'selected_path':None if got['selected_object'] is None else got['selected_object']['path'],'passed':passed})

        # Autonomous two-turn dialogue: the model sees ambiguity and the human clarification,
        # but never sees the expected file path or oracle.
        q='Где старый C-код, в котором были printf и struct?'
        inp1=t/'dialogue-1-input.json'; plan1=t/'dialogue-1-plan.json'
        make_input(inp1,q); p1=local_plan(a,inp1,plan1); r1=bridge(a.root,plan1)
        if r1['action']!='ambiguous':
            if r1['action']=='found': false_confident+=1
            raise SystemExit('turn1 must remain ambiguous: '+json.dumps({'plan':p1,'result':r1},ensure_ascii=False))
        previous={'action':r1['action'],'candidate_paths':[x['path'] for x in r1['ranked_candidates'][:6]],'probe_reports':r1['probe_reports']}
        inp2=t/'dialogue-2-input.json'; plan2=t/'dialogue-2-plan.json'
        make_input(inp2,q,previous_state=previous,human_clarification='Я имел в виду файл про разбор полей строки; помню названия fieldread и fieldmake.')
        p2=local_plan(a,inp2,plan2); r2=bridge(a.root,plan2)
        dialogue_ok=r2['action']=='found' and (r2['selected_object'] or {}).get('path')=='canterbury/fields.c'
        if r2['action']=='found' and not dialogue_ok: false_confident+=1
        results.append({'id':'dialogue-turn2','turn1_probes':p1['probes'],'turn2_probes':p2['probes'],'turn1_action':r1['action'],'action':r2['action'],'selected_path':None if r2['selected_object'] is None else r2['selected_object']['path'],'passed':dialogue_ok})

    passed=sum(1 for x in results if x['passed'])
    report={
      'format':'GLYPH_LOCAL_LLM_RUNTIME_GATE_V0',
      'planner_kind':'autonomous_local_gguf_llm_via_llama_cpp',
      'model':a.model,
      'runtime_llm_used':True,
      'assistant_authored_runtime_probes':False,
      'oracle_visible_to_llm':False,
      'cases':len(results),
      'passed':passed,
      'false_confident_answers':false_confident,
      'all_checks_passed':passed==len(results) and false_confident==0,
      'results':results,
      'important_non_claim':'This proves a real local GGUF LLM can generate GLYPH probes at runtime on the CI host. It is not yet an iPhone build and does not claim this small model is sufficient for arbitrary real personal data.'
    }
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
    if not report['all_checks_passed']: raise SystemExit(1)

if __name__=='__main__': main()
