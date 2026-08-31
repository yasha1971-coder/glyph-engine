#!/usr/bin/env python3
import argparse,json,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
SCORER=HERE/'evidence_scoring_v4.py'

CASES=[
  {'id':'alice','query':'Найди книгу про Alice, Чеширского кота и Wonderland.','action':'found','path':'canterbury/alice29.txt'},
  {'id':'paradise','query':'Найди Paradise Lost Джона Мильтона.','action':'ambiguous','path':None},
  {'id':'xargs','query':'Найди старую man-страницу команды xargs.','action':'found','path':'canterbury/xargs.1'},
  {'id':'missing','query':'Найди документ с точным маркером GLYPH_LOCAL_LLM_NEVER_PRESENT_81C4.','action':'not_found','path':None},
  {'id':'c_vague','query':'Где старый C-код, в котором были printf и struct?','action':'ambiguous','path':None},
  {'id':'c_clarified','query':'Где старый C-код, в котором были printf и struct?','clar':'Я имел в виду файл про разбор полей строки; помню fieldread и fieldmake.','action':'found','path':'canterbury/fields.c'},
]

def call(root,q,clar=None):
    cmd=['python3',SCORER,root,'--query',q]
    if clar is not None: cmd += ['--clarification',clar]
    return json.loads(subprocess.check_output([str(x) for x in cmd],text=True))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path)
    a=ap.parse_args(); results=[]; false_confident=0
    for c in CASES:
        r=call(a.root,c['query'],c.get('clar'))
        ok=r['action']==c['action'] and (c['path'] is None or r.get('selected_path')==c['path'])
        if r['action']=='found' and not ok: false_confident+=1
        results.append({'id':c['id'],'action':r['action'],'selected_path':r.get('selected_path'),'margin_bits':r.get('margin_bits'),'ranked':r.get('ranked',[])[:3],'passed':ok})
    passed=sum(1 for x in results if x['passed'])
    report={
      'format':'GLYPH_EVIDENCE_SCORING_GATE_V4',
      'architecture':'deterministic IDF-like object evidence scoring; no LLM',
      'cases':len(results),
      'passed':passed,
      'false_confident_answers':false_confident,
      'all_checks_passed':passed==len(results) and false_confident==0,
      'results':results,
      'important_non_claim':'Thresholds are provisional and corpus-scale dependent. This gate proves only that common-word intersections cannot directly create FOUND on the frozen 30-object fixture.'
    }
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
    if not report['all_checks_passed']: raise SystemExit(1)

if __name__=='__main__': main()
