#!/usr/bin/env python3
import json,subprocess,tempfile
from pathlib import Path

ROOT=Path('/tmp/pv0')
BRIDGE='experiments/personal_vault_v0/ai_glyph_bridge_v0.py'

def run_plan(plan):
    with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False,encoding='utf-8') as f:
        json.dump(plan,f,ensure_ascii=False,separators=(',',':'))
        path=f.name
    out=subprocess.check_output(['python3',BRIDGE,str(ROOT),'--plan',path],text=True)
    return json.loads(out)

first={
 'id':'dialogue-c-source-turn1',
 'human_query':'Где старый C-код, в котором были printf и struct?',
 'probes':['printf','struct']
}
r1=run_plan(first)
assert r1['action']=='ambiguous',r1
assert r1['clarification'] and r1['clarification']['needed'] is True,r1
assert len(r1['clarification']['candidate_paths'])>=2,r1

second={
 'id':'dialogue-c-source-turn2',
 'human_query':'Я имел в виду файл про разбор полей строки.',
 'probes':['fieldread','fieldmake']
}
r2=run_plan(second)
assert r2['action']=='found',r2
assert r2['selected_object']['path']=='canterbury/fields.c',r2

report={
 'format':'GLYPH_DIALOGUE_CLARIFICATION_GATE_V0',
 'turns':2,
 'turn1_action':r1['action'],
 'turn1_candidates':r1['clarification']['candidate_paths'],
 'turn2_action':r2['action'],
 'turn2_selected_path':r2['selected_object']['path'],
 'oracle_used_inside_bridge':False,
 'human_refinement_required':True,
 'all_checks_passed':True,
 'important_non_claim':'The clarification probes are still externally supplied; this gate validates dialogue state transition, not autonomous generation of the follow-up probes.'
}
(ROOT/'dialogue-clarification-gate.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps(report,ensure_ascii=False,sort_keys=True))
