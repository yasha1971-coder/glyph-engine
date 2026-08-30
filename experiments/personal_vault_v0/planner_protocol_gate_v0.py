#!/usr/bin/env python3
import json,subprocess,tempfile
from pathlib import Path

ROOT=Path('/tmp/pv0')
PROTO='experiments/personal_vault_v0/planner_protocol_v0.py'
BRIDGE='experiments/personal_vault_v0/ai_glyph_bridge_v0.py'


def write_tmp(doc):
    f=tempfile.NamedTemporaryFile('w',suffix='.json',delete=False,encoding='utf-8')
    json.dump(doc,f,ensure_ascii=False,separators=(',',':')); f.close(); return Path(f.name)


def validate(mode,doc):
    p=write_tmp(doc)
    subprocess.check_call(['python3',PROTO,mode,str(p)],stdout=subprocess.DEVNULL)
    return p


def bridge(plan):
    p=write_tmp({'id':'planner-protocol-gate','human_query':plan['human_query'],'probes':plan['probes']})
    return json.loads(subprocess.check_output(['python3',BRIDGE,str(ROOT),'--plan',str(p)],text=True))

# Turn 1: planner sees only human words, no oracle.
input1={'format':'GLYPH_PLANNER_INPUT_V0','human_query':'Где старый C-код, в котором были printf и struct?','previous_state':None,'human_clarification':None}
output1={'format':'GLYPH_PLANNER_OUTPUT_V0','probes':['printf','struct'],'reason':'Use exact technical tokens explicitly remembered by the human.','stop_if_unverified':True}
validate('validate-input',input1); validate('validate-output',output1)
r1=bridge({'human_query':input1['human_query'],'probes':output1['probes']})
assert r1['action']=='ambiguous',r1

# Only evidence returned by GLYPH is carried into turn 2.
prev={'action':r1['action'],'candidate_paths':r1['clarification']['candidate_paths'],'probe_reports':r1['probe_reports']}
input2={'format':'GLYPH_PLANNER_INPUT_V0','human_query':input1['human_query'],'previous_state':prev,'human_clarification':'Я имел в виду файл про разбор полей строки.'}
output2={'format':'GLYPH_PLANNER_OUTPUT_V0','probes':['fieldread','fieldmake'],'reason':'The clarification narrows the intent to field parsing; use exact field-related identifiers as discriminators.','stop_if_unverified':True}
validate('validate-input',input2); validate('validate-output',output2)
r2=bridge({'human_query':input2['human_clarification'],'probes':output2['probes']})
assert r2['action']=='found',r2
assert r2['selected_object']['path']=='canterbury/fields.c',r2

# Negative protocol tests: oracle leakage and unsafe acceptance must fail validation.
negative=[]
for name,mode,doc in [
 ('input-oracle-leak','validate-input',{'format':'GLYPH_PLANNER_INPUT_V0','human_query':'x','previous_state':{'action':'found','candidate_paths':[],'probe_reports':[],'expected_path':'secret'},'human_clarification':None}),
 ('output-oracle-leak','validate-output',{'format':'GLYPH_PLANNER_OUTPUT_V0','probes':['x'],'reason':'oracle says expected_path','stop_if_unverified':True}),
 ('unsafe-stop-flag','validate-output',{'format':'GLYPH_PLANNER_OUTPUT_V0','probes':['x'],'reason':'guess','stop_if_unverified':False})
]:
    p=write_tmp(doc)
    rc=subprocess.run(['python3',PROTO,mode,str(p)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode
    assert rc!=0,(name,rc)
    negative.append(name)

report={'format':'GLYPH_PLANNER_PROTOCOL_GATE_V0','turn1_action':r1['action'],'turn2_action':r2['action'],'turn2_selected_path':r2['selected_object']['path'],'oracle_visible_to_planner':False,'stop_if_unverified_required':True,'negative_protocol_cases_rejected':negative,'all_checks_passed':True,'important_non_claim':'The planner outputs in this gate are replayed external-LLM outputs; CI validates the protocol and evidence boundary, not an on-device model invocation.'}
(ROOT/'planner-protocol-gate.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
print(json.dumps(report,ensure_ascii=False,sort_keys=True))
