#!/usr/bin/env python3
import argparse,json,sys
from pathlib import Path

ALLOWED_INPUT_KEYS={'format','human_query','previous_state','human_clarification'}
ALLOWED_OUTPUT_KEYS={'format','probes','reason','stop_if_unverified'}


def fail(msg):
    raise SystemExit(msg)


def validate_input(doc):
    if doc.get('format')!='GLYPH_PLANNER_INPUT_V0': fail('bad planner input format')
    if set(doc)-ALLOWED_INPUT_KEYS: fail('planner input contains forbidden keys')
    if not isinstance(doc.get('human_query'),str) or not doc['human_query']: fail('human_query required')
    prev=doc.get('previous_state')
    if prev is not None:
        if not isinstance(prev,dict): fail('previous_state must be object')
        allowed={'action','candidate_paths','probe_reports'}
        if set(prev)-allowed: fail('previous_state contains forbidden keys')
        if prev.get('action') not in ('ambiguous','not_found','found'): fail('bad previous action')
        if prev.get('action')=='ambiguous' and not isinstance(prev.get('candidate_paths'),list): fail('ambiguous state needs candidate_paths')
    hc=doc.get('human_clarification')
    if hc is not None and (not isinstance(hc,str) or not hc): fail('bad clarification')
    forbidden_text=('expected_path','expected_action','oracle','golden','correct_file')
    blob=json.dumps(doc,ensure_ascii=False).lower()
    if any(x in blob for x in forbidden_text): fail('oracle-like token in planner input')


def validate_output(doc):
    if doc.get('format')!='GLYPH_PLANNER_OUTPUT_V0': fail('bad planner output format')
    if set(doc)-ALLOWED_OUTPUT_KEYS: fail('planner output contains forbidden keys')
    probes=doc.get('probes')
    if not isinstance(probes,list) or not (1<=len(probes)<=8): fail('planner output needs 1..8 probes')
    for p in probes:
        if not isinstance(p,str) or not p or len(p.encode('utf-8'))>256: fail('bad probe')
    if not isinstance(doc.get('reason'),str) or not doc['reason']: fail('reason required')
    if doc.get('stop_if_unverified') is not True: fail('stop_if_unverified must be true')
    blob=json.dumps(doc,ensure_ascii=False).lower()
    if 'expected_path' in blob or 'oracle' in blob: fail('oracle-like token in planner output')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('mode',choices=['validate-input','validate-output'])
    ap.add_argument('path',type=Path)
    a=ap.parse_args(); doc=json.loads(a.path.read_text())
    validate_input(doc) if a.mode=='validate-input' else validate_output(doc)
    print(json.dumps({'format':'GLYPH_PLANNER_PROTOCOL_VALIDATION_V0','mode':a.mode,'valid':True},sort_keys=True))

if __name__=='__main__': main()
