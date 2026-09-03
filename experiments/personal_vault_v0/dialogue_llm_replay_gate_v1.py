#!/usr/bin/env python3
import json,subprocess,tempfile
from pathlib import Path

ROOT=Path('/tmp/pv0')
BRIDGE='experiments/personal_vault_v0/ai_glyph_bridge_v0.py'
REPLAY=Path('experiments/personal_vault_v0/dialogue_llm_replay_v1.json')


def run_bridge(plan):
    with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False,encoding='utf-8') as f:
        json.dump(plan,f,ensure_ascii=False,separators=(',',':'))
        p=f.name
    out=subprocess.check_output(['python3',BRIDGE,str(ROOT),'--plan',p],text=True)
    return json.loads(out)


def main():
    doc=json.loads(REPLAY.read_text())
    assert doc['planner']['oracle_visible'] is False,doc['planner']
    results=[]
    for d in doc['dialogues']:
        p1={'id':d['id']+'-turn1','human_query':d['turn1']['human_query'],'probes':d['turn1']['probes']}
        r1=run_bridge(p1)
        assert r1['action']==d['expected_transition_after_turn1'],(d,r1)
        assert r1['action']=='ambiguous' and r1['clarification']['needed'] is True,(d,r1)

        # turn2 uses only the human clarification + externally generated probes;
        # no expected object/path is passed into the bridge.
        p2={'id':d['id']+'-turn2','human_query':d['turn2']['human_clarification'],'probes':d['turn2']['probes']}
        r2=run_bridge(p2)
        assert r2['action']=='found',(d,r2)

        # independent oracle check only after the bridge has selected an object.
        assert r2['selected_object']['path']=='canterbury/fields.c',(d,r2)
        results.append({
            'id':d['id'],
            'turn1_action':r1['action'],
            'turn1_candidates':r1['clarification']['candidate_paths'],
            'turn2_action':r2['action'],
            'turn2_selected_path':r2['selected_object']['path'],
            'turn2_probes':d['turn2']['probes'],
        })

    report={
      'format':'GLYPH_DIALOGUE_LLM_REPLAY_GATE_V1',
      'dialogues':len(results),
      'dialogues_passed':len(results),
      'all_dialogues_passed':True,
      'oracle_visible_to_bridge':False,
      'turn2_planner_source':'external LLM replay artifact',
      'retrieval_substrate':'RLB3X+LOC2+object-boundary-filter',
      'important_non_claim':'This validates an externally generated second-turn plan replay. It does not yet run an autonomous local/on-device LLM inside CI.',
      'results':results,
    }
    (ROOT/'dialogue-llm-replay-gate.json').write_text(json.dumps(report,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='results'},ensure_ascii=False,sort_keys=True))

if __name__=='__main__': main()
