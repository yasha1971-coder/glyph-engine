#!/usr/bin/env python3
import argparse,collections,json,subprocess
from pathlib import Path


def glyph_query(root,probe):
    out=subprocess.check_output([
        'python3','experiments/personal_vault_v0/query_rlb3x_object.py',
        '--rlb3x',str(root/'bwt.rlb3x'),
        '--locate-core',str(root/'locate.loc2'),
        '--objects',str(root/'objects.json'),
        '--pattern-hex',probe.encode('utf-8').hex(),
    ],text=True)
    got=json.loads(out)
    if got.get('rlb2_not_used') is not True:
        raise SystemExit('bridge requires RLB3X query path')
    return got


def evaluate(root,plan):
    probes=plan.get('probes') or []
    if not isinstance(probes,list) or not probes or len(probes)>8:
        raise SystemExit('plan.probes must contain 1..8 strings')
    if any((not isinstance(x,str) or not x or len(x.encode('utf-8'))>256) for x in probes):
        raise SystemExit('invalid probe')

    scores=collections.Counter(); evidence=collections.defaultdict(list); probe_reports=[]
    for probe in probes:
        got=glyph_query(root,probe)
        object_keys=set()
        for hit in got['valid_hits']:
            key=(hit['object_id'],hit['path'])
            object_keys.add(key)
            evidence[key].append({
                'probe':probe,
                'object_offset':hit['object_offset'],
                'corpus_offset':hit['corpus_offset'],
                'pattern_bytes':hit['pattern_bytes'],
            })
        for key in object_keys: scores[key]+=1
        probe_reports.append({
            'probe':probe,
            'raw_count':got['raw_count'],
            'valid_count':got['valid_count'],
            'matched_objects':sorted({p for _,p in object_keys}),
        })

    ranked=sorted(scores.items(),key=lambda kv:(-kv[1],kv[0][1],kv[0][0]))
    top_score=ranked[0][1] if ranked else 0
    top=[key for key,score in ranked if score==top_score]
    if not ranked:
        action='not_found'; selected=None
    elif len(top)==1 and top_score==len(probes):
        action='found'; selected=top[0]
    else:
        action='ambiguous'; selected=None

    ranked_out=[]
    for (oid,path),score in ranked[:8]:
        ranked_out.append({
            'object_id':oid,'path':path,'matched_probes':score,'probe_count':len(probes),
            'score_fraction':score/len(probes),'evidence':evidence[(oid,path)],
        })

    clarification=None
    if action=='ambiguous':
        clarification={
            'needed':True,
            'reason':'multiple or partial candidates remain; AI should ask the human for one more distinguishing fact instead of guessing',
            'candidate_paths':[x['path'] for x in ranked_out[:5]],
            'matched_probe_summary':[{ 'path':x['path'],'matched_probes':x['matched_probes']} for x in ranked_out[:5]],
        }

    return {
        'format':'GLYPH_AI_GLYPH_BRIDGE_V0',
        'human_query':plan.get('human_query',''),
        'plan_id':plan.get('id'),
        'probes':probes,
        'action':action,
        'selected_object':None if selected is None else {'object_id':selected[0],'path':selected[1]},
        'ranked_candidates':ranked_out,
        'probe_reports':probe_reports,
        'clarification':clarification,
        'contract':{
            'ai_role':'generate search probes and explain/clarify',
            'glyph_role':'exact-byte evidence and object coordinates',
            'no_answer_without_full_unique_evidence':True,
            'oracle_not_used':True,
        }
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('root',type=Path)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    plan=json.loads(a.plan.read_text())
    result=evaluate(a.root,plan)
    text=json.dumps(result,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n'
    if a.out: a.out.write_text(text)
    print(text,end='')

if __name__=='__main__': main()
