#!/usr/bin/env python3
import argparse,collections,json,subprocess,sys
from pathlib import Path

HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
import glyph_vault_cli_v0 as vaultcli

QUERY=HERE/'query_rlb3x_object.py'


def validate_plan(plan):
    probes=plan.get('probes') or []
    if not isinstance(probes,list) or not (1<=len(probes)<=8):
        raise SystemExit('plan.probes must contain 1..8 strings')
    for p in probes:
        if not isinstance(p,str) or not p or len(p.encode('utf-8'))>256:
            raise SystemExit('invalid probe')
    return probes


def query_segment(seg,probe):
    out=subprocess.check_output([
        'python3',str(QUERY),
        '--rlb3x',str(seg/'bwt.rlb3x'),
        '--locate-core',str(seg/'locate.loc2'),
        '--objects',str(seg/'objects.json'),
        '--pattern-hex',probe.encode('utf-8').hex(),
    ],text=True)
    got=json.loads(out)
    if got.get('rlb2_not_used') is not True:
        raise SystemExit('global bridge requires RLB3X path')
    return got


def evaluate(vault,plan):
    probes=validate_plan(plan)
    root_report=vaultcli.verify_root(vault)
    if not root_report.get('root_present'):
        raise SystemExit('vault has no committed root')
    segs=vaultcli.iter_segments(vault)

    scores=collections.Counter()
    evidence=collections.defaultdict(list)
    probe_reports=[]

    for probe in probes:
        matched_global=set(); segment_reports=[]
        for seg in segs:
            got=query_segment(seg,probe)
            matched_here=set()
            for hit in got['valid_hits']:
                key=(seg.name,hit['object_id'],hit['path'])
                matched_here.add(key); matched_global.add(key)
                evidence[key].append({
                    'probe':probe,
                    'segment_id':seg.name,
                    'object_offset':hit['object_offset'],
                    'segment_corpus_offset':hit['corpus_offset'],
                    'pattern_bytes':hit['pattern_bytes'],
                })
            segment_reports.append({
                'segment_id':seg.name,
                'raw_count':got['raw_count'],
                'valid_count':got['valid_count'],
                'matched_objects':sorted({k[2] for k in matched_here}),
            })
        for key in matched_global:
            scores[key]+=1
        probe_reports.append({
            'probe':probe,
            'matched_object_versions':len(matched_global),
            'segments':segment_reports,
        })

    ranked=sorted(scores.items(),key=lambda kv:(-kv[1],kv[0][2],kv[0][0],kv[0][1]))
    fully=[key for key,score in ranked if score==len(probes)]
    if not ranked:
        action='not_found'; selected=None
    elif not fully:
        action='partial'; selected=None
    elif len(fully)==1:
        action='found'; selected=fully[0]
    else:
        action='ambiguous'; selected=None

    ranked_out=[]
    for (sid,oid,path),score in ranked[:12]:
        ranked_out.append({
            'segment_id':sid,'object_id':oid,'path':path,
            'matched_probes':score,'probe_count':len(probes),
            'score_fraction':score/len(probes),
            'fully_supported':score==len(probes),
            'evidence':evidence[(sid,oid,path)],
        })

    clarification=None
    if action in ('ambiguous','partial'):
        clarification={
            'needed':True,
            'reason':(
                'multiple fully-supported object versions remain; ask for a distinguishing fact or version clue'
                if action=='ambiguous' else
                'some exact evidence exists but no committed object version supports the whole plan'
            ),
            'candidates':[
                {'segment_id':x['segment_id'],'path':x['path'],'matched_probes':x['matched_probes']}
                for x in ranked_out[:6]
            ],
        }

    return {
        'format':'GLYPH_AI_GLYPH_VAULT_BRIDGE_V0',
        'human_query':plan.get('human_query',''),
        'plan_id':plan.get('id'),
        'probes':probes,
        'vault_root':{
            'name':root_report['root_name'],
            'sha256':root_report['root_sha256'],
            'segment_count':len(segs),
            'root_binding_verified':True,
            'root_parent_binding_verified':root_report['parent_binding_verified'],
        },
        'action':action,
        'selected_object':None if selected is None else {
            'segment_id':selected[0],'object_id':selected[1],'path':selected[2]
        },
        'ranked_candidates':ranked_out,
        'probe_reports':probe_reports,
        'clarification':clarification,
        'contract':{
            'segment_fanout_hidden_from_ai':True,
            'only_latest_committed_root_searched':True,
            'object_version_identity':['segment_id','object_id','path'],
            'states':['found','ambiguous','partial','not_found'],
            'no_answer_without_full_unique_evidence':True,
            'oracle_not_used':True,
            'rlb2_not_used':True,
        }
    }


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('vault',type=Path)
    ap.add_argument('--plan',type=Path,required=True)
    ap.add_argument('--out',type=Path)
    a=ap.parse_args()
    plan=json.loads(a.plan.read_text())
    result=evaluate(a.vault,plan)
    text=json.dumps(result,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n'
    if a.out: a.out.write_text(text)
    print(text,end='')

if __name__=='__main__': main()
