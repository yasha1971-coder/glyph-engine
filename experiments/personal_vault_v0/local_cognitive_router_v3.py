#!/usr/bin/env python3
import argparse,json,re,subprocess
from pathlib import Path

HERE=Path(__file__).resolve().parent
QUERY=HERE/'query_rlb3x_object.py'
ASCII_RUN_RE=re.compile(r'[A-Za-z0-9_.+/-]+(?:[ \t]+[A-Za-z0-9_.+/-]+){0,2}')
TOKEN_RE=re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.+/-]*')


def literal_candidates(query,clarification=None):
    text=query + (' ' + clarification if clarification else '')
    out=[]
    def add(x):
        x=' '.join(x.strip(' .,:;!?()[]{}"\'').split())
        if len(x)>=2 and x not in out: out.append(x)
    # Preserve exact marker-like tokens first.
    for t in TOKEN_RE.findall(text):
        if '_' in t or any(c.isdigit() for c in t): add(t)
    # Add contiguous ASCII runs as phrases, then their tokens.
    for m in ASCII_RUN_RE.finditer(text):
        run=' '.join(m.group(0).split())
        add(run)
        parts=run.split()
        if len(parts)>=2:
            for n in (2,):
                for i in range(len(parts)-n+1): add(' '.join(parts[i:i+n]))
        for p in parts: add(p)
    return out[:20]


def is_exact_marker_intent(query,clarification=None):
    text=query + (' ' + clarification if clarification else '')
    toks=TOKEN_RE.findall(text)
    return any('_' in t and any(c.isdigit() for c in t) and len(t)>=8 for t in toks)


def query_glyph(root,probe):
    out=subprocess.check_output([
        'python3',str(QUERY),
        '--rlb3x',str(root/'bwt.rlb3x'),
        '--locate-core',str(root/'locate.loc2'),
        '--objects',str(root/'objects.json'),
        '--pattern-hex',probe.encode('utf-8').hex(),
    ],text=True)
    got=json.loads(out)
    return {
        'probe':probe,
        'raw_count':got['raw_count'],
        'valid_count':got['valid_count'],
        'paths':sorted({h['path'] for h in got['valid_hits']}),
    }


def run_literal_stage(root,query,clarification=None):
    cands=literal_candidates(query,clarification)
    evidence=[]
    for probe in cands:
        r=query_glyph(root,probe); r['source']='H1_human_literal'; evidence.append(r)
    nonempty=[x for x in evidence if x['paths']]
    if not nonempty:
        return {'action':'not_found' if is_exact_marker_intent(query,clarification) else 'unresolved','selected_path':None,'candidate_paths':[],'evidence':evidence,'candidates':cands}
    # Start from the most selective literal evidence; intersect only evidence that remains compatible.
    ordered=sorted(nonempty,key=lambda x:(len(x['paths']),-len(x['probe'])))
    current=set(ordered[0]['paths'])
    for e in ordered[1:]:
        overlap=current & set(e['paths'])
        if overlap: current=overlap
    if len(current)==1:
        return {'action':'found','selected_path':next(iter(current)),'candidate_paths':sorted(current),'evidence':evidence,'candidates':cands}
    return {'action':'ambiguous','selected_path':None,'candidate_paths':sorted(current),'evidence':evidence,'candidates':cands}


def semantic_hypotheses(llama,model,query,clarification,literal_result):
    schema={
      'type':'object','additionalProperties':False,
      'properties':{
        'probes':{'type':'array','minItems':1,'maxItems':3,'items':{'type':'string'}},
        'reason':{'type':'string'}
      },
      'required':['probes','reason']
    }
    summary=[{'probe':x['probe'],'paths':x['paths']} for x in literal_result['evidence'] if x['paths']]
    prompt=f'''You are the semantic fallback router for GLYPH, a byte-exact personal memory vault.
The deterministic literal stage could not uniquely identify one object.
Generate 1 to 3 SHORT English literal hypotheses that are likely to appear exactly in the sought document and can distinguish the remaining candidates.
You may translate/transliterate a remembered name or concept into the likely original-language literal, but do not invent unrelated facts.
Do not answer the human and do not name a file. GLYPH will verify every hypothesis exactly; unverified hypotheses are never treated as evidence.

Human query: {query}
Human clarification: {clarification or 'null'}
Current candidate paths: {json.dumps(literal_result['candidate_paths'],ensure_ascii=False)}
Verified human-literal evidence: {json.dumps(summary,ensure_ascii=False)}
Return only the constrained JSON object.'''
    cmd=[str(llama),'--jinja','--single-turn','--no-display-prompt','-n','96','--temp','0','-c','2048','-j',json.dumps(schema,separators=(',',':')),'-p',prompt,'-m',str(model)]
    try:
        p=subprocess.run(cmd,text=True,capture_output=True,timeout=120)
    except subprocess.TimeoutExpired as e:
        raise SystemExit('semantic fallback model invocation exceeded 120 s') from e
    if p.returncode!=0: raise SystemExit('llama-completion failed: '+p.stderr[-3000:])
    text=p.stdout.strip(); start=text.find('{'); end=text.rfind('}')
    if start<0 or end<start: raise SystemExit('constrained model emitted no JSON: '+text[-1000:])
    doc=json.loads(text[start:end+1])
    probes=[]
    for x in doc['probes']:
        x=' '.join(str(x).strip().split())
        if x and len(x.encode('utf-8'))<=128 and x not in probes: probes.append(x)
    return probes,doc.get('reason','')


def run_router(root,llama,model,query,clarification=None):
    literal=run_literal_stage(root,query,clarification)
    report={'architecture':'H0 human memory -> H1 deterministic literals -> GLYPH -> H2 constrained semantic hypotheses -> GLYPH','literal_stage':literal,'semantic_stage':None}
    if literal['action'] in ('found','not_found'):
        report.update({'action':literal['action'],'selected_path':literal['selected_path']})
        return report
    # Safety rule: H2 can only narrow an already verified H1 candidate set; it can never create FOUND from no human evidence.
    if not literal['candidate_paths']:
        report.update({'action':'partial','selected_path':None})
        return report
    probes,reason=semantic_hypotheses(llama,model,query,clarification,literal)
    h2=[]; current=set(literal['candidate_paths'])
    for probe in probes:
        r=query_glyph(root,probe); r['source']='H2_model_hypothesis'; h2.append(r)
        overlap=current & set(r['paths'])
        if overlap: current=overlap
        if len(current)==1: break
    semantic={'probes':probes,'reason':reason,'evidence':h2,'remaining_candidates':sorted(current)}
    report['semantic_stage']=semantic
    if len(current)==1:
        report.update({'action':'found','selected_path':next(iter(current))})
    elif current:
        report.update({'action':'ambiguous','selected_path':None})
    else:
        report.update({'action':'partial','selected_path':None})
    return report


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--llama-cli',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--query',required=True); ap.add_argument('--clarification')
    a=ap.parse_args(); print(json.dumps(run_router(a.root,a.llama_cli,a.model,a.query,a.clarification),ensure_ascii=False,sort_keys=True))

if __name__=='__main__': main()
