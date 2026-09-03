#!/usr/bin/env python3
import argparse,json,re,subprocess,tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
QUERY=HERE/'query_rlb3x_object.py'

TOKEN_RE=re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.+/-]*')


def human_candidates(query,clarification=None):
    text=query + (' ' + clarification if clarification else '')
    toks=TOKEN_RE.findall(text)
    out=[]
    def add(x):
        x=x.strip('.,;:!?()[]{}"\'')
        if len(x)>=2 and x not in out: out.append(x)
    # Exact marker-like tokens first.
    for t in toks:
        if '_' in t or any(c.isdigit() for c in t): add(t)
    # Adjacent ASCII phrases are often titles or remembered literal phrases.
    for n in (3,2):
        for i in range(len(toks)-n+1): add(' '.join(toks[i:i+n]))
    for t in toks: add(t)
    return out[:24]


def query_glyph(root,probe):
    out=subprocess.check_output([
        'python3',str(QUERY),
        '--rlb3x',str(root/'bwt.rlb3x'),
        '--locate-core',str(root/'locate.loc2'),
        '--objects',str(root/'objects.json'),
        '--pattern-hex',probe.encode('utf-8').hex(),
    ],text=True)
    got=json.loads(out)
    paths=sorted({h['path'] for h in got['valid_hits']})
    return {'probe':probe,'raw_count':got['raw_count'],'valid_count':got['valid_count'],'paths':paths}


def choose_probe(llama,model,human_query,candidates,history,clarification=None):
    if not candidates: return None
    schema={
      'type':'object','additionalProperties':False,
      'properties':{
        'probe':{'type':'string','enum':candidates},
        'reason':{'type':'string'}
      },
      'required':['probe','reason']
    }
    prompt=f'''You are the local tool router for GLYPH, a byte-exact personal memory vault.
Choose exactly ONE literal probe from the supplied candidate list. You are not allowed to invent or rewrite a probe.
Prefer the most distinctive remembered title, proper name, command, identifier, or exact marker. Avoid generic words when a distinctive candidate exists.
GLYPH will execute the probe and return exact object evidence. If prior probes were ambiguous or empty, choose a different candidate that best separates the remaining possibilities.
Do not answer the human and do not name a file.

Human query: {human_query}
Human clarification: {clarification or 'null'}
Allowed probes: {json.dumps(candidates,ensure_ascii=False)}
Prior exact GLYPH tool results: {json.dumps(history,ensure_ascii=False)}
Return the constrained JSON object only.'''
    cmd=[str(llama),'--jinja','--single-turn','--no-display-prompt','-n','64','--temp','0','-c','2048','-j',json.dumps(schema,separators=(',',':')),'-p',prompt,'-m',str(model)]
    try:
        p=subprocess.run(cmd,text=True,capture_output=True,timeout=120)
    except subprocess.TimeoutExpired as e:
        raise SystemExit('frontier agent model invocation exceeded 120 s') from e
    if p.returncode!=0: raise SystemExit('llama-completion failed: '+p.stderr[-3000:])
    text=p.stdout.strip()
    # Constrained generation should make the whole generated object parseable; tolerate wrappers from runtime.
    start=text.find('{'); end=text.rfind('}')
    if start<0 or end<start: raise SystemExit('constrained model emitted no JSON object: '+text[-1000:])
    doc=json.loads(text[start:end+1])
    if doc['probe'] not in candidates: raise SystemExit('decoder constraint violation')
    return doc


def run_agent(root,llama,model,human_query,clarification=None,max_turns=4):
    candidates=human_candidates(human_query,clarification)
    if not candidates:
        return {'action':'not_found','selected_path':None,'history':[],'candidate_source':'human_extract_only'}
    remaining=list(candidates); history=[]
    for _ in range(max_turns):
        choice=choose_probe(llama,model,human_query,remaining,history,clarification)
        if choice is None: break
        probe=choice['probe']; remaining=[x for x in remaining if x!=probe]
        tool=query_glyph(root,probe); tool['model_reason']=choice.get('reason','')
        history.append(tool)
        if len(tool['paths'])==1:
            return {'action':'found','selected_path':tool['paths'][0],'history':history,'candidate_source':'human_extract_only'}
        # zero or multiple hits: let the model choose another literal human-supplied clue.
    # Intersect all non-empty evidence sets. This is exact evidence only.
    sets=[set(x['paths']) for x in history if x['paths']]
    if not sets:
        action='not_found'; selected=None
    else:
        inter=set.intersection(*sets) if len(sets)>1 else sets[0]
        if len(inter)==1: action='found'; selected=next(iter(inter))
        elif len(inter)>1: action='ambiguous'; selected=None
        else: action='partial'; selected=None
    return {'action':action,'selected_path':selected,'history':history,'candidate_source':'human_extract_only'}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--llama-cli',type=Path,required=True); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--query',required=True); ap.add_argument('--clarification')
    a=ap.parse_args(); result=run_agent(a.root,a.llama_cli,a.model,a.query,a.clarification)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True))

if __name__=='__main__': main()
