#!/usr/bin/env python3
import argparse,json,math,re,subprocess
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
QUERY=HERE/'query_rlb3x_object.py'
TOKEN_RE=re.compile(r'[A-Za-z0-9_][A-Za-z0-9_.+/-]*')
STOP={'C','code','old','find','page','man','book','document'}


def candidates(text):
    toks=TOKEN_RE.findall(text)
    out=[]
    def add(x):
        x=' '.join(x.strip(' .,:;!?()[]{}"\'').split())
        if len(x)>=2 and x not in out and x not in STOP: out.append(x)
    # exact marker-like tokens first
    for t in toks:
        if '_' in t or any(c.isdigit() for c in t): add(t)
    # preserve explicit adjacent ASCII phrase when useful
    for n in (3,2):
        for i in range(len(toks)-n+1):
            phrase=' '.join(toks[i:i+n])
            if all(x not in STOP for x in toks[i:i+n]): add(phrase)
    for t in toks: add(t)
    return out[:20]


def query(root,probe):
    out=subprocess.check_output([
        'python3',str(QUERY),
        '--rlb3x',str(root/'bwt.rlb3x'),
        '--locate-core',str(root/'locate.loc2'),
        '--objects',str(root/'objects.json'),
        '--pattern-hex',probe.encode('utf-8').hex(),
    ],text=True)
    got=json.loads(out)
    return sorted({h['path'] for h in got['valid_hits']})


def score(root,text,total_objects=30):
    probes=candidates(text)
    ev=[]; by_path=defaultdict(lambda:{'score':0.0,'probes':[],'strong':0,'independent':0})
    for p in probes:
        paths=query(root,p); df=len(paths)
        if df==0:
            ev.append({'probe':p,'df':0,'bits':None,'paths':[]}); continue
        bits=math.log2(total_objects/df)
        # Phrase/identifier bonus is capped; df remains dominant.
        lexical_bonus=0.75 if (' ' in p or '_' in p) else 0.0
        weight=bits+lexical_bonus
        strong=bits>=3.0 or df<=3
        ev.append({'probe':p,'df':df,'bits':bits,'weight':weight,'strong':strong,'paths':paths})
        for path in paths:
            by_path[path]['score']+=weight
            by_path[path]['probes'].append(p)
            if strong: by_path[path]['strong']+=1
    ranked=sorted((dict(v,path=k) for k,v in by_path.items()),key=lambda x:(-x['score'],-x['strong'],x['path']))
    if not ranked:
        exact_marker=any('_' in p and any(c.isdigit() for c in p) for p in probes)
        return {'action':'not_found' if exact_marker else 'partial','selected_path':None,'ranked':[],'evidence':ev,'probes':probes}
    top=ranked[0]; second=ranked[1] if len(ranked)>1 else None
    margin=top['score']-(second['score'] if second else 0.0)
    # FOUND is conservative: at least one strong clue, enough total information,
    # and a clear lead over the runner-up. Common-word intersections cannot pass alone.
    found=(top['strong']>=1 and top['score']>=4.0 and margin>=2.0)
    return {'action':'found' if found else 'ambiguous','selected_path':top['path'] if found else None,'ranked':ranked[:8],'evidence':ev,'probes':probes,'margin_bits':margin,'thresholds':{'min_score':4.0,'min_margin':2.0,'min_strong':1}}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('root',type=Path); ap.add_argument('--query',required=True); ap.add_argument('--clarification')
    a=ap.parse_args(); text=a.query+(' '+a.clarification if a.clarification else '')
    print(json.dumps(score(a.root,text),ensure_ascii=False,sort_keys=True))

if __name__=='__main__': main()
