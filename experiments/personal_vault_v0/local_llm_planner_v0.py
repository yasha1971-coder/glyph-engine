#!/usr/bin/env python3
import argparse,json,re,subprocess,tempfile
from pathlib import Path


def extract_json(text):
    # Prefer a fenced JSON object, otherwise take the first balanced-looking object.
    m=re.search(r'```(?:json)?\s*(\{.*?\})\s*```',text,re.S|re.I)
    candidates=[m.group(1)] if m else []
    candidates+=re.findall(r'(\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\})',text,re.S)
    for raw in candidates:
        try:
            doc=json.loads(raw)
            if isinstance(doc,dict) and 'probes' in doc:
                return doc
        except Exception:
            pass
    raise SystemExit('local LLM did not emit parseable planner JSON: '+text[-2000:])


def build_prompt(inp):
    prev=inp.get('previous_state')
    clarification=inp.get('human_clarification')
    return f'''/no_think
You are the local search planner for a personal byte-exact archive called GLYPH.
Your job is NOT to answer the user and NOT to name a file. Generate 1 to 4 short literal byte probes that are likely to occur exactly in the sought document.
Prefer distinctive proper names, titles, code identifiers, commands, or exact phrases already present in the user's memory. Do not translate distinctive English words into Russian. Do not invent facts.
If the user gives an exact marker/token, preserve it exactly as a probe.
If prior GLYPH evidence is ambiguous, use the human clarification to generate NEW distinguishing probes.
Return ONLY one JSON object with exactly these keys:
{{"format":"GLYPH_PLANNER_OUTPUT_V0","probes":["..."],"reason":"brief","stop_if_unverified":true}}

Human query: {inp['human_query']}
Previous GLYPH state: {json.dumps(prev,ensure_ascii=False) if prev is not None else 'null'}
Human clarification: {json.dumps(clarification,ensure_ascii=False) if clarification is not None else 'null'}
'''


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--llama-cli',type=Path,required=True)
    ap.add_argument('--model',required=True,help='GGUF path or -hf style model argument value')
    ap.add_argument('--input',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--hf',action='store_true',help='Pass model through llama.cpp -hf instead of -m')
    a=ap.parse_args()
    inp=json.loads(a.input.read_text())
    if inp.get('format')!='GLYPH_PLANNER_INPUT_V0': raise SystemExit('bad planner input format')
    prompt=build_prompt(inp)
    cmd=[str(a.llama_cli),'-n','160','--temp','0','--no-display-prompt','-p',prompt]
    if a.hf: cmd += ['-hf',a.model]
    else: cmd += ['-m',a.model]
    p=subprocess.run(cmd,text=True,capture_output=True)
    if p.returncode!=0:
        raise SystemExit('llama-cli failed: '+p.stderr[-4000:])
    raw=p.stdout.strip()
    doc=extract_json(raw)
    out={
        'format':'GLYPH_PLANNER_OUTPUT_V0',
        'probes':doc.get('probes'),
        'reason':doc.get('reason') or 'local LLM planner',
        'stop_if_unverified':doc.get('stop_if_unverified'),
    }
    a.output.write_text(json.dumps(out,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\n')
    print(json.dumps({'ok':True,'planner':'local_llm','probes':out['probes']},ensure_ascii=False,sort_keys=True))

if __name__=='__main__': main()
