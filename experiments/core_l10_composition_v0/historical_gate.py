#!/usr/bin/env python3
"""Run unchanged historical Composition checker, including its independent replay."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

REF='81628f843dc8aa633c65594a66d2386bfc3cb5c2'


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--historical',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    a.output.mkdir(exist_ok=False)
    ref=a.historical.resolve()
    assert subprocess.check_output(['git','-C',str(ref),'rev-parse','HEAD'],text=True).strip()==REF
    paths=('docs/specs/GLYPH_COMPOSITION_SEMANTICS_V1.md','tools/check_composition_reference_v1.py',
           'tools/replay_composition_reference_v1.py','tools/glyph_operator_manifest_v1.py',
           'tools/glyph_operator_index_v1.py','tools/glyph_operator_query_v1.py')
    identities={}
    for name in paths:
        raw=(ref/name).read_bytes()
        assert raw==subprocess.check_output(['git','-C',str(ref),'show',REF+':'+name])
        identities[name]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
    command=[sys.executable,'-I',str(ref/'tools/check_composition_reference_v1.py')]
    start=time.monotonic()
    result=subprocess.run(command,capture_output=True,text=True,cwd=a.output,timeout=900)
    (a.output/'original-checker.log').write_text(result.stdout+result.stderr)
    assert result.returncode==0, result.stderr[-1500:]
    d=json.JSONDecoder().raw_decode(result.stdout)[0]
    assert 'GLYPH COMPOSITION REFERENCE OK' in result.stdout
    assert d['normative_exact_count']==25 and d['normative_open_count']==0 and d['normative_closure_complete']
    assert d['independent_composition_replay_verified'] and d['independent_replay_negative_case_count']==3
    assert d['runtime_corpus_id']=='4765ce3496058719db1436c56a33b325b16dc1c4cdfecd86664d0bdfb94dc950'
    assert d['source_manifest_id']=='a21e00fea40c4f7fb31b22c86400436acd90dba6e6efba855b4395ce7d162e09'
    report=dict(status='HISTORICAL_COMPOSITION_PASS',commit=REF,command=command,exit_code=result.returncode,
                wall_seconds=time.monotonic()-start,identities=identities,original_result=d)
    (a.output/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    print(json.dumps(report,sort_keys=True),flush=True)
    print('CORE_L10_HISTORICAL_25_EXACT_PASS',flush=True)


if __name__=='__main__':
    main()
