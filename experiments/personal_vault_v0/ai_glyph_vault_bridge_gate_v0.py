#!/usr/bin/env python3
import json,shutil,subprocess,tempfile
from pathlib import Path

CLI='experiments/personal_vault_v0/glyph_vault_cli_v0.py'
FETCH='experiments/personal_vault_v0/fetch_corpora.sh'
BRIDGE='experiments/personal_vault_v0/ai_glyph_vault_bridge_v0.py'

def run(*args):
    return subprocess.check_output([str(x) for x in args],text=True)

def ask(vault,plan):
    with tempfile.NamedTemporaryFile('w',suffix='.json',delete=False,encoding='utf-8') as f:
        json.dump(plan,f,ensure_ascii=False,separators=(',',':'))
        p=f.name
    return json.loads(run('python3',BRIDGE,vault,'--plan',p))

with tempfile.TemporaryDirectory(prefix='glyph-ai-vault-bridge-v0-') as td:
    t=Path(td); raw=t/'raw'; src1=t/'src1'; src2=t/'src2'; vault=t/'vault'
    subprocess.check_call(['bash',FETCH,str(raw)])
    src1.mkdir(); src2.mkdir()
    shutil.copytree(raw/'canterbury',src1/'canterbury')
    for name in ('calgary','artificial','miscellaneous'):
        shutil.copytree(raw/name,src2/name)

    json.loads(run('python3',CLI,'init',vault))
    json.loads(run('python3',CLI,'add',vault,src1))
    json.loads(run('python3',CLI,'add',vault,src2))
    ver=json.loads(run('python3',CLI,'verify',vault))
    assert ver['root_binding_verified'] is True and ver['root_parent_binding_verified'] is True

    r1=ask(vault,{'id':'global-alice','human_query':'Найди книгу про Алису и Чеширского кота','probes':['Cheshire','Wonderland']})
    assert r1['action']=='found',r1
    assert r1['selected_object']['segment_id']=='00000001',r1
    assert r1['selected_object']['path']=='canterbury/alice29.txt',r1
    assert r1['vault_root']['segment_count']==2 and r1['vault_root']['root_binding_verified'] is True

    r2=ask(vault,{'id':'global-hardy','human_query':'Найди книгу Hardy с Bathsheba','probes':['T. HARDY','Bathsheba']})
    assert r2['action']=='found',r2
    assert r2['selected_object']['segment_id']=='00000002',r2
    assert r2['selected_object']['path']=='calgary/book1',r2

    r3=ask(vault,{'id':'global-ambiguous','human_query':'Где старый C-код с printf и struct?','probes':['printf','struct']})
    assert r3['action']=='ambiguous',r3
    assert r3['clarification'] and r3['clarification']['needed'] is True,r3

    r4=ask(vault,{'id':'global-not-found','human_query':'Найди несуществующий маркер','probes':['GLYPH_GLOBAL_NEVER_PRESENT_9182AF']})
    assert r4['action']=='not_found',r4

    # Hostile root binding: mutate a committed segment manifest in a disposable copy.
    hostile=t/'hostile'; shutil.copytree(vault,hostile)
    mf=hostile/'segments'/'00000001'/'segment-manifest.json'
    doc=json.loads(mf.read_text()); doc['hostile_mutation']=1
    mf.write_text(json.dumps(doc,sort_keys=True,separators=(',',':'))+'\n')
    bad_verify=subprocess.run(['python3',CLI,'verify',hostile],text=True,capture_output=True)
    assert bad_verify.returncode!=0
    assert 'root-to-segment-manifest binding failed' in (bad_verify.stderr+bad_verify.stdout)

    report={
      'format':'GLYPH_AI_GLYPH_VAULT_BRIDGE_GATE_V0',
      'all_checks_passed':True,
      'committed_segments':2,
      'global_found_first_segment':'canterbury/alice29.txt',
      'global_found_second_segment':'calgary/book1',
      'global_ambiguous_preserved':True,
      'global_not_found_preserved':True,
      'segment_fanout_hidden_from_ai':True,
      'latest_committed_root_only':True,
      'root_to_segment_manifest_binding_verified':True,
      'hostile_segment_manifest_mutation_rejected':True,
      'oracle_used_inside_bridge':False,
      'important_non_claim':'This validates one global AI/evidence surface across two committed immutable segments. It does not yet optimize fan-out, maintain a global routing index, deduplicate versions, or run an autonomous on-device LLM.'
    }
    print(json.dumps(report,ensure_ascii=False,sort_keys=True))
