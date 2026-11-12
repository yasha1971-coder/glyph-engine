"""Clean packed-only replay. No Vault/importer/producer import or source oracle."""
import argparse,copy,json,shutil,subprocess,sys,tempfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE))
from packed_storage import view
from packed_history import authenticate,history,restore
from independent_scale import verify
from independent_verify import roots,read,parse,sha,canon,demand
from lifecycle_costs import measured,disk

def replay(vault,case,claim):
    with tempfile.TemporaryDirectory(prefix='glyph-packed-replay-') as tmp:
        target=Path(tmp);view(vault,target)
        expanded=sum(p.stat().st_size for p in target.rglob('*') if p.is_file())
        result=verify(target,case['pin'],bytes.fromhex(case['query_hex']),case.get('limit',256),claim)
        result['expanded_view_file_bytes']=expanded
        return result

def run(capsule,pins,output):
    output.mkdir(parents=True,exist_ok=False)
    if type(pins) is list:
        demand(len(pins)==3,'legacy case count');cases=pins;head=cases[-1]['pin'];selected=[]
    else:
        demand(type(pins) is dict and set(pins)=={'format','head','cases','selected'},'external pin envelope')
        demand(pins['format']=='GLYPH_PACKED_LIFECYCLE_PINS_V1','external pin format')
        cases,head,selected=pins['cases'],pins['head'],pins['selected']
        demand(type(cases) is list and 3<=len(cases)<=64 and type(selected) is list and len(selected)<=64,'case/selection bound')
    demand(set(p.name for p in capsule.iterdir())=={'vault',*(str(i)+'.json' for i in range(len(cases)))},'capsule must contain only packed Vault and claims')
    vault=capsule/'vault'
    # Artifact ZIPs omit empty directories. Absent staging is empty, whereas
    # any persistent staging entry still invalidates an index-only capsule.
    staging=vault/'staging'
    demand(not staging.exists() or (staging.is_dir() and not staging.is_symlink() and not any(staging.iterdir())),
           'capsule contains persistent staging; index-only transfer not established')
    before=disk(vault)
    for name in before['files']:
        parts=Path(name).parts
        allowed=(name in ('CURRENT','LOCK','decoder') or
            (len(parts)==2 and parts[0]=='snapshots' and len(parts[1])==64) or
            (len(parts)==3 and parts[0]=='objects' and len(parts[1])==64 and parts[2] in
             ('empty.json','root.json','storage.json','rlb2.packed','rlr2.packed','ef.packed')))
        demand(allowed,'capsule unaccounted source/representation: '+name)
    budget,cost=measured(lambda:authenticate(vault,head))
    results=[];metrics=[];selections=[]
    for i,c in enumerate(cases):
        demand(c['pin'] in budget['chain'],'query root outside pinned history')
        raw=read(capsule/(str(i)+'.json'),1<<20)
        demand(sha(raw)==c['claim_sha256'],'separately expected claim pin')
        claim=parse(raw)
        result,m=measured(lambda:replay(vault,c,claim));results.append(result);metrics.append(m)
    for s in selected:
        demand(s['pin'] in budget['chain'],'selected root outside pinned history')
        r=roots(vault,s['pin'],'GLYPH_BOUNDED_IMPORT_STREAM_V2',100000000)
        demand(type(s['doc_id']) is int and 0<=s['doc_id']<len(r['documents']),'selected doc ID')
        d=r['documents'][s['doc_id']]
        demand((d['revision'],d['sha256'],d['bytes'])==(s['revision'],s['sha256'],s['bytes']),'external selected source identity')
        with tempfile.TemporaryDirectory(prefix='glyph-selected-') as tmp:
            dest=restore(vault,d,Path(tmp))
            demand(sha(dest.read_bytes())==s['sha256'] and dest.stat().st_size==s['bytes'],'external selected restore SHA')
        selections.append(s)
    # Mutate only a disposable copy. Each negative names its precise reason.
    mutation=output/'negative-vault';shutil.copytree(vault,mutation)
    negatives=[]
    c=next(c for c in reversed(cases) if any(d['bytes'] for d in roots(vault,c['pin'],'GLYPH_BOUNDED_IMPORT_STREAM_V2',100000000)['documents']))
    index=cases.index(c);claim=parse(read(capsule/(str(index)+'.json'),1<<20))
    def reject(label,fn,exception,reason):
        try:fn()
        except exception as e:
            demand(reason(e),'wrong negative reason '+label+': '+str(e));negatives.append(dict(name=label,reason=str(e)))
        else:raise AssertionError('negative accepted: '+label)
    wrong=copy.deepcopy(claim);wrong['count']+=1
    reject('wrong-count',lambda:replay(mutation,c,wrong),ValueError,lambda e:str(e)=='independent answer mismatch')
    wrongpin=dict(c,pin='0'*64)
    reject('wrong-external-root-pin',lambda:replay(mutation,wrongpin,claim),FileNotFoundError,lambda e:Path(e.filename).parts[-2:]==('snapshots','0'*64))
    component=next((mutation/'objects').glob('*/ef.packed'));saved=component.read_bytes()
    try:
        component.unlink()
        reject('missing-component',lambda:replay(mutation,c,claim),FileNotFoundError,lambda e:Path(e.filename)==component)
        component.write_bytes(saved+b'x')
        reject('corrupt-component',lambda:replay(mutation,c,claim),ValueError,lambda e:str(e) in ('incomplete codec stream','decoded component identity'))
    finally:component.write_bytes(saved)
    if type(pins) is dict:
        from packed_history_negatives import run as history_negatives
        negatives+=history_negatives(vault,head,c,claim,replay,output/'history-negatives')
    demand(disk(vault)==before,'input Vault changed during independent replay')
    report=dict(status='CLEAN_PACKED_CAPSULE_PASS',head=head,replays=results,history=budget,
        retained_bytes=budget['retained_bytes'],distinct_source_bytes=budget['distinct_source_bytes'],
        selected_restores=selections,retained=before,history_cost=cost,replay_costs=metrics,
        negatives_rejected=negatives,original_sources_transferred=False,independent_index_source_semantics=False)
    report['checkout_head']=subprocess.check_output(['git','-C',str(HERE.parents[2]),'rev-parse','HEAD'],text=True).strip()
    report['checkout_tree']=subprocess.check_output(['git','-C',str(HERE.parents[2]),'rev-parse','HEAD^{tree}'],text=True).strip()
    return report

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--capsule',type=Path,required=True);p.add_argument('--pins-json',required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    try:
        report,cost=measured(lambda:run(a.capsule,json.loads(a.pins_json),a.output))
        imported=sorted(set(sys.modules)&{'vault','vault_packed','snapshot_contract','physical','scale_leaf','producer'})
        demand(not imported,'independent process imported writer/query producer')
        report['producer_modules_imported']=imported
        report['whole_replay_cost']=cost
        report['scratch_file_lengths']=sum(p.stat().st_size for p in a.output.rglob('*') if p.is_file())
        (a.output/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
        print(json.dumps(report,sort_keys=True),flush=True);print('PACKED_COMPLETE_HISTORY_INDEPENDENT_PASS',flush=True)
    except Exception as e:
        print(json.dumps(dict(status='UNTRUSTED',reason=str(e)[:500])),file=sys.stderr);sys.exit(3)
