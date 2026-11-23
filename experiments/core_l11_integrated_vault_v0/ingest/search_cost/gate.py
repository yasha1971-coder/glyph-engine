"""Pinned existing packed-capsule search cost experiment; no new product SLA."""
import argparse,hashlib,json,shutil,statistics,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;sys.path.insert(0,str(HERE));sys.path.insert(1,str(HERE.parent))
from monitor import run
from vault import canon
from lifecycle_costs import disk
from packed_history import authenticate

def main():
    p=argparse.ArgumentParser()
    for n in ('capsule','pins','reference','output'):p.add_argument('--'+n,type=Path,required=True)
    a=p.parse_args();a.output.mkdir(parents=True);lock_raw=(HERE/'LOCK.json').read_bytes();lock=json.loads(lock_raw)
    envelope=json.loads(a.pins.read_bytes());vault=a.capsule/'vault';head=envelope['head']
    before=disk(vault);history=authenticate(vault,head)
    cases=[c for c in envelope['cases'] if c['pin']==head];assert len(cases)==5
    expected={}
    for c in cases:
        i=envelope['cases'].index(c);raw=(a.capsule/(str(i)+'.json')).read_bytes()
        assert hashlib.sha256(raw).hexdigest()==c['claim_sha256'];expected[(c['query_hex'],c.get('limit',256))]=raw
    sequence=cases*lock['cycles'];trials=[]
    for trial,order in enumerate(lock['trial_order']):
        rows={}
        for mode in order:
            folder=a.output/(str(trial)+'-'+mode)
            if mode=='fresh_process':
                folder.mkdir();individual=[run('baseline',vault,head,a.reference,[c],folder/str(k),lock) for k,c in enumerate(sequence)]
                answers=[x['answers'][0] for x in individual]
                row=dict(wall_seconds=sum(x['wall_seconds'] for x in individual),answers=answers,
                    cold_command_response_seconds=[x['first_answer_seconds'] for x in individual],individual=individual)
            else:row=run('session' if mode=='session_batch' else 'baseline',vault,head,a.reference,sequence,folder,lock)
            for case,answer in zip(sequence,row['answers']):
                assert canon(answer['claim'])==expected[(case['query_hex'],case.get('limit',256))],'canonical claim differs'
                assert answer['claim']['independent_index_source_semantics'] is False
            rows[mode]=row
        trials.append(rows)
    # Attribution is a separately profiled original query, excluded from speed table.
    profile=run('baseline',vault,head,a.reference,[cases[1]],a.output/'profile',lock,profile=True)
    stats=profile['answers'][0]['profile'];startup=sum(r['cumulative_seconds'] for r in stats['rows'] if r['file'].endswith('/scale_leaf.py') and r['function']=='__init__')
    fraction=startup/stats['total_seconds'];ratios=[x['session_batch']['wall_seconds']/x['baseline_batch']['wall_seconds'] for x in trials]
    from cost_hostiles import run as hostile_tests
    negatives=hostile_tests(vault,head,a.reference,a.output/'hostiles',lock)
    assert disk(vault)==before,'read-only query mutated Vault'
    assert (HERE/'LOCK.json').read_bytes()==lock_raw,'locked protocol changed'
    # Export actual session-produced claims, not benchmark success flags.
    capsule=a.output/'capsule';shutil.copytree(a.capsule,capsule)
    answers=trials[-1]['session_batch']['answers'][:len(cases)]
    for c,answer in zip(cases,answers):
        index=envelope['cases'].index(c);raw=canon(answer['claim'])
        assert hashlib.sha256(raw).hexdigest()==c['claim_sha256']
        (capsule/(str(index)+'.json')).write_bytes(raw)
    (a.output/'pins.json').write_bytes(a.pins.read_bytes())
    report=dict(status='PACKED_SEARCH_COST_MEASURED',head=head,checkout_head=subprocess.check_output(['git','-C',str(HERE.parents[3]),'rev-parse','HEAD'],text=True).strip(),
        checkout_tree=subprocess.check_output(['git','-C',str(HERE.parents[3]),'rev-parse','HEAD^{tree}'],text=True).strip(),
        lock_sha256=hashlib.sha256(lock_raw).hexdigest(),lock=lock,hostiles=negatives,
        history=history,retained=before,profile=profile,trials=trials,startup_fraction=fraction,session_over_baseline_batch_ratios=ratios,
        startup_hypothesis_met=fraction>=lock['hypotheses']['state_startup_fraction_at_least'],
        batch_improvement_hypothesis_met=statistics.median(ratios)<=lock['hypotheses']['median_session_batch_wall_over_baseline_batch_wall_at_most'],
        independent_index_source_semantics=False,os_page_cache_cleared=False)
    (a.output/'result.json').write_text(json.dumps(report,sort_keys=True,indent=2)+'\n')
    summary={k:v for k,v in report.items() if k not in ('trials','profile','retained','lock')}
    summary['batches']=[{m:dict(wall_seconds=r['wall_seconds'],median_query_round_trip=statistics.median(x['round_trip_seconds'] for x in r['answers']),
        **(dict(median_cold_command_response=statistics.median(r['cold_command_response_seconds']),max_sampled_tree_rss_bytes=max(x['peak_sampled_tree_rss_bytes'] for x in r['individual']),
            summed_worker_self_cpu_seconds=sum(x['usage']['self_cpu_seconds'] for x in r['individual']),
            summed_reaped_children_cpu_seconds=sum(x['usage']['reaped_children_cpu_seconds'] for x in r['individual']),
            summed_client_cpu_seconds=sum(x['client_cpu_seconds'] for x in r['individual']),
            max_sampled_temp_file_lengths=max(x['peak_sampled_temp_file_lengths'] for x in r['individual'])) if m=='fresh_process' else {}),
        **({k:r[k] for k in ('ready_seconds','first_answer_seconds','peak_sampled_tree_rss_bytes','peak_sampled_temp_file_lengths','client_cpu_seconds','client_self_peak_rss_kib_cumulative','usage')} if m!='fresh_process' else {})) for m,r in t.items()} for t in trials]
    print(json.dumps(summary,sort_keys=True),flush=True);print('PACKED_SEARCH_COST_MEASURED',flush=True)
if __name__=='__main__':main()
