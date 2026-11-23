"""Paired full-catalog scheduling probe. External pins remain mandatory."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(1, str(HERE.parent))
from session import Session
from parallel_session import ParallelHashSession
from vault import canon
from lifecycle_costs import disk, measured


def candidate_worker(args):
    session = ParallelHashSession(args.capsule / 'vault', args.pin, args.reference)
    try:
        session.query(args.pin, b'absent')
    finally:
        session.close()


def hostiles(vault, head, reference, output):
    # Inject the real candidate into the existing assertions, not mock replies.
    import cost_hostiles
    old = cost_hostiles.Session
    try:
        cost_hostiles.Session = ParallelHashSession
        legacy = cost_hostiles.run(vault, head, reference, output / 'original',
                                  json.loads((HERE / 'LOCK.json').read_bytes()))
    finally:
        cost_hostiles.Session = old
    # The original RSS assertion launches its old worker. Exercise a real
    # candidate worker separately; its process group is the only kill target.
    cmd = [sys.executable, '-I', str(Path(__file__).resolve()),
           '--candidate-worker', '--capsule', str(vault.parent), '--pin', head,
           '--reference', str(reference)]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, start_new_session=True)
    killed = False
    try:
        deadline = time.monotonic() + 30
        while process.poll() is None and time.monotonic() < deadline:
            rss = 0
            for entry in Path('/proc').iterdir():
                if not entry.name.isdigit():
                    continue
                try:
                    fields = (entry / 'stat').read_text().rsplit(')', 1)[1].split()
                    if int(fields[2]) == process.pid:
                        rss += int((entry / 'statm').read_text().split()[1]) * os.sysconf('SC_PAGE_SIZE')
                except (OSError, ValueError, IndexError):
                    continue
            if rss > 1:
                os.killpg(process.pid, signal.SIGKILL)
                killed = True
                break
            time.sleep(0.01)
        code = process.wait(timeout=5)
        assert killed and code == -signal.SIGKILL, (killed, code)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
    return dict(original_assertions=legacy,
                original_constructor_cases_against_candidate=16,
                original_RSS_case_uses_original_worker=True,
                candidate_owned_worker_RSS_kill=dict(argv=cmd, exit_code=code,
                                                     reason='sampled RSS exceeds one-byte budget'))


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    lock_raw = (HERE / 'PARALLEL_GUARD_LOCK.json').read_bytes()
    lock = json.loads(lock_raw)
    envelope = json.loads(args.pins.read_bytes())
    cases = envelope if type(envelope) is list else envelope['cases']
    head = cases[-1]['pin'] if type(envelope) is list else envelope['head']
    current = [case for case in cases if case['pin'] == head]
    assert len(current) == (3 if args.profile == 'public' else 5)
    expected = {}
    for case in current:
        raw = (args.capsule / (str(cases.index(case)) + '.json')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == case['claim_sha256']
        expected[(case['query_hex'], case.get('limit', 256))] = raw
    vault = args.capsule / 'vault'
    before = disk(vault)
    resources = lock['public_resource_arguments'] if args.profile == 'public' else {}
    states, opens, trials = {}, {}, []
    stages = []
    try:
        for mode, cls in [('serial', Session), ('parallel', ParallelHashSession)]:
            states[mode], opens[mode] = measured(lambda cls=cls: cls(vault, head, args.reference, **resources))
            stages.append(Path(states[mode].stage.name))
            stages.extend(Path(leaf.stage.name) for leaf in states[mode].states.values())
        baseline = states['serial']
        assert states['parallel']._catalogs() == dict(live=baseline.live, **baseline.private)
        if args.profile == 'public':
            documents = baseline.root['documents']
            assert len(documents) == 1 and documents[0]['bytes'] == 100000000
            assert documents[0]['sha256'] == lock['public_source_sha256']
        bytes_per_guard = sum(item['bytes'] for item in baseline.live.values())
        bytes_per_guard += sum(item['bytes'] for group in baseline.private.values() for item in group.values())
        sequence = current * lock['cycles'][args.profile]
        actual_parallel = {}
        for order in lock['trial_order']:
            trial = {}
            for mode in order:
                answers = []
                for case in sequence:
                    state = states[mode]
                    claim, cost = measured(lambda: state.query(head, bytes.fromhex(case['query_hex']), case.get('limit', 256)))
                    key = (case['query_hex'], case.get('limit', 256))
                    assert canon(claim) == expected[key], 'canonical receipt differs from external pin'
                    assert claim['independent_index_source_semantics'] is False
                    answers.append(dict(cost=cost, metrics=state.last_metrics,
                                        receipt_sha256=hashlib.sha256(canon(claim)).hexdigest()))
                    if mode == 'parallel':
                        actual_parallel[key] = canon(claim)
                trial[mode] = answers
            trials.append(trial)
    finally:
        for state in states.values():
            state.close()
    assert not any(path.exists() for path in stages), 'normal close leaked private state'
    ratios = [statistics.median(a['metrics']['full_guards_seconds'] for a in t['parallel']) /
              statistics.median(a['metrics']['full_guards_seconds'] for a in t['serial']) for t in trials]
    negatives = hostiles(vault, head, args.reference, args.output) if args.hostiles else None
    assert disk(vault) == before, 'input Vault changed'
    assert (HERE / 'PARALLEL_GUARD_LOCK.json').read_bytes() == lock_raw
    shutil.copytree(args.capsule, args.output / 'capsule')
    for case in current:
        (args.output / 'capsule' / (str(cases.index(case)) + '.json')).write_bytes(
            actual_parallel[(case['query_hex'], case.get('limit', 256))])
    (args.output / 'pins.json').write_bytes(args.pins.read_bytes())
    report = dict(status='PARALLEL_FULL_SHA_GUARD_CORRECTNESS_PASS', profile=args.profile,
                  checkout_head=subprocess.check_output(['git', '-C', str(HERE.parents[3]), 'rev-parse', 'HEAD'], text=True).strip(),
                  lock_sha256=hashlib.sha256(lock_raw).hexdigest(), opens=opens, trials=trials,
                  paired_guard_ratios=ratios, public_hypothesis_met=None if args.profile != 'public' else
                  statistics.median(ratios) <= lock['hypothesis_public_median_paired_full_guard_ratio_at_most'],
                  canonical_replies=len(sequence) * 2 * len(trials),
                  live_and_private_bytes_hashed_per_query=2 * bytes_per_guard,
                  code_hash_reads_additional=True, full_catalogs_equal=True,
                  hostiles=negatives, retained=before, normal_close_private_state_removed=True,
                  independent_index_source_semantics=False)
    (args.output / 'result.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('status', 'profile', 'checkout_head', 'canonical_replies', 'paired_guard_ratios', 'public_hypothesis_met', 'live_and_private_bytes_hashed_per_query')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--capsule', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--pins', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--profile', choices=['small', 'public'])
    parser.add_argument('--hostiles', action='store_true')
    parser.add_argument('--candidate-worker', action='store_true')
    parser.add_argument('--pin')
    arguments = parser.parse_args()
    if arguments.candidate_worker:
        candidate_worker(arguments)
    else:
        parser.error('--pins, --output and --profile are required') if not all((arguments.pins, arguments.output, arguments.profile)) else run(arguments)
