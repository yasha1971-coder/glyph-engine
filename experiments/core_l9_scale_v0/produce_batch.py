#!/usr/bin/env python3
"""Separate producer process: receives queries/index state, never oracle source."""
import argparse
import importlib.util
import json
from pathlib import Path
import resource
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    for name in ('state', 'reference', 'queries', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--test-fixture', action='store_true')
    a = p.parse_args()
    spec = importlib.util.spec_from_file_location('scale_query_producer', HERE/'producer.py')
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    a.output.mkdir(exist_ok=False)
    state = a.state.resolve()
    raw = (state/'root.json').read_bytes()
    q = m.QueryState(state/'root.json', m.sha(raw), state/'rlb2', state/'rlr2', state/'ef',
                     state/'decoder', a.reference.resolve(), a.test_fixture)
    cases = []
    try:
        assert not (Path(q.stage.name)/'ephemeral-authenticated-source').exists()
        for query in json.loads(a.queries.read_bytes()):
            assert set(query) == {'name', 'pattern_hex', 'limit'} and '/' not in query['name']
            pattern = bytes.fromhex(query['pattern_hex'])
            assert pattern.hex() == query['pattern_hex']
            receipt = q.query(pattern, query['limit'])
            first_metrics = dict(q.last_metrics)
            path = a.output/(query['name']+'.json')
            path.write_bytes(m.canonical(receipt))
            assert path.read_bytes() == m.canonical(q.query(pattern, query['limit'])), 'nondeterministic query receipt'
            cases.append(dict(query=query, receipt=path.name, status=receipt['status'],
                count=receipt['count'], offsets=receipt['offsets'], receipt_bytes=path.stat().st_size,
                producer_metrics=first_metrics, determinism_replay_metrics=q.last_metrics))
        report = dict(startup=q.startup, cases=cases, raw_source_input=False,
                      child_peak_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss,
                      retained_query_source_archive_bytes=0)
        (a.output/'producer-result.json').write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
        print(json.dumps(report, sort_keys=True), flush=True)
    finally:
        q.close()


if __name__ == '__main__':
    main()
