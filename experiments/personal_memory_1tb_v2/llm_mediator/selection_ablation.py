"""Known-fixture diagnosis: direct, instruction-only, and oracle-intent selection."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import compare_settings as transport
import temporal_pipeline as pipeline
import version_choice as probe

MODES = ('direct', 'instruction_only', 'oracle_intent')
# Human-authored question interpretation; never a selected document ID or quote.
# This is privileged information, explicitly NOT autonomous intent recognition.
ORACLE_HINTS = {
    'old': {'temporal': 'before_event', 'anchor': 'отмены поезда'},
    'latest': {'temporal': 'latest', 'anchor': 'последнее согласованное решение'},
    'ukrainian': {'temporal': 'earlier_unspecified', 'anchor': 'старе рішення'},
    'ambiguous': {'temporal': 'unspecified', 'anchor': ''},
    'missing': {'temporal': 'unspecified', 'anchor': ''},
}


def body_for(mode, case, question, docs, model):
    if mode not in MODES:
        raise ValueError('unknown mode')
    body = pipeline.selection_body(question, docs, model)
    if mode != 'direct':
        body['messages'][0]['content'] += '\n' + pipeline.SELECTION_RULES
    if mode == 'oracle_intent':
        data = json.loads(body['messages'][1]['content'])
        data['temporal_hint'] = dict(ORACLE_HINTS[case])
        body['messages'][1]['content'] = json.dumps(data, ensure_ascii=False)
    return body


def jobs():
    for reverse in (False, True):
        for index, case in enumerate(probe.CASES):
            offset = (index + int(reverse)) % len(MODES)
            for mode in MODES[offset:] + MODES[:offset]:
                yield reverse, case, mode


def run(endpoint, model, output, caller=transport.exchange):
    transport.origin(endpoint)
    scores = {mode: [] for mode in MODES}
    with Path(output).open('x', encoding='utf-8') as out:
        def save(record):
            out.write(json.dumps(record, ensure_ascii=False) + '\n')
            out.flush()
            os.fsync(out.fileno())

        save({'type': 'header', 'format': 'GLYPH_SELECTION_ABLATION_V1',
              'synthetic_only': True, 'planned_calls': 30,
              'scope': 'known five-case diagnostic, NOT held-out or production accuracy',
              'oracle_intent_uses_privileged_labels': True,
              'oracle_intent_is_autonomous': False,
              'cache': 'uncontrolled; mode order rotated, not cold latency',
              'model_weights_verified': False, 'timeout_kind': 'socket inactivity',
              'timeout_seconds': 120,
              'fixture_sha256': hashlib.sha256(json.dumps([probe.DOCS, probe.CASES], ensure_ascii=False).encode()).hexdigest(),
              'code_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                              for p in [Path(__file__), Path(probe.__file__),
                                        Path(pipeline.__file__), Path(transport.__file__)]}})
        complete = True
        for reverse, case_tuple, mode in jobs():
            case, question, expected_action, expected_id = case_tuple
            docs = list(reversed(probe.DOCS)) if reverse else probe.DOCS
            body = body_for(mode, case, question, docs, model)
            row = {'type': 'measurement', 'mode': mode, 'case': case,
                   'reversed_order': reverse, 'request': body}
            start = time.perf_counter()
            try:
                response = caller(endpoint, 'POST', '/v1/chat/completions', body, 120)
                row['response'] = response
                row['assessment'] = transport.assess(response, docs, expected_action, expected_id)
                if not row['assessment']['completed']:
                    complete = False
            except Exception as error:
                row['error_type'] = type(error).__name__
                complete = False
            row['seconds'] = time.perf_counter() - start
            correct = row.get('assessment', {}).get('accepted_correct', False)
            scores[mode].append(correct)
            save(row)
            print(json.dumps({'mode': mode, 'case': case, 'reverse': reverse,
                              'correct': correct, 'seconds': round(row['seconds'], 2),
                              'error': row.get('error_type')}), flush=True)
            if not complete:
                break
        summary = {'type': 'summary', 'complete': complete,
                   'results': {mode: {'correct': sum(v), 'attempted': len(v), 'planned': 10,
                                     'uses_oracle_intent': mode == 'oracle_intent'}
                               for mode, v in scores.items()}}
        save(summary)
        print(json.dumps(summary), flush=True)
    return complete


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--endpoint', default='http://127.0.0.1:8080')
    p.add_argument('--output', required=True)
    args = p.parse_args()
    if Path(args.output).exists():
        p.error('output already exists')
    models = transport.exchange(args.endpoint, 'GET', '/v1/models', timeout=15)['data']
    if len(models) != 1:
        p.error('exactly one model required')
    ok = run(args.endpoint, models[0]['id'], args.output)
    print('Report:', args.output)
    raise SystemExit(0 if ok else 2)


if __name__ == '__main__':
    main()
