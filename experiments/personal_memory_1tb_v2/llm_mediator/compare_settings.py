"""Synthetic local-only A/B probe; preserves version_choice.py and its oracle."""
import argparse
import hashlib
import http.client
import ipaddress
import json
import os
from pathlib import Path
import time
from urllib.parse import urlsplit

import version_choice as probe

PROFILES = {
    'greedy': {'temperature': 0},
    'qwen-card-general': {'temperature': 0.7, 'top_p': 0.8, 'top_k': 20,
                          'min_p': 0.0, 'presence_penalty': 1.5,
                          'repeat_penalty': 1.0},
}


def origin(endpoint):
    u = urlsplit(endpoint)
    if (u.scheme != 'http' or u.path not in ('', '/') or u.query or
            u.fragment or u.username or u.password or
            not ipaddress.ip_address(u.hostname).is_loopback):
        raise ValueError('literal loopback HTTP origin required')
    return u.hostname, u.port or 80


def exchange(endpoint, method, path, body=None, timeout=120):
    host, port = origin(endpoint)
    conn = http.client.HTTPConnection(host, port, timeout=timeout)
    try:
        payload = None if body is None else json.dumps(body).encode('utf-8')
        conn.request(method, path, payload, {'Content-Type': 'application/json'})
        response = conn.getresponse()
        raw = response.read(262145)
        if response.status != 200 or len(raw) > 262144:
            raise ValueError('HTTP status or response size limit')
        return json.loads(raw)
    finally:
        conn.close()


def schedule(repeats):
    for repeat in range(repeats):
        for reverse in (False, True):
            for index, case in enumerate(probe.CASES):
                profiles = list(PROFILES)
                if (repeat + index + int(reverse)) % 2:
                    profiles.reverse()
                for profile in profiles:
                    yield repeat, reverse, case, profile


def make_body(question, docs, model, profile, seed):
    body = probe.request_body(question, docs, model)
    body.update(PROFILES[profile])
    body.update(stream=False, seed=seed,
                chat_template_kwargs={'enable_thinking': False})
    return body


def assess(response, docs, action, version):
    choice = response['choices'][0]
    result = probe.grade(choice['message'].get('content'), docs, action, version)
    result['finish_reason'] = choice.get('finish_reason')
    result['completed'] = result['finish_reason'] == 'stop'
    result['accepted_correct'] = bool(result['completed'] and result['decision_matches_oracle'])
    return result


def totals(rows):
    result = {}
    for profile in PROFILES:
        subset = [r for r in rows if r['profile'] == profile]
        result[profile] = {
            'attempted': len(subset),
            'correct': sum(r.get('assessment', {}).get('accepted_correct', False) for r in subset),
            'errors': sum('error_type' in r for r in subset),
            'by_case': {case[0]: {
                'attempted': sum(r['case'] == case[0] for r in subset),
                'correct': sum(r['case'] == case[0] and r.get('assessment', {}).get('accepted_correct', False)
                               for r in subset),
            } for case in probe.CASES},
        }
    return result


def run(endpoint, model, output, repeats=1, timeout=120, caller=exchange):
    origin(endpoint)
    if not 1 <= repeats <= 5 or not 1 <= timeout <= 300:
        raise ValueError('repeats 1..5; socket timeout 1..300 seconds')
    rows = []
    # Append-only JSONL: previous complete lines survive interruption.
    with Path(output).open('x', encoding='utf-8') as out:
        def save(obj):
            out.write(json.dumps(obj, ensure_ascii=False) + '\n')
            out.flush()
            os.fsync(out.fileno())
        save({'type': 'header', 'format': 'GLYPH_SETTINGS_AB_V1',
              'synthetic_only': True, 'model': model, 'model_weights_verified': False,
              'scope': 'known five-case diagnostic; NOT held-out, retrieval or GUI evaluation',
              'profiles': PROFILES, 'requested_parameters_verified_by_server': False,
              'cache_policy': 'uncontrolled; interleaved profiles; raw server timings retained',
              'timeout_kind': 'socket inactivity, not whole-request deadline',
              'timeout_seconds': timeout, 'planned_calls': repeats * 20,
              'fixture_sha256': hashlib.sha256(json.dumps([probe.DOCS, probe.CASES], ensure_ascii=False).encode()).hexdigest(),
              'probe_sha256': hashlib.sha256(Path(probe.__file__).read_bytes()).hexdigest()})
        complete = True
        for repeat, reverse, case, profile in schedule(repeats):
            ident, question, action, version = case
            docs = list(reversed(probe.DOCS)) if reverse else probe.DOCS
            body = make_body(question, docs, model, profile, 1700 + repeat)
            row = {'type': 'measurement', 'repeat': repeat + 1,
                   'reversed_order': reverse, 'case': ident, 'profile': profile,
                   'request': body}
            start = time.perf_counter()
            try:
                response = caller(endpoint, 'POST', '/v1/chat/completions', body, timeout)
                row['response'] = response
                row['assessment'] = assess(response, docs, action, version)
                if not row['assessment']['completed']:
                    complete = False
            except Exception as error:
                row.update(error_type=type(error).__name__)
                complete = False
            row['seconds'] = time.perf_counter() - start
            rows.append(row)
            save(row)
            print(json.dumps({'done': len(rows), 'profile': profile, 'case': ident,
                              'seconds': round(row['seconds'], 2),
                              'correct': row.get('assessment', {}).get('accepted_correct', False),
                              'error': row.get('error_type')}), flush=True)
            if not complete:
                break
        save({'type': 'summary', 'complete': complete, 'results': totals(rows)})
    return complete


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--endpoint', default='http://127.0.0.1:8080')
    parser.add_argument('--output', required=True)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--timeout', type=int, default=120)
    args = parser.parse_args()
    if Path(args.output).exists():
        parser.error('output already exists')
    models = exchange(args.endpoint, 'GET', '/v1/models', timeout=15)['data']
    if len(models) != 1:
        parser.error('exactly one loaded model required')
    complete = run(args.endpoint, models[0]['id'], args.output, args.repeats, args.timeout)
    print('Report:', args.output)
    raise SystemExit(0 if complete else 2)


if __name__ == '__main__':
    main()
