"""Read-only LLM planner over GLYPH's existing evidence bridge.

The model proposes an exact substring query, never permissions or operations.
Returned excerpts are source evidence, not model-authored answers.
"""
import http.client
import ipaddress
import json
import time
from urllib.parse import urlsplit

SCHEMA = {
    'type': 'object', 'additionalProperties': False,
    'properties': {'action': {'type': 'string', 'enum': ['search', 'clarify']},
                   'query': {'type': 'string', 'maxLength': 256},
                   'question': {'type': 'string', 'maxLength': 300}},
    'required': ['action', 'query', 'question']}
PROMPT = '''You are GLYPH's read-only query planner. Return exactly one JSON object
with action, query, question. For search, query must be a short exact substring
likely to occur in the user's documents; question must be empty. Preserve names,
numbers and language. Do not invent dates or translate a literal quoted phrase.
For an ambiguous request, or requests to delete, edit, run commands or transmit
files, use action=clarify, query="", and a short question in the user's language.
You cannot change permissions, access files or execute commands. Do not claim
that a search has already happened. No document text is supplied to this stage.'''


def parse(raw):
    if not isinstance(raw, str) or len(raw.encode('utf-8')) > 4096:
        raise ValueError('planner response budget')
    def unique(pairs):
        obj = {}
        for k, v in pairs:
            if k in obj:
                raise ValueError('duplicate JSON key')
            obj[k] = v
        return obj
    value = json.loads(raw, object_pairs_hook=unique)
    if not isinstance(value, dict) or set(value) != {'action', 'query', 'question'}:
        raise ValueError('unexpected planner fields')
    if any(not isinstance(v, str) for v in value.values()):
        raise ValueError('planner fields must be strings')
    action, query, question = value['action'], value['query'], value['question']
    if len(query.encode('utf-8')) > 256 or len(question) > 300:
        raise ValueError('planner field budget')
    if action == 'search' and query.strip() and not question:
        return value
    if action == 'clarify' and not query and question.strip():
        return value
    raise ValueError('invalid planner action')


def local_plan(question, *, endpoint='http://127.0.0.1:8080', model='local', timeout=30):
    if not isinstance(question, str) or not 0 < len(question.encode('utf-8')) <= 4096:
        raise ValueError('question budget')
    url = urlsplit(endpoint)
    if url.scheme != 'http' or url.username or url.password or url.path not in ('', '/') or url.query or url.fragment:
        raise ValueError('expected local HTTP origin')
    try:
        if not ipaddress.ip_address(url.hostname).is_loopback:
            raise ValueError('non-local endpoint')
    except (TypeError, ValueError) as e:
        raise ValueError('literal loopback IP required') from e
    body = json.dumps({'model': model, 'temperature': 0, 'max_tokens': 256,
        'messages': [{'role': 'system', 'content': PROMPT}, {'role': 'user', 'content': question}],
        'response_format': {'type': 'json_schema', 'json_schema': {'name': 'glyph_plan', 'strict': True, 'schema': SCHEMA}}}).encode()
    start = time.perf_counter()
    connection = http.client.HTTPConnection(url.hostname, url.port or 80, timeout=timeout)
    try:
        connection.request('POST', '/v1/chat/completions', body, {'Content-Type': 'application/json'})
        response = connection.getresponse()
        raw = response.read(32769)
        if response.status != 200 or len(raw) > 32768:
            raise ValueError('local model HTTP failure or response budget')
        doc = json.loads(raw)
        plan = parse(doc['choices'][0]['message']['content'])
        return plan, {'model_requested': model, 'seconds': time.perf_counter() - start,
                      'usage': doc.get('usage'), 'model_invoked': True}
    finally:
        connection.close()


def execute(plan_json, views, host_grants):
    """Host supplies immutable view/grant bindings. No model-selected grants."""
    from local_memory_bridge import evidence
    plan = parse(plan_json)
    if plan['action'] == 'clarify':
        return {'status': 'CLARIFY', 'question': plan['question'], 'operation_executed': False}
    result = evidence(views, plan['query'], frozenset(host_grants))
    return {'status': result['status'], 'query': plan['query'],
            'scope': result['scope'], 'coverage_complete': result['coverage_complete'],
            'skipped': result['skipped'], 'context_truncated': result['context_truncated'],
            'evidence': result['snippets'], 'answer_kind': 'verified excerpts; not generated synthesis'}
