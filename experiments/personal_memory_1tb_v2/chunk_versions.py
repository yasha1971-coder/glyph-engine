"""Gear CDC reference pilot, not FastCDC. Depth-one legacy ranges, no delta chain."""
import os
import hashlib
import json
import zlib

MIN, TARGET, MAX = 4096, 16384, 65536
MASK64 = (1 << 64) - 1
GEAR = tuple(int.from_bytes(hashlib.sha256(b'GLYPH-GEAR-V1' + bytes([i])).digest()[:8], 'little') for i in range(256))
FORMAT = 'GLYPH_CHUNK_RECIPE_V1'


def split_python(data):
    start, rolling = 0, 0
    for i, byte in enumerate(data):
        rolling = ((rolling << 1) + GEAR[byte]) & MASK64
        size = i + 1 - start
        if size >= MAX or (size >= MIN and rolling & (TARGET - 1) == 0):
            yield start, data[start:i+1]
            start, rolling = i + 1, 0
    if start < len(data):
        yield start, data[start:]


def split(data):
    native = os.environ.get('GLYPH_CDC_NATIVE')
    if native and isinstance(data, bytes):
        from native_cdc import split as native_split
        yield from native_split(data, GEAR, MIN, TARGET, MAX, native)
    else:
        yield from split_python(data)


def pack(data):
    compressed = zlib.compress(data, 6)
    return b'Z' + compressed if len(compressed) < len(data) else b'R' + data


def unpack(payload, size):
    if payload[:1] == b'R':
        data = payload[1:]
    elif payload[:1] == b'Z':
        decoder = zlib.decompressobj()
        data = decoder.decompress(payload[1:], size + 1)
        if not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
            raise ValueError('invalid chunk stream')
    else:
        raise ValueError('unknown chunk codec')
    if len(data) != size:
        raise ValueError('chunk size mismatch')
    return data


def recipe(m, item):
    import incremental_memory as inc
    if not inc.HEX.fullmatch(item.get('recipe', '')):
        raise inc.Error('invalid recipe pin')
    raw = inc.read_regular(m.root, 'recipes/' + item['recipe'], inc.META_LIMIT)
    if inc.digest(raw) != item['recipe']:
        raise inc.Error('recipe corruption')
    doc = json.loads(raw)
    if doc['format'] != FORMAT or doc['sha256'] != item['sha256'] or doc['bytes'] != item['bytes']:
        raise inc.Error('recipe identity mismatch')
    if len(doc['chunks']) > inc.LIMIT // MIN + 1:
        raise inc.Error('too many chunks')
    total = 0
    for chunk in doc['chunks']:
        if not inc.HEX.fullmatch(chunk['sha256']) or type(chunk['bytes']) is not int or not 0 < chunk['bytes'] <= MAX:
            raise inc.Error('invalid chunk')
        if 'source' in chunk:
            source = chunk['source']
            if source['storage'] not in ('base', 'raw', 'deflate') or not inc.HEX.fullmatch(source['sha256']):
                raise inc.Error('recursive or invalid source')
            if type(source['bytes']) is not int or not 0 <= source['bytes'] <= 64 * 1024**2:
                raise inc.Error('source budget exceeded')
            if type(chunk['offset']) is not int or chunk['offset'] < 0 or chunk['offset'] + chunk['bytes'] > source['bytes']:
                raise inc.Error('range outside source')
        total += chunk['bytes']
    if total != item['bytes']:
        raise inc.Error('recipe total mismatch')
    return doc


def read(m, item):
    import incremental_memory as inc
    cache, parts = {}, []
    for chunk in recipe(m, item)['chunks']:
        if 'source' in chunk:
            source = chunk['source']
            key = (source['storage'], source['sha256'], source['bytes'])
            if key not in cache:
                # One recipe may depend on one legacy whole object only.
                if cache:
                    raise inc.Error('too many legacy sources')
                cache[key] = m.read_item(source)
            data = cache[key][chunk['offset']:chunk['offset'] + chunk['bytes']]
        else:
            payload = inc.read_regular(m.root, 'chunks/' + chunk['sha256'], MAX + 1)
            data = unpack(payload, chunk['bytes'])
        if inc.digest(data) != chunk['sha256']:
            raise inc.Error('chunk digest mismatch')
        parts.append(data)
    result = b''.join(parts)
    if len(result) != item['bytes'] or inc.digest(result) != item['sha256']:
        raise inc.Error('file digest mismatch')
    return result


def store(m, data, previous):
    """Choose by newly needed payload+recipe bytes; immutable old data retained."""
    import incremental_memory as inc
    whole = zlib.compress(data, 6)
    codec, whole = ('deflate', whole) if len(whole) < len(data) else ('raw', data)
    sha = inc.digest(data)
    reused = {}
    if previous and previous['storage'] == 'chunks-v1':
        # Validate the previous recipe and its bytes before reusing references.
        m.read_item(previous)
        reused = {c['sha256']: c for c in recipe(m, previous)['chunks']}
    elif previous and previous['bytes'] <= inc.LIMIT:
        old = m.read_item(previous)
        reused = {inc.digest(part): {'sha256': inc.digest(part), 'bytes': len(part),
                  'source': previous, 'offset': offset} for offset, part in split(old)}
    chunks, pending = [], {}
    for _, part in split(data):
        h = inc.digest(part)
        if h in reused:
            chunks.append(reused[h])
        else:
            path = m.root / 'chunks' / h
            if path.exists() or path.is_symlink():
                payload = inc.read_regular(m.root, 'chunks/' + h, MAX + 1)
                if unpack(payload, len(part)) != part:
                    raise inc.Error('existing chunk mismatch')
            else:
                pending[h] = pack(part)
            chunks.append({'sha256': h, 'bytes': len(part)})
    doc = {'format': FORMAT, 'sha256': sha, 'bytes': len(data), 'chunks': chunks}
    raw = inc.base.canonical_json(doc)
    if len(raw) + sum(map(len, pending.values())) < len(whole):
        for directory in ('chunks', 'recipes'):
            (m.root / directory).mkdir(exist_ok=True)
        for h, payload in pending.items():
            inc.publish(m.root / 'chunks' / h, payload)
        h = inc.digest(raw)
        inc.publish(m.root / 'recipes' / h, raw)
        item = {'sha256': sha, 'bytes': len(data), 'storage': 'chunks-v1', 'recipe': h}
    else:
        inc.publish(m.root / 'objects' / sha, whole)
        item = {'sha256': sha, 'bytes': len(data), 'storage': codec}
    if m.read_item(item) != data:
        raise inc.Error('stored version roundtrip failed')
    return item
