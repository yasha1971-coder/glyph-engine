"""Frozen sample-bz-xz6 V1 selection, validated independently in codec_frontier.

No previous corpus results, paths, extensions or hashes influence selection.
Codec names describe the actual encoder preset; old readers reject xz-6.
"""
import bz2
import lzma

POLICY = 'sample-bz-xz6'
LIMIT = 64 * 1024 * 1024
EXCLUDED = {'тут все.txt', 'тут все.txt\\'}


def choose(data):
    width = 64 * 1024
    if len(data) < 3 * width:
        return 'bzip2-9'
    sizes = {'bzip2-9': 0, 'xz-6': 0}
    for start in (0, (len(data) - width) // 2, len(data) - width):
        sample = data[start:start + width]
        sizes['bzip2-9'] += len(bz2.compress(sample, 9))
        sizes['xz-6'] += len(lzma.compress(sample, preset=6))
    gain = sizes['bzip2-9'] - sizes['xz-6']
    estimated = gain * len(data) / (3 * width)
    return 'xz-6' if gain >= .10 * sizes['bzip2-9'] and estimated >= 256 * 1024 else 'bzip2-9'


def encode(data):
    if len(data) > LIMIT:
        raise ValueError('adaptive file limit is 64 MiB')
    codec = choose(data)
    payload = (bz2.compress(data, 9) if codec == 'bzip2-9'
               else lzma.compress(data, format=lzma.FORMAT_XZ, preset=6))
    return ('raw', data) if len(payload) >= len(data) else (codec, payload)
