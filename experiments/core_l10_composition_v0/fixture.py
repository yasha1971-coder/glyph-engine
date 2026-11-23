#!/usr/bin/env python3
"""Construction-only process: real frozen RLB2/RLR2 + physical EF, then delete SA/BWT."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import struct
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def documents():
    # Exactly the original Composition checker fixture, followed by three new slots.
    return [('00-empty.bin', b''), ('10-alpha.bin', b'alpha LEFT'), ('20-edge-a.bin', b'EDGE-A'),
        ('30-edge-b.bin', b'B-EDGE shared\0\xff'), ('40-shared.bin', b'shared middle'),
        ('50-dup-a.bin', b'dup'), ('60-dup-b.bin', b'dup'), ('70-tail.bin', b'tail shared'),
        ('80-omega.bin', b'omega'), ('90-append.bin', b'ONLY_IN_V1 shared'),
        ('91-new-dup.bin', b'dup'), ('92-new-binary.bin', b'added \xff\0 shared  spaced')]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--stage', choices=('initial', 'append'), required=True)
    for name in ('bundle', 'reference'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--decoder', type=Path)
    a = p.parse_args()
    b, ref = a.bundle.resolve(), a.reference.resolve()
    m = module(HERE/'producer.py', 'construction_identity_helpers')
    assert subprocess.check_output(['git', '-C', str(ref), 'rev-parse', 'HEAD'], text=True).strip() == m.REF
    subprocess.run(['git', '-C', str(ref), 'diff', '--exit-code', 'HEAD', '--'], check=True)
    if a.stage == 'initial':
        assert not b.exists() and a.decoder is not None
        for folder in (b/'leaves', b/'blocks', b/'roots'):
            folder.mkdir(parents=True)
        shutil.copy2(a.decoder, b/'decoder')
        start, stop, generation = 0, 9, 0
    else:
        start, stop, generation = 9, 12, 1
        assert b.is_dir() and a.decoder is None
    all_docs = documents()
    records = [dict(doc_id=i, relative_path_bytes_hex=name.encode().hex(), byte_length=len(raw),
                    sha256=hashlib.sha256(raw).hexdigest()) for i, (name, raw) in enumerate(all_docs[:stop])]
    sys.path.insert(0, str(REPO/'experiments/loc_text_ef_v0'))
    from real_enwik8_ef_physical_gate import build, verify
    container = module(ref/'tools/rlbwt_container_v2.py', 'frozen_construction_container')
    for i in range(start, stop):
        leaf = b/'leaves'/str(i)
        leaf.mkdir(exist_ok=False)
        source = all_docs[i][1]
        if not source:
            binding = dict(format=m.PROFILE+'_LEAF', document=records[i], kind='EMPTY_MANIFEST_ONLY', index_root=None, artifacts={})
        else:
            construction = leaf/'construction'
            construction.mkdir()
            mapped = tuple(x+1 for x in source)+(0,)
            sa = sorted(range(len(mapped)), key=lambda j: mapped[j:])
            symbols = [256 if j == 0 else source[j-1] for j in sa]
            payload = struct.pack('<'+'H'*len(symbols), *symbols)
            fnv = container.Fnv1a64()
            fnv.update(payload)
            bwt = construction/'source.bwt'
            bwt.write_bytes(container.BWT_HEADER.pack(container.BWT_MAGIC, container.BWT_VERSION,
                len(source), len(mapped), 257, 256, 16, len(payload), fnv.value)+payload)
            subprocess.run([sys.executable, str(ref/'tools/rlbwt_container_v2.py'), 'encode', str(bwt), str(leaf/'rlb2')],
                           capture_output=True, check=True)
            subprocess.run([sys.executable, str(ref/'tools/rlbwt_rank_v2.py'), 'build', str(leaf/'rlb2'), str(leaf/'rlr2'),
                            '--rank-step', '4'], capture_output=True, check=True)
            sap = construction/'source.sa'
            sap.write_bytes(b'\0'*60+struct.pack('<'+'I'*len(sa), *sa))
            geometry = build(sap, len(sa), 2, leaf/'ef')
            assert verify(sap, leaf/'ef', geometry)['sampled_sa_records_verified'] > 0
            artifacts = {name: m.identity(leaf/name) for name in ('rlb2', 'rlr2', 'ef')}
            root = dict(schema='GLYPH_CORE_L8_PINNED_SOURCE_MATERIALIZATION_V0', frozen_commit=m.REF,
                source=dict(bytes=len(source), sha256=records[i]['sha256']), artifacts=artifacts,
                decoder=dict(sha256=m.identity(b/'decoder')['sha256']))
            (leaf/'root.json').write_bytes(m.canonical(root))
            binding = dict(format=m.PROFILE+'_LEAF', document=records[i], kind='RLB2_RLR2_EF',
                           index_root=m.identity(leaf/'root.json'), artifacts=artifacts)
            shutil.rmtree(construction)
        (leaf/'binding.json').write_bytes(m.canonical(binding))
    leaves = [dict(doc_id=i, binding=m.identity(b/'leaves'/str(i)/'binding.json')) for i in range(stop)]
    cid, sid = m.global_ids(records)
    if generation == 0:
        assert cid == '4765ce3496058719db1436c56a33b325b16dc1c4cdfecd86664d0bdfb94dc950'
        assert sid == 'a21e00fea40c4f7fb31b22c86400436acd90dba6e6efba855b4395ce7d162e09'
    for layout, sizes in [('A', [3, 3, 3]), ('B', [2, 4, 3])]:
        if generation:
            sizes.append(3)
        blocks, first = [], 0
        for ordinal, size in enumerate(sizes):
            block = dict(format=m.PROFILE+'_BLOCK', ordinal=ordinal, first_doc_id=first,
                         document_count=size, leaves=leaves[first:first+size])
            raw = m.canonical(block)
            path = b/'blocks'/(m.sha(raw)+'.json')
            if path.exists():
                assert path.read_bytes() == raw
            else:
                with path.open('xb') as out:
                    out.write(raw)
            blocks.append(dict(ordinal=ordinal, first_doc_id=first, document_count=size, manifest=m.identity(path)))
            first += size
        parent = None
        if generation:
            old_path = b/'roots'/(layout+'0.json')
            old = m.decode(old_path.read_bytes())
            assert old['documents'] == records[:9] and old['leaves'] == leaves[:9]
            parent = dict(sha256=m.sha(old_path.read_bytes()), research_composition_root_id=old['research_composition_root_id'], document_count=9)
        root = dict(format=m.PROFILE, historical_reference=m.HIST, generation=generation, documents=records,
            leaves=leaves, blocks=blocks, runtime_corpus_id=cid, source_manifest_id=sid, parent=parent,
            dependencies=dict(decoder=m.identity(b/'decoder'), code=m.code_identity(ref)))
        root['research_composition_root_id'] = m.root_id(root)
        with (b/'roots'/(layout+str(generation)+'.json')).open('xb') as out:
            out.write(m.canonical(root))
    print(json.dumps(dict(stage=a.stage, document_count=stop, new_leaf_slots=list(range(start, stop)),
                         runtime_corpus_id=cid, source_manifest_id=sid, raw_source_archive_bytes=0), sort_keys=True))


if __name__ == '__main__':
    main()
