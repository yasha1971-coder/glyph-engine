"""Bounded physical builder: uses unchanged frozen RLB2/RankV2 and EF geometry."""
import hashlib, shutil, struct, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'experiments/core_l10_composition_v0'))
from fixture import module

def sha(raw): return hashlib.sha256(raw).hexdigest()
def build_object(source, leaf, ref, decoder):
    if not 0 < len(source) <= 256: raise ValueError('bounded nonempty source required')
    if subprocess.check_output(['git','-C',str(ref),'rev-parse','HEAD'],text=True).strip()!='c33cb493bc75f13ee4a904fabb190a29a196c94b': raise ValueError('wrong frozen reference')
    subprocess.run(['git','-C',str(ref),'diff','--exit-code','HEAD','--'],check=True,capture_output=True)
    leaf.mkdir()
    b=leaf.parent
    m=module(REPO/'experiments/core_l10_composition_v0/producer.py','ingest_build_helpers')
    sys.path.insert(0,str(REPO/'experiments/loc_text_ef_v0'))
    from real_enwik8_ef_physical_gate import build, verify
    container=module(ref/'tools/rlbwt_container_v2.py','ingest_frozen_container')
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
        source=dict(bytes=len(source), sha256=sha(source)), artifacts=artifacts,
        decoder=dict(sha256=m.identity(decoder)['sha256']))
    (leaf/'root.json').write_bytes(m.canonical(root))
    shutil.rmtree(construction)
