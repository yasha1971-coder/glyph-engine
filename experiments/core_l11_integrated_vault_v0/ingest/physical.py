"""Bounded physical builder: uses unchanged frozen RLB2/RankV2 and EF geometry."""
import hashlib, shutil, subprocess, sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
REPO=HERE.parents[2]
sys.path.insert(0,str(REPO/'experiments/core_l10_composition_v0'))
from fixture import module

def sha(raw): return hashlib.sha256(raw).hexdigest()
def build_object(source, leaf, ref, decoder,source_limit=4096):
    if not 0 < len(source) <= source_limit: raise ValueError('bounded nonempty source required')
    if subprocess.check_output(['git','-C',str(ref),'rev-parse','HEAD'],text=True).strip()!='c33cb493bc75f13ee4a904fabb190a29a196c94b': raise ValueError('wrong frozen reference')
    subprocess.run(['git','-C',str(ref),'diff','--exit-code','HEAD','--'],check=True,capture_output=True)
    leaf.mkdir()
    m=module(REPO/'experiments/core_l10_composition_v0/producer.py','ingest_build_helpers')
    sys.path.insert(0,str(REPO/'experiments/loc_text_ef_v0'))
    from real_enwik8_ef_physical_gate import build, verify
    construction = leaf/'construction'
    construction.mkdir()
    raw = construction/'source.raw'
    raw.write_bytes(source)
    sap = construction/'source.sa'
    bwt = construction/'source.bwt'
    subprocess.run([str(ref/'build/build_sa_binary_v1'),str(raw),str(sap)],check=True,capture_output=True,timeout=120)
    subprocess.run([str(ref/'build/build_bwt_binary_v1'),str(raw),str(sap),str(bwt)],check=True,capture_output=True,timeout=120)
    subprocess.run([sys.executable, str(ref/'tools/rlbwt_container_v2.py'), 'encode', str(bwt), str(leaf/'rlb2')],
                   capture_output=True, check=True)
    subprocess.run([sys.executable, str(ref/'tools/rlbwt_rank_v2.py'), 'build', str(leaf/'rlb2'), str(leaf/'rlr2'),
                    '--rank-step', '4' if len(source)<=256 else '128' if len(source)<=4096 else '4096'], capture_output=True, check=True)
    geometry = build(sap, len(source)+1, 2 if len(source)<=256 else 128, leaf/'ef')
    assert verify(sap, leaf/'ef', geometry)['sampled_sa_records_verified'] > 0
    def identity(path):
        if source_limit<=4096:return m.identity(path)
        size=path.stat().st_size
        if path.is_symlink() or not 0<size<=128<<20:raise ValueError('scaled component admission')
        h=hashlib.sha256();total=0
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1<<20),b''):
                total+=len(chunk)
                if total>size:raise ValueError('changing component')
                h.update(chunk)
        if total!=size:raise ValueError('component truncated')
        return dict(bytes=total,sha256=h.hexdigest())
    artifacts = {name: identity(leaf/name) for name in ('rlb2', 'rlr2', 'ef')}
    root = dict(schema='GLYPH_CORE_L8_PINNED_SOURCE_MATERIALIZATION_V0', frozen_commit=m.REF,
        source=dict(bytes=len(source), sha256=sha(source)), artifacts=artifacts,
        decoder=dict(sha256=m.identity(decoder)['sha256']))
    (leaf/'root.json').write_bytes(m.canonical(root))
    shutil.rmtree(construction)
