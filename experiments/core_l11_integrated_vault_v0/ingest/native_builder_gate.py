"""Native construction parity and 4KiB primitive experiment; Vault cap unchanged."""
import hashlib, json, random, struct, subprocess, sys, tempfile, time
from pathlib import Path
from physical import build_object, REPO
from fixture import module

def sha(b): return hashlib.sha256(b).hexdigest()
def run(reference,decoder,output):
    started=time.monotonic(); reference=Path(reference); decoder=Path(decoder)
    m=module(REPO/'experiments/core_l10_composition_v0/producer.py','native_parity_helpers')
    container=module(reference/'tools/rlbwt_container_v2.py','native_parity_container')
    sys.path.insert(0,str(REPO/'experiments/loc_text_ef_v0'))
    from real_enwik8_ef_physical_gate import build
    rng=random.Random(881)
    cases=[b'\0',b'\xff',b'ababa',bytes(range(256)),b'x'*256,bytes(rng.randrange(256) for _ in range(255))]
    with tempfile.TemporaryDirectory(prefix='glyph-native-parity-') as tmp:
        tmp=Path(tmp)
        for i,source in enumerate(cases):
            leaf=tmp/str(i); build_object(source,leaf,reference,decoder)
            mapped=tuple(x+1 for x in source)+(0,)
            sa=sorted(range(len(mapped)),key=lambda j:mapped[j:])
            symbols=[256 if j==0 else source[j-1] for j in sa]
            payload=struct.pack('<'+'H'*len(symbols),*symbols)
            fnv=container.Fnv1a64(); fnv.update(payload)
            bwt=tmp/'oracle.bwt'; bwt.write_bytes(container.BWT_HEADER.pack(container.BWT_MAGIC,container.BWT_VERSION,len(source),len(mapped),257,256,16,len(payload),fnv.value)+payload)
            subprocess.run([sys.executable,str(reference/'tools/rlbwt_container_v2.py'),'encode',str(bwt),str(tmp/'oracle.rlb2')],check=True,capture_output=True)
            subprocess.run([sys.executable,str(reference/'tools/rlbwt_rank_v2.py'),'build',str(tmp/'oracle.rlb2'),str(tmp/'oracle.rlr2'),'--rank-step','4'],check=True,capture_output=True)
            sap=tmp/'oracle.sa'; sap.write_bytes(b'\0'*60+struct.pack('<'+'I'*len(sa),*sa))
            build(sap,len(sa),2,tmp/'oracle.ef')
            for name in ('rlb2','rlr2','ef'):
                path=tmp/('oracle.'+name)
                assert (leaf/name).read_bytes()==path.read_bytes(),(i,name)
                path.unlink()
        source=bytes(rng.randrange(256) for _ in range(4096))
        leaf=tmp/'large'; build_object(source,leaf,reference,decoder)
        from producer import QueryState
        root=leaf/'root.json'; pin=sha(root.read_bytes())
        state=QueryState(root,pin,leaf/'rlb2',leaf/'rlr2',leaf/'ef',decoder,reference)
        try:
            for i,pattern in enumerate((source[2045:2062],b'not-present-native-4k')):
                claim=state.query(pattern,256)
                expected=[j for j in range(len(source)) if source.startswith(pattern,j)]
                assert claim['count']==len(expected) and claim['offsets']==expected
                materialized=tmp/('restore'+str(i)); receipt=tmp/'receipt.json';receipt.write_bytes(m.canonical(claim))
                args=[sys.executable,str(REPO/'experiments/loc_text_ef_v0/trusted_materialize_v0.py'),'--manifest',str(root),'--root-sha256',pin,'--rlb2',str(leaf/'rlb2'),'--rlr2',str(leaf/'rlr2'),'--ef',str(leaf/'ef'),'--decoder',str(decoder),'--output',str(materialized),'--threads','2','--mode','rank']
                subprocess.run(args,check=True,capture_output=True,timeout=90)
                assert materialized.read_bytes()==source
                args=[sys.executable,str(REPO/'experiments/core_l9_exact_query_v0/verifier.py'),'--manifest',str(root),'--root-sha256',pin,'--rlb2',str(leaf/'rlb2'),'--rlr2',str(leaf/'rlr2'),'--ef',str(leaf/'ef'),'--decoder',str(decoder),'--materialized',str(materialized),'--receipt',str(receipt),'--pattern-hex',pattern.hex(),'--limit','256','--research',str(REPO),'--reference',str(reference)]
                subprocess.run(args,check=True,capture_output=True,timeout=90)
        finally: state.close()
        result=dict(status='NATIVE_BUILDER_PARITY_AND_4K_PRIMITIVE_PASS',parity_sources=len(cases),physical_components_identical=18,large_source_bytes=4096,large_query_replays=2,byte_perfect_restore=True,vault_file_cap_unchanged=256,wall_seconds=time.monotonic()-started)
        Path(output).write_text(json.dumps(result,sort_keys=True)+'\n'); print(json.dumps(result,sort_keys=True))
