#!/usr/bin/env python3
"""Compare the real RC1 CLI and candidate byte-for-byte, preserving receipts."""
from pathlib import Path
import argparse,array,hashlib,importlib.util,json,struct,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[1]
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--work',type=Path,required=True);args=ap.parse_args()
    work=args.work;work.mkdir(parents=True,exist_ok=False)
    legacy=ROOT/'tests/reference/build_locate_fixture_rc1.py'
    candidate=ROOT/'tools/build_locate_fixture_v1.py'
    reference=json.loads((legacy.parent/'RC1.json').read_text())
    # Git may check .py out as CRLF on Windows; the reference's source content
    # remains the frozen LF-normalized RC1 member.
    assert hashlib.sha256(legacy.read_bytes().replace(b'\r\n',b'\n')).hexdigest()==reference['source_sha256']
    receipts=[];checks=[]
    def run(script,folder,bwt,sa,step):
        folder.mkdir();argv=[sys.executable,str(script),'--bwt',str(bwt),'--sa',str(sa),'--out-dir',str(folder),'--checkpoint-step','8192']
        if step is not None:argv+=['--sample-step',str(step)]
        start=time.perf_counter();p=subprocess.run(argv,capture_output=True)
        (folder/'stdout').write_bytes(p.stdout);(folder/'stderr').write_bytes(p.stderr)
        receipts.append({'argv':argv,'returncode':p.returncode,'seconds':time.perf_counter()-start,'stdout':str(folder/'stdout'),'stderr':str(folder/'stderr')})
        (work/'attempts.json').write_text(json.dumps(receipts,indent=2)+'\n')
        return p
    cases=[(n,s) for n in [0,1,2,3,127,128,129,255,256,257] for s in [1,16,128,n+7,0,-1]]
    cases += [(n,s) for n in [262143,262144,262145] for s in [3,128,129]]
    cases += [(129,None)]
    for ordinal,(n,step) in enumerate(cases):
        d=work/f'normal-{ordinal:03d}';d.mkdir()
        sa=d/'sa.bin';bwt=d/'bwt.bin'
        # Exercise the entire unsigned range as a format test, not a suffix
        # array construction test; RC1 has no coordinate-range validation.
        values=[(i*2654435761+0x80000000)&0xffffffff for i in range(n)]
        with sa.open('wb') as f:
            for value in values:f.write(struct.pack('<I',value))
        bwt.write_bytes(bytes((i%251)+1 for i in range(n)))
        old=run(legacy,d/'old',bwt,sa,step);new=run(candidate,d/'new',bwt,sa,step)
        assert old.returncode==new.returncode==0,(n,step)
        assert old.stdout==new.stdout,(n,step,'stdout')
        filename=f'locate_core_s{16 if step is None else step}.bin'
        for file in ['fm_core.bin',filename]:
            assert (d/'old'/file).read_bytes()==(d/'new'/file).read_bytes(),(n,step,file)
        data=(d/'new'/filename).read_bytes();magic,size,stride,count=struct.unpack_from('<4sQIQ',data)
        expected_step=max(1,16 if step is None else step)
        rows=list(range(0,n,expected_step))
        assert (magic,size,stride,count)==(b'LOC1',n,expected_step,len(rows))
        assert len(data)==24+16*len(rows)
        assert list(struct.iter_unpack('<QQ',data[24:]))==[(i,values[i]) for i in rows]
        checks.append({'n':n,'step':step,'ok':True,'loc1_sha256':digest(d/'new'/filename),'fmv1_sha256':digest(d/'new/fm_core.bin')})
        (work/'checks.json').write_text(json.dumps(checks,indent=2)+'\n')
    for case,sa_data,bwt_data,error in [
        ('bad-sa-length',b'abc',b'x','SA file size is not divisible by 4'),
        ('sa-bwt-mismatch',struct.pack('<3I',0,1,2),b'abcd','SA length 3 != BWT length 4')]:
        d=work/case;d.mkdir();sa=d/'sa.bin';bwt=d/'bwt.bin';sa.write_bytes(sa_data);bwt.write_bytes(bwt_data)
        old=run(legacy,d/'old',bwt,sa,128);new=run(candidate,d/'new',bwt,sa,128)
        assert old.returncode!=0 and new.returncode!=0
        assert error.encode() in old.stderr and error.encode() in new.stderr
        assert not (d/'new/fm_core.bin').exists() and not (d/'new/locate_core_s128.bin').exists()
        checks.append({'case':case,'rejected_old_and_new':True,'ok':True})
    spec=importlib.util.spec_from_file_location('candidate_locate',candidate);module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    api=work/'array-api.bin';api.write_bytes(struct.pack('<4I',0,0x80000000,0xffffffff,7))
    values=module.read_sa_u32(api);assert isinstance(values,array.array)
    assert list(values[::2])==[0,0xffffffff]
    sliced=values[::2];sliced.byteswap();sliced.byteswap();assert list(sliced)==[0,0xffffffff]
    with (work/'array-api-copy.bin').open('wb') as f:values.tofile(f)
    assert (work/'array-api-copy.bin').read_bytes()==api.read_bytes()
    result={'ok':True,'normal_cases':len(cases),'rejection_cases':2,'array_api_for_search_preserved':True,'reference':reference,'checks':checks}
    (work/'PASS.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ['ok','normal_cases','rejection_cases','array_api_for_search_preserved']}))
if __name__=='__main__':main()
