import json, os, pathlib, random, subprocess, sys, tempfile, time

exe = str(pathlib.Path(sys.argv[1]).resolve())
rng = random.Random(1977)
rows = []
def run(args, ok=True):
    p = subprocess.run([exe, *map(str,args)], capture_output=True, timeout=30)
    assert (p.returncode == 0) == ok, (args, p.returncode, p.stderr.decode(errors='replace'))
    for marker in (b'AddressSanitizer', b'LeakSanitizer', b'runtime error:'):
        assert marker not in p.stderr, p.stderr
    return p.stdout

with tempfile.TemporaryDirectory(prefix='glyph-native-tests-') as tmp:
    d = pathlib.Path(tmp)
    cases = [b'', b'\0', b'aaaaa', bytes(range(256))*4097,
             b'x'*(1048576-1)+b'ABC\0ABC', rng.randbytes(1048576+71)]
    for i, data in enumerate(cases):
        src, arc, dst = [d/f'{i}.{s}' for s in ('src','arc','restored')]
        src.write_bytes(data)
        run(['pack',src,arc]); run(['verify',arc]); run(['restore',arc,dst])
        assert dst.read_bytes() == data
        second=d/f'{i}.second';run(['pack',src,second]);assert arc.read_bytes()==second.read_bytes()
        for query in (b'\0',b'aa',b'ABC',bytes(range(256))):
            expect=sum(data.startswith(query,k) for k in range(len(data)))
            output=run(['count',arc,query.hex()])
            assert f'count={expect} '.encode() in output,(i,query,output,expect)
        run(['pack',src,arc],False); run(['restore',arc,dst],False)
        assert dst.read_bytes()==data
        rows.append({'case':f'roundtrip-search-{i}','bytes':len(data),'pass':True})
    good=(d/'5.arc').read_bytes()
    mutations=[good[:n] for n in (0,1,7,8,20,51,len(good)-1,len(good)-44)]
    mutations += [good+b'junk',b'BADMAGIC'+good[8:]]
    for offset in (8,12,16,20,52,len(good)-1):
        v=bytearray(good);v[offset]^=255;mutations.append(bytes(v))
    for i, data in enumerate(mutations):
        arc=d/'bad.arc';arc.write_bytes(data);dst=d/'must-not-publish'
        run(['verify',arc],False);run(['restore',arc,dst],False)
        assert not dst.exists()
        assert not list(d.glob('.glyph-native-*'))
        rows.append({'case':f'corruption-{i}','pass':True})
    for query in ('','0','zz','00'*4097):
        run(['count',d/'5.arc',query],False)
    link=d/'input-link';link.symlink_to(d/'5.src')
    run(['pack',link,d/'no.arc'],False)
    linkout=d/'out-link';linkout.symlink_to(d/'5.src')
    run(['restore',d/'5.arc',linkout],False)
    fifo=d/'fifo';os.mkfifo(fifo);run(['pack',fifo,d/'fifo.arc'],False)
    rows.append({'case':'invalid-query-symlink-fifo','pass':True})
print(json.dumps({'executable':exe,'groups':len(rows),'passed':True,'rows':rows},indent=2))
