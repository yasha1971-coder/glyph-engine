"""Synthetic replay of recovered preservation; not a full indexed Vault gate."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT/'experiments/personal_memory_1tb_v2'
sys.path.insert(0, str(OLD))
sys.path.insert(0, str(OLD/'tests'))
sys.path.insert(0, str(HERE))
import verified_hybrid_archive as hybrid
from test_verified_hybrid_archive import make_inventory
from storage_budget import assess

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=False)
reports = []
for name, mixed in [('random', False), ('mixed', True)]:
    work = a.output/name
    work.mkdir()
    source = work/'source'
    source.mkdir()
    (source/'random.bin').write_bytes(random.Random(94718).randbytes(1 << 20))
    if mixed:
        (source/'text.txt').write_bytes((b'GLYPH exact preservation and search\n'*40000)[:1 << 20])
    make_inventory(source, work/'inventory')
    hybrid.build(work/'inventory', work/'candidate', 0)
    hybrid.restore(work/'candidate', work/'restore')
    original = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in source.iterdir()}
    restored = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in (work/'restore').iterdir()}
    assert original == restored
    r = assess(work/'candidate', sum(f.stat().st_size for f in source.iterdir()))
    assert r['storage_budget_pass'] == mixed
    reports.append(dict(case=name, budget=r, restored_file_sha256=restored,
                        byte_perfect_compare=True))
(a.output/'result.json').write_text(json.dumps(reports, indent=2)+'\n')
print(json.dumps(reports), flush=True)
