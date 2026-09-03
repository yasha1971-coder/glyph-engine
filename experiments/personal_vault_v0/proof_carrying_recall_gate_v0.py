#!/usr/bin/env python3
import json,shutil,subprocess,tempfile
from pathlib import Path

HERE=Path(__file__).resolve().parent
CLI=HERE/'glyph_vault_cli_v0.py'
FETCH=HERE/'fetch_corpora.sh'
MAKE=HERE/'proof_carrying_recall_v0.py'
VERIFY=HERE/'verify_proof_carrying_recall_v0.py'


def run_json(*args):
    return json.loads(subprocess.check_output([str(x) for x in args],text=True))

with tempfile.TemporaryDirectory(prefix='glyph-proof-carrying-recall-v0-') as td:
    t=Path(td); raw=t/'raw'; src=t/'source'; vault=t/'vault'; receipt=t/'receipt.json'
    subprocess.check_call(['bash',str(FETCH),str(raw)])
    src.mkdir(); shutil.copytree(raw/'canterbury',src/'canterbury')
    run_json('python3',CLI,'init',vault)
    add=run_json('python3',CLI,'add',vault,src)
    assert add['objects']==11 and add['source_deleted'] is False
    run_json('python3',CLI,'verify',vault)

    made=run_json('python3',MAKE,vault,'--pattern','Wonderland','--out',receipt)
    assert made['path']=='canterbury/alice29.txt' and made['occurrences']>=1

    # Critical property: destroy the original source tree before verification.
    shutil.rmtree(src)
    assert not src.exists()
    verified=run_json('python3',VERIFY,vault,receipt)
    assert verified['ok'] and verified['original_source_required'] is False and verified['ai_required'] is False
    assert verified['path']=='canterbury/alice29.txt'

    # Hostile mutation of the receipt must fail independently.
    bad=json.loads(receipt.read_text())
    bad['evidence']['object_offsets'][0]+=1
    bad_path=t/'bad-receipt.json'; bad_path.write_text(json.dumps(bad,sort_keys=True,separators=(',',':'))+'\n')
    p=subprocess.run(['python3',str(VERIFY),str(vault),str(bad_path)],text=True,capture_output=True)
    assert p.returncode!=0 and 'VERIFY_FAIL' in (p.stderr+p.stdout)

    print(json.dumps({
      'format':'GLYPH_PROOF_CARRYING_RECALL_GATE_V0',
      'all_checks_passed':True,
      'source_tree_removed_before_verification':True,
      'ai_not_required_for_verification':True,
      'committed_root_bound':True,
      'object_restored_bit_exact':True,
      'literal_offsets_replayed':True,
      'hostile_receipt_mutation_rejected':True,
      'wow_claim':'A human/AI recall result can carry a compact receipt that a separate verifier replays against the committed GLYPH Vault after the original source tree is gone.'
    },sort_keys=True))
