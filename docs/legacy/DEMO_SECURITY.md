Problem:
grep on 4GB binary corpus is slow

Solution:
GLYPH exact retrieval

Steps:
1. build index
2. query pattern
3. get exact positions

Result:
- deterministic
- fast
- exact

Comparison:
grep → scan
glyph → index lookup


## Live server demo result

Command:

    curl -s http://127.0.0.1:18080/health
    ./glyph_cli.py --hex "$(xxd -p -c 999999 /tmp/query_41905.bin)"

Result:

- health: ok
- outcome: EXACT_MULTI
- total_merged: 8
- query_time_sec: ~0.0027 sec
- fm_calls: 64
- deterministic byte-exact shortlist returned

Meaning:

GLYPH service layer is operational.
The HTTP/CLI demo proves exact byte retrieval over prepared segmented artifacts.
