# GLYPH — Live Demo

Command:

curl -s http://127.0.0.1:18080/health

./glyph_cli.py --hex "$(xxd -p -c 999999 /tmp/query_41905.bin)"

Output:

- outcome: EXACT_MULTI
- total_merged: 8
- query_time_sec: ~0.0027 sec
- deterministic byte-exact shortlist returned

Meaning:

GLYPH returns exact byte matches from pre-indexed data.

No ranking. No fuzzy. No interpretation.
