#!/usr/bin/env python3
"""Explain the preserved selection FAIL without running either index engine."""
from array import array
import bisect
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
from independent_oracle import select_sa_prefix

work = Path(sys.argv[1])
oracle = json.loads((work / "oracle/oracle.json").read_text())
data = Path(oracle["corpus"]["path"]).read_bytes()
assert hashlib.sha256(data).hexdigest() == oracle["corpus"]["sha256"]
positions = array("I")
with Path(oracle["all_matches"]["path"]).open("rb") as src:
    positions.fromfile(src, oracle["count"])
if sys.byteorder != "little":
    positions.byteswap()
frequencies = Counter(data)
ordered = sorted(frequencies, key=lambda c: (-frequencies[c], c))
mapping = bytearray(256)
for rank, byte in enumerate(ordered, 1):
    mapping[byte] = rank
translated = data.translate(bytes(mapping))
expected_frequency_order = select_sa_prefix(translated, positions, 100)
lines = (work / "control/Move-r-preflight.stdout").read_text().splitlines()[1:]
records = []
for limit, line in zip((1, 10, 100), lines):
    reply = json.loads(line)
    offsets = reply["offsets"]
    bytes_ok = all(0 <= x <= len(data) - 3 and data[x:x + 3] == b"the" for x in offsets)
    membership_ok = all(positions[bisect.bisect_left(positions, x)] == x for x in offsets)
    records.append({"limit": limit, "count_matches_direct_search": reply["count"] == oracle["count"],
                    "all_returned_bytes_PASS": bytes_ok, "all_offsets_in_full_direct_list": membership_ok,
                    "exact_frequency_order_prefix_matches": offsets == expected_frequency_order[:limit],
                    "raw_byte_order_prefix_matches": sorted(offsets) == oracle["offsets_by_limit"][str(limit)]})
assert all(x["count_matches_direct_search"] and x["all_returned_bytes_PASS"]
           and x["all_offsets_in_full_direct_list"] and x["exact_frequency_order_prefix_matches"] for x in records)
result = {"diagnosis": "Move-r default reorder_alphabet=true sorts its internal alphabet by decreasing frequency",
          "raw_byte_prefix_contract": "FAIL; original attempt retained",
          "underlying_exact_count_and_returned_bytes": "PASS in all three replies",
          "hypothesis_confirmed_by_independent_translated_suffix_comparison": True,
          "records": records, "new_index_engine_queries": 0, "performance_samples": 0}
print(json.dumps(result, sort_keys=True, indent=2))
