# PIZZA_SENTINEL_LOCATE_500MB_V1

Status:
PASS

Corpus:
Pizza & Chili english 500MB sentinel-safe prefix

Query:
Ten Days that Shook the World

Pipeline:
text query
text_to_hex_queries.py
query_fm_batch_v1
FM interval
SA offset lookup
snippet extraction

FM result:
123701937 123701938 1

SA index:
123701937

Resolved offset:
53

Snippet:
This etext was produced by Normal Wolcott.

Ten Days that Shook the World

by John Reed

Conclusion:
Sentinel-safe pipeline supports the full evidence loop:
text query -> FM count -> SA offset -> human-readable snippet.
