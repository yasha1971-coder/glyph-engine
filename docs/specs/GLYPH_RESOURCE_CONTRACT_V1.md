GLYPH RESOURCE CONTRACT V1

Purpose

Define runtime resources that must exist before GLYPH can enter READY state.

Capability Contract describes what the machine can do.

Resource Contract describes what is available now.

Required resource checks

backend_binary_exists

manifest_exists

fm_artifact_exists

bwt_artifact_exists

index_directory_exists

Current required resources

backend binary:

build/query_fm_server_v1

manifest:

examples/mini/out/manifest.json

FM artifact:

examples/mini/out/fm.bin

BWT artifact:

examples/mini/out/bwt.bin

Rule

Missing required resource means:

READY=false

engine must fail hard

no query serving

Principle

Resources verified before READY.

Silent resource drift is worse than hard failure.

Future extensions

ram_available

disk_free

corpus_present

artifact_size_expected

artifact_checksum_expected

mmap_open_success

page_cache_state

file_descriptor_budget

shard_set_complete

Interpretation

GLYPH readiness is not only process startup.

GLYPH readiness requires:

runtime compatibility

capability compatibility

retrieval contract compatibility

resource availability
