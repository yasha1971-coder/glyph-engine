GLYPH CAPABILITY CONTRACT V1

Purpose

Runtime capability is part of deterministic retrieval.

Determinism depends not only on corpus and artifacts.

Determinism also depends on execution capability.

Required capability fields

cpu_arch

compiler_family

compiler_version

simd_capability

build_type

sa_container_version

fm_artifact_version

query_protocol_version

server_protocol_version

http_protocol_version

retrieval_contract_version

Optional capability fields

runtime_os

runtime_kernel

chunk_container_version

batch_protocol_version

Contract rule

Artifact validates runtime.

Runtime validates artifact.

Capability validates compatibility.

Compatibility mismatch creates incompatible state.

System behavior

Compatible:
execute

Incompatible:
fail hard

Unknown:
warn + compatibility mode

Determinism invariant

identical corpus

+

identical retrieval artifacts

+

identical protocol versions

+

identical capability contract

+

identical query

must produce identical retrieval output

Drift invariant

Silent capability drift

is worse

than hard failure

Examples

cpu_arch:
x86_64

compiler_family:
gcc

compiler_version:
13.x

simd_capability:
AVX2

fm_artifact_version:
FMBINv2

query_protocol_version:
GLYPH_QUERY_PROTOCOL_V1

retrieval_contract_version:
GLYPH_RETRIEVAL_CONTRACT_V1

Future extensions

runtime fingerprint

build fingerprint

binary capability fingerprint

SIMD verification layer

cross-machine deterministic verification
