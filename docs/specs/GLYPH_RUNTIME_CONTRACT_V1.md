GLYPH RUNTIME CONTRACT V1

Purpose

Runtime compatibility boundary.

Engine startup requires:

Capability Contract V1

Retrieval Contract V1

Manifest integrity PASS

FM artifact integrity PASS

Required runtime fields

runtime_os

runtime_kernel

cpu_arch

compiler_family

query_protocol_version

server_protocol_version

http_protocol_version

retrieval_contract_version

capability_contract_version

Rules

Contract mismatch:

engine refuses startup.

Silent drift forbidden.

Hard failure preferred.

Determinism preserved.
