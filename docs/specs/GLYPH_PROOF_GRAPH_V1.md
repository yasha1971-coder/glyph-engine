# GLYPH_PROOF_GRAPH_V1

Status: executable proof graph  
Version: 1  
Date: 2026-07-13

## Root inputs

- authoritative source bytes;
- ordered document table;
- binary query bytes;
- declared format versions;
- declared boundary policy.

## Graph

    P1  Virtual Sentinel Total Order
     |
     +----> P2  Suffix Array Validity
     |       |
     |       +----> P3  Suffix BWT Relation
     |                 |
     |                 +----+
     |                      |
    P4  Canonical Corpus Identity
     |                      |
     +----------+-----------+
                |
                v
    P5  FM Token / Rank / LF Consistency
                |
                v
    P6  FM Backward Search Exactness
                |
                v
    P7  Locate Coordinate Exactness
                |
         +------+------+
         |             |
         v             v
    P8 Binary       P9 Document
    Query Transport Boundary Semantics
         \             /
          \           /
           v         v
          P10 Replay Determinism
                    |
                    v
          P11 Bundle Completeness
                    |
                    v
          P12 Verify Chain Closure
                    |
                    v
          GLYPH PROOF GRAPH OK
                    |
                    v
                VERIFY OK

## Obligations

| Proof | Establishes | Direct dependencies |
|---|---|---|
| P1 | Virtual sentinel token order | none |
| P2 | Suffix-array permutation and lexical validity | P1 |
| P3 | Exact suffix-array/BWT relation | P1, P2 |
| P4 | Canonical corpus and document identity | none |
| P5 | FM token, C, rank, and LF consistency | P2, P3, P4 |
| P6 | Exact binary backward search | P5 |
| P7 | Exact canonical locate coordinates | P4, P5, P6 |
| P8 | Binary-safe query transport | P1, P6 |
| P9 | Document-local match semantics | P4, P6, P7, P8 |
| P10 | Deterministic authoritative replay artifact | P4, P6, P7, P8, P9 |
| P11 | Complete self-contained portable bundle | P4, P7, P8, P9, P10 |
| P12 | Executable closure of P1 through P11 | P1 through P11 |

## End-to-end chain

    source bytes
      -> canonical corpus identity        P4
      -> suffix-array validity            P1, P2
      -> suffix BWT                       P3
      -> FM rank and LF                   P5
      -> backward search                  P6
      -> locate coordinates               P7
      -> binary query transport           P8
      -> document-local filtering         P9
      -> deterministic artifact           P10
      -> complete portable bundle         P11
      -> executable proof closure         P12
      -> VERIFY OK

## Failure localization

- sentinel collision or byte-order error: P1;
- invalid suffix order or permutation: P2;
- rotation/suffix BWT mismatch: P3;
- corpus or document identity mismatch: P4;
- C/rank/LF inconsistency: P5;
- incorrect interval or count: P6;
- incorrect coordinate: P7;
- NUL truncation or query transport corruption: P8;
- cross-document match: P9;
- nondeterministic artifact: P10;
- missing bundle dependency or integrity failure: P11;
- skipped obligation or false `VERIFY OK`: P12.

## Executable representation

The executable graph is defined by:

    tools/check_verify_chain_v1.py
    tools/run_glyph_proof_graph_v1.sh

The machine-readable result is:

    benchmarks/results/GLYPH_PROOF_GRAPH_V1.json

`VERIFY OK` is forbidden unless the result contains twelve ordered PASS nodes.
