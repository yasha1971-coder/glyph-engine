# RLBWT Binary-Safe Container V2 — enwik8

- Status: measured experimental container result
- Corpus: canonical Matt Mahoney `enwik8`
- Corpus bytes: 100,000,000
- Corpus MD5: `a1fa5ffddb56f4953e226637dabbb36a`
- Corpus SHA-256: `2b49720ec4d78c3c9fabaee6e4179a5e997302b3a70029f30f2d582218c024a8`
- Source commit: `ebe825eb44c650f38dccf3dea7d15a5c83a7cded`

## Measured container result

| Metric | Value |
|---|---:|
| Canonical BWT rows | 100,000,001 |
| BWT runs | 36,661,040 |
| RLB2 payload bytes | 73,364,945 |
| RLB2 header bytes | 160 |
| Complete RLB2 file bytes | 73,365,105 |
| RLB2 bytes / corpus bytes | 0.733651050 |
| Corpus bytes / RLB2 bytes | 1.363045824 |
| Saving versus source | 26,634,895 bytes |
| Encode wall time | 85.330 s |
| Encode peak RSS | 56.836 MiB |
| Inspect wall time | 4.930 s |
| Decode wall time | 68.040 s |
| Decode peak RSS | 34.598 MiB |

The decoded canonical BWT is bit-identical to the input BWT:

`d00b44f165ba78dc11bdc048f8578021003f17fa19587fea944e499fb6514195`

The measured RLB2 artifact SHA-256 is:

`4ca1218cc84b38539b6d12b0aacdfa30b21e387b24e4f8abaa828f1a6fffc0d3`

The modeled payload and the encoded payload are exactly equal.

## Binary-safe result

The container preserves the 257-symbol alphabet and logical sentinel 256.
Adaptive escape encoding adds one byte over the impossible one-byte
257-symbol run-head reference on this corpus.

RLB2 is additive. It does not redefine RLB1 or RLR1.

## Hostile gate

- Positive cases: 4
- Rejected mutations: 33
- All mutations rejected: true
- Deterministic encode verified: true
- Different working directory verified: true
- Tool SHA-256: `80d6e8c9714c2cd3acd40b58f3e49a7bcf8ec4d16778929043fb3671348040ac`
- Checker SHA-256: `eb4ed5a6f64086e90b3c2d7ceee189af893e1348439d2f99c8c44404398a6483`

## Remaining budget

To keep a future complete indexed runtime below the 100,000,000-byte corpus,
all additional rank, locate, metadata and evidence structures together must
fit within 26,634,895 bytes.

## Claim boundary

This is a measured lossless binary-safe compressed-BWT container result.

It is not yet a complete query runtime. It does not establish count, locate,
coordinate retrieval, byte verification, or a full runtime below 1x corpus.
The reported Python timings characterize the experimental reference tool,
not a compiled production implementation.

The next gate is the exact enwik8 rank, locate and metadata byte budget.
