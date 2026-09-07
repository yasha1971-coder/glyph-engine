# Personal Vault V0 — closed-loop experiment

This branch is based on the measured Runtime V2 research line, not on main. It does not change RLB2, RLR2, LOC1, or query semantics.

Goal: test whether 30 independent Canterbury/Calgary/Artificial/Miscellaneous files can be treated as one logical Personal Vault while preserving per-object identity.

Gates:
1. concatenate 30 objects into one canonical byte corpus plus an object map;
2. build canonical SA/BWT and Runtime V2 (RLB2 + RLR2 + LOC1);
3. reconstruct the corpus from BWT and require byte-identical equality;
4. restore every object by its canonical range and require original SHA-256/content;
5. run >=20 unique exact queries through Runtime V2 and require exact offsets;
6. construct >=20 byte strings that exist only by crossing adjacent object boundaries. Raw concatenated GLYPH is expected to find them; this is recorded as proof that Personal Vault needs a mandatory object-boundary result filter.

Important: V0 does NOT claim that RLB2 alone is a general archive format. The inverse-BWT gate proves lossless recoverability from canonical BWT state; a later vault format must define whether reconstruction metadata is retained inside the packaged vault and account for every byte in its ratio.

No source file deletion is permitted by this experiment.
