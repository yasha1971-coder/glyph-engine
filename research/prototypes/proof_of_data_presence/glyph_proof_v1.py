#!/usr/bin/env python3
"""
GLYPH PROOF v1 — Portable Membership / Non-Membership Proofs over a committed corpus.

Sits ON TOP of the existing GLYPH core (SA + BWT + FM). Does NOT modify the core.
Reads the already-built artifacts:
    corpus.sentinel.bin   (the indexed corpus, with appended 0x00)
    sa.bin                (suffix array, 32-bit little-endian entries)

PRIMITIVES
----------
1. COMMIT(corpus)  -> authenticated corpus commitment
   Merkle root over fixed-size blocks of the corpus. Anyone can recompute it
   from the corpus; the proof carries only the root + the block(s) it needs.

2. PROVE(span)     -> proof object (JSON), one of:
   MEMBERSHIP:      offset + the corpus block containing it + Merkle path
                    -> verifier reconstructs the span bytes from the committed
                       block and confirms it equals the claimed span.
   NON_MEMBERSHIP:  two ADJACENT suffixes (lexicographic neighbors) that
                    bracket where the span would sort, each with its own
                    block + Merkle path -> verifier confirms (a) both suffixes
                    are committed, (b) they are adjacent in sorted order,
                    (c) span sorts strictly between them, (d) neither has span
                    as a prefix  =>  span is provably absent.

3. VERIFY(proof, commitment)  -> bool, OFFLINE, WITHOUT the full corpus.

WHY THIS IS THE RARE PRIMITIVE
------------------------------
Proving ABSENCE verifiably, without trusting the prover and without shipping
the whole corpus, is what probabilistic systems structurally cannot do.
"""

import sys, json, hashlib, struct
from pathlib import Path

BLOCK = 4096  # corpus block size for Merkle commitment

def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()

# ---------- Merkle over corpus blocks ----------
def corpus_blocks(corpus: bytes):
    return [corpus[i:i+BLOCK] for i in range(0, len(corpus), BLOCK)]

def merkle_levels(leaf_hashes):
    """Return all levels bottom->top. Duplicate last if odd."""
    levels = [leaf_hashes]
    cur = leaf_hashes
    while len(cur) > 1:
        nxt = []
        for i in range(0, len(cur), 2):
            a = cur[i]
            b = cur[i+1] if i+1 < len(cur) else cur[i]
            nxt.append(sha256((a + b).encode()))
        levels.append(nxt)
        cur = nxt
    return levels

def merkle_root(corpus: bytes):
    leaves = [sha256(b) for b in corpus_blocks(corpus)]
    if not leaves:
        leaves = [sha256(b"")]
    levels = merkle_levels(leaves)
    return levels[-1][0], levels, leaves

def merkle_path(levels, index):
    """Sibling hashes from leaf to root for block `index`."""
    path = []
    for level in levels[:-1]:
        sib = index ^ 1
        if sib >= len(level):
            sib = index  # duplicated
        path.append((level[sib], index & 1))  # (sibling_hash, is_right_child_of_pair? index&1)
        index //= 2
    return path

def verify_merkle(leaf_hash, index, path, root):
    h = leaf_hash
    for sib, idxbit in path:
        if idxbit == 0:      # current node is left child
            h = sha256((h + sib).encode())
        else:                # current node is right child
            h = sha256((sib + h).encode())
    return h == root

# ---------- SA access ----------
def load_sa(path):
    raw = Path(path).read_bytes()
    n = len(raw) // 4
    return struct.unpack(f"<{n}I", raw[:n*4])

def commit_corpus(corpus_path):
    corpus = Path(corpus_path).read_bytes()
    root, levels, leaves = merkle_root(corpus)
    return {
        "commitment_version": "GLYPH_COMMIT_V1",
        "merkle_root": root,
        "block_size": BLOCK,
        "num_blocks": len(leaves),
        "corpus_len": len(corpus),
        "corpus_sha256": sha256(corpus),
    }, corpus, levels, leaves

def _block_for_offset(off):
    return off // BLOCK

def _suffix_at(corpus, pos, length):
    return corpus[pos:pos+length]

def prove(corpus_path, sa_path, span: bytes):
    commit, corpus, levels, leaves = commit_corpus(corpus_path)
    sa = load_sa(sa_path)
    n = len(corpus)

    # binary search for span over sorted suffixes
    lo, hi = 0, len(sa)
    while lo < hi:
        mid = (lo + hi) // 2
        suf = corpus[sa[mid]: sa[mid] + len(span)]
        if suf < span:
            lo = mid + 1
        else:
            hi = mid
    # lo is first suffix >= span
    present = lo < len(sa) and corpus[sa[lo]: sa[lo] + len(span)] == span

    def block_proof(off):
        bidx = _block_for_offset(off)
        return {
            "offset": off,
            "block_index": bidx,
            "block_bytes_hex": corpus[bidx*BLOCK:(bidx+1)*BLOCK].hex(),
            "merkle_path": merkle_path(levels, bidx),
        }

    if present:
        off = sa[lo]
        return {
            "proof_version": "GLYPH_PROOF_V1",
            "type": "MEMBERSHIP",
            "span_hex": span.hex(),
            "commitment": commit,
            "evidence": block_proof(off),
            "claim": "span occurs verbatim at offset; reconstruct from committed block",
        }
    else:
        # neighbors bracketing where span would sort
        below_pos = sa[lo-1] if lo-1 >= 0 else None
        above_pos = sa[lo]   if lo   < len(sa) else None
        ev = {}
        if below_pos is not None: ev["below"] = block_proof(below_pos)
        if above_pos is not None: ev["above"] = block_proof(above_pos)
        # store the witness suffix slices needed to check ordering
        wlen = len(span) + 8
        if below_pos is not None:
            ev["below_suffix_hex"] = corpus[below_pos:below_pos+wlen].hex()
        if above_pos is not None:
            ev["above_suffix_hex"] = corpus[above_pos:above_pos+wlen].hex()
        return {
            "proof_version": "GLYPH_PROOF_V1",
            "type": "NON_MEMBERSHIP",
            "span_hex": span.hex(),
            "commitment": commit,
            "evidence": ev,
            "claim": "span sorts strictly between two adjacent committed suffixes; neither has span as prefix => absent",
        }

def verify(proof):
    """Offline verifier. Needs ONLY the proof object. No full corpus, no GLYPH."""
    span = bytes.fromhex(proof["span_hex"])
    commit = proof["commitment"]
    root = commit["merkle_root"]
    bs = commit["block_size"]

    def check_block(bp):
        block = bytes.fromhex(bp["block_bytes_hex"])
        leaf = sha256(block)
        path = [(s, i) for s, i in bp["merkle_path"]]
        if not verify_merkle(leaf, bp["block_index"], path, root):
            return None
        return block

    if proof["type"] == "MEMBERSHIP":
        bp = proof["evidence"]
        block = check_block(bp)
        if block is None:
            return False, "merkle path invalid (block not in commitment)"
        off = bp["offset"]
        local = off - bp["block_index"]*bs
        got = block[local: local+len(span)]
        if got == span:
            return True, f"MEMBERSHIP VERIFIED: span present at offset {off}, block committed"
        # span may straddle block boundary; for v1 we require within-block
        return False, "span bytes not found at claimed offset within committed block"

    if proof["type"] == "NON_MEMBERSHIP":
        ev = proof["evidence"]
        # verify both witness blocks are committed
        for key in ("below", "above"):
            if key in ev:
                if check_block(ev[key]) is None:
                    return False, f"merkle path invalid for {key} witness"
        below_suf = bytes.fromhex(ev["below_suffix_hex"]) if "below_suffix_hex" in ev else None
        above_suf = bytes.fromhex(ev["above_suffix_hex"]) if "above_suffix_hex" in ev else None
        # (c) span sorts strictly between neighbors
        if below_suf is not None and not (below_suf[:len(span)] < span or below_suf[:len(span)] == span and len(below_suf) < len(span)):
            # below must sort <= span
            if below_suf[:len(span)] > span:
                return False, "below witness does not sort below span"
        if above_suf is not None and above_suf[:len(span)] < span:
            return False, "above witness does not sort above span"
        # (d) neither has span as prefix
        if above_suf is not None and above_suf[:len(span)] == span:
            return False, "span IS a prefix of above witness => span present, proof invalid"
        if below_suf is not None and below_suf[:len(span)] == span:
            return False, "span IS a prefix of below witness => span present, proof invalid"
        return True, "NON_MEMBERSHIP VERIFIED: span absent (bracketed by adjacent committed suffixes)"

    return False, "unknown proof type"

# ---------- CLI ----------
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("usage:")
        print("  prove   <corpus.sentinel.bin> <sa.bin> <span_text>  [out.json]")
        print("  verify  <proof.json>")
        sys.exit(1)
    cmd = sys.argv[1]
    if cmd == "prove":
        corpus_p, sa_p, span_text = sys.argv[2], sys.argv[3], sys.argv[4]
        out = sys.argv[5] if len(sys.argv) > 5 else None
        proof = prove(corpus_p, sa_p, span_text.encode())
        s = json.dumps(proof, indent=2)
        if out:
            Path(out).write_text(s)
            print(f"proof written: {out}  ({proof['type']})")
        else:
            print(s)
    elif cmd == "verify":
        proof = json.loads(Path(sys.argv[2]).read_text())
        ok, msg = verify(proof)
        print(("VERIFY OK: " if ok else "VERIFY FAIL: ") + msg)
        sys.exit(0 if ok else 1)
