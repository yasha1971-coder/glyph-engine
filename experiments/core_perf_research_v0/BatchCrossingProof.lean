import Mathlib

/-!
# GLYPH P02 — mixed-length query carry arithmetic

If a streaming batch retains M-1 bytes for the longest pattern length M,
a shorter pattern m <= M must not be double-counted merely because it is
present inside the longer carry. New matches are filtered by end > b,
where b is the already processed prefix length.

These are arithmetic invariants ONLY. They do not prove SHA-256,
Python implementation, index semantics, external witness, or OS reads.
-/

namespace GLYPH.PerfP02

/-- A match of length m <= M crossing the processed-prefix boundary b
begins in the last M-1 bytes, including the case b < M-1. -/
theorem mixed_length_crossing_covered
    (m M b p : ℕ)
    (hm : 0 < m) (hbound : m ≤ M)
    (hbefore : p < b) (hcross : b < p + m) :
    b - (M - 1) ≤ p := by
  omega

/-- A match already completed by b cannot be considered a new match. -/
theorem completed_match_is_not_new
    (m b p : ℕ)
    (hcomplete : p + m ≤ b) :
    ¬ b < p + m := by
  omega

/-- A match whose endpoint lies after b must not have been completed
in a prior chunk. This applies to shorter patterns found inside M-1 carry. -/
theorem new_match_not_previously_completed
    (m b p : ℕ)
    (hnew : b < p + m) :
    ¬ p + m ≤ b := by
  omega

/-- Reconstructing the absolute coordinate from a carry window start
cannot change the original byte coordinate. -/
theorem carry_absolute_coordinate
    (b kept p : ℕ)
    (hkept : kept ≤ b)
    (hposition : b - kept ≤ p) :
    (b - kept) + (p - (b - kept)) = p := by
  omega

/-- A crossing occurrence, completely arrived in the new chunk,
is inside the retained carry + new block window. -/
theorem mixed_length_new_block_window
    (m M b p newBytes : ℕ)
    (hm : 0 < m) (hbound : m ≤ M)
    (hbefore : p < b) (hcross : b < p + m)
    (harrived : p + m ≤ b + newBytes) :
    b - (M - 1) ≤ p ∧
    (p - (b - (M - 1))) + m ≤
      (b - (b - (M - 1))) + newBytes := by
  constructor
  · omega
  · omega

end GLYPH.PerfP02
