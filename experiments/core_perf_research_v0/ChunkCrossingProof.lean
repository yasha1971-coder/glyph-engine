import Mathlib

/-!
# GLYPH CORE-PERF-P01: chunk crossing arithmetic obligations

Known finite-string scanning math, not a new theorem.
The verifier's one-pass optimization retains the last m-1 bytes of the
preceding block when m>0, and searches this carry + next block.

This Lean proof checks the arithmetic coverage of a match crossing a
block boundary. It does NOT prove Python code, SHA collision resistance,
file I/O, a signed root, concurrency, or semantic index integrity.
-/

namespace GLYPH.PerfP01

/-- Any match beginning before b and ending after b starts in the final
    m-1 bytes of the preceding region, unless that region is shorter. -/
theorem crossing_starts_in_carry
    (m b p : ℕ)
    (hm : 0 < m)
    (starts_before : p < b)
    (ends_after : b < p + m) :
    b - (m - 1) ≤ p := by
  omega

/-- Conversely, a match beginning within a carry of m-1 bytes cannot
    have been fully completed before the previous block boundary b. -/
theorem carry_cannot_repeat_completed_match
    (m b p : ℕ)
    (hm : 0 < m)
    (carry_start : b - (m - 1) ≤ p)
    (starts_before : p < b) :
    b < p + m := by
  omega

/-- If the full match has arrived by the end of the new block, it fits
    entirely into the carry+new-block window. -/
theorem crossing_match_fits_window
    (m b p newBytes : ℕ)
    (hm : 0 < m)
    (starts_before : p < b)
    (ends_after : b < p + m)
    (fully_arrived : p + m ≤ b + newBytes) :
    b - (m - 1) ≤ p ∧
    (p - (b - (m - 1))) + m ≤ (b - (b - (m - 1))) + newBytes := by
  constructor
  · omega
  · omega

end GLYPH.PerfP01
