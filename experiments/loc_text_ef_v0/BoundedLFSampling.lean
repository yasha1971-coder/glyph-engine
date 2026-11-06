import Mathlib

/-!
# LOC-TEXT-EF: arithmetic obligation, NOT a full FM-index proof

For a positive sample step s, starting at a suffix-array text position x,
decrementing it by x % s reaches a sampled text position divisible by s.
The number of decrements is strictly less than s.

This proof does not establish that a particular RLB2/RLR2 implementation
correctly computes LF. Independent byte oracles and actual runtime parity
provide separate empirical evidence for the implementation.

Classical text-order suffix-array sampling and modular arithmetic are
known results, not a claim of mathematical novelty.
-/

namespace GLYPH.LocTextEF

theorem text_sample_hit_and_bound (x s : ℕ) (hs : 0 < s) :
    (x - (x % s)) % s = 0 ∧ x % s < s := by
  have hmod : x % s < s := Nat.mod_lt x hs
  have hdiv : x % s + s * (x / s) = x := Nat.mod_add_div x s
  have hsub : x - x % s = s * (x / s) := by omega
  constructor
  · rw [hsub]
    simp
  · exact hmod

theorem text_sample_max_lf_steps (x s : ℕ) (hs : 0 < s) :
    x % s ≤ s - 1 := by
  have h := (text_sample_hit_and_bound x s hs).2
  omega

end GLYPH.LocTextEF
