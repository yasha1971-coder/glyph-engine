import Mathlib

/-!
# GLYPH CORE-L6: finite-run LF affine identity

Scope: a mathematical model of symbol-prefix ranks and LF. A BWT run
is a maximal constant-symbol interval. Given a constant symbol on
[a, a+d], the LF mapping is affine on that interval.

This is a known BWT property, NOT a novel theorem. The proof does NOT
certify a C++ parser, SHA256, EF provenance, r-index construction,
runtime error handling or any P/R/O cryptographic claim.

The proof is deliberately separate from source/CI byte oracles.
-/

namespace GLYPH.RunIntervalLF

/-- Prefix count of character c in positions [0, i). -/
def rankPrefix (bwt : ℕ → ℕ) (c : ℕ) : ℕ → ℕ
  | 0 => 0
  | i + 1 => rankPrefix bwt c i + if bwt i = c then 1 else 0

/-- On a constant BWT-symbol run, prefix rank rises exactly by d. -/
theorem rankPrefix_on_constant_run
    (bwt : ℕ → ℕ) (c a : ℕ) :
    ∀ d : ℕ,
      (∀ k : ℕ, k < d → bwt (a + k) = c) →
      rankPrefix bwt c (a + d) = rankPrefix bwt c a + d := by
  intro d
  induction d with
  | zero =>
      intro _
      simp
  | succ d ih =>
      intro hs
      have hsmall : ∀ k : ℕ, k < d → bwt (a + k) = c := by
        intro k hk
        exact hs k (Nat.lt_trans hk (Nat.lt_succ_self d))
      have hd : bwt (a + d) = c := hs d (Nat.lt_succ_self d)
      calc
        rankPrefix bwt c (a + (d + 1))
            = rankPrefix bwt c (a + d) + 1 := by
                rw [Nat.add_assoc]
                simp only [rankPrefix, if_pos hd, add_zero]
        _ = (rankPrefix bwt c a + d) + 1 := by rw [ih hsmall]
        _ = rankPrefix bwt c a + (d + 1) := by omega

/-- Classical last-to-first mapping over prefix rank. -/
def LF (bwt : ℕ → ℕ) (C : ℕ → ℕ) (i : ℕ) : ℕ :=
  C (bwt i) + rankPrefix bwt (bwt i) i

/-- Affine LF inside a BWT run: this is the run-index exactness obligation. -/
theorem lf_affine_within_run
    (bwt : ℕ → ℕ) (C : ℕ → ℕ) (c a d : ℕ)
    (hconstant : ∀ k : ℕ, k ≤ d → bwt (a + k) = c) :
    LF bwt C (a + d) = LF bwt C a + d := by
  have hfirst : bwt a = c := by
    simpa using hconstant 0 (Nat.zero_le d)
  have hlast : bwt (a + d) = c := hconstant d (le_refl d)
  have hprefix := rankPrefix_on_constant_run bwt c a d
    (by
      intro k hk
      exact hconstant k (Nat.le_of_lt hk))
  simp only [LF, hfirst, hlast, hprefix]
  omega

end GLYPH.RunIntervalLF
