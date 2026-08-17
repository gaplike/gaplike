# Determinant feasibility study — summary of findings

**Branch:** `flexible_spline_TD` · **Dates:** 2026-08-14/15 · **Spec:** `lanczos_determinant_study.md`
**Status:** implemented and validated by Claude under Ollie's direction; **all numbers
below are quick-pass, single-machine measurements awaiting Ollie's independent re-run
before anything is quoted in the DHF proposal** (pre-registration rule).

Setup throughout: scenario-C comb gaps (80% duty — LISA-like), TDI2 A-channel
two-component reference spectrum, Δt = 15 s, flexible noise = 12-coefficient
ratio-spline `S_ref(f)·exp(Σ c_k B_k(log10 f))`. Ground truth: dense Cholesky of the
restricted covariance (`logdet_dense`), affordable to N = 2¹⁴.

## What was built

- `src/gaplike/slq.py` — stochastic Lanczos quadrature (SLQ) log-determinants;
  **shared-probe determinant differences** (the MCMC-relevant primitive);
  batched, no-reorthogonalisation fast path; `ratio_spline` flexible PSD
  component; **`ComplementFactor`** — exact log-determinant *and* exact quadratic
  forms via the Schur-complement identity with an exact DC downdate.
- `tests/test_slq.py` — 11 new tests (51 total pass): quadrature exactness,
  dense-truth agreement, shared-vs-independent variance, batched≡serial
  equivalence, no-reorth accuracy, complement identity to machine precision,
  DC-floor independence, quad-form vs CG.
- `paper/fig_det_scaling.py`, `paper/fig_det_flexible.py`, `paper/mkfig_det.py`
  — study scripts (JSON out) + figure.

## Findings

### F1 — Absolute stochastic log-dets are useless for accept ratios
SLQ converges to ~10⁻⁴ **relative** error in under a second — but the determinants
are ~10⁵–10⁶ in magnitude, so **absolute** errors sit at 25–200 log-likelihood (LL)
units at any tested budget. Data: `paper/results/det_scaling.json`; right panel of
`paper/figures/det_scaling.{pdf,png}` (grey triangles).

### F2 — Shared-probe *differences* are the right stochastic primitive
For MCMC-sized spline steps (σ = 0.01 per coefficient), sharing the Rademacher
probes across the two operators collapses the variance by **~10⁵–10⁶×** versus
independent probes: median errors **0.21 / 0.35 / 0.37 LL** at N = 2¹²/2¹³/2¹⁴
with only 16 probes (≈5 quadratic-form equivalents per difference), flat in N.
Larger steps degrade linearly (σ = 0.1 → ~1.6–4.4 LL). Data:
`paper/results/det_flexible_k{12,13,14}.json`; green stars / olive crosses in the
figure. The pre-registered 0.1-LL criterion is narrowly missed at M = 16 and would
be met by a probe-count increase (M ≈ 64–128, ~1–2 s) — rendered moot by F4 within
its range.

### F3 — CPU acceleration: measured, including one instructive dead end
- rfft matvec (real half-spectrum): ~2×, benefits the whole library incl. CG.
- **Naive batched Lanczos with full reorthogonalisation is 3–4× SLOWER**: the
  stored-basis re-reads become DRAM-bound once the batch exceeds cache
  (~335 MB/step at 32 probes, 2¹⁴). Dead end worth remembering.
- Standard no-reorthogonalisation SLQ, batched: **validated** (estimates within
  <0.001 LL of full-reorth and within scatter of dense truth; permanent test) —
  net **~2× vs serial, ~3× vs the original baseline**; measured scaling N^0.87;
  crossover vs dense moved to ~2¹³. Left panel of the figure.
- GPU remains an identified route (the batched structure is the on-ramp);
  projections stay out of print per house rule.

### F4 — The complement identity: exact, and faster than everything in range
For SPD Σ split over observed (O) and gap (G) samples:

    log|Σ_OO| = log|Σ| + log|(Σ⁻¹)_GG|        (exact; Schur complement algebra)
    Σ_OO⁻¹    = A_OO − A_OG A_GG⁻¹ A_GO,  A = Σ⁻¹   (same g×g factor)

In the circulant model log|Σ| is the closed-form Whittle sum and Σ⁻¹ is the
inverse-spectrum circulant, so the whole cost is **one g×g Cholesky on the GAP
side** — at 80% duty, 64× fewer flops than dense on the observed side. Measured
(min over repeats; multithreaded LAPACK):

| N | data span | dense-O | SLQ (estimate) | complement (**exact**) | |err| vs truth |
|---|---|---|---|---|---|
| 2¹² | 17 h | 0.06 s | 0.18 s | **0.004 s** | 7×10⁻¹⁰ |
| 2¹⁴ | 2.8 d | 2.57 s | 0.65 s | **0.113 s** | 1.4×10⁻⁹ |
| 2¹⁵ | 5.7 d | — | 1.33 s | **0.56 s** | (exact) |
| 2¹⁶ | 11 d | — | 2.87 s | 3.57 s | (exact) |

**Quadratic forms from the same cached factor: 46 ms vs 458 ms of CG (988
iterations) at 2¹⁴, agreeing to 3×10⁻¹⁵ relative.** One factorisation per noise
state buys the exact determinant *and* all quadratic forms: full exact likelihood
≈ 0.16 s/state at 2¹⁴ → ~2×10⁴ exact evaluations per hour on one machine.

**Two-tier picture:** exact complement route to ~2¹⁵–2¹⁶ (its g³ wall, exactly as
the (m/g)³ arithmetic predicts); validated shared-probe SLQ beyond. Lineage note:
this is classical Gaussian missing-data algebra (the marginalised cousin of gap
data augmentation) — the contribution is its deployment inside the
restricted-circulant likelihood, not the identity.

### F5 — Numerical honesty notes
- The library zeroes the DC bin (mean removal), so the identity runs on a
  DC-floored model and the floor is removed **exactly** by a rank-one downdate
  (matrix determinant lemma / Sherman–Morrison along the constant mode). The
  floor is therefore a *conditioning* knob: small floors (≲10⁻² of max) are
  machine-tight; large floors cancel catastrophically in the downdate log
  (sweep-tested; default `dc_scale = 1e-3`).
- Everything is deterministic given seeds; batched ≡ serial pinned by test;
  timings are multithreaded (relevant when comparing to single-threaded numbers
  elsewhere, e.g. the manuscript's dense timings).

## Plots and data

- `paper/figures/det_scaling.pdf` / `.png` — cost (left) and accuracy-in-LL-units
  (right); **does not yet include the complement (exact) curve** — next step.
- `paper/results/det_scaling.json` — dense/SLQ timings + errors, N = 2⁹–2¹⁴.
- `paper/results/det_flexible_k{12,13,14}.json` — difference experiments.
- Complement benchmark numbers: table above (inline run; script-ify on request).

## Open items

1. Add the complement curve to the figure (rewrites its story: an exact-and-fast
   tier to 2¹⁵⁺, stochastic tier beyond).
2. Ollie: independent re-run + code review + commit; then decide what (if
   anything) enters the DHF proposal, as one "preliminary experiments" sentence.
3. Unimplemented accelerants, in rough value order: preconditioned/whitened
   difference operator (spec E3-iii); leading-edge truncation study (spec E4 —
   the premerger case; note Toeplitz structure also unlocks Levinson streaming
   and Szegő asymptotics, untried); exact low-moment control variates;
   deflation; Hutch++/XTrace; mixed precision; pyfftw; GPU backend.
4. Error bars on the F2 medians (more proposal pairs, second chain state).

## Provenance

Code and experiments drafted by Claude (Anthropic) under Ollie Burke's direction,
2026-08-14/15, with the test suite as the arbiter throughout; all design
decisions, the underlying formalism, and final acceptance are Ollie's. This file
is part of the repo's research record; the DHF application's separate AI-use log
covers anything that crosses into the proposal.
