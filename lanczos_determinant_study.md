# Determinant feasibility study: stochastic Lanczos/Hutchinson log-determinants for LISA time-domain noise inference

**Status:** design brief — seed document for a standalone repository.
**Timebox:** 2 weeks hard cap. Week 1: E0–E2. Week 2: E3 + write-up. E4 is a stretch goal.
**Owner:** Ollie Burke. Implementation assistance: Claude (this document is the spec).
**Decision gate:** results enter the DHF proposal only if clean, as ONE sentence labelled
"preliminary experiments", with measured numbers only. Otherwise the study is private
Y1 de-risking + interview material. Nothing here couples to the [7] arXiv posting.

---

## 1. Why this study exists

The DHF proposal's WP1 promises Bayesian **noise** inference in the time domain with
flexible (ratio-spline) noise models. Quadratic forms `d^T Σ^{-1} d` are demonstrated
in [7] (CG + FFT matvecs, empirically ~O(N^1.25)). The likelihood's other half,

    log L ⊃ -1/2 · log|Σ(θ_noise)|,

changes with **every noise-parameter update** and is NOT demonstrated in [7] beyond
outline. An expert reviewer's sharpest feasibility attack (confirmed by independent
review, 2026-08): *"whether the log-determinant is tractable for flexible spline noise
models is the sharpest unanswered question."* This study answers it with numbers —
or finds out early that the convolved-diagonal/SMC fallback must carry more weight.

Key insight to exploit: **MCMC needs determinant differences, not absolute values.**
For a proposed noise update θ → θ′ the accept ratio needs
Δ = log|Σ(θ′)| − log|Σ(θ)| = tr(log Σ(θ′) − log Σ(θ)),
and stochastic estimators of the *difference* with **shared probe vectors** enjoy
variance cancellation when Σ(θ′) ≈ Σ(θ) — which is the typical MCMC step.

## 2. Mathematical setup

- Stationary noise, cadence Δt (baseline **15 s**, matching [7]), record length N
  samples, frequencies f ∈ [1/(NΔt), 1/(2Δt)] with the analysis band bottoming at
  ~1e-4 Hz.
- One-sided PSD S(f; θ) → autocovariance (Wiener–Khinchin) → symmetric positive
  definite Toeplitz covariance Σ(θ) ∈ R^{N×N}.
- Exact matvec Σv via **circulant embedding** (size 2N FFT). Watch for negative
  eigenvalues of the embedding at small N / steep PSDs — assert nonnegativity, pad
  if needed.
- Noise model (the WP1 parameterisation): **ratio-spline**
  S(f; c) = S_ref(f) · exp( Σ_k c_k B_k(log f) ),
  with B_k cubic B-splines (K ≈ 10–20 knots) and S_ref a fixed reference PSD.
- Identity used throughout: log|Σ| = tr(log Σ) (Σ is SPD).

### Estimators

1. **Hutchinson trace estimator:** tr(f(Σ)) ≈ (1/M) Σ_{i=1..M} z_i^T f(Σ) z_i with
   Rademacher probes z_i. Optional upgrade: Hutch++ (Meyer et al. 2021) if variance
   is the binding constraint.
2. **Stochastic Lanczos quadrature (SLQ)** (Ubaru–Chen–Saad 2017): each z^T log(Σ) z
   evaluated by k Lanczos steps on Σ (matvecs by FFT), giving tridiagonal T_k;
   z^T log(Σ) z ≈ ||z||² Σ_j (τ_j)² log(λ_j) over eigenpairs (λ_j, τ_j) of T_k
   (τ_j = first component of the j-th eigenvector). Use full reorthogonalisation at
   these k (memory is trivial for k ≤ a few hundred).
3. **Determinant difference, shared probes:**
   Δ̂ = (1/M) Σ_i [ z_i^T log(Σ(θ′)) z_i − z_i^T log(Σ(θ)) z_i ],
   with the SAME z_i in both terms (control-variate cancellation).
4. **Reference preconditioning (the designed WP1 answer, to be TESTED not assumed):**
   let C_ref be the *circulant* operator built from S_ref — its inverse square root
   is FFT-diagonal, hence cheap. Form the whitened operator
   W(θ) = C_ref^{-1/2} Σ(θ) C_ref^{-1/2}.
   For small spline perturbations W ≈ I + (edge terms): spectrum clustered near 1
   → fast Lanczos convergence and low SLQ variance. Note
   log|Σ| = log|C_ref| + log|W| only up to the circulant-vs-Toeplitz edge mismatch —
   the mismatch is itself a leading-edge effect. **Measure it, do not assume it.**
   (Split option: log|Σ| = log|Σ_ref| + log|Σ_ref^{-1}Σ| with the reference term
   precomputed once per run.)

### Known subtlety to address head-on (reviewers will probe it)

A stochastic log-likelihood inside MCMC is not exactly valid (pseudo-marginal theory
requires unbiased *likelihood* estimates; SLQ gives consistent-but-biased *log*
estimates). Three practical stances, in increasing rigour:
(a) drive bias+std below a tolerance (~0.1 in log-likelihood units) and treat the
surrogate as exact; (b) fix the probe set per run (common random numbers →
deterministic surrogate likelihood); (c) debiasing schemes (Russian roulette) if ever
needed. **E1 therefore measures BIAS separately from variance** (vs dense truth).

## 3. Experiments

Two PSD models throughout:
- **P1 (toy, smooth):** power law + white floor. Sanity and scaling.
- **P2 (realistic):** TDI2 A-channel-like PSD **including the transfer-function
  zero(s)** (~30 mHz and harmonics). This is the crux case — the zeros make Σ
  ill-conditioned, which is exactly where Lanczos convergence and SLQ variance get
  hurt, and where reference preconditioning must earn its keep. Source: Ollie
  supplies (gaplike / lisatools / analytic formula) — do not hand-roll a guess.

| ID | Question | Setup | Output |
|---|---|---|---|
| **E0** | Validation: does SLQ reproduce the truth? | N ∈ {2^10..2^14}, dense Cholesky log-det as ground truth (2^14 dense ≈ tens of seconds — cross-check the number quoted in [7] Sec. VII C) | rel. error tables; also validates the dense timing for the proposal |
| **E1** | Accuracy: bias and variance vs budget | M ∈ {4,8,...,128} probes × k ∈ {10,25,50,100,200} Lanczos steps, at N ∈ {2^12, 2^14, 2^16}, P1 and P2 | bias(M,k), std(M,k) heat maps; the P2-vs-P1 gap quantifies the conditioning penalty |
| **E2** | Cost: wall-time scaling at fixed accuracy | pick (M,k) meeting the tolerance from E1; N ∈ {2^12..2^17} (2^18 if cheap) | wall time vs N (log–log, fitted slope); compare to quadratic-form cost at same N |
| **E3** | **The money experiment:** determinant differences under spline perturbations | draw c′ = c + δ, δ_k ~ N(0, σ²), σ ∈ {0.01, 0.1, 0.5}; estimate Δ via (i) two absolute SLQ estimates, independent probes; (ii) shared probes; (iii) shared probes + reference preconditioning (W-operator form). Truth from dense at N = 2^14 | variance of Δ̂ vs method vs σ; the (iii)/(i) variance ratio is the headline number; report cost per accepted-quality Δ̂ in units of quadratic-form evaluations |
| **E4** (stretch) | Does it survive the observed-sample restriction? | leading-edge truncation: Σ_obs = R Σ R^T (delete trailing rows/cols; matvec = mask ∘ circulant ∘ mask), rerun E3(iii) on the restricted operator | same metrics; flags whether gap-edge deflation is needed (expected from the WP1 plan) |

**Success criterion (pre-registered):** |Δ̂ − Δ| ≤ 0.1 (bias + 1σ) at N = 2^16–2^17
for σ ≤ 0.1 perturbations, at a cost ≤ ~10 quadratic-form equivalents per difference.
**Stopping rule:** if E1/E3 cannot reach tolerance with M·k ≤ 10^4 matvecs on P2,
stop, write up the negative result honestly, and strengthen the fallback framing
instead. That outcome is also a success of the study.

## 4. Implementation notes

- Python; numpy/scipy only (`scipy.sparse.linalg.LinearOperator` + FFT matvec;
  `scipy.linalg.cholesky` for truth; own 20-line Lanczos with reorthogonalisation —
  clearer than adapting a library and we need the T_k internals).
- Deterministic: fixed seeds everywhere; all tunables in one CONFIG block (house
  style of the proposal figure scripts). No GPU — batched-FFT GPU throughput stays a
  *projection* in the proposal; this study is single-core honest numbers.
- Outputs: `figures/*.pdf` (grayscale, B&W-safe, same conventions as the proposal
  figures) + `metrics.json` (every number that could enter prose) + `SUMMARY.md`
  (one paragraph; the sentence-ready result with its caveats).
- Repo hygiene: MIT, showyourwork-compatible layout if convenient (matches the DMP's
  stated practice), `uv` environment, tests for the Lanczos routine vs
  `scipy.linalg.eigh_tridiagonal` on small cases.

## 5. Risk register

| Risk | Mitigation |
|---|---|
| TDI zeros → ill-conditioning → slow Lanczos / high SLQ variance | reference preconditioning (E3-iii); near-null-mode deflation as identified route; report the P2 penalty honestly |
| Circulant embedding indefinite at small N / steep PSD | assert min eigenvalue ≥ 0; enlarge embedding; document |
| Finite-k bias masquerading as convergence | E1 measures bias vs dense truth explicitly, per k |
| Scope creep (the 2-week wall) | E0–E3 only; E4 stretch; stopping rule above |
| Overclaiming in the proposal | decision gate: one sentence, measured numbers, "preliminary"; else nothing |

## 6. What Ollie supplies

1. The P2 PSD function (or pointer to the gaplike/lisatools call) with the TDI2 zero.
2. The reference PSD S_ref and spline-knot layout consistent with the WP1 plan.
3. The [7] Sec. VII C dense-solve timing at N = 2^14 (validation cross-check — also
   settles the proposal's 20-min-vs-26-s discrepancy).
4. Sign-off on the success criterion before any code runs (pre-registration keeps
   the result honest either way).
