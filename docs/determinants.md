# Determinants

The exact time-domain likelihood has two halves. The quadratic form
$r^{T}\Sigma_{OO}^{-1}r$ is what {class}`~gaplike.RestrictedCG` evaluates
matrix-free at any scale. The other half,

$$\log L \supset -\tfrac{1}{2}\,\log\lvert\Sigma_{OO}(\lambda)\rvert,$$

changes with **every noise-parameter update**, and once the noise model is
richer than the two-component pencil there is no closed form. The dense route
— build $\Sigma_{OO}$, factorize, read the determinant off the factor — costs
O(m³) and stops existing at the same record lengths that motivated the CG
solver in the first place.

{mod}`gaplike.slq` provides two matrix-free answers, which tier by record
length.

## The complement identity: exact

For an SPD covariance partitioned over observed (O) and gap (G) samples,
Schur-complement algebra gives the exact identities

$$\log\lvert\Sigma_{OO}\rvert = \log\lvert\Sigma\rvert
    + \log\lvert(\Sigma^{-1})_{GG}\rvert,
\qquad
\Sigma_{OO}^{-1} = A_{OO} - A_{OG}A_{GG}^{-1}A_{GO},
\quad A = \Sigma^{-1}.$$

In the circulant model both ingredients are cheap: $\log\lvert\Sigma\rvert$
is the closed-form Whittle sum, and $\Sigma^{-1}$ is the circulant of the
*inverse* spectrum, so $A_{GG}$ is a dense restriction to the **gap** samples
— a g×g problem where g is the number of *missing* samples. At an 80% duty
cycle that is $(m/g)^3 = 64\times$ fewer flops than dense on the observed
side, with no probes and no iteration. The same cached factor turns every
quadratic form into two FFTs and one triangular solve — no conjugate
gradients either.

```python
import numpy as np
import gaplike as gl

n, dt = 2**14, 15.0
mask = gl.gaps.periodic_mask(n, 10, 50)
comps = list(gl.psd.lisa_tdi2_ae().values())

eig = gl.cg.circulant_eigenvalues(comps, n, dt)
fac = gl.ComplementFactor(eig, mask)      # one g x g Cholesky
fac.logdet                                # exact log|Sigma_OO|
fac.quad_form(resid)                      # exact quadratic forms, same factor
```

One factorization per noise state buys the exact determinant *and* all
quadratic forms. Measured on the paper's scenario-C comb at N = 2¹⁴ the
factorization takes ~0.1 s against ~2.3 s for dense build + Cholesky, and the
result agrees with dense truth to ~10⁻⁹; the identity stays affordable to
N ≈ 2¹⁵–2¹⁶, where its own g³ wall arrives — exactly as the $(m/g)^3$
arithmetic predicts.

One convention note: the library zeroes the DC bin (mean removal), which
makes the full circulant singular. {class}`~gaplike.ComplementFactor`
therefore evaluates the identity on a DC-floored model and removes the floor
**exactly** by a rank-one downdate (matrix determinant lemma along the
constant mode). The floor is a conditioning knob, not an approximation.

## Stochastic Lanczos quadrature: beyond the wall

Past the g³ wall, {func}`~gaplike.slq.logdet_slq` estimates
$\log\lvert\Sigma_{OO}\rvert = \operatorname{tr}\log\Sigma_{OO}$ by
Hutchinson probing with Lanczos quadrature (Ubaru, Chen & Saad 2017), built
on the same FFT matvec the CG solver uses — its cost scales like the
quadratic form's, measured ~N⁰·⁹ per estimate, with a batched path that
executes all probes through one multithreaded real-FFT pair per Lanczos step.

The estimator converges to ~10⁻⁴ *relative* error in under a second — but
the determinants are ~10⁵–10⁶ in magnitude, so an *absolute* stochastic
estimate is useless inside an accept ratio. What an MCMC actually needs is
the **difference**

$$\Delta = \log\lvert\Sigma_{OO}(\lambda')\rvert
         - \log\lvert\Sigma_{OO}(\lambda)\rvert,$$

and {func}`~gaplike.slq.logdet_diff_slq` estimates it with the **same
Rademacher probes on both operators**, so the estimator sees only the small
difference operator, never the two large absolute determinants. For
MCMC-sized noise steps the shared probes collapse the variance by orders of
magnitude relative to differencing two independent estimates — fractions of
a log-likelihood unit from a handful of probes, roughly flat in N.

```python
rng = np.random.default_rng(0)
out = gl.slq.logdet_slq(eig, mask, n_probes=32, k=100, rng=rng)
out["est"], out["per_probe"].std()        # estimate + empirical scatter

diff = gl.slq.logdet_diff_slq(eig1, eig2, mask, n_probes=16, k=100,
                              rng=rng, shared=True)
```

## Flexible noise models

The component that makes all this necessary:
{func}`~gaplike.slq.ratio_spline` builds a PSD
$S_{\mathrm{ref}}(f)\,\exp\bigl(\sum_k c_k B_k(\log_{10} f)\bigr)$ — a fixed
reference spectrum times a free log-ratio cubic B-spline. It is an ordinary
component, accepted anywhere `gaplike` takes one (`RestrictedCG`,
`circulant_eigenvalues`, …); zero coefficients reproduce the reference
exactly, and outside the knot span the ratio extrapolates flat.

```python
S_ref = lambda f: sum(c(f) for c in comps)
knots = gl.slq.spline_knots(1e-4, 1.0 / (2 * dt), n_coeff=12)
flex = gl.slq.ratio_spline(S_ref, knots, coeffs)     # a PSD component
eig = gl.cg.circulant_eigenvalues([flex], n, dt)
```

## Choosing a route

**Two components entering linearly, m up to ~10⁴?** The closed form of
{class}`~gaplike.TimeDomainExact` already includes the determinant; nothing
here is needed.

**Anything richer, N up to ~2¹⁵–2¹⁶?** The complement identity: exact
determinant and exact quadratic forms from one g×g Cholesky per noise state.

**Beyond that?** Shared-probe SLQ differences for the accept ratio, with
{func}`~gaplike.slq.logdet_dense` as ground truth wherever dense is still
affordable (validation, small-N cross-checks).

The measured cost and accuracy of all three routes on the paper's comb —
dense, complement, SLQ — come from `paper/fig_det_scaling.py`
(`paper/mkfig_det.py` draws the figure), and the shared-probe difference
experiments from `paper/fig_det_flexible.py`.
