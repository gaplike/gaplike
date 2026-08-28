# The likelihood hierarchy

| class | covariance model | cost per evaluation | guarantees |
|---|---|---|---|
| {class}`~gaplike.TimeDomainExact` | stationary covariance restricted to the observed samples | O(m) after one O(m³) setup | **exact**: scatter/width and width/true-width are 1 by construction; no window anywhere |
| {class}`~gaplike.FullCovariance` | dense windowed frequency-domain covariance, band-restricted | O(r) after one O(N_b³) setup | exact within the band-restricted, circularly-symmetric reduction |
| {meth}`~gaplike.DiagonalLikelihood.convolved` | exact diagonal of the windowed covariance | O(N_b) | unbiased widths *per bin*; ignores bin-to-bin correlations |
| {meth}`~gaplike.DiagonalLikelihood.whittle` | raw PSD, optionally scaled by the window power | O(N_b) | the Whittle / "normalizing constant" approximation |
| {class}`~gaplike.RestrictedCG` | same model as `TimeDomainExact`, matrix-free | 4 FFTs × a few hundred iterations, **no setup, no storage** | exact quadratic forms to a chosen tolerance; any number of components; determinant from {mod}`gaplike.slq` |

## Conventions

Noise parameters are the same throughout: $\lambda_k$ are log10 deviations of
the component **powers** from their reference values,

$$\Sigma(\lambda) = \sum_k 10^{\lambda_k} C_k ,$$

with the truth at $\lambda = 0$. Residuals are multichannel, shape
`(nch, ·)` — channels independent with identical covariances, as LISA's A and
E are.

Both *pencil* classes, `TimeDomainExact` and `FullCovariance`, additionally
assume the covariance is linear in **exactly two** components. One
simultaneous diagonalization then makes every likelihood evaluation, Fisher
matrix and determinant closed-form. The diagonal tiers and the
conjugate-gradient route accept any number of components.

## How to choose

**Is the noise model two components entering linearly?** If yes, and the
number of observed samples is up to ~10⁴, `TimeDomainExact` is both exact and
the cheapest thing you can run: one O(m³) factorization at setup, then O(m)
per evaluation with a closed-form determinant. There is no reason to use
anything else.

**Is it something less accommodating** — free spline knots, a third component,
a nonlinearly parametrized shape? The pencil dies, because it would need a
fresh factorization at every evaluation. That is what `gaplike.cg` is for.

**Is the record long?** The dense covariance is what stops you first. At an
80% duty cycle it reaches 5.5 GB at N = 2¹⁵ and about 350 GB at N = 2¹⁸: not
slow, simply unavailable. The conjugate-gradient route never forms it.

**Do you only need the widths to be honest, not optimal?** The convolved
diagonal is accurate — its point estimates are unbiased — and costs O(N_b).
Its failure mode is inflation, not bias: it quotes intervals wider than the
data allow, badly so under a severe gap pattern.

## Why the diagonal fails, and how

Multiplication in time is convolution in frequency, so a window couples
frequency bins. The convolved diagonal corrects the modelled PSD for that
smearing but keeps the covariance diagonal, which means it must treat the
aliased variance in every bin as independent noise. The exact treatments
instead exploit the strong bin-to-bin correlations of the gap pattern to unmix
it.

The consequence is a loss of *precision*, not of accuracy. In the paper's
drastic-comb scenario the convolved diagonal remains calibrated — its
scatter-to-width ratios sit at 0.85–0.95 in the signal sector — while its
quoted widths are 8.6 to 10.2 times wider than the data allow. It does not
give you a confidently wrong answer; it gives you a correct answer with error
bars an order of magnitude too loose. (Those two numbers are the $\Upsilon$
and $\Xi$ diagnostics; [their own page](upsilon_xi.md) defines them and shows
why a PP plot cannot separate them.)

Because both the curvature and the score variance are quadratic in
$\partial h / \partial\theta$, that ratio is independent of SNR at leading
order: the cost of the approximation neither grows nor shrinks with how loud
the source is. What it does mean is that the analysis behaves as though the
SNR were an order of magnitude smaller — so the approximation stops being
usable when the true SNR is not comfortably above that factor times threshold.

## The matrix-free route

The exact time-domain likelihood needs quadratic forms in the inverse of

$$\Sigma_{OO} = R\,\Sigma\,R^{T},$$

the stationary (circulant) covariance with the gap rows and columns deleted.
$\Sigma_{OO}$ is neither circulant nor Toeplitz, so the FFT does not
diagonalize it and the classical route is to build and factorize it.

Conjugate gradients never needs the matrix — only its *action* on a vector,
which costs three steps, two of them FFTs: embed the short vector into the
full grid with zeros, multiply by the circulant eigenvalues in Fourier space,
restrict back to the observed samples. The identity $R\Sigma R^{T} =
\Sigma_{OO}$ makes this exact rather than approximate: the zero-fill is the
embedding $R^{T}$, never a statement about the data.

What remains is conditioning. CG converges in roughly $\sqrt{\kappa}$
iterations and $\kappa$ is enormous, because the LISA spectrum spans many
orders of magnitude — but that dynamic range is present without gaps too, and
without gaps it inverts exactly. Preconditioning with the gap-free covariance,
applied by the same FFT pair dividing instead of multiplying, removes it and
leaves the iteration to deal only with the geometry of the gaps. The count
falls from thousands to a few hundred.

Measured on the paper's comb, the cost of one quadratic form scales as
$N^{1.25}$ against $N^{2.8}$ for the dense route: at N = 2¹⁸, over 2×10⁵
observed samples, a minute against an extrapolated fifteen hours. The gap
between $N^{1.25}$ and the $N\log N$ of a single iteration is the growth of
the iteration count, and that growth decelerates — its local slope falls from
$N^{0.96}$ at the smallest sizes to $N^{0.11}$ over the last octave, as one
would expect if the conditioning is set by the geometry of the gaps rather
than by the length of the record.

### Which preconditioner

The circulant (Whittle) preconditioner above is one of two proposed in
[Baghi et al. (2016)](https://arxiv.org/abs/1608.08530). The other builds a
*sparse* approximation of $\Sigma_{OO}$ itself: taper the time-domain
autocovariance to zero beyond a lag $L$, evaluate it at the true lag
$t_i - t_j$ between observed samples, and factorize the resulting banded
matrix with `scipy.linalg.cholesky_banded`.

The two fail in opposite directions, and the reason is worth stating plainly:

* The **circulant** preconditioner is the exact inverse of the *gap-free*
  covariance. It is perfect where the record is uninterrupted, so what it
  leaves behind is entirely the gaps — and its iteration count grows with
  their **number**, measured as $n_{\rm g}^{0.64}$ at fixed missing time.
* The **sparse** preconditioner is built from the *true restricted*
  covariance. It knows where the holes are, but throws away every correlation
  beyond lag $L$. That cut only costs it when a surviving segment is longer
  than $L$ — and when the gaps are closely spaced, none of them are. Its
  iteration count is a function of $\ell/L$ alone, with $\ell$ the mean
  surviving-segment length: curves for different $L$ collapse to about 20% on
  that variable, rising roughly as $(\ell/L)^{1/2}$.

At $N = 8192$ with 15% of the record missing, chopped into $n_{\rm g}$ equal
gaps, the circulant route runs from 20 iterations at $n_{\rm g} = 1$ to 633 at
$n_{\rm g} = 256$, while the sparse route at $L = 32$ runs from 635 down to 34.
Pricing one likelihood call — $\mathcal{O}(N)$ to rebuild the circulant
eigenvalues, against $\mathcal{O}(mL^2)$ for the banded Cholesky, since
$\Sigma_{OO}$ moves with the noise parameters at every proposal — puts the
crossover near $n_{\rm g} \simeq 20$. Both return the same quadratic form to
$5\times10^{-15}$. The rule of thumb:

> **A few long gaps → circulant. Many short gaps → sparse.**

Even spacing is the best case for both. At $n_{\rm g} = 16$, scattering the
same gaps at random takes the circulant count from 145 to 187 and the sparse
one (at $L=128$) from 40 to 79, without changing which is cheaper.

Positive-definiteness of the tapered autocovariance is not automatic, and the
banded Cholesky is what checks it. By the Schur product theorem it is
guaranteed when the taper is itself a positive-definite function of the lag:
the Bartlett taper (the Fejér kernel) qualifies and never failed, but costs
4–10× more iterations. The Hann taper is *not* PD as a function — the minimum
of its lag-spectrum is $-3.4$ — yet succeeded for every pattern and every $L$
tried, and is the better default. Hard truncation (no taper, spectral minimum
$-55.8$) genuinely fails: it was not positive definite for three of four gap
patterns tested. Catch the `LinAlgError` and fall back to Bartlett.

`gaplike.cg` ships the circulant preconditioner as
{func}`~gaplike.cg.whittle_preconditioner` because it carries no tuning
parameter; the sparse one is a drop-in `LinearOperator` on the `M=` argument
of {func}`~gaplike.cg.solve`, and `paper/fig_precond_compare.py` builds it in
about thirty lines. Note that its per-component band arrays do not depend on
the noise parameters — $\gamma$ is linear in $10^{\lambda}$ — so they are
built once and recombined at each proposal; only the Cholesky is per-call.

The determinant is deliberately not provided by `gaplike.cg` itself. With
exactly two components the closed form of `TimeDomainExact` already supplies
it; in general {mod}`gaplike.slq` provides the two matrix-free companions —
the exact Schur-complement identity (one Cholesky on the *gap* samples per
noise state, which also yields exact quadratic forms), and stochastic Lanczos
quadrature with shared-probe differences for the accept ratios an MCMC
actually needs. [Determinants](determinants.md) lays out when each one is the
right choice.
