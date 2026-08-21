# Quickstart

The shape of every analysis is the same: build a gap pattern, choose a noise
model, pick a likelihood tier, hand it residuals.

## Gaps

Every builder in {mod}`gaplike.gaps` returns a boolean **mask**, `True` where
a sample survives. A mask is just an array, so patterns compose: `mask_a &
mask_b` is a valid pattern.

```python
import numpy as np
import gaplike as gl

n, dt = 2880, 15.0                                   # 12 h at 15 s cadence

mask = gl.gaps.periodic_mask(n, 10, 50)              # comb: 150 s out of 750 s
# mask = gl.gaps.random_mask(n, dt, rate_per_day=40, duration_s=600, rng=1)
# mask = gl.gaps.mask_from_intervals(n, dt, [(21000, 25000)])
```

A **gate** is the same thing as a float array in `[0, 1]`, optionally with
Planck-tapered edges; the **effective window** multiplies it by whatever taper
you apply to the record itself.

```python
gate = gl.gaps.gate_from_mask(mask, dt, taper_s=0.0)     # 0 = rectangular
w = gl.gaps.effective_window(gate, gl.gaps.segment_window(n, 0.05))
```

Tapering the gap edges is not cosmetic. A sharp edge is a discontinuity, and
a record with six sharp gap edges leaks *more* far-field power than one with
no window at all.

## Noise

A noise model is a list of **components**, callables `S_k(f)` giving the
one-sided PSD of component `k` at its reference amplitude. The modelled
spectrum is

$$\Sigma(\lambda) = \sum_k 10^{\lambda_k} C_k,$$

so the noise parameters $\lambda_k$ are log10 deviations of the component
**powers**, with the truth at $\lambda = 0$. The two-component LISA TDI-2 A/E
model is built in.

```python
comps = list(gl.psd.lisa_tdi2_ae().values())         # [S_tm(f), S_oms(f)]
S_tot = gl.psd.one_sided_grid(lambda f: sum(c(f) for c in comps), n, dt)
x = gl.simulate.noise_td(S_tot, dt, np.random.default_rng(0), nch=2)
```

## Likelihoods

Four tiers, all with the same interface: `transform` maps a full-grid time
series to that tier's data vector, `loglike(resid, lam)` evaluates.

```python
L_td   = gl.TimeDomainExact(mask, comps, dt)                       # exact, no window
L_full = gl.FullCovariance.from_window(w, comps, dt, 1e-4, 3.1e-2) # dense windowed cov
L_conv = gl.DiagonalLikelihood.convolved(w, comps, dt, 1e-4, 3.1e-2)
L_whit = gl.DiagonalLikelihood.whittle(w, comps, dt, 1e-4, 3.1e-2,
                                       scale_by_window_power=True)

for L in (L_td, L_full, L_conv, L_whit):
    r = L.transform(x)                               # residual = data - template
    print(type(L).__name__,
          L.loglike(r, (0.0, 0.0)) - L.loglike(r, (0.05, 0.0)))
```

:::{warning}
Each tier carries its own constant offset from its internal normalization, so
log-likelihoods are **not comparable across tiers**. Only *differences* —
between parameter points, or between templates — are meaningful.
:::

## Templates

Wrap any frequency-domain callable in {class}`gaplike.Waveform` and use
`.td(theta)`:

```python
wf = gl.Waveform(fd_func, n, dt)
resid = L_td.transform(data_td - wf.td(theta))
```

For LISA massive black-hole binaries,
{func}`gaplike.waveform.lisabeta_mbhb_ae` gives the (A, E) TDI-2 waveform used
in the paper — with any source parameters, any parametrization (`to_params`)
and any approximant (`wf_kw`).

## At scale

`TimeDomainExact` and `FullCovariance` each pay a one-off dense factorization:
fine at m ≈ 2300, impossible at m ≈ 10⁵, where the covariance alone would be
hundreds of gigabytes. {class}`gaplike.RestrictedCG` evaluates the identical
time-domain quadratic form without ever forming the matrix.

```python
rcg = gl.RestrictedCG(mask, comps, dt, rtol=1e-8)
quad, iters = rcg.quad_form((0.0, 0.0), L_td.transform(x))
```

Pair it with a determinant: closed-form from `TimeDomainExact` when there are
exactly two components; otherwise {mod}`gaplike.slq` supplies the exact
complement identity ({class}`~gaplike.ComplementFactor`) and stochastic
Lanczos quadrature at any scale — see [Determinants](determinants.md). See
[the likelihood hierarchy](likelihoods.md) for when this is the right choice.
