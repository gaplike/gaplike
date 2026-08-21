# gaplike

**Inference on gapped and windowed stationary Gaussian data.**

`gaplike` does joint signal + noise-parameter estimation when a stationary
time series is interrupted by data gaps, or multiplied by any window. It
provides gap-pattern generation for arbitrary patterns, windowed
frequency-domain covariances, and a hierarchy of likelihoods running from the
cheap Whittle approximation up to the **exact time-domain likelihood** —
evaluated either in closed form, when the noise is linear in two component
powers, or **matrix-free at any scale** by preconditioned conjugate gradients.

Only `numpy` and `scipy` are required. The two-component LISA TDI-2 A/E noise
model is built in; user-defined PSD components and any waveform work the same
way.

Companion package to *"Zurückbleiben bitte: the impact of window functions on
noise and signal parameter inference"* (O. Burke & F. Pozzoli). The
[`paper/`](https://github.com/FedericoPozzoli/gaplike/tree/main/paper)
directory reproduces every figure of that work.

```bash
git clone https://github.com/FedericoPozzoli/gaplike
cd gaplike && uv pip install -e ".[pe]"
```

## Why gaps are not a detail

Multiplying a time series by a gate is a multiplication in time, so it is a
*convolution* in frequency. Every gap edge is a discontinuity, and each one
throws power tens of bins away from where it belongs. On a spectrum as steep
as LISA's, that power lands where the true spectrum is orders of magnitude
smaller — so a diagonal noise model, the Whittle likelihood, is no longer
describing the data.

Two things follow, and `gaplike` is built to measure both.

**Accuracy.** The convolved diagonal, which corrects the modelled PSD for
leakage, stays unbiased under quite severe gap patterns. Its point estimates
are fine. It is the widths that go wrong.

**Precision.** A diagonal model must treat the aliased variance in every bin
as independent noise, while the exact treatments exploit the strong bin-to-bin
correlations of the gap pattern to unmix it. So the intervals a diagonal model
quotes can be far wider than the data allow — by an order of magnitude under
the drastic comb studied in the paper. The failure mode is inflation, not
bias: not a confidently wrong answer, but a correct answer with error bars an
order of magnitude too loose.

<img class="gl-figure" src="_static/gap_leakage_web.gif" alt="gaps, tapers, leakage and correlation"/>

*Four acts on the same record: a raw cut, then a taper on the record edges,
then sharp-edged gaps, then tapered gap edges. Note that sharp gaps leak
several times worse than the raw cut — six new discontinuities instead of
two — and that tapering does not remove the correlations, it makes them
local.*

The remedy is one of the two exact treatments: the full windowed covariance in
the frequency domain, or the time-domain likelihood of the observed samples,
in which the missing samples are simply marginalized and no window appears
anywhere. [The likelihood hierarchy](likelihoods.md) lays out when each one is
the right choice.

```{toctree}
:maxdepth: 1
:caption: Guide

installation
quickstart
likelihoods
determinants
reproducing
```

```{toctree}
:maxdepth: 1
:caption: Reference

api
```

## Citing

Please cite the software (see `CITATION.cff`) together with Burke & Pozzoli
(2026), *Zurückbleiben bitte: the impact of window functions on noise and
signal parameter inference*.

* {ref}`genindex`
* {ref}`modindex`
