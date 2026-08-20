# gaplike

**Inference on gapped / windowed stationary Gaussian data.**

`gaplike` provides everything needed to do joint signal + noise-parameter
estimation when a stationary time series is interrupted by data gaps (or
multiplied by any window): gap-pattern generation with arbitrary patterns,
windowed frequency-domain covariances, and a hierarchy of likelihoods — from
the cheap Whittle approximation to the dense windowed covariance and the
**exact time-domain likelihood**, evaluated either in closed form (two
components, simultaneous diagonalization) or **matrix-free at any scale by
preconditioned conjugate gradients** (`gaplike.cg`: the covariance is never
formed, one solve is a few hundred FFT pairs).

Companion package to *"Zurückbleiben bitte: the impact of window functions on
noise and signal parameter inference"* (O. Burke & F. Pozzoli); the `paper/`
folder reproduces every figure of that paper.

Only `numpy` and `scipy` are required. The two-component LISA TDI-2 A/E noise
model is built in; any user-defined PSD components and any waveform work the
same way.

<img src="assets/gap_leakage_web.gif" width="600"/>

## Install

```bash
uv venv && uv pip install -e .          # library only (numpy + scipy)
uv pip install -e ".[pe]"               # + emcee, corner, matplotlib (paper pipeline, notebooks)
uv pip install -e ".[dev]"              # + pytest
```

(or the same lines with plain `pip`; there is nothing uv-specific in the
package).

The MBHB waveform adapter additionally needs
[lisabeta](https://gitlab.in2p3.fr/marsat/lisabeta), which is **not** pulled
in by any of the lines above:

```bash
uv pip install -e ".[pe,mbhb]"          # + lisabeta (from PyPI)
```

`[mbhb]` is a separate extra because exactly one factory needs it. Everything
else — every likelihood, every gap builder, the CG solver, the paper figures
from the cached chains, and `notebooks/exact_inference_demo.ipynb` — runs
without lisabeta, and the import happens inside
`gaplike.waveform.lisabeta_mbhb_ae`, so a missing install fails only there.

Installing from PyPI gives wheels. Building the GitLab sources instead
requires FFTW and a CMake toolchain, and the FFTW location has to be handed
to the build explicitly:

```bash
brew install fftw                       # macOS;  apt install libfftw3-dev on Debian/Ubuntu
SKBUILD_CMAKE_DEFINE="FFTW_ROOT=$(brew --prefix fftw)" \
  uv pip install "lisabeta @ git+https://gitlab.in2p3.fr/marsat/lisabeta.git"
```

## Quickstart

```python
import numpy as np
import gaplike as gl

n, dt = 2880, 15.0                          # 12 h at 15 s cadence
f_lo, f_hi = 1e-4, 3.1e-2                   # analysis band [Hz]

# --- 1. gaps: ANY pattern -------------------------------------------------
mask = gl.gaps.periodic_mask(n, 10, 50)              # comb: 150 s out of 750 s
# mask = gl.gaps.random_mask(n, dt, rate_per_day=40, duration_s=600, rng=1)
# mask = gl.gaps.mask_from_intervals(n, dt, [(21000, 25000)])
gate = gl.gaps.gate_from_mask(mask, dt, taper_s=0.0) # rectangular (or tapered)
w    = gl.gaps.effective_window(gate, gl.gaps.segment_window(n, 0.05))

# --- 2. noise model: two components; lam = log10 deviations of the component
#        POWERS, so Sigma(lam) = 10^(lam_0) C_0 + 10^(lam_1) C_1, truth at 0
comps = list(gl.psd.lisa_tdi2_ae().values())         # [S_tm(f), S_oms(f)]
S_tot = gl.psd.one_sided_grid(lambda f: sum(c(f) for c in comps), n, dt)
x = gl.simulate.noise_td(S_tot, dt, np.random.default_rng(0), nch=2)

# --- 3. likelihood hierarchy ------------------------------------------------
L_full = gl.FullCovariance.from_window(w, comps, dt, f_lo, f_hi)   # dense windowed cov
L_conv = gl.DiagonalLikelihood.convolved(w, comps, dt, f_lo, f_hi) # exact diagonal
L_whit = gl.DiagonalLikelihood.whittle(w, comps, dt, f_lo, f_hi,
                                       scale_by_window_power=True) # ~ Whittle
L_td   = gl.TimeDomainExact(mask, comps, dt)                       # EXACT, no window

for L in (L_td, L_full, L_conv, L_whit):
    r = L.transform(x)                       # residual = data - template
    print(type(L).__name__, L.loglike(r, (0.0, 0.0)) - L.loglike(r, (0.05, 0.0)))

# --- 4. the same exact likelihood, matrix-free (large N) --------------------
rcg  = gl.RestrictedCG(mask, comps, dt, rtol=1e-8)   # no O(m^2) storage, no O(m^3) setup
quad, iters = rcg.quad_form((0.0, 0.0), L_td.transform(x))
# pair with a determinant: closed form from TimeDomainExact (2 components),
# or stochastic Lanczos quadrature in general
```

Each tier carries its own constant offset from its internal normalization, so
log-likelihoods are **not** comparable across tiers — only *differences*
(between parameter points, or between templates) are meaningful.

Templates from any waveform: wrap a frequency-domain callable in
`gl.Waveform(fd_func, n, dt)` and use `.td(theta)`; then
`resid = L.transform(data_td - h_td)`. For LISA MBHBs,
`gl.waveform.lisabeta_mbhb_ae(n, dt, f_lo, f_hi)` gives the (A, E) TDI-2
IMRPhenomHM waveform used in the paper — with any source parameters, any
parametrization (`to_params`) and any approximant (`wf_kw`); see
`notebooks/mbhb_gaps_demo.ipynb`.

## Reproducing the paper figures

All commands run from `paper/` with the `[pe]` extra installed.  The cached
PE chains ship in `paper/results/`, so every chain-based figure regenerates
in minutes without lisabeta and without sampling; lisabeta and the from-
scratch runs are needed only to regenerate the chains themselves.

```bash
cd paper

# -- chain-based figures, from the cached results/ (no lisabeta needed) -----
python make_figures.py        # corner_key_{A,B}, corner_noise_{A,B},
                              # corner_fullkey_{A,B}, upsilon_xi(.png/_heatmap),
                              # overview                           (~10 min)
python plot_scenC.py          # corner_key_C_full_diag_td, corner_noise_C_...,
                              # cov_colormap_C
python plot_ABC.py            # corner_key_full_ABC
python fig_cov_colormap.py    # cov_colormap_ABC

# -- the CG scaling figure (gaplike only: no lisabeta, no chains) -----------
python fig_cg_scaling.py      # times a^T Sigma_OO^-1 a, dense vs matrix-free
                              # preconditioned CG -> results/cg_scaling.json
                              # (~15-30 min; --kmax-cg/--kmax-dense to shorten)
python _mkfig.py              # -> figures/cg_scaling.{png,pdf}

# -- full regeneration of the chains (lisabeta + hours of sampling) ---------
python driver.py              # A/B x {full,diag,bare,psd}   (~80 min, 2 cores)
python scenC_run.py           # C: FD full + convolved diagonal + exact TD PE
python ensemble_noise.py      # 300-realization ensemble (Table: pseudo-true
                              # points, sandwich scatter)
```

Figure-to-script map: every `corner_*`, `upsilon_xi*` and `overview` come
from `make_figures.py`; the three standalone scripts above cover the
scenario-C corner, the A/B/C overlay corner and the covariance colormaps;
`fig_cg_scaling.py` + `_mkfig.py` produce the dense-vs-CG scaling figure.

## The likelihood hierarchy

| class | covariance model | cost / eval | guarantees |
|---|---|---|---|
| `TimeDomainExact` | stationary covariance restricted to observed samples | O(m) after one O(m³) setup | **exact**: scatter/width = width/true-width = 1 by construction; no window anywhere |
| `FullCovariance` | dense windowed FD covariance (band-restricted) | O(r) after one O(N_b³) setup | exact within the band-restricted, circularly-symmetric FD reduction |
| `DiagonalLikelihood.convolved` | exact diagonal of the windowed covariance | O(N_b) | unbiased widths *per bin*; ignores bin–bin correlations |
| `DiagonalLikelihood.whittle` | raw PSD (optionally × window power W₂) | O(N_b) | Whittle / "normalizing constant" approximation |
| `gaplike.cg` / `RestrictedCG` | same model as `TimeDomainExact`, matrix-free | 4 FFTs × a few hundred iterations, **no setup, no storage** | exact quadratic forms to a chosen tolerance; any number of components; determinant not included |

Noise parameters are the same throughout: `lam_k` are log10 deviations of the
component **powers** from their reference values, `Sigma(lam) = sum_k
10^(lam_k) C_k`, with the truth at `lam = 0`.

Both pencil classes (`TimeDomainExact`, `FullCovariance`) additionally assume
the covariance is linear in **exactly two** components: one simultaneous
diagonalization then makes every likelihood evaluation, Fisher matrix and
determinant closed-form. The diagonal tiers and the conjugate gradient route
accept any number of components; `gaplike.cg` is the escape hatch when the
two-component structure is not available (spline-knot spectra, extra
components) or when m is too large to factorize anything.

The paper's analytic mis-specification machinery (Fisher blocks under the
true windowed covariance, leading-order biases, MLE scatter, the Υ/Ξ
diagnostics, KL pseudo-true parameters, Godambe–White sandwich) lives in
`paper/diagnostics.py`, on top of the package.

## Repository layout

| path | content |
|---|---|
| `src/gaplike/` | the package: `gaps`, `psd`, `covariance`, `simulate`, `likelihood`, `cg`, `waveform` |
| `tests/` | unit tests (small-N brute-force exactness of every tier, CG vs dense) + machine-precision regression against the paper pipeline |
| `notebooks/exact_inference_demo.ipynb` | executable end-to-end example (**no lisabeta**): gapped two-channel LISA noise + toy chirp, joint 4-parameter PE with `TimeDomainExact` and `FullCovariance`, comparison corner |
| `notebooks/mbhb_gaps_demo.ipynb` | inject an MBHB with arbitrary parameters (lisabeta), compare five gap configurations, and set up a full 13-parameter inference ready to launch |
| `paper/` | full paper reproduction (scenarios A/B/C, PE, every figure incl. the CG scaling benchmark) — see `paper/README.md` |

## Tests

```bash
uv pip install -e ".[dev]"
pytest                    # ~40 s; the paper regression needs tests/data/reference.json
```

`tests/test_paper_regression.py` rebuilds the paper's three gap scenarios
purely from package primitives and reproduces the original pipeline's
windows, covariances, noise realization and all likelihood values to ~1e-9.

## Citing

If you use `gaplike`, please cite the software (see `CITATION.cff`) together
with Burke & Pozzoli (2026), *Zurückbleiben bitte: the impact of window
functions on noise and signal parameter inference* (in prep.).

## License

MIT — see `LICENSE`.
