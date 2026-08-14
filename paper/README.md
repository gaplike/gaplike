# Paper reproduction — gapped-MBHB study

Reproduces every figure of *"Zurückbleiben bitte"* (Burke & Pozzoli): joint
13-parameter (11 signal + 2 noise) inference of an IMRPhenomHM MBHB and the
TM/OMS noise amplitudes on a 12 h LISA TDI-2 A/E segment, under three
controlled gap scenarios:

* **A — two long gaps** (1 h each, 0.3 h Planck tapers), SNR 585
* **B — twelve short gaps** (9 min hourly, 0.05 h tapers), SNR 743
* **C — drastic gaps** (rectangular comb, 150 s removed every 750 s), SNR 319
  (no-gap reference: SNR 902)

Every algorithm lives in the `gaplike` package; the modules here only pin the
paper configuration and drive the runs.

## Requirements

```bash
uv pip install -e "..[pe]"   # gaplike + emcee + corner + matplotlib
# plus lisabeta (waveform): https://gitlab.in2p3.fr/marsat/lisabeta
# (lisabeta is NOT needed for the CG scaling benchmark, step 4)
```

## Reproduce

```bash
# 1. all paper figures from the CACHED chains in results/ (no sampling)
python make_figures.py             # ~10 min -> figures/
python plot_scenC.py               # scenario-C 3-way corner + C colormap
python plot_ABC.py                 # A/B/C full-covariance corner overlay
python fig_cov_colormap.py         # A/B/C covariance colormaps

# 2. full re-run of the PE from scratch (overwrites results/)
python driver.py                   # A/B x {full,diag,bare,psd}: ~80 min on 2 cores
python scenC_run.py                # C: FD full + conv-diag + exact TD (pencil) PE
python ensemble_noise.py           # 300-realization ensemble verification

# 3. the CG-vs-dense scaling benchmark (Fig. cg_scaling) — gaplike only,
#    no lisabeta, no chains: dense solves against matrix-free
#    preconditioned conjugate gradients, no structure granted to either
python fig_cg_scaling.py           # ~15-30 min -> results/cg_scaling.json
python _mkfig.py                   # -> figures/cg_scaling.{png,pdf}
```

The regression suite that used to live here (`verify_pipeline.py`) moved into
the package tests: `pytest ../tests` reproduces the validated pipeline
numbers from `tests/data/reference.json`.

Cached chains are thinned ×2 and stored as float32 (posteriors are visually
identical; re-run step 3 for full-resolution chains).

## Layout

| file | content |
|---|---|
| `core.py` | paper configuration (grids, band, truth, priors' parent constants) + thin facades over `gaplike` (`FullCovTrick`, `ApproxModel`, `freq_cov`, ...) and the lisabeta waveform/derivative wiring |
| `scenarios.py` | scenario builder A/B (+ no-gap): windows, per-component covariances, models, shared data realization, SNRs |
| `scenC.py` | scenario C builder + `TDPencil` facade over `gaplike.likelihood.TimeDomainExact`; TD templates/derivatives |
| `pe.py` | emcee PE (13 parameters, 4 covariance models), exact profile noise MLE, KL pseudo-true + Godambe/White sandwich (via the local `diagnostics.py`) |
| `plots.py` | corner plots (PE vs truncated-Gaussian analytics vs exact MLEs), Υ/Ξ figures, overview figure |
| `driver.py`, `scenC_run.py`, `ensemble_noise.py` | batch runners (PE jobs, scenario-C analytics + TD PE, ensemble verification) |
| `fig_cov_colormap.py`, `plot_scenC.py`, `plot_ABC.py` | standalone figure scripts (covariance colormaps, scenario-C 3-way corners, A/B/C full-covariance overlay) |
| `make_figures.py` | one entry point: regenerates the chain-based figures from `results/` |
| `diagnostics.py` | analytic machinery on top of gaplike: Fisher under the true covariance, ApproxModel (biases, scatter, Υ/Ξ), KL pseudo-true, Godambe/White sandwich |
| `fig_cg_scaling.py`, `_mkfig.py` | CG-vs-dense scaling benchmark and its figure (depends on `gaplike` only) |
| `results/`, `figures/` | cached PE chains + `cg_scaling.json` (reference timings: 2 cores, Intel Xeon 2.80 GHz); regenerated figures |
