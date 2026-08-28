"""gaplike — inference on gapped/windowed stationary Gaussian data.

Gap-pattern generation (arbitrary patterns, tapered or rectangular),
windowed frequency-domain covariances, and a hierarchy of likelihoods for
joint signal + noise-parameter estimation:

* exact time domain (restricted stationary covariance, two-component pencil),
* exact time domain at scale (matrix-free preconditioned conjugate
  gradients, :mod:`gaplike.cg` — no O(m^2) storage, no O(m^3) setup),
* log-determinants for the matrix-free route (:mod:`gaplike.slq` — the
  exact Schur-complement identity, and stochastic Lanczos quadrature with
  shared-probe differences for flexible noise models),
* full windowed frequency-domain covariance (simultaneous diagonalization),
* exact convolved diagonal (leakage-corrected Whittle),
* raw / window-power-scaled Whittle.

Companion package to "Zurückbleiben bitte: the impact of window functions on
noise and signal parameter inference" (Burke & Pozzoli).
"""
from . import gaps, psd, covariance, simulate, likelihood, waveform, cg, slq
from .likelihood import DiagonalLikelihood, FullCovariance, TimeDomainExact
from .cg import RestrictedCG
from .slq import ComplementFactor
from .waveform import Waveform

__version__ = "0.2.0"

__all__ = [
    "gaps", "psd", "covariance", "simulate", "likelihood", "waveform", "cg",
    "slq", "DiagonalLikelihood", "FullCovariance", "TimeDomainExact",
    "RestrictedCG", "ComplementFactor", "Waveform", "__version__",
]
