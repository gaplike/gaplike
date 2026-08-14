"""gaplike — inference on gapped/windowed stationary Gaussian data.

Gap-pattern generation (arbitrary patterns, tapered or rectangular),
windowed frequency-domain covariances, and a hierarchy of likelihoods for
joint signal + noise-parameter estimation:

* exact time domain (restricted stationary covariance, two-component pencil),
* exact time domain at scale (matrix-free preconditioned conjugate
  gradients, :mod:`gaplike.cg` — no O(m^2) storage, no O(m^3) setup),
* full windowed frequency-domain covariance (simultaneous diagonalization),
* exact convolved diagonal (leakage-corrected Whittle),
* raw / window-power-scaled Whittle.

Companion package to "Zurückbleiben bitte: the impact of window functions on
noise and signal parameter inference" (Burke & Pozzoli).
"""
from . import gaps, psd, covariance, simulate, likelihood, waveform, cg
from .likelihood import DiagonalLikelihood, FullCovariance, TimeDomainExact
from .cg import RestrictedCG
from .waveform import Waveform

__version__ = "1.0.0"

__all__ = [
    "gaps", "psd", "covariance", "simulate", "likelihood", "waveform", "cg",
    "DiagonalLikelihood", "FullCovariance", "TimeDomainExact", "RestrictedCG",
    "Waveform", "__version__",
]
