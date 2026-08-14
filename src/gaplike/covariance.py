"""Windowed frequency-domain covariances and restricted time-domain
autocovariances of stationary noise.

Frequency-domain data convention
--------------------------------
The banded data vector of a windowed real series ``x`` is

    y_j = sqrt(2 dt / n) * rfft(w * x)_j ,   j in the analysis band,

so that ``E[y y^H]`` carries one-sided-PSD units and, for ``w = 1``, its
diagonal is the PSD.  ``full_covariance`` returns exactly ``E[y y^H]`` for
stationary noise of two-sided spectrum ``S2``; ``convolved_diag`` returns its
exact diagonal on the full one-sided grid at ``O(n log n)`` cost.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "band_indices", "rfft_scale", "transform",
    "convolved_diag", "full_covariance", "restricted_autocov",
]


def band_indices(n, dt, f_lo, f_hi):
    """Indices of the rfft bins strictly inside ``(f_lo, f_hi)``."""
    f = np.fft.rfftfreq(n, dt)
    return np.where((f > f_lo) & (f < f_hi))[0]


def rfft_scale(n, dt):
    """Periodogram normalization ``sqrt(2 dt / n)`` of the rfft."""
    return np.sqrt(2.0 * dt / n)


def transform(x_td, window, idx, dt):
    """Banded, periodogram-normalized rfft of the windowed series
    (``x_td``: shape ``(..., n)``)."""
    window = np.asarray(window, float)
    n = window.size
    return np.fft.rfft(window * x_td, axis=-1)[..., idx] * rfft_scale(n, dt)


def convolved_diag(window, S2):
    """Exact diagonal of the windowed covariance on the one-sided grid:
    circular convolution of ``|W|^2/n^2`` with the two-sided PSD, O(n log n)."""
    window = np.asarray(window, float)
    n = window.size
    k = np.abs(np.fft.fft(window))**2 / n**2
    return np.real(np.fft.ifft(np.fft.fft(S2) * np.fft.fft(k)))[:n // 2 + 1]


def full_covariance(window, S2, idx):
    """Dense windowed covariance ``E[y y^H]`` restricted to the rfft bins
    ``idx`` (band-restricted Algorithm 1 of Burke et al. 2025): the inner sum
    runs over the ENTIRE two-sided grid, so out-of-band leakage into the band
    is fully retained.  Cost ``O(len(idx)^2 n)`` via one matrix product; never
    forms an ``n x n`` object."""
    window = np.asarray(window, float)
    n = window.size
    idx = np.asarray(idx)
    W = np.fft.fft(window)
    l = np.arange(n)
    D = W[(idx[:, None] - l[None, :]) % n] * np.sqrt(S2)[None, :]
    return (D @ D.conj().T) / n**2


def restricted_autocov(S2, obs, dt):
    """Stationary autocovariance matrix restricted to the observed samples:
    ``C[i, j] = gamma[(obs_i - obs_j) mod n]`` with
    ``gamma = Re ifft(S2) / (2 dt)`` (circulant embedding of the Toeplitz
    covariance; exact for data generated on the circulant grid)."""
    S2 = np.asarray(S2, float)
    n = S2.size
    obs = np.asarray(obs)
    gamma = np.real(np.fft.ifft(S2)) / (2.0 * dt)
    lag = (obs[:, None] - obs[None, :]) % n
    return gamma[lag]
