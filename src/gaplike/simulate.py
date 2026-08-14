"""Simulation of stationary Gaussian noise in the time domain."""
from __future__ import annotations

import numpy as np

__all__ = ["noise_td"]


def noise_td(S_onesided, dt, rng, nch=1):
    """``(nch, n)`` independent stationary noise realizations with one-sided
    PSD ``S_onesided`` on the rfft grid (length ``n//2 + 1``, ``n`` even,
    ``S[0]`` ignored / DC-free).

    Exact circulant sampling: independent complex-normal rfft coefficients of
    variance ``S * n / (4 dt)`` per real/imaginary part (real Nyquist bin),
    inverse-transformed to the time domain.
    """
    S = np.asarray(S_onesided, float)
    n = 2 * (S.size - 1)
    out = np.empty((nch, n))
    s = np.sqrt(S * n / (4 * dt))
    for c in range(nch):
        Xf = rng.normal(0, s) + 1j * rng.normal(0, s)
        Xf[0] = 0.0
        Xf[-1] = rng.normal(0, np.sqrt(S[-1] * n / (2 * dt)))
        out[c] = np.fft.irfft(Xf, n=n)
    return out
