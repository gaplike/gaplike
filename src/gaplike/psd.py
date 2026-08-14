"""Noise power spectral densities.

A noise model is a sequence of **components**: callables ``S_k(f)`` returning
the one-sided PSD of component ``k`` at its *reference* amplitude.  The
modelled spectrum is

    S(f; lam) = sum_k 10^(2 lam_k) S_k(f),

with ``lam_k`` the (dimensionless) log10 deviations of the component
amplitudes from their reference values — the noise parameters inferred by
every likelihood in :mod:`gaplike.likelihood`.

The built-in model is the two-component (test-mass + OMS) LISA TDI-2 A/E
spectrum (SciRDv1 shapes).  Any user-defined callables work the same way;
components may also be passed to the likelihoods directly as precomputed
grids.
"""
from __future__ import annotations

import numpy as np

__all__ = [
    "C_LIGHT", "lisa_tdi2_ae", "two_sided_grid", "one_sided_grid", "as_two_sided",
]

C_LIGHT = 299792458.0


# --------------------------------------------------------------------------
# built-in: LISA TDI-2 A/E, test-mass + OMS components (SciRDv1 shapes)
# --------------------------------------------------------------------------
def lisa_tdi2_ae(tm_asd=3e-15, oms_asd=15e-12, arm_m=2.5e9, generation=2):
    """The two A/E-channel PSD components of second-generation TDI.

    Returns ``{"tm": S_tm(f), "oms": S_oms(f)}`` — callables giving the
    channel PSD (fractional frequency) of each single-link noise, transferred
    through the TDI combination, at the reference amplitudes ``tm_asd``
    [m s^-2 Hz^-1/2] and ``oms_asd`` [m Hz^-1/2].
    """
    avg_d = arm_m / C_LIGHT

    def _tm_link(f):
        return tm_asd**2 * (1 + (0.4e-3 / f)**2) * (1 + (f / 8e-3)**4) \
            / (2 * np.pi * f * C_LIGHT)**2

    def _oms_link(f):
        return oms_asd**2 * (1 + (2e-3 / f)**4) * (2 * np.pi * f / C_LIGHT)**2

    def _c_xx(f):
        wL = 2 * np.pi * f * avg_d
        c = 4 * np.sin(wL)**2
        if generation == 2:
            c *= 4 * np.sin(2 * wL)**2
        return c

    def s_tm(f):
        f = np.asarray(f, float)
        wL = 2 * np.pi * f * avg_d
        return _tm_link(f) * 4 * _c_xx(f) * (3 + 2 * np.cos(wL) + np.cos(2 * wL))

    def s_oms(f):
        f = np.asarray(f, float)
        wL = 2 * np.pi * f * avg_d
        return _oms_link(f) * 2 * _c_xx(f) * (2 + np.cos(wL))

    return {"tm": s_tm, "oms": s_oms}


# --------------------------------------------------------------------------
# grids
# --------------------------------------------------------------------------
def two_sided_grid(S, n, dt):
    """Two-sided PSD on the full DFT grid ``fftfreq(n, dt)`` (0 at DC)."""
    f = np.fft.fftfreq(n, dt)
    out = np.zeros(n)
    nz = f != 0
    out[nz] = S(np.abs(f[nz]))
    return out


def one_sided_grid(S, n, dt):
    """One-sided PSD on the rfft grid ``rfftfreq(n, dt)`` (0 at DC)."""
    f = np.fft.rfftfreq(n, dt)
    out = np.zeros(f.size)
    out[1:] = S(f[1:])
    return out


def as_two_sided(component, n, dt):
    """Normalize a component (callable or precomputed length-``n`` two-sided
    grid) to a two-sided grid."""
    if callable(component):
        return two_sided_grid(component, n, dt)
    a = np.asarray(component, float)
    if a.shape != (n,):
        raise ValueError(f"two-sided grid must have length n={n}, got {a.shape}")
    return a
