"""Waveform interfaces.

The package is waveform-agnostic: a :class:`Waveform` wraps ANY callable
``theta -> h~(f)`` returning the multichannel continuous-Fourier-transform
waveform on the rfft grid ``rfftfreq(n, dt)`` (numpy sign convention), shape
``(nch, n//2 + 1)``.  From there:

* ``.td(theta)`` gives the physical time-domain template via the CFT->DFT
  rule ``h = irfft(h~ / dt)`` (so Whittle SNRs match the continuous integral),
* any :mod:`gaplike.likelihood` object consumes ``.td`` output through its
  ``transform`` / ``loglike_td`` methods.

An adapter for lisabeta massive-black-hole-binary TDI waveforms is provided
(:func:`lisabeta_mbhb_ae`); lisabeta is an optional dependency imported only
inside the factory.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Waveform", "central_derivs", "default_mbhb_params", "lisabeta_mbhb_ae"]


class Waveform:
    """Wrap a frequency-domain waveform callable.

    Parameters
    ----------
    fd_func : callable
        ``theta -> (nch, n//2 + 1)`` complex CFT waveform on the rfft grid.
    n, dt : grid length and cadence.
    """

    def __init__(self, fd_func, n, dt):
        self.fd_func = fd_func
        self.n = int(n)
        self.dt = float(dt)

    def fd(self, theta):
        """Frequency-domain CFT template ``(nch, n//2 + 1)``."""
        return np.asarray(self.fd_func(theta))

    def td(self, theta):
        """Physical time-domain template ``(nch, n)``: ``irfft(h~ / dt)``."""
        return np.fft.irfft(self.fd(theta) / self.dt, n=self.n, axis=-1)


def central_derivs(f, theta, steps):
    """Generic central finite differences of ``f(theta)`` (any array output):
    returns a list of ``df/dtheta_a`` for the parameters with ``steps[a]``
    not None; entries with ``steps[a] is None`` are returned as ``None``
    (caller supplies analytic derivatives)."""
    theta = np.asarray(theta, float)
    out = []
    for a, st in enumerate(steps):
        if st is None:
            out.append(None)
            continue
        tp = theta.copy(); tp[a] += st
        tm = theta.copy(); tm[a] -= st
        out.append((np.asarray(f(tp)) - np.asarray(f(tm))) / (2 * st))
    return out


# --------------------------------------------------------------------------
# lisabeta MBHB adapter (optional dependency)
# --------------------------------------------------------------------------
def default_mbhb_params(x):
    """Physical vector -> lisabeta parameter dict.

    ``x = (Mtot, q, chi1, chi2, log10 dL[Gpc], iota, phi, lambda, beta, psi,
    tc_frac)`` with L-frame angles; ``tc_frac`` (element 10) is applied
    separately as a frequency-domain phase."""
    Mtot, q, chi1, chi2, lgd, inc, phi, lam, bet, psi = x[:10]
    m1 = Mtot * q / (1 + q)
    m2 = Mtot / (1 + q)
    return {"m1": m1, "m2": m2, "chi1": chi1, "chi2": chi2, "Deltat": 0.0,
            "dist": 10**lgd * 1e3, "inc": inc, "phi": phi,
            "lambda": lam, "beta": bet, "psi": psi, "Lframe": True}


def lisabeta_mbhb_ae(n, dt, f_lo, f_hi, wf_kw=None, to_params=None,
                     tc_index=10):
    """:class:`Waveform` for the (A, E) TDI channels of an MBHB, evaluated by
    lisabeta on the analysis band only and summed over all modes of the
    approximant (default IMRPhenomHM).

    ``tc_index``: index of the coalescence-time parameter (as a fraction of
    the segment), applied as the FD phase ``exp(-2 i pi f tc T)`` common to
    every mode; ``None`` disables it.  ``to_params`` maps the parameter
    vector to a lisabeta dict (default :func:`default_mbhb_params`).
    """
    import lisabeta.lisa.lisa as lisa   # optional dependency

    wf_kw = dict(TDI='TDI2AET', TDIrescaled=False,
                 approximant="IMRPhenomHM") if wf_kw is None else dict(wf_kw)
    to_params = default_mbhb_params if to_params is None else to_params
    freqs = np.fft.rfftfreq(n, dt)
    band = (freqs > f_lo) & (freqs < f_hi)
    idx_band = np.where(band)[0]
    fb = freqs[band]
    t_obs = n * dt

    def fd_func(theta):
        o = lisa.GenerateLISATDIFreqseries_SMBH(to_params(theta), fb, **wf_kw)
        if tc_index is not None:
            ph = np.exp(-2j * np.pi * fb * (theta[tc_index] * t_obs))
        else:
            ph = 1.0
        sA = np.zeros(fb.size, complex)
        sE = np.zeros(fb.size, complex)
        for mkey in o:
            if isinstance(mkey, tuple):
                sA += np.conj(np.asarray(o[mkey]['chan1']))
                sE += np.conj(np.asarray(o[mkey]['chan2']))
        ht = np.zeros((2, freqs.size), complex)
        ht[0, idx_band] = sA * ph
        ht[1, idx_band] = sE * ph
        return ht

    return Waveform(fd_func, n, dt)
