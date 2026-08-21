"""Likelihoods for gapped/windowed stationary Gaussian data.

Four tiers, in decreasing order of fidelity and cost:

* :class:`TimeDomainExact` — the exact likelihood of the observed samples
  (marginalization of a stationary Gaussian over the gap samples = row/column
  deletion).  No window appears anywhere.  With two noise components entering
  linearly, one simultaneous diagonalization ("pencil") gives an O(m)
  likelihood per evaluation with a closed-form determinant.
* :class:`FullCovariance` — complex-normal likelihood of the windowed,
  band-restricted rfft data with the DENSE windowed covariance
  ``Sigma(lam) = 10^(lam_0) C_0 + 10^(lam_1) C_1``; same two-component
  pencil trick, reduced to the rank retained by the gaps.
* :class:`DiagonalLikelihood` (``convolved``) — Whittle-like likelihood with
  the EXACT diagonal of the windowed covariance (leakage-corrected PSD).
* :class:`DiagonalLikelihood` (``whittle``) — the raw PSD on the band,
  optionally rescaled by the window power ``W_2`` ("normalizing constant"
  approximation).

Common conventions
------------------
* Noise parameters ``lam = (lam_0, lam_1, ...)``: log10 deviations of the
  component POWERS (PSDs), ``Sigma(lam) = sum_k 10^(lam_k) C_k``; truth at 0.
  (v0.1 used amplitude deviations, ``10^(2 lam)``: lam_new = 2 lam_old.)
* Residuals are multichannel, shape ``(nch, .)`` — channels independent with
  identical covariances (e.g. LISA A and E).
* Frequency-domain classes expose ``transform(x_td)`` mapping time series to
  the internally normalized data vector, and ``loglike_td`` for convenience;
  ``loglike(resid, lam)`` takes already-transformed residuals.
* Log-likelihoods contain constant offsets that depend on the internal
  normalization ``s_ref``; every DIFFERENCE of log-likelihoods (between
  parameter points, or between templates) is independent of it.
"""
from __future__ import annotations

import numpy as np

from . import covariance as _cov
from . import gaps as _gaps
from .psd import as_two_sided

__all__ = ["DiagonalLikelihood", "FullCovariance", "TimeDomainExact", "LN10"]

LN10 = np.log(10.0)


def _as_grids(components, n, dt):
    return [as_two_sided(c, n, dt) for c in components]


def _safe_inv(F, rcond=1e-12):
    """Scale-invariant eigenvalue-regularized inverse of a symmetric Fisher
    block (near-null directions floored instead of amplified)."""
    F = 0.5 * (np.asarray(F) + np.asarray(F).T)
    d = np.sqrt(np.diag(F))
    Fn = F / np.outer(d, d)
    w, U = np.linalg.eigh(0.5 * (Fn + Fn.T))
    w = np.maximum(w, rcond * w.max())
    Ci = (U / w) @ U.T
    return Ci / np.outer(d, d)


# ==========================================================================
# frequency-domain base: window / band / normalization bookkeeping
# ==========================================================================
class _FDLikelihood:
    """Shared machinery: y = rfft(w x)[band] * sqrt(2 dt / n) / sqrt(s_ref)."""

    def _setup_fd(self, window, dt, idx, s_ref):
        self.window = np.asarray(window, float)
        self.n = self.window.size
        self.dt = float(dt)
        self.idx = np.asarray(idx)
        self.n_band = self.idx.size
        self.s_ref = float(s_ref)

    def transform(self, x_td):
        """Time series ``(..., n)`` -> normalized banded data ``(..., n_band)``."""
        y = _cov.transform(x_td, self.window, self.idx, self.dt)
        return y / np.sqrt(self.s_ref)

    def loglike_td(self, resid_td, lam):
        """Log-likelihood from a time-domain residual ``(nch, n)``."""
        return self.loglike(self.transform(resid_td), lam)


# ==========================================================================
# diagonal tiers (convolved / Whittle / scaled Whittle) — any K components
# ==========================================================================
class DiagonalLikelihood(_FDLikelihood):
    """Complex-normal likelihood with a diagonal covariance model
    ``var_j(lam) = sum_k 10^(lam_k) v_kj`` on the analysis band."""

    def __init__(self, variances):
        self.v = [np.real(np.asarray(v, float)) for v in variances]
        self.k = len(self.v)

    # ---- constructors -----------------------------------------------------
    @classmethod
    def convolved(cls, window, components, dt, f_lo, f_hi, s_ref=None):
        """Exact leading diagonal of the windowed covariance, per component
        (O(n log n) each; no dense matrix is ever formed)."""
        window = np.asarray(window, float)
        n = window.size
        idx = _cov.band_indices(n, dt, f_lo, f_hi)
        grids = _as_grids(components, n, dt)
        v = [_cov.convolved_diag(window, g)[idx] for g in grids]
        if s_ref is None:
            s_ref = float(np.median(sum(v)))
        obj = cls([vk / s_ref for vk in v])
        obj._setup_fd(window, dt, idx, s_ref)
        return obj

    @classmethod
    def whittle(cls, window, components, dt, f_lo, f_hi,
                scale_by_window_power=False, s_ref=None):
        """Raw one-sided PSD on the band (no leakage correction); with
        ``scale_by_window_power=True`` every component is multiplied by
        ``W_2 = sum(w^2)/n`` (the 'normalizing constant' approximation)."""
        window = np.asarray(window, float)
        n = window.size
        idx = _cov.band_indices(n, dt, f_lo, f_hi)
        fb = np.fft.rfftfreq(n, dt)[idx]
        v = []
        for c in components:
            if callable(c):
                v.append(np.asarray(c(fb), float))
            else:
                v.append(as_two_sided(c, n, dt)[idx])
        if scale_by_window_power:
            w2 = _gaps.window_power(window)
            v = [w2 * vk for vk in v]
        if s_ref is None:
            s_ref = float(np.median(sum(v)))
        obj = cls([vk / s_ref for vk in v])
        obj._setup_fd(window, dt, idx, s_ref)
        return obj

    # ---- evaluation ---------------------------------------------------------
    def variances(self, lam):
        lam = np.atleast_1d(np.asarray(lam, float))
        return sum(10.0**(lam[k]) * self.v[k] for k in range(self.k))

    def loglike(self, resid, lam):
        """``resid``: (nch, n_band) transformed residuals."""
        var = self.variances(lam)
        nch = resid.shape[0]
        return float(-nch * np.sum(np.log(var))
                     - np.sum(np.abs(resid)**2 / var[None, :]))


# ==========================================================================
# full dense windowed covariance — two components, simultaneous diagonalization
# ==========================================================================
class FullCovariance(_FDLikelihood):
    """Dense-covariance complex-normal likelihood,
    ``Sigma(lam) = 10^(lam_0) C_0 + 10^(lam_1) C_1`` (exactly two
    components), diagonalized ONCE in the rank retained by the gaps:

    * eigendecomposition of ``C_0 + C_1`` truncated at ``rtol_rank`` discards
      the singular directions annihilated by the gaps,
    * ``C_1`` is whitened and the remaining symmetric pencil diagonalized,
      producing a fixed projector ``Tproj`` with
      ``Tproj C_0 Tproj^H = diag(mu)``, ``Tproj C_1 Tproj^H = I``,
    * every likelihood evaluation is then O(rank) per channel with a
      closed-form log-determinant.

    Extending to K > 2 components (or nonlinearly-parametrized shapes) would
    require a fresh factorization per evaluation — use the convolved diagonal
    or the time-domain solver instead.
    """

    def __init__(self, components, rtol_rank=1e-8):
        if len(components) != 2:
            raise ValueError("FullCovariance requires exactly 2 component matrices")
        C0, C1 = (np.asarray(c) for c in components)
        A0 = 0.5 * ((C0 + C1) + (C0 + C1).conj().T)
        wv, U = np.linalg.eigh(A0)
        keep = wv > rtol_rank * wv.max()
        V = U[:, keep]
        C0r = V.conj().T @ C0 @ V
        C1r = V.conj().T @ C1 @ V
        C0r = 0.5 * (C0r + C0r.conj().T)
        C1r = 0.5 * (C1r + C1r.conj().T)
        sc, Uc = np.linalg.eigh(C1r)
        sc = np.clip(sc, 1e-30, None)
        Bi = (Uc * (1.0 / np.sqrt(sc))[None, :]).conj().T
        mu, Wm = np.linalg.eigh(Bi @ C0r @ Bi.conj().T)
        self.rank = int(keep.sum())
        self.mu = np.clip(np.real(mu), 0.0, None)
        self.Tproj = (Wm.conj().T @ Bi) @ V.conj().T
        self.logdet_C1 = float(np.sum(np.log(sc)))

    # ---- constructor from a window ------------------------------------------
    @classmethod
    def from_window(cls, window, components, dt, f_lo, f_hi,
                    rtol_rank=1e-8, s_ref=None):
        """Build the two banded windowed covariance matrices from a window and
        two PSD components (callables or two-sided grids), then diagonalize."""
        window = np.asarray(window, float)
        n = window.size
        idx = _cov.band_indices(n, dt, f_lo, f_hi)
        grids = _as_grids(components, n, dt)
        if len(grids) != 2:
            raise ValueError("FullCovariance requires exactly 2 components")
        mats = [_cov.full_covariance(window, g, idx) for g in grids]
        if s_ref is None:
            s_ref = float(np.median(np.real(np.diag(mats[0] + mats[1]))))
        obj = cls([m / s_ref for m in mats], rtol_rank=rtol_rank)
        obj._setup_fd(window, dt, idx, s_ref)
        return obj

    # ---- evaluation -----------------------------------------------------------
    @property
    def r(self):
        return self.rank

    def variances(self, lam):
        a0, a1 = lam
        return 10.0**(a0) * self.mu + 10.0**(a1)

    def loglike(self, resid, lam):
        """``resid``: (nch, n_band) transformed residuals."""
        Z = self.Tproj @ resid.T
        v = self.variances(lam)
        nch = resid.shape[0]
        return float(-np.sum(np.abs(Z)**2 / v[:, None])
                     - nch * (np.sum(np.log(v)) + self.logdet_C1
                              + self.rank * np.log(np.pi)))


# ==========================================================================
# exact time domain — restricted stationary covariance, two-component pencil
# ==========================================================================
class TimeDomainExact:
    """EXACT likelihood of gapped stationary data: the observed samples of a
    stationary Gaussian are jointly Gaussian with covariance
    ``Sigma_OO(lam) = 10^(2 lam_0) C_0,OO + 10^(2 lam_1) C_1,OO`` — the
    stationary (circulant) covariance with the gap rows/columns deleted.
    No window or taper appears anywhere; scatter/width and model/true width
    ratios are 1 by construction.

    With two components entering linearly, one simultaneous diagonalization
    (cost O(m^3), once) reduces every likelihood evaluation to O(m) with a
    closed-form determinant — practical for m up to ~10^4 observed samples.

    Parameters
    ----------
    mask : bool or float array, length n
        Observed-sample mask (or gate; strictly-positive entries = observed).
    components : sequence of two callables or ndarrays
        The two PSD components, as callables of ``f`` or two-sided grids.
    dt : float
        Sample cadence [s].
    """

    def __init__(self, mask, components, dt):
        mask = np.asarray(mask)
        n = mask.size
        obs = np.where(mask > 0)[0] if mask.dtype != bool else np.where(mask)[0]
        grids = _as_grids(components, n, dt)
        if len(grids) != 2:
            raise ValueError("TimeDomainExact requires exactly 2 components")
        C0 = _cov.restricted_autocov(grids[0], obs, dt)
        C1 = _cov.restricted_autocov(grids[1], obs, dt)
        wv, U = np.linalg.eigh(0.5 * (C1 + C1.T))
        wv = np.clip(wv, wv.max() * 1e-14, None)
        Bi = (U / np.sqrt(wv)).T                     # Bi C1 Bi^T = I
        Mm = Bi @ C0 @ Bi.T
        mu, W = np.linalg.eigh(0.5 * (Mm + Mm.T))
        self.mu = np.clip(mu, 0.0, None)
        self.T = W.T @ Bi                            # T C0 T^T = diag(mu), T C1 T^T = I
        self.logdet_C1 = float(np.sum(np.log(wv)))
        self.obs = obs
        self.m = int(obs.size)
        self.n = int(n)
        self.dt = float(dt)

    # ---- data reduction -------------------------------------------------------
    def transform(self, x_td):
        """Time series ``(..., n)`` -> observed samples ``(..., m)``."""
        return np.asarray(x_td)[..., self.obs]

    def loglike_td(self, resid_td, lam):
        return self.loglike(self.transform(resid_td), lam)

    # ---- evaluation -----------------------------------------------------------
    def variances(self, lam):
        a0, a1 = lam
        return 10.0**(a0) * self.mu + 10.0**(a1)

    def loglike(self, resid_obs, lam):
        """``resid_obs``: (nch, m) real residuals on the observed samples.
        (Constant ``-nch m log(2 pi)/2`` omitted.)"""
        z = self.T @ resid_obs.T                     # (m, nch)
        v = self.variances(lam)
        nch = resid_obs.shape[0]
        return float(-0.5 * (np.sum(z**2 / v[:, None])
                             + nch * np.sum(np.log(v))
                             + nch * self.logdet_C1))

    # ---- exact Fisher blocks ---------------------------------------------------
    def fisher_noise(self, nch=1, lam=(0.0, 0.0)):
        """2x2 noise-parameter Fisher matrix at ``lam``."""
        v = self.variances(lam)
        wa = LN10 * 10.0**(lam[0]) * self.mu / v
        wb = LN10 * 10.0**(lam[1]) * 1.0 / v
        G = 0.5 * np.array([[np.sum(wa * wa), np.sum(wa * wb)],
                            [np.sum(wa * wb), np.sum(wb * wb)]])
        return nch * G

    def fisher_signal(self, dh_obs, lam=(0.0, 0.0)):
        """Signal Fisher matrix; ``dh_obs``: (nch, m, npar) template derivatives
        on the observed samples."""
        v = self.variances(lam)
        G = 0.0
        for c in range(dh_obs.shape[0]):
            Z = self.T @ dh_obs[c]                   # (m, npar)
            G = G + Z.T @ (Z / v[:, None])
        return G

    # ---- linearized MLEs --------------------------------------------------------
    def score_noise(self, resid_obs, lam=(0.0, 0.0)):
        z = self.T @ resid_obs.T
        v = self.variances(lam)
        p = (z**2).sum(axis=1)                       # sum over channels
        nch = resid_obs.shape[0]
        sa = 0.5 * np.sum((p / v**2 - nch / v) * (LN10 * 10.0**(lam[0]) * self.mu))
        sb = 0.5 * np.sum((p / v**2 - nch / v) * (LN10 * 10.0**(lam[1]) * 1.0))
        return np.array([sa, sb])

    def mle_noise(self, resid_obs, lam=(0.0, 0.0)):
        nch = resid_obs.shape[0]
        return np.linalg.solve(self.fisher_noise(nch, lam),
                               self.score_noise(resid_obs, lam))

    def mle_signal(self, dh_obs, resid_obs, lam=(0.0, 0.0), rcond=1e-12):
        v = self.variances(lam)
        s = 0.0
        for c in range(dh_obs.shape[0]):
            Z = self.T @ dh_obs[c]
            z = self.T @ resid_obs[c]
            s = s + Z.T @ (z / v)
        G = self.fisher_signal(dh_obs, lam)
        return _safe_inv(G, rcond=rcond) @ s
