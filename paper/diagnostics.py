"""Analytic diagnostics for (possibly mis-specified) covariance models.

NOISE CONVENTION: Sigma'(lam) = sum_k 10^(lam_k) C_k (log10-POWER deviations)
(paper-side module: kept OUT of the gaplike package on purpose).

Everything here follows the circularly-symmetric complex-normal convention
for ``nch`` independent channels with identical covariances:

* :func:`build_truth` — Fisher information under the TRUE windowed covariance
  ``Q`` (noise block ``nch Tr[Q+ Q_a Q+ Q_b]``, signal block
  ``2 sum_ch Re dh^H Q+ dh``).
* :class:`ApproxModel` — predictions for a model covariance ``Sigma'(lam)``
  analysed on data of true covariance ``Q``: Fisher blocks, expected score
  (leading-order noise bias), MLE scatter, and the two paper diagnostics
  ``Upsilon`` (true scatter / quoted width) and ``Xi`` (quoted width / width
  of the correct analysis).
* :func:`kl_pseudo_true_diag` / :func:`kl_pseudo_true_full` — the point the
  nonlinear noise MLE converges to in probability: the minimizer of the
  Kullback-Leibler divergence of the model family from the truth.
* :func:`sandwich_diag` — Godambe/White sandwich covariance ``H^-1 J H^-1``
  of the mis-specified noise MLE about the pseudo-true point.
"""
from __future__ import annotations

import numpy as np

LN10 = np.log(10.0)

__all__ = [
    "safe_inv", "build_truth", "ApproxModel", "snr_windowed",
    "kl_pseudo_true_diag", "kl_pseudo_true_full", "sandwich_diag",
]


def _dense(C):
    C = np.asarray(C)
    return np.diag(C).astype(complex) if C.ndim == 1 else C.astype(complex)


def _herm(A):
    return 0.5 * (A + A.conj().T)


def safe_inv(F, rcond=1e-12):
    """Scale-invariant eigenvalue-regularized inverse of a symmetric Fisher
    block: normalized to unit diagonal, eigenvalues of the correlation-like
    matrix floored at ``rcond * max`` (near-null directions get a huge-but-
    finite variance instead of float noise)."""
    F = 0.5 * (np.asarray(F) + np.asarray(F).T)
    d = np.sqrt(np.diag(F))
    Fn = F / np.outer(d, d)
    w, U = np.linalg.eigh(0.5 * (Fn + Fn.T))
    w = np.maximum(w, rcond * w.max())
    Ci = (U / w) @ U.T
    return Ci / np.outer(d, d)


def build_truth(true_components, dh, rcond=1e-13, nch=1):
    """Fisher information under the true covariance ``Q = sum(components)``.

    ``true_components``: list of banded matrices (or diagonals); ``dh``:
    ``(nch, nb, npar)`` signal derivatives of the transformed template."""
    Dc = [_dense(c) for c in true_components]
    Q = _herm(sum(Dc))
    Qi = np.linalg.pinv(Q, rcond=rcond, hermitian=True)
    dQ = [LN10 * d for d in Dc]
    Mi = [Qi @ dq for dq in dQ]
    npar = len(Dc)
    fisher_n = nch * np.array([[np.real(np.einsum('ij,ji->', Mi[a], Mi[b]))
                                for b in range(npar)] for a in range(npar)])
    fisher_s = sum(2.0 * np.real(dh[c].conj().T @ (Qi @ dh[c]))
                   for c in range(nch))
    return dict(Q=Q, Qi=Qi, fisher_n=fisher_n, cov_n=np.linalg.inv(fisher_n),
                fisher_s=fisher_s, cov_s=safe_inv(fisher_s))


class ApproxModel:
    """Analytic predictions for one model covariance ``Sigma'(lam) =
    sum_k 10^(2 lam_k) C_k`` under the true covariance ``truth['Q']``.

    ``comps``: model components — all 1-D (diagonal model) or all 2-D (dense);
    ``dh``: ``(nch, nb, npar)`` transformed signal derivatives;
    ``truth``: output of :func:`build_truth`.
    """

    def __init__(self, comps, dh, truth, rcond=1e-13, nch=1):
        comps = [np.asarray(c) for c in comps]
        self.diagonal = all(c.ndim == 1 for c in comps)
        self.npar = len(comps)
        self.nch = nch
        Q = truth["Q"]

        if self.diagonal:
            sp = np.real(sum(comps))
            keep = sp > rcond * sp.max()
            spi = np.where(keep, 1.0 / np.where(keep, sp, 1.0), 0.0)
            dS = [LN10 * np.real(c) for c in comps]
            A = [spi * d * spi for d in dS]
            T = np.array([np.sum(spi * d) for d in dS])
            fisher_blk = np.array([[np.sum(A[a] * dS[b]) for b in range(self.npar)]
                                   for a in range(self.npar)])
            AQ = [A[a][:, None] * Q for a in range(self.npar)]
            Spi_dh = spi[None, :, None] * dh
            QSpi_dh = np.einsum('ij,cjs->cis', Q, Spi_dh)
        else:
            Sp = _herm(sum(_dense(c) for c in comps))
            Spi = np.linalg.pinv(Sp, rcond=rcond, hermitian=True)
            dS = [LN10 * _dense(c) for c in comps]
            A = [Spi @ d @ Spi for d in dS]
            T = np.array([np.real(np.einsum('ij,ji->', Spi, d)) for d in dS])
            fisher_blk = np.array([[np.real(np.einsum('ij,ji->', A[a], dS[b]))
                                    for b in range(self.npar)]
                                   for a in range(self.npar)])
            AQ = [A[a] @ Q for a in range(self.npar)]
            Spi_dh = np.einsum('ij,cjs->cis', Spi, dh)
            QSpi_dh = np.einsum('ij,cjs->cis', Q, Spi_dh)

        self._A, self._T = A, T
        self._Spi_dh = Spi_dh

        # ---- noise sector ----
        self.fisher_n = nch * fisher_blk
        self.cov_n = np.linalg.inv(self.fisher_n)
        self.score_mean = nch * (np.array([np.real(np.trace(AQ[a]))
                                           for a in range(self.npar)]) - T)
        self.bias_n = self.cov_n @ self.score_mean
        conn = nch * np.array([[np.real(np.einsum('ij,ji->', AQ[a], AQ[b]))
                                for b in range(self.npar)] for a in range(self.npar)])
        self.scatter_n = self.cov_n @ (conn + np.outer(self.score_mean,
                                                       self.score_mean)) @ self.cov_n
        self.variance_n = self.cov_n @ conn @ self.cov_n

        # ---- signal sector ----
        self.fisher_s = sum(2.0 * np.real(dh[c].conj().T @ Spi_dh[c])
                            for c in range(nch))
        self.cov_s = safe_inv(self.fisher_s)
        Mmat = sum(2.0 * np.real(Spi_dh[c].conj().T @ QSpi_dh[c]) for c in range(nch))
        self.scatter_s = self.cov_s @ Mmat @ self.cov_s

        self.fisher_true_n = truth["fisher_n"]
        self.cov_true_n = truth["cov_n"]
        self.fisher_true_s = truth["fisher_s"]
        self.cov_true_s = truth["cov_s"]

        # ---- scatter-to-width (Upsilon) and width-to-width (Xi) ----
        wm_s = np.sqrt(np.diag(self.cov_s))
        wm_n = np.sqrt(np.diag(self.cov_n))
        self.upsilon_s = np.sqrt(np.abs(np.diag(self.scatter_s))) / wm_s
        self.upsilon_n = np.sqrt(np.abs(np.diag(self.scatter_n))) / wm_n
        self.xi_s = wm_s / np.sqrt(np.diag(self.cov_true_s))
        self.xi_n = wm_n / np.sqrt(np.diag(self.cov_true_n))

    def upsilon_n_bias(self, bias):
        """Noise-sector Upsilon with an externally supplied bias vector (e.g.
        the nonlinear KL pseudo-true bias) in place of the leading-order one."""
        return np.sqrt(np.abs(np.diag(self.variance_n)) + np.asarray(bias)**2) \
            / np.sqrt(np.diag(self.cov_n))

    def score_n(self, r):
        if self.diagonal:
            a2 = (np.abs(r)**2).sum(0)
            return np.array([np.sum(self._A[a] * a2)
                             for a in range(self.npar)]) - self.nch * self._T
        return np.array([sum(np.real(r[c].conj() @ (self._A[a] @ r[c]))
                             for c in range(self.nch))
                         for a in range(self.npar)]) - self.nch * self._T

    def mle_noise(self, r):
        return self.cov_n @ self.score_n(r)

    def mle_signal(self, r):
        s = sum(2.0 * np.real(self._Spi_dh[c].conj().T @ r[c])
                for c in range(self.nch))
        return self.cov_s @ s


def snr_windowed(h, Qi):
    """Matched-filter SNR of the transformed template under the true windowed
    covariance: ``sqrt(2 sum_ch Re h^H Q+ h)``."""
    return float(np.sqrt(sum(2.0 * np.real(h[c].conj() @ (Qi @ h[c]))
                             for c in range(h.shape[0]))))


# ==========================================================================
# nonlinear completions: KL pseudo-true point and Godambe/White sandwich
# ==========================================================================
def kl_pseudo_true_diag(v_components, q_diag):
    """Pseudo-true noise parameters of a two-component DIAGONAL model under a
    true covariance with diagonal ``q_diag``: argmin of the expected negative
    log-likelihood (= KL divergence up to a constant)."""
    from scipy.optimize import minimize
    v1, v2 = [np.real(np.asarray(v)) for v in v_components]
    qd = np.real(np.asarray(q_diag))

    def nll(x):
        var = 10.0**(x[0]) * v1 + 10.0**(x[1]) * v2
        return float(np.sum(np.log(var)) + np.sum(qd / var))

    res = minimize(nll, np.zeros(2), method="Nelder-Mead",
                   options=dict(xatol=1e-7, fatol=1e-10, maxiter=3000))
    return res.x


def kl_pseudo_true_full(fullcov, Q):
    """Pseudo-true noise parameters of a :class:`~gaplike.likelihood.FullCovariance`
    model under an arbitrary true covariance ``Q`` (in the same normalized
    units as the model matrices)."""
    from scipy.optimize import minimize
    wq = np.real(np.einsum('ij,jk,ik->i', fullcov.Tproj, Q, fullcov.Tproj.conj()))

    def nll(x):
        v = 10.0**(x[0]) * fullcov.mu + 10.0**(x[1])
        return float(np.sum(np.log(v)) + np.sum(wq / v))

    res = minimize(nll, np.zeros(2), method="Nelder-Mead",
                   options=dict(xatol=1e-7, fatol=1e-10, maxiter=3000))
    return res.x


def sandwich_diag(v_components, Q, nch=1):
    """KL pseudo-true point and Godambe/White sandwich covariance
    ``H^-1 J H^-1`` of the noise MLE of a two-component diagonal model under
    the true covariance ``Q``:

    * ``H``: Hessian of the EXPECTED negative log-likelihood at the
      pseudo-true point,
    * ``J``: covariance of the score there — for circularly-symmetric complex
      data ``Cov(|r_j|^2, |r_k|^2) = |Q_jk|^2`` per channel, so
      ``J = nch A_a^T |Q|^2 A_b`` with ``A_a = dS_a / var^2``.

    Returns ``(lam_star, cov)``.
    """
    v1, v2 = [np.real(np.asarray(v)) for v in v_components]
    qd = np.real(np.diag(Q))
    lam_star = kl_pseudo_true_diag([v1, v2], qd)
    var = 10.0**(lam_star[0]) * v1 + 10.0**(lam_star[1]) * v2
    dS = [LN10 * 10.0**(lam_star[0]) * v1,
          LN10 * 10.0**(lam_star[1]) * v2]
    A = [d / var**2 for d in dS]
    absQ2 = np.abs(Q)**2
    J = nch * np.array([[float(A[a] @ (absQ2 @ A[b])) for b in range(2)]
                        for a in range(2)])

    def nll_exp(x):
        v = 10.0**(x[0]) * v1 + 10.0**(x[1]) * v2
        return nch * float(np.sum(np.log(v)) + np.sum(qd / v))

    h = 1e-4
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            xpp = lam_star.copy(); xpp[i] += h; xpp[j] += h
            xpm = lam_star.copy(); xpm[i] += h; xpm[j] -= h
            xmp = lam_star.copy(); xmp[i] -= h; xmp[j] += h
            xmm = lam_star.copy(); xmm[i] -= h; xmm[j] -= h
            H[i, j] = (nll_exp(xpp) - nll_exp(xpm) - nll_exp(xmp)
                       + nll_exp(xmm)) / (4 * h * h)
    H = 0.5 * (H + H.T)
    Hi = np.linalg.inv(H)
    return lam_star, Hi @ J @ Hi
