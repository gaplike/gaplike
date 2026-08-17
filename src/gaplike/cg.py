"""Matrix-free preconditioned conjugate gradients for gapped stationary data.

The exact likelihood of the observed samples of a stationary Gaussian needs
quadratic forms in the inverse of

    Sigma_OO = R Sigma R^T,

the stationary (circulant) covariance restricted to the observed samples
(rows/columns of the gaps deleted).  ``Sigma_OO`` is neither circulant nor
Toeplitz, and at large ``N`` it cannot even be stored: with an 80% duty cycle
it reaches 5.5 GB at ``N = 2^15`` and ~350 GB at ``N = 2^18``.

Conjugate gradients sidesteps the matrix entirely.  It solves
``Sigma_OO u = b`` given only the *action* of ``Sigma_OO``, which costs three
steps, two of them FFTs:

    embed with zeros  ->  multiply by the circulant eigenvalues  ->  restrict.

The identity ``R Sigma R^T = Sigma_OO`` makes this exact, not approximate:
the zero-fill is the embedding ``R^T``, never a statement about the data.
Convergence is governed by the conditioning *after* preconditioning with the
gap-free (Whittle) covariance --- the same FFT pair, dividing by the
eigenvalues instead --- which removes the dynamic range of the PSD and leaves
only the geometry of the gaps.  Per iteration: four length-``N`` FFTs.

Nothing here assumes anything about the noise model beyond stationarity ---
in particular, none of the two-component (TM/OMS) structure that the
closed-form routes in :mod:`gaplike.likelihood` exploit.  This is the general
route: it survives spline-parametrized spectra, extra components, anything
that defeats a simultaneous diagonalization.

The log-determinant is *not* provided here: pair the quadratic form with the
closed-form determinant of :class:`gaplike.likelihood.TimeDomainExact` when
there are exactly two components (as in the companion paper), or with
stochastic Lanczos quadrature in general.

Conventions match :func:`gaplike.covariance.restricted_autocov`: the
circulant eigenvalues are ``eig = S2 / (2 dt)`` with ``S2`` the two-sided PSD
on the full DFT grid, so ``gamma = Re ifft(eig)`` is the autocovariance.
"""
from __future__ import annotations

import inspect

import numpy as np
from scipy.sparse.linalg import LinearOperator, cg as _scipy_cg

from .psd import as_two_sided

__all__ = [
    "circulant_eigenvalues", "sigma_oo", "whittle_preconditioner",
    "solve", "quad_form", "dense_restricted", "RestrictedCG",
]

# scipy renamed cg(tol=...) to cg(rtol=...) in 1.12
_TOL_KW = "rtol" if "rtol" in inspect.signature(_scipy_cg).parameters else "tol"


def circulant_eigenvalues(components, n, dt):
    """DFT eigenvalues of the stationary circulant covariance for a sum of
    PSD components (callables of ``f`` or two-sided grids): ``sum_k S2_k/(2 dt)``."""
    return sum(as_two_sided(c, n, dt) for c in components) / (2.0 * dt)


def _obs_array(obs_or_mask):
    """Boolean or float arrays are masks/gates over the full grid (strictly
    positive = observed); integer arrays are observed-sample indices."""
    a = np.asarray(obs_or_mask)
    if a.dtype == bool or np.issubdtype(a.dtype, np.floating):
        return np.flatnonzero(a > 0)
    return a.astype(np.intp)


def sigma_oo(eig, obs):
    """``LinearOperator`` applying ``Sigma_OO = R Sigma R^T`` matrix-free:
    zero-embed, multiply by ``eig`` in Fourier space, restrict.  Two FFTs."""
    eig = np.asarray(eig, float)
    n = eig.size
    obs = _obs_array(obs)
    m = obs.size

    eig_r = eig[: n // 2 + 1]           # real-input half spectrum

    def mv(v):
        z = np.zeros(n)
        z[obs] = v
        return np.fft.irfft(eig_r * np.fft.rfft(z), n)[obs]

    return LinearOperator((m, m), matvec=mv, dtype=float)


def whittle_preconditioner(eig, obs, floor_rel=1e-6):
    """Inverse of the gap-free (Whittle) covariance applied by the same FFT
    pair, as a preconditioner.  ``floor_rel`` floors the eigenvalues at a
    fraction of their maximum, taming transfer-function nulls (e.g. the TDI2
    zeros) that the restricted operator does not share."""
    eig = np.asarray(eig, float)
    ef = np.maximum(eig, floor_rel * eig.max())
    n = eig.size
    obs = _obs_array(obs)
    m = obs.size
    ef_r = ef[: n // 2 + 1]             # real-input half spectrum

    def mv(v):
        z = np.zeros(n)
        z[obs] = v
        return np.fft.irfft(np.fft.rfft(z) / ef_r, n)[obs]

    return LinearOperator((m, m), matvec=mv, dtype=float)


def solve(eig, obs, b, rtol=1e-8, x0=None, maxiter=None,
          floor_rel=1e-6, precondition=True):
    """Solve ``Sigma_OO u = b`` by (preconditioned) conjugate gradients.

    Parameters
    ----------
    eig : (n,) circulant eigenvalues of the full covariance
    obs : observed-sample indices, or a boolean/gate mask over the full grid
    b : (m,) right-hand side on the observed samples
    rtol : relative residual tolerance ``||Sigma_OO u - b|| / ||b||``
    x0 : optional warm start (e.g. the previously accepted solution in MCMC)

    Returns ``(u, iterations)``.
    """
    obs = _obs_array(obs)
    A = sigma_oo(eig, obs)
    M = whittle_preconditioner(eig, obs, floor_rel) if precondition else None
    it = [0]
    kw = {_TOL_KW: rtol, "atol": 0.0, "M": M, "x0": x0,
          "callback": lambda xk: it.__setitem__(0, it[0] + 1)}
    if maxiter is not None:
        kw["maxiter"] = maxiter
    u, info = _scipy_cg(A, b, **kw)
    if info > 0:
        raise RuntimeError(f"CG did not converge in {info} iterations "
                           f"(rtol={rtol}); raise maxiter or loosen rtol")
    return u, it[0]


def quad_form(eig, obs, b, **kw):
    """``b^T Sigma_OO^{-1} b`` by CG.  ``b`` may be ``(m,)`` or multichannel
    ``(nch, m)`` (channels independent, identical covariance --- the forms are
    summed).  Returns ``(value, [iterations per solve])``."""
    b = np.atleast_2d(np.asarray(b, float))
    q, iters = 0.0, []
    for row in b:
        u, ni = solve(eig, obs, row, **kw)
        q += float(row @ u)
        iters.append(ni)
    return q, iters


def dense_restricted(eig, obs, block_bytes=2e8):
    """Dense ``Sigma_OO`` built from the autocovariance, in row blocks so the
    integer lag array never rivals the matrix itself.  Reference/benchmark
    only: this is exactly the object the CG route exists to avoid."""
    eig = np.asarray(eig, float)
    n = eig.size
    obs = _obs_array(obs)
    m = obs.size
    gamma = np.real(np.fft.ifft(eig))
    S = np.empty((m, m))
    blk = max(1, int(block_bytes / 8 // max(m, 1)))
    for i0 in range(0, m, blk):
        i1 = min(i0 + blk, m)
        lag = (obs[i0:i1, None] - obs[None, :]) % n
        S[i0:i1] = gamma[lag]
    return S


class RestrictedCG:
    """Quadratic forms in ``Sigma_OO(lam)^{-1}`` for a component-parametrized
    spectrum ``Sigma(lam) = sum_k 10^(lam_k) C_k``, evaluated matrix-free.

    The companion of :class:`gaplike.likelihood.TimeDomainExact`: same model,
    same parametrization, any number of components, no O(m^3) setup and no
    O(m^2) storage --- but no determinant (see the module docstring).

    Parameters
    ----------
    mask : observed-sample mask/gate (strictly positive = observed), length n
    components : PSD components (callables or two-sided grids)
    dt : cadence [s]
    """

    def __init__(self, mask, components, dt, rtol=1e-8, floor_rel=1e-6):
        mask = np.asarray(mask)
        self.n = mask.size
        self.obs = np.flatnonzero(mask > 0)
        self.m = self.obs.size
        self.dt = float(dt)
        self.eig_k = [as_two_sided(c, self.n, dt) / (2.0 * dt)
                      for c in components]
        self.rtol = float(rtol)
        self.floor_rel = float(floor_rel)

    def eigenvalues(self, lam):
        lam = np.atleast_1d(np.asarray(lam, float))
        return sum(10.0**lam[k] * e for k, e in enumerate(self.eig_k))

    def solve(self, lam, b, x0=None, maxiter=None):
        return solve(self.eigenvalues(lam), self.obs, b, rtol=self.rtol,
                     x0=x0, maxiter=maxiter, floor_rel=self.floor_rel)

    def quad_form(self, lam, resid_obs, maxiter=None):
        """``resid_obs``: (nch, m) residuals on the observed samples.
        Returns ``(sum_ch r^T Sigma_OO^{-1} r, [iterations per solve])``."""
        return quad_form(self.eigenvalues(lam), self.obs, resid_obs,
                         rtol=self.rtol, maxiter=maxiter,
                         floor_rel=self.floor_rel)
