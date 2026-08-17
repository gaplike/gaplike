"""Stochastic Lanczos quadrature (SLQ) log-determinants for gapped data.

The exact time-domain likelihood needs, besides the quadratic form that
:mod:`gaplike.cg` provides matrix-free, the log-determinant

    log|Sigma_OO(theta)|,

which changes with every noise-parameter update.  With exactly two PSD
components the closed form of :class:`gaplike.likelihood.TimeDomainExact`
applies; for anything richer --- in particular the ratio-spline spectra this
module also provides --- there is no closed form, and the dense route costs
O(m^3).  This module supplies the general, matrix-free alternative:

*   ``logdet_slq``            Hutchinson + Lanczos quadrature estimate of
                              ``tr log Sigma_OO`` (Ubaru, Chen & Saad 2017).
*   ``logdet_diff_slq``       the quantity an MCMC accept ratio actually
                              needs, ``log|Sigma_OO(theta')| -
                              log|Sigma_OO(theta)|``, estimated with SHARED
                              probe vectors so that the estimator sees only
                              the (small) operator difference, not the two
                              (large) absolute determinants.
*   ``logdet_dense``          Cholesky ground truth, for validation at small
                              ``m`` (this is the object everything else
                              exists to avoid).
*   ``circulant_logdet``      the gap-free (Whittle) determinant in closed
                              form --- context lines and sanity checks.
*   ``ratio_spline``          a flexible PSD component
                              ``S_ref(f) * exp(sum_k c_k B_k(log10 f))``
                              that plugs into ``circulant_eigenvalues`` like
                              any other component.

Probes are Rademacher: for a fixed symmetric operator ``D`` the estimator
variance is ``(2/M) * (||D||_F^2 - sum_i D_ii^2)`` --- the diagonal does not
contribute.  For the *difference* operator ``log Sigma(2) - log Sigma(1)``
under a smooth spectral perturbation most of the energy sits on the
diagonal, which is why shared-probe differences can be dramatically better
than differencing two absolute estimates; measuring that gain on realistic
spectra is the point of ``paper/fig_det_flexible.py``.

Everything is deterministic given ``rng``; no state is kept between calls.
"""
from __future__ import annotations

import numpy as np
import scipy.fft as sfft
from scipy.interpolate import BSpline
from scipy.linalg import cho_factor, eigh_tridiagonal

from .cg import _obs_array, dense_restricted, sigma_oo

__all__ = [
    "rademacher", "lanczos_tridiag", "quad_log", "logdet_slq",
    "logdet_diff_slq", "logdet_dense", "circulant_logdet", "ratio_spline",
    "ComplementFactor", "logdet_complement",
    "spline_knots",
]


def rademacher(m, n_probes, rng):
    """(n_probes, m) array of +-1 probe vectors."""
    return rng.integers(0, 2, size=(n_probes, m)).astype(float) * 2.0 - 1.0


def lanczos_tridiag(matvec, v0, k, reorth=True):
    """``k`` Lanczos steps from ``v0`` with full reorthogonalisation.

    Returns ``(alpha, beta, V)`` --- diagonal, off-diagonal and the basis
    actually built (early breakdown truncates all three).  Memory is
    ``m * k`` doubles; at the ``m``, ``k`` of this study that is trivial and
    full reorthogonalisation keeps the Ritz values honest.
    """
    v = np.asarray(v0, float)
    v = v / np.linalg.norm(v)
    V = np.empty((k, v.size))
    alpha = np.empty(k)
    beta = np.empty(k)          # beta[j] links step j to j+1
    v_prev = np.zeros_like(v)
    b_prev = 0.0
    for j in range(k):
        V[j] = v
        w = matvec(v)
        a = float(v @ w)
        alpha[j] = a
        w = w - a * v - b_prev * v_prev
        if reorth:              # twice is enough (Parlett)
            w -= V[: j + 1].T @ (V[: j + 1] @ w)
            w -= V[: j + 1].T @ (V[: j + 1] @ w)
        b = float(np.linalg.norm(w))
        if b == 0.0 or not np.isfinite(b):
            return alpha[: j + 1], beta[:j], V[: j + 1]
        beta[j] = b
        v_prev, b_prev, v = v, b, w / b
    return alpha, beta[: k - 1], V


def quad_log(matvec, z, k, eig_floor=0.0):
    """Lanczos-quadrature estimate of ``z^T log(A) z`` for SPD ``A``.

    ``eig_floor > 0`` clips non-positive Ritz values (finite-precision guards
    near-singular operators); the number clipped is returned for monitoring.
    """
    z = np.asarray(z, float)
    nz2 = float(z @ z)
    alpha, beta, _ = lanczos_tridiag(matvec, z, k)
    theta, U = eigh_tridiagonal(alpha, beta)
    clipped = int(np.sum(theta <= eig_floor))
    theta = np.maximum(theta, max(eig_floor, np.finfo(float).tiny))
    tau2 = U[0] ** 2
    return nz2 * float(tau2 @ np.log(theta)), clipped


# ------------------------------------------------------------- batched path
#
# A CG solve is a serial chain of matvecs, but SLQ's probes are independent:
# stacking them into a (B, n) array turns every Lanczos step into ONE batched
# real FFT (multithreaded via scipy.fft's ``workers``), and the
# reorthogonalisations into batched matmuls.  Same mathematics, same probes,
# same answers to floating-point order --- just executed together.

def _batched_restricted_matvec(eig_r, obs, n, workers):
    """(B, m) -> (B, m) action of Sigma_OO on a block of vectors: one batched
    rfft/irfft pair.  Keeps a reusable zero-embedding workspace per block
    size."""
    buf = {}

    def mv(V):
        B = V.shape[0]
        Z = buf.get(B)
        if Z is None:
            Z = buf.setdefault(B, np.zeros((B, n)))
        Z[:] = 0.0
        Z[:, obs] = V
        F = sfft.rfft(Z, axis=1, workers=workers)
        F *= eig_r
        return sfft.irfft(F, n=n, axis=1, workers=workers)[:, obs]

    return mv


def _lanczos_batched(mv, Z0, k, reorth="full"):
    """Batched Lanczos.  Returns ``(alpha (B,k), beta (B,k-1),
    start_norms (B,))``.

    ``reorth="full"`` stores the bases (``B * k * m`` doubles) and
    reorthogonalises twice per step --- numerically pristine, but the
    repeated basis reads are memory-bandwidth bound once the batch no
    longer fits in cache.  ``reorth="none"`` is the standard SLQ practice
    (Ubaru, Chen & Saad 2017): plain three-term recurrence, O(B*m) memory,
    FFT-dominated cost; the orthogonality loss duplicates Ritz values,
    which the log quadrature tolerates --- VALIDATED against dense truth in
    the test suite rather than assumed."""
    B, m = Z0.shape
    nrm0 = np.linalg.norm(Z0, axis=1)
    v = Z0 / nrm0[:, None]
    store = reorth == "full"
    V = np.empty((B, k, m)) if store else None
    alpha = np.empty((B, k))
    beta = np.empty((B, max(k - 1, 1)))
    v_prev = np.zeros_like(v)
    b_prev = np.zeros(B)
    for j in range(k):
        if store:
            V[:, j] = v
        w = mv(v)
        a = np.einsum("bm,bm->b", v, w)
        alpha[:, j] = a
        w = w - a[:, None] * v - b_prev[:, None] * v_prev
        if store:
            Vj = V[:, : j + 1]
            for _ in range(2):
                c = np.einsum("bjm,bm->bj", Vj, w)
                w -= np.einsum("bj,bjm->bm", c, Vj)
        if j == k - 1:
            break
        b = np.linalg.norm(w, axis=1)
        if np.any(b <= 1e3 * np.finfo(float).tiny):
            raise RuntimeError("Lanczos breakdown in batched mode; "
                               "rerun with batched=False for this operator")
        beta[:, j] = b
        v_prev, b_prev = v, b
        v = w / b[:, None]
    return alpha, beta[:, : k - 1], nrm0


def _slq_block(eig, obs, Z, k, eig_floor, workers, max_bytes, reorth="full"):
    """Per-probe ``z^T log(Sigma_OO) z`` for a block of probes, chunked so the
    stored Lanczos bases stay under ``max_bytes`` (no storage when
    ``reorth="none"`` --- the whole block runs at once)."""
    eig = np.asarray(eig, float)
    n = eig.size
    obs = _obs_array(obs)
    m = obs.size
    eig_r = eig[: n // 2 + 1]
    M = Z.shape[0]
    if reorth == "full":
        chunk = max(1, min(M, int(max_bytes / (k * m * 8))))
    else:
        chunk = M
    mv = _batched_restricted_matvec(eig_r, obs, n, workers)
    per = np.empty(M)
    clipped = 0
    for i0 in range(0, M, chunk):
        Zc = Z[i0:i0 + chunk]
        alpha, beta, nrm0 = _lanczos_batched(mv, Zc, k, reorth)
        for b in range(Zc.shape[0]):
            theta, U = eigh_tridiagonal(alpha[b], beta[b])
            clipped += int(np.sum(theta <= eig_floor))
            theta = np.maximum(theta, max(eig_floor, np.finfo(float).tiny))
            per[i0 + b] = nrm0[b] ** 2 * float(U[0] ** 2 @ np.log(theta))
    return per, clipped


def logdet_slq(eig, obs, n_probes, k, rng, eig_floor=0.0, batched=True,
               workers=-1, max_bytes=1.0e9, reorth="none"):
    """SLQ estimate of ``log|Sigma_OO|``.

    Parameters
    ----------
    eig : (n,) circulant eigenvalues of the full covariance
        (:func:`gaplike.cg.circulant_eigenvalues`)
    obs : observed-sample indices, or mask/gate over the full grid
    n_probes, k : Rademacher probes and Lanczos steps per probe
    rng : ``numpy.random.Generator``

    Returns a dict: ``est`` (the estimate), ``per_probe`` (individual probe
    estimates, for empirical scatter), ``matvecs``, ``clipped``.

    ``batched=True`` (default) executes all probes together --- one batched
    real FFT per Lanczos step, multithreaded via ``workers`` --- identical
    probes, identical mathematics, floating-point-order differences only.
    ``max_bytes`` caps the stored Lanczos bases (probes are chunked).
    """
    obs = _obs_array(obs)
    Z = rademacher(obs.size, n_probes, rng)
    if batched:
        per, clipped = _slq_block(eig, obs, Z, k, eig_floor, workers,
                                  max_bytes, reorth)
    else:
        A = sigma_oo(eig, obs)
        per = np.empty(n_probes)
        clipped = 0
        for i, z in enumerate(Z):
            per[i], c = quad_log(A.matvec, z, k, eig_floor)
            clipped += c
    return dict(est=float(per.mean()), per_probe=per,
                matvecs=n_probes * k, clipped=clipped)


def logdet_diff_slq(eig1, eig2, obs, n_probes, k, rng, shared=True,
                    eig_floor=0.0, batched=True, workers=-1,
                    max_bytes=1.0e9, reorth="none"):
    """``log|Sigma_OO(eig2)| - log|Sigma_OO(eig1)|`` by SLQ.

    ``shared=True`` uses the same Rademacher probes on both operators (the
    estimator of the *difference* operator); ``shared=False`` draws fresh
    probes for each --- the naive differencing of two absolute estimates,
    kept for comparison.  Same return structure as :func:`logdet_slq` with
    ``per_probe`` holding per-probe differences; ``batched`` as there.
    """
    obs = _obs_array(obs)
    Z1 = rademacher(obs.size, n_probes, rng)
    Z2 = Z1 if shared else rademacher(obs.size, n_probes, rng)
    if batched:
        per2, c2 = _slq_block(eig2, obs, Z2, k, eig_floor, workers,
                              max_bytes, reorth)
        per1, c1 = _slq_block(eig1, obs, Z1, k, eig_floor, workers,
                              max_bytes, reorth)
        per = per2 - per1
        clipped = c1 + c2
    else:
        A1 = sigma_oo(eig1, obs)
        A2 = sigma_oo(eig2, obs)
        per = np.empty(n_probes)
        clipped = 0
        for i in range(n_probes):
            q2, c2 = quad_log(A2.matvec, Z2[i], k, eig_floor)
            q1, c1 = quad_log(A1.matvec, Z1[i], k, eig_floor)
            per[i] = q2 - q1
            clipped += c1 + c2
    return dict(est=float(per.mean()), per_probe=per,
                matvecs=2 * n_probes * k, clipped=clipped)


def logdet_dense(eig, obs):
    """Ground truth ``log|Sigma_OO|`` by dense Cholesky (O(m^3); small ``m``
    only).  Returns ``(logdet, seconds_spent_in_factorisation)``."""
    import time
    S = dense_restricted(eig, obs)
    t0 = time.perf_counter()
    cf, _ = cho_factor(S, lower=True, overwrite_a=True, check_finite=False)
    dt = time.perf_counter() - t0
    return 2.0 * float(np.sum(np.log(np.diag(cf)))), dt


def circulant_logdet(eig, skip_nonpositive=False):
    """Closed-form ``log|Sigma|`` of the FULL (gap-free) circulant --- the
    Whittle determinant.  Reference/context only: restriction to the
    observed samples has no closed form (that is the whole point).

    NB the library's grid convention zeroes the DC bin (mean removal), so
    eigenvalue grids built from callables contain an exact zero and the
    unrestricted determinant is ``-inf`` by design; ``skip_nonpositive=True``
    sums the strictly positive (in-band) bins instead, which is the Whittle
    practice."""
    eig = np.asarray(eig, float)
    if skip_nonpositive:
        eig = eig[eig > 0]
    return float(np.sum(np.log(eig)))


# ---------------------------------------------------------------- flexible PSD

def spline_knots(f_lo, f_hi, n_coeff, degree=3):
    """Clamped knot vector in ``log10 f`` for ``n_coeff`` B-spline basis
    functions of the given degree spanning ``[f_lo, f_hi]``."""
    lo, hi = np.log10(f_lo), np.log10(f_hi)
    interior = np.linspace(lo, hi, n_coeff - degree + 1)
    return np.concatenate([[lo] * degree, interior, [hi] * degree])


def ratio_spline(S_ref, knots, coeffs, degree=3):
    """Flexible PSD component ``S_ref(f) * exp(spline(log10 f))``.

    ``S_ref`` is any PSD callable of ``f``; ``coeffs`` are the log-ratio
    B-spline coefficients on ``knots`` (from :func:`spline_knots`).  The
    spline argument is clipped to the knot span, so the ratio extrapolates
    flat --- deliberately: tails behave like the reference.  ``coeffs = 0``
    reproduces ``S_ref`` exactly.  The return is a callable component
    accepted anywhere ``gaplike`` takes PSD components.
    """
    coeffs = np.asarray(coeffs, float)
    spl = BSpline(np.asarray(knots, float), coeffs, degree, extrapolate=False)
    lo, hi = knots[degree], knots[-degree - 1]

    def component(f):
        f = np.asarray(f, float)
        x = np.clip(np.log10(np.maximum(np.abs(f), 1e-300)), lo, hi)
        return S_ref(f) * np.exp(spl(x))

    return component


# ------------------------------------------------------- complement identity
#
# For SPD Sigma partitioned over observed (O) and gap (G) samples, Schur
# complement algebra gives the EXACT identities
#
#     log|Sigma_OO|  =  log|Sigma| + log|(Sigma^-1)_GG|
#     Sigma_OO^-1    =  A_OO - A_OG A_GG^-1 A_GO,        A = Sigma^-1.
#
# In the circulant model both ingredients are cheap: log|Sigma| is the
# closed-form Whittle sum, and Sigma^-1 is the circulant of the INVERSE
# spectrum, so A_GG is just ``dense_restricted`` on the gap samples --- a
# g x g problem where g is the number of MISSING samples.  At LISA-like
# duty (~80% observed) that is (m/g)^3 = 64x fewer flops than dense on the
# observed side, exact, with no probes and no Lanczos; and the SAME g x g
# Cholesky then turns every quadratic form into two FFTs and one
# triangular solve --- no conjugate gradients.
#
# The library's zeroed bins (DC by convention) make Sigma singular, so the
# identity is evaluated on an eps-floored spectrum; the +-log(eps) parts
# cancel between the two terms and the residual is an O(eps) model
# perturbation --- ``eps_rel`` is swept in the tests against dense truth.
# This is the classical Gaussian missing-data algebra (the marginalised
# cousin of gap data augmentation); the point here is its deployment
# inside the restricted-circulant likelihood.

class ComplementFactor:
    """One g x g Cholesky of ``(Sigma_c^-1)_GG`` and everything it unlocks:
    the exact log-determinant (``.logdet``) and exact quadratic forms
    (:meth:`quad_form`) of ``Sigma_OO``.

    The library's grid convention zeroes the DC bin, making the full
    circulant singular; the identity is therefore evaluated on a model with
    the DC eigenvalue set to ``dc_scale * max(eig)``, and the exact
    rank-one downdate (matrix determinant lemma / Sherman--Morrison along
    the constant mode) removes it again EXACTLY --- the floor is a
    conditioning choice, not an approximation.  Any zeroed bin other than
    DC is not handled and raises.

    Parameters
    ----------
    eig : (n,) circulant eigenvalues of the full covariance
    obs : observed-sample indices, or mask/gate over the full grid
    dc_scale : DC floor as a fraction of the largest eigenvalue
    """

    def __init__(self, eig, obs, dc_scale=1e-3):
        eig = np.asarray(eig, float)
        n = eig.size
        obs = _obs_array(obs)
        gap = np.setdiff1d(np.arange(n), obs)
        floored = np.flatnonzero(eig <= 0)
        if floored.size and not np.array_equal(floored, [0]):
            raise ValueError("only a zeroed DC bin is supported; "
                             f"non-positive bins at {floored}")
        c_val = dc_scale * eig.max()
        ef = eig.copy()
        if floored.size:
            ef[0] = c_val
        self.n, self.obs, self.gap = n, obs, gap
        self.ef = ef
        self.ef_r = ef[: n // 2 + 1]
        logdet_c = float(np.sum(np.log(ef)))
        if gap.size:
            gg = dense_restricted(1.0 / ef, gap)
            self._cf = cho_factor(gg, lower=True, overwrite_a=True,
                                  check_finite=False)
            logdet_c += 2.0 * float(np.sum(np.log(np.diag(self._cf[0]))))
        else:
            self._cf = None
        # exact removal of the DC floor: Sigma_OO = Sigma_c,OO - (c/n) 1 1^T
        if floored.size:
            ones = np.ones(obs.size)
            v = self._apply_inv_c(ones)
            alpha = c_val / n
            q_c = float(ones @ v)
            self._v = v
            self._denom = 1.0 / alpha - q_c          # > 0 iff Sigma_OO SPD
            self.logdet = logdet_c + np.log1p(-alpha * q_c)
        else:
            self._v = None
            self.logdet = logdet_c

    def _apply_inv_c(self, d):
        """``Sigma_c,OO^{-1} d`` as a vector, via the block-inverse formula:
        two rfft/irfft applications of the inverse circulant and one solve
        against the cached g x g factor."""
        from scipy.linalg import cho_solve
        z = np.zeros(self.n)
        z[self.obs] = d
        u = np.fft.irfft(np.fft.rfft(z) / self.ef_r, self.n)
        if self._cf is None:
            return u[self.obs]
        t = cho_solve(self._cf, u[self.gap], check_finite=False)
        z2 = np.zeros(self.n)
        z2[self.gap] = t
        y = np.fft.irfft(np.fft.rfft(z2) / self.ef_r, self.n)
        return u[self.obs] - y[self.obs]

    def quad_form(self, d):
        """Exact ``d^T Sigma_OO^{-1} d`` (``d``: (m,) or (nch, m)) from the
        cached factor; the DC downdate is applied exactly."""
        d = np.atleast_2d(np.asarray(d, float))
        q = 0.0
        for row in d:
            w = self._apply_inv_c(row)
            qi = float(row @ w)
            if self._v is not None:
                s = float(self._v @ row)
                qi += s * s / self._denom
            q += qi
        return q


def logdet_complement(eig, obs, dc_scale=1e-3):
    """Exact ``log|Sigma_OO|`` via the complement identity (see
    :class:`ComplementFactor`); cost is one g x g Cholesky on the GAP
    samples."""
    return ComplementFactor(eig, obs, dc_scale).logdet
