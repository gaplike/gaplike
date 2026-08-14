"""Machine-precision checks of every likelihood tier against direct dense
linear algebra at small N."""
import numpy as np
import pytest

from gaplike import covariance as cov
from gaplike import gaps, psd, simulate
from gaplike.likelihood import DiagonalLikelihood, FullCovariance, TimeDomainExact

N, DT = 128, 5.0
F_LO, F_HI = 1.0 / (N * DT) * 2.5, 0.4 / DT / 2


def _components(n=N, dt=DT):
    S1 = lambda f: 1.0 / (1.0 + (f / 0.01)**2)          # red component
    S2 = lambda f: 0.5 * np.ones_like(np.asarray(f))    # white component
    return [psd.two_sided_grid(S1, n, dt), psd.two_sided_grid(S2, n, dt)]


def _resid(nb, nch=2, seed=1, complex_=True):
    rng = np.random.default_rng(seed)
    if complex_:
        return rng.standard_normal((nch, nb)) + 1j * rng.standard_normal((nch, nb))
    return rng.standard_normal((nch, nb))


# --------------------------------------------------------------------------
# FullCovariance vs direct dense complex-normal
# --------------------------------------------------------------------------
def test_fullcov_vs_dense_full_rank():
    comps = _components()
    w = gaps.segment_window(N, 0.2)                      # no gaps: full rank
    idx = cov.band_indices(N, DT, F_LO, F_HI)
    mats = [cov.full_covariance(w, g, idx) for g in comps]

    L = FullCovariance(mats, rtol_rank=1e-13)
    r = _resid(idx.size)

    for lam in [(0.0, 0.0), (0.13, -0.21)]:
        Sig = 10.0**(lam[0]) * mats[0] + 10.0**(lam[1]) * mats[1]
        sgn, logdet = np.linalg.slogdet(Sig)
        assert sgn > 0
        Si = np.linalg.inv(Sig)
        nch = r.shape[0]
        ll_direct = float(
            -sum(np.real(r[c].conj() @ (Si @ r[c])) for c in range(nch))
            - nch * (logdet + idx.size * np.log(np.pi)))
        assert np.isclose(L.loglike(r, lam), ll_direct, rtol=1e-9), lam


def test_fullcov_rank_cut_insensitive():
    """With gaps (rank-deficient covariance), likelihood DIFFERENCES are
    insensitive to the rank tolerance across 1e-6 .. 1e-10."""
    comps = _components()
    gate = gaps.gate_from_mask(gaps.periodic_mask(N, 12, 32), DT)
    w = gaps.effective_window(gate, gaps.segment_window(N, 0.2))
    idx = cov.band_indices(N, DT, F_LO, F_HI)
    mats = [cov.full_covariance(w, g, idx) for g in comps]
    r = _resid(idx.size, seed=4)
    lamA, lamB = (0.0, 0.0), (0.11, -0.07)

    deltas = []
    for tol in (1e-6, 1e-8, 1e-10):
        L = FullCovariance(mats, rtol_rank=tol)
        deltas.append(L.loglike(r, lamA) - L.loglike(r, lamB))
    assert np.allclose(deltas, deltas[0], rtol=1e-6)


def test_fullcov_from_window_transform_consistency():
    """from_window + transform reproduces the direct matrix route."""
    comps = _components()
    w = gaps.segment_window(N, 0.2)
    L = FullCovariance.from_window(w, comps, DT, F_LO, F_HI, rtol_rank=1e-13)
    rng = np.random.default_rng(7)
    x = rng.standard_normal((2, N))
    y = L.transform(x)
    assert y.shape == (2, L.n_band)
    # the same likelihood through matrices with explicit s_ref
    idx = cov.band_indices(N, DT, F_LO, F_HI)
    mats = [cov.full_covariance(w, g, idx) for g in comps]
    L2 = FullCovariance([m / L.s_ref for m in mats], rtol_rank=1e-13)
    lam = (0.05, -0.02)
    assert np.isclose(L.loglike(y, lam), L2.loglike(y, lam), rtol=1e-10)
    assert np.isclose(L.loglike_td(x, lam), L.loglike(y, lam), rtol=1e-12)


# --------------------------------------------------------------------------
# DiagonalLikelihood
# --------------------------------------------------------------------------
def test_diagonal_formula():
    v1 = np.linspace(1.0, 2.0, 11)
    v2 = np.linspace(0.5, 0.1, 11)
    L = DiagonalLikelihood([v1, v2])
    r = _resid(11, seed=2)
    lam = (0.1, -0.3)
    var = 10.0**0.1 * v1 + 10.0**-0.3 * v2
    ll = -2 * np.sum(np.log(var)) - np.sum(np.abs(r)**2 / var[None, :])
    assert np.isclose(L.loglike(r, lam), ll, rtol=1e-12)


def test_convolved_diag_matches_full_diag():
    """Convolved-diagonal variances equal the diagonal of the dense windowed
    covariance (same normalization)."""
    comps = _components()
    gate = gaps.gate_from_mask(gaps.periodic_mask(N, 8, 32), DT)
    w = gaps.effective_window(gate, gaps.segment_window(N, 0.2))
    idx = cov.band_indices(N, DT, F_LO, F_HI)
    Ld = DiagonalLikelihood.convolved(w, comps, DT, F_LO, F_HI, s_ref=1.0)
    mats = [cov.full_covariance(w, g, idx) for g in comps]
    for vk, mk in zip(Ld.v, mats):
        assert np.allclose(vk, np.real(np.diag(mk)), rtol=1e-9)


def test_whittle_scaled():
    comps = _components()
    gate = gaps.gate_from_mask(gaps.periodic_mask(N, 8, 32), DT)
    w = gaps.effective_window(gate, gaps.segment_window(N, 0.2))
    L0 = DiagonalLikelihood.whittle(w, comps, DT, F_LO, F_HI, s_ref=1.0)
    L1 = DiagonalLikelihood.whittle(w, comps, DT, F_LO, F_HI,
                                    scale_by_window_power=True, s_ref=1.0)
    w2 = gaps.window_power(w)
    for v0, v1 in zip(L0.v, L1.v):
        assert np.allclose(v1, w2 * v0, rtol=1e-12)


# --------------------------------------------------------------------------
# TimeDomainExact vs direct dense Gaussian
# --------------------------------------------------------------------------
def test_td_exact_vs_dense():
    comps = _components()
    mask = gaps.periodic_mask(N, 10, 40)
    L = TimeDomainExact(mask, comps, DT)
    obs = L.obs
    r = _resid(L.m, nch=2, seed=3, complex_=False)

    C0 = cov.restricted_autocov(comps[0], obs, DT)
    C1 = cov.restricted_autocov(comps[1], obs, DT)
    for lam in [(0.0, 0.0), (0.2, -0.15)]:
        Sig = 10.0**(lam[0]) * C0 + 10.0**(lam[1]) * C1
        sgn, logdet = np.linalg.slogdet(Sig)
        assert sgn > 0
        Si = np.linalg.inv(Sig)
        nch = r.shape[0]
        ll_direct = float(-0.5 * (sum(r[c] @ (Si @ r[c]) for c in range(nch))
                                  + nch * logdet))
        assert np.isclose(L.loglike(r, lam), ll_direct, rtol=1e-9), lam


def test_td_score_matches_numerical_gradient():
    comps = _components()
    mask = gaps.periodic_mask(N, 10, 40)
    L = TimeDomainExact(mask, comps, DT)
    r = _resid(L.m, nch=2, seed=5, complex_=False)
    lam0 = (0.03, -0.06)
    s = L.score_noise(r, lam=lam0)
    eps = 1e-6
    for a in range(2):
        lp = list(lam0); lp[a] += eps
        lm = list(lam0); lm[a] -= eps
        num = (L.loglike(r, lp) - L.loglike(r, lm)) / (2 * eps)
        assert np.isclose(s[a], num, rtol=1e-5), a


def test_td_fisher_noise_positive_and_symmetric():
    comps = _components()
    mask = gaps.periodic_mask(N, 10, 40)
    L = TimeDomainExact(mask, comps, DT)
    G = L.fisher_noise(nch=2)
    assert np.allclose(G, G.T)
    assert np.all(np.linalg.eigvalsh(G) > 0)


def test_td_transform():
    comps = _components()
    mask = gaps.periodic_mask(N, 10, 40)
    L = TimeDomainExact(mask, comps, DT)
    x = np.arange(2 * N, dtype=float).reshape(2, N)
    assert np.array_equal(L.transform(x), x[:, L.obs])


# --------------------------------------------------------------------------
# statistical sanity: simulated noise is unit-consistent with every tier
# --------------------------------------------------------------------------
def test_simulated_noise_consistent_units():
    """Whittle-level check: mean periodogram of simulated noise over many
    realizations matches the PSD within MC error, and the TD likelihood
    prefers lam = 0 over strongly wrong values."""
    n, dt = 512, 5.0
    comps = _components(n, dt)
    # one-sided grid = first n//2+1 entries of the two-sided grid (fftfreq
    # ordering: [0, f_1, ..., f_{n/2-1}, +-f_nyq, ...]; S evaluated at |f|)
    S_1s = (comps[0] + comps[1])[:n // 2 + 1]
    rng = np.random.default_rng(11)
    x = simulate.noise_td(S_1s, dt, rng, nch=400)
    y = cov.transform(x, np.ones(n), np.arange(1, n // 2), dt)
    P = np.mean(np.abs(y)**2, axis=0)
    # per-bin MC error ~ 1/sqrt(400) = 5%; 6 sigma tolerance
    assert np.allclose(P, S_1s[1:n // 2], rtol=0.30)
    assert np.isclose(np.median(P / S_1s[1:n // 2]), 1.0, atol=0.03)

    mask = gaps.periodic_mask(n, 20, 100)
    L = TimeDomainExact(mask, comps, dt)
    r = L.transform(x[:2])
    ll0 = L.loglike(r, (0.0, 0.0))
    assert ll0 > L.loglike(r, (0.4, 0.4))
    assert ll0 > L.loglike(r, (-0.4, -0.4))
