"""gaplike.cg: matrix-free CG against the dense restricted covariance."""
import numpy as np
import pytest
from scipy.linalg import cho_factor, cho_solve

from gaplike import cg, gaps, psd
from gaplike.likelihood import TimeDomainExact

DT = 15.0
RNG = np.random.default_rng(7)


def _setup(n=1024, lam=(0.0, 0.0)):
    comps = psd.lisa_tdi2_ae()
    grids = [psd.two_sided_grid(comps["tm"], n, DT),
             psd.two_sided_grid(comps["oms"], n, DT)]
    eig = (10.0**lam[0] * grids[0] + 10.0**lam[1] * grids[1]) / (2.0 * DT)
    mask = gaps.periodic_mask(n, 10, 50)          # scenario-C comb, 80% duty
    obs = np.flatnonzero(np.asarray(mask) > 0)
    return eig, obs, grids


def test_operator_matches_dense_columns():
    eig, obs, _ = _setup(n=256)
    A = cg.sigma_oo(eig, obs)
    S = cg.dense_restricted(eig, obs)
    for j in (0, 17, obs.size - 1):
        e = np.zeros(obs.size)
        e[j] = 1.0
        assert np.allclose(A @ e, S[:, j], rtol=1e-12, atol=S.max() * 1e-14)


def test_dense_restricted_blocking_invariant():
    eig, obs, _ = _setup(n=512)
    full = cg.dense_restricted(eig, obs, block_bytes=1e12)   # one block
    tiny = cg.dense_restricted(eig, obs, block_bytes=8 * obs.size)  # 1-row blocks
    assert np.array_equal(full, tiny)


def test_quad_form_matches_dense_solve():
    eig, obs, _ = _setup()
    b = RNG.standard_normal(obs.size)
    q_cg, iters = cg.quad_form(eig, obs, b, rtol=1e-10)
    S = cg.dense_restricted(eig, obs)
    u = cho_solve(cho_factor(S, lower=True), b)
    assert abs(q_cg - b @ u) / abs(b @ u) < 1e-8
    assert 0 < iters[0] < obs.size


def test_preconditioner_pays():
    eig, obs, _ = _setup()
    b = RNG.standard_normal(obs.size)
    _, it_pre = cg.solve(eig, obs, b, rtol=1e-8)
    _, it_bare = cg.solve(eig, obs, b, rtol=1e-8, precondition=False,
                          maxiter=50000)
    assert it_pre < it_bare


def test_warm_start_pays():
    eig, obs, _ = _setup()
    b = RNG.standard_normal(obs.size)
    u, it_cold = cg.solve(eig, obs, b, rtol=1e-8)
    _, it_warm = cg.solve(eig, obs, 1.02 * b, x0=1.02 * u, rtol=1e-8)
    assert it_warm < it_cold


def test_multichannel_sums():
    eig, obs, _ = _setup()
    b = RNG.standard_normal((2, obs.size))
    q2, it2 = cg.quad_form(eig, obs, b, rtol=1e-10)
    q0, _ = cg.quad_form(eig, obs, b[0], rtol=1e-10)
    q1, _ = cg.quad_form(eig, obs, b[1], rtol=1e-10)
    assert len(it2) == 2
    assert np.isclose(q2, q0 + q1, rtol=1e-9)


def test_mask_and_indices_agree():
    eig, obs, _ = _setup(n=512)
    mask = np.zeros(512)
    mask[obs] = 1.0
    b = RNG.standard_normal(obs.size)
    qa, _ = cg.quad_form(eig, obs, b, rtol=1e-10)
    qb, _ = cg.quad_form(eig, mask, b, rtol=1e-10)
    assert np.isclose(qa, qb, rtol=1e-12)


def test_restricted_cg_class_vs_pencil_likelihood():
    """Full circle: CG quadratic form + pencil determinant must reproduce
    TimeDomainExact.loglike at off-truth noise parameters."""
    n = 1024
    comps = psd.lisa_tdi2_ae()
    mask = gaps.periodic_mask(n, 10, 50)
    td = TimeDomainExact(mask, [comps["tm"], comps["oms"]], DT)
    rcg = cg.RestrictedCG(mask, [comps["tm"], comps["oms"]], DT, rtol=1e-11)
    r = RNG.standard_normal((2, td.m)) * 1e-21     # ~ instrument scale
    for lam in [(0.0, 0.0), (0.35, -0.2)]:
        quad, _ = rcg.quad_form(lam, r)
        v = td.variances(lam)
        ll_cg = -0.5 * (quad + r.shape[0] * (np.sum(np.log(v)) + td.logdet_C1))
        assert np.isclose(ll_cg, td.loglike(r, lam), rtol=1e-8)


def test_nonconvergence_raises():
    eig, obs, _ = _setup(n=512)
    b = RNG.standard_normal(obs.size)
    with pytest.raises(RuntimeError):
        cg.solve(eig, obs, b, rtol=1e-12, maxiter=3)
