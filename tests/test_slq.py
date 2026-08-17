"""Tests for gaplike.slq: Lanczos quadrature log-determinants + ratio splines.

Ground truth throughout is the dense Cholesky of the restricted covariance
(small n only).  Stochastic assertions are made against the estimator's own
reported scatter, with seeds fixed.
"""
import numpy as np
import pytest

from gaplike import cg as gcg
from gaplike import gaps, psd, slq

DT = 15.0


def _setup(n=512, n_gap=10, period=50):
    comps = list(psd.lisa_tdi2_ae().values())
    eig = gcg.circulant_eigenvalues(comps, n, DT)
    obs = np.flatnonzero(np.asarray(gaps.periodic_mask(n, n_gap, period)) > 0)
    return comps, eig, obs


def test_quad_log_exact_at_full_krylov():
    """With k = m the quadrature is exact: compare to eigendecomposition."""
    rng = np.random.default_rng(1)
    m = 24
    B = rng.standard_normal((m, m))
    A = B @ B.T + m * np.eye(m)
    w, U = np.linalg.eigh(A)
    z = rng.standard_normal(m)
    exact = float((U.T @ z) ** 2 @ np.log(w))
    est, clipped = slq.quad_log(lambda v: A @ v, z, k=m)
    assert clipped == 0
    assert est == pytest.approx(exact, rel=1e-10)


def test_circulant_logdet_matches_dense_when_gap_free():
    # The library grid convention zeroes the DC bin (mean removal), which
    # makes the UNRESTRICTED circulant singular by design.  The identity is
    # therefore tested on a strictly positive precomputed two-sided grid,
    # which as_two_sided passes through untouched.
    n = 256
    f = np.fft.fftfreq(n, DT)
    s2 = 1e-40 * (1.0 + (np.abs(f) / 1e-3) ** 2)      # > 0 everywhere, DC incl.
    eig = gcg.circulant_eigenvalues([s2], n, DT)
    obs = np.arange(n)                        # no gaps: Sigma_OO = circulant
    truth, _ = slq.logdet_dense(eig, obs)
    assert slq.circulant_logdet(eig) == pytest.approx(truth, rel=1e-10)


def test_logdet_slq_against_dense():
    _, eig, obs = _setup(n=512)
    truth, _ = slq.logdet_dense(eig, obs)
    out = slq.logdet_slq(eig, obs, n_probes=24, k=100,
                         rng=np.random.default_rng(2))
    assert out["clipped"] == 0
    scatter = out["per_probe"].std(ddof=1) / np.sqrt(len(out["per_probe"]))
    # within 5 standard errors of its own scatter, and scatter itself small
    # relative to the (large) determinant magnitude
    assert abs(out["est"] - truth) < 5 * scatter
    assert scatter < 0.02 * abs(truth)


def test_shared_probe_difference_beats_independent():
    comps, eig1, obs = _setup(n=512)
    S_ref = lambda f: sum(c(f) for c in comps)
    knots = slq.spline_knots(1e-4, 1.0 / (2 * DT), 12)
    rng_c = np.random.default_rng(3)
    coeffs = 0.1 * rng_c.standard_normal(12)
    comp2 = slq.ratio_spline(S_ref, knots, coeffs)
    eig2 = gcg.circulant_eigenvalues([comp2], 512, DT)

    t2, _ = slq.logdet_dense(eig2, obs)
    t1, _ = slq.logdet_dense(eig1, obs)
    truth = t2 - t1

    sh = slq.logdet_diff_slq(eig1, eig2, obs, n_probes=16, k=100,
                             rng=np.random.default_rng(4), shared=True)
    ind = slq.logdet_diff_slq(eig1, eig2, obs, n_probes=16, k=100,
                              rng=np.random.default_rng(4), shared=False)
    std_sh = sh["per_probe"].std(ddof=1)
    std_ind = ind["per_probe"].std(ddof=1)
    se_sh = std_sh / np.sqrt(len(sh["per_probe"]))
    assert abs(sh["est"] - truth) < 5 * max(se_sh, 1e-12)
    # the whole point: sharing probes must collapse the variance
    assert std_sh < std_ind / 3.0


def test_ratio_spline_identity_and_clipping():
    comps = list(psd.lisa_tdi2_ae().values())
    S_ref = lambda f: sum(c(f) for c in comps)
    knots = slq.spline_knots(1e-4, 3e-2, 10)
    comp0 = slq.ratio_spline(S_ref, knots, np.zeros(10))
    f = np.geomspace(1e-4, 3e-2, 64)
    assert np.allclose(comp0(f), S_ref(f), rtol=1e-12)
    # outside the knot span the log-ratio extrapolates flat (finite, positive)
    comp = slq.ratio_spline(S_ref, knots, 0.3 * np.ones(10))
    lo, hi = comp(np.array([1e-5])), comp(np.array([1e-1]))
    assert np.isfinite(lo).all() and np.isfinite(hi).all()
    assert (lo > 0).all() and (hi > 0).all()


def test_batched_matches_serial():
    """Same probes, same mathematics: the batched path must reproduce the
    serial per-probe estimates to floating-point order."""
    comps, eig, obs = _setup(n=512)
    a = slq.logdet_slq(eig, obs, n_probes=8, k=60,
                       rng=np.random.default_rng(7), batched=False)
    b = slq.logdet_slq(eig, obs, n_probes=8, k=60,
                       rng=np.random.default_rng(7), batched=True,
                       reorth="full")
    assert np.allclose(a["per_probe"], b["per_probe"], rtol=1e-6)

    S_ref = lambda f: sum(c(f) for c in comps)
    knots = slq.spline_knots(1e-4, 1.0 / (2 * DT), 12)
    comp2 = slq.ratio_spline(S_ref, knots,
                             0.05 * np.random.default_rng(8).standard_normal(12))
    eig2 = gcg.circulant_eigenvalues([comp2], 512, DT)
    d_ser = slq.logdet_diff_slq(eig, eig2, obs, 8, 60,
                                np.random.default_rng(9), batched=False)
    d_bat = slq.logdet_diff_slq(eig, eig2, obs, 8, 60,
                                np.random.default_rng(9), batched=True,
                                reorth="full")
    assert np.allclose(d_ser["per_probe"], d_bat["per_probe"], rtol=1e-6,
                       atol=1e-8)


def test_batched_chunking_consistent():
    """Chunked execution (tiny max_bytes) must agree with one-shot batched."""
    _, eig, obs = _setup(n=512)
    one = slq.logdet_slq(eig, obs, n_probes=8, k=40,
                         rng=np.random.default_rng(11), batched=True)
    chk = slq.logdet_slq(eig, obs, n_probes=8, k=40,
                         rng=np.random.default_rng(11), batched=True,
                         max_bytes=200_000)      # forces several chunks
    assert np.allclose(one["per_probe"], chk["per_probe"], rtol=1e-10)


def test_no_reorth_accuracy():
    """SLQ without reorthogonalisation (the fast batched mode) must still
    agree with dense truth within its own scatter --- the standard SLQ
    robustness result, verified rather than assumed."""
    _, eig, obs = _setup(n=512)
    truth, _ = slq.logdet_dense(eig, obs)
    out = slq.logdet_slq(eig, obs, n_probes=24, k=100,
                         rng=np.random.default_rng(21), batched=True,
                         reorth="none")
    se = out["per_probe"].std(ddof=1) / np.sqrt(len(out["per_probe"]))
    assert abs(out["est"] - truth) < 5 * se
    assert se < 0.02 * abs(truth)


def test_complement_logdet_exact():
    """The complement identity must reproduce dense truth to near machine
    precision, across a sweep of the DC-floor regulariser."""
    _, eig, obs = _setup(n=1024)
    truth, _ = slq.logdet_dense(eig, obs)
    # the DC floor is removed EXACTLY by the rank-one downdate, so the
    # result must be floor-independent and machine-tight
    for scale in (1e-6, 1e-4, 1e-2):
        got = slq.logdet_complement(eig, obs, dc_scale=scale)
        assert abs(got - truth) < 1e-6, f"dc_scale={scale}: {got} vs {truth}"


def test_complement_quad_form_matches_cg():
    """The cached factor must reproduce CG quadratic forms."""
    _, eig, obs = _setup(n=1024)
    rng = np.random.default_rng(31)
    d = rng.standard_normal(obs.size)
    q_cg, _ = gcg.quad_form(eig, obs, d, rtol=1e-12, maxiter=50000)
    fac = slq.ComplementFactor(eig, obs)
    q_c = fac.quad_form(d)
    assert q_c == pytest.approx(q_cg, rel=1e-8)


def test_complement_diff_matches_dense():
    """Determinant differences through two complement factors vs dense."""
    comps, eig1, obs = _setup(n=1024)
    S_ref = lambda f: sum(c(f) for c in comps)
    knots = slq.spline_knots(1e-4, 1.0 / (2 * DT), 12)
    coeffs = 0.1 * np.random.default_rng(33).standard_normal(12)
    eig2 = gcg.circulant_eigenvalues(
        [slq.ratio_spline(S_ref, knots, coeffs)], 1024, DT)
    t2, _ = slq.logdet_dense(eig2, obs)
    t1, _ = slq.logdet_dense(eig1, obs)
    d_c = (slq.logdet_complement(eig2, obs)
           - slq.logdet_complement(eig1, obs))
    assert d_c == pytest.approx(t2 - t1, abs=1e-6)
