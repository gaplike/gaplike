"""Regression against the paper pipeline (Burke & Pozzoli, gapped-MBHB study).

Rebuilds the paper's scenarios A / B / C purely from gaplike primitives and
checks windows, covariances, the shared noise realization and every
likelihood tier against numbers dumped from the original validated pipeline
(tests/data/reference.json).  The waveform-dependent fingerprint additionally
requires lisabeta and is skipped when it is missing.

These tests are slower than the unit tests (a few dense 1335^2 covariances
and one 2304-point pencil): ~2 minutes total.
"""
import json
import os

import numpy as np
import pytest

from gaplike import covariance as cov
from gaplike import gaps, psd, simulate
from gaplike.likelihood import DiagonalLikelihood, FullCovariance, TimeDomainExact

REF_PATH = os.path.join(os.path.dirname(__file__), "data", "reference.json")
pytestmark = pytest.mark.skipif(not os.path.exists(REF_PATH),
                                reason="reference.json not available")

# ---- paper configuration --------------------------------------------------
DT = 15.0
M = int(round(12.0 * 3600 / DT))          # 2880
F_LO, F_HI = 1.0e-4, 3.1e-2
ALPHA = 0.05
SEED_NOISE = 7
RTOL_RANK = 1e-8

SCEN_GAPS_H = {
    "A": (([(5.9, 1.0), (9.0, 1.0)]), 0.3),
    "B": ([(0.2 + 1.0 * k, 0.15) for k in range(12)], 0.05),
}


@pytest.fixture(scope="module")
def ref():
    with open(REF_PATH) as f:
        return json.load(f)


@pytest.fixture(scope="module")
def setup():
    comps_f = psd.lisa_tdi2_ae()
    grids = {k: psd.two_sided_grid(s, M, DT) for k, s in comps_f.items()}
    S_seg = psd.one_sided_grid(lambda f: comps_f["tm"](f) + comps_f["oms"](f), M, DT)
    idx = cov.band_indices(M, DT, F_LO, F_HI)
    w_seg = gaps.segment_window(M, ALPHA)
    th = np.arange(M) * DT / 3600.0        # hour grid (paper convention)

    weff = {}
    for k, (gh, taper_h) in SCEN_GAPS_H.items():
        weff[k] = w_seg * gaps.gate_from_gaps(gh, taper_h, t=th)
    gateC = gaps.periodic_mask(M, 10, 50)
    weff["C"] = w_seg * gateC.astype(float)

    rng = np.random.default_rng(SEED_NOISE)
    n_td = simulate.noise_td(S_seg, DT, rng, nch=2)
    return dict(grids=grids, S_seg=S_seg, idx=idx, weff=weff, gateC=gateC,
                n_td=n_td)


def test_window_and_grids(ref, setup):
    w = setup["weff"]["A"]
    for i, v in zip((0, 700, 1440, 2879), ref["A"]["weff_spot"]):
        assert np.isclose(w[i], v, rtol=1e-12, atol=1e-300), i
    assert setup["idx"].size == ref["A"]["NB"]


def test_noise_realization_fingerprint(ref, setup):
    n_td = setup["n_td"]
    fp = ref["noise_fingerprint"]
    assert np.isclose(n_td[0, 0], fp[0], rtol=1e-12)
    assert np.isclose(n_td[1, 100], fp[1], rtol=1e-12)
    assert np.isclose(float(np.sum(n_td**2)), fp[2], rtol=1e-12)


def test_covariance_spots(ref, setup):
    CT = cov.full_covariance(setup["weff"]["A"], setup["grids"]["tm"], setup["idx"])
    assert np.isclose(np.real(CT[0, 0]), ref["A"]["CT_00"][0], rtol=1e-10)
    assert np.isclose(np.imag(CT[0, 0]), ref["A"]["CT_00"][1], atol=1e-25)
    assert np.isclose(np.real(CT[3, 7]), ref["A"]["CT_37"][0], rtol=1e-10)
    assert np.isclose(np.imag(CT[3, 7]), ref["A"]["CT_37"][1], rtol=1e-10)
    cd = cov.convolved_diag(setup["weff"]["A"], setup["grids"]["tm"])
    for i, v in zip((5, 100, 700, 1400), ref["A"]["convdiag_spot"]):
        assert np.isclose(cd[i], v, rtol=1e-10), i


@pytest.mark.parametrize("sk", ["A", "B", "C"])
def test_likelihood_tiers_regression(ref, setup, sk):
    """FullCovariance and convolved-diagonal likelihoods on the paper's noise
    realization reproduce the pipeline values (the pipeline residual at the
    true signal equals the transformed pure-noise series)."""
    w = setup["weff"][sk]
    comps = [setup["grids"]["tm"], setup["grids"]["oms"]]
    R = ref[sk]

    Lfull = FullCovariance.from_window(w, comps, DT, F_LO, F_HI,
                                       rtol_rank=RTOL_RANK)
    assert np.isclose(Lfull.s_ref, R["S_ref"], rtol=1e-9)
    assert Lfull.rank == R["rank"]
    assert np.isclose(gaps.window_power(w), R["beta_eff"], rtol=1e-12)

    r = Lfull.transform(setup["n_td"])
    # reference.json was dumped in the v4 AMPLITUDE convention (10^(2 lam));
    # the package now uses POWER deviations (10^(lam)): lam_new = 2 lam_old,
    # so the same covariance point is reached at twice the old test lam.
    lam_t = tuple(2 * v for v in ref["test_lam"])
    assert np.isclose(Lfull.loglike(r, (0.0, 0.0)), R["ll_full_0"], rtol=1e-7)
    assert np.isclose(Lfull.loglike(r, lam_t), R["ll_full_t"], rtol=1e-7)

    # pipeline 'diag' = diagonal of the dense covariance == convolved diagonal
    Ldiag = DiagonalLikelihood.convolved(w, comps, DT, F_LO, F_HI,
                                         s_ref=Lfull.s_ref)
    rd = Ldiag.transform(setup["n_td"])
    assert np.isclose(Ldiag.loglike(rd, (0.0, 0.0)), R["ll_diag_0"], rtol=1e-7)
    assert np.isclose(Ldiag.loglike(rd, lam_t), R["ll_diag_t"], rtol=1e-7)


def test_time_domain_exact_regression(ref, setup):
    comps = [setup["grids"]["tm"], setup["grids"]["oms"]]
    L = TimeDomainExact(setup["gateC"], comps, DT)
    R = ref["C"]
    assert L.m == R["td_m"]
    assert np.isclose(L.logdet_C1, R["td_logdet_Co"], rtol=1e-9)
    r = L.transform(setup["n_td"])
    lam_t = tuple(2 * v for v in ref["test_lam"])   # old->new convention
    assert np.isclose(L.loglike(r, (0.0, 0.0)), R["td_ll_0"], rtol=1e-9)
    assert np.isclose(L.loglike(r, lam_t), R["td_ll_t"], rtol=1e-9)
    # Fisher in lam scales as (d lam_old / d lam_new)^2 = 1/4
    assert np.allclose(L.fisher_noise(nch=2), np.array(R["td_fisher_n"]) / 4.0,
                       rtol=1e-9)


def test_waveform_fingerprint(ref):
    lisabeta = pytest.importorskip("lisabeta.lisa.lisa")  # noqa: F841
    from gaplike.waveform import lisabeta_mbhb_ae
    wf = lisabeta_mbhb_ae(M, DT, F_LO, F_HI)
    theta = np.array([2.5e7, 1.5, 0.3, 0.1, np.log10(40.0),
                      0.6, 1.0, 1.0, 0.4, 0.3, 0.80])
    h_td = wf.td(theta)
    fp = ref["h_td_fingerprint"]
    assert np.isclose(h_td[0, 1000], fp[0], rtol=1e-9)
    assert np.isclose(float(np.sum(h_td**2)), fp[1], rtol=1e-9)
