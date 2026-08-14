"""Scenario C — drastic rectangular comb (no taper on the gaps): N_GAP
consecutive samples removed at the start of every PERIOD-sample block
(10 of every 50: 150 s removed every 750 s, duty 80%).

Two analyses on the same data realization:
  (i)  frequency domain, as in scenarios A/B, but ONLY the two models
       full (dense windowed covariance) and diag (exact convolved diagonal);
  (ii) exact time domain (gaplike.likelihood.TimeDomainExact): the observed
       samples of gapped stationary data are exactly stationary-restricted;
       with two amplitude components this is the two-component pencil regime
       (m <~ 1e4): one simultaneous diagonalization, then O(m) exact
       likelihood, closed-form determinant, Upsilon = Xi = 1 by construction.
       No window/taper appears anywhere in (ii).
"""
import numpy as np

import gaplike as gl
import core as C
import scenarios as S

LN10 = C.LN10

# rectangular comb: 10 of every 50 (the 1-of-5 variant: N_GAP, PERIOD = 1, 5)
N_GAP, PERIOD = 10, 50


def build_scenario_C(verbose=True):
    gate = gl.gaps.periodic_mask(C.M, N_GAP, PERIOD).astype(float)
    weff = C.W_SEG * gate
    beta_eff = float(np.sum(weff**2) / C.M)

    CT = C.freq_cov(weff, C.S2_TM, C.idx_band)
    CO = C.freq_cov(weff, C.S2_OMS, C.idx_band)
    S_ref = float(np.median(np.real(np.diag(CT + CO))))
    CTn, COn = CT / S_ref, CO / S_ref

    h0, dh = C.signal_derivs(weff)
    h0n, dhn = h0 / np.sqrt(S_ref), dh / np.sqrt(S_ref)

    truth = C.build_truth([CTn, COn], dhn, rcond=S.RTOL_RANK)
    comps = {"full": [CTn, COn],
             "diag": [np.real(np.diag(CTn)), np.real(np.diag(COn))]}
    rcond_of = {"full": S.RTOL_RANK, "diag": S.RCOND_DIAG}
    models = {k: C.ApproxModel(v, dhn, truth, rcond=rcond_of[k])
              for k, v in comps.items()}
    trick = C.FullCovTrick(CTn, COn, rtol_rank=S.RTOL_RANK)

    rng = np.random.default_rng(S.SEED_NOISE)          # same realization as A/B
    n_td = C.gen_noise_td(rng)
    ht_true = C.wf_AE_fullgrid(C.THETA_SIG_TRUE)
    h_td_true = np.fft.irfft(ht_true / C.DT, n=C.M, axis=-1)
    d_td = n_td + h_td_true
    y = C.y_of_td(d_td, weff) / np.sqrt(S_ref)
    resid = y - h0n
    snr = C.snr_windowed(h0n, truth["Qi"])

    sc = dict(key="C", desc="C: drastic gaps",
              weff=weff, gate=gate,
              beta_eff=beta_eff, S_ref=S_ref, comps=comps, models=models,
              trick=trick, truth=truth, h0=h0n, dh=dhn, y=y, resid=resid,
              snr=snr, n_td=n_td, h_td=h_td_true, d_td=d_td)
    if verbose:
        print(f"[C] {sc['desc']}")
        print(f"    beta_eff={beta_eff:.4f}  rank r={trick.r}/{C.NB}  "
              f"SNR_w(A+E)={snr:.1f}")
    return sc


# ---------------------------------------------------------------------------
# exact time-domain pencil on Sigma_OO (gaplike.likelihood.TimeDomainExact)
# ---------------------------------------------------------------------------
class TDPencil(gl.likelihood.TimeDomainExact):
    """v4-compatible facade: constructed from the observed-sample indices,
    methods keyed on (a, b) = (lam_tm, lam_oms)."""

    def __init__(self, obs):
        mask = np.zeros(C.M, bool)
        mask[np.asarray(obs)] = True
        super().__init__(mask, [C.S2_TM, C.S2_OMS], C.DT)

    @property
    def logdet_Co(self):
        return self.logdet_C1

    def v(self, a, b):
        return self.variances((a, b))

    def loglike(self, a, b, r):
        """r: (nch, m) real residuals on the observed samples."""
        return super().loglike(r, (a, b))

    def fisher_noise(self, nch=C.NCH):
        return super().fisher_noise(nch=nch)

    def fisher_signal(self, gamma, lam=(0.0, 0.0)):
        return super().fisher_signal(gamma, lam=lam)

    def score_noise(self, r):
        return super().score_noise(r)

    def mle_noise(self, r, nch=C.NCH):
        return np.linalg.solve(self.fisher_noise(nch), self.score_noise(r))

    def mle_signal(self, gamma, r):
        return super().mle_signal(gamma, r)


def signal_derivs_td():
    """h0_td: (2, M); dh_td: (2, M, NSIG) — physical TD templates and
    derivatives (same FD steps as core.signal_derivs, no window)."""
    ht0 = C.wf_AE_fullgrid(C.THETA_SIG_TRUE)
    h0 = np.fft.irfft(ht0 / C.DT, n=C.M, axis=-1)
    dh = np.zeros((2, C.M, C.NSIG))
    for a, name in enumerate(C.SIG_NAMES):
        st = C.FD_STEPS[name]
        if name == "log10dL":
            dh[:, :, a] = -LN10 * h0
            continue
        if name == "tc_frac":
            dht = ht0 * (-2j * np.pi * C.freqs * C.T_OBS)[None, :]
            dh[:, :, a] = np.fft.irfft(dht / C.DT, n=C.M, axis=-1)
            continue
        tp = C.THETA_SIG_TRUE.copy(); tp[a] += st
        tm = C.THETA_SIG_TRUE.copy(); tm[a] -= st
        hp = np.fft.irfft(C.wf_AE_fullgrid(tp) / C.DT, n=C.M, axis=-1)
        hm = np.fft.irfft(C.wf_AE_fullgrid(tm) / C.DT, n=C.M, axis=-1)
        dh[:, :, a] = (hp - hm) / (2 * st)
    return h0, dh


def template_td(theta_sig):
    ht = C.wf_AE_fullgrid(theta_sig)
    return np.fft.irfft(ht / C.DT, n=C.M, axis=-1)
