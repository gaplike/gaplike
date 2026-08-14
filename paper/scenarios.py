"""Scenario builder: controlled gaps + covariances + analytics for each case."""
import numpy as np
import core as C

SEED_NOISE = 7        # one shared noise realization across scenarios (paired comparison)
RCOND_DIAG = 1e-13
RTOL_RANK = 1e-8      # rank cut used CONSISTENTLY by the full-cov trick and the
                      # analytic pseudo-inverses (rcond caveat of gaps_noise)

# controlled-gap scenarios (verbatim definition from the request)
SCEN_DEF = {
    "A": dict(label="A: two long gaps",
              gaps=[(5.9, 1.0), (9., 1.0)], taper_h=0.3),
    "B": dict(label="B: twelve short gaps",
              gaps=[(0.2 + 1. * k, 0.15) for k in range(12)], taper_h=0.05),
}

SCENARIOS = {"nogap": dict(gaps=[], taper_h=0.0,
                           desc="no gap (Tukey %.2f segment window only)" % C.ANALYSIS_ALPHA)}
for k, v in SCEN_DEF.items():
    SCENARIOS[k] = dict(gaps=v["gaps"], taper_h=v["taper_h"], desc=v["label"])


def build_scenario(key, verbose=True):
    sc = SCENARIOS[key]
    weff, gate = C.make_weff(sc["gaps"], sc.get("taper_h", 0.12))
    beta_eff = float(np.sum(weff**2) / C.M)

    CT = C.freq_cov(weff, C.S2_TM, C.idx_band)
    CO = C.freq_cov(weff, C.S2_OMS, C.idx_band)
    S_ref = float(np.median(np.real(np.diag(CT + CO))))
    CTn, COn = CT / S_ref, CO / S_ref

    h0, dh = C.signal_derivs(weff)
    h0n, dhn = h0 / np.sqrt(S_ref), dh / np.sqrt(S_ref)

    truth = C.build_truth([CTn, COn], dhn, rcond=RTOL_RANK)

    comps = {
        "full": [CTn, COn],
        "diag": [np.real(np.diag(CTn)), np.real(np.diag(COn))],
        "bare": [beta_eff * C.STM_BARE / S_ref, beta_eff * C.SOMS_BARE / S_ref],
        "psd":  [C.STM_BARE / S_ref, C.SOMS_BARE / S_ref],
    }
    rcond_of = {"full": RTOL_RANK, "diag": RCOND_DIAG, "bare": RCOND_DIAG, "psd": RCOND_DIAG}
    models = {k: C.ApproxModel(v, dhn, truth, rcond=rcond_of[k]) for k, v in comps.items()}
    trick = C.FullCovTrick(CTn, COn, rtol_rank=RTOL_RANK)

    rng = np.random.default_rng(SEED_NOISE)
    n_td = C.gen_noise_td(rng)                              # (2, M)
    ht_true = C.wf_AE_fullgrid(C.THETA_SIG_TRUE)            # (2, nfreq)
    h_td_true = np.fft.irfft(ht_true / C.DT, n=C.M, axis=-1)
    d_td = n_td + h_td_true
    y = C.y_of_td(d_td, weff) / np.sqrt(S_ref)              # (2, NB)
    resid_true = y - h0n

    snr = C.snr_windowed(h0n, truth["Qi"])
    out = dict(key=key, desc=sc["desc"], weff=weff, gate=gate, beta_eff=beta_eff,
               S_ref=S_ref, comps=comps, models=models, trick=trick, truth=truth,
               h0=h0n, dh=dhn, y=y, resid=resid_true, snr=snr,
               n_td=n_td, h_td=h_td_true, d_td=d_td)
    if verbose:
        print(f"[{key}] {sc['desc']}")
        print(f"    beta_eff={beta_eff:.4f}  rank r={trick.r}/{C.NB}  SNR_w(A+E)={snr:.1f}")
    return out
