"""Driver for scenario C: analytics printout, then 3 PE runs on 2 processes
(worker 1: FD full; worker 2: FD diag, then TD exact pencil)."""
import os
import json
import time

os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import emcee

import core as C
import scenarios as S
import pe
import scenC

t0 = time.time()
SC = scenC.build_scenario_C(verbose=True)

# ---------------- analytics printout ----------------
mfull, mdiag = SC["models"]["full"], SC["models"]["diag"]
print("\n--- FD analytics (band-restricted) ---")
print("noise widths  full:", np.round(np.sqrt(np.diag(mfull.cov_n)), 4),
      " diag:", np.round(np.sqrt(np.diag(mdiag.cov_n)), 4))
print("diag Upsilon_n:", np.round(mdiag.upsilon_n, 3),
      " Xi_n:", np.round(mdiag.xi_n, 3))
print("diag Ups_s range:", np.round([mdiag.upsilon_s.min(), mdiag.upsilon_s.max()], 3),
      " Xi_s range:", np.round([mdiag.xi_s.min(), mdiag.xi_s.max()], 3))
print("diag lin-MLE:", np.round(mdiag.mle_noise(SC["resid"]), 4),
      " full lin-MLE:", np.round(mfull.mle_noise(SC["resid"]), 4))
# alias-fill of the convolved diagonal vs stationary level at low f
Sconv = C.conv_diag(SC["weff"], C.S2)
ratio = Sconv[C.idx_band] / (SC["beta_eff"] * C.S_SEG[C.idx_band])
i_lo = np.argmin(np.abs(C.fb - 3e-4))
print(f"conv-diag / (beta_eff S) at 3e-4 Hz: {ratio[i_lo]:.1f};  max over band: {ratio.max():.1f}")

# ---------------- TD pencil analytics ----------------
obs = np.flatnonzero(SC["gate"] > 0.0)
print(f"\n--- TD exact pencil: m = {obs.size} observed samples ---")
tp = scenC.TDPencil(obs)
h0_td, dh_td = scenC.signal_derivs_td()
gamma = dh_td[:, obs, :]
Gn_td = tp.fisher_noise()
Gs_td = tp.fisher_signal(gamma)
cov_n_td = np.linalg.inv(Gn_td)
cov_s_td = C._safe_inv(Gs_td, rcond=1e-12)
print("TD noise widths:", np.round(np.sqrt(np.diag(cov_n_td)), 4))
print("FD-full noise widths:", np.round(np.sqrt(np.diag(mfull.cov_n)), 4))
print("TD/FD-full noise width ratio:", np.round(np.sqrt(np.diag(cov_n_td) / np.diag(mfull.cov_n)), 3))
w_td = np.sqrt(np.diag(cov_s_td)); w_fd = np.sqrt(np.diag(mfull.cov_s)); w_dg = np.sqrt(np.diag(mdiag.cov_s))
print("signal widths TD/FD-full ratio:", np.round(w_td / w_fd, 3))
print("signal widths diag/TD ratio (Xi-like vs exact):", np.round(w_dg / w_td, 2))
r_td = (SC["d_td"] - h0_td)[:, obs]
print("TD exact loglike at truth:", tp.loglike(0.0, 0.0, r_td))
print("TD lin-MLE noise:", np.round(tp.mle_noise(r_td), 4))
snr_td = np.sqrt(sum(np.sum((tp.T @ h0_td[c, obs])**2 / tp.v(0.0, 0.0))
                     for c in range(2)))
print(f"TD SNR (template on obs, full grid): {snr_td:.1f}   "
      f"FD SNR_w (band): {SC['snr']:.1f}")

json.dump(dict(beta_eff=SC["beta_eff"], rank=int(SC["trick"].r), snr_fd=SC["snr"],
               m_obs=int(obs.size),
               widths_full=np.sqrt(np.diag(mfull.cov_n)).tolist(),
               widths_diag=np.sqrt(np.diag(mdiag.cov_n)).tolist(),
               widths_td=np.sqrt(np.diag(cov_n_td)).tolist(),
               ups_diag=mdiag.upsilon_n.tolist(), xi_diag=mdiag.xi_n.tolist(),
               ups_s_diag=[float(mdiag.upsilon_s.min()), float(mdiag.upsilon_s.max())],
               xi_s_diag=[float(mdiag.xi_s.min()), float(mdiag.xi_s.max())],
               sig_w_diag_over_td=(w_dg / w_td).tolist(),
               sig_w_fdfull_over_td=(w_fd / w_td).tolist()),
          open("results/C_analytics.json", "w"), indent=1)
print(f"\n[analytics done at {time.time()-t0:.0f}s]", flush=True)

# ---------------- TD PE runner ----------------
def run_pe_td(nwalkers=48, nburn=1800, nsteps=4200, seed=0, thin=7):
    lo, hi = pe.prior_box(SC)
    d_td = SC["d_td"]
    sig0 = np.concatenate([np.sqrt(np.diag(cov_s_td)), np.sqrt(np.diag(cov_n_td))])

    def logpost(x):
        if np.any(x < lo) or np.any(x > hi):
            return -np.inf
        h = scenC.template_td(x[:C.NSIG])
        r = (d_td - h)[:, obs]
        return tp.loglike(x[C.NSIG], x[C.NSIG + 1], r)

    x0 = pe.X_TRUE.copy()
    scat = np.clip(0.3 * sig0, 0.02 * (hi - lo), 0.12 * (hi - lo))
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(np.maximum(lo + 1e-9 * (hi - lo), x0 - scat),
                     np.minimum(hi - 1e-9 * (hi - lo), x0 + scat),
                     size=(nwalkers, pe.NDIM))
    sam = emcee.EnsembleSampler(nwalkers, pe.NDIM, logpost)
    state = sam.run_mcmc(p0, nburn, progress=False)
    sam.reset()
    sam.run_mcmc(state, nsteps, progress=False)
    chain = sam.get_chain(flat=True, thin=thin)
    lp = sam.get_log_prob(flat=True, thin=thin)
    acc = float(np.mean(sam.acceptance_fraction))
    np.savez("results/C_td.npz", chain=chain, log_prob=lp, x_true=pe.X_TRUE,
             lo=lo, hi=hi, acc=acc, labels=np.array(pe.LAB), model="td",
             scenario="C")
    return acc

# ---------------- run the three PEs on 2 processes ----------------
from multiprocessing import Process

def worker_full():
    t = time.time()
    _, acc = pe.run_pe(SC, "full")
    print(f"[C full] done {time.time()-t:.0f}s acc={acc:.2f}", flush=True)

def worker_diag_td():
    t = time.time()
    _, acc = pe.run_pe(SC, "diag")
    print(f"[C diag] done {time.time()-t:.0f}s acc={acc:.2f}", flush=True)
    t = time.time()
    acc = run_pe_td()
    print(f"[C td]   done {time.time()-t:.0f}s acc={acc:.2f}", flush=True)

if __name__ == "__main__":
    p1 = Process(target=worker_full)
    p2 = Process(target=worker_diag_td)
    p1.start(); p2.start(); p1.join(); p2.join()
    print(f"ALL DONE in {(time.time()-t0)/60:.1f} min", flush=True)
