"""Ensemble verification of the noise-sector scatter formulas.

For each scenario (A, B, C) and covariance model, draw N_ENS independent
stationary noise realizations, window them, and compute
  (i)  the EXACT profile noise MLE (2D Nelder-Mead of the model likelihood),
  (ii) the LINEARIZED MLE statistic  dlam = Gamma'^-1 s(r)  (paper formula).
Ensemble means/stds are compared against
  - leading order: bias_n (eq:biased_noise_params_mismodelling) and
    variance_n (eq:noise_mismodelling_params_covariance)  -> exact for (ii);
  - nonlinear: KL pseudo-true lam* and Godambe/White sandwich  -> describes (i).

Scenarios A and B run the four frequency-domain tiers of the hierarchy;
scenario C runs the two tiers that survive it (full, convolved diagonal)
plus the exact time-domain solver of Sec. td_exact ('td'), whose model is
correct by construction, so that lam* = 0 and the sandwich collapses to the
inverse Fisher matrix -- the ensemble then *measures* Upsilon = Xi = 1
instead of asserting it.

Writes results/ensemble_noise.{npz,json} and prints a summary table.
"""
import json
import numpy as np
from scipy.optimize import minimize

import core as C
import scenarios as S
import scenC
import pe

N_ENS = 300
SEED0 = 20000
MODELS = {"A": ["full", "diag", "bare", "psd"],
          "B": ["full", "diag", "bare", "psd"],
          "C": ["full", "diag", "td"]}

out = {}
for skey in ["A", "B", "C"]:
    sc = (S.build_scenario(skey, verbose=False) if skey in ("A", "B")
          else scenC.build_scenario_C(verbose=False))
    weff, S_ref = sc["weff"], sc["S_ref"]
    trick = sc["trick"]
    models = MODELS[skey]

    # exact time-domain tier: pencil on Sigma_OO over the observed samples
    tp, obs = None, None
    if "td" in models:
        obs = np.flatnonzero(sc["gate"] > 0.0)
        tp = scenC.TDPencil(obs)
        cov_td = np.linalg.inv(tp.fisher_noise(nch=C.NCH))

    # per-model fast profile-MLE closures on precomputed sufficient statistics
    prep = {}
    for mk in models:
        if mk in ("full", "td"):
            prep[mk] = None          # projection recomputed per realization
        else:
            v1, v2 = [np.real(np.asarray(c)) for c in sc["comps"][mk]]
            prep[mk] = (v1, v2)

    lam_star = {mk: (np.zeros(2) if mk in ("full", "td")
                     else pe.noise_pseudo_true(sc, mk)) for mk in models}

    exact = {mk: np.zeros((N_ENS, 2)) for mk in models}
    lin = {mk: np.zeros((N_ENS, 2)) for mk in models}

    for i in range(N_ENS):
        rng = np.random.default_rng(SEED0 + i)
        n_td = C.gen_noise_td(rng)
        r = C.y_of_td(n_td, weff) / np.sqrt(S_ref)      # (2, NB) pure noise
        r_obs = n_td[:, obs] if tp is not None else None

        for mk in models:
            if mk == "td":
                lin[mk][i] = tp.mle_noise(r_obs)

                def nll(x, r_obs=r_obs):
                    return -tp.loglike(x[0], x[1], r_obs)
            else:
                m = sc["models"][mk]
                lin[mk][i] = m.mle_noise(r)
                if mk == "full":
                    w = (np.abs(trick.Tproj @ r.T)**2).sum(axis=1)    # (r,)

                    def nll(x, w=w):
                        lam = 10.0**(x[0]) * trick.mu + 10.0**(x[1])
                        return float(np.sum(w / lam) + C.NCH * np.sum(np.log(lam)))
                else:
                    v1, v2 = prep[mk]
                    a2 = (np.abs(r)**2).sum(0)

                    def nll(x, a2=a2, v1=v1, v2=v2):
                        var = 10.0**(x[0]) * v1 + 10.0**(x[1]) * v2
                        return float(np.sum(a2 / var) + C.NCH * np.sum(np.log(var)))
            res = minimize(nll, lam_star[mk], method="Nelder-Mead",
                           options=dict(xatol=1e-6, fatol=1e-8, maxiter=2000))
            exact[mk][i] = res.x

    out[skey] = {}
    print(f"\n===== scenario {skey}  (N_ens = {N_ENS}) =====")
    print("model |   width      | exact MLE: mean +/- std   vs  lam* +/- sd_sw"
          "      | lin stat: mean +/- std  vs  bias_lin +/- sd_lin")
    for mk in models:
        if mk == "td":
            wdt = np.sqrt(np.diag(cov_td))
            ls, sand = np.zeros(2), cov_td           # correct model: J = H = Gamma
            sd_lin, bias_lin = wdt, np.zeros(2)
            ups_lin = np.ones(2)
        else:
            m = sc["models"][mk]
            wdt = np.sqrt(np.diag(m.cov_n))
            ls, sand = pe.noise_sandwich(sc, mk)
            sd_lin = np.sqrt(np.diag(m.variance_n))
            bias_lin, ups_lin = m.bias_n, m.upsilon_n
        sd_sw = np.sqrt(np.diag(sand))
        e_mean, e_std = exact[mk].mean(0), exact[mk].std(0, ddof=1)
        l_mean, l_std = lin[mk].mean(0), lin[mk].std(0, ddof=1)
        ups_emp = np.sqrt((exact[mk]**2).mean(0)) / wdt
        ups_sw = np.sqrt(np.diag(sand) + ls**2) / wdt   # sandwich-based Upsilon
        out[skey][mk] = dict(width=wdt.tolist(), lam_star=np.asarray(ls).tolist(),
                             sd_sw=sd_sw.tolist(),
                             bias_lin=np.asarray(bias_lin).tolist(),
                             sd_lin=sd_lin.tolist(),
                             exact_mean=e_mean.tolist(), exact_std=e_std.tolist(),
                             lin_mean=l_mean.tolist(), lin_std=l_std.tolist(),
                             ups_emp=ups_emp.tolist(),
                             ups_sw=ups_sw.tolist(),
                             ups_lin=np.asarray(ups_lin).tolist())
        for j, nm in enumerate(["tm ", "oms"]):
            print(f"{mk:5s} {nm} w={wdt[j]:8.4f} | "
                  f"{e_mean[j]:+8.4f} +/- {e_std[j]:7.4f}  vs  "
                  f"{ls[j]:+8.4f} +/- {sd_sw[j]:7.4f}   | "
                  f"{l_mean[j]:+8.3f} +/- {l_std[j]:7.3f}  vs  "
                  f"{bias_lin[j]:+8.3f} +/- {sd_lin[j]:7.3f}")
        print(f"      Ups_emp = {ups_emp[0]:7.2f} {ups_emp[1]:7.2f}   "
              f"Ups_sw = {ups_sw[0]:7.2f} {ups_sw[1]:7.2f}   "
              f"Ups_lin = {out[skey][mk]['ups_lin'][0]:8.2f} "
              f"{out[skey][mk]['ups_lin'][1]:8.2f}")

np.savez("results/ensemble_noise.npz",
         summary=json.dumps(out), n_ens=N_ENS, seed0=SEED0)
with open("results/ensemble_noise.json", "w") as f:
    json.dump(out, f, indent=1)
print("\nsaved results/ensemble_noise.{npz,json}")
