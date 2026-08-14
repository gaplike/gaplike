"""Joint signal+noise PE with emcee for the four covariance models (13 parameters,
physical parametrization of gaps_mbhb_mle.ipynb)."""
import os
import numpy as np
import emcee

import core as C

NDIM = C.NSIG + 2
LAB = C.SIG_NAMES + C.NOISE_NAMES
X_TRUE = np.concatenate([C.THETA_SIG_TRUE, [0.0, 0.0]])

# prior half-width caps (uniform boxes centered on truth), order = LAB:
# [Mtot, q, chi1, chi2, log10dL, iota, phi, lambda, beta, psi, tc_frac, lam_tm, lam_oms]
# Local-mode boxes: the (2,2)-only source carries curved quasi-degeneracies
# (chi_eff surface, iota-D_L, phi-psi, sky); the caps bracket the locally-linear
# neighbourhood of the truth where the Hessian-based description under test is
# meaningful. tc cap = 900 s in segment-fraction units.
CAPS = np.array([1.2e6, 0.25, 0.35, 0.60, 0.065,
                 0.15, 0.30, 0.20, 0.20, 0.30, 900.0 / C.T_OBS, 1.80, 1.80])
FLOORS = np.array([1e3, 1e-3, 1e-3, 2e-3, 5e-4,
                   2e-3, 2e-3, 2e-3, 2e-3, 2e-3, 2.0 / C.T_OBS, 1e-2, 1e-2])
# physical bounds (intersected with the box); q >= 1.02 keeps one branch of the
# exact m1 <-> m2 relabelling degeneracy
PHYS_LO = np.array([1e5, 1.02, -0.99, -0.99, -np.inf,
                    0.02, -np.inf, -np.inf, -np.pi/2 + 0.02, -np.inf, 0.0,
                    -np.inf, -np.inf])
PHYS_HI = np.array([np.inf, np.inf, 0.99, 0.99, np.inf,
                    np.pi - 0.02, np.inf, np.inf, np.pi/2 - 0.02, np.inf, 1.0,
                    np.inf, np.inf])


def prior_box(sc):
    """Per-scenario uniform prior: truth +/- min(cap, max(12*width, 6*sqrt(scatter)))."""
    w = np.zeros(NDIM)
    ns = C.NSIG
    for m in sc["models"].values():
        ws = np.sqrt(np.diag(m.cov_s))
        wsc = np.sqrt(np.abs(np.diag(m.scatter_s)))
        wn = np.sqrt(np.diag(m.cov_n))
        wnsc = np.sqrt(np.abs(np.diag(m.scatter_n)))
        w[:ns] = np.maximum(w[:ns], np.maximum(12 * ws, 6 * wsc))
        w[ns:] = np.maximum(w[ns:], np.maximum(12 * wn, 6 * wnsc))
    w = np.clip(w, FLOORS, CAPS)
    return np.maximum(X_TRUE - w, PHYS_LO), np.minimum(X_TRUE + w, PHYS_HI)


def make_logpost(sc, model_key):
    lo, hi = prior_box(sc)
    y = sc["y"]
    S_ref = sc["S_ref"]
    weff = sc["weff"]
    ns = C.NSIG
    if model_key == "full":
        trick = sc["trick"]

        def loglike(r, a, b):
            return trick.loglike(a, b, r)
    else:
        v1, v2 = [np.real(np.asarray(c)) for c in sc["comps"][model_key]]
        nch = y.shape[0]

        def loglike(r, a, b):
            var = 10.0**(a) * v1 + 10.0**(b) * v2
            return float(-nch * np.sum(np.log(var)) - np.sum(np.abs(r)**2 / var[None, :]))

    def logpost(x):
        if np.any(x < lo) or np.any(x > hi):
            return -np.inf
        h = C.template_y(x[:ns], weff) / np.sqrt(S_ref)
        return loglike(y - h, x[ns], x[ns + 1])

    return logpost, (lo, hi)


def noise_profile_mle(sc, model_key):
    """Exact (nonlinear) profile MLE of the 2 noise parameters at fixed true signal."""
    from scipy.optimize import minimize
    r = sc["resid"]
    if model_key == "full":
        trick = sc["trick"]
        nll = lambda x: -trick.loglike(x[0], x[1], r)
    else:
        v1, v2 = [np.real(np.asarray(c)) for c in sc["comps"][model_key]]
        nch = r.shape[0]
        a2 = (np.abs(r)**2).sum(0)

        def nll(x):
            var = 10.0**(x[0]) * v1 + 10.0**(x[1]) * v2
            return float(nch * np.sum(np.log(var)) + np.sum(a2 / var))
    res = minimize(nll, np.zeros(2), method="Nelder-Mead",
                   options=dict(xatol=1e-6, fatol=1e-8, maxiter=2000))
    x = res.x
    h = 1e-4
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            xpp = x.copy(); xpp[i] += h; xpp[j] += h
            xpm = x.copy(); xpm[i] += h; xpm[j] -= h
            xmp = x.copy(); xmp[i] -= h; xmp[j] += h
            xmm = x.copy(); xmm[i] -= h; xmm[j] -= h
            H[i, j] = (nll(xpp) - nll(xpm) - nll(xmp) + nll(xmm)) / (4 * h * h)
    return x, np.linalg.inv(0.5 * (H + H.T))


def noise_pseudo_true(sc, model_key):
    """Nonlinear (population-level) noise bias: the KL pseudo-true parameters,
    argmin over (lam_tm, lam_oms) of the EXPECTED negative log-likelihood of the
    model under the true windowed covariance Q (local diagnostics module)."""
    import diagnostics as D
    Q = sc["truth"]["Q"]
    if model_key == "full":
        return D.kl_pseudo_true_full(sc["trick"], Q)
    v1, v2 = [np.real(np.asarray(c)) for c in sc["comps"][model_key]]
    return D.kl_pseudo_true_diag([v1, v2], np.real(np.diag(Q)))


def noise_sandwich(sc, model_key):
    """KL pseudo-true parameters lam* and the Godambe/White sandwich covariance
    H^-1 J H^-1 of the mis-specified noise MLE about lam* (local diagnostics module;
    for 'full' the model is correct: (0, variance_n))."""
    import diagnostics as D
    m = sc["models"][model_key]
    if model_key == "full":
        return np.zeros(2), m.variance_n
    v1, v2 = [np.real(np.asarray(c)) for c in sc["comps"][model_key]]
    return D.sandwich_diag([v1, v2], sc["truth"]["Q"], nch=C.NCH)


def run_pe(sc, model_key, nwalkers=48, nburn=1800, nsteps=4200, seed=0, thin=7,
           outdir="results", tag=""):
    os.makedirs(outdir, exist_ok=True)
    fname = os.path.join(outdir, f"{sc['key']}{tag}_{model_key}.npz")
    logpost, (lo, hi) = make_logpost(sc, model_key)

    m = sc["models"][model_key]
    # truth-centred init (the linearized MLE can extrapolate off the curved
    # likelihood ridge along the quasi-degenerate directions)
    x0 = X_TRUE.copy()
    sig0 = np.concatenate([np.sqrt(np.diag(m.cov_s)), np.sqrt(np.diag(m.cov_n))])
    scat = np.clip(0.3 * sig0, 0.02 * (hi - lo), 0.12 * (hi - lo))
    rng = np.random.default_rng(seed)
    p0 = rng.uniform(np.maximum(lo + 1e-9 * (hi - lo), x0 - scat),
                     np.minimum(hi - 1e-9 * (hi - lo), x0 + scat),
                     size=(nwalkers, NDIM))

    sampler = emcee.EnsembleSampler(nwalkers, NDIM, logpost)
    state = sampler.run_mcmc(p0, nburn, progress=False)
    sampler.reset()
    sampler.run_mcmc(state, nsteps, progress=False)
    chain = sampler.get_chain(flat=True, thin=thin)
    lp = sampler.get_log_prob(flat=True, thin=thin)
    acc = float(np.mean(sampler.acceptance_fraction))
    np.savez(fname, chain=chain, log_prob=lp, x_true=X_TRUE, lo=lo, hi=hi,
             acc=acc, labels=np.array(LAB), model=model_key, scenario=sc["key"])
    return chain, acc
