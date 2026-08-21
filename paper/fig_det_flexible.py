"""Determinant DIFFERENCES under flexible (ratio-spline) noise models:
the quantity an MCMC accept ratio actually needs, on gapped data.

Setup: scenario-C comb (80% duty), TDI2 A-channel reference spectrum,
record length n at 15 s cadence ("much shorter data sets": the default
n = 2^12 is ~17 hours).  The noise model is the ratio-spline

    S(f; c) = S_ref(f) * exp( sum_k c_k B_k(log10 f) ),

the flexible case where no two-component closed form exists.  The chain
state is a fixed non-trivial coefficient vector c; proposals are
c' = c + delta with delta ~ N(0, sigma^2) per coefficient, for a range of
step sizes sigma.  For each proposal the target is

    Delta = log|Sigma_OO(c')| - log|Sigma_OO(c)|,

with dense Cholesky as ground truth (n kept small enough to afford it) and
two SLQ estimators compared:

    shared      same Rademacher probes on both operators (the difference
                estimator --- variance sees only the small operator change)
    independent fresh probes for each (naive differencing of two absolute
                estimates)

Reported per step size: absolute errors, per-probe scatters, the
shared/independent variance ratio (the headline number), matvec counts, and
the cost of one quadratic form at the same n for scale.

    python fig_det_flexible.py                       # default: n=2^12
    python fig_det_flexible.py --k 11 --pairs 4      # faster pass
"""
import argparse
import json
import os
import time

import numpy as np

from gaplike import cg as gcg
from gaplike import gaps, psd, slq

DT = 15.0
N_GAP, PERIOD = 10, 50          # scenario-C duty cycle

p = argparse.ArgumentParser()
p.add_argument("--k", type=int, default=12, help="record length 2^k")
p.add_argument("--n-coeff", type=int, default=12)
p.add_argument("--state-scale", type=float, default=0.1,
               help="scale of the fixed chain state c")
p.add_argument("--sigmas", type=float, nargs="+", default=[0.01, 0.1])
p.add_argument("--pairs", type=int, default=6, help="proposals per sigma")
p.add_argument("--probes", type=int, default=16)
p.add_argument("--lanczos", type=int, default=100)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="results/det_flexible.json")
args = p.parse_args()

n = 2 ** args.k
obs = np.flatnonzero(np.asarray(gaps.periodic_mask(n, N_GAP, PERIOD)) > 0)
m = obs.size
f_nyq = 1.0 / (2 * DT)

comps = psd.lisa_tdi2_ae()
S_ref = lambda f: comps["tm"](f) + comps["oms"](f)
knots = slq.spline_knots(1e-4, f_nyq, args.n_coeff)

rng = np.random.default_rng(args.seed)
c_state = args.state_scale * rng.standard_normal(args.n_coeff)
eig1 = gcg.circulant_eigenvalues(
    [slq.ratio_spline(S_ref, knots, c_state)], n, DT)

print(f"n=2^{args.k}={n} ({n * DT / 3600:.1f} h), m={m}, "
      f"{args.n_coeff} spline coefficients, state scale {args.state_scale}")

t0 = time.perf_counter()
ld1, t_chol = slq.logdet_dense(eig1, obs)
t_dense_full = time.perf_counter() - t0
print(f"dense at state: build+Cholesky {t_dense_full:.2f} s "
      f"(factorisation alone {t_chol:.2f} s), logdet {ld1:.6e}")

# cost unit: one quadratic form (CG) at the same n
a = rng.standard_normal(m)
t0 = time.perf_counter()
_, it_cg = gcg.quad_form(eig1, obs, a, rtol=1e-10, maxiter=50000)
t_quad = time.perf_counter() - t0
print(f"one quadratic form (CG, {it_cg[0]} iters): {t_quad:.3f} s")

out = dict(k=args.k, N=n, m=int(m), n_coeff=args.n_coeff,
           state_scale=args.state_scale, probes=args.probes,
           lanczos=args.lanczos, t_dense=t_dense_full, t_quad=t_quad,
           logdet_state=ld1, sigmas=[])

for sig in args.sigmas:
    rows = []
    for j in range(args.pairs):
        delta = sig * rng.standard_normal(args.n_coeff)
        eig2 = gcg.circulant_eigenvalues(
            [slq.ratio_spline(S_ref, knots, c_state + delta)], n, DT)
        ld2, _ = slq.logdet_dense(eig2, obs)
        truth = ld2 - ld1

        r_sh = slq.logdet_diff_slq(
            eig1, eig2, obs, args.probes, args.lanczos,
            np.random.default_rng(args.seed + 100 + j), shared=True)
        r_in = slq.logdet_diff_slq(
            eig1, eig2, obs, args.probes, args.lanczos,
            np.random.default_rng(args.seed + 100 + j), shared=False)

        std_sh = float(r_sh["per_probe"].std(ddof=1))
        std_in = float(r_in["per_probe"].std(ddof=1))
        rows.append(dict(
            truth=truth,
            est_shared=r_sh["est"], err_shared=abs(r_sh["est"] - truth),
            std_shared=std_sh,
            est_indep=r_in["est"], err_indep=abs(r_in["est"] - truth),
            std_indep=std_in,
            var_ratio=(std_in / std_sh) ** 2 if std_sh > 0 else np.inf,
            matvecs=r_sh["matvecs"], clipped=r_sh["clipped"]))
        print(f"  sigma={sig:5.3f} pair {j}: truth {truth:+9.4f}  "
              f"shared {r_sh['est']:+9.4f} (err {rows[-1]['err_shared']:.3f}, "
              f"std {std_sh:.3f})  indep err {rows[-1]['err_indep']:.3f} "
              f"(std {std_in:.3f})  var.ratio {rows[-1]['var_ratio']:.0f}",
              flush=True)
    med_ratio = float(np.median([r["var_ratio"] for r in rows]))
    med_err = float(np.median([r["err_shared"] for r in rows]))
    out["sigmas"].append(dict(sigma=sig, rows=rows,
                              median_var_ratio=med_ratio,
                              median_err_shared=med_err))
    print(f"  sigma={sig}: median shared-probe error {med_err:.3f} "
          f"(log-likelihood units), median variance ratio {med_ratio:.0f}",
          flush=True)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print(f"wrote {args.out}")
