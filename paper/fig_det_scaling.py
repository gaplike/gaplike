"""Cost and accuracy of the log-determinant on gapped data: dense Cholesky
vs stochastic Lanczos quadrature (SLQ), as the record length grows.

The determinant companion to ``fig_cg_scaling.py`` (same scenario-C comb,
same TDI2 A-channel spectrum, same cadence, same timing conventions): where
that script times the quadratic form, this one times ``log|Sigma_OO|`` ---
the other half of the exact time-domain likelihood, and the half with no
closed form once the noise model is richer than two components.

Per record length N = 2^k:
  dense : build Sigma_OO, Cholesky, read the determinant off the factor
          (the general-case reference route, exactly as in fig_cg_scaling)
  SLQ   : matrix-free Hutchinson + Lanczos quadrature at each (n_probes,
          k_lanczos) setting of a small grid; reported are the estimate,
          its per-probe scatter, wall time (min over repeats) and, where
          dense truth exists, the absolute error.

    python fig_det_scaling.py                          # default quick pass
    python fig_det_scaling.py --kmax-slq 14 --kmax-dense 14 --reps 3
"""
import argparse
import json
import os
import time

import numpy as np

from gaplike import cg as gcg
from gaplike import gaps, psd, slq

DT = 15.0
N_GAP, PERIOD = 10, 50          # scenario-C duty cycle (80% observed)

p = argparse.ArgumentParser()
p.add_argument("--kmin", type=int, default=9)
p.add_argument("--kmax-slq", type=int, default=13)
p.add_argument("--kmax-dense", type=int, default=13,
               help="largest dense point; 2^14 needs ~1.4 GB, 2^15 ~5.5 GB")
p.add_argument("--probes", type=int, nargs="+", default=[8, 32])
p.add_argument("--lanczos", type=int, nargs="+", default=[50, 100])
p.add_argument("--reps", type=int, default=3)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--out", default="results/det_scaling.json")
args = p.parse_args()

comps = psd.lisa_tdi2_ae()
COMPONENTS = [comps["tm"], comps["oms"]]

out = []
for k in range(args.kmin, args.kmax_slq + 1):
    n = 2 ** k
    obs = np.flatnonzero(np.asarray(gaps.periodic_mask(n, N_GAP, PERIOD)) > 0)
    m = obs.size
    eig = gcg.circulant_eigenvalues(COMPONENTS, n, DT)
    row = dict(k=k, N=n, m=int(m))

    # ---- dense: build + Cholesky, determinant off the factor -------------
    if k <= args.kmax_dense:
        t_d, ld_d = np.inf, None
        for _ in range(args.reps):
            t0 = time.perf_counter()
            ld_d, _ = slq.logdet_dense(eig, obs)   # includes the build
            t_d = min(t_d, time.perf_counter() - t0)
        row.update(t_dense=t_d, logdet_dense=ld_d)

    # ---- SLQ grid ---------------------------------------------------------
    for M in args.probes:
        for kl in args.lanczos:
            t_s, res = np.inf, None
            for r in range(args.reps):
                rng = np.random.default_rng(args.seed + r)
                t0 = time.perf_counter()
                res = slq.logdet_slq(eig, obs, n_probes=M, k=kl, rng=rng)
                t_s = min(t_s, time.perf_counter() - t0)
            se = float(res["per_probe"].std(ddof=1) / np.sqrt(M))
            entry = dict(probes=M, lanczos=kl, t=t_s, est=res["est"],
                         stderr=se, clipped=res["clipped"],
                         matvecs=res["matvecs"])
            if "logdet_dense" in row:
                entry["abs_err"] = abs(res["est"] - row["logdet_dense"])
                entry["rel_err"] = entry["abs_err"] / abs(row["logdet_dense"])
            row.setdefault("slq", []).append(entry)

    out.append(row)
    msg = f"k={k:2d}  N={n:6d}  m={m:6d}"
    if "t_dense" in row:
        msg += f"  dense {row['t_dense']:8.2f} s  logdet {row['logdet_dense']:.6e}"
    print(msg, flush=True)
    for e in row.get("slq", []):
        line = (f"    SLQ M={e['probes']:3d} k={e['lanczos']:3d}  "
                f"{e['t']:7.2f} s  est {e['est']:.6e}  se {e['stderr']:.2e}")
        if "abs_err" in e:
            line += f"  abs.err {e['abs_err']:.3e}  rel {e['rel_err']:.2e}"
        print(line, flush=True)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print(f"wrote {args.out}")
