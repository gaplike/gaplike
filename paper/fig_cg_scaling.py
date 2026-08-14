"""Cost of one quadratic form a^T Sigma_OO^-1 a on gapped data, dense vs
matrix-free preconditioned conjugate gradients, as the record length grows
(left panel of Fig. `cg_scaling` in the paper; the figure itself is drawn by
_mkfig.py from the JSON this script writes).

Gap pattern: the scenario-C comb (10 of every 50 samples removed, 80% duty)
scaled with N.  Noise: the TDI2 A-channel SciRDv1 spectrum at the reference
amplitudes, on the DFT grid of each N at fixed cadence dt = 15 s.  Depends
only on `gaplike` (+ numpy/scipy); no waveform code is touched.

Timed, per route, is everything needed at a *new* set of noise parameters:
  dense : build Sigma_OO from the autocovariance, Cholesky, triangular solve
          (deliberately given NO structure to exploit --- no simultaneous
          diagonalization, no two-component trick: the general case)
  CG    : preconditioned conjugate gradients, matrix never formed
Reported is the minimum over several repeats.

    python fig_cg_scaling.py                       # full range, ~15-30 min
    python fig_cg_scaling.py --kmax-cg 12 --kmax-dense 12   # quick pass
    python _mkfig.py                               # then draw the figure
"""
import argparse
import json
import os
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from gaplike import cg as gcg
from gaplike import gaps, psd

DT = 15.0
N_GAP, PERIOD = 10, 50          # scenario-C duty cycle

p = argparse.ArgumentParser()
p.add_argument("--kmin", type=int, default=8)
p.add_argument("--kmax-cg", type=int, default=18)
p.add_argument("--kmax-dense", type=int, default=15,
               help="largest dense point; 2^15 needs ~5.5 GB for the matrix")
p.add_argument("--rtol", type=float, default=1e-10)
p.add_argument("--out", default="results/cg_scaling.json")
args = p.parse_args()

comps = psd.lisa_tdi2_ae()
COMPONENTS = [comps["tm"], comps["oms"]]

out = []
rng = np.random.default_rng(0)
for k in range(args.kmin, args.kmax_cg + 1):
    n = 2 ** k
    obs = np.flatnonzero(np.asarray(gaps.periodic_mask(n, N_GAP, PERIOD)) > 0)
    m = obs.size
    eig = gcg.circulant_eigenvalues(COMPONENTS, n, DT)
    a = rng.standard_normal(m)
    reps = 10 if k <= 12 else (5 if k <= 15 else 3)

    # ---- CG: preconditioned, matrix-free ---------------------------------
    t_cg, q_cg, nit = np.inf, None, None
    for _ in range(reps):
        t0 = time.perf_counter()
        q_cg, iters = gcg.quad_form(eig, obs, a, rtol=args.rtol, maxiter=50000)
        t_cg = min(t_cg, time.perf_counter() - t0)
        nit = iters[0]
    row = dict(k=k, N=n, m=int(m), t_cg=t_cg, q_cg=q_cg, iters=nit)

    # ---- dense: build, factorise, solve -----------------------------------
    if k <= args.kmax_dense:
        try:
            t_d, q_d = np.inf, None
            for _ in range(min(reps, 3)):
                t0 = time.perf_counter()
                S = gcg.dense_restricted(eig, obs)
                cf = cho_factor(S, lower=True, overwrite_a=True,
                                check_finite=False)
                u = cho_solve(cf, a, check_finite=False)
                t_d = min(t_d, time.perf_counter() - t0)
                q_d = float(a @ u)
                del S, cf
            row.update(t_dense=t_d, q_dense=q_d,
                       rel=abs(q_cg - q_d) / abs(q_d))
        except MemoryError:
            print(f"  k={k}: dense out of memory")

    out.append(row)
    msg = f"k={k:2d}  N={n:7d}  m={m:6d}  CG {t_cg*1e3:9.2f} ms ({nit:5d} it)"
    if "t_dense" in row:
        msg += f"   dense {row['t_dense']*1e3:10.2f} ms   rel.diff {row['rel']:.2e}"
    print(msg, flush=True)

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
json.dump(out, open(args.out, "w"), indent=1)
print(f"wrote {args.out}\nnow run:  python _mkfig.py")
