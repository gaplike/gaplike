"""Two preconditioners for the matrix-free time-domain likelihood, compared as
the same missing time is chopped into more and more gaps (Fig. `precond_compare`
in the paper).

`gaplike.cg` preconditions with the inverse of the *gap-free* (Whittle)
covariance, applied by the same FFT pair.  Ref. [Baghi et al., PRD 93, 122007
(2016)] instead tapers the time-domain autocovariance to zero beyond a lag L
and factorizes the resulting banded matrix.  The two are complementary, and
which one wins is a property of the gap pattern, not of the record:

  circulant : exact where there are no gaps, so its residual is the gap
              geometry -- iterations grow with the NUMBER of gaps.
  sparse    : built from the true restricted covariance, so it knows the holes
              exactly, but discards every correlation beyond lag L.  That cut
              only costs it when a surviving segment is longer than L -- and
              when the gaps are close together, none of them are.  Iterations
              are a function of (segment length)/L alone.

Sweep: N = 8192 at dt = 15 s, a fixed 15% of the record missing, chopped into
n_g equal gaps evenly spaced, n_g = 1 ... 256.  The paper's scenarios sit at
n_g = 2 (A), 12 (B) and 57 (C).  Depends only on `gaplike` (+ numpy/scipy/
matplotlib); no waveform code, no chains.

Panel (b) prices one likelihood call at a NEW set of noise parameters, since
Sigma_OO depends on lambda and nothing survives between MCMC proposals:

  circulant : recombine the circulant eigenvalues   O(N)      + CG iterations
  sparse    : recombine the cached component bands,
              then a banded Cholesky                O(m L^2)  + CG iterations

The per-component band arrays do NOT depend on lambda (gamma is linear in
10^lambda), so they are built once and excluded from the per-call cost;
counting them would overstate the sparse route.

    python fig_precond_compare.py            # ~4 min -> results/ + figures/
    python fig_precond_compare.py --plot     # redraw from the cached JSON
"""
import argparse
import inspect
import json
import os
import time

import numpy as np
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.sparse.linalg import LinearOperator, cg as _scipy_cg

from gaplike import cg as gcg
from gaplike import psd

DT = 15.0
N = 8192
DUTY = 0.85                     # 15% of the record missing, at every n_g
NGAPS = (1, 2, 4, 8, 16, 32, 64, 128, 256)
LS = (32, 256)                  # taper supports, in samples
RTOL = 1e-8
REPEAT = 5                      # timings are the MINIMUM over repeats

_TOL_KW = "rtol" if "rtol" in inspect.signature(_scipy_cg).parameters else "tol"


# --------------------------------------------------------------- gap patterns
def chopped_mask(n, n_gaps, duty):
    """`n_gaps` equal gaps, evenly spaced, removing exactly (1-duty) of `n`.

    The remainder of the integer division is distributed over the gaps so that
    the total missing time is identical at every `n_gaps`; letting it round
    down inflates the duty cycle at large `n_gaps` and flatters precisely the
    finely-chopped end of the sweep.
    """
    miss = int(round((1.0 - duty) * n))
    base, rem = divmod(miss, n_gaps)
    per = n // n_gaps
    m = np.ones(n, bool)
    for k in range(n_gaps):
        g = base + (1 if k < rem else 0)
        s = np.clip(k * per + (per - g) // 2, 0, n - g)
        m[s:s + g] = False
    return m


# ------------------------------------------------------- sparse preconditioner
def hann_lag_taper(n, L):
    """Taper on the autocovariance lag: 1 at zero lag, 0 beyond lag L.

    Positive-definiteness of the tapered autocovariance is not automatic.  By
    the Schur product theorem it is guaranteed if the taper is itself a
    positive-definite function of the lag: the Bartlett taper (the Fejer
    kernel) qualifies, and never failed here, but needs 4-10x more iterations.
    The Hann taper used below is not PD as a function -- the minimum of its
    lag-spectrum is -3.4 -- yet the banded Cholesky succeeded for every pattern
    and every L tried in this paper.  Hard truncation (no taper) has spectral
    minimum -55.8 and does fail: it is not positive definite for 3 of the 4
    patterns tested.  The Cholesky is the check; fall back to Bartlett if it
    raises.
    """
    lag = np.arange(n)
    return np.where(lag <= L, 0.5 * (1.0 + np.cos(np.pi * lag / L)), 0.0)


def component_bands(gammas, obs, L, n):
    """Banded (upper) form of the tapered restricted covariance, per component.

    Banded in OBSERVED-index space with half-bandwidth L, but the
    autocovariance is evaluated at the TRUE lag ``obs[i] - obs[j]``: the
    preconditioner knows where the holes are.  Independent of lambda, so this
    is built once and reused at every noise state.
    """
    w = hann_lag_taper(n, L)
    out = []
    for g in gammas:
        gt = g * w
        ab = np.zeros((L + 1, obs.size))
        ab[L, :] = gt[0]
        for d in range(1, L + 1):
            lag = obs[d:] - obs[:-d]           # >= d, because of the gaps
            ab[L - d, d:] = np.where(lag < n, gt[np.minimum(lag, n - 1)], 0.0)
        out.append(ab)
    return out


# --------------------------------------------------------------------- timing
def timeit(fn, repeat=REPEAT):
    ts = []
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn()
        ts.append(time.perf_counter() - t0)
    return out, float(np.min(ts)) * 1e3


def cg_run(A, b, M, maxiter=6000):
    it = [0]
    x, info = _scipy_cg(A, b, M=M, maxiter=maxiter,
                        callback=lambda xk: it.__setitem__(0, it[0] + 1),
                        atol=0.0, **{_TOL_KW: RTOL})
    if info != 0:
        raise RuntimeError(f"CG did not reach rtol={RTOL} (info={info})")
    return it[0]


# ----------------------------------------------------------------- the sweep
def sweep(out_path):
    comps = psd.lisa_tdi2_ae()
    eig_k = [psd.as_two_sided(comps[k], N, DT) / (2.0 * DT) for k in ("tm", "oms")]
    gam_k = [np.real(np.fft.ifft(e)) for e in eig_k]
    lam = np.zeros(2)                          # truth; any lambda times the same

    rows = []
    hdr = f"{'n_g':>5} {'seg':>6} |{'circ it':>9}{'circ ms':>9} |"
    for L in LS:
        hdr += f"{f'L={L} it':>10}{'chol':>8}{'solve':>8}{'total':>8} |"
    print(hdr, flush=True)

    for ng in NGAPS:
        obs = np.flatnonzero(chopped_mask(N, ng, DUTY))
        m = obs.size
        b = np.random.default_rng(0).standard_normal(m)
        A = gcg.sigma_oo(10.0 ** lam[0] * eig_k[0] + 10.0 ** lam[1] * eig_k[1], obs)

        # ---- circulant (Whittle): gaplike's default ----------------------
        eig, ms_eig = timeit(
            lambda: 10.0 ** lam[0] * eig_k[0] + 10.0 ** lam[1] * eig_k[1])
        Mc = gcg.whittle_preconditioner(eig, obs, floor_rel=1e-6)
        it_c, ms_c = timeit(lambda: cg_run(A, b, Mc))
        row = dict(n_gaps=ng, m=int(m), seg=m / ng, circ_it=it_c,
                   circ_ms=ms_eig + ms_c, band={})
        line = f"{ng:5d} {m/ng:6.0f} |{it_c:9d}{ms_eig + ms_c:9.1f} |"

        # ---- sparse tapered, two supports --------------------------------
        for L in LS:
            abk = component_bands(gam_k, obs, L, N)      # cached, not timed
            def build_chol():
                return cholesky_banded(
                    10.0 ** lam[0] * abk[0] + 10.0 ** lam[1] * abk[1],
                    lower=False)
            try:
                c, ms_chol = timeit(build_chol)
            except np.linalg.LinAlgError:                # taper not PD here
                row["band"][str(L)] = None
                line += f"{'NOT PD':>10}{'':>24} |"
                continue
            Mb = LinearOperator((m, m), dtype=float,
                                matvec=lambda v, c=c: cho_solve_banded((c, False), v))
            it_b, ms_b = timeit(lambda: cg_run(A, b, Mb))
            row["band"][str(L)] = dict(it=it_b, chol_ms=ms_chol, solve_ms=ms_b,
                                       total_ms=ms_chol + ms_b)
            line += f"{it_b:10d}{ms_chol:8.1f}{ms_b:8.1f}{ms_chol + ms_b:8.1f} |"

        print(line, flush=True)
        rows.append(row)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(rows, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")
    return rows


# --------------------------------------------------------------- the figure
def draw(rows, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 8, "axes.labelsize": 8.5, "axes.titlesize": 8.5,
        "legend.fontsize": 6.8, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "lines.markeredgewidth": 0.6,
        "legend.frameon": True, "legend.framealpha": 0.93,
        "legend.edgecolor": "0.7",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    CIRC = "#0d3b78"
    BCOL, BMK = {32: "#e8853a", 256: "#8c2d04"}, {32: "s", 256: "^"}

    ng = np.array([r["n_gaps"] for r in rows], float)
    cit = np.array([r["circ_it"] for r in rows], float)
    cms = np.array([r["circ_ms"] for r in rows], float)

    def band(L, key):
        return np.array([r["band"][str(L)][key] for r in rows], float)

    fig, ax = plt.subplots(1, 2, figsize=(7.0, 2.95))

    a = ax[0]
    a.loglog(ng, cit, "o-", color=CIRC, lw=1.4, ms=3.6, zorder=3,
             label=r"circulant (Whittle) \textemdash\ \texttt{gaplike} default")
    for L in LS:
        a.loglog(ng, band(L, "it"), BMK[L] + "-", color=BCOL[L], lw=1.4, ms=3.6,
                 zorder=3, label=rf"sparse tapered, $L={L}$")
    a.set_xlabel(r"number of gaps $n_{\rm g}$")
    a.set_ylabel(r"CG iterations to $\|r\|/\|b\| < 10^{-8}$")
    a.set_xlim(0.75, 420)
    a.set_ylim(11, 3000)
    a.legend(loc="upper left", handlelength=1.5, borderpad=0.4,
             handletextpad=0.5, labelspacing=0.3)
    a.set_title(r"(a)\ \ iteration count", loc="left")

    a = ax[1]
    a.loglog(ng, cms, "o-", color=CIRC, lw=1.4, ms=3.6, zorder=3)
    for L in LS:
        a.loglog(ng, band(L, "total_ms"), BMK[L] + "-", color=BCOL[L], lw=1.4,
                 ms=3.6, zorder=3)
    a.set_xlabel(r"number of gaps $n_{\rm g}$")
    a.set_ylabel(r"cost of one likelihood call \ [ms]")
    a.set_xlim(0.75, 420)
    a.set_ylim(8, 900)
    a.set_title(r"(b)\ \ cost per likelihood call", loc="left")

    fig.tight_layout(w_pad=1.4)
    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    print(f"wrote {stem}.pdf / .png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true",
                   help="skip the sweep, redraw from the cached JSON")
    p.add_argument("--out", default="results/precond_compare.json")
    p.add_argument("--fig", default="figures/precond_compare")
    args = p.parse_args()
    rows = json.load(open(args.out)) if args.plot else sweep(args.out)
    draw(rows, args.fig)
