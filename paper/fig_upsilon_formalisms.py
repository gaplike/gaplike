"""When does the linearised formalism reproduce the Godambe/White one?
(Fig. `upsilon_formalisms` in the paper.)

Each of the four covariance models quotes a scatter-to-width ratio Upsilon.
It can be computed twice:

  linearised     : one Newton step from the truth.  The bias is
                   b = F^-1 <score>, the scatter is the linearised MLE
                   variance, and Upsilon = sqrt(var + b^2) / (quoted width).
  Godambe/White  : the exact pseudo-true point lambda*, the minimizer of the
                   KL divergence of the model family from the truth, and the
                   sandwich covariance H^-1 J H^-1 about it.

The two are the same object evaluated at two different points, so they agree
exactly when the pseudo-true point does not move: lambda* = lambda_true.  That
happens whenever the model reproduces the TRUE diagonal of the windowed
covariance, because then the expected score vanishes term by term.  It is a
property of the model, not of the gap pattern -- which is what this figure
shows, by walking the gap pattern from one 2 h gap to the 57-gap comb of
scenario C while keeping the missing time fixed.

Away from that point the disagreement is set by how far lambda* has moved:
to leading order the relative error of the linearised formalism is
(ln 10 / 2) |lambda* - lambda_true| in dex.

Noise sector only -- no waveform, no lisabeta, no chains.  `build_truth` still
wants a `dh` array for the signal block, so it gets a dummy column whose
outputs are never read.

    python fig_upsilon_formalisms.py           # ~6 min -> results/ + figures/
    python fig_upsilon_formalisms.py --plot    # redraw from the cached JSON
"""
import argparse
import json
import os

import numpy as np
from scipy.optimize import minimize

import core as C
import diagnostics as D

LN10 = np.log(10.0)
MISSING_H = 2.0                  # total missing time, fixed across the sweep
NGAPS = list(range(1, 58))       # 57 = the scenario-C comb
TIERS = ("full", "diag", "bare", "psd")
RTOL_RANK, RCOND_DIAG = 1e-8, 1e-13
NG_MAX_PLOT = 32

COL = {"full": "#d62728", "diag": "#1f77b4", "bare": "#2ca02c", "psd": "#9467bd"}
LAB = {"full": "Full covariance", "diag": "Convolved diagonal",
       "bare": r"Scalar $W_c$ correction", "psd": "Raw PSD"}


# ------------------------------------------------------ noise-sector scenario
def build(gaps, taper_h):
    """The four model tiers on a gapped 12 h segment, noise parameters only."""
    weff, _ = C.make_weff(gaps, taper_h)
    beta_eff = float(np.sum(weff ** 2) / C.M)
    CT = C.freq_cov(weff, C.S2_TM, C.idx_band)
    CO = C.freq_cov(weff, C.S2_OMS, C.idx_band)
    S_ref = float(np.median(np.real(np.diag(CT + CO))))
    CTn, COn = CT / S_ref, CO / S_ref

    dh = np.ones((C.NCH, C.NB, 1), complex)          # dummy signal block
    truth = D.build_truth([CTn, COn], dh, rcond=RTOL_RANK, nch=C.NCH)
    comps = {
        "full": [CTn, COn],
        "diag": [np.real(np.diag(CTn)), np.real(np.diag(COn))],
        "bare": [beta_eff * C.STM_BARE / S_ref, beta_eff * C.SOMS_BARE / S_ref],
        "psd": [C.STM_BARE / S_ref, C.SOMS_BARE / S_ref],
    }
    rcond_of = {"full": RTOL_RANK}
    mods = {k: D.ApproxModel(v, dh, truth, rcond=rcond_of.get(k, RCOND_DIAG),
                             nch=C.NCH) for k, v in comps.items()}
    return dict(truth=truth, comps=comps, models=mods)


# -------------------------------------------- high-precision pseudo-true point
def kl_pseudo_true_precise(v1, v2, qd):
    """argmin_x  sum log v + sum q/v  with  v = 10^x0 v1 + 10^x1 v2, by Newton.

    `diagnostics.kl_pseudo_true_diag` uses Nelder-Mead with xatol = 1e-7, which
    is ample for the paper's cases (|lambda*| ~ 0.1-1) but would leave the
    `diag` tier's lambda* = 0 unresolved -- and that null result is the whole
    point of the lower panel.  The objective and the sandwich formulas are the
    paper's, unchanged; only the optimiser is sharpened.
    """
    v1, v2, qd = (np.real(np.asarray(a)) for a in (v1, v2, qd))

    def fgh(x):
        p = 10.0 ** np.asarray(x)
        v = p[0] * v1 + p[1] * v2
        dv = [LN10 * p[0] * v1, LN10 * p[1] * v2]
        f = float(np.sum(np.log(v)) + np.sum(qd / v))
        w1 = (v - qd) / v ** 2
        w2 = (2.0 * qd - v) / v ** 3
        g = np.array([float(np.sum(dv[a] * w1)) for a in range(2)])
        H = np.array([[float(np.sum(dv[a] * dv[b] * w2))
                       + (LN10 * float(np.sum(dv[a] * w1)) if a == b else 0.0)
                       for b in range(2)] for a in range(2)])
        return f, g, H

    res = minimize(lambda x: fgh(x)[0], np.zeros(2), jac=lambda x: fgh(x)[1],
                   hess=lambda x: fgh(x)[2], method="trust-exact",
                   options=dict(gtol=1e-14, maxiter=500))
    # trust-exact stalls around |g| ~ 1e-5; a few plain Newton steps from there
    # reach machine precision, which is what the null result needs.
    x, step = res.x.copy(), np.full(2, np.inf)
    for _ in range(20):
        _, g, H = fgh(x)
        step = np.linalg.solve(H, g)
        x = x - step
        if np.max(np.abs(step)) < 1e-15:
            break
    return x, float(np.max(np.abs(step)))


def sandwich_at(v1, v2, Q, lam_star, nch):
    """`diagnostics.sandwich_diag`, evaluated at a pseudo-true point supplied
    from outside rather than at its own Nelder-Mead one."""
    v1, v2 = np.real(v1), np.real(v2)
    qd = np.real(np.diag(Q))
    var = 10.0 ** lam_star[0] * v1 + 10.0 ** lam_star[1] * v2
    dS = [LN10 * 10.0 ** lam_star[0] * v1, LN10 * 10.0 ** lam_star[1] * v2]
    A = [d / var ** 2 for d in dS]
    absQ2 = np.abs(Q) ** 2
    J = nch * np.array([[float(A[a] @ (absQ2 @ A[b])) for b in range(2)]
                        for a in range(2)])

    def nll_exp(x):
        v = 10.0 ** x[0] * v1 + 10.0 ** x[1] * v2
        return nch * float(np.sum(np.log(v)) + np.sum(qd / v))

    h = 1e-4
    H = np.zeros((2, 2))
    for i in range(2):
        for j in range(2):
            pp = lam_star.copy(); pp[i] += h; pp[j] += h
            pm = lam_star.copy(); pm[i] += h; pm[j] -= h
            mp = lam_star.copy(); mp[i] -= h; mp[j] += h
            mm = lam_star.copy(); mm[i] -= h; mm[j] -= h
            H[i, j] = (nll_exp(pp) - nll_exp(pm)
                       - nll_exp(mp) + nll_exp(mm)) / (4 * h * h)
    H = 0.5 * (H + H.T)
    Hi = np.linalg.inv(H)
    return Hi @ J @ Hi


# ------------------------------------------------------------------ the sweep
def sweep(out_path):
    out = []
    for n_g in NGAPS:
        dur = MISSING_H / n_g
        gaps = [((k + 0.5) * C.SEG_HOURS / n_g, dur) for k in range(n_g)]
        sc = build(gaps, 0.3 * dur)
        Q = sc["truth"]["Q"]
        qd = np.real(np.diag(Q))
        rec = dict(n_gaps=n_g, dur_min=dur * 60.0, tiers={})
        for tk in TIERS:
            m = sc["models"][tk]
            w = np.sqrt(np.diag(m.cov_n))
            if tk == "full":                       # correct by construction
                lam_star, sand = np.zeros(2), m.variance_n
            else:
                v1, v2 = (np.real(np.asarray(c)) for c in sc["comps"][tk])
                lam_star, _ = kl_pseudo_true_precise(v1, v2, qd)
                sand = sandwich_at(v1, v2, Q, lam_star, C.NCH)
            rec["tiers"][tk] = dict(
                ups_lin=np.asarray(m.upsilon_n).tolist(),
                ups_sw=(np.sqrt(np.abs(np.diag(sand)) + np.asarray(lam_star) ** 2)
                        / w).tolist(),
                lam_star=np.asarray(lam_star).tolist(),
                bias_lin=np.asarray(m.bias_n).tolist())
        out.append(rec)
        print(f"n_g={n_g:3d}  gap={dur * 60:6.2f} min  " + "  ".join(
            f"{t}: {np.max(rec['tiers'][t]['ups_lin']):9.2f}/"
            f"{np.max(rec['tiers'][t]['ups_sw']):8.2f}" for t in TIERS), flush=True)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(out, open(out_path, "w"), indent=1)
    print(f"\nwrote {out_path}")
    return out


# ----------------------------------------------------------------- the figure
def draw(records, stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    plt.rcParams.update({
        "text.usetex": True, "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.size": 8, "axes.labelsize": 8.5, "legend.fontsize": 6.6,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.7,
        "xtick.direction": "in", "ytick.direction": "in", "ytick.right": True,
        "xtick.major.width": 0.7, "ytick.major.width": 0.7,
        "xtick.minor.width": 0.5, "ytick.minor.width": 0.5,
        "legend.frameon": True, "legend.framealpha": 0.95,
        "legend.edgecolor": "0.7",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })

    d = [r for r in records if r["n_gaps"] <= NG_MAX_PLOT]
    ng = np.array([r["n_gaps"] for r in d], float)

    def ups(tk, key):
        return np.array([np.max(r["tiers"][tk][key]) for r in d])

    fig, (hi, lo) = plt.subplots(2, 1, figsize=(3.9, 3.5), sharex=True,
                                 gridspec_kw=dict(height_ratios=[1.85, 1],
                                                  hspace=0.09))
    for ax in (hi, lo):
        ax.set_xscale("log")
        for tk in TIERS:
            ax.plot(ng, ups(tk, "ups_sw"), "-", color=COL[tk], lw=2.8,
                    alpha=0.9, zorder=3, label=LAB[tk] if ax is hi else None)
            ax.plot(ng, ups(tk, "ups_lin"), color=COL[tk], lw=0.9, zorder=4,
                    dashes=(4, 2.6))
            # open markers make coincidence unmistakable: where the two
            # formalisms agree the circles sit centred on the thick line
            ax.plot(ng[::3], ups(tk, "ups_lin")[::3], "o", mfc="white",
                    mec=COL[tk], ms=4.4, mew=1.1, zorder=5)

    hi.set_yscale("log")
    hi.set_ylim(7, 1.2e4)
    lo.set_ylim(0.93, 1.66)
    lo.set_yticks([1.0, 1.2, 1.4, 1.6])

    hi.spines["bottom"].set_visible(False)          # break the axis
    lo.spines["top"].set_visible(False)
    hi.tick_params(bottom=False, labelbottom=False)
    kw = dict(marker=[(-1, -0.55), (1, 0.55)], markersize=7, linestyle="none",
              color="k", mec="k", mew=0.8, clip_on=False)
    hi.plot([0, 1], [0, 0], transform=hi.transAxes, **kw)
    lo.plot([0, 1], [1, 1], transform=lo.transAxes, **kw)

    lo.set_xlabel(r"number of gaps $n_{\rm g}$")
    lo.set_xlim(0.92, NG_MAX_PLOT * 1.08)
    fig.supylabel(r"$\Upsilon$ \ (true scatter / quoted width)", x=0.015,
                  fontsize=8.5)

    h, _ = hi.get_legend_handles_labels()
    h += [Line2D([], [], color="0.35", lw=2.8, label=r"Godambe--White"),
          Line2D([], [], color="0.35", lw=0.9, dashes=(4, 2.6), marker="o",
                 mfc="white", mec="0.35", ms=4.4, mew=1.1, label=r"Linearised")]
    hi.legend(handles=h, loc="upper left", handlelength=1.8, borderpad=0.35,
              handletextpad=0.5, labelspacing=0.24)

    # secondary_xaxis with a reciprocal transform silently reverses the tick
    # labels; place them explicitly on a twin axis instead.
    t = hi.twiny()
    t.set_xscale("log")
    t.set_xlim(hi.get_xlim())
    ticks = [1, 2, 4, 8, 16, 32]
    t.set_xticks(ticks)
    t.set_xticklabels([f"{MISSING_H * 60 / n:.0f}" if MISSING_H * 60 / n >= 10
                       else f"{MISSING_H * 60 / n:.1f}" for n in ticks])
    t.set_xlabel(r"gap duration \ [min]", labelpad=4)
    t.tick_params(direction="in", labelsize=7)
    t.minorticks_off()

    os.makedirs(os.path.dirname(stem) or ".", exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}.{ext}", dpi=300)
    print(f"wrote {stem}.pdf / .png")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--plot", action="store_true",
                   help="skip the sweep, redraw from the cached JSON")
    p.add_argument("--out", default="results/upsilon_formalisms.json")
    p.add_argument("--fig", default="figures/upsilon_formalisms")
    args = p.parse_args()
    recs = json.load(open(args.out)) if args.plot else sweep(args.out)
    draw(recs, args.fig)
