"""Figures: overview panels and PE-vs-analytic corner plots (13 parameters)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import corner
import emcee as _emcee

import core as C

plt.rcParams.update({
    "font.family": "serif", "mathtext.fontset": "cm", "text.usetex": False,
    "font.size": 15, "axes.labelsize": 18, "axes.titlesize": 20,
    "legend.fontsize": 14, "xtick.labelsize": 14, "ytick.labelsize": 14,
    "figure.dpi": 110,
})

MCOL = {"full": "tab:red", "diag": "tab:blue", "bare": "tab:green", "psd": "tab:purple"}
MNAME = {"full": "Full covariance", "diag": "Convolved diagonal",
         "bare": r"Scalar $W_c$ correction", "psd": "Raw PSD"}
MODELS = ["full", "diag", "bare", "psd"]

LAB_TEX = [r"$\Delta M_{\rm tot}/10^{6}$", r"$\Delta q$",
           r"$\Delta\chi_1$", r"$\Delta\chi_2$", r"$\Delta\log_{10}d_L$",
           r"$\Delta\iota$", r"$\Delta\varphi$", r"$\Delta\lambda$",
           r"$\Delta\beta$", r"$\Delta\psi$", r"$\Delta t_c$ [s]",
           r"$\lambda_{\rm tm}$", r"$\lambda_{\rm oms}$"]
# display deltas: Mtot in 1e6 Msun, t_c (segment fraction) in seconds
SCALE = np.array([1e-6, 1, 1, 1, 1, 1, 1, 1, 1, 1, C.T_OBS, 1, 1])

BLOCKS = {
    "signal": list(range(11)),
    "noise": [11, 12],
    "key": [0, 1, 2, 10, 11, 12],
    "joint": list(range(13)),
}


def _delta(x, x_true):
    return (x - x_true[None, :]) * SCALE[None, :]


def load_chains(skey, resdir="results", dlogl=60.0):
    """Load chains; drop stranded-walker samples with logL < max - dlogl.
    (Isolated walkers can get permanently stuck at catastrophically low logL —
    a known stretch-move pathology; the cut is far outside the posterior spread,
    which is ~chi2(13)/2 ~ 7 +- 4 nats.)"""
    out = {}
    for mk in MODELS:
        z = np.load(f"{resdir}/{skey}_{mk}.npz")
        keep = z["log_prob"] > z["log_prob"].max() - dlogl
        out[mk] = {"chain": z["chain"][keep], "lo": z["lo"], "hi": z["hi"],
                   "acc": float(z["acc"]), "frac_cut": float(1.0 - keep.mean())}
    return out


def gauss_samples(model, resid, x_true, lo, hi, n=60000, seed=3,
                  noise_override=None):
    """Analytic prediction: the Hessian-based-likelihood posterior — a Gaussian with
    the model Fisher covariance, centred on the linearized MLE, TRUNCATED by the same
    uniform prior box used in the PE. Sampled with a fast emcee run (the density is
    a 13-d quadratic form, so this costs seconds).

    noise_override=(mle, cov): replace the noise-sector centre/covariance with the
    EXACT (nonlinear, Godambe--White regime) profile MLE and its curvature — used
    for the non-convolved models, whose linearized noise MLEs leave their domain
    of validity (they overshoot and pile at the prior boundary)."""
    ns = C.NSIG
    mean = x_true.copy()
    mean[:ns] = mean[:ns] + model.mle_signal(resid)
    cov = np.zeros((ns + 2, ns + 2))
    cov[:ns, :ns] = model.cov_s
    if noise_override is None:
        mean[ns:] = mean[ns:] + model.mle_noise(resid)
        cov[ns:, ns:] = model.cov_n
    else:
        mle_n, cov_n = noise_override
        mean[ns:] = np.asarray(mle_n)          # absolute (noise truth = 0)
        cov[ns:, ns:] = np.asarray(cov_n)
    P = np.linalg.inv(cov)

    def logp(x):
        if np.any(x < lo) or np.any(x > hi):
            return -np.inf
        d = x - mean
        return -0.5 * float(d @ P @ d)

    ndim = ns + 2
    nw = 64
    rng = np.random.default_rng(seed)
    sig = np.sqrt(np.diag(cov))
    x0 = np.clip(mean, lo + 0.03 * (hi - lo), hi - 0.03 * (hi - lo))
    scat = np.clip(0.5 * sig, 0.02 * (hi - lo), 0.25 * (hi - lo))
    p0 = rng.uniform(np.maximum(lo + 1e-9 * (hi - lo), x0 - scat),
                     np.minimum(hi - 1e-9 * (hi - lo), x0 + scat), size=(nw, ndim))
    sam = _emcee.EnsembleSampler(nw, ndim, logp)
    state = sam.run_mcmc(p0, 600, progress=False)
    sam.reset()
    sam.run_mcmc(state, int(np.ceil(n / nw)) + 50, progress=False)
    return sam.get_chain(flat=True)[:n]


def corner_block(sc, chains, x_true, block, fname, models=MODELS, title=None,
                 n_gauss=60000, exact_noise_mle=None, noise_pred=None):
    """PE posteriors (solid) vs truncated analytic Gaussians (dashed).
    For block='noise', pass exact_noise_mle={model: (mle, cov)} to overlay the
    exact profile-MLE Gaussians (dotted) as the nonlinear analytic reference.
    noise_pred={model: (mle, cov)}: use the exact (nonlinear/Godambe--White)
    profile MLE + curvature as the noise-sector centre of the DASHED analytic
    prediction for those models (signal sector stays linearized)."""
    cols = BLOCKS[block]
    labels = [LAB_TEX[i] for i in cols]
    noise_pred = noise_pred or {}

    ga = {}
    for mk in models:
        rz = chains[mk]
        ga[mk] = gauss_samples(sc["models"][mk], sc["resid"], x_true,
                               rz["lo"], rz["hi"], n=n_gauss,
                               noise_override=noise_pred.get(mk))

    gexa = {}
    if exact_noise_mle and block == "noise":
        rng0 = np.random.default_rng(11)
        for mk in models:
            mle, cvn = exact_noise_mle[mk]
            gexa[mk] = x_true[None, 11:] + rng0.multivariate_normal(mle, cvn, size=n_gauss)

    allsets = []
    for mk in models:
        allsets.append(_delta(np.asarray(chains[mk]["chain"]), x_true)[:, cols])
        allsets.append(_delta(ga[mk], x_true)[:, cols])
    # quantile-based ranges: robust to stray walker islands with <~0.1% weight
    lo_r = np.min([np.quantile(a, 0.001, axis=0) for a in allsets], axis=0)
    hi_r = np.max([np.quantile(a, 0.999, axis=0) for a in allsets], axis=0)
    pad = 0.08 * (hi_r - lo_r)
    rng_plot = [(l - p, h + p) for l, h, p in zip(lo_r, hi_r, pad)]

    fig = None
    for mk in models:
        fig = corner.corner(_delta(ga[mk], x_true)[:, cols], color=MCOL[mk], fig=fig,
                            range=rng_plot, plot_datapoints=False, plot_density=False,
                            no_fill_contours=True, levels=(0.68, 0.95), smooth=1.0,
                            hist_kwargs={"density": True, "ls": "--", "lw": 1.1, "alpha": .85},
                            contour_kwargs={"linestyles": "--", "linewidths": 1.1, "alpha": .85},
                            labels=labels, truths=np.zeros(len(cols)), truth_color="0.35",
                            label_kwargs={"fontsize": 12}, max_n_ticks=3)
    for mk in gexa:
        fig = corner.corner(gexa[mk] - x_true[None, 11:], color=MCOL[mk], fig=fig,
                            range=rng_plot, plot_datapoints=False, plot_density=False,
                            no_fill_contours=True, levels=(0.68, 0.95), smooth=1.0,
                            hist_kwargs={"density": True, "ls": ":", "lw": 1.5},
                            contour_kwargs={"linestyles": ":", "linewidths": 1.5},
                            labels=labels, truths=np.zeros(len(cols)), truth_color="0.35",
                            label_kwargs={"fontsize": 19}, max_n_ticks=3)
    for mk in models:
        d = _delta(np.asarray(chains[mk]["chain"]), x_true)[:, cols]
        fig = corner.corner(d, color=MCOL[mk], fig=fig, range=rng_plot,
                            plot_datapoints=False, plot_density=False,
                            no_fill_contours=True, levels=(0.68, 0.95), smooth=1.0,
                            hist_kwargs={"density": True, "lw": 1.6},
                            contour_kwargs={"linewidths": 1.6},
                            labels=labels, truths=np.zeros(len(cols)), truth_color="0.35",
                            label_kwargs={"fontsize": 19}, max_n_ticks=3)

    handles = [Line2D([0], [0], color=MCOL[mk], lw=2, label=MNAME[mk]) for mk in models]
    if noise_pred:
        lab_dash = ("Analytic prediction: linearized MLE (signal);\n"
                    "exact profile MLE + curvature in the\n"
                    r"Godambe–White regime ($W_c$ correction and raw PSD)")
    else:
        lab_dash = ("Analytic prediction: truncated Gaussian\n"
                    "at the linearized MLE")
    handles += [Line2D([0], [0], color="0.25", lw=1.8, label="PE (emcee)"),
                Line2D([0], [0], color="0.25", lw=1.4, ls="--", label=lab_dash)]
    if gexa:
        handles.append(Line2D([0], [0], color="0.25", lw=1.6, ls=":",
                              label="Analytic: exact profile MLE (nonlinear)"))
    if len(cols) <= 3:
        # small corner: legend to the right of the grid, outside the axes
        fig.legend(handles=handles, loc="center left", fontsize=15,
                   bbox_to_anchor=(0.99, 0.72), frameon=False)
    else:
        # large corner: legend in the empty upper triangle, clear of all axes
        fig.legend(handles=handles, loc="upper right", fontsize=16,
                   bbox_to_anchor=(0.985, 0.975), frameon=False)
    if title:
        fig.suptitle(title, fontsize=22, x=0.44, y=1.004)
    fig.savefig(fname, dpi=(150 if len(cols) > 8 else 200), bbox_inches="tight")
    fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return fname


PL_TEX = [r"$M_{\rm tot}$", r"$q$", r"$\chi_1$", r"$\chi_2$", r"$\log_{10}d_L$",
          r"$\iota$", r"$\varphi$", r"$\lambda$", r"$\beta$", r"$\psi$", r"$t_c$",
          r"$\lambda_{\rm tm}$", r"$\lambda_{\rm oms}$"]


def fig_upsilon_xi(scs, sandwiches=None, fname="figures/upsilon_xi.png"):
    """Upsilon / Xi diagnostics, paper-style: leading-order values only,
    minimal titles, CM-mathtext fonts, legend outside the axes."""
    rc = {"font.family": "serif", "mathtext.fontset": "cm",
          "font.size": 15, "axes.titlesize": 21, "axes.labelsize": 19,
          "xtick.labelsize": 16, "ytick.labelsize": 14}
    with plt.rc_context(rc):
        keys = list(scs)
        fig, axes = plt.subplots(len(keys), 2, figsize=(14.2, 4.3 * len(keys)),
                                 sharex=True)
        if len(keys) == 1:
            axes = axes[None, :]
        xpos = np.arange(13)
        off = {"full": -0.27, "diag": -0.09, "bare": 0.09, "psd": 0.27}
        for r, key in enumerate(keys):
            sc = scs[key]
            axU, axX = axes[r]
            for ax in (axU, axX):
                ax.axvspan(10.5, 12.6, color="tab:blue", alpha=.06)
                ax.axvline(10.5, color="0.55", lw=1.0)
                ax.axhline(1.0, color="0.4", ls="--", lw=1.1)
                ax.set_xticks(xpos)
                ax.set_xticklabels(PL_TEX, rotation=30, ha="right",
                                   rotation_mode="anchor")
                ax.set_xlim(-0.6, 12.6)
                ax.grid(alpha=.22, axis="y")
            for mk in [k for k in MODELS if k in sc["models"]]:
                m = sc["models"][mk]
                ups = np.concatenate([m.upsilon_s, m.upsilon_n])
                xi = np.concatenate([m.xi_s, m.xi_n])
                axU.plot(xpos + off[mk], ups, "o", color=MCOL[mk], ms=8)
                axX.plot(xpos + off[mk], xi, "o", color=MCOL[mk], ms=8)
            axU.set_yscale("log")
            axU.set_ylabel(f"{key}")
            if r == 0:
                axU.set_title(r"$\Upsilon_A$")
                axX.set_title(r"$\Xi_A$")
        handles = [Line2D([0], [0], color=MCOL[mk], lw=0, marker="o", ms=10,
                          label=MNAME[mk]) for mk in MODELS]
        leg = fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=15,
                         frameon=False, bbox_to_anchor=(0.5, 1.045),
                         handletextpad=0.15, columnspacing=1.1)
        fig.tight_layout()
        fig.savefig(fname, dpi=190, bbox_inches="tight", bbox_extra_artists=(leg,))
        fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight",
                    bbox_extra_artists=(leg,))
        plt.close(fig)
    return fname


def fig_upsxi_heatmap(scs, sandwiches, fname="figures/upsilon_xi_heatmap.png"):
    """Upsilon / Xi as heatmaps: models (rows) x parameters (columns), one
    (Upsilon, Xi) pair per scenario. Colors: log10 of the value, diverging around
    the honest value 1. Noise-sector Upsilon cells show the nonlinear sandwich
    value (annotated), with the paper's leading-order number in parentheses."""
    import matplotlib.colors as mcolors
    keys = list(scs)
    fig, axes = plt.subplots(2 * len(keys), 1,
                             figsize=(13.8, 2.35 * 2 * len(keys)))
    axes = np.atleast_1d(axes)
    cmap = plt.get_cmap("coolwarm")
    normU = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-0.35, vmax=2.35)
    normX = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-0.14, vmax=0.20)

    def _annot(ax, Mat, norm, dual=None):
        for i in range(Mat.shape[0]):
            for j in range(Mat.shape[1]):
                v = 10.0**Mat[i, j]
                rgba = cmap(norm(Mat[i, j]))
                lum = 0.299 * rgba[0] + 0.587 * rgba[1] + 0.114 * rgba[2]
                col = "k" if lum > 0.55 else "w"
                txt = f"{v:.2f}" if v < 9.95 else f"{v:.0f}"
                if dual is not None and j >= 11:
                    vl = dual[i, j - 11]
                    tl = f"{vl:.2f}" if vl < 9.95 else f"{vl:.0f}"
                    ax.text(j, i - 0.18, txt, ha="center", va="center",
                            color=col, fontsize=9, fontweight="bold")
                    ax.text(j, i + 0.26, f"({tl})", ha="center", va="center",
                            color=col, fontsize=7)
                else:
                    ax.text(j, i, txt, ha="center", va="center", color=col,
                            fontsize=9)

    for r, key in enumerate(keys):
        sc = scs[key]
        U = np.zeros((4, 13))
        Ulin_noise = np.zeros((4, 2))
        X = np.zeros((4, 13))
        for i, mk in enumerate(MODELS):
            m = sc["models"][mk]
            ls_, Var = sandwiches[key][mk]
            ups_sw = np.sqrt(np.diag(Var) + ls_**2) / np.sqrt(np.diag(m.cov_n))
            U[i] = np.log10(np.concatenate([m.upsilon_s, ups_sw]))
            Ulin_noise[i] = m.upsilon_n
            X[i] = np.log10(np.concatenate([m.xi_s, m.xi_n]))

        axU, axX = axes[2 * r], axes[2 * r + 1]
        imU = axU.imshow(U, cmap=cmap, norm=normU, aspect="auto")
        imX = axX.imshow(X, cmap=cmap, norm=normX, aspect="auto")
        _annot(axU, U, normU, dual=Ulin_noise)
        _annot(axX, X, normX)
        for ax, tag in ((axU, r"$\Upsilon_A$"), (axX, r"$\Xi_A$")):
            ax.set_yticks(range(4))
            ax.set_yticklabels(["full", "diag", "bare", "psd"], fontsize=15)
            ax.set_xticks(range(13))
            ax.set_xticklabels(PL_TEX, fontsize=16)
            ax.axvline(10.5, color="k", lw=1.2)
            ax.set_title(f"{key} — {tag}", fontsize=18, loc="left")
        fig.colorbar(imU, ax=axU, pad=0.012, fraction=0.05,
                     ticks=[np.log10(v) for v in (0.5, 1, 3, 10, 30, 100)],
                     format=lambda x, _: f"{10**x:g}")
        fig.colorbar(imX, ax=axX, pad=0.012, fraction=0.05,
                     ticks=[np.log10(v) for v in (0.8, 0.9, 1.0, 1.2, 1.5)],
                     format=lambda x, _: f"{10**x:g}")
    fig.tight_layout()
    fig.savefig(fname, dpi=190, bbox_inches="tight")
    fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    return fname


def fig_overview(scs, fname="figures/overview.png"):
    """Per scenario: TD lane in the fig01 style (gap bands, windowed data,
    MBHB visible: no-gap signal black + gapped signal red dashed, physical
    units); band spectra on the right. Legends outside the axes (top row)."""
    keys = [k for k in scs]
    nrow = len(keys)
    fig, axes = plt.subplots(nrow, 2, figsize=(14, 2.9 * nrow))
    if nrow == 1:
        axes = axes[None, :]
    th = np.arange(C.M) * C.DT / 3600.0
    F_LP = 1.0e-2
    # display low-pass (10 mHz): the MBHB carries 99% of its power below
    # ~1.6 mHz, while >99.9% of the TD noise variance sits in the loud
    # OMS/TDI2-transfer region above ~10 mHz -- excising it for DISPLAY
    # only makes the signal visible without touching any analysis.
    lp_of = {}
    for k in keys:
        wd = scs[k]["weff"] * scs[k]["d_td"][0]
        lp_of[k] = np.fft.irfft(np.where(C.freqs <= F_LP, np.fft.rfft(wd), 0),
                                n=C.M)
    # ONE vertical scale for all rows (so noise levels are comparable across
    # scenarios), as tight as the signal allows: the merger peak sets the
    # floor, and the displayed noise trace is cut at a high quantile instead
    # of its extreme value (~1% of samples clipped in the worst row).
    ym = 1.03 * max(max(np.quantile(np.abs(lp_of[k]), 0.95),
                        np.max(np.abs(scs[k]["h_td"][0]))) for k in keys)
    for i, k in enumerate(keys):
        sc = scs[k]
        axL, axR = axes[i]
        d = sc["d_td"][0]
        h0_td = sc["h_td"][0]                       # non-gapped MBHB (physical)
        # gap pattern as shaded bands (dense combs: fine vertical striping)
        gap = sc["gate"] < 0.5
        edges = np.diff(np.concatenate([[0], gap.astype(int), [0]]))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        if starts.size > 60:
            axL.vlines(th[gap], 0, 1, transform=axL.get_xaxis_transform(),
                       color="orange", alpha=0.16, lw=0.5, zorder=0)
        else:
            for s0, e0 in zip(starts, ends):
                axL.axvspan(s0 * C.DT / 3600.0, e0 * C.DT / 3600.0,
                            color="orange", alpha=0.30, lw=0)
        wd_lp = lp_of[k]
        axL.plot(th, wd_lp, color="gray", lw=.5, alpha=.5, zorder=1)
        axL.plot(th, h0_td, color="black", lw=1.5, zorder=3)
        axL.plot(th, sc["weff"] * h0_td, color="tab:red", lw=1.2, ls="--",
                 zorder=4)
        axL.axvline(C.THETA_SIG_TRUE[10] * C.SEG_HOURS, color="tab:blue",
                    lw=1.4, ls="--", zorder=2)
        axL.axhline(0, color="0.5", lw=.9, ls="--", zorder=2)
        axL.set_ylim(-ym, ym)
        axL.set_xlim(0, C.SEG_HOURS)
        axL.set_ylabel(r"TDI ($A$)", color="0.35")
        axL.text(.015, .96, sc["desc"] + "\n" + r"SNR$_w$(A+E) = %.0f" % sc["snr"],
                 transform=axL.transAxes, va="top", fontsize=13,
                 bbox=dict(fc="w", ec="0.8", alpha=.9))
        if i == nrow - 1:
            axL.set_xlabel(r"$t$ [h]")

        P = np.abs(np.fft.rfft(sc["weff"] * d))**2 * 2 / (C.FS * C.M)
        Sconv = C.conv_diag(sc["weff"], C.S2)
        axR.loglog(C.freqs[1:], P[1:], color="0.7", lw=.4)
        axR.loglog(C.freqs[1:], sc["beta_eff"] * C.S_SEG[1:], "k-", lw=1.5)
        axR.loglog(C.freqs[1:], Sconv[1:], "r--", lw=1.5)
        ht_abs2 = np.abs(np.fft.rfft(sc["weff"] * sc["h_td"][0]))**2 * 2 / (C.FS * C.M)
        axR.loglog(C.freqs[1:], ht_abs2[1:], color="tab:orange", lw=1.0, alpha=.85)
        axR.set_xlim(C.F_LO_BAND, C.FS / 2)
        axR.set_ylim(3e-47, 3e-38)
        axR.set_ylabel(r"PSD [1/Hz]")
        if i == nrow - 1:
            axR.set_xlabel(r"$f$ [Hz]")

    from matplotlib.patches import Patch
    hL = [Patch(fc="orange", alpha=.30, label="gap"),
          Line2D([0], [0], color="gray", lw=1.4, alpha=.7,
                 label="data TDI"),
          Line2D([0], [0], color="black", lw=1.6, label="MBHB - no gap"),
          Line2D([0], [0], color="tab:red", ls="--", lw=1.5,
                 label=r"$w_{\rm eff}\,h$"),
          Line2D([0], [0], color="tab:blue", ls="--", lw=1.5, label=r"$t_c$")]
    hR = [Line2D([0], [0], color="0.7", lw=1.4, label="periodogram"),
          Line2D([0], [0], color="k", lw=1.6, label=r"$W_c\,S_n$"),
          Line2D([0], [0], color="r", ls="--", lw=1.6, label="conv. diagonal"),
          Line2D([0], [0], color="tab:orange", lw=1.5, label=r"$|h_w|^2$")]
    leg1 = fig.legend(handles=hL, loc="lower left", ncol=5, frameon=False,
                      fontsize=14, bbox_to_anchor=(0.01, 0.985),
                      handletextpad=0.4, columnspacing=0.9)
    leg2 = fig.legend(handles=hR, loc="lower right", ncol=4, frameon=False,
                      fontsize=14, bbox_to_anchor=(0.99, 0.985),
                      handletextpad=0.4, columnspacing=0.9)
    fig.tight_layout()
    fig.savefig(fname, dpi=200, bbox_inches="tight",
                bbox_extra_artists=(leg1, leg2))
    if fname.endswith(".png"):
        fig.savefig(fname[:-4] + ".pdf", bbox_inches="tight",
                    bbox_extra_artists=(leg1, leg2))
    plt.close(fig)
    return fname
