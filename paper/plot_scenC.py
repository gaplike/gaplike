"""Scenario C figures: 3-way corners (FD full, FD conv-diag, TD exact pencil),
noise-sector corner with exact profile MLEs, and the alias-structure colormap."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.optimize import minimize
import corner as corner_pkg

import core as C
import pe
import plots as P          # global style
import scenC

SC = scenC.build_scenario_C(verbose=False)
obs = np.flatnonzero(SC["gate"] > 0.0)
tp = scenC.TDPencil(obs)
h0_td, dh_td_full = scenC.signal_derivs_td()
r_td = (SC["d_td"] - h0_td)[:, obs]

COLS = {"full": "tab:red", "diag": "tab:blue", "td": "tab:orange"}
NAME = {"full": "FD full covariance", "diag": "FD convolved diagonal",
        "td": "TD exact (restricted stationary)"}


def load(mk):
    z = np.load(f"results/C_{mk}.npz")
    keep = z["log_prob"] > z["log_prob"].max() - 60.0
    return z["chain"][keep]


chains = {mk: load(mk) for mk in ["full", "diag", "td"]}
for mk in chains:
    print(mk, chains[mk].shape)

# ---- exact (nonlinear) profile noise MLEs + curvatures ----
exact, exact_cov = {}, {}
for mk in ["full", "diag"]:
    exact[mk], exact_cov[mk] = pe.noise_profile_mle(SC, mk)
res = minimize(lambda x: -tp.loglike(x[0], x[1], r_td), np.zeros(2),
               method="Nelder-Mead", options=dict(xatol=1e-6, fatol=1e-8))
exact["td"] = res.x
exact_cov["td"] = np.linalg.inv(tp.fisher_noise(nch=2))
print("exact profile MLEs:", {k: np.round(v, 4) for k, v in exact.items()})

# ---- full 13-parameter analytic predictions (dashed overlays) --------------
# signal sector: linearized MLE (exact as a statistic -- linear in the data)
# + the model's Fisher widths; noise sector: exact nonlinear MLE + curvature
# (Godambe-White; coincides with Fisher for the two exact tiers).
dh_obs = dh_td_full[:, obs, :]
pred_mean, pred_cov = {}, {}
for mk in ["full", "diag"]:
    m = SC["models"][mk]
    mu = pe.X_TRUE.copy()
    mu[:11] = mu[:11] + m.mle_signal(SC["resid"])
    mu[11:] = exact[mk]
    cv = np.zeros((13, 13))
    cv[:11, :11] = m.cov_s
    cv[11:, 11:] = exact_cov[mk]
    pred_mean[mk], pred_cov[mk] = mu, cv
mu = pe.X_TRUE.copy()
mu[:11] = mu[:11] + tp.mle_signal(dh_obs, r_td)
mu[11:] = exact["td"]
cv = np.zeros((13, 13))
cv[:11, :11] = C._safe_inv(tp.fisher_signal(dh_obs), rcond=1e-12)
cv[11:, 11:] = exact_cov["td"]
pred_mean["td"], pred_cov["td"] = mu, cv

# ---- corner blocks ----
def corner_C(block_cols, fname, ranges_from="td"):
    labels = [P.LAB_TEX[i] for i in block_cols]

    def delta(mk):
        return (chains[mk][:, block_cols] - pe.X_TRUE[block_cols]) * \
            np.array([P.SCALE[i] for i in block_cols])

    lo = [np.quantile(np.concatenate([delta(m)[:, k] for m in chains]), 0.001)
          for k in range(len(block_cols))]
    hi = [np.quantile(np.concatenate([delta(m)[:, k] for m in chains]), 0.999)
          for k in range(len(block_cols))]
    rng = list(zip(lo, hi))
    fig = None
    for mk in ["diag", "full", "td"]:
        fig = corner_pkg.corner(
            delta(mk), labels=labels, color=COLS[mk], fig=fig, range=rng,
            plot_datapoints=False, plot_density=False, bins=45, smooth=1.0,
            levels=(0.393, 0.865), hist_kwargs=dict(density=True, lw=1.7),
            contour_kwargs=dict(linewidths=1.5), label_kwargs=dict(fontsize=19))
    # analytic dashed overlays: FULL prediction (signal + noise sectors)
    noise_dims = [k for k, c in enumerate(block_cols) if c in (11, 12)]
    rng0 = np.random.default_rng(5)
    scale_cols = np.array([P.SCALE[i] for i in block_cols])
    for mk in ["diag", "full", "td"]:
        sub = np.ix_(block_cols, block_cols)
        gsam = rng0.multivariate_normal(pred_mean[mk][block_cols],
                                        pred_cov[mk][sub], size=40000)
        dsam = (gsam - pe.X_TRUE[block_cols][None, :]) * scale_cols[None, :]
        # dashed prediction on the 2D panels only: the 1D histograms stay
        # clean (hist overlay drawn fully transparent)
        fig = corner_pkg.corner(dsam, color=COLS[mk], fig=fig, range=rng,
                                plot_datapoints=False, plot_density=False,
                                no_fill_contours=True, bins=45, smooth=1.0,
                                levels=(0.393, 0.865),
                                hist_kwargs=dict(density=True, lw=0.0,
                                                 alpha=0.0),
                                contour_kwargs=dict(linestyles="--",
                                                    linewidths=1.2, alpha=0.9))
    tru = np.zeros(len(block_cols))
    nd = len(block_cols)
    axes = np.array(fig.axes).reshape(nd, nd)
    for i in range(nd):
        for j in range(i + 1):
            ax = axes[i, j]
            if j < i:
                ax.axvline(tru[j], color="0.4", lw=1.2)
                ax.axhline(tru[i], color="0.4", lw=1.2)
                ax.plot(tru[j], tru[i], "s", color="0.3", ms=4)
            else:
                ax.axvline(tru[i], color="0.4", lw=1.2)
    if noise_dims:
        # dotted vertical exact-MLE markers on the noise histograms
        for k in noise_dims:
            comp = 0 if block_cols[k] == 11 else 1        # lam_tm / lam_oms
            for mk in ["diag", "full", "td"]:
                axes[k, k].axvline(exact[mk][comp], color=COLS[mk],
                                   ls=":", lw=1.4, alpha=0.9)
    handles = [Line2D([0], [0], color=COLS[m], lw=2.2, label=NAME[m])
               for m in ["full", "diag", "td"]] + \
              [Line2D([0], [0], color="0.4", lw=1.4, label="truth"),
               Line2D([0], [0], color="0.35", lw=1.5, ls="--",
                      label="Analytic prediction: linearized MLE + Fisher widths (signal);\n"
                            "exact profile MLE + curvature (noise, Godambe–White)")]
    loc = "upper right" if nd > 3 else (0.97, 0.75)
    if nd > 3:
        fig.legend(handles=handles, loc="upper right", fontsize=15,
                   frameon=False, bbox_to_anchor=(0.98, 0.97))
    else:
        fig.legend(handles=handles, loc="center left", fontsize=14,
                   frameon=False, bbox_to_anchor=(0.62, 0.78))
    fig.suptitle("Scenario C (drastic gaps)", fontsize=22,
                 y=1.0 if nd > 3 else 1.04)
    fig.savefig(fname, dpi=170, bbox_inches="tight")
    fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


corner_C(P.BLOCKS["key"], "figures/corner_key_C_full_diag_td.png")
corner_C(P.BLOCKS["noise"], "figures/corner_noise_C_full_diag_td.png")

# ---- alias-structure covariance colormap ----
Q = SC["truth"]["Q"]
d = np.sqrt(np.real(np.diag(Q)))
R = np.abs(Q) / np.outer(d, d)
fmhz = C.fb * 1e3
fig, ax = plt.subplots(figsize=(7.6, 6.4))
im = ax.imshow(np.log10(np.clip(R, 1e-12, None)), origin="lower",
               extent=[fmhz[0], fmhz[-1], fmhz[0], fmhz[-1]],
               cmap="magma", vmin=-4, vmax=0, aspect="auto")
for k in (1, 2):
    off = k * 1e3 / (5 * C.DT)        # alias spacing f_s/5 = 13.33 mHz
    ax.plot([fmhz[0], fmhz[-1] - off], [fmhz[0] + off, fmhz[-1]],
            color="w", lw=.6, ls=":", alpha=.8)
    ax.plot([fmhz[0] + off, fmhz[-1]], [fmhz[0], fmhz[-1] - off],
            color="w", lw=.6, ls=":", alpha=.8)
ax.set_xlabel(r"$f$ [mHz]")
ax.set_ylabel(r"$f$ [mHz]")
ax.set_title(r"Scenario C (drastic gaps) — $\log_{10}|Q_{jk}|/\sqrt{Q_{jj}Q_{kk}}$")
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
fig.savefig("figures/cov_colormap_C.png", dpi=180, bbox_inches="tight")
fig.savefig("figures/cov_colormap_C.pdf", bbox_inches="tight")
print("saved figures/cov_colormap_C.png")

# ---- TD vs FD-full log-likelihood surface congruence (noise sector) ----
trick = SC["trick"]
rfd = SC["resid"]
grid = np.linspace(-0.25, 0.25, 9)
dl_fd, dl_td = [], []
l0_fd = trick.loglike(0.0, 0.0, rfd)
l0_td = tp.loglike(0.0, 0.0, r_td)
for da in grid:
    for db in np.linspace(-0.02, 0.02, 5):
        dl_fd.append(trick.loglike(da, db, rfd) - l0_fd)
        dl_td.append(tp.loglike(da, db, r_td) - l0_td)
dl_fd, dl_td = np.array(dl_fd), np.array(dl_td)
cc = np.corrcoef(dl_fd, dl_td)[0, 1]
slope = np.polyfit(dl_td, dl_fd, 1)[0]
print(f"Delta-lnL surfaces (noise sector): corr = {cc:.5f}, slope FD/TD = {slope:.3f}")
