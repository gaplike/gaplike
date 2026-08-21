"""Draw figures/det_scaling.{png,pdf} from results/det_scaling.json and
results/det_flexible_k*.json (produced by fig_det_scaling.py and
fig_det_flexible.py).

Left  : wall time of one log-determinant vs record length --- dense
        build+Cholesky against the EXACT complement identity (one g x g
        Cholesky on the gap samples) and matrix-free SLQ; measured power
        laws, dense and complement extrapolated beyond their largest
        affordable points (the companion of the quadratic-form panel in
        fig `cg_scaling`).
Right : what the accept ratio actually feels --- ABSOLUTE error in
        log-likelihood units.  Absolute SLQ estimates sit at 10^1-10^2
        regardless of budget (they are relative-accuracy estimators of a
        ~10^5 quantity); shared-probe determinant DIFFERENCES for
        MCMC-sized spline steps sit near the 0.1 tolerance line, with
        naive independent-probe differencing shown for contrast.
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

scal = json.load(open("results/det_scaling.json"))
flex = [json.load(open(p)) for p in sorted(glob.glob("results/det_flexible_k*.json"))]
os.makedirs("figures", exist_ok=True)

BEST = dict(probes=32, lanczos=100)     # the SLQ setting shown as the main line
CHEAP = dict(probes=8, lanczos=50)


def slq_entry(row, sel):
    for e in row.get("slq", []):
        if e["probes"] == sel["probes"] and e["lanczos"] == sel["lanczos"]:
            return e
    return None


kk = np.array([r["k"] for r in scal if slq_entry(r, BEST)])
t_slq = np.array([slq_entry(r, BEST)["t"] for r in scal if slq_entry(r, BEST)])
kd = np.array([r["k"] for r in scal if "t_dense" in r])
td = np.array([r["t_dense"] for r in scal if "t_dense" in r])
kc = np.array([r["k"] for r in scal if "t_comp" in r])
tc = np.array([r["t_comp"] for r in scal if "t_comp" in r])

# measured power laws  t ~ N^p  (fit the last few points)
pd_ = np.polyfit(kd[-4:] * np.log(2), np.log(td[-4:]), 1)[0]
ps_ = np.polyfit(kk[-4:] * np.log(2), np.log(t_slq[-4:]), 1)[0]
pc_ = (np.polyfit(kc[-4:] * np.log(2), np.log(tc[-4:]), 1)[0]
       if kc.size >= 4 else np.nan)
print(f"measured exponents: dense N^{pd_:.2f}   SLQ N^{ps_:.2f}"
      + (f"   complement N^{pc_:.2f}" if np.isfinite(pc_) else ""))

rc = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 14,
      "axes.labelsize": 17, "axes.titlesize": 16, "xtick.labelsize": 13,
      "ytick.labelsize": 13, "legend.fontsize": 12.5}
with plt.rc_context(rc):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12.8, 5.0))

    # ---- left: cost --------------------------------------------------------
    ke = np.arange(kd[-1], 19)                      # extrapolated dense
    te = td[-1] * 2.0 ** (pd_ * (ke - kd[-1]))
    axL.plot(ke, te, "o--", color="tab:red", ms=8, lw=1.6, mfc="white",
             label="dense, extrapolated")
    axL.plot(kd, td, "o-", color="tab:red", ms=7.5, lw=2,
             label=rf"dense: build + Cholesky  $\propto N^{{{pd_:.1f}}}$")
    for x, y in zip(ke[1:], te[1:]):
        lab = f"{y/3600:.0f} h" if y > 3600 else (
            f"{y/60:.0f} min" if y > 60 else f"{y:.0f} s")
        axL.annotate(lab, (x, y), textcoords="offset points", xytext=(7, -12),
                     fontsize=10, color="tab:red", ha="left", va="top")
    if kc.size:
        if np.isfinite(pc_):
            kce = np.arange(kc[-1], 19)
            tce = tc[-1] * 2.0 ** (pc_ * (kce - kc[-1]))
            axL.plot(kce, tce, "D--", color="tab:green", ms=6.5, lw=1.4,
                     mfc="white")
        lab = r"complement identity (exact)"
        if np.isfinite(pc_):
            lab += rf"  $\propto N^{{{pc_:.1f}}}$"
        axL.plot(kc, tc, "D-", color="tab:green", ms=7, lw=2, label=lab)
    ks = np.arange(kk[-1], 19)
    ts = t_slq[-1] * 2.0 ** (ps_ * (ks - kk[-1]))
    axL.plot(ks, ts, "s--", color="tab:blue", ms=7, lw=1.4, mfc="white")
    axL.plot(kk, t_slq, "s-", color="tab:blue", ms=7.5, lw=2,
             label=r"SLQ, matrix-free ($M{=}32$, $k{=}100$)  "
                   + rf"$\propto N^{{{ps_:.2f}}}$")
    axL.set_yscale("log")
    axL.set_xlabel(r"record length  $\log_2 N$")
    axL.set_ylabel("wall time of one log-determinant [s]")
    axL.legend(loc="upper left", framealpha=0.95, fontsize=11)
    axL.grid(alpha=0.25)

    # ---- right: accuracy in log-likelihood units ---------------------------
    for sel, c, mk, lab in [(CHEAP, "0.6", "v", r"$M{=}8$, $k{=}50$"),
                            (BEST, "0.35", "^", r"$M{=}32$, $k{=}100$")]:
        ka = np.array([r["k"] for r in scal if slq_entry(r, sel) and
                       "abs_err" in slq_entry(r, sel)])
        ea = np.array([slq_entry(r, sel)["abs_err"] for r in scal
                       if slq_entry(r, sel) and "abs_err" in slq_entry(r, sel)])
        axR.plot(ka, ea, mk, color=c, ms=8, ls="-", lw=1.2,
                 label=rf"absolute SLQ, {lab}")

    if flex:
        kf = np.array([F["k"] for F in flex])
        for sig, c, mk in [(0.01, "tab:green", "*"), (0.1, "tab:olive", "P")]:
            med_sh, med_in, kk_f = [], [], []
            for F in flex:
                blk = next((s for s in F["sigmas"] if s["sigma"] == sig), None)
                if blk:
                    kk_f.append(F["k"])
                    med_sh.append(np.median([r["err_shared"] for r in blk["rows"]]))
                    med_in.append(np.median([r["err_indep"] for r in blk["rows"]]))
            axR.plot(kk_f, med_in, mk, color="0.75", ms=11, ls=":",
                     lw=1.0, mec="0.4")
            axR.plot(kk_f, med_sh, mk, color=c, ms=13, ls="-", lw=1.6,
                     mec="k", label=r"difference, shared probes ($\sigma{=}"
                                    + rf"{sig}$)")
        axR.plot([], [], "s", color="0.75", mec="0.4",
                 label="same, independent probes")

    # the complement identity is exact --- its error (~1e-9 vs dense truth)
    # sits ten decades below the axis, so it enters as a legend line only
    err_c = [r["abs_err_comp"] for r in scal if "abs_err_comp" in r]
    if err_c:
        axR.plot([], [], "D", color="tab:green",
                 label=rf"complement identity: exact "
                       rf"($|$err$|<10^{{{int(np.ceil(np.log10(max(err_c))))}}}$, off scale)")

    axR.axhline(0.1, color="k", ls="--", lw=1.2)
    axR.annotate("0.1 log-likelihood units", (kk.min() + 0.1, 0.1),
                 textcoords="offset points", xytext=(0, 5), fontsize=11)
    axR.set_yscale("log")
    axR.set_xlabel(r"record length  $\log_2 N$")
    axR.set_ylabel(r"absolute error in $\log|\Sigma_{OO}|$  [LL units]")
    axR.legend(loc="upper left", framealpha=0.95, fontsize=11)
    axR.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig("figures/det_scaling.pdf", bbox_inches="tight")
    fig.savefig("figures/det_scaling.png", dpi=170, bbox_inches="tight")
print("wrote figures/det_scaling.{pdf,png}")
