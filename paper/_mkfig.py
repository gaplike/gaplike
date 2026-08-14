"""Draw figures/cg_scaling.{png,pdf} from results/cg_scaling.json
(produced by fig_cg_scaling.py)."""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
out = json.load(open("results/cg_scaling.json"))
os.makedirs("figures", exist_ok=True)
kk = np.array([r["k"] for r in out]); tc = np.array([r["t_cg"] for r in out])
it = np.array([r["iters"] for r in out])
kd = np.array([r["k"] for r in out if "t_dense" in r])
td = np.array([r["t_dense"] for r in out if "t_dense" in r])
kr = np.array([r["k"] for r in out if "rel" in r])
rr = np.array([r["rel"] for r in out if "rel" in r])

# measured power laws  t ~ N^p   (fit the last few points)
pd = np.polyfit(kd[-4:]*np.log(2), np.log(td[-4:]), 1)[0]
pc = np.polyfit(kk[-5:]*np.log(2), np.log(tc[-5:]), 1)[0]
pi = np.polyfit(kk[-5:]*np.log(2), np.log(it[-5:]), 1)[0]
print(f"measured exponents: dense N^{pd:.2f}   CG N^{pc:.2f}   iters N^{pi:.2f}")

rc = {"font.family":"serif","mathtext.fontset":"cm","font.size":14,
      "axes.labelsize":17,"axes.titlesize":16,"xtick.labelsize":13,
      "ytick.labelsize":13,"legend.fontsize":12.5}
with plt.rc_context(rc):
    fig,(axL,axR)=plt.subplots(1,2,figsize=(12.8,5.0))
    kg=np.linspace(kk.min(),kk.max(),60)
    axL.plot(kg, td[-1]*2.0**(pd*(kg-kd[-1])), ":", color="tab:red", lw=1.5)
    axL.plot(kg, tc[-1]*2.0**(pc*(kg-kk[-1])), ":", color="tab:blue", lw=1.5)
    ke = np.arange(kd[-1], kk.max()+1)          # extrapolated dense
    te = td[-1]*2.0**(pd*(ke-kd[-1]))
    axL.plot(ke, te, "o--", color="tab:red", ms=8, lw=1.6, mfc="white",
             label="dense, extrapolated")
    axL.plot(kd, td, "o-", color="tab:red", ms=7.5, lw=2,
             label=rf"dense: build, Cholesky, solve  $\propto N^{{{pd:.1f}}}$")
    for x,y in zip(ke[1:], te[1:]):
        lab = f"{y/3600:.0f} h" if y>3600 else f"{y/60:.0f} min"
        axL.annotate(lab,(x,y),textcoords="offset points",xytext=(7,-12),
                     fontsize=10,color="tab:red",ha="left",va="top")
    print("extrapolated dense:", [(int(a), f"{b/3600:.1f} h") for a,b in zip(ke,te)])
    axL.plot(kk, tc, "s-", color="tab:blue", ms=7.5, lw=2,
             label=rf"preconditioned CG, matrix-free  $\propto N^{{{pc:.2f}}}$")
    axL.set_xlim(kk.min()-0.4, kk.max()+0.8)
    axL.set_yscale("log"); axL.set_xlabel(r"$\log_{2} N$")
    axL.set_ylabel("wall time per quadratic form  [s]")
    axL.grid(alpha=.25); h,l=axL.get_legend_handles_labels(); o=[1,2,0]
    axL.legend([h[i] for i in o],[l[i] for i in o],frameon=False,loc="upper left")
    axL.set_title("cost of one $a^{T}\\Sigma_{OO}^{-1}a$ on gapped data")

    axR.semilogy(kk, it, "s-", color="tab:blue", ms=7.5, lw=2, label="measured")
    kf = kg[kg >= kk[-5]]                       # show the fit only where fitted
    axR.semilogy(kf, it[-1]*2.0**(pi*(kf-kk[-1])), ":", color="tab:blue", lw=1.6,
                 label=rf"$\propto N^{{{pi:.2f}}}$ over the top of the range")
    loc = np.diff(np.log2(it))/np.diff(kk)      # local logarithmic slope
    axR.set_xlim(kk.min()-0.4, kk.max()+0.9)
    axR.set_xlabel(r"$\log_{2} N$")
    axR.set_ylabel("preconditioned CG iterations")
    axR.set_yticks([100, 200, 500, 1000, 2000])
    axR.get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    axR.grid(alpha=.25, which="both")
    axR.legend(frameon=False, loc="upper left")
    axR.set_title("iterations to a relative residual of $10^{-10}$")
    print("local slopes:", " ".join(f"{p:.2f}" for p in loc))
    print(f"agreement CG vs dense: {rr.min():.1e} to {rr.max():.1e} over k={kr.min()}..{kr.max()}")
    fig.tight_layout()
    for e in ("png","pdf"):
        fig.savefig(f"figures/cg_scaling.{e}", dpi=180, bbox_inches="tight")
print("saved")
