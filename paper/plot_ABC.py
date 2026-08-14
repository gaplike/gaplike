"""Full-covariance posteriors of scenarios A, B, C overlaid (key + noise)."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import corner as corner_pkg

import plots as P
import pe

SCEN = ["A", "B", "C"]
COLS = {"A": "tab:blue", "B": "tab:red", "C": "tab:orange"}
NAME = {"A": "Scenario A (two long gaps)",
        "B": "Scenario B (twelve short gaps)",
        "C": "Scenario C (drastic gaps)"}


def load(sk):
    z = np.load(f"results/{sk}_full.npz")
    keep = z["log_prob"] > z["log_prob"].max() - 60.0
    return z["chain"][keep]


chains = {sk: load(sk) for sk in SCEN}


def corner_ABC(block, fname):
    cols = P.BLOCKS[block]
    labels = [P.LAB_TEX[i] for i in cols]

    def delta(sk):
        return (chains[sk][:, cols] - pe.X_TRUE[cols]) * \
            np.array([P.SCALE[i] for i in cols])

    lo = [np.quantile(np.concatenate([delta(s)[:, k] for s in SCEN]), 0.001)
          for k in range(len(cols))]
    hi = [np.quantile(np.concatenate([delta(s)[:, k] for s in SCEN]), 0.999)
          for k in range(len(cols))]
    rng = list(zip(lo, hi))
    fig = None
    for sk in ["C", "A", "B"]:
        fig = corner_pkg.corner(
            delta(sk), labels=labels, color=COLS[sk], fig=fig, range=rng,
            plot_datapoints=False, plot_density=False, bins=45, smooth=1.0,
            levels=(0.393, 0.865), hist_kwargs=dict(density=True, lw=1.7),
            contour_kwargs=dict(linewidths=1.5), label_kwargs=dict(fontsize=19))
    nd = len(cols)
    axes = np.array(fig.axes).reshape(nd, nd)
    for i in range(nd):
        for j in range(i + 1):
            ax = axes[i, j]
            if j < i:
                ax.axvline(0, color="0.4", lw=1.2)
                ax.axhline(0, color="0.4", lw=1.2)
                ax.plot(0, 0, "s", color="0.3", ms=4)
            else:
                ax.axvline(0, color="0.4", lw=1.2)
    handles = [Line2D([0], [0], color=COLS[s], lw=2.2, label=NAME[s])
               for s in SCEN] + \
              [Line2D([0], [0], color="0.4", lw=1.4, label="truth")]
    if nd > 3:
        fig.legend(handles=handles, loc="upper right", fontsize=15,
                   frameon=False, bbox_to_anchor=(0.98, 0.97))
    else:
        fig.legend(handles=handles, loc="center left", fontsize=13,
                   frameon=False, bbox_to_anchor=(0.60, 0.80))
    fig.suptitle("Full covariance model", fontsize=22,
                 y=1.0 if nd > 3 else 1.05)
    fig.savefig(fname, dpi=170, bbox_inches="tight")
    fig.savefig(fname.replace(".png", ".pdf"), bbox_inches="tight")
    plt.close(fig)
    print("saved", fname)


if __name__ == "__main__":
    corner_ABC("key", "figures/corner_key_full_ABC.png")
    corner_ABC("noise", "figures/corner_noise_full_ABC.png")
