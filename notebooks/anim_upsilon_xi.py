"""Two talk animations for the Upsilon/Xi diagnostics of the paper.

    Upsilon = true scatter of the estimate / quoted posterior width
    Xi      = quoted posterior width / width of the exact analysis

``upsilon_xi_quadrants.gif`` — four analyses side by side, one repeated
experiment per row: the quoted 68% interval against the width the data
actually allow (grey band), with the coverage counter running.  The four
cases are anchored to the paper's measured values (scenario C unless noted).

``upsilon_xi_pp.gif`` — the same four analyses through the field's standard
diagnostic, the PP plot, injections accumulating: it reads Upsilon and ONLY
Upsilon.  The calibrated-but-inflated analysis is indistinguishable from the
optimal one — Xi is invisible to a PP plot.

Every case is a 1-D Gaussian sampling model in units of the exact-analysis
width: quoted width Xi, scatter about truth Upsilon*Xi (split into bias +
variance where the paper's case is bias-driven).  Same standard-normal draws
across panels, so the panels differ only in (Upsilon, Xi, bias).

    python anim_upsilon_xi.py             # writes ../assets/*.gif
"""
from pathlib import Path

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

ASSETS = Path(__file__).resolve().parents[1] / "assets"

# ---------------------------------------------------------------- the cases
#
# All four cases are MEASURED paper values, quoted directly from Table III
# of the manuscript: the signal-sector entries are the 11-parameter averages
# (Upsilon_s bar, Xi_s bar), the noise-sector entry is lambda_tm.  The full
# per-parameter set regenerates via paper/dump_upsxi.py into
# paper/results/upsxi_dump.json.  NB the model tiers: paper's "scalar W_c
# correction" is the pipeline's "bare", the paper's "raw PSD" is "psd".
# ``bias`` is the systematic offset of the estimate from truth in
# exact-width units --- zero for all four anchors; it is kept as a knob
# (the interactive explorer exposes it).
CASES = [
    dict(ups=1.0, xi=1.0, bias=0.0, color="0.25",
         head=r"$\Upsilon \simeq 1,\ \ \Xi \simeq 1$",
         tag="honest and optimal",
         anchor="exact time domain, scenario C  (1 by construction)"),
    dict(ups=0.92, xi=9.3, bias=0.0, color="tab:blue",
         head=r"$\Upsilon < 1,\ \ \Xi \gg 1$",
         tag="calibrated, but inflated",
         anchor=r"convolved diagonal, scenario C, signal sector:"
                "  $\\bar\\Upsilon_s=0.92$, $\\bar\\Xi_s=9.3$"),
    dict(ups=1.47, xi=0.82, bias=0.0, color="tab:red",
         head=r"$\Upsilon > 1,\ \ \Xi < 1$",
         tag="overconfident: too narrow",
         anchor=r"$W_c$-scaled Whittle, scenario B, signal sector:"
                "  $\\bar\\Upsilon_s=1.47$, $\\bar\\Xi_s=0.82$"),
    dict(ups=1.23, xi=13.7, bias=0.0, color="tab:purple",
         head=r"$\Upsilon > 1,\ \ \Xi > 1$",
         tag="overconfident and inflated",
         anchor=r"convolved diagonal, scenario C, $\lambda_{\rm tm}$:"
                "  $\\Upsilon=1.23$, $\\Xi=13.7$"),
]

N_ROWS = 40           # experiments per panel in the forest view
N_INJ = 800           # injections for the PP view
SEED = 7
FPS = 8
RC = {"font.family": "serif", "mathtext.fontset": "cm", "font.size": 11,
      "axes.titlesize": 12, "axes.labelsize": 11.5,
      "xtick.labelsize": 9.5, "ytick.labelsize": 9.5}


def _draws(n):
    """One shared set of standardized draws: panels differ only in the case
    parameters, so identical rows are comparable across panels."""
    return np.random.default_rng(SEED).standard_normal(n)


def _estimates(case, u):
    """theta_hat about truth 0, in exact-width units: bias + scatter."""
    s_tot = case["ups"] * case["xi"]
    b = case["bias"]
    s = np.sqrt(max(s_tot**2 - b**2, 1e-12))
    return b + s * u


# ------------------------------------------------------- quadrant animation
def make_quadrants(fname="upsilon_xi_quadrants.gif"):
    u = _draws(N_ROWS)
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(11.4, 7.8))
        fig.subplots_adjust(top=0.795, bottom=0.07, hspace=0.44, wspace=0.14)
        fig.suptitle(
            r"$\Upsilon = \dfrac{\rm true\ scatter}{\rm quoted\ width}$"
            r"$\qquad\qquad$"
            r"$\Xi = \dfrac{\rm quoted\ width}{\rm width\ of\ the\ exact\ "
            r"analysis}$", y=0.982, fontsize=15)
        fig.text(0.5, 0.885,
                 "one repeated experiment per row — dot: estimate, bar: quoted "
                 "68% interval, grey band: the width the data allow\n"
                 "faded bars miss the truth; an honest analysis misses 32% "
                 "of the time", ha="center", fontsize=10.5, color="0.35")

        artists, texts = [], []
        for ax, case in zip(axes.ravel(), CASES):
            th = _estimates(case, u)
            half = max(case["xi"], abs(case["bias"]) + case["ups"] * case["xi"])
            ax.set_xlim(-2.9 * half, 2.9 * half)
            ax.set_ylim(-(N_ROWS + 1.5), 6.4)
            ax.axvspan(-1, 1, color="0.82", alpha=0.75, lw=0, zorder=0)
            ax.axvline(0.0, color="k", ls="--", lw=1.0, zorder=1)
            ax.set_yticks([])
            ax.set_xlabel(r"$(\hat\theta - \theta_{\rm true})\ /\ "
                          r"\sigma_{\rm exact}$", labelpad=1.5)
            ax.set_title(f"{case['head']} — {case['tag']}\n", pad=10)
            ax.text(0.5, 1.02, case["anchor"], transform=ax.transAxes,
                    ha="center", fontsize=8.6, color="0.4")
            # the Xi visual: quoted vs exact width as two stacked reference
            # bars (the grey band can be a sliver when Xi is large)
            ax.plot([-case["xi"], case["xi"]], [4.6, 4.6], color=case["color"],
                    lw=4, solid_capstyle="butt")
            ax.annotate("quoted", (max(case["xi"], 0.14 * half), 4.6),
                        textcoords="offset points", xytext=(5, -3.5),
                        fontsize=8, color=case["color"])
            ax.plot([-1, 1], [2.6, 2.6], color="0.35", lw=4,
                    solid_capstyle="butt")
            ax.annotate("exact", (max(case["xi"], 0.14 * half), 2.6),
                        textcoords="offset points", xytext=(5, -3.5),
                        fontsize=8, color="0.35")
            rows = []
            for i in range(N_ROWS):
                covers = abs(th[i]) <= case["xi"]
                bar, = ax.plot([], [], color=case["color"], lw=1.7,
                               alpha=0.95 if covers else 0.38,
                               solid_capstyle="butt", zorder=3)
                dot, = ax.plot([], [], "o", color=case["color"], ms=2.8,
                               mfc=case["color"] if covers else "white",
                               mew=0.8, zorder=4)
                rows.append((bar, dot, th[i], covers))
            cnt = ax.text(0.985, 0.02, "", transform=ax.transAxes,
                          ha="right", va="bottom", fontsize=9.5, color="0.25")
            artists.append(rows)
            texts.append(cnt)

        hold = 2 * FPS

        def update(frame):
            n = min(frame + 1, N_ROWS)
            changed = []
            for rows, cnt, case in zip(artists, texts, CASES):
                nc = 0
                for i in range(n):
                    bar, dot, thi, covers = rows[i]
                    y = -(i + 1)
                    bar.set_data([thi - case["xi"], thi + case["xi"]], [y, y])
                    dot.set_data([thi], [y])
                    nc += covers
                    changed += [bar, dot]
                cnt.set_text(f"cover: {nc}/{n}  ({100 * nc / n:.0f}%)")
                changed.append(cnt)
            return changed

        anim = FuncAnimation(fig, update, frames=N_ROWS + hold, blit=False)
        out = ASSETS / fname
        anim.save(out, writer=PillowWriter(fps=FPS))
        plt.close(fig)
    print(f"wrote {out}")


# ------------------------------------------------------------- PP animation
def _pp_expected(case, p):
    """Analytic coverage of the central quoted p-interval, Gaussian case."""
    z = norm.ppf(0.5 * (1.0 + p))
    s_tot = case["ups"] * case["xi"]
    b = case["bias"]
    s = np.sqrt(max(s_tot**2 - b**2, 1e-12))
    return (norm.cdf((z * case["xi"] - b) / s)
            - norm.cdf((-z * case["xi"] - b) / s))


def make_pp(fname="upsilon_xi_pp.gif"):
    u = _draws(N_INJ)
    p_grid = np.linspace(0, 1, 400)
    # reveal schedule: a few injections per frame at first, then whole batches
    counts = np.unique(np.geomspace(1, N_INJ, 46).astype(int))
    with plt.rc_context(RC):
        fig, axes = plt.subplots(2, 2, figsize=(9.6, 9.4))
        fig.subplots_adjust(top=0.865, bottom=0.06, hspace=0.36, wspace=0.24)
        fig.suptitle("the PP plot measures $\\Upsilon$ — and only $\\Upsilon$",
                     y=0.975, fontsize=16)
        fig.text(0.5, 0.925,
                 "fraction of injections whose true value falls inside the "
                 "quoted $p$-credible interval, as injections accumulate",
                 ha="center", fontsize=10.5, color="0.35")

        curves, texts = [], []
        for ax, case in zip(axes.ravel(), CASES):
            th = _estimates(case, u)
            p_i = 2.0 * norm.cdf(np.abs(th) / case["xi"]) - 1.0
            ax.plot([0, 1], [0, 1], color="k", ls="--", lw=1.0, zorder=1)
            ax.plot(p_grid, _pp_expected(case, p_grid), color=case["color"],
                    lw=1.1, alpha=0.55, zorder=2)
            emp, = ax.step([], [], where="post", color=case["color"], lw=2.4,
                           zorder=3)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_aspect("equal")
            ax.grid(alpha=0.22)
            ax.set_title(f"{case['head']} — {case['tag']}\n", pad=8)
            ax.text(0.5, 1.015, case["anchor"], transform=ax.transAxes,
                    ha="center", fontsize=8.2, color="0.4")
            ax.set_xlabel("credible level $p$", labelpad=1.5)
            ax.set_ylabel("fraction inside")
            cnt = ax.text(0.03, 0.965, "", transform=ax.transAxes, va="top",
                          fontsize=9.5, color="0.25")
            curves.append((emp, p_i))          # injection order; sorted per frame
            texts.append(cnt)
        # the punchline, on the calibrated-but-inflated panel
        axes.ravel()[1].text(0.96, 0.06,
                             "the gentle bow is $\\Upsilon=0.92$ —\n"
                             "the $9.3\\times$ inflation ($\\Xi$) is invisible",
                             transform=axes.ravel()[1].transAxes, ha="right",
                             fontsize=9.5, color="tab:blue", style="italic")

        hold = 2 * FPS

        def update(frame):
            n = counts[min(frame, len(counts) - 1)]
            changed = []
            for (emp, p_i), cnt in zip(curves, texts):
                ps = np.sort(p_i[:n])
                x = np.concatenate([[0.0], ps, [1.0]])
                y = np.concatenate([[0.0], np.arange(1, ps.size + 1) / n,
                                    [1.0]])
                emp.set_data(x, y)
                cnt.set_text(f"$n = {n}$ injections")
                changed += [emp, cnt]
            return changed

        anim = FuncAnimation(fig, update, frames=len(counts) + hold, blit=False)
        out = ASSETS / fname
        anim.save(out, writer=PillowWriter(fps=FPS))
        plt.close(fig)
    print(f"wrote {out}")


if __name__ == "__main__":
    make_quadrants()
    make_pp()
