"""Regenerate EVERY paper figure from the cached PE chains in results/.

No sampling is run here: the expensive PE outputs are loaded from results/
(re-create them with driver.py, scenC_run.py and ensemble_noise.py).  The
scenarios themselves (windows, covariances, analytics) are rebuilt live, so
this script needs lisabeta (waveform derivatives) and takes ~10 minutes.

    python make_figures.py            # everything
    python make_figures.py corners    # only a group (corners | diag | over |
                                      #   colormaps | scenC | abc)
"""
import os
import subprocess
import sys

os.makedirs("figures", exist_ok=True)
GROUPS = sys.argv[1:] or ["corners", "diag", "over", "colormaps", "scenC", "abc"]

import numpy as np           # noqa: E402
import core as C             # noqa: E402
import scenarios as S        # noqa: E402
import pe                    # noqa: E402
import plots                 # noqa: E402
import scenC                 # noqa: E402

x_true = pe.X_TRUE
print("building scenarios A, B, C ...")
scs = {k: S.build_scenario(k, verbose=False) for k in ["A", "B"]}
scC = scenC.build_scenario_C(verbose=False)

if "corners" in GROUPS:
    exact_mle = {k: {mk: pe.noise_profile_mle(scs[k], mk) for mk in plots.MODELS}
                 for k in ["A", "B"]}
    for k in ["A", "B"]:
        ch = plots.load_chains(k)
        # scenario A: pure linearized overlays (they work there);
        # scenario B: Godambe-White noise-sector prediction for the two
        # non-convolved models (the linearized ones leave their validity)
        npred = ({mk: exact_mle[k][mk] for mk in ("bare", "psd")}
                 if k == "B" else None)
        plots.corner_block(scs[k], ch, x_true, "key",
                           f"figures/corner_key_{k}.png", title=f"Scenario {k}",
                           noise_pred=npred)
        plots.corner_block(scs[k], ch, x_true, "noise",
                           f"figures/corner_noise_{k}.png", title=f"Scenario {k}",
                           exact_noise_mle=exact_mle[k])
        plots.corner_block(scs[k], ch, x_true, "key",
                           f"figures/corner_fullkey_{k}.png", models=["full"],
                           title=f"Scenario {k} — full covariance")
        plots.corner_block(scs[k], ch, x_true, "joint",
                           f"figures/corner_full13_{k}.png", models=["full"],
                           title=f"Scenario {k} — full covariance")
    print("corners A/B done")

if "diag" in GROUPS:
    sand = {k: {mk: pe.noise_sandwich(scs[k], mk) for mk in plots.MODELS}
            for k in ["A", "B"]}
    scs3 = {"A": scs["A"], "B": scs["B"], "C": scC}
    plots.fig_upsilon_xi(scs3, sand, "figures/upsilon_xi.png")
    plots.fig_upsxi_heatmap(scs, sand, "figures/upsilon_xi_heatmap.png")
    print("Upsilon/Xi figures done")

if "over" in GROUPS:
    plots.fig_overview({"A": scs["A"], "B": scs["B"], "C": scC},
                       "figures/overview.png")
    print("overview done")

# standalone scripts (each rebuilds what it needs)
SCRIPTS = {"colormaps": "fig_cov_colormap.py",
           "scenC": "plot_scenC.py",
           "abc": "plot_ABC.py"}
for g, script in SCRIPTS.items():
    if g in GROUPS:
        print(f"running {script} ...")
        subprocess.run([sys.executable, script], check=True)

print("all requested figure groups regenerated in figures/")
