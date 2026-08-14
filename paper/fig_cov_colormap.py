"""Colormaps of the windowed frequency-domain covariance, scenarios A, B, C.

Shown as the correlation magnitude |Q_jk| / sqrt(Q_jj Q_kk) (log10), which
normalizes out the PSD dynamic range and exposes the window-induced structure:
  A (two long gaps, 0.3 h lobes): a smooth near-diagonal leakage halo of width
    ~ 1/(1 h), plus beating from the two-gap separation;
  B (twelve short hourly gaps): a comb — discrete sideband diagonals at
    multiples of 1/(1 h) = 0.278 mHz — and a bright cross at the TDI2 null
    (~30 mHz), whose bins contain only leaked power and are therefore highly
    correlated with the rest of the band;
  C (drastic 150 s / 750 s rectangular comb): dense alias diagonals at
    multiples of 1/750 s = 1.33 mHz — the structure the exact solvers unmix.
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import plots  # global style
import core as C
import scenarios as S
import scenC

ZOOM = (1.0, 4.0)          # [mHz] zoom window (shows the comb diagonals crisply)
F_NULL = 1e3 / (4 * C.AVG_D) / 1e3   # ~0.02998 Hz

fmhz = C.fb * 1e3
ext_full = [fmhz[0], fmhz[-1], fmhz[0], fmhz[-1]]

fig, axes = plt.subplots(3, 2, figsize=(12.6, 16.2),
                         gridspec_kw=dict(width_ratios=[1, 1]))
ims = []
for r, key in enumerate(["A", "B", "C"]):
    sc = (S.build_scenario(key, verbose=False) if key in ("A", "B")
          else scenC.build_scenario_C(verbose=False))
    Q = sc["truth"]["Q"]
    d = np.sqrt(np.real(np.diag(Q)))
    R = np.abs(Q) / np.outer(d, d)
    logR = np.log10(np.clip(R, 1e-12, None))

    axF, axZ = axes[r]
    im = axF.imshow(logR, origin="lower", extent=ext_full, cmap="magma",
                    vmin=-4, vmax=0, interpolation="antialiased", aspect="auto")
    ims.append(im)
    axF.set_ylabel(f"Scenario {key}\n" + r"$f$ [mHz]", fontsize=12)
    axF.set_title(r"$\log_{10}|Q_{jk}|/\sqrt{Q_{jj}Q_{kk}}$ — full band" if r == 0 else "",
                  fontsize=11)
    # mark zoom region and the TDI2 null
    axF.add_patch(Rectangle((ZOOM[0], ZOOM[0]), ZOOM[1]-ZOOM[0], ZOOM[1]-ZOOM[0],
                            fill=False, ec="w", lw=1.0, ls="--"))
    axF.axvline(F_NULL * 1e3 * 0 + 29.98, color="w", lw=.5, ls=":", alpha=.7)
    axF.axhline(29.98, color="w", lw=.5, ls=":", alpha=.7)
    axF.text(29.6, 2, "TDI2 null", color="w", fontsize=8, rotation=90, va="bottom")

    mz = (fmhz >= ZOOM[0]) & (fmhz <= ZOOM[1])
    iz = np.where(mz)[0]
    ext_z = [fmhz[iz[0]], fmhz[iz[-1]], fmhz[iz[0]], fmhz[iz[-1]]]
    axZ.imshow(logR[np.ix_(iz, iz)], origin="lower", extent=ext_z, cmap="magma",
               vmin=-4, vmax=0, interpolation="nearest", aspect="auto")
    axZ.set_title(f"zoom [{ZOOM[0]:.0f}, {ZOOM[1]:.0f}] mHz" if r == 0 else "",
                  fontsize=11)
    if r == 2:
        axF.set_xlabel(r"$f$ [mHz]", fontsize=12)
        axZ.set_xlabel(r"$f$ [mHz]", fontsize=12)
    lab = dict(A="two long gaps (0.3 h lobes)", B="twelve short hourly gaps",
               C="drastic comb (150 s / 750 s, rectangular)")[key]
    axZ.text(0.03, 0.965, lab, transform=axZ.transAxes, color="w", fontsize=10,
             va="top")

cb = fig.colorbar(ims[0], ax=axes, fraction=0.035, pad=0.02,
                  label=r"$\log_{10}$ |bin-bin correlation|")
fig.savefig("figures/cov_colormap_ABC.png", dpi=180, bbox_inches="tight")
fig.savefig("figures/cov_colormap_ABC.pdf", bbox_inches="tight")
print("saved figures/cov_colormap_ABC.png/.pdf")
