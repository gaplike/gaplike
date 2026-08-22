"""Dump the paper's leading-order Upsilon/Xi diagnostics for every scenario
and model tier as JSON (results/upsxi_dump.json) — the numbers behind the
talk animations in notebooks/anim_upsilon_xi.py and the interactive explorer
in assets/upsilon_xi_explorer.html.

    Upsilon = true scatter of the estimate / quoted posterior width
    Xi      = quoted posterior width / width of the exact analysis

Analytic only (ApproxModel + the Godambe/White sandwich): no sampling, but
scenarios A/B need lisabeta for the waveform derivatives.  Run from paper/.
"""
import json

import numpy as np

import scenarios as S
import scenC
import pe

MODELS = ["full", "diag", "bare", "psd"]

out = {}
scs = {k: S.build_scenario(k, verbose=False) for k in ["A", "B"]}
scs["C"] = scenC.build_scenario_C(verbose=False)
for key, sc in scs.items():
    row = {"snr": float(sc["snr"])}
    for mk in [m for m in MODELS if m in sc["models"]]:
        m = sc["models"][mk]
        ls_, Var = pe.noise_sandwich(sc, mk)
        ups_sw = np.sqrt(np.abs(np.diag(Var)) + ls_**2) / np.sqrt(np.diag(m.cov_n))
        row[mk] = dict(
            upsilon_s=[float(x) for x in m.upsilon_s],
            xi_s=[float(x) for x in m.xi_s],
            upsilon_n=[float(x) for x in m.upsilon_n],
            xi_n=[float(x) for x in m.xi_n],
            upsilon_n_sandwich=[float(x) for x in ups_sw],
            bias_n_lin=[float(x) for x in m.bias_n],
            lam_star=[float(x) for x in ls_],
            quoted_n=[float(x) for x in np.sqrt(np.diag(m.cov_n))],
            true_n=[float(x) for x in np.sqrt(np.diag(m.cov_true_n))],
        )
    out[key] = row

with open("results/upsxi_dump.json", "w") as f:
    json.dump(out, f, indent=1)
print("wrote results/upsxi_dump.json")
