#!/usr/bin/env python3
"""Paper copy (draft/scripts, LaTeX fonts via paper_style) of analysis/paper_figs/fig_n126_error_budget.py.

Error budget of the N = 126 half-life predictions, per nucleus, all in log10 T units (one panel per functional):
  couplings   = spread of the surface over the posterior (param_sd in the prediction table)
  theory      = fitted theory error (posterior median of sigma_HL_the, same for all nuclei)
  Q value     = Q-correction term 5 sigma_Q / (Q_corr ln 10) (sigma_Q_i in the table)
  surface     = error of the surface step: out-of-sample RMS (surface fitted to batch 30001, evaluated at the
                independent batch-30201 pnFAM points) if available, else the in-sample fit RMS (open marker)
  total (solid) = quadrature sum of the four = standard deviation of the predictive distribution
  68% half-width of the predictive band (dashed): for the variance-matched Student-t(4) draw this is
                0.80 x the standard deviation (t_{0.84,4} sqrt(2/4) = 0.802), which is why it lies below
                the total (2026-10-20, user question).
Inputs: analysis/frozen/n126_final/quadratic/predictions_<F>_batch30001.tsv; traces results/<F>/..._sem21001;
batch tables calibration/<F>/HL/<nuc>/batch{30001,30201}/. Output: ../figs/fig_n126_error_budget.{pdf,png}.
"""
import os
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps   # LaTeX fonts, colours, ps.save()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import repo_paths as RP  # noqa: E402  (all data paths of the repository)
sys.path.insert(0, RP.CALIBRATION)
from predict_from_draws import design, read_batch_table  # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs", "fig_n126_error_budget")
FUNCS = [(F, os.path.join(RP.posterior(F, "full", "studentt"), "trace.nc")) for F in RP.FUNCTIONALS]
NUCS = ["Ba182", "Ce184", "Nd186", "Sm188", "Gd190", "Dy192", "Er194", "Yb196", "Hf198", "W200", "Os202", "Pt204", "Hg206"]
ZOF = {n: 56 + 2 * i for i, n in enumerate(NUCS)}
PRED = RP.PAPER_DATA   # predictions_<F>_batch30001.tsv
# contributions: (marker, colour, label); Okabe-Ito colours and markers not used for the functionals
# (SkO' blue o, SLy4 vermillion s, UNEDF1-HFB bluish green ^); labels as in the caption
STY = {"couplings": ("o", "0.35", "parameter uncertainty"),
       "theory": ("v", "#E69F00", "theory error"),
       "Q value": ("p", "#56B4E9", r"$Q_\beta$ error"),
       "surface": ("D", "#CC79A7", "surface-fit error")}


def load(F, nuc, B):
    d = os.path.join(RP.N126, F, nuc, f"batch{B}")
    pf, hf = os.path.join(d, f"params_batch{B}.txt"), os.path.join(d, f"halflives_batch{B}.txt")
    if not (os.path.exists(pf) and os.path.exists(hf)):
        return None
    P = {int(float(l.split()[0])): (float(l.split()[1]), float(l.split()[2]))
         for l in open(pf) if not l.startswith("#") and l.strip()}
    H = read_batch_table(hf)
    ids = sorted(set(P) & set(H))
    return np.array([P[i][0] for i in ids]), np.array([P[i][1] for i in ids]), np.array([H[i][0] for i in ids])


fig, axes = plt.subplots(1, 3, figsize=(ps.TEXT_WIDTH, 2.5), sharey=True, gridspec_kw=dict(wspace=0.08))
for ax, (F, trace) in zip(axes, FUNCS):
    t = pd.read_csv(os.path.join(PRED, f"predictions_{F}_batch30001.tsv"), sep="\t", comment="#").set_index("nucleus")
    sthe = float(np.median(az.from_netcdf(trace).posterior["sigma_hl_the"].values.ravel()))
    z = np.array([ZOF[n] for n in NUCS])
    half68 = np.array([0.5 * (t.loc[n, "p84"] - t.loc[n, "p16"]) for n in NUCS])
    coup = np.array([t.loc[n, "param_sd"] for n in NUCS])
    qterm = np.array([t.loc[n, "sigma_Q_i"] for n in NUCS])
    surf, oos = [], []
    for n in NUCS:
        a, b = load(F, n, 30001), load(F, n, 30201)
        v, c, y = a
        beta = np.linalg.lstsq(design(v, c, "quadratic"), y, rcond=None)[0]
        if b is not None and len(b[2]) >= 10:
            r = design(b[0], b[1], "quadratic") @ beta - b[2]; oos.append(True)
        else:
            r = design(v, c, "quadratic") @ beta - y; oos.append(False)
        surf.append(np.sqrt(np.mean(r ** 2)))
    surf, oos = np.array(surf), np.array(oos)
    total = np.sqrt(coup ** 2 + sthe ** 2 + qterm ** 2 + surf ** 2)
    ax.plot(z, total, "-", color="k", lw=1.2, label="total", zorder=3)
    # (68% half-width line dropped 2026-10-20, user; the ratio to the total is printed below)
    ax.plot(z, coup, STY["couplings"][0], color=STY["couplings"][1], ms=3.5, label=STY["couplings"][2])
    ax.plot(z, np.full_like(z, sthe, dtype=float), STY["theory"][0], color=STY["theory"][1], ms=3.5, label=STY["theory"][2])
    ax.plot(z, qterm, STY["Q value"][0], color=STY["Q value"][1], ms=3.5, label=STY["Q value"][2])
    ax.plot(z[oos], surf[oos], STY["surface"][0], color=STY["surface"][1], ms=3.5, label=STY["surface"][2])
    if (~oos).any():
        ax.plot(z[~oos], surf[~oos], STY["surface"][0], color=STY["surface"][1], ms=3.5, mfc="white")
    ax.set_yscale("log"); ax.set_ylim(1e-5, 4)
    ax.set_xlabel("$Z$"); ax.set_xticks(range(56, 82, 4))
    ax.grid(alpha=0.25, ls=":")
    ax.text(0.04, 0.96, ps.LABELS[F], transform=ax.transAxes, ha="left", va="top", color=ps.OKABE_ITO[F])
    print(f"{F}: theory {sthe:.3f}; couplings {coup.min():.3f}-{coup.max():.3f}; Q term {qterm.min():.3f}-{qterm.max():.3f}; "
          f"surface {surf.min():.4f}-{surf.max():.4f} ({'out-of-sample' if oos.all() else 'in-sample'}); "
          f"total {total.min():.2f}-{total.max():.2f}; 68% half-width {half68.min():.2f}-{half68.max():.2f}; "
          f"ratio half68/total {np.min(half68/total):.2f}-{np.max(half68/total):.2f}")
axes[0].set_ylabel(r"uncertainty in $\log_{10}(T_{1/2}/{\rm s})$")
axes[0].legend(fontsize=6.5, frameon=False, loc="lower left", handlelength=1.8, labelspacing=0.3)
fig.tight_layout()
ps.save(fig, OUT)
