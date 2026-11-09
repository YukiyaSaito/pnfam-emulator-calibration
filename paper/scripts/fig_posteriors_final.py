#!/usr/bin/env python3
"""Final posteriors, one panel per functional: Full set = filled contours
(68/95%), the adopted ten-nucleus set (L10 for every functional) = dashed, Set A =
dotted; marginals on top and right. Student-t runs.

Density: bayes_plot.boundary_kde with each run's prior spec (taper divided
out, reflection at the domain edges), bandwidth = Scott (as in posterior_A).
Shaded: V0 below the functional's stability limit and C1s < 50.

Runs: final calibrations (test emulator errors, Q term), results/<F>/*_sem21001;
Set A = the final-settings Set A runs (Student-t, 2026-10-19). Output: ../figs/fig_posteriors_final.{pdf,png}
"""
import os
import sys

import numpy as np
import arviz as az
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import repo_paths as RP  # noqa: E402  (all data paths of the repository)
sys.path.insert(0, RP.CALIBRATION)
from bayes_plot import boundary_kde
from bayes_mcmc import read_prior_spec_from_manifest

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs", "fig_posteriors_final")

# ---- knobs ------------------------------------------------------------------
LIK = "t4gtt4"                    # headline likelihood tag in the run names
PANELS = [  # (functional key, [(label, run dir, style)])
    ("SKOP", [("Set A", RP.posterior("SKOP", "setA", LIK), "faded"),
              ("L10", RP.posterior("SKOP", "L10", LIK), "dashed"),
              ("Full", RP.posterior("SKOP", "full", LIK), "fill")]),
    ("SLY4", [("Set A", RP.posterior("SLY4", "setA", LIK), "faded"),
              ("L10", RP.posterior("SLY4", "L10", LIK), "dashed"),
              ("Full", RP.posterior("SLY4", "full", LIK), "fill")]),
    ("HFB1", [("Set A", RP.posterior("HFB1", "setA", LIK), "faded"),
              ("G10 (= L10)", RP.posterior("HFB1", "L10", LIK), "dashed"),
              ("Full", RP.posterior("HFB1", "full", LIK), "fill")]),
]
LEVELS = (0.68, 0.95)
ME_POINT = (-346.352, 128.279)   # Mustonen & Engel (2016), SkO'
XLIM, YLIM = (-460, 0), (50, 180)
BW_FACTOR = 1.0
STYLE = {"fill":    dict(fill=(0.10, 0.65), ls="solid",   lw=(0.7, 1.0), levels=(0, 1), alpha=0.6),
         "dashed":  dict(fill=None,         ls="dashed",  lw=(0.7, 1.0), levels=(0, 1), alpha=0.9),
         "dashdot": dict(fill=None,         ls="dashdot", lw=(0.7, 1.0), levels=(0, 1), alpha=0.9),
         "faded":   dict(fill=None,         ls="dotted",  lw=(0.7, 0.7), levels=(0, 1), alpha=0.7)}
# ------------------------------------------------------------------------------


def mass_levels(f, dv, dc, levels):
    d = np.sort(f[np.isfinite(f)].ravel())[::-1]
    cum = np.cumsum(d) * dv * dc
    cum /= cum[-1]
    return sorted(d[np.searchsorted(cum, lv)] for lv in levels)


fig = plt.figure(figsize=(ps.TEXT_WIDTH, ps.TEXT_WIDTH * 0.40))
gs = fig.add_gridspec(2, 8, width_ratios=(4, 1, 0.5, 4, 1, 0.5, 4, 1), height_ratios=(1, 4),
                      hspace=0.05, wspace=0.05)
vg = np.linspace(-500, 0, 501)
cg = np.linspace(0, 250, 251)
for ip, (func, runs) in enumerate(PANELS):
    color = ps.OKABE_ITO[func]
    c0 = 3 * ip
    ax = fig.add_subplot(gs[1, c0])
    ax_top = fig.add_subplot(gs[0, c0], sharex=ax)
    ax_right = fig.add_subplot(gs[1, c0 + 1], sharey=ax)
    edge = None
    for label, run, style in runs:
        st = STYLE[style]
        post = az.from_netcdf(os.path.join(run, "trace.nc")).posterior
        v = post["vpair"].values.ravel()
        c = post["cs0"].values.ravel()
        spec = read_prior_spec_from_manifest(os.path.join(run, "manifest.txt"))
        edge = spec["vpair"]["lower"]
        est = boundary_kde(v, c, spec=spec, vg=vg, cg=cg, bw_factor=BW_FACTOR)
        f = np.where(np.isfinite(est["f"]), est["f"], 0.0)
        lv = mass_levels(f, vg[1] - vg[0], cg[1] - cg[0], LEVELS)
        if st["fill"]:
            fl = np.linspace(lv[0], f.max(), 9)
            al = np.linspace(*st["fill"], 8)
            ax.contourf(est["VV"], est["CC"], f, levels=fl, colors=[to_rgba(color, a) for a in al])
        ax.contour(est["VV"], est["CC"], f, levels=[lv[i] for i in st["levels"]],
                   colors=[to_rgba(color, st["alpha"])], linewidths=st["lw"], linestyles=st["ls"])
        ax.plot([], [], color=color, lw=1.0, ls=st["ls"], label=label)
        mv = np.nan_to_num(est["mv"]); mc = np.nan_to_num(est["mc"])
        mv, mc = mv / mv.max(), mc / mc.max()
        if style in ("faded", "fill"):
            ax_top.fill_between(vg, 0, mv, color=color, alpha=0.12 if style == "faded" else 0.25, lw=0)
            ax_right.fill_betweenx(cg, 0, mc, color=color, alpha=0.12 if style == "faded" else 0.25, lw=0)
        ax_top.plot(vg, mv, color=color, lw=0.8, ls=st["ls"])
        ax_right.plot(mc, cg, color=color, lw=0.8, ls=st["ls"])
    ax.axvspan(XLIM[0], edge, color="0.90", zorder=0)
    ax.axhspan(YLIM[0], 50.0, color="0.90", zorder=0)
    ax_top.axvspan(XLIM[0], edge, color="0.90", zorder=0)
    if func == "SKOP":
        ax.plot(*ME_POINT, marker="*", ms=8, color=color, mec="k", mew=0.4, ls="none",
                label=r"Mustonen \& Engel (2016), SkO$'$")
    ax.set_xlim(XLIM); ax.set_ylim(YLIM)
    ax.set_xlabel(r"$V_0$ (MeV\,fm$^3$)")
    if ip == 0:
        ax.set_ylabel(r"$C_1^s$ (MeV\,fm$^3$)")
    else:
        ax.tick_params(labelleft=False)
    ax.text(0.04, 0.96, ps.LABELS[func], transform=ax.transAxes, ha="left", va="top", color=color)
    for a in (ax_top, ax_right):
        a.axis("off")
    ax_top.set_ylim(0, 1.05); ax_right.set_xlim(0, 1.05)

from matplotlib.lines import Line2D
handles = [Line2D([], [], color="0.3", ls="dotted", lw=1.0, label="Set A"),
           Line2D([], [], color="0.3", ls="dashed", lw=1.0, label="L10"),
           Line2D([], [], color="0.3", ls="solid", lw=1.0, label="Full"),
           Line2D([], [], marker="*", ms=8, color=ps.OKABE_ITO["SKOP"], mec="k", mew=0.4, ls="none",
                  label=r"Mustonen \& Engel (2016), SkO$'$")]
fig.legend(handles=handles, frameon=False, fontsize=7, loc="lower center", ncol=4,
           bbox_to_anchor=(0.5, -0.16), columnspacing=1.5, handlelength=2.2)
ps.save(fig, OUT)
