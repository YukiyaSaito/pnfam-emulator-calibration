#!/usr/bin/env python3
"""Set-A posterior figure: SkO', SLy4 and UNEDF1-HFB in one (V0, C1s) panel
with marginal distributions. Mustonen & Engel (2016) point for SkO' only.

Density: the boundary-aware estimate of bayes_plot.boundary_kde (taper-aware,
reflected at the prior edges; adopted 2026-10-12), read with each run's prior
spec from its manifest. Contours enclose 68% and 95% of the posterior mass.
The lower V0 edge of each functional's domain is drawn as a dotted line in
its colour; C1s < 50 (common to all) is shaded.

Runs: Set A with the FINAL settings (Q term, test emulator errors; GT = Set A six
nuclei), Normal likelihood (Fig. 3)
(user): Sec. III.B compares with the chi^2 fit of Mustonen & Engel and passes the
normal-likelihood theory errors and V0 interval to the set selection. The
Student-t Set A runs remain in Table III and as dotted contours in
fig_posteriors_final.
Output: ../figs/posterior_A.{pdf,png}  (file name as referenced in main.tex).
"""
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
import arviz as az

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import repo_paths as RP  # noqa: E402  (all data paths of the repository)
sys.path.insert(0, RP.CALIBRATION)
from bayes_plot import boundary_kde
from bayes_mcmc import read_prior_spec_from_manifest

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs", "posterior_A")

# ---- knobs ------------------------------------------------------------------
RUNS = [  # (functional key, run directory)
    # FINAL-settings Set A runs (2026-10-19): Q term, test emulator errors, Normal (2026-11-03); GT = Set A six nuclei
    ("SKOP", RP.posterior("SKOP", "setA", "normal")),
    ("SLY4", RP.posterior("SLY4", "setA", "normal")),
    ("HFB1", RP.posterior("HFB1", "setA", "normal")),
]
LEVELS = (0.68, 0.95)           # credible-mass contours
ME_POINT = (-346.352, 128.279)  # Mustonen & Engel (2016), SkO'
XLIM, YLIM = (-460, 0), (50, 180)     # 2026-11-11: same limits as Fig. 5 (fig_posteriors_final.py); was (-470, 0), (40, 180)
C1S_LOWER = 50.0                # common lower bound of the C1s prior
FILL = True                     # graded fill inside the 95% contour (False = lines only)
LINE_ALPHA = 0.6                # transparency of the contour lines
VLINE_YMAX = 0.70               # V_min lines in the main panel end at this axes fraction (below the legend)
BW_FACTOR = 1.0                 # KDE bandwidth / Scott's factor (bayes_plot default 0.5 is wobbly at this size)
OUT_SUFFIX = ""                 # set e.g. "_lines" to keep a variant beside the main file
# ------------------------------------------------------------------------------


def mass_levels(f, dv, dc, levels):
    """Density thresholds enclosing the given posterior masses on the grid."""
    d = np.sort(f[np.isfinite(f)].ravel())[::-1]
    cum = np.cumsum(d) * dv * dc
    cum /= cum[-1]
    return sorted(d[np.searchsorted(cum, lv)] for lv in levels)


fig = plt.figure(figsize=(ps.COLUMN_WIDTH, ps.COLUMN_WIDTH))
gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                      hspace=0.05, wspace=0.05)
ax = fig.add_subplot(gs[1, 0])
ax_top = fig.add_subplot(gs[0, 0], sharex=ax)
ax_right = fig.add_subplot(gs[1, 1], sharey=ax)

vg = np.linspace(-500, 0, 501)
cg = np.linspace(0, 250, 251)
for func, run in RUNS:
    color = ps.OKABE_ITO[func]
    post = az.from_netcdf(os.path.join(run, "trace.nc")).posterior
    v = post["vpair"].values.ravel()
    c = post["cs0"].values.ravel()
    spec = read_prior_spec_from_manifest(os.path.join(run, "manifest.txt"))
    est = boundary_kde(v, c, spec=spec, vg=vg, cg=cg, bw_factor=BW_FACTOR)
    f = np.where(np.isfinite(est["f"]), est["f"], 0.0)
    lv = mass_levels(f, vg[1] - vg[0], cg[1] - cg[0], LEVELS)
    n_bands = 8
    fill_levels = np.linspace(lv[0], f.max(), n_bands + 1)
    alphas = np.linspace(0.10, 0.65, n_bands)
    if FILL:
        ax.contourf(est["VV"], est["CC"], f, levels=fill_levels,
                    colors=[to_rgba(color, a) for a in alphas])
    ax.contour(est["VV"], est["CC"], f, levels=lv, colors=[to_rgba(color, LINE_ALPHA)],
               linewidths=(0.7, 1.0)[-len(lv):], linestyles=("dashed", "solid")[-len(lv):])
    ax.plot([], [], color=color, lw=1.4, label=ps.LABELS[func])   # legend proxy
    vlo = spec["vpair"]["lower"]
    ax.axvline(vlo, color=color, lw=0.8, ls=":", ymax=VLINE_YMAX)   # stops below the legend; full height in the top marginal
    ax_top.axvline(vlo, color=color, lw=0.8, ls=":")
    ax_top.fill_between(vg, 0.0, np.nan_to_num(est["mv"]), color=color, alpha=0.3, lw=0)
    ax_top.plot(vg, est["mv"], color=color, lw=0.8)
    ax_right.fill_betweenx(cg, 0.0, np.nan_to_num(est["mc"]), color=color, alpha=0.3, lw=0)
    ax_right.plot(est["mc"], cg, color=color, lw=0.8)
    print(f"{func}: V0 median {np.median(v):.1f}  C1s median {np.median(c):.1f}  edge {vlo}")

ax.axhspan(YLIM[0], C1S_LOWER, color="0.85", zorder=0)
ax_right.axhspan(YLIM[0], C1S_LOWER, color="0.85", zorder=0)
ax.plot(*ME_POINT, marker="*", ms=9, color=ps.OKABE_ITO["SKOP"], mec="k", mew=0.4,
        ls="none", label=r"Mustonen \& Engel (2016)")   # SkO' stated in the caption; shorter label keeps the legend clear of the V_min lines
ax.set_xlim(XLIM); ax.set_ylim(YLIM)
ax.set_xlabel(r"$V_0$ (MeV\,fm$^3$)")
ax.set_ylabel(r"$C_1^s$ (MeV\,fm$^3$)")
ax.legend(frameon=False, fontsize=7, loc="upper right")   # 2026-11-11: moved from upper left, where the dotted V_min lines crossed the legend text
for a in (ax_top, ax_right):
    a.axis("off")
ax_top.set_ylim(bottom=0)
ax_right.set_xlim(left=0)

ps.save(fig, OUT + OUT_SUFFIX)
