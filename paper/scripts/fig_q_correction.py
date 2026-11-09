#!/usr/bin/env python3
"""Paper copy (draft/scripts, LaTeX fonts via paper_style; 2026-10-21) of analysis/paper_figs/fig_q_correction.py.
Q-value correction, one functional (paper figure). Even-even nuclei only.

(a) Q_HFB - Q_exp versus N-Z, before the correction;
(b) Q_corr - Q_exp versus Q_exp, after it.
Q_corr = Q_HFB - [a + b(N-Z) + c(N+Z)], with a, b, c read from the header of
Qsurvey/results/q_shift_<F>.txt (fit to all 154 measured even-even nuclei, none excluded).
Mean, sd, histogram and Gaussian fit use the measured nuclei only; the AME2020
extrapolated ones are shown as grey circles with the same correction applied.
Calibration pool = the nuclei listed in q_shift_<F>.txt.

Simple by design -- edit the knobs below.
"""
import os
import re

import sys

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import repo_paths as RP  # noqa: E402  (all data paths of the repository)
import paper_style as ps   # LaTeX fonts, colours, ps.save()

RES = RP.QCORR_RESULTS
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs", "fig_q_correction")

# ---- knobs ------------------------------------------------------------------
FUNC = "HFB1"
TITLE = ps.LABELS[FUNC]
HIST_LIM = 4.5           # inset x range +-HIST_LIM [MeV]
FIGSIZE = (ps.COLUMN_WIDTH, 5.4)   # single column, two stacked panels; fonts at the paper's 9 pt
C_MEAS, C_BAND, C_FIT = "#0072B2", "#009E73", "#D55E00"   # measured '+', mean/sd band, Gaussian fit
POOL_MARKER = "s"        # 2026-10-21 (user): squares for the calibration-pool nuclei
# ------------------------------------------------------------------------------


def load(path):
    """dQ_<F>.txt rows -> (measured, extrapolated) dicts of arrays; Qexp, dQ in MeV."""
    rows = [t for t in (l.split() for l in open(path)) if t and t[0].isdigit()]
    out = []
    for src in ("AME", "AME#"):
        r = [t for t in rows if t[11] == src]
        out.append({k: np.array([float(t[i]) for t in r]) for k, i in
                    (("Z", 0), ("N", 1), ("Qexp", 4), ("dQ", 6))})
    return out


def gauss(x, amp, mu, sig):
    return amp * np.exp(-0.5 * ((x - mu) / sig) ** 2)


# correction coefficients and calibration pool from q_shift_<F>.txt
qshift = os.path.join(RES, "q_shift_%s.txt" % FUNC)
a, b, c = [float(re.search(k + r"=([+-]?[\d.]+)", open(qshift).read()).group(1)) for k in "abc"]
pool = {(int(t[0]), int(t[1])) for t in (l.split() for l in open(qshift)) if t and t[0].isdigit()}

meas, extr = load(os.path.join(RES, "dQ_%s.txt" % FUNC))   # dQ_<F>.txt lists the AME# nuclei too
for d in (meas, extr):
    d["NZ"] = d["N"] - d["Z"]
    d["res"] = d["dQ"] - (a + b * d["NZ"] + c * (d["N"] + d["Z"]))   # Q_corr - Q_exp [MeV]
    d["pool"] = np.array([(int(z), int(n)) in pool for z, n in zip(d["Z"], d["N"])])


def panel(ax, xkey, ykey, letter):
    y = meas[ykey]
    m, s = y.mean(), y.std()   # population sd (divide by n), as sigma_Q in Qsurvey/q_correction.py and Table I (2026-11-03)
    ax.axhspan(m - s, m + s, color=C_BAND, alpha=0.25, lw=0, zorder=0)
    ax.axhline(0.0, color="k", lw=0.8, zorder=1)
    ax.axhline(m, color=C_BAND, lw=1.2, zorder=1)
    ax.plot(extr[xkey], extr[ykey], "o", mfc="none", mec="#b0b0b0", ms=2.8, mew=0.7, zorder=2,
            label="AME2020 systematics")
    ax.plot(meas[xkey], y, "+", color=C_MEAS, ms=5, mew=0.9, zorder=3, label="measured")
    npool = 0
    for d in (meas, extr):
        ax.plot(d[xkey][d["pool"]], d[ykey][d["pool"]], POOL_MARKER, mfc="none", mec="k", ms=5.5, mew=0.8, zorder=4)
        npool += d["pool"].sum()
    ax.plot([], [], POOL_MARKER, mfc="none", mec="k", ms=5.5, mew=0.8, label="calibration pool")   # counts (102 / 154 / 42) go in the caption
    ax.text(0.965, 0.53, "$%+.2f \\pm %.2f$ MeV" % (m, s), transform=ax.transAxes, fontsize=8,
            ha="right", va="top", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="k", lw=0.6))   # just below the inset
    ax.text(0.03, 0.965, letter, transform=ax.transAxes, ha="left", va="top")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.92), fontsize=7, frameon=False, handletextpad=0.3,
              borderaxespad=0.4, labelspacing=0.3)

    # inset: histogram of the measured values + Gaussian fit to the counts
    ins = ax.inset_axes([0.68, 0.66, 0.30, 0.30])
    bins = np.arange(-HIST_LIM, HIST_LIM + 0.25, 0.25)
    counts, _, _ = ins.hist(y, bins=bins, color=C_MEAS, alpha=0.6, edgecolor="k", lw=0.4)
    xc = 0.5 * (bins[1:] + bins[:-1])
    (amp, mu, sig), _ = curve_fit(gauss, xc, counts, p0=(counts.max(), m, s))
    xx = np.linspace(-HIST_LIM, HIST_LIM, 300)
    ins.plot(xx, gauss(xx, amp, mu, sig), "-", color=C_FIT, lw=1.0)
    ins.set_xlim(-HIST_LIM, HIST_LIM)
    ins.axvline(0.0, color="k", lw=0.6)
    ins.axvline(m, color=C_BAND, lw=1.0)
    ins.set_ylabel("Counts", fontsize=6.5, labelpad=2)
    ins.set_xlabel("MeV", fontsize=6.5, labelpad=1)
    ins.tick_params(labelsize=6, direction="in", pad=2)
    print("%s  measured %+.3f +- %.3f MeV | Gaussian fit mu %+.3f sigma %.3f MeV | pool shown %d of %d"
          % (letter, m, s, mu, abs(sig), npool, len(pool)))


fig, (ax1, ax2) = plt.subplots(2, 1, figsize=FIGSIZE, sharey=True)
panel(ax1, "NZ", "dQ", "(a)")
ax1.text(0.97, 0.03, TITLE, transform=ax1.transAxes, ha="right", va="bottom", color=ps.OKABE_ITO[FUNC])
ax1.set_xlabel("$N - Z$")
ax1.set_ylabel("$Q_\\beta^{\\mathrm{HFB}} - Q_\\beta^{\\mathrm{exp}}$ (MeV)")
ax1.set_xlim(0, 62)
panel(ax2, "Qexp", "res", "(b)")
ax2.set_xlabel("$Q_\\beta^{\\mathrm{exp}}$ (MeV)")
ax2.set_ylabel("$Q_\\beta^{\\mathrm{corr}} - Q_\\beta^{\\mathrm{exp}}$ (MeV)")
ax2.set_xlim(-0.3, 18.6)
ax2.text(0.04, 0.66, "$%+.3f$\n$%+.4f\\,(N-Z)$\n$%+.4f\\,(N+Z)$" % (a, b, c), transform=ax2.transAxes,
         ha="left", va="top", fontsize=8.5, color="k", linespacing=1.15)   # fitted shift, three lines under the legend (back by user request 2026-10-21)
lo, hi = ax1.get_ylim()                 # shared y range; room above the data for the inset
ax1.set_ylim(lo - 0.08 * (hi - lo), hi + 0.60 * (hi - lo))
fig.tight_layout(h_pad=0.8)
ps.save(fig, OUT)
print("fit a=%+.4f b=%+.5f c=%+.5f" % (a, b, c))
