#!/usr/bin/env python3
"""Two-panel emulator-accuracy summary figure for the paper.

(a) half-life emulators: cross-validation error per nucleus (log10 T units)
(b) GT-centroid emulators: cross-validation error per nucleus (MeV)
Reference lines: fitted theory error per functional (from the Set-A anchor
fits) and the experimental errors.

Simple by design -- edit the knobs below.
"""
import glob
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps   # LaTeX fonts, shared colours, ps.save()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import repo_paths as RP  # noqa: E402  (all data paths of the repository)
sys.path.insert(0, RP.CALIBRATION)
import config

CV = RP.CV
# TEST errors (2026-10-19): the certified per-nucleus values from the frozen batch 21001
# (40 test points per nucleus, never trained on; RMS incl. bias) that enter the final likelihood.
TEST_FILE = os.path.join(RP.ERRORS, "sigma_em_frozen_{F}_batch21001.txt")
SHOW_CV_BEHIND = True   # small grey '+' = cross-validation error of the same emulator
# output goes to ../figs (the draft's figure directory)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs",
                   "fig_emulator_summary")

# ---- knobs ------------------------------------------------------------------
# Updated 2026-10-12 for the Q-corrected chain (V7): three functionals, POOL40
# pools, Set-A anchor theory errors (Normal likelihood, Set A).
FUNCTIONALS = ["SKOP", "SLY4", "HFB1"]
COLORS = {"SKOP": "#0072B2", "SLY4": "#D55E00", "HFB1": "#009E73"}  # Okabe-Ito
MARKERS = {"SKOP": "o", "SLY4": "s", "HFB1": "^"}
LABELS = {"SKOP": r"SkO$'$", "SLY4": "SLy4", "HFB1": "UNEDF1-HFB"}
SIGMA_TH_HL = {"SKOP": 0.48, "SLY4": 0.46, "HFB1": 0.31}  # Full-set posterior medians, Student-t (Table III)
SIGMA_TH_GT = {"SKOP": 1.16, "SLY4": 0.79, "HFB1": 0.92}  # MeV, same runs (Table III)
SETS = {"SKOP": "L10_SKOP", "SLY4": "L10_SLY4", "HFB1": "G10_HFB1"}  # filled symbols = members of the adopted (L10) set
POOLS = {"SKOP": "POOL40_SKOP", "SLY4": "POOL40_SLY4", "HFB1": "POOL40_HFB1"}  # only deployed emulators are shown
EXCLUDE_HL = {"Pt200_quad"}               # non-production CV directories
FIGWIDTH = ps.COLUMN_WIDTH
# ------------------------------------------------------------------------------


def mass(nuc):
    return int("".join(c for c in nuc if c.isdigit()))


def pooled_rms(pattern):
    """RMS (incl. bias) of pooled fold validation residuals, or nan."""
    r = []
    for f in sorted(glob.glob(pattern)):
        d = np.loadtxt(f, ndmin=2)
        if d.size == 0:
            continue
        ok = (d[:, 1] > 0.0) & (d[:, 2] > 0.0)
        r.append(d[ok, 2] - d[ok, 1])
    if not r:
        return float("nan")
    r = np.concatenate(r)
    return float(np.sqrt(np.mean(r * r)))


def pooled_rms_log(pattern):
    """Same, but residuals of log10 (for half-lives stored in seconds)."""
    r = []
    for f in sorted(glob.glob(pattern)):
        d = np.loadtxt(f, ndmin=2)
        if d.size == 0:
            continue
        ok = (d[:, 1] > 0.0) & (d[:, 2] > 0.0)
        r.append(np.log10(d[ok, 2]) - np.log10(d[ok, 1]))
    if not r:
        return float("nan")
    r = np.concatenate(r)
    return float(np.sqrt(np.mean(r * r)))



def line_tag(ax, y, ls, color, text, above=True):
    """Legend-style tag (line sample + text) at the right end, just above (or below) y.
    Built as a separate Legend artist so the axes' own legend is kept."""
    from matplotlib.lines import Line2D as _L
    from matplotlib.legend import Legend as _Leg
    frac = ax.transAxes.inverted().transform(ax.transData.transform((0.0, y)))[1]
    leg = _Leg(ax, [_L([], [], color=color, ls=ls, lw=0.8)], [text], frameon=False,
               fontsize=7, loc="lower right" if above else "upper right",
               bbox_to_anchor=(0.97, frac + (0.025 if above else -0.025)), handlelength=1.6, handletextpad=0.4,
               borderpad=0.0, borderaxespad=0.1)
    ax.add_artist(leg)
    return leg


fig, (ax_a, ax_b) = plt.subplots(
    2, 1, figsize=(FIGWIDTH, 4.6), sharex=False,
    gridspec_kw=dict(hspace=0.42))

# ---- panel (a): half-lives ---------------------------------------------------
for func in FUNCTIONALS:
    nuclei = sorted(
        (n for n in os.listdir(os.path.join(CV, func, "HL"))
         if os.path.isdir(os.path.join(CV, func, "HL", n))
         and n not in EXCLUDE_HL
         and n in config.select_hl_nuclei(POOLS[func])),
        key=mass)
    members = set(config.select_hl_nuclei(SETS[func])) if SETS[func] else set()
    test = {}
    for ln in open(TEST_FILE.format(F=func)):
        if "=" in ln and not ln.startswith("#"):
            k, v = ln.split("=")
            test[k.strip()] = float(v)
    for filled in (False, True):
        sel = [n for n in nuclei if (n in members) == filled]
        A = [mass(n) for n in sel]
        if SHOW_CV_BEHIND:
            cv = [pooled_rms_log(os.path.join(
                CV, func, "HL", n, "fold_*", "parity_half_life_val.txt")) for n in sel]
            ax_a.plot(A, cv, "+", ms=3, color="0.6", mew=0.6, zorder=1)
        err = [test[n] for n in sel]
        ax_a.plot(A, err, MARKERS[func], ms=3.5,
                  mfc=COLORS[func] if filled else "none",
                  color=COLORS[func], zorder=2,
                  label=LABELS[func] if not filled else None)
    if SIGMA_TH_HL[func] is not None:
        ax_a.axhline(SIGMA_TH_HL[func], color=COLORS[func], lw=0.8, ls="--")

# median experimental error in log10(T)
sexp = [config.HL_ERR[n] / (math.log(10.0) * config.HL_EXP[n])
        for n in config.HL_EXP]
ax_a.axhline(np.median(sexp), color="0.4", lw=0.8, ls=":")
ax_a.set_yscale("log")
ax_a.set_xlabel(r"$A$")
ax_a.set_ylabel(r"error in $\log_{10}(T_{1/2}/{\rm s})$")
ax_a.text(0.02, 0.05, "(a)", transform=ax_a.transAxes, va="bottom")
# ax_a.text(0.98, 0.04, "filled: in adopted calibration set",
#           transform=ax_a.transAxes, ha="right", fontsize=6, color="0.3")
ax_a.set_ylim(1e-4, 3e1)
from matplotlib.lines import Line2D
from matplotlib.legend import Legend
h_a, l_a = ax_a.get_legend_handles_labels()
if SHOW_CV_BEHIND:
    h_a.append(Line2D([], [], marker="+", color="0.6", ls="", ms=3, mew=0.6)); l_a.append("CV error")
ax_a.legend(h_a, l_a, frameon=False, fontsize=7, loc="upper left", ncol=2,
            columnspacing=0.8, handletextpad=0.3)
line_tag(ax_a, max(v for v in SIGMA_TH_HL.values() if v is not None), "--", "0.3", r"$\sigma_{\rm th}$ (Full)")
line_tag(ax_a, np.median(sexp), ":", "0.4", r"median $\sigma_{\rm exp}$")
# labels next to the reference lines (right end)
_sth = np.mean([v for v in SIGMA_TH_HL.values() if v is not None])

# ---- panel (b): GT centroids -------------------------------------------------
for func in FUNCTIONALS:
    base = os.path.join(CV, func, "GT")
    if not os.path.isdir(base):
        print("no GT-centroid CV for", func, "- skipped in panel (b)")
        continue
    nuclei = sorted(os.listdir(base), key=mass)
    members = ({n.replace("_", "") for n in config.GT_SETS[SETS[func]]}
               if SETS[func] else set())
    for filled in (False, True):
        sel = [n for n in nuclei if (n in members) == filled]
        A = [mass(n) for n in sel]
        err = [pooled_rms(os.path.join(
            base, n, "full", "fold_*", "parity_centroid_val.txt"))
            for n in sel]
        ax_b.plot(A, err, MARKERS[func], ms=3.5,
                  mfc=COLORS[func] if filled else "none",
                  color=COLORS[func])
    if SIGMA_TH_GT[func] is not None:
        ax_b.axhline(SIGMA_TH_GT[func], color=COLORS[func], lw=0.8, ls="--")

# largest experimental uncertainty of the GT centroids (config.GT_E_ERR)
ax_b.axhspan(min(config.GT_E_ERR.values()), max(config.GT_E_ERR.values()), color="0.85", zorder=0)

ax_b.set_yscale("log")
ax_b.set_xlabel(r"$A$")
ax_b.set_ylabel(r"CV error in $\bar{E}_{\rm GT}$ (MeV)")
ax_b.text(0.02, 0.95, "(b)", transform=ax_b.transAxes, va="top")
ax_b.set_ylim(5e-4, 2e1)
_sgt = max(v for v in SIGMA_TH_GT.values() if v is not None)
line_tag(ax_b, _sgt, "--", "0.3", r"$\sigma_{\rm th}$ (Full)")
import math as _m
_ymid = _m.sqrt(min(config.GT_E_ERR.values()) * max(config.GT_E_ERR.values()))
ax_b.text(ax_b.get_xlim()[1] - 4, _ymid, r"$\sigma_{\rm exp}$", ha="right", va="center", fontsize=7, color="black", zorder=10)

ps.save(fig, OUT)
