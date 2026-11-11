#!/usr/bin/env python3
"""Fig. 9 (adopted 2026-11-03): two-panel version of fig_n126_halflives_compare.py, same data.

(a) our three calibrated functionals: line + markers at the posterior-median couplings
    (log10T_at_median), 16-84% predictive band (p16-p84), measured Pt204 / Hg206.
(b) log10(T_model / T_SkO') relative to our SkO' at its posterior-median couplings, at the 13 even-Z
    nuclei only (our predictions exist for even Z only): our SLy4 and UNEDF1-HFB (medians), the
    published DD-PC1, SkO' (Ney et al.), RHB-1(D3C*) updated and FRDM, our SkO' 68% band around 0,
    and the measured points.
Inputs and loaders are copied from fig_n126_halflives_compare.py (frozen/n126_final/quadratic,
FRIB_projects/beta_decay/data, config.HL_EXP / HL_ERR). Units: log10 of seconds.
Writes ../figs/fig_n126_halflives_ratio.{pdf,png}; fig_n126_halflives_compare.* (the previous Fig. 9) is kept.
"""
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedLocator, FuncFormatter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps   # LaTeX fonts, colours, ps.save()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import repo_paths as RP  # noqa: E402  (all data paths of the repository)
sys.path.insert(0, RP.CALIBRATION)
import config

PRED = RP.PAPER_DATA   # predictions_<F>_batch30001.tsv
PUB = RP.LITERATURE
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs", "fig_n126_halflives_ratio")

# ---- knobs (styles as in fig_n126_halflives_compare.py) -----------------------
FUNCTIONALS = ["SKOP", "SLY4", "HFB1"]
REF = "SKOP"                                   # reference of panel (b)
COLORS, LABELS, MARKERS = ps.OKABE_ITO, ps.LABELS, ps.MARKERS
MSIZE = {"SKOP": 5.0, "SLY4": 4.6, "HFB1": 5.6}
LSTYLE = {"SKOP": "-", "SLY4": "--", "HFB1": "-."}
MZORDER = {"SLY4": 4.8, "SKOP": 5.0, "HFB1": 5.2}
LINE_LW, MEC, MEW = 1.0, "white", 0.8
ZOF = {"Ba182": 56, "Ce184": 58, "Nd186": 60, "Sm188": 62, "Gd190": 64, "Dy192": 66, "Er194": 68,
       "Yb196": 70, "Hf198": 72, "W200": 74, "Os202": 76, "Pt204": 78, "Hg206": 80}
ZGRID = np.array(sorted(ZOF.values()))
MEASURED = ["Pt204", "Hg206"]
ZMIN, ZMAX, NN = 54, 82, 126
PUBSTYLE = [   # label, loader key, grey, linestyle, marker (as in Fig. 9)
    ("DD-PC1",              "DDPC1", "0.20", "-",           "s"),
    ("SkO$'$ (Ney et al.)", "Evan",  "0.45", (0, (3, 1.5)), "o"),
    ("RHB-1(D3C*) updated", "RQRPA", "0.30", (0, (1, 1)),   "x"),
    ("FRDM",                "FRDM",  "0.45", (0, (1, 1)),   "d"),
]
PUB_LW, PUB_MS = 0.7, 3.5
BAND_ALPHA = 0.22
FIGWIDTH, FIGHEIGHT = ps.COLUMN_WIDTH, 5.4
HEIGHT_RATIOS = (1.35, 1.0)
YLIM_A = (-4.0, 6.5)                           # log10 s, as in Fig. 9
YLIM_B = (-1.75, 2.3)                          # decades; top part left free for the legend
# ------------------------------------------------------------------------------


def load_ours(func):
    path = os.path.join(PRED, "predictions_%s_batch30001.tsv" % func)
    rows = [l.rstrip("\n").split("\t") for l in open(path) if not l.startswith("#")]
    head, rows = rows[0], rows[1:]
    col = {k: i for i, k in enumerate(head)}
    d = {"Z": ZGRID.astype(float)}
    for k in ("p16", "p50", "p84", "log10T_at_median"):
        d[k] = np.full(len(ZGRID), np.nan)
    for r in rows:
        i = int(np.where(ZGRID == ZOF[r[0]])[0][0])
        for k in ("p16", "p50", "p84", "log10T_at_median"):
            d[k][i] = float(r[col[k]])
    return d


def load_pub(key):
    """-> (Z, log10 T [log10 s]) for N = 126, ZMIN <= Z <= ZMAX, finite values only."""
    if key == "DDPC1":
        a = np.genfromtxt(os.path.join(PUB, "data_DDPC1.txt"), comments="#")
        n, z, y = a[:, 0], a[:, 1], a[:, 5]
    elif key in ("Evan", "FRDM"):
        a = np.genfromtxt(os.path.join(PUB, "data_%s.txt" % key), comments="#")
        n, z, y = a[:, 0], a[:, 1], a[:, 2]
    elif key == "RQRPA":
        a = np.genfromtxt(os.path.join(PUB, "half-lives.dat"), comments="#")
        z, n, y = a[:, 0], a[:, 1], np.log10(a[:, 2])
    else:
        raise KeyError(key)
    m = (np.abs(n - NN) < 0.5) & (z > ZMIN - 0.5) & (z < ZMAX + 0.5) & np.isfinite(y)
    o = np.argsort(z[m])
    return z[m][o], y[m][o]


ours = {f: load_ours(f) for f in FUNCTIONALS}
ref = ours[REF]["log10T_at_median"]            # log10 s at the 13 even-Z nuclei

fig, (axa, axb) = plt.subplots(2, 1, sharex=True, figsize=(FIGWIDTH, FIGHEIGHT),
                               gridspec_kw=dict(height_ratios=HEIGHT_RATIOS, hspace=0.06))

# ---- (a) absolute half-lives, this work + experiment ----
handles_a = []
for func in FUNCTIONALS:
    d, c = ours[func], COLORS[func]
    ok = ~np.isnan(d["log10T_at_median"])
    axa.fill_between(d["Z"], d["p16"], d["p84"], where=ok, color=c, alpha=BAND_ALPHA, lw=0, zorder=2)
    axa.plot(d["Z"], d["log10T_at_median"], LSTYLE[func], color=c, lw=LINE_LW, zorder=3)
    axa.plot(d["Z"], d["log10T_at_median"], ls="none", marker=MARKERS[func], ms=MSIZE[func],
             mfc=c, mec=MEC, mew=MEW, zorder=MZORDER[func])
    handles_a.append(Line2D([], [], ls=LSTYLE[func], color=c, lw=LINE_LW, marker=MARKERS[func],
                            ms=MSIZE[func], mfc=c, mec=MEC, mew=MEW, label=LABELS[func]))
zm = np.array([ZOF[n] for n in MEASURED])
T = np.array([config.HL_EXP[n] for n in MEASURED])     # s
dT = np.array([config.HL_ERR[n] for n in MEASURED])    # s
yexp, eexp = np.log10(T), dT / (T * math.log(10.0))
handles_a.append(axa.errorbar(zm, yexp, yerr=eexp, fmt="o", color="k", mfc="k", mec="white", mew=0.8,
                              ms=6.0, capsize=2, lw=0.8, zorder=6, label="Experiment"))
axa.set_ylabel(r"$T_{1/2}$ (s)")
_k = np.arange(math.floor(YLIM_A[0]), math.ceil(YLIM_A[1]) + 1)
axa.yaxis.set_major_locator(FixedLocator(_k[_k % 2 == 0]))
axa.yaxis.set_major_formatter(FuncFormatter(lambda y, _: r"$10^{%d}$" % round(y)))
axa.yaxis.set_minor_locator(FixedLocator([k + math.log10(m) for k in _k for m in range(1, 10)
                                          if YLIM_A[0] <= k + math.log10(m) <= YLIM_A[1]]))
axa.set_ylim(*YLIM_A)
axa.legend(handles=handles_a, loc="upper left", fontsize=7.5, frameon=False, handlelength=2.6,
           handletextpad=0.4, borderpad=0.4, labelspacing=0.3)
axa.text(0.95, 0.04, r"$N = 126$", transform=axa.transAxes, ha="right", va="bottom")
axa.text(0.97, 0.96, "(a)", transform=axa.transAxes, ha="right", va="top")

# ---- (b) ratio to our SkO' at its median couplings, even Z only ----
c = COLORS[REF]
dref = ours[REF]
axb.fill_between(ZGRID, dref["p16"] - ref, dref["p84"] - ref, color=c, alpha=BAND_ALPHA, lw=0, zorder=2)
axb.axhline(0.0, color=c, lw=LINE_LW, zorder=3)
handles_b = [Line2D([], [], color=c, lw=6, alpha=BAND_ALPHA + 0.1, label=r"SkO$'$ (this work), 68\%")]
for func in FUNCTIONALS:
    if func == REF:
        continue
    d, cf = ours[func], COLORS[func]
    r = d["log10T_at_median"] - ref
    axb.plot(ZGRID, r, LSTYLE[func], color=cf, lw=LINE_LW, marker=MARKERS[func], ms=MSIZE[func] - 0.6,
             mfc=cf, mec=MEC, mew=MEW, zorder=MZORDER[func])
rng = {}
for label, key, grey, ls, mk in PUBSTYLE:
    z, y = load_pub(key)
    sel = np.isin(z.astype(int), ZGRID)
    zz, yy = z[sel].astype(int), y[sel]
    r = yy - ref[np.searchsorted(ZGRID, zz)]
    rng[label] = (len(zz), np.nanmin(r), np.nanmax(r))
    ms_, mew_ = (PUB_MS + 1.5, 1.0) if mk == "x" else (PUB_MS, 0.6)
    h, = axb.plot(zz, r, linestyle=ls, color=grey, lw=PUB_LW, marker=mk, ms=ms_, mfc="white", mec=grey,
                  mew=mew_, label=label, zorder=2.5)
    handles_b.append(h)
rexp = yexp - ref[np.searchsorted(ZGRID, zm)]
axb.errorbar(zm, rexp, yerr=eexp, fmt="o", color="k", mfc="k", mec="white", mew=0.8, ms=6.0, capsize=2,
             lw=0.8, zorder=6)
axb.set_ylabel(r"$\log_{10}(T_{1/2}/T_{1/2}^{\mathrm{SkO}'})$")
axb.set_xlabel(r"$Z$")
axb.set_xlim(ZMIN, ZMAX)
axb.set_xticks(range(ZMIN, ZMAX + 1, 4))
axb.set_ylim(*YLIM_B)
# 2026-11-11 (user): single column; the 2-column legend let "RHB-1(D3C*) updated" stick out past the axes
axb.legend(handles=handles_b, loc="upper left", fontsize=6.5, frameon=False, ncol=1, handlelength=2.4,
           handletextpad=0.4, borderpad=0.3, labelspacing=0.25)
axb.text(0.03, 0.04, "(b)", transform=axb.transAxes, ha="left", va="bottom")

ps.save(fig, OUT)

print("panel (b): log10(T/T_SkO'(ours, median couplings)) at even Z; n points, min, max:")
for f in FUNCTIONALS:
    if f != REF:
        r = ours[f]["log10T_at_median"] - ref
        print("  %-22s n=%2d  %+.2f .. %+.2f" % (LABELS[f], np.sum(np.isfinite(r)), np.nanmin(r), np.nanmax(r)))
for label, (n, lo, hi) in rng.items():
    print("  %-22s n=%2d  %+.2f .. %+.2f" % (label, n, lo, hi))
print("  %-22s Z=78 %+.2f  Z=80 %+.2f" % ("Experiment", rexp[0], rexp[1]))
print("  SkO' 68%% band half-width: %.2f .. %.2f" % (np.nanmin((dref["p84"] - dref["p16"]) / 2),
                                                    np.nanmax((dref["p84"] - dref["p16"]) / 2)))
