#!/usr/bin/env python3
"""Paper copy (draft/scripts, LaTeX fonts via paper_style) of analysis/paper_figs/fig_n126_halflives_compare.py.
N = 126 beta-decay half-lives: our three calibrated functionals overlaid on
the published curves of Fig. 3 of Ravlic, Saito, Nazarewicz, PRC 113, 045802 (2026).

Ours (colour, markers, 16-84% band): Q-term Student-t quadratic-surface predictions
of batch 30001, calibration/analysis/frozen/n126_qterm/quadratic/predictions_<F>_batch30001.tsv
(band p16-p84 of log10 T [log10 s]; line + markers at log10T_at_median = surface
f(median V0, median C1s), no error terms, changed 2026-10-15 from p50); missing nuclei are gaps (UNEDF1-HFB on 2026-10-15:
Nd186, Sm188, Gd190, Hg206 pending).  Even-even nuclei only.

Published (thin grey/black lines, no bands; odd and even Z, N = 126, Z = 54-82).
Since 2026-10-15 only DD-PC1, SkO' (Ney et al.) and RHB-1(D3C*) updated are drawn (SHOW_ALL_PUB),
files in FRIB_projects/beta_decay/data/:
  data_DDPC1.txt, data_DDPCX.txt  columns N Z E(beta) beta2 Q HL_log10 FF_percent  -> HL_log10 [log10 s]
                                  (Z = 81 is NaN in both files and is dropped)
  data_Evan.txt     SkO' (Ney et al.)   columns N Z log10T [log10 s]
  data_FRDM.txt     FRDM                columns N Z log10T [log10 s]
  data_Marketin.txt RHB-1(D3C*)         columns N Z log10T [log10 s]; values >= 100 are a
                                        placeholder for stable / no decay and are dropped
  half-lives.dat    Ref. 72             columns Z N T_RQRPA ... T_RQTBA ... (T in s; even Z only)
                    RQRPA -> "RHB-1(D3C*) updated", RQTBA -> "RQTBA"; plotted as log10 T
Measured points (Pt204, Hg206): Beta_decay/config.py HL_EXP / HL_ERR [s] (NUBASE2020).
"""
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps   # LaTeX fonts, colours, ps.save()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import repo_paths as RP  # noqa: E402  (all data paths of the repository)
sys.path.insert(0, RP.CALIBRATION)
import config

PRED = RP.PAPER_DATA   # predictions_<F>_batch30001.tsv
PUB = RP.LITERATURE
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs", "fig_n126_halflives_compare")

# ---- knobs ------------------------------------------------------------------
FUNCTIONALS = ["SKOP", "SLY4", "HFB1"]
COLORS, LABELS, MARKERS = ps.OKABE_ITO, ps.LABELS, ps.MARKERS   # same markers as the other paper figures
MSIZE = {"SKOP": 5.0, "SLY4": 4.6, "HFB1": 5.6}   # pt; sized to look alike
LSTYLE = {"SKOP": "-", "SLY4": "--", "HFB1": "-."}
OPEN = {"SKOP": False, "SLY4": False, "HFB1": False}   # 2026-10-15 marker2: all filled, white edge (SLY4 was open)
LINE_LW = 1.0                             # pt, our lines (was 1.4)
MEC, MEW = "white", 0.8                   # marker edge of our markers
MZORDER = {"SLY4": 4.8, "SKOP": 5.0, "HFB1": 5.2}   # largest marker (SLy4 triangle) at the bottom so its corners show
DODGE = {"SKOP": 0.0, "SLY4": 0.0, "HFB1": 0.0}   # horizontal offset in Z of line+markers (bands stay at Z); off
ZOF = {"Ba182": 56, "Ce184": 58, "Nd186": 60, "Sm188": 62, "Gd190": 64, "Dy192": 66, "Er194": 68,
       "Yb196": 70, "Hf198": 72, "W200": 74, "Os202": 76, "Pt204": 78, "Hg206": 80}
ZGRID = np.array(sorted(ZOF.values()))
MEASURED = ["Pt204", "Hg206"]
ZMIN, ZMAX, NN = 54, 82, 126
# published curves: label -> (loader key, grey level, linestyle, marker)
PUBSTYLE = [
    ("DD-PC1",                "DDPC1",   "0.20", "-",              "s"),
    ("DD-PCX",                "DDPCX",   "0.10", (0, (4, 1.5)),    "x"),
    ("SkO$'$ (Ney et al.)",   "Evan",    "0.45", (0, (3, 1.5)),    "o"),
    ("RHB-1(D3C*)",           "Marketin", "0.30", (0, (6, 1.5, 1, 1.5)), "+"),
    ("RHB-1(D3C*) updated",   "RQRPA",   "0.30", (0, (1, 1)),      "x"),   # 2026-10-15: x instead of hexagon (user: easier to see)
    ("RQTBA",                 "RQTBA",   "0.30", (0, (3, 1.5)),    "v"),
    ("FRDM",                  "FRDM",    "0.45", (0, (1, 1)),      "d"),    # 2026-10-20: after RHB-1, thin-diamond marker (user)
]
# 2026-10-15 trim: only these published curves are drawn.  Set SHOW_ALL_PUB = True to restore
# all seven (DD-PCX, FRDM, RHB-1(D3C*) (Marketin), RQTBA); the old 2-column legend (6 pt) was used then.
SHOW_ALL_PUB = False
PUB_SHOWN = ("DDPC1", "Evan", "RQRPA", "FRDM")   # 2026-10-20 (user): FRDM in place of RQTBA
PUB_LW = 0.7
PUB_MS = 3.5                              # pt; 2026-10-15 pubms (was 4.5, before that 2.2); open (white-filled) grey markers
FIGWIDTH = ps.COLUMN_WIDTH
FIGHEIGHT = 4.2
BAND_ALPHA = 0.22   # 2026-10-15: was 0.12 (user: less transparent)
YLIM = (-4.0, 6.5)                        # log10 s; shows the full 68% bands at Z = 80
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
    """-> (Z array, log10 T [log10 s] array) for N = 126, ZMIN <= Z <= ZMAX, finite values only."""
    if key in ("DDPC1", "DDPCX"):
        a = np.genfromtxt(os.path.join(PUB, "data_%s.txt" % key), comments="#")
        n, z, y = a[:, 0], a[:, 1], a[:, 5]               # N Z E(beta) beta2 Q HL_log10 FF
    elif key in ("Evan", "FRDM", "Marketin"):
        a = np.genfromtxt(os.path.join(PUB, "data_%s.txt" % key), comments="#")
        n, z, y = a[:, 0], a[:, 1], a[:, 2]               # N Z log10T
        if key == "Marketin":
            y = np.where(y >= 100.0, np.nan, y)           # 1e2 = stable / no-decay placeholder
    elif key in ("RQRPA", "RQTBA"):
        a = np.genfromtxt(os.path.join(PUB, "half-lives.dat"), comments="#")
        z, n = a[:, 0], a[:, 1]                           # Z N T_RQRPA FF.. FF0 FF1 FF2 T_RQTBA ...
        t = a[:, 2] if key == "RQRPA" else a[:, 7]        # s
        y = np.log10(t)
    else:
        raise KeyError(key)
    m = (np.abs(n - NN) < 0.5) & (z > ZMIN - 0.5) & (z < ZMAX + 0.5) & np.isfinite(y)
    o = np.argsort(z[m])
    return z[m][o], y[m][o]


fig, ax = plt.subplots(figsize=(FIGWIDTH, FIGHEIGHT))

handles = []
# our functionals: bands (bottom), lines + markers (top)
for func in FUNCTIONALS:
    d = load_ours(func)
    c = COLORS[func]
    ok = ~np.isnan(d["log10T_at_median"])
    ax.fill_between(d["Z"], d["p16"], d["p84"], where=ok, color=c, alpha=BAND_ALPHA, lw=0, zorder=2.5)   # 2026-10-15: bands in front of the published curves (zorder 2), behind our lines (3)
    zx = d["Z"] + DODGE[func]
    ax.plot(zx, d["log10T_at_median"], LSTYLE[func], color=c, lw=LINE_LW, zorder=3)              # line
    ax.plot(zx, d["log10T_at_median"], ls="none", marker=MARKERS[func], ms=MSIZE[func],          # markers above all lines
            mfc="white" if OPEN[func] else c, mec=MEC, mew=MEW, zorder=MZORDER[func])
    handles.append(Line2D([], [], ls=LSTYLE[func], color=c, lw=LINE_LW, marker=MARKERS[func], ms=MSIZE[func],
                          mfc="white" if OPEN[func] else c, mec=MEC, mew=MEW, label=LABELS[func]))

# published curves
sanity = {}
for label, key, grey, ls, mk in PUBSTYLE:
    if not SHOW_ALL_PUB and key not in PUB_SHOWN:
        continue
    z, y = load_pub(key)
    sanity[label] = dict(zip(z.astype(int), y))
    # 2026-10-15: the updated RHB-1 uses an "x" (no fill); draw it a little larger and heavier so it stands out
    ms_, mew_ = (PUB_MS + 1.5, 1.0) if mk == "x" else (PUB_MS, 0.6)
    h, = ax.plot(z, y, linestyle=ls, color=grey, lw=PUB_LW, marker=mk, ms=ms_, mfc="white",
                 mec=grey, mew=mew_, label=label, zorder=2)
    handles.append(h)

# measured half-lives (NUBASE2020 via config)
zm = np.array([ZOF[n] for n in MEASURED])
T = np.array([config.HL_EXP[n] for n in MEASURED])     # s
dT = np.array([config.HL_ERR[n] for n in MEASURED])    # s
h = ax.errorbar(zm, np.log10(T), yerr=dT / (T * math.log(10.0)), fmt="o", color="k", mfc="k", mec="white",
                mew=0.8, ms=6.0, capsize=2, lw=0.8, zorder=6, label="Experiment")
handles.append(h)

ax.set_xlabel(r"$Z$")
ax.set_ylabel(r"$T_{1/2}$ (s)")
# 2026-10-23 (user): the axis shows T_1/2 in s on a log scale. Curves are drawn in log10(T/s); major ticks at
# even powers of ten labelled 10^k, minor ticks at every m*10^k (m = 1..9), i.e. exactly a logarithmic axis.
from matplotlib.ticker import FixedLocator, FuncFormatter
_k = np.arange(math.floor(YLIM[0]), math.ceil(YLIM[1]) + 1)
ax.yaxis.set_major_locator(FixedLocator(_k[_k % 2 == 0]))
ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: r"$10^{%d}$" % round(y)))
ax.yaxis.set_minor_locator(FixedLocator([k + math.log10(m) for k in _k for m in range(1, 10)
                                         if YLIM[0] <= k + math.log10(m) <= YLIM[1]]))
ax.set_xlim(ZMIN, ZMAX)
ax.set_xticks(range(ZMIN, ZMAX + 1, 4))
ax.set_ylim(*YLIM)
ax.text(0.95, 0.04, r"$N = 126$", transform=ax.transAxes, ha="right", va="bottom")
if SHOW_ALL_PUB:
    ax.legend(handles=handles, loc="upper left", ncol=2, fontsize=6.0, frameon=True, fancybox=False,
              edgecolor="0.3", handlelength=2.6, handletextpad=0.4, columnspacing=0.8, borderpad=0.4,
              labelspacing=0.35)
else:
    ax.legend(handles=handles, loc="upper left", ncol=1, fontsize=7.5, frameon=False,
              handlelength=2.6, handletextpad=0.4, borderpad=0.4, labelspacing=0.3)

ps.save(fig, OUT)

print("published log10 T [log10 s] at N = 126 (number of Z points plotted, Z = 78, Z = 80):")
for label, v in sanity.items():
    print("  %-22s n=%2d  Z78 %s  Z80 %s" % (label, len(v), "%+.3f" % v[78] if 78 in v else "  -  ",
                                               "%+.3f" % v[80] if 80 in v else "  -  "))
