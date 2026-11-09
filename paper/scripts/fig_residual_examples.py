#!/usr/bin/env python3
"""Emulator residual examples for the paper (SkO', two nuclei).

Each point: emulated minus pnFAM log10 half-life for one parameter sample
the emulator was not trained on, against the pnFAM value.

SOURCE = "test" (2026-10-19): residuals on the 40 frozen test points per
nucleus (batch 21001, never trained on), from
analysis/frozen/test_residuals_batch21001.tsv. SOURCE = "validation" keeps
the five-fold cross-validation held-out residuals (the 2026-10-13 placeholder).

Panels: (a) a typical calibration nucleus, (b) the calibration nucleus with
the largest test error (TEST_ERROR_2026-10-17.txt: Cr56 0.013, Sn134 0.0005).
The dotted vertical line is the measured half-life (config.HL_EXP).
"""
import glob
import math
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import repo_paths as RP  # noqa: E402  (all data paths of the repository)
sys.path.insert(0, RP.CALIBRATION)
import config

CV = RP.CV
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs",
                   "fig_residual_examples")

# ---- knobs ------------------------------------------------------------------
FUNCTIONAL = "SKOP"
NUCLEI = ["Sn134", "Cr56"]                # (a) typical, (b) largest error among set members
SOURCE = "test"                           # "test" (frozen batch 21001) | "validation" (CV folds)
TEST_TSV = os.path.join(RP.ERRORS, "test_residuals_batch21001.tsv")
# ------------------------------------------------------------------------------


def residuals(func, nuc):
    """(log10 T_pnFAM, log10 T_emu - log10 T_pnFAM): test points, or pooled CV folds."""
    if SOURCE == "test":
        x, r = [], []
        for ln in open(TEST_TSV):
            p = ln.split("\t")
            if p[0] == func and p[1] == nuc:
                x.append(float(p[6])); r.append(float(p[8]))
        return np.array(x), np.array(r)
    x, r = [], []
    for f in sorted(glob.glob(os.path.join(CV, func, "HL", nuc, "fold_*",
                                           "parity_half_life_val.txt"))):
        d = np.loadtxt(f, ndmin=2)
        ok = (d[:, 1] > 0.0) & (d[:, 2] > 0.0)
        x.append(np.log10(d[ok, 1]))
        r.append(np.log10(d[ok, 2]) - np.log10(d[ok, 1]))
    return np.concatenate(x), np.concatenate(r)


fig, axes = plt.subplots(2, 1, figsize=(ps.COLUMN_WIDTH, 3.4), sharey=False,
                         gridspec_kw=dict(hspace=0.32))
for ax, nuc, lab in zip(axes, NUCLEI, "ab"):
    x, r = residuals(FUNCTIONAL, nuc)
    rms = math.sqrt(np.mean(r * r))
    ax.axhline(0.0, color="0.6", lw=0.6)
    ax.axvline(math.log10(config.HL_EXP[nuc]), color="0.3", lw=0.7, ls=":")
    ax.plot(x, r, ps.MARKERS[FUNCTIONAL], ms=2.5, color=ps.OKABE_ITO[FUNCTIONAL],
            mfc="none", mew=0.6)
    ax.text(0.03, 0.92, f"({lab}) {ps.nuc(nuc)}", transform=ax.transAxes, va="top")
    ax.text(0.97, 0.08, rf"rms $= {rms:.4f}$" if rms < 0.002 else rf"rms $= {rms:.3f}$", transform=ax.transAxes, va="bottom", ha="right")
    ax.set_xlabel(r"$\log_{10}(T_{1/2}/\mathrm{s})$ (pnFAM)")
    lim = 1.15 * np.max(np.abs(r))
    ax.set_ylim(-lim, lim)
fig.subplots_adjust(left=0.2)
fig.supylabel(r"$\log_{10} T_{1/2}$: emulator $-$ pnFAM", x=0.02)
ps.save(fig, OUT)
print("source:", SOURCE, "| nuclei:", NUCLEI, "| functional:", FUNCTIONAL)
