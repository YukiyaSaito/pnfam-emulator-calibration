#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-0 nuclei-selection table for the SKO'/SKOP 2-parameter half-life
calibration. For each SKOP/HL nucleus, quantify how strongly log10(T1/2)
responds to the two parameters and how tightly that constrains the posterior.

Model per nucleus:  log10(T)  =  b0 + b_vp * a + b_cs * b
  a = scaled vpair_t0 in [-1,1] over (-500,0)
  b = scaled override_cs0 in [-1,1] over (0,250)
Sensitivity slopes b_vp,b_cs are d log10(T) per half-box.

Constraint (Fisher) per parameter, single-nucleus Gaussian likelihood on
y = log10(T):   F_θ = (b_θ)^2 / sigma_log10^2
  sigma_log10 = sigma_exp / (ln10 * T_exp)      (experimental floor)
sigma_em (from CV parity, where available) is reported for context; it is
NOT folded into the headline Fisher because (i) only the 10 SET_A nuclei have
CV runs and (ii) its log-vs-linear convention must first be confirmed against
bayes_mcmc. Primary metric therefore uses sigma_exp (consistent across all 20).

Set conditioning: summed Fisher matrix F = Σ_i g_i g_i^T / sigma_log10,i^2 with
g_i = [b_vp, b_cs]; report eigenvalues and condition number (large cond => the
weak eigen-direction is poorly constrained => degenerate posterior).

Read-only. Double precision; no float '==' comparisons.

2026-08-05: generalized to --functional (default SKOP, unchanged behavior;
output names <functional>_hl_fisher.csv/.txt) and --out-dir. The scaled
box (-500,0)x(0,250) is kept IDENTICAL across functionals so slopes stay
comparable, even though UNE1 samples only span vpair [-398.4, -3.9].
"""
import argparse
import os
import math
import numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument("--functional", default="SKOP", help="SKOP | UNE1 | ...")
_ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)),
                 help="where to write <functional>_hl_fisher.csv/.txt")
_ap.add_argument("--include-batch-ids", action="store_true",
                 help="include densification-batch samples (ids >= 10001) in "
                      "the slope fits. DEFAULT OFF (protocol 2026-08-05): "
                      "global slopes are fit on the original space-filling "
                      "design only -- batch windows oversample the rough "
                      "corner and would design-weight the OLS slopes; "
                      "batches inform emulators and localized slopes.")
_args = _ap.parse_args()
FUNCTIONAL = _args.functional
BATCH_ID_MIN = None if _args.include_batch_ids else 10001
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import repo_paths as RP  # noqa: E402
ROOT = os.path.join(RP.TRAINING, FUNCTIONAL, "HL")   # params.txt / halflives.txt per nucleus
CVROOT = os.path.join(RP.CV, FUNCTIONAL, "HL")       # cross-validation parity files
OUT = _args.out_dir
STEM = f"{FUNCTIONAL.lower()}_hl_fisher"

import sys
sys.path.insert(0, RP.CALIBRATION)
import config

LN10 = math.log(10.0)

# curated candidate sets (from config.py), intersected with SKOP availability
SET_A = config.HL_SET_A
SET_APLUS = config.HL_SET_APLUS   # A ∪ {Ti52,Pt200,Zr100}; Zr100 absent in SKOP


def read_two(path, ncol):
    idx, vals = [], []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        c = ln.split()
        idx.append(int(c[0]))
        vals.append([float(x) for x in c[1:1 + ncol]])
    return idx, np.array(vals, dtype=np.float64)


def load_sigma_em(nuc, T_exp):
    """Emulator sigma_em in log10(T) NEAR the experimental half-life:
    RMS (including bias, 2026-07-08 convention) of the pooled CV fold
    validation residuals log10(pred/true), restricted to true T within one
    decade of T_exp (falls back to all points when fewer than 8 land in
    the decade window). Returns (sem_log_near_Texp, sem_log_all).

    2026-08-05: replaces the earlier linear-seconds heteroscedastic-bin
    estimate -- linear bins straddle decades, so the bin sigma divided by
    ln10*T_exp can overstate the log error at T_exp by >10x (Pd114: bin
    1.74 dex vs log-space 0.045 dex).
    """
    import glob
    files = sorted(glob.glob(os.path.join(
        CVROOT, nuc, "fold_*", "parity_half_life_val.txt")))
    if not files or T_exp is None:
        return float("nan"), float("nan")
    r_all, r_near = [], []
    for f in files:
        d = np.loadtxt(f, ndmin=2)
        if d.size == 0:
            continue
        ok = (d[:, 1] > 0.0) & (d[:, 2] > 0.0)
        lr = np.log10(d[ok, 2]) - np.log10(d[ok, 1])
        r_all.append(lr)
        near = (d[ok, 1] > T_exp / 10.0) & (d[ok, 1] < T_exp * 10.0)
        r_near.append(lr[near])
    if not r_all:
        return float("nan"), float("nan")
    r_all = np.concatenate(r_all)
    r_near = np.concatenate(r_near)
    sem_all = float(np.sqrt(np.mean(r_all ** 2))) if len(r_all) else float("nan")
    if len(r_near) >= 8:
        return float(np.sqrt(np.mean(r_near ** 2))), sem_all
    return sem_all, sem_all


def fit(nuc):
    pidx, praw = read_two(os.path.join(ROOT, nuc, "params.txt"), 2)
    hidx, hval = read_two(os.path.join(ROOT, nuc, "halflives.txt"), 1)
    hmap = {int(i): float(v[0]) for i, v in zip(hidx, hval)}
    keep = lambda i: (int(i) in hmap
                      and (BATCH_ID_MIN is None or int(i) < BATCH_ID_MIN))
    idx = [i for i in pidx if keep(i)]
    sel = [k for k, i in enumerate(pidx) if keep(i)]
    vp = praw[sel, 0]
    cs = praw[sel, 1]
    a = (vp + 250.0) / 250.0          # scale (-500,0) -> [-1,1]
    b = (cs - 125.0) / 125.0          # scale (0,250)  -> [-1,1]
    y = np.log10(np.array([hmap[int(i)] for i in idx], dtype=np.float64))
    X = np.column_stack([np.ones_like(a), a, b])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    yhat = X @ beta
    sst = ((y - y.mean()) ** 2).sum()
    R2 = 1.0 - ((y - yhat) ** 2).sum() / sst

    def reduced(cols):
        Xr = np.column_stack([np.ones_like(a)] + cols)
        br, *_ = np.linalg.lstsq(Xr, y, rcond=None)
        return 1.0 - ((y - Xr @ br) ** 2).sum() / sst

    pR2_vp = R2 - reduced([b])
    pR2_cs = R2 - reduced([a])
    # slope standard errors (OLS): se(b_j) = sqrt(sigma_resid^2 * (X'X)^-1_jj)
    resid = y - yhat
    dof = max(len(y) - X.shape[1], 1)
    sresid2 = float((resid ** 2).sum() / dof)
    XtXinv = np.linalg.inv(X.T @ X)
    se_vp = float(np.sqrt(sresid2 * XtXinv[1, 1]))
    se_cs = float(np.sqrt(sresid2 * XtXinv[2, 2]))
    return dict(n=len(y), R2=R2, b_vp=beta[1], b_cs=beta[2],
                pR2_vp=pR2_vp, pR2_cs=pR2_cs,
                se_vp=se_vp, se_cs=se_cs, sresid=float(np.sqrt(sresid2)))


def main():
    rows = []
    for nuc in sorted(os.listdir(ROOT)):
        if not os.path.isdir(os.path.join(ROOT, nuc)):
            continue
        r = fit(nuc)
        T = config.HL_EXP.get(nuc)
        sE = config.HL_ERR.get(nuc)
        slog = sE / (LN10 * T) if (T and sE) else float("nan")
        sem_log, sem_log_all = load_sigma_em(nuc, T)
        F_vp = (r["b_vp"] ** 2) / slog ** 2 if slog == slog else float("nan")
        F_cs = (r["b_cs"] ** 2) / slog ** 2 if slog == slog else float("nan")
        r.update(nuc=nuc, T=T, sE=sE, slog=slog, sem=sem_log_all, sem_log=sem_log,
                 F_vp=F_vp, F_cs=F_cs)
        # self-checks
        assert -0.01 <= r["R2"] <= 1.01 or r["R2"] < 0, "R2 out of expected range"
        rows.append(r)

    rows.sort(key=lambda x: -(x["F_vp"] if x["F_vp"] == x["F_vp"] else -1))

    lines = []
    def p(s):
        lines.append(s); print(s)

    p(f"=== {FUNCTIONAL} 2-parameter HL nuclei-selection table ===")
    p("sensitivity = d log10(T) per half-box; F = Fisher info (sigma_exp floor); "
      "sem_log = CV log10 RMS within a decade of T_exp (context); sem/sE ratio at T_exp")
    hdr = (f"{'nuc':7s}{'n':>4s}{'R2':>6s}{'b_vp':>7s}{'b_cs':>7s}"
           f"{'pR2vp':>7s}{'pR2cs':>7s}{'T(s)':>10s}{'slog':>7s}{'semlog':>8s}"
           f"{'sem/sE':>8s}{'F_vp':>9s}{'F_cs':>9s}")
    p(hdr)
    for r in rows:
        ratio = (r["sem_log"] / r["slog"]) if (r["sem_log"] == r["sem_log"]) else float("nan")
        p(f"{r['nuc']:7s}{r['n']:4d}{r['R2']:6.2f}{r['b_vp']:7.2f}{r['b_cs']:7.2f}"
          f"{r['pR2_vp']:7.2f}{r['pR2_cs']:7.2f}{r['T']:10.3g}{r['slog']:7.3f}"
          f"{r['sem_log']:8.3f}{ratio:8.2f}{r['F_vp']:9.0f}{r['F_cs']:9.0f}")

    def fisher(setn, use_sem=False):
        F = np.zeros((2, 2))
        used = []
        for r in rows:
            if r["nuc"] not in setn:
                continue
            s2 = r["slog"] ** 2
            if use_sem and r["sem_log"] == r["sem_log"]:
                s2 = r["slog"] ** 2 + r["sem_log"] ** 2
            if s2 > 0:
                g = np.array([r["b_vp"], r["b_cs"]])
                F += np.outer(g, g) / s2
                used.append(r["nuc"])
        ev = np.linalg.eigvalsh(F)
        assert (ev >= -1e-6).all(), "Fisher not PSD"
        cond = ev[-1] / ev[0] if ev[0] > 0 else float("inf")
        return ev, cond, sorted(used)

    p("")
    p("--- set conditioning (summed Fisher matrix) ---")
    for name, setn in [("SET_A", SET_A),
                       ("SET_A + Ti52,Pt200", SET_A | {"Ti52", "Pt200"}),
                       ("SET_APLUS(∩SKOP)", SET_APLUS)]:
        ev, cond, used = fisher(setn)
        p(f"{name:22s} eig={ev.round(0)}  cond={cond:.1f}  n={len(used)}")
    ev, cond, _ = fisher(SET_A, use_sem=True)
    p(f"{'SET_A (+sigma_em)':22s} eig={ev.round(0)}  cond={cond:.1f}  "
      f"(refinement using sigma_total for the 10 CV nuclei)")

    # save
    cols = ("nucleus,n,R2,b_vp,b_cs,pR2_vp,pR2_cs,T_s,sigma_log10,sem_log10,"
            "F_vp,F_cs,se_vp,se_cs,sigma_resid")
    with open(os.path.join(OUT, STEM + ".csv"), "w") as f:
        f.write(cols + "\n")
        for r in rows:
            f.write(f"{r['nuc']},{r['n']},{r['R2']:.4f},{r['b_vp']:.4f},{r['b_cs']:.4f},"
                    f"{r['pR2_vp']:.4f},{r['pR2_cs']:.4f},{r['T']:.6g},{r['slog']:.5f},"
                    f"{r['sem_log']:.5f},{r['F_vp']:.1f},{r['F_cs']:.1f},"
                    f"{r['se_vp']:.5f},{r['se_cs']:.5f},{r['sresid']:.5f}\n")
    with open(os.path.join(OUT, STEM + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    p(f"\nsaved: {OUT}/{STEM}.csv and .txt")


if __name__ == "__main__":
    main()
