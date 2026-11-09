#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase-0 GT-resonance nuclei-selection table for the SKO'/SKOP 2-parameter
calibration, and the COMBINED HL+GT Fisher conditioning in (vpair,cs0) space.

For each SKOP/GT nucleus, compute the windowed Gamow-Teller centroid per sample
and regress it on the two scaled parameters:
  centroid_Ex = sum(E_x * S) / sum(S)   over E_x in the nucleus GT window
  E_x = E_math - shift  (config.get_gt_shift);  window from config.get_gt_window
Sensitivity b_θ = d centroid[MeV] / d(scaled param); Fisher F_θ = b_θ^2/sigma^2
with sigma = experimental GT centroid error (config.GT_E_ERR, MeV).

The 2x2 Fisher matrix lives in parameter space, so HL (log10 T) and GT (MeV)
contributions are ADDITIVE information about (vpair,cs0); we report HL-only,
GT-only, and combined condition numbers to quantify HL<->GT complementarity.

Physics check: mean computed centroid should sit within a few MeV of the
experimental GTR energy (config.GT_E_EXP). Read-only; double precision.

2026-08-05: generalized to --functional (default SKOP, unchanged behavior;
output names <functional>_gt_fisher.csv/.txt) and --out-dir. The energy
shift is functional-keyed (get_gt_shift(nuc, FUNCTIONAL)); the HL CSV for
the combined conditioning is read from the same out-dir and functional.
"""
import argparse
import os
import math
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import repo_paths as RP  # noqa: E402
sys.path.insert(0, RP.CALIBRATION)
import config  # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--functional", default="SKOP", help="SKOP | UNE1 | ...")
_ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)),
                 help="where to write <functional>_gt_fisher.csv/.txt; must "
                      "already hold <functional>_hl_fisher.csv")
_args = _ap.parse_args()
FUNCTIONAL = _args.functional
GTROOT = os.path.join(RP.TRAINING, FUNCTIONAL, "GT")   # <nuc>/full/sample_<i>.dat strength functions
OUT = _args.out_dir
STEM = f"{FUNCTIONAL.lower()}_gt_fisher"
LN10 = math.log(10.0)


def read_params(path):
    idx, vp, cs = [], [], []
    for ln in open(path):
        s = ln.strip()
        if not s or s.startswith("#"):
            continue
        c = s.split()
        idx.append(int(c[0])); vp.append(float(c[1])); cs.append(float(c[2]))
    return idx, np.array(vp), np.array(cs)


def centroid(nuc, i):
    """Windowed GT centroid in physical E_x for one sample, or nan."""
    f = os.path.join(GTROOT, nuc, "full", f"sample_{i}.dat")
    if not os.path.exists(f):
        return float("nan")
    d = np.loadtxt(f, comments="#")
    E, S = d[:, 0], d[:, 1]
    shift = config.get_gt_shift(nuc, FUNCTIONAL)
    Ex = E - shift
    lo, hi = config.get_gt_window(nuc)
    m = (Ex >= lo) & (Ex <= hi) & (S > 0.0)
    if m.sum() < 2 or S[m].sum() <= 0.0:
        return float("nan")
    return float((Ex[m] * S[m]).sum() / S[m].sum())


def fit_gt(nuc):
    idx, vp, cs = read_params(os.path.join(GTROOT, nuc, "full", "params.txt"))
    a = (vp + 250.0) / 250.0
    b = (cs - 125.0) / 125.0
    c = np.array([centroid(nuc, i) for i in idx])
    good = np.isfinite(c)
    a, b, c = a[good], b[good], c[good]
    X = np.column_stack([np.ones_like(a), a, b])
    beta, *_ = np.linalg.lstsq(X, c, rcond=None)
    sst = ((c - c.mean()) ** 2).sum()
    R2 = 1.0 - ((c - X @ beta) ** 2).sum() / sst if sst > 0 else float("nan")

    def reduced(cols):
        Xr = np.column_stack([np.ones_like(a)] + cols)
        br, *_ = np.linalg.lstsq(Xr, c, rcond=None)
        return 1.0 - ((c - Xr @ br) ** 2).sum() / sst

    resid = c - X @ beta
    dof = max(len(c) - X.shape[1], 1)
    sresid2 = float((resid ** 2).sum() / dof)
    XtXinv = np.linalg.inv(X.T @ X)
    se_vp = float(np.sqrt(sresid2 * XtXinv[1, 1]))
    se_cs = float(np.sqrt(sresid2 * XtXinv[2, 2]))
    return dict(n=len(c), R2=R2, b_vp=beta[1], b_cs=beta[2],
                pR2_vp=R2 - reduced([b]), pR2_cs=R2 - reduced([a]),
                cbar=float(c.mean()), se_vp=se_vp, se_cs=se_cs,
                sresid=float(np.sqrt(sresid2)))


def hl_contribs(setn):
    """Load HL per-nucleus (b_vp,b_cs,sigma_log10) from the HL table CSV."""
    path = os.path.join(OUT, f"{FUNCTIONAL.lower()}_hl_fisher.csv")
    out = []
    for ln in open(path):
        if ln.startswith("nucleus") or not ln.strip():
            continue
        c = ln.split(",")
        nuc = c[0]
        if nuc in setn:
            out.append((np.array([float(c[3]), float(c[4])]), float(c[8])))  # g, slog
    return out


def fisher_from(contribs):
    F = np.zeros((2, 2))
    for g, s in contribs:
        if s > 0:
            F += np.outer(g, g) / s ** 2
    ev = np.linalg.eigvalsh(F)
    cond = ev[-1] / ev[0] if ev[0] > 0 else float("inf")
    return F, ev, cond


def main():
    rows = []
    lines = []
    def p(s):
        lines.append(s); print(s)

    p(f"=== {FUNCTIONAL} GT-centroid nuclei-selection table ===")
    p("centroid sensitivity b = d(centroid[MeV]) per half-box; "
      "F = b^2/sigma_exp^2 (sigma from GT_E_ERR)")
    p(f"{'nuc':7s}{'n':>4s}{'R2':>6s}{'b_vp':>7s}{'b_cs':>7s}{'pR2vp':>7s}{'pR2cs':>7s}"
      f"{'cbar':>8s}{'cexp':>7s}{'sig':>6s}{'F_vp':>8s}{'F_cs':>8s}")
    gt_contribs = []
    for nuc in sorted(os.listdir(GTROOT)):
        if not os.path.isdir(os.path.join(GTROOT, nuc)):
            continue
        r = fit_gt(nuc)
        key = config.normalize_nucname(nuc)
        cexp = config.GT_E_EXP.get(key, float("nan"))
        sig = config.GT_E_ERR.get(key, float("nan"))
        F_vp = (r["b_vp"] ** 2) / sig ** 2 if sig == sig else float("nan")
        F_cs = (r["b_cs"] ** 2) / sig ** 2 if sig == sig else float("nan")
        r.update(nuc=nuc, cexp=cexp, sig=sig, F_vp=F_vp, F_cs=F_cs)
        rows.append(r)
        gt_contribs.append((np.array([r["b_vp"], r["b_cs"]]), sig))
        p(f"{nuc:7s}{r['n']:4d}{r['R2']:6.2f}{r['b_vp']:7.2f}{r['b_cs']:7.2f}"
          f"{r['pR2_vp']:7.2f}{r['pR2_cs']:7.2f}{r['cbar']:8.2f}{cexp:7.2f}{sig:6.2f}"
          f"{r['F_vp']:8.0f}{r['F_cs']:8.0f}")

    # physics sanity check
    dmax = max(abs(r["cbar"] - r["cexp"]) for r in rows if r["cexp"] == r["cexp"])
    p(f"\n[physics check] max |computed centroid - experimental GTR| across grid "
      f"means = {dmax:.1f} MeV (calibration should close the model-data gap; "
      f"a few MeV is expected, tens of MeV would signal a window/shift bug)")

    # --- combined HL+GT conditioning ---
    p("\n--- Fisher conditioning in (vpair,cs0): HL-only vs GT-only vs combined ---")
    hl = hl_contribs(config.HL_SET_A)
    _, evh, ch = fisher_from(hl)
    _, evg, cg = fisher_from(gt_contribs)
    _, evc, cc = fisher_from(hl + gt_contribs)
    p(f"HL SET_A          eig={evh.round(0)}  cond={ch:.1f}")
    p(f"GT SET_A(7 nuc)   eig={evg.round(1)}  cond={cg:.1f}")
    p(f"HL + GT combined  eig={evc.round(0)}  cond={cc:.1f}")
    hlp = hl_contribs(config.HL_SET_A | {"Ti52", "Pt200"})
    _, evcp, ccp = fisher_from(hlp + gt_contribs)
    p(f"HL(A+Ti52,Pt200)+GT combined  eig={evcp.round(0)}  cond={ccp:.1f}")

    with open(os.path.join(OUT, STEM + ".csv"), "w") as f:
        f.write("nucleus,n,R2,b_vp,b_cs,pR2_vp,pR2_cs,centroid_mean,centroid_exp,"
                "sigma_MeV,F_vp,F_cs,se_vp,se_cs,sigma_resid\n")
        for r in rows:
            f.write(f"{r['nuc']},{r['n']},{r['R2']:.4f},{r['b_vp']:.4f},{r['b_cs']:.4f},"
                    f"{r['pR2_vp']:.4f},{r['pR2_cs']:.4f},{r['cbar']:.4f},{r['cexp']:.4f},"
                    f"{r['sig']:.4f},{r['F_vp']:.1f},{r['F_cs']:.1f},"
                    f"{r['se_vp']:.5f},{r['se_cs']:.5f},{r['sresid']:.5f}\n")
    with open(os.path.join(OUT, STEM + ".txt"), "w") as f:
        f.write("\n".join(lines) + "\n")
    p(f"\nsaved: {OUT}/{STEM}.csv and .txt")


if __name__ == "__main__":
    main()
