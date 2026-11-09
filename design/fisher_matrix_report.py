#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Paper-ready Fisher (precision) matrix report for candidate nucleus sets, 2D
SKO' calibration in (vpair, cs0).

For each set, the summed DATA Fisher matrix (no prior) is
    F = sum_i g_i g_i^T / sigma_i^2
with g_i the sensitivity slopes (per half-box, scaled [-1,1] parametrization so
the two parameters are dimensionally comparable and the condition number is
meaningful) and sigma_i the observable error. We report:
  - the 2x2 matrix entries (F_vv, F_vc, F_cc),
  - eigenvalues + eigenvectors (principal constrained/unconstrained directions),
  - condition number (degeneracy indicator), under sigma_exp and sigma_total,
  - F^-1 = data-only posterior covariance (Laplace/Gaussian approx), giving the
    implied posterior std devs in scaled AND physical units, and the vpair-cs0
    correlation; compared against the actual 2D MCMC posterior spreads.

Read-only. Double precision. Note: condition number is parametrization-dependent
(computed in the normalized [-1,1] box); F^-1 here excludes the prior.
"""
import os
import numpy as np
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import repo_paths as RP  # noqa: E402
sys.path.insert(0, RP.CALIBRATION)
import config

OUT = os.path.dirname(os.path.abspath(__file__))
VPAIR_HALF, CS0_HALF = 250.0, 125.0   # physical half-box widths for de-scaling

# DOMINANT model/theory discrepancy fitted by bayes_mcmc (results/SKOP/bayes/*/
# summary.csv): sigma_hl_the ~0.52 in log10(T), sigma_gt_the ~1.66 MeV. This is
# the term that actually sets the posterior width; sigma_exp/sigma_em are
# subdominant. sigma_total^2 = sigma_exp^2 + sigma_em^2 + sigma_the^2.
SIGMA_THE_HL = 0.517
SIGMA_THE_GT = 1.66

# actual 2D MCMC posterior spreads (raw units) for validation, from
# results/SKOP/bayes/*/summary.csv (centroid mode): vpair sd~72, cs0 sd~19
OBS_POST = {"vpair_sd": 72.0, "cs0_sd": 19.0}


def load(csv, is_gt):
    d = {}
    for ln in open(os.path.join(OUT, csv)):
        if ln.startswith("nucleus") or not ln.strip():
            continue
        c = ln.split(",")
        g = np.array([float(c[3]), float(c[4])])
        if is_gt:
            sig_exp = float(c[9])
            sig_tot = np.sqrt(sig_exp ** 2 + SIGMA_THE_GT ** 2)
        else:
            slog = float(c[8]); sem = float(c[9]) if c[9] not in ("", "nan") else np.nan
            sem = sem if np.isfinite(sem) else 0.0
            sig_exp = slog
            sig_tot = np.sqrt(slog ** 2 + sem ** 2 + SIGMA_THE_HL ** 2)
        d[("GT:" if is_gt else "HL:") + c[0]] = (g, sig_exp, sig_tot)
    return d


def fisher(pool, labels, which):
    F = np.zeros((2, 2))
    for lab in labels:
        if lab not in pool:
            continue
        g, se, st = pool[lab]
        s = se if which == "exp" else st
        F += np.outer(g, g) / s ** 2
    return F


def report(name, pool, labels, out):
    labs = [l for l in labels if l in pool]
    Fexp = fisher(pool, labs, "exp")
    Ftot = fisher(pool, labs, "tot")
    out.append(f"\n{'='*72}\n{name}  (n={len(labs)})")
    out.append("nuclei: " + ", ".join(l.split(':')[1] for l in labs))
    for tag, F in [("sigma_exp only (omits theory err - WRONG scale)", Fexp),
                   ("sigma_total = exp(+)emu(+)theory (correct)", Ftot)]:
        ev, V = np.linalg.eigh(F)
        cond = ev[-1] / ev[0] if ev[0] > 0 else np.inf
        out.append(f"\n[{tag}] Fisher matrix F (scaled units):")
        out.append(f"    [ {F[0,0]:12.1f}  {F[0,1]:12.1f} ]   (vpair,vpair)  (vpair,cs0)")
        out.append(f"    [ {F[1,0]:12.1f}  {F[1,1]:12.1f} ]   (cs0,vpair)    (cs0,cs0)")
        out.append(f"  eigenvalues: lambda_min={ev[0]:.1f}  lambda_max={ev[-1]:.1f}"
                   f"   condition#={cond:.1f}")
        out.append(f"  weak-direction eigenvector (vpair,cs0)=({V[0,0]:+.2f},{V[1,0]:+.2f})"
                   f"   strong=({V[0,1]:+.2f},{V[1,1]:+.2f})")
        # data-only posterior covariance = F^-1
        try:
            C = np.linalg.inv(F)
            sdv_s, sdc_s = np.sqrt(C[0, 0]), np.sqrt(C[1, 1])
            corr = C[0, 1] / (sdv_s * sdc_s)
            out.append(f"  F^-1 -> posterior sd (scaled): vpair={sdv_s:.3f}  cs0={sdc_s:.3f}"
                       f"   corr={corr:+.2f}")
            out.append(f"           posterior sd (physical): vpair={sdv_s*VPAIR_HALF:.0f}"
                       f"   cs0={sdc_s*CS0_HALF:.0f}   [MCMC obs: vpair~{OBS_POST['vpair_sd']:.0f}"
                       f", cs0~{OBS_POST['cs0_sd']:.0f}]")
        except np.linalg.LinAlgError:
            out.append("  F singular (rank-deficient) -> a parameter direction is unconstrained")


def main():
    hl = load("skop_hl_fisher.csv", False)
    gt = load("skop_gt_fisher.csv", True)
    pool = {**hl, **gt}

    A = [f"HL:{n}" for n in config.HL_SET_A]
    Ap = A + ["HL:Ti52", "HL:Pt200"]
    recHL = [f"HL:{n}" for n in ["Pt200", "Sn134", "Rn226", "Cd126",
                                 "Ba148", "Zn82", "Sr102", "Ar48"]]
    recGT = [f"GT:{n}" for n in ["Pb208", "Sn112", "Te130", "Zr90", "Sn132"]]

    out = ["SKO' 2D Fisher-matrix report (vpair, cs0). Scaled [-1,1] parametrization.",
           "F = sum g g^T / sigma^2 (data only, no prior). F^-1 = Laplace posterior cov."]
    report("SET_A (canonical 10)", pool, A, out)
    report("SET_A + Ti52, Pt200", pool, Ap, out)
    report("Recommended HL (8)", pool, recHL, out)
    report("Recommended HL + GT (joint)", pool, recHL + recGT, out)

    text = "\n".join(out)
    print(text)
    with open(os.path.join(OUT, "fisher_matrix_report.txt"), "w") as f:
        f.write(text + "\n")
    print(f"\nsaved: {OUT}/fisher_matrix_report.txt")


if __name__ == "__main__":
    main()
