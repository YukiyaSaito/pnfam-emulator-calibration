#!/usr/bin/env python3
"""First-forbidden share of the N = 126 decay rates, lambda_FF / (lambda_GT + lambda_FF), versus Z, with the
parameter uncertainty of the final calibrations (2026-10-28).

Method (same as the N = 126 half-life predictions): per nucleus and functional, quadratic surfaces in
(V0, C1s) are fitted to log10 lambda_GT and log10 lambda_FF of the 20 batch-30001 pnFAM runs
(data/n126_ff_rates.txt, extracted by extract_n126_ff.py). Both surfaces are evaluated at every sample of
the Full-set Student-t posterior and the fraction is formed per sample; the band is the 16-84% range of
those fractions, the marker the fraction at the posterior-median couplings. No theory-error term: the
inferred sigma_th concerns the total half-life, not the GT/FF split, so the band is parameter uncertainty
only. Published values for comparison: Ney et al. (SkO', FF% column of their table) and DD-PC1
(FF_percent column of data_DDPC1.txt), even Z only.

Output: ../figs/fig_n126_ff_fraction.{pdf,png}, table ../data/n126_ff_fraction.txt
"""
import os
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import repo_paths as RP  # noqa: E402  (all data paths of the repository)
sys.path.insert(0, RP.CALIBRATION)
from predict_from_draws import design  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
RATES = os.path.join(HERE, "..", "data", "n126_ff_rates.txt")
OUT = os.path.join(HERE, "..", "figs", "fig_n126_ff_fraction")
OUT_TXT = os.path.join(HERE, "..", "data", "n126_ff_fraction.txt")
PUB = RP.LITERATURE
TRACE = {F: os.path.join(RP.posterior(F, "full", "studentt"), "trace.nc") for F in RP.FUNCTIONALS}
NUCS = ["Ba182", "Ce184", "Nd186", "Sm188", "Gd190", "Dy192", "Er194", "Yb196", "Hf198", "W200", "Os202", "Pt204", "Hg206"]
ZOF = {n: 56 + 2 * i for i, n in enumerate(NUCS)}
NN = 126
RATE_FLOOR = 1e-9          # 1/s, resolution of the contour integration (same rule as the pool)
SHOW_PUB = True
LSTYLE = {"SKOP": "-", "SLY4": "--", "HFB1": "-."}


def load_rates():
    d = {}
    for ln in open(RATES):
        if ln.startswith("#") or ln.startswith("functional") or not ln.strip():
            continue
        F, nuc, Z, sid, v, c, tot, gt, ff, pc, hc, fc = ln.rstrip("\n").split("\t")
        if hc != "Yes" or fc != "Yes":
            continue
        d.setdefault((F, nuc), []).append((float(v), float(c), float(gt), float(ff)))
    return {k: np.array(x) for k, x in d.items()}


def load_pub():
    out = {}
    # Ney et al.: whitespace table, columns ID Z N Str Z-Blk N-Blk 2KP Energy beta_2 Q Half-Life FF% sigma_r
    ney = {}
    for ln in open(os.path.join(PUB, "supplementary_halflives.txt")):
        q = ln.split()
        if len(q) >= 13 and q[0].isdigit():
            try:
                z, n, ff = int(q[2]), int(q[3]), float(q[-2])
            except ValueError:
                continue
            if n == NN:
                ney[z] = ff / 100.0
    out["Ney \\textit{et al.} (SkO$'$)"] = ney
    # DD-PC1 (Ravlic, Saito, Nazarewicz): columns N Z E(beta) beta2 Q HL_log10 FF_percent
    dd = {}
    for ln in open(os.path.join(PUB, "data_DDPC1.txt")):
        q = ln.split()
        if len(q) >= 7 and not ln.startswith("#") and int(float(q[0])) == NN:
            dd[int(float(q[1]))] = float(q[6]) / 100.0
    out["DD-PC1"] = dd
    return out


def main():
    rates = load_rates()
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH, 2.7))
    txt = ["# FF share lambda_FF/(lambda_GT+lambda_FF) of the N = 126 isotones (draft/scripts/fig_n126_ff_fraction.py)",
           "# surfaces: quadratic in (V0,C1s) to log10 rate_GT and log10 rate_FF, 20 batch-30001 runs; posterior = Full set Student-t",
           "# p16/p50/p84 over the posterior (parameters only); at_median = fraction at the posterior-median couplings; fit RMS in log10 rate",
           "functional\tnucleus\tZ\tn_runs\tfit_rms_GT\tfit_rms_FF\tp16\tp50\tp84\tat_median\tpcFF_run_min\tpcFF_run_max"]
    z = np.array([ZOF[n] for n in NUCS])
    for F in TRACE:
        post = az.from_netcdf(TRACE[F]).posterior
        vp = post["vpair"].values.ravel(); cs = post["cs0"].values.ravel()
        vmed, cmed = float(np.median(vp)), float(np.median(cs))
        Xpost = design(vp, cs, "quadratic"); Xmed = design(np.array([vmed]), np.array([cmed]), "quadratic")
        lo, mid, hi, atmed = [], [], [], []
        for nuc in NUCS:
            a = rates[(F, nuc)]
            v, c, gt, ff = a[:, 0], a[:, 1], np.maximum(a[:, 2], RATE_FLOOR), np.maximum(a[:, 3], RATE_FLOOR)
            X = design(v, c, "quadratic")
            bg, *_ = np.linalg.lstsq(X, np.log10(gt), rcond=None)
            bf, *_ = np.linalg.lstsq(X, np.log10(ff), rcond=None)
            rg = float(np.sqrt(np.mean((X @ bg - np.log10(gt)) ** 2)))
            rf = float(np.sqrt(np.mean((X @ bf - np.log10(ff)) ** 2)))
            lg, lf = Xpost @ bg, Xpost @ bf
            frac = 1.0 / (1.0 + 10.0 ** (lg - lf))          # FF/(GT+FF) per posterior sample
            q = np.percentile(frac, [16, 50, 84])
            fm = float(1.0 / (1.0 + 10.0 ** ((Xmed @ bg - Xmed @ bf)[0])))
            lo.append(q[0]); mid.append(q[1]); hi.append(q[2]); atmed.append(fm)
            pc = a[:, 3] / (a[:, 2] + a[:, 3])
            txt.append("\t".join([F, nuc, str(ZOF[nuc]), str(len(a)), f"{rg:.4f}", f"{rf:.4f}",
                                  f"{q[0]:.4f}", f"{q[1]:.4f}", f"{q[2]:.4f}", f"{fm:.4f}", f"{pc.min():.4f}", f"{pc.max():.4f}"]))
        col = ps.OKABE_ITO[F]
        ax.fill_between(z, lo, hi, color=col, alpha=0.22, lw=0, zorder=2)
        ax.plot(z, atmed, LSTYLE[F], color=col, lw=1.0, marker=ps.MARKERS[F], ms=4.5, mec="white", mew=0.6,
                label=ps.LABELS[F], zorder=3)
    if SHOW_PUB:
        for (label, tab), (mk, ls) in zip(load_pub().items(), (("o", (0, (3, 1.5))), ("s", "-"))):
            zz = sorted(k for k in tab if 54 <= k <= 82 and k % 2 == 0)
            ax.plot(zz, [tab[k] for k in zz], ls=ls, color="0.35", lw=0.7, marker=mk, ms=3.5, mfc="white", mec="0.35",
                    mew=0.8, label=label, zorder=1)
    ax.set_xlabel("$Z$"); ax.set_ylabel(r"$\lambda_{\rm FF}/(\lambda_{\rm GT}+\lambda_{\rm FF})$")
    ax.set_xlim(54, 82); ax.set_xticks(range(54, 83, 4)); ax.set_ylim(0, 1.05)
    ax.text(0.04, 0.04, r"$N=126$", transform=ax.transAxes, ha="left", va="bottom")
    # two legends inside the frame (user 2026-10-28): functionals in the free lower-right corner,
    # published curves in the free upper-left corner (DD-PC1 stays below 0.8 for Z <= 64)
    h, l = ax.get_legend_handles_labels()
    leg1 = ax.legend(h[:3], l[:3], loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=7, frameon=False,
                     handlelength=1.8, handletextpad=0.4, labelspacing=0.3)
    ax.add_artist(leg1)
    ax.legend(h[3:], l[3:], loc="upper left", bbox_to_anchor=(0.0, 0.99), fontsize=7, frameon=False,
              handlelength=1.8, handletextpad=0.4, labelspacing=0.3)
    fig.tight_layout()
    ps.save(fig, OUT)
    open(OUT_TXT, "w").write("\n".join(txt) + "\n")
    for l in txt[4:]:
        q = l.split("\t")
        if q[1] in ("Ba182", "Er194", "W200", "Pt204", "Hg206"):
            print(f"{q[0]:5} {q[1]:6} fitRMS GT {q[4]} FF {q[5]} | FF share p16 {q[6]} p50 {q[7]} p84 {q[8]} | runs {q[10]}..{q[11]}")
    print("wrote", OUT_TXT)


if __name__ == "__main__":
    main()
