#!/usr/bin/env python3
"""Posterior-predictive residuals of the eight GT centroids, final posteriors
(port of calibration/analysis/paper_figs/fig_v7_gt_residuals.py, with the L10 runs added
so the figure mirrors fig_pool_residuals: filled marker = Full posterior, open = L10 posterior).

One panel, the 8 GT nuclei on the x axis, the three functionals side by side per nucleus.
Plotted: predicted centroid minus measured centroid [MeV], posterior-predictive median (marker),
68% (thick) and 95% (thin) predictive intervals. All 8 nuclei are fitted in every run (GT set = all),
so there is no out-of-sample case here, unlike the half-life figure.

Predictive draw per posterior sample s, exactly the GT likelihood of bayes_mcmc.py:
  E = tracked_pred_Ex_s + scale * t_nu,  scale = sqrt(sig_gt_the_s^2 + err_exp_i^2 + sig_em_gt_i^2) * sqrt((nu-2)/nu)
(Normal if the run was Normal). err_exp_i = config.GT_E_ERR, sig_em_gt_i (test errors) from the manifest.

Inputs : results/<F>/bayes_t4gtt4_sthe1_domw10_q_<set>_gtall[_<F>]_sem21001/centroid/{trace.nc,manifest.txt}
Outputs: ../figs/fig_gt_residuals.{pdf,png}; table ../data/gt_residuals.txt
"""
import ast
import os
import sys

import arviz as az
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps   # LaTeX fonts, colours, ps.save()
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import repo_paths as RP  # noqa: E402  (all data paths of the repository)
sys.path.insert(0, RP.CALIBRATION)
import config  # noqa: E402
from fig_pool_residuals import RUNS, SEED, read_manifest, PLAIN_LABELS  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figs", "fig_gt_residuals")
OUT_TXT = os.path.join(HERE, "..", "data", "gt_residuals.txt")
PCT = (2.5, 16.0, 50.0, 84.0, 97.5)
OFF = {"SKOP": -0.27, "SLY4": 0.0, "HFB1": 0.27}   # functional offset within a nucleus column
DX = 0.07                                         # Full (left) / L10 (right) offset within a functional
YLIM = (-5.0, 5.0)


def tex_gt(key):
    el, a = key.split("_")
    return rf"$^{{{a}}}${el}"


def load_run(F, sub):
    d = RP.posterior(F, sub, "studentt")
    m = read_manifest(os.path.join(d, "manifest.txt"))
    post = az.from_netcdf(os.path.join(d, "trace.nc")).posterior
    return dict(post=post, order=ast.literal_eval(m["gt_nuclei_order"]),
                sig_em=ast.literal_eval(m["sigma_em_gt"]), lik=m["gt_likelihood"],
                nu=float(m.get("gt_nu", 4.0)), sthe=post["sigma_gt_the"].values.ravel())


def predictive(run, nuc, rng):
    pred_ex = run["post"][f"tracked_pred_Ex_{nuc}"].values.ravel()
    err = config.GT_E_ERR[nuc]
    tot = np.sqrt(run["sthe"] ** 2 + err ** 2 + run["sig_em"].get(nuc, 0.0) ** 2)
    if run["lik"] == "studentt":
        nu = run["nu"]
        eps = tot * np.sqrt((nu - 2.0) / nu) * rng.standard_t(nu, size=pred_ex.shape)
    else:
        eps = rng.normal(0.0, tot)
    return pred_ex + eps


def main():
    rng = np.random.default_rng(SEED)
    fig, ax = plt.subplots(figsize=(ps.COLUMN_WIDTH, 2.8))
    out = ["# GT centroid posterior-predictive residuals [MeV] (draft/scripts/fig_gt_residuals.py)",
           "# resid = predictive percentile of E_GT minus measured E_GT; 'full' = Full posterior, 'set' = L10 posterior",
           "# all eight nuclei are fitted in both runs (GT set = all)",
           "functional\tnucleus\tE_exp\terr_exp\tsig_em\tfull_p2.5\tfull_p16\tfull_p50\tfull_p84\tfull_p97.5\tfull_in68\tfull_in95"
           "\tset_p2.5\tset_p16\tset_p50\tset_p84\tset_p97.5\tset_in68\tset_in95"]
    summary = []
    nuclei = None
    for F in RUNS:
        full, sett = load_run(F, RUNS[F][0]), load_run(F, RUNS[F][1])
        if nuclei is None:
            nuclei = sorted(full["order"], key=lambda k: int(k.split("_")[1]))
        col = ps.OKABE_ITO[F]
        cov = {"full68": 0, "full95": 0, "set68": 0, "set95": 0}
        res_med, var_model = [], []
        sthe_med = float(np.median(full["sthe"]))
        for i, nuc in enumerate(nuclei):
            y, err = config.GT_E_EXP[nuc], config.GT_E_ERR[nuc]
            row = [PLAIN_LABELS[F], nuc, f"{y:.2f}", f"{err:.2f}", f"{full['sig_em'].get(nuc, 0.0):.4f}"]
            for tag, run, dx, filled in (("full", full, -DX, True), ("set", sett, +DX, False)):
                q = np.percentile(predictive(run, nuc, rng), PCT) - y
                in68, in95 = (q[1] <= 0.0 <= q[3]), (q[0] <= 0.0 <= q[4])
                cov[tag + "68"] += in68; cov[tag + "95"] += in95
                x = i + OFF[F] + dx
                a = 1.0 if filled else 0.7
                ax.plot([x, x], [q[0], q[4]], color=col, lw=0.6, alpha=a, zorder=2)
                ax.plot([x, x], [q[1], q[3]], color=col, lw=1.8, alpha=a, zorder=2)
                ax.plot(x, q[2], marker=ps.MARKERS[F], ms=3.5, color=col, mec=col,
                        mfc=col if filled else "white", mew=0.8, zorder=3)
                row += [f"{v:.3f}" for v in q] + [str(int(in68)), str(int(in95))]
                if tag == "full":
                    res_med.append(q[2])
                    var_model.append(sthe_med ** 2 + err ** 2 + full["sig_em"].get(nuc, 0.0) ** 2)
            out.append("\t".join(row))
        rms = float(np.sqrt(np.mean(np.square(res_med)))); rms_model = float(np.sqrt(np.mean(var_model)))
        summary.append(f"{PLAIN_LABELS[F]}: Full inside 68% {cov['full68']}/8, 95% {cov['full95']}/8; "
                       f"L10 inside 68% {cov['set68']}/8, 95% {cov['set95']}/8; "
                       f"Full RMS residual at predictive median {rms:.2f} MeV, model expectation {rms_model:.2f} MeV "
                       f"(median sigma_GT {sthe_med:.2f} MeV)")
    ax.axhline(0.0, color="k", lw=0.8, zorder=1)
    for i in range(1, len(nuclei)):
        ax.axvline(i - 0.5, color="0.85", lw=0.5, zorder=0)
    ax.set_xticks(range(len(nuclei)))
    ax.set_xticklabels([tex_gt(k) for k in nuclei])
    ax.set_xlim(-0.6, len(nuclei) - 0.4)
    ax.set_ylim(*YLIM)
    ax.set_ylabel(r"$E_{\rm GT}^{\rm pred}-E_{\rm GT}^{\rm exp}$ (MeV)")
    ax.grid(axis="y", alpha=0.3, ls=":")
    h = [plt.Line2D([], [], marker=ps.MARKERS[F], ms=3.5, color=ps.OKABE_ITO[F], lw=0, label=ps.LABELS[F]) for F in RUNS]
    h += [plt.Line2D([], [], marker="o", ms=3.5, color="0.3", lw=1.8, label="Full"),
          plt.Line2D([], [], marker="o", ms=3.5, mfc="white", mec="0.3", color="0.3", lw=1.8, alpha=0.7, label="L10")]
    ax.legend(handles=h, loc="upper center", bbox_to_anchor=(0.5, -0.12), fontsize=7, ncol=5, frameon=False,
              columnspacing=0.9, handletextpad=0.4, handlelength=1.4)
    fig.tight_layout()
    ps.save(fig, OUT)
    out += ["", "# summary"] + summary
    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(out) + "\n")
    print("\n".join(summary))
    print("wrote", OUT_TXT)


if __name__ == "__main__":
    main()
