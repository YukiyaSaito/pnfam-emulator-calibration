#!/usr/bin/env python3
"""Posterior-predictive residuals of the 40-nucleus calibration pool, V7 final posteriors (2026-10-19).

One panel per functional. For every pool nucleus two posterior-predictive intervals of
log10(T_1/2 / s) minus the measured value:
  filled marker = full-pool posterior (the nucleus was fitted),
  open marker   = closest 10-nucleus set posterior (out-of-sample for the 30 non-members;
                  set members are marked by a shaded column).
Thick bar = 68% predictive interval (16-84%), thin bar = 95% (2.5-97.5%).

Predictive draw per posterior sample s and nucleus i, exactly the likelihood of bayes_mcmc.py:
  log10 T = f_i(V0_s, C1s_s) + scale * t_nu,  scale = sqrt(sig_the_s^2 + sig_exp_i^2 + sig_em_i^2 + sig_Q_i^2) * sqrt((nu-2)/nu)
(variance-matched Student-t, nu = 4, hl_likelihood/hl_nu read from the manifest; Normal if the run was Normal).
sig_exp_i, sig_em_i (test errors, batch 21001) and sig_Q_i are the manifest values of the full-pool run.

Coverage = fraction of nuclei whose measured value lies inside the interval. For the set posterior the
coverage is counted over the non-members only. Tick labels in grey: nuclei where the Q-correction term
sig_Q_i exceeds the posterior-median theory error (the interval is then dominated by the Q term).

Inputs : results/<F>/bayes_t4gtt4_sthe1_domw10_q_<set>_gtall[_<F>]_sem21001/centroid/{trace.nc,manifest.txt}
         results/<F>/HL/<nuc>/{params_best_n5_only_HL.txt,train_set.txt}  (promoted emulators)
         Beta_decay/config.py HL_EXP / HL_ERR (seconds)
Outputs: fig_v7_pool_residuals.{pdf,png}; table calibration/analysis/V7_POOL_RESIDUALS_2026-10-19.txt
"""
import argparse
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
COLORS, LABELS = ps.OKABE_ITO, ps.LABELS
PLAIN_LABELS = {"SKOP": "SkO'", "SLY4": "SLy4", "HFB1": "UNEDF1-HFB"}
mass = lambda n: int("".join(c for c in n if c.isdigit()))
tex_nuc = ps.nuc

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "figs", "fig_pool_residuals")
OUT_TXT = os.path.join(HERE, "..", "data", "pool_residuals.txt")
# 2026-10-26 (user): SkO' half-lives of Ney, Engel, Schunck, PRC 102, 034326 (2020), shown in the SkO' panel as
# log10(T_Ney / T_exp). Global table columns N, Z, log10(T/s); Ar48 (Z = 18) and Pt200 are not in it.
NEY_FILE = os.path.join(RP.LITERATURE, "data_Evan.txt")   # Ney et al. (2020) SkO' half-lives
NEY_PANEL = "SKOP"
TAG_TABLE = os.path.join(RP.DESIGN, "nuclei_tag_table.csv")   # Z, N per nucleus   # same content as analysis/V7_POOL_RESIDUALS_2026-10-19.txt
SEED = 20261019
RUNS = {
    "SKOP": ("full", "L10"),   # (Full set, adopted set) posteriors, Student-t likelihood; see repo_paths.posterior
    "SLY4": ("full", "L10"),
    "HFB1": ("full", "L10"),   # L10 = G10 for UNEDF1-HFB
}
PCT = (2.5, 16.0, 50.0, 84.0, 97.5)
G8_SETS = {F: RP.SET_KEYS[F]["G10"] for F in RP.FUNCTIONALS}   # G10 membership (hatched columns)
# RUNS second entry = the adopted set (L10 for every functional, 2026-11-05): L10_SKOP, L10_SLY4, G10_HFB1 (= L10_HFB1)
# 2026-11-19 (user): optional subset of functionals, e.g. "--funcs SKOP,HFB1 --suffix _2func" writes
# fig_pool_residuals_2func.{pdf,png} and pool_residuals_2func.txt; without options the three-panel figure is unchanged.
_ap = argparse.ArgumentParser()
_ap.add_argument("--funcs", default=",".join(RUNS), help="comma-separated subset of SKOP,SLY4,HFB1 (panel order as in RUNS)")
_ap.add_argument("--suffix", default="", help="suffix for the output stem and data file")
_args = _ap.parse_args()
FUNCS = [F for F in RUNS if F in _args.funcs.split(",")]
OUT += _args.suffix
OUT_TXT = OUT_TXT[:-4] + _args.suffix + ".txt"


class HLEmulator:
    """numpy copy of bayes_mcmc.HalfLifeEmulator: middle eigenvalue of D + da*S1 + db*S2 (n = 5)."""
    def __init__(self, functional, nuc, n=5):
        d = os.path.join(RP.EMULATORS, functional, "HL", nuc)
        ts = np.loadtxt(os.path.join(d, "train_set.txt"), delimiter=",")
        k = int(np.argmin(np.linalg.norm(ts - ts.mean(axis=0), axis=1)))
        self.cp_a, self.cp_b = float(ts[k, 0]), float(ts[k, 1])
        p = np.loadtxt(os.path.join(d, f"params_best_n{n}_only_HL.txt"))
        nu = n * (n + 1) // 2
        self.n = n
        self.D = np.diag(p[:n])
        iu = np.triu_indices(n)
        S1 = np.zeros((n, n)); S1[iu] = p[n:n + nu]; self.S1 = S1 + S1.T - np.diag(np.diag(S1))
        S2 = np.zeros((n, n)); S2[iu] = p[n + nu:n + 2 * nu]; self.S2 = S2 + S2.T - np.diag(np.diag(S2))

    def __call__(self, v0, c1s):
        lo_v, hi_v = config.PARAM_BOUNDS["vpair_t0"]
        lo_c, hi_c = config.PARAM_BOUNDS["override_cs0"]
        da = 2.0 * (np.asarray(v0) - lo_v) / (hi_v - lo_v) - 1.0 - self.cp_a
        db = 2.0 * (np.asarray(c1s) - lo_c) / (hi_c - lo_c) - 1.0 - self.cp_b
        M = self.D[None] + da[:, None, None] * self.S1[None] + db[:, None, None] * self.S2[None]
        return np.linalg.eigvalsh(M)[:, self.n // 2]


def read_manifest(path):
    m = {}
    for ln in open(path):
        if "=" in ln:
            k, v = ln.rstrip("\n").split("=", 1)
            m[k] = v
    return m


def load_run(functional, sub):
    d = RP.posterior(functional, sub, "studentt")
    m = read_manifest(os.path.join(d, "manifest.txt"))
    post = az.from_netcdf(os.path.join(d, "trace.nc")).posterior
    s = dict(v0=post["vpair"].values.ravel(), c1s=post["cs0"].values.ravel(),
             sthe=post["sigma_hl_the"].values.ravel(),
             lik=m["hl_likelihood"], nu=float(m.get("hl_nu", 4.0)),
             nuclei=ast.literal_eval(m["hl_nuclei_order"]),
             sig_em=ast.literal_eval(m["sigma_em_hl"]), sig_q=ast.literal_eval(m["sigma_q_hl"]))
    return s


def predictive(sample, emu, y_exp, sig_exp, sig_em, sig_q, rng):
    """Predictive log10 T draws (one per posterior sample) and the emulator value at every sample."""
    f = emu(sample["v0"], sample["c1s"])
    tot = np.sqrt(sample["sthe"] ** 2 + sig_exp ** 2 + sig_em ** 2 + sig_q ** 2)
    if sample["lik"] == "studentt":
        nu = sample["nu"]
        eps = tot * np.sqrt((nu - 2.0) / nu) * rng.standard_t(nu, size=f.shape)
    else:
        eps = rng.normal(0.0, tot)
    return f, f + eps


def load_ney():
    import csv
    a = np.genfromtxt(NEY_FILE, comments="#")
    table = {(int(round(z)), int(round(nn))): y for nn, z, y in a[:, :3]}
    zn = {r["nucleus"]: (int(r["Z"]), int(r["N"])) for r in csv.DictReader(open(TAG_TABLE))}
    return table, zn


def main():
    rng = np.random.default_rng(SEED)
    ney_table, zn_of = load_ney()
    ney_res = {}            # nucleus -> log10(T_Ney/T_exp)
    ours_res_skop = {}      # nucleus -> SkO' Full surface at median couplings minus measured (same quantity as f_med_full)
    fig, axes = plt.subplots(len(FUNCS), 1, figsize=(ps.TEXT_WIDTH, 2.2 * len(FUNCS)), sharex=True)
    axes = np.atleast_1d(axes)
    ax_of = dict(zip(FUNCS, axes))
    scratch = plt.figure()   # functionals not in FUNCS are still computed (same random stream as the three-panel figure) but drawn here and discarded
    # common x axis: the 42-nucleus pool; a functional's two excluded nuclei leave an empty column
    pool42 = sorted(config.HL_SET_POOL42, key=lambda n: (mass(n), n))
    xpos = {n: i for i, n in enumerate(pool42)}
    lines = ["# Posterior-predictive residuals of the calibration pool (draft/scripts/fig_pool_residuals.py, copy of paper_figs/fig_v7_pool_residuals.py)",
             "# resid = predictive percentile of log10(T/s) minus measured log10(T/s); 'full' = full-pool posterior, 'set' = 10-nucleus-set posterior",
             "# member = 1 if the nucleus is in the 10-nucleus set; qdom = 1 if sigma_Q_i > posterior-median sigma_the (full-pool run)",
             "# f_med = emulator at the posterior-median couplings minus measured value (full-pool run)",
             "# model expectation of the RMS residual = sqrt(mean over nuclei of sigma_the^2 + sigma_exp^2 + sigma_em^2 + sigma_Q^2) at the posterior-median sigma_the",
             "functional\tnucleus\tA\tmember\tqdom\tsig_exp\tsig_em\tsig_Q\tfull_p2.5\tfull_p16\tfull_p50\tfull_p84\tfull_p97.5\tfull_in68\tfull_in95"
             "\tset_p2.5\tset_p16\tset_p50\tset_p84\tset_p97.5\tset_in68\tset_in95\tf_med_full\tney_resid"]
    summary = []
    for F in RUNS:
        ax = ax_of[F] if F in ax_of else scratch.add_subplot()
        full = load_run(F, RUNS[F][0])
        sett = load_run(F, RUNS[F][1])
        nuclei = sorted(full["nuclei"], key=lambda n: (mass(n), n))
        members = set(sett["nuclei"])
        g8_members = set(config.select_hl_nuclei(G8_SETS[F]))
        sthe_med = float(np.median(full["sthe"]))
        v0_med, c1s_med = float(np.median(full["v0"])), float(np.median(full["c1s"]))
        cov = {"full68": 0, "full95": 0, "set68": 0, "set95": 0}
        n_non = 0
        fmed_res = []
        var_model = []      # per-nucleus total variance of the likelihood at the posterior-median theory error
        col = COLORS[F]
        for nuc in nuclei:
            i = xpos[nuc]
            emu = HLEmulator(F, nuc)
            y = np.log10(config.HL_EXP[nuc])
            sig_exp = config.HL_ERR[nuc] / (config.HL_EXP[nuc] * np.log(10.0))
            sig_em, sig_q = full["sig_em"][nuc], full["sig_q"][nuc]
            res = {}
            for tag, smp in (("full", full), ("set", sett)):
                f, pred = predictive(smp, emu, y, sig_exp, sig_em, sig_q, rng)
                q = np.percentile(pred, PCT) - y
                res[tag] = (q, (q[1] <= 0.0 <= q[3]), (q[0] <= 0.0 <= q[4]))
                if tag == "full":
                    fm = float(emu(np.array([v0_med]), np.array([c1s_med]))[0] - y)
                    fmed_res.append(fm)
                    var_model.append(sthe_med ** 2 + sig_exp ** 2 + sig_em ** 2 + sig_q ** 2)
            cov["full68"] += res["full"][1]; cov["full95"] += res["full"][2]
            is_member = nuc in members
            if not is_member:
                n_non += 1
                cov["set68"] += res["set"][1]; cov["set95"] += res["set"][2]
            qdom = sig_q > sthe_med
            # drawing
            if is_member:
                ax.axvspan(i - 0.5, i + 0.5, color="0.88", zorder=0)
            if nuc in g8_members:
                ax.axvspan(i - 0.5, i + 0.5, facecolor="none", edgecolor="0.6", hatch="////", lw=0, zorder=0)
            q = res["full"][0]
            ax.plot([i - 0.18] * 2, [q[0], q[4]], color=col, lw=0.6, zorder=2)
            ax.plot([i - 0.18] * 2, [q[1], q[3]], color=col, lw=1.8, zorder=2)
            ax.plot(i - 0.18, q[2], marker="o", ms=3, color=col, mec=col, zorder=3)
            q = res["set"][0]
            ax.plot([i + 0.18] * 2, [q[0], q[4]], color=col, lw=0.6, alpha=0.7, zorder=2)
            ax.plot([i + 0.18] * 2, [q[1], q[3]], color=col, lw=1.8, alpha=0.7, zorder=2)
            ax.plot(i + 0.18, q[2], marker="o", ms=3, mfc="white", mec=col, mew=0.8, zorder=3)
            if F == NEY_PANEL:
                ours_res_skop[nuc] = fmed_res[-1]
                y_ney = ney_table.get(zn_of[nuc])
                if y_ney is not None:
                    ney_res[nuc] = float(y_ney) - y
                    ax.plot(i, ney_res[nuc], marker="x", ms=4, mew=0.9, color="k", ls="none", zorder=4)
            lines.append("\t".join([F, nuc, str(mass(nuc)), str(int(is_member)), str(int(qdom)),
                                    f"{sig_exp:.4f}", f"{sig_em:.4f}", f"{sig_q:.4f}"]
                                   + [f"{v:.3f}" for v in res["full"][0]] + [str(int(res["full"][1])), str(int(res["full"][2]))]
                                   + [f"{v:.3f}" for v in res["set"][0]] + [str(int(res["set"][1])), str(int(res["set"][2]))]
                                   + [f"{fmed_res[-1]:.3f}"]
                                   + [f"{ney_res[nuc]:.3f}" if (F == NEY_PANEL and nuc in ney_res) else "nan"]))
        n = len(nuclei)
        rms_med = float(np.sqrt(np.mean(np.square(fmed_res))))
        rms_model = float(np.sqrt(np.mean(var_model)))   # model's expected RMS residual (same nuclei)
        ax.axhline(0.0, color="k", lw=0.8, zorder=1)
        ax.set_xlim(-0.6, len(pool42) - 0.4)
        ax.set_xticks(range(len(pool42)))
        ax.set_xticklabels([tex_nuc(nn) for nn in pool42], rotation=90, fontsize=7)
        # (grey tick labels for Q-dominated nuclei dropped, user 2026-10-19; the qdom flag stays in the table)
        ax.set_ylabel(r"$\log_{10}(T_{1/2}^{\rm pred}/T_{1/2}^{\rm exp})$")
        ax.grid(axis="y", alpha=0.3, ls=":")
        ax.text(0.01, 0.96, LABELS[F], transform=ax.transAxes, ha="left", va="top", color=col, zorder=6,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.7))
        # (coverage text removed 2026-10-20, user; the numbers stay in the data file summary)
        ax.set_ylim(-7.5, 7.5)
        summary.append((F, n, cov, n_non, rms_med, rms_model, sthe_med))
    plt.close(scratch)
    lines = [l for l in lines if l.startswith("#") or l.startswith("functional") or l.split("\t")[0] in FUNCS]
    summary = [t for t in summary if t[0] in FUNCS]
    # legend on the top panel (user 2026-10-28), lower left, below the SkO' bars
    ax = axes[0]
    h = [plt.Line2D([], [], marker="o", ms=3, color="0.3", lw=1.8, label="Full (68\\% thick, 95\\% thin)"),
         plt.Line2D([], [], marker="o", ms=3, mfc="white", mec="0.3", color="0.3", lw=1.8, alpha=0.7, label="L10"),
         plt.Rectangle((0, 0), 1, 1, fc="0.88", ec="none", label="L10 member"),
         plt.Rectangle((0, 0), 1, 1, fc="none", ec="0.6", hatch="////", lw=0, label="G10 member"),
         plt.Line2D([], [], marker="x", ms=4, mew=0.9, color="k", ls="none", label=r"Ney \textit{et al.} (SkO$'$)")]
    ax.legend(handles=h, loc="lower left", fontsize=7, ncol=5, frameon=True, framealpha=0.7,
              edgecolor="none", fancybox=False).set_zorder(6)
    fig.tight_layout(h_pad=0.8)
    ps.save(fig, OUT)
    lines.append("")
    lines.append("# summary")
    for F, n, cov, n_non, rms_med, rms_model, sthe_med in summary:
        lines.append(f"{PLAIN_LABELS[F]}: full pool coverage 68% {cov['full68']}/{n} = {cov['full68']/n:.2f}, 95% {cov['full95']}/{n} = {cov['full95']/n:.2f}; "
                     f"set posterior on non-members 68% {cov['set68']}/{n_non} = {cov['set68']/n_non:.2f}, 95% {cov['set95']}/{n_non} = {cov['set95']/n_non:.2f}; "
                     f"RMS residual at median couplings {rms_med:.3f} log10 s vs model expectation sqrt(mean(sigma_the^2+sigma_exp^2+sigma_em^2+sigma_Q^2)) = {rms_model:.3f} (median sigma_HL {sthe_med:.3f})")
        print(lines[-1])
    common = sorted(set(ney_res) & set(ours_res_skop))
    if common:   # empty when SKOP is not among FUNCS
        rms_ney = float(np.sqrt(np.mean([ney_res[k] ** 2 for k in common])))
        rms_ours = float(np.sqrt(np.mean([ours_res_skop[k] ** 2 for k in common])))
        lines.append(f"Ney et al. SkO' (log10(T_Ney/T_exp)) on {len(common)} pool nuclei (no Ar48, Pt200): RMS {rms_ney:.3f}; "
                     f"our SkO' Full surface at median couplings on the same nuclei: RMS {rms_ours:.3f}")
        print(lines[-1])
    with open(OUT_TXT, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("wrote", OUT_TXT)


if __name__ == "__main__":
    main()
