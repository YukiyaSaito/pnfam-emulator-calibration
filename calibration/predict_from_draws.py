#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Predicted half-life and rate with uncertainty from direct pnFAM runs at posterior draws.

Per nucleus: fit a local surface log10 T = f(vpair, cs0) (linear or quadratic,
least squares) to the K runs of batch 30001, report the fit residual RMS
(fit_rms = sqrt(sum resid^2 / (K - n_coef)); the emulator-adequacy number),
then push EVERY posterior sample through f and add the theory error drawn
jointly from the trace and the Q-correction term, in quadrature:
    log10 T_pred = f(vpair_s, cs0_s) + N(0, sqrt(sigma_hl_the_s^2 + sigma_Q,i^2))
    sigma_Q,i = 5 sigma_Q / (Q_corr,i ln 10)   [log10 s]
The draw follows the run's half-life likelihood (--pred-dist auto, the default, reads
hl_likelihood / hl_nu from <trace-dir>/manifest.txt): for hl_likelihood=studentt the
error term is s * t_nu with s = sqrt(sigma_hl_the_s^2 + sigma_Q,i^2) * sqrt((nu-2)/nu),
the variance-matched scale of Beta_decay/bayes_mcmc.py (same variance as the Normal draw);
for hl_likelihood=normal it is the Normal draw above (2026-10-15).
with sigma_Q the per-functional constant below and Q_corr,i the median of the
corrected Q (beta.out HFB_Qval) over the nucleus's runs. --no-q-term sets
sigma_Q,i = 0 (reproduces the pre-2026-10-13 behaviour bit for bit).
Outputs the predictive median and 16/84, 2.5/97.5 percentiles of log10 T and
of the rate ln2/T [1/s] (same predictive samples), the parameter-only spread,
the surface at the posterior median couplings, log10T_at_median = f(median vpair,
median cs0) [log10 s] (no error terms) and rate_at_median = ln2 / 10^log10T_at_median [1/s],
and, if the nucleus has a measured half-life in config.HL_EXP, the residual
and whether it falls inside the 68% / 95% bands. Also prints the fraction of
posterior samples outside the convex hull and outside the bounding box of the
draws in (V0, C1s); warning if either exceeds 5%.

  python3 predict_from_draws.py --functional SKOP|SLY4|HFB1 --trace-dir results/<F>/<run>/centroid \
      [--nuclei Pt204,Hg206] [--model linear|quadratic] [--no-q-term] [--out-dir DIR]
      [--pred-dist auto|normal|studentt]
Self-test (Pt200 SKOP_PT200LOC frozen points as the 'runs'):
  python3 predict_from_draws.py --fixture --trace-dir results/SKOP/<run>/centroid [--no-q-term]
Synthetic self-test (analytic surface, known coefficients):
  python3 predict_from_draws.py --selftest --trace-dir results/<F>/<run>/centroid
"""
import argparse
import glob
import os
import sys
import tempfile

import numpy as np

from frozen_common import CAL, SMLR, load_posterior, read_cols, read_manifest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # config.py lives beside this file
import config  # noqa: E402

OUT_DIR = os.path.join(SMLR, "paper", "data")   # predictions_<F>_batch30001.tsv read by the N = 126 figures
BATCH = 30001
PCT = [2.5, 16, 50, 84, 97.5]
LN2 = np.log(2.0)

# sigma_Q [MeV]: global Q-correction uncertainty per functional = residual standard deviation of the
# Q_HFB - Q_exp fit over 154 measured even-even nuclei; source FRIB_projects/Qsurvey/q_correction.py COEFF
# (qcorr/q_correction.py; per-nucleus values in data/errors/sigma_q_<F>.dat).
SIGMA_Q = {"SKOP": 0.580, "SLY4": 0.619, "HFB1": 0.564}
# Fixture only: the frozen Pt200 points have no Q column. Corrected Q of SkO' Pt200 = beta.out HFB_Qval of
# scratch/Calibration/SKOP/outputs/rates/Pt200_sample_{1,37,100}_final.tar.gz (0.18801 MeV in all three;
# logfile HFB_Qval 0.110013 + Q_eff 0.078). Read 2026-10-13.
FIXTURE_QCORR = 0.18801


def design(v, c, model):
    a = (v + 250.0) / 250.0; b = (c - 125.0) / 125.0          # box-scaled to [-1,1]
    cols = [np.ones_like(a), a, b]
    if model == "quadratic":
        cols += [a * a, a * b, b * b]
    return np.column_stack(cols)


def fit_predict(v_run, c_run, y_run, model, vpair, cs0, sthe, sq=0.0, seed=0, dist="normal", nu=None):
    """sq = sigma_Q,i [log10 s]; drawn in quadrature with sthe in ONE draw,
    so sq = 0 reproduces the no-Q-term draws exactly (hypot(x, 0) == |x|).
    dist = "normal": N(0, hypot(sthe, sq)) (unchanged from before 2026-10-15).
    dist = "studentt": hypot(sthe, sq) * sqrt((nu-2)/nu) * t_nu, the variance-matched
    Student-t of the bayes_mcmc.py studentt branch (variance = hypot(sthe, sq)^2)."""
    X = design(v_run, c_run, model)
    beta, *_ = np.linalg.lstsq(X, y_run, rcond=None)
    resid = y_run - X @ beta
    dof = max(len(y_run) - X.shape[1], 1)
    fit_rms = float(np.sqrt(np.sum(resid ** 2) / dof))
    f_post = design(vpair, cs0, model) @ beta
    rng = np.random.default_rng(seed)
    if dist == "studentt":
        s = np.hypot(sthe, sq) * np.sqrt((nu - 2.0) / nu)
        pred = f_post + s * rng.standard_t(nu, size=np.shape(f_post))
    else:
        pred = f_post + rng.normal(0.0, np.hypot(sthe, sq))
    return beta, fit_rms, f_post, pred


def resolve_pred_dist(choice, trace_dir):
    """(dist, nu, text) for --pred-dist; auto follows hl_likelihood in <trace_dir>/manifest.txt,
    nu = hl_nu from the same manifest."""
    path = os.path.join(trace_dir, "manifest.txt")
    man = read_manifest(trace_dir) if os.path.isfile(path) else {}
    nu = float(man["hl_nu"]) if "hl_nu" in man else None
    if choice == "auto":
        if "hl_likelihood" not in man:
            raise SystemExit(f"--pred-dist auto: no hl_likelihood in {path}; give --pred-dist normal|studentt")
        dist, src = man["hl_likelihood"], f"auto: manifest hl_likelihood={man['hl_likelihood']}"
        if dist not in ("normal", "studentt"):
            raise SystemExit(f"--pred-dist auto: unknown hl_likelihood={dist} in {path}")
    else:
        dist, src = choice, "set by --pred-dist"
    if dist == "studentt" and (nu is None or nu <= 2.0):
        raise SystemExit(f"Student-t draw needs hl_nu > 2 in {path} (got {nu})")
    text = (f"studentt nu={nu:g} (scale = hypot(sigma_the, sigma_Q_i)*sqrt((nu-2)/nu)), {src}"
            if dist == "studentt" else f"normal, {src}")
    return dist, nu, text


def surface_at(beta, v, c, model):
    """fitted surface f(v, c) [log10 s] at scalar couplings v = vpair, c = cs0 [MeV fm^3]."""
    return (design(np.atleast_1d(np.asarray(v, float)), np.atleast_1d(np.asarray(c, float)), model) @ beta)[0]


def sigma_q_i(sigma_q, q_corr):
    """5 sigma_Q / (Q_corr ln 10) [log10 s] (allowed phase space, T ~ Q^-5)."""
    return 5.0 * sigma_q / (q_corr * np.log(10.0))


def read_batch_table(path):
    """halflives_batch<B>.txt -> {id: (log10T [log10 s], Q_corr [MeV] or nan)}, usable rows only.
    New format (extract_prediction_batch.py, 9 columns):
        sample vpair_t0 override_cs0 log10T rate Q_corr EQRPAmax hfb_conv fam_conv
    Old format (extract_frozen_batch.py): sample total_hl[s] [HFB_Conv FAM_Conv].
    Rows are dropped only on pnFAM flags / nonpositive half-life."""
    h = {}
    for ln in open(path):
        if ln.lstrip().startswith("#") or not ln.strip():
            continue
        q = ln.split()
        i = int(float(q[0]))
        if len(q) == 9:
            y = float(q[3])
            if np.isfinite(y) and q[7] == "Yes" and q[8] == "Yes":
                h[i] = (y, float(q[5]))
        elif float(q[1]) > 0 and (len(q) < 4 or (q[2] == "Yes" and q[3] == "Yes")):
            h[i] = (float(np.log10(float(q[1]))), np.nan)
    return h


def outside_fractions(draw_v, draw_c, vpair, cs0):
    """(fraction outside convex hull, fraction outside bounding box) of posterior samples
    relative to the draws in (V0, C1s)."""
    from scipy.spatial import Delaunay
    pts = np.column_stack([vpair, cs0])
    hull = Delaunay(np.column_stack([draw_v, draw_c]))
    f_hull = float(np.mean(hull.find_simplex(pts) < 0))
    inbox = ((vpair >= draw_v.min()) & (vpair <= draw_v.max()) &
             (cs0 >= draw_c.min()) & (cs0 <= draw_c.max()))
    return f_hull, float(np.mean(~inbox))


def coverage_line(label, draw_v, draw_c, vpair, cs0):
    fh, fb = outside_fractions(draw_v, draw_c, vpair, cs0)
    warn = "  WARNING: > 5% of the posterior outside the draws" if max(fh, fb) > 0.05 else ""
    return (f"coverage {label}: {len(draw_v)} draws; posterior outside convex hull {100 * fh:.1f}%, "
            f"outside bounding box {100 * fb:.1f}%{warn}")


def mixture_quantile(p, f, s, cdf, pdf):
    """p-quantile of f_i + s_i X (equal-weight mixture over samples i; X with cdf/pdf) and the
    mixture density there."""
    from scipy.optimize import brentq
    g = lambda q: np.mean(cdf((q - f) / s)) - p
    lo, hi = float(np.min(f) - 50 * np.max(s)), float(np.max(f) + 50 * np.max(s))
    q = brentq(g, lo, hi, xtol=1e-12)
    return q, float(np.mean(pdf((q - f) / s) / s))


def selftest(vpair, cs0, sthe, functional, nu=4.0):
    """Analytic surface log10 T = a + b V0 + c C1s at 20 V0-stratified posterior draws."""
    ok = True
    a0, b0, c0 = 2.0, -0.004, 0.003                    # log10 s, per MeV fm^3, per MeV fm^3
    rng = np.random.default_rng(12345)
    order = np.argsort(vpair)
    idx = [rng.choice(s) for s in np.array_split(order, 20)]
    dv, dc = vpair[idx], cs0[idx]
    qc = 1.5                                            # MeV, fake corrected Q
    with tempfile.TemporaryDirectory() as td:
        p = os.path.join(td, "halflives_batch30001.txt")
        with open(p, "w") as f:
            f.write("# sample vpair_t0 override_cs0 log10T rate Q_corr EQRPAmax hfb_conv fam_conv\n")
            for k, (v, c) in enumerate(zip(np.round(dv, 4), np.round(dc, 4))):   # 4 decimals as written
                y = a0 + b0 * v + c0 * c
                f.write(f"{30001 + k}\t{v:.4f}\t{c:.4f}\t{y:.12f}\t{LN2 / 10 ** y:.6e}\t{qc:.5f}\t5.0\tYes\tYes\n")
            f.write("30099\t-300.0\t100.0\tnan\t-1e-3\t1.5\t5.0\tYes\tYes\n")      # must be dropped
            f.write("30098\t-300.0\t100.0\t1.0\t0.069\t1.5\t5.0\tYes\tNo\n")        # must be dropped
        h = read_batch_table(p)
    ids = sorted(h)
    y = np.array([h[i][0] for i in ids])
    rows_ok = len(ids) == 20
    print(f"SELFTEST reader: {len(ids)} usable rows of 22 (expect 20)  {'ok' if rows_ok else 'FAIL'}")
    ok &= rows_ok
    v4, c4 = np.round(dv, 4), np.round(dc, 4)          # couplings as written to the table
    sq = sigma_q_i(SIGMA_Q.get(functional, 0.58), np.median([h[i][1] for i in ids]))
    beta, fit_rms, f_post, pred = fit_predict(v4, c4, y, "linear", vpair, cs0, sthe, sq)
    b_fit, c_fit = beta[1] / 250.0, beta[2] / 125.0
    a_fit = beta[0] + beta[1] - beta[2]               # f = b0 + b1 (V+250)/250 + b2 (C-125)/125
    err = max(abs(a_fit - a0), abs(b_fit - b0) * 250, abs(c_fit - c0) * 125)
    print(f"SELFTEST surface: a {a_fit:.10f} (true {a0})  b {b_fit:.10f} (true {b0})  c {c_fit:.10f} "
          f"(true {c0})  fit_rms {fit_rms:.2e}  max scaled error {err:.1e}  {'ok' if err < 1e-8 else 'FAIL'}")
    ok &= err < 1e-8
    vm, cm = np.median(vpair), np.median(cs0)
    f_med = a0 + b0 * vm + c0 * cm
    f_mean = a0 + b0 * vpair.mean() + c0 * cs0.mean()
    # (i) error terms off: predictive samples == surface values; their median == surface at the
    #     posterior median (exact in 1-D, close in 2-D) within the sampling noise of a median
    _, _, f0, pred0 = fit_predict(v4, c4, y, "linear", vpair, cs0, np.zeros_like(sthe), 0.0)
    same = np.allclose(pred0, f0, rtol=0, atol=1e-12)
    print(f"SELFTEST zero error terms: predictive samples == surface values  {'ok' if same else 'FAIL'}")
    ok &= same
    med0 = float(np.median(pred0))
    noise0 = np.sqrt(np.pi / 2.0) * pred0.std() / np.sqrt(len(pred0))
    d0 = med0 - f_med
    print(f"SELFTEST median (error terms off): predictive {med0:.4f} vs surface at posterior median "
          f"(V0 {vm:.2f}, C1s {cm:.2f}) {f_med:.4f}, difference {d0:+.4f}; sampling noise of the median "
          f"{noise0:.4f} (sqrt(pi/2) sd/sqrt(n), n = {len(pred0)})  {'ok' if abs(d0) < 3 * noise0 else 'FAIL'}")
    ok &= abs(d0) < 3 * noise0
    # (ii) error terms on: zero-mean errors -> predictive MEAN == surface at the posterior mean (exact for a
    #      linear surface); the predictive MEDIAN moves from f(median) toward f(mean) when the error terms
    #      are wide compared with the parameter spread and the posterior is skewed (printed, not tested)
    mn = float(pred.mean())
    noise1 = pred.std() / np.sqrt(len(pred))
    d1 = mn - f_mean
    print(f"SELFTEST mean (error terms on): predictive {mn:.4f} vs surface at posterior mean {f_mean:.4f}, "
          f"difference {d1:+.4f}; sampling noise {noise1:.4f}  {'ok' if abs(d1) < 3 * noise1 else 'FAIL'}")
    ok &= abs(d1) < 3 * noise1
    print(f"SELFTEST info (error terms on): predictive median {float(np.median(pred)):.4f}; surface at posterior "
          f"median {f_med:.4f}, at posterior mean {f_mean:.4f} (parameter-only sd {f0.std():.4f}, "
          f"error-term sd {np.sqrt(np.mean(sthe ** 2) + sq ** 2):.4f})")
    # variance bookkeeping: var(pred - f_post) = mean(sthe^2) + sq^2
    vr, vexp = float(np.var(pred - f_post)), float(np.mean(sthe ** 2) + sq ** 2)
    rel = abs(vr / vexp - 1.0)
    tol = 4 * np.sqrt(2.0 / len(pred))
    print(f"SELFTEST variance: var(pred - f) {vr:.4f} vs mean(sigma_the^2) + sigma_Q,i^2 {vexp:.4f} "
          f"(sigma_Q,i {sq:.4f} at Q_corr {qc} MeV)  rel. diff {rel:.3f} (tol {tol:.3f})  "
          f"{'ok' if rel < tol else 'FAIL'}")
    ok &= rel < tol
    # rate percentiles from the same samples
    rq = np.percentile(LN2 / 10.0 ** pred, PCT)
    tq = np.percentile(pred, PCT)
    rr = np.max(np.abs(rq[::-1] * 10.0 ** tq / LN2 - 1.0))
    print(f"SELFTEST rate: max |rate_p(100-x) * 10^log10T_p(x) / ln2 - 1| = {rr:.1e}  "
          f"{'ok' if rr < 1e-3 else 'FAIL'}")
    ok &= rr < 1e-3
    # hull: unit square
    sv, sc = np.array([0.0, 1, 1, 0]), np.array([0.0, 0, 1, 1])
    fh, fb = outside_fractions(sv, sc, np.array([0.5, 2.0, 0.1, -0.1]), np.array([0.5, 0.5, 0.9, 0.5]))
    hull_ok = abs(fh - 0.5) < 1e-12 and abs(fb - 0.5) < 1e-12
    print(f"SELFTEST hull/box on the unit square: outside {fh:.2f} / {fb:.2f} (expect 0.50 / 0.50)  "
          f"{'ok' if hull_ok else 'FAIL'}")
    ok &= hull_ok
    # surface at the posterior median couplings (log10T_at_median column) vs the analytic surface
    fm_fit = float(surface_at(beta, vm, cm, "linear"))
    dm = abs(fm_fit - f_med)
    print(f"SELFTEST surface at posterior median: fitted {fm_fit:.10f} vs analytic {f_med:.10f}  "
          f"|diff| {dm:.1e}  {'ok' if dm < 1e-8 else 'FAIL'}")
    ok &= dm < 1e-8
    # ---- Student-t predictive draw (2026-10-15) ----
    from scipy import stats
    S = np.hypot(sthe, sq)                                # log10 s, Normal-draw sd per sample
    k = np.sqrt((nu - 2.0) / nu)                          # variance-matching factor
    # (a) Normal path unchanged: same numbers as the pre-2026-10-15 expression
    ref = f_post + np.random.default_rng(0).normal(0.0, S)
    same_n = np.array_equal(pred, ref)
    print(f"SELFTEST normal draw unchanged: pred == f + default_rng(0).normal(0, hypot(sigma_the, sigma_Q_i))  "
          f"{'ok' if same_n else 'FAIL'}")
    ok &= same_n
    # (b) t draw: exact scale, (pred - f) / (hypot * sqrt((nu-2)/nu)) == default_rng(0).standard_t(nu)
    _, _, f_t, pred_t = fit_predict(v4, c4, y, "linear", vpair, cs0, sthe, sq, dist="studentt", nu=nu)
    tz = (pred_t - f_t) / (S * k)
    dz = float(np.max(np.abs(tz - np.random.default_rng(0).standard_t(nu, size=len(tz)))))
    print(f"SELFTEST t draw scale: max |(pred - f)/(hypot*sqrt((nu-2)/nu)) - standard_t({nu:g})| {dz:.1e}  "
          f"{'ok' if dz < 1e-9 else 'FAIL'}")
    ok &= dz < 1e-9
    # (c) variance: analytic var(t_nu) * k^2 = 1, and the sample variance. t_4 has no finite 4th moment, so
    #     the sample variance of n = len(posterior) draws scatters beyond the Normal tolerance; the test uses
    #     the posterior tiled R times (n = R * len) with tolerance 0.03.
    va = float(stats.t.var(nu) * k ** 2)
    R = 500
    _, _, f_R, pred_R = fit_predict(v4, c4, y, "linear", np.tile(vpair, R), np.tile(cs0, R), np.tile(sthe, R), sq,
                                    dist="studentt", nu=nu)
    vr_t = float(np.var(pred_R - f_R))
    rel_t = abs(vr_t / vexp - 1.0)
    vr_t1 = float(np.var(pred_t - f_t))
    print(f"SELFTEST t variance: var(t_{nu:g}) * (nu-2)/nu = {va:.12f} (scipy; expect 1); var(pred - f) over "
          f"{R} x {len(pred)} draws {vr_t:.4f} vs mean(sigma_the^2) + sigma_Q,i^2 {vexp:.4f}  rel. diff {rel_t:.4f} "
          f"(tol 0.030)  {'ok' if rel_t < 0.03 and abs(va - 1) < 1e-9 else 'FAIL'}   "
          f"[info: over the {len(pred)} posterior draws alone {vr_t1:.4f}, rel. diff {abs(vr_t1 / vexp - 1):.4f}]")
    ok &= rel_t < 0.03 and abs(va - 1) < 1e-9
    # (d) band widths vs the exact quantiles of the mixture f_i + S_i X (Normal) and f_i + S_i k t_nu (t)
    zn, zt = stats.norm, stats.t(nu)
    for lo_p, hi_p, lab in ((0.16, 0.84, "68% (p84-p16)"), (0.025, 0.975, "95% (p97.5-p2.5)")):
        res_w = {}
        for name, smp, sc, D in (("normal", pred, S, zn), ("studentt", pred_t, S * k, zt)):
            (ql, dl), (qh, dh) = (mixture_quantile(pp, f_post, sc, D.cdf, D.pdf) for pp in (lo_p, hi_p))
            w_ex = qh - ql
            w_s = float(np.percentile(smp, 100 * hi_p) - np.percentile(smp, 100 * lo_p))
            sd = np.sqrt(lo_p * (1 - lo_p) / len(smp) / dl ** 2 + hi_p * (1 - hi_p) / len(smp) / dh ** 2)
            good = abs(w_s - w_ex) < 4 * sd
            res_w[name] = (w_s, w_ex)
            print(f"SELFTEST width {lab} {name:8s}: sample {w_s:.4f} vs exact mixture {w_ex:.4f}  "
                  f"diff {w_s - w_ex:+.4f} (4 sd {4 * sd:.4f})  {'ok' if good else 'FAIL'}")
            ok &= good
        z_ref = (zt.ppf(hi_p) - zt.ppf(lo_p)) * k / (zn.ppf(hi_p) - zn.ppf(lo_p))
        print(f"SELFTEST width ratio t/normal {lab}: sample {res_w['studentt'][0] / res_w['normal'][0]:.4f}, exact "
              f"mixture {res_w['studentt'][1] / res_w['normal'][1]:.4f}; single-scale reference "
              f"t_{nu:g} q{hi_p:g} {zt.ppf(hi_p):.4f} x {k:.4f} / normal q{hi_p:g} {zn.ppf(hi_p):.4f} = {z_ref:.4f} "
              f"(scipy.stats)")
    print("SELFTEST " + ("PASSED" if ok else "FAILED"))
    return ok


def main():
    global BATCH
    ap = argparse.ArgumentParser()
    ap.add_argument("--functional", choices=["SKOP", "SLY4", "HFB1", "UNE1"])
    ap.add_argument("--trace-dir", required=True)
    ap.add_argument("--nuclei", default=None)
    ap.add_argument("--model", default="linear", choices=["linear", "quadratic"])
    ap.add_argument("--fixture", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--no-q-term", action="store_true", help="sigma_Q,i = 0 (pre-2026-10-13 behaviour)")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--batch", type=int, default=30001, help="pnFAM draw batch (default 30001; 30201 = final-posterior draws, 2026-10-17)")
    ap.add_argument("--pred-dist", default="auto", choices=["auto", "normal", "studentt"],
                    help="distribution of the predictive error draw; auto = hl_likelihood in <trace-dir>/manifest.txt")
    args = ap.parse_args()
    BATCH = args.batch   # module default 30001 overridden per call
    vpair, cs0, sthe = load_posterior(args.trace_dir)
    pdist, pnu, ptxt = resolve_pred_dist(args.pred_dist, args.trace_dir)

    if args.selftest:
        print(f"predictive draw in production for this trace: {ptxt}")
        raise SystemExit(0 if selftest(vpair, cs0, sthe, args.functional or "SKOP", pnu if pnu else 4.0) else 1)

    F = "SKOP" if args.fixture else args.functional
    if F is None:
        ap.error("--functional required")
    if not args.no_q_term and F not in SIGMA_Q:
        ap.error(f"no sigma_Q for {F} (have {sorted(SIGMA_Q)}); use --no-q-term")
    sig_q = 0.0 if args.no_q_term else SIGMA_Q[F]
    os.makedirs(args.out_dir, exist_ok=True)

    jobs = []                    # (nuc, v, c, y, q_corr array, draw_v, draw_c)
    if args.fixture:
        d = os.path.join(CAL, "SKOP_PT200LOC_retired20260924", "HL", "Pt200")
        p, h = read_cols(os.path.join(d, "params_frozen.txt")), read_cols(os.path.join(d, "halflives_frozen.txt"))
        ids = sorted(set(p) & set(h))
        v, c = np.array([p[i][0] for i in ids]), np.array([p[i][1] for i in ids])
        jobs.append(("Pt200", v, c, np.log10(np.array([h[i][0] for i in ids])),
                     np.full(len(ids), FIXTURE_QCORR), v, c))
    else:
        root = os.path.join(SMLR, "data", "n126", F)
        nuclei = (args.nuclei.split(",") if args.nuclei else
                  sorted(os.path.basename(os.path.dirname(os.path.dirname(q))) for q in
                         glob.glob(os.path.join(root, "*", f"batch{BATCH}", f"halflives_batch{BATCH}.txt"))))
        for nuc in nuclei:
            bdir = os.path.join(root, nuc, f"batch{BATCH}")
            p = read_cols(os.path.join(bdir, f"params_batch{BATCH}.txt"))
            h = read_batch_table(os.path.join(bdir, f"halflives_batch{BATCH}.txt"))
            ids = sorted(set(p) & set(h))
            pid = sorted(p)
            jobs.append((nuc, np.array([p[i][0] for i in ids]), np.array([p[i][1] for i in ids]),
                         np.array([h[i][0] for i in ids]), np.array([h[i][1] for i in ids]),
                         np.array([p[i][0] for i in pid]), np.array([p[i][1] for i in pid])))

    qtxt = "OFF (--no-q-term)" if args.no_q_term else f"sigma_Q {sig_q:.3f} MeV"
    v_med, c_med = float(np.median(vpair)), float(np.median(cs0))      # posterior medians [MeV fm^3]
    print(f"{F}  trace {args.trace_dir}  model {args.model}  sigma_the median {np.median(sthe):.3f}  "
          f"Q term {qtxt}  posterior samples {len(vpair)}")
    print(f"predictive draw: {ptxt}")
    print(f"posterior medians: vpair {v_med:.3f}, cs0 {c_med:.3f} (log10T_at_median = surface there, no error terms)")
    # coverage of the posterior by the draws (per functional; draws are shared by all nuclei)
    if jobs:
        dv0, dc0 = jobs[0][5], jobs[0][6]
        same = all(len(j[5]) == len(dv0) and np.allclose(j[5], dv0) and np.allclose(j[6], dc0) for j in jobs)
        print(coverage_line(F + (" (all nuclei share these draws)" if same else f" ({jobs[0][0]} draws)"),
                            dv0, dc0, vpair, cs0))
        if not same:
            print("  NOTE: draw sets differ between nuclei; see per-nucleus lines below")
    print(f"{'nuc':7s}{'K':>3s}{'fit_rms':>8s}{'param_sd':>9s}{'Q_corr':>8s}{'sig_Q':>7s}"
          f"{'median':>8s}{'16%':>7s}{'84%':>7s}{'2.5%':>7s}{'97.5%':>7s}{'exp':>7s}{'resid':>7s}  in68 in95"
          f"{'f(med)':>8s}")
    rows, notes = [], []
    for nuc, v, c, y, qc, dv, dc in jobs:
        if len(y) < design(v, c, args.model).shape[1] + 1:
            notes.append(f"{nuc}: only {len(y)} usable runs, skipped")
            continue
        qfin = qc[np.isfinite(qc)]
        if args.no_q_term:
            q_med, sq = (float(np.median(qfin)) if len(qfin) else np.nan), 0.0
        else:
            if not len(qfin):
                raise SystemExit(f"{nuc}: no Q_corr column (old-format table); use --no-q-term")
            q_med = float(np.median(qfin))
            sq = float(sigma_q_i(sig_q, q_med))
        if len(qfin) and qfin.max() - qfin.min() > 1e-4:
            notes.append(f"{nuc}: Q_corr varies across runs: {qfin.min():.5f}..{qfin.max():.5f} MeV "
                         f"(median used)")
        if len(y) < len(dv):
            notes.append(f"{nuc}: {len(y)}/{len(dv)} runs usable; " +
                         coverage_line(f"{nuc} usable runs", v, c, vpair, cs0))
        beta, fit_rms, f_post, pred = fit_predict(v, c, y, args.model, vpair, cs0, sthe, sq, dist=pdist, nu=pnu)
        q = np.percentile(pred, PCT)
        rq = np.percentile(LN2 / 10.0 ** pred, PCT)
        psd = float(f_post.std())
        f_med = float(surface_at(beta, v_med, c_med, args.model))       # log10 s, no error terms
        exp = np.log10(config.HL_EXP[nuc]) if nuc in config.HL_EXP else np.nan
        res = exp - q[2] if np.isfinite(exp) else np.nan
        in68 = "yes" if np.isfinite(exp) and q[1] <= exp <= q[3] else ("no" if np.isfinite(exp) else "-")
        in95 = "yes" if np.isfinite(exp) and q[0] <= exp <= q[4] else ("no" if np.isfinite(exp) else "-")
        print(f"{nuc:7s}{len(y):3d}{fit_rms:8.3f}{psd:9.3f}{q_med:8.4f}{sq:7.3f}{q[2]:8.2f}{q[1]:7.2f}{q[3]:7.2f}"
              f"{q[0]:7.2f}{q[4]:7.2f}{exp:7.2f}{res:+7.2f}  {in68:4s} {in95:4s}{f_med:8.2f}")
        rows.append((nuc, len(y), fit_rms, psd, q_med, sq, *q, *rq, exp, res, in68, in95,
                     f_med, float(LN2 / 10.0 ** f_med)))

    if rows:
        print(f"\nrate [1/s] = ln2 / T from the same predictive samples")
        print(f"{'nuc':7s}{'median':>11s}{'16%':>11s}{'84%':>11s}{'2.5%':>11s}{'97.5%':>11s}")
        for r in rows:
            rq = r[11:16]
            print(f"{r[0]:7s}{rq[2]:11.3e}{rq[1]:11.3e}{rq[3]:11.3e}{rq[0]:11.3e}{rq[4]:11.3e}")
    for n in notes:
        print("NOTE " + n)

    tag = "fixture" if args.fixture else f"{F}_batch{BATCH}"
    tag += "_noq" if args.no_q_term else ("_q" if args.fixture else "")
    out = os.path.join(args.out_dir, f"predictions_{tag}.tsv")
    with open(out, "w") as f:
        f.write(f"# trace={args.trace_dir} model={args.model} q_term={'off' if args.no_q_term else 'on'} "
                f"sigma_Q[MeV]={sig_q} pred_dist={ptxt} | fit_rms = sqrt(sum resid^2/(K-n_coef)) [log10 s]; param_sd = sd of "
                f"f over the posterior; sigma_Q_i = 5 sigma_Q/(Q_corr ln10) [log10 s]; p* = percentiles of "
                f"log10(T/s); rate_p* = percentiles of ln2/T [1/s]; log10T_at_median = f(median vpair, median cs0) "
                f"[log10 s] (surface only, no error terms; medians vpair {v_med:.3f} cs0 {c_med:.3f}); "
                f"rate_at_median = ln2/10^log10T_at_median [1/s]\n")
        f.write("nucleus\tK\tfit_rms\tparam_sd\tQ_corr\tsigma_Q_i\tp2.5\tp16\tp50\tp84\tp97.5\t"
                "rate_p2.5\trate_p16\trate_p50\trate_p84\trate_p97.5\tlog10T_exp\tresid\tin68\tin95\t"
                "log10T_at_median\trate_at_median\n")
        for r in rows:
            f.write("\t".join(f"{x:.4e}" if isinstance(x, float) and (11 <= i <= 15 or i == 21) else
                              (f"{x:.4f}" if isinstance(x, float) else str(x))
                              for i, x in enumerate(r)) + "\n")
    print(f"\nwrote {out}")
    if args.fixture:
        print("FIXTURE expectation (recorded 2026-09-10, analysis/frozen/predictions_fixture.tsv, Q term off): "
              "K 25, fit_rms 0.1920, median 4.4504, 16-84% 3.9215-5.0050, 2.5-97.5% 3.2893-5.7394, "
              "exp 4.6567 in68 yes.")


if __name__ == "__main__":
    main()
