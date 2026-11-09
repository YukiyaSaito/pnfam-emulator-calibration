#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Combine K=0 and K=1 full-spectrum CV runs into sigma_em estimates for the
K-summed (deformed-nucleus) windowed-centroid, global peak-location, and
full-window peak-location observables.

For each fold, loads the fold's best K=0 and K=1 emulator parameters, predicts
S_K0(E; theta) and S_K1(E; theta) on each val sample's energy grid, forms the
summed spectrum
    S_full(E) = S_K0(E) + 2 * S_K1(E)        [K=+/-1 degeneracy weight = 2]
computes the windowed centroid, global peak location, and window-constrained
peak location on both predicted and true summed spectra, and pools residuals
across folds to estimate sigma_em on the combined observables.

Assumes K=0 and K=1 CVs were run with the same --seed0 and --n-folds so their
fold i's val sets hold the same sample indices. The script verifies this.

Usage example:
    python combine_full_cv.py \\
        --k0-base    <cv_root>/<F>/GT/Nd150/K0 \\
        --k1-base    <cv_root>/<F>/GT/Nd150/K1 \\
        --data-dir-k0 data/training/<F>/GT/Nd150/K0 \\
        --data-dir-k1 data/training/<F>/GT/Nd150/K1 \\
        --nucleus     Nd_150 \\
        --window-mode gt_window \\
        --out-dir     <cv_root>/<F>/GT/Nd150/full_combined
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force CPU before TF import to match the training scripts.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
import tensorflow as tf

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
import config
import helper


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--k0-base", required=True,
                   help="CV base dir for K=0 (contains fold_0, fold_1, ... subdirs).")
    p.add_argument("--k1-base", required=True,
                   help="CV base dir for K=1.")
    p.add_argument("--data-dir-k0", required=True,
                   help="Raw K=0 strength data dir (sample_<idx>.dat files).")
    p.add_argument("--data-dir-k1", required=True,
                   help="Raw K=1 strength data dir.")
    p.add_argument("--nucleus", required=True,
                   help="Nucleus label (e.g. 'Nd_150') for window/shift lookup.")
    p.add_argument("--functional", default=None,
                   help="Functional key for the energy shift (e.g. SKOP, UNE1). "
                        "Default: inferred from the K-base/data-dir paths; "
                        "ERRORS if it cannot be inferred (never silently SKOP).")
    p.add_argument("--window-mode", choices=["gt_window", "ex_nonneg"], default="gt_window",
                   help="Observable window definition (matches main_mod.py).")
    p.add_argument("--tau", type=float, default=None,
                   help="Sigmoid edge width (MeV). Default: config.GT_WINDOW_TAU.")
    p.add_argument("--out-dir", required=True,
                   help="Output dir for per-fold combined parity + pooled sigma_em summary.")
    p.add_argument("--n-bins", type=int, default=0,
                   help="Quantile bins for heteroscedastic sigma_em (0 disables).")
    p.add_argument("--k1-weight", type=float, default=2.0,
                   help="Degeneracy weight on K=1 when summing (default 2).")
    return p.parse_args()


# --------------------------------------------------------------------------
# Fold I/O helpers
# --------------------------------------------------------------------------
def _scale(raw, name):
    lo, hi = config.PARAM_BOUNDS[name]
    return 2.0 * (raw - lo) / (hi - lo) - 1.0


def find_best_params_file(fold_dir):
    """Locate params_best_n<n>_retain<retain>.txt; return (path, n, retain)."""
    matches = glob.glob(os.path.join(fold_dir, "params_best_n*_retain*.txt"))
    if not matches:
        raise FileNotFoundError(f"No params_best_n*_retain*.txt in {fold_dir}")
    fname = os.path.basename(matches[0])
    stem = fname.replace("params_best_n", "").replace(".txt", "")
    n_str, retain_str = stem.split("_retain")
    return matches[0], int(n_str), float(retain_str)


def central_point_from_train(fold_dir):
    ts = os.path.join(fold_dir, "train_set_scaled.txt")
    arr = np.loadtxt(ts)
    c = arr.mean(axis=0)
    k = int(np.argmin(np.linalg.norm(arr - c, axis=1)))
    return (float(arr[k, 0]), float(arr[k, 1]))


def load_val_set(fold_dir):
    """Return sample_idx[N], alpha_beta_scaled[N, 2] ordered as saved."""
    us = os.path.join(fold_dir, "val_set_unscaled.txt")
    arr = np.loadtxt(us)
    if arr.ndim == 1:
        arr = arr[None, :]
    sample_idx = arr[:, 0].astype(int)
    alpha = _scale(arr[:, 1], "vpair_t0")
    beta  = _scale(arr[:, 2], "override_cs0")
    return sample_idx, np.column_stack([alpha, beta])


# --------------------------------------------------------------------------
# Observable
# --------------------------------------------------------------------------
def windowed_centroid(E, S, E_lo, E_hi, tau):
    x = tf.constant(E, dtype=tf.float64)
    y = tf.constant(S, dtype=tf.float64)
    _S, mu, _var = helper.windowed_moments_spectrum_tf(x, y, E_lo, E_hi, tau)
    return float(mu.numpy())


def window_peak(E, S, E_lo, E_hi):
    mask = (E >= E_lo) & (E <= E_hi)
    if not np.any(mask):
        raise ValueError(
            f"No energy-grid points inside window [{E_lo:.6f}, {E_hi:.6f}]"
        )
    return helper.peak_from_spectrum(E[mask], S[mask])


# --------------------------------------------------------------------------
# Per-fold processing
# --------------------------------------------------------------------------
def process_fold(fold_i, k0_base, k1_base, data_k0, data_k1,
                 e_min, e_max, window_E, tau, k1_weight):
    fd_k0 = os.path.join(k0_base, f"fold_{fold_i}")
    fd_k1 = os.path.join(k1_base, f"fold_{fold_i}")
    if not (os.path.isdir(fd_k0) and os.path.isdir(fd_k1)):
        return None

    idx_k0, _ab_k0 = load_val_set(fd_k0)
    idx_k1, _ab_k1 = load_val_set(fd_k1)
    if not np.array_equal(np.sort(idx_k0), np.sort(idx_k1)):
        raise ValueError(
            f"Fold {fold_i}: K=0 and K=1 val-set sample indices differ.\n"
            f"  K=0 sorted: {np.sort(idx_k0)}\n"
            f"  K=1 sorted: {np.sort(idx_k1)}\n"
            "Rerun both CVs with the same --seed0 and --n-folds."
        )

    # Align on sample_index (sorted order).
    order_k0 = np.argsort(idx_k0)
    sample_idx = idx_k0[order_k0]
    ab = load_val_set(fd_k0)[1][order_k0]

    pf_k0, n_k0, retain_k0 = find_best_params_file(fd_k0)
    pf_k1, n_k1, retain_k1 = find_best_params_file(fd_k1)
    params_k0 = tf.Variable(np.loadtxt(pf_k0), dtype=tf.float64)
    params_k1 = tf.Variable(np.loadtxt(pf_k1), dtype=tf.float64)

    cp_k0 = central_point_from_train(fd_k0)
    cp_k1 = central_point_from_train(fd_k1)

    Lors_k0 = helper.data_table_from_files(sample_idx, data_k0, e_min, e_max)
    Lors_k1 = helper.data_table_from_files(sample_idx, data_k1, e_min, e_max)

    E_lo, E_hi = window_E
    mu_true_arr, mu_pred_arr = [], []
    peak_true_arr, peak_pred_arr = [], []
    win_peak_true_arr, win_peak_pred_arr = [], []
    for i in range(len(sample_idx)):
        E_k0, y_k0_true = Lors_k0[i][:, 0], Lors_k0[i][:, 1]
        E_k1, y_k1_true = Lors_k1[i][:, 0], Lors_k1[i][:, 1]
        if not np.allclose(E_k0, E_k1):
            raise ValueError(
                f"Sample {sample_idx[i]}: K=0 and K=1 energy grids differ; cannot sum."
            )
        E = E_k0
        S_true = y_k0_true + k1_weight * y_k1_true

        _ek0, _Bk0, S_k0_pred = helper.predict_strength_spectrum(
            params_k0, n_k0, ab[i], E, cp_k0, retain_k0)
        _ek1, _Bk1, S_k1_pred = helper.predict_strength_spectrum(
            params_k1, n_k1, ab[i], E, cp_k1, retain_k1)
        S_pred = S_k0_pred + k1_weight * S_k1_pred

        mu_true_arr.append(windowed_centroid(E, S_true, E_lo, E_hi, tau))
        mu_pred_arr.append(windowed_centroid(E, S_pred, E_lo, E_hi, tau))
        peak_true_arr.append(helper.peak_from_spectrum(E, S_true))
        peak_pred_arr.append(helper.peak_from_spectrum(E, S_pred))
        win_peak_true_arr.append(window_peak(E, S_true, E_lo, E_hi))
        win_peak_pred_arr.append(window_peak(E, S_pred, E_lo, E_hi))

    return (sample_idx,
            np.asarray(mu_true_arr), np.asarray(mu_pred_arr),
            np.asarray(peak_true_arr), np.asarray(peak_pred_arr),
            np.asarray(win_peak_true_arr), np.asarray(win_peak_pred_arr))


# --------------------------------------------------------------------------
# Pooling
# --------------------------------------------------------------------------
def binned_sigma(true_all, resid, n_bins):
    if n_bins <= 0 or len(true_all) < n_bins * 2:
        return []
    edges = np.quantile(true_all, np.linspace(0.0, 1.0, n_bins + 1))
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        mask = (true_all >= lo) & (true_all <= hi if b == n_bins - 1 else true_all < hi)
        if mask.sum() < 2:
            continue
        r_b = resid[mask]
        mean_b = float(r_b.mean())
        rows.append((b, float(lo), float(hi), int(mask.sum()), mean_b,
                     float(np.sqrt(np.mean((r_b - mean_b) ** 2)))))
    return rows


def write_sigma_summary(out_summary, *, title, section, args, ex_lo, ex_hi,
                        window_E, shift, tau, per_fold, true_all, pred_all,
                        n_bins):
    resid = pred_all - true_all
    bias = float(np.mean(resid))
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    sigma = float(np.sqrt(np.mean((resid - bias) ** 2)))
    bin_rows = binned_sigma(true_all, resid, n_bins)

    with open(out_summary, "w") as f:
        f.write(f"# {title}\n")
        f.write(f"# k0_base:  {args.k0_base}\n")
        f.write(f"# k1_base:  {args.k1_base}\n")
        f.write(f"# nucleus:  {args.nucleus}\n")
        f.write(f"# window_mode: {args.window_mode}\n")
        f.write(f"# Ex_window:    [{ex_lo:.6f}, {ex_hi:.6f}]\n")
        f.write(f"# math_window:  [{window_E[0]:.6f}, {window_E[1]:.6f}]\n")
        f.write(f"# shift:    {shift:.6f}\n")
        f.write(f"# tau:      {tau}\n")
        f.write(f"# k1_weight: {args.k1_weight}\n")
        f.write(f"# folds:    {len(per_fold)}\n\n")
        f.write(f"[{section}]\n")
        f.write(f"N={len(resid)}\n")
        f.write(f"bias={bias:.6e}\n")
        f.write(f"rmse={rmse:.6e}\n")
        f.write(f"sigma_em={sigma:.6e}\n")
        f.write("# per-fold (fold, N, bias, rmse):\n")
        for fi, N_, m_, r_ in per_fold:
            f.write(f"  fold_{fi}  N={N_}  bias={m_:.6e}  rmse={r_:.6e}\n")
        if bin_rows:
            f.write("# heteroscedastic (bin, true_lo, true_hi, N, bias, sigma):\n")
            for b, lo, hi, N, m, s in bin_rows:
                f.write(f"  bin={b}  [{lo:.6e}, {hi:.6e}]  N={N}  bias={m:.6e}  sigma={s:.6e}\n")

    return bias, rmse, sigma


def plot_val_parity(pred, true, *, quantity, units, save_path):
    pred = np.asarray(pred, dtype=float)
    true = np.asarray(true, dtype=float)
    finite = np.isfinite(pred) & np.isfinite(true)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111)
    if np.any(finite):
        ax.scatter(pred[finite], true[finite], color="#ff7f0e", marker="s",
                   s=60, alpha=0.9, edgecolor="k", label="Validation", zorder=5)
        vals = np.concatenate([pred[finite], true[finite]])
        span = float(np.max(vals) - np.min(vals))
        pad = 0.05 * span if span > 0.0 else 1.0
        lo = float(np.min(vals)) - pad
        hi = float(np.max(vals)) + pad
        resid = pred[finite] - true[finite]
        bias = float(np.mean(resid))
        rmse = float(np.sqrt(np.mean(resid ** 2)))
        ax.text(0.05, 0.95, f"N={finite.sum()}\nbias={bias:+.3e}\nRMSE={rmse:.3e}",
                transform=ax.transAxes, va="top", ha="left",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))
    else:
        lo, hi = 0.0, 1.0
        ax.text(0.5, 0.5, f"No finite {quantity} values",
                transform=ax.transAxes, ha="center", va="center")

    ax.plot([lo, hi], [lo, hi], "r--", label="Ideal", zorder=1)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(f"Emulated {quantity} ({units})")
    ax.set_ylabel(f"True QRPA {quantity} ({units})")
    ax.set_title(f"K-summed parity - {quantity}")
    ax.grid(True, alpha=0.4)
    ax.legend()
    fig.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    E_MIN = config.E_MIN
    E_MAX = config.E_MAX
    tau = args.tau if args.tau is not None else float(getattr(config, "GT_WINDOW_TAU", 0.5))
    functional = config.resolve_functional(
        args.functional, args.data_dir_k0, args.data_dir_k1,
        args.k0_base, args.k1_base)
    shift = float(config.get_gt_shift(args.nucleus, functional))
    print(f"nucleus={args.nucleus}  functional={functional}  shift={shift}")

    if args.window_mode == "ex_nonneg":
        ex_lo, ex_hi = 0.0, E_MAX - shift
    else:
        ex_lo, ex_hi = config.get_gt_window(args.nucleus)
    window_E = (ex_lo + shift, ex_hi + shift)

    print(f"Nucleus: {args.nucleus}  window_mode={args.window_mode}")
    print(f"  Ex window: [{ex_lo:.3f}, {ex_hi:.3f}]   shift: {shift:.3f}")
    print(f"  math-E window: [{window_E[0]:.3f}, {window_E[1]:.3f}]   tau: {tau}")
    print(f"  K=1 weight: {args.k1_weight}")

    k0_fold_dirs = sorted(glob.glob(os.path.join(args.k0_base, "fold_*")))
    if not k0_fold_dirs:
        raise SystemExit(f"No fold_* under {args.k0_base}")
    K = len(k0_fold_dirs)
    print(f"Detected {K} folds under {args.k0_base}")

    all_mu_true, all_mu_pred, per_fold_mu = [], [], []
    all_peak_true, all_peak_pred, per_fold_peak = [], [], []
    all_win_peak_true, all_win_peak_pred, per_fold_win_peak = [], [], []
    for fi in range(K):
        print(f"Fold {fi}...")
        res = process_fold(fi, args.k0_base, args.k1_base,
                           args.data_dir_k0, args.data_dir_k1,
                           E_MIN, E_MAX, window_E, tau, args.k1_weight)
        if res is None:
            print(f"  (missing K0 or K1 fold_{fi} dir; skipping)")
            continue
        (sample_idx, mu_true, mu_pred, peak_true, peak_pred,
         win_peak_true, win_peak_pred) = res
        out_parity_dir = os.path.join(args.out_dir, f"fold_{fi}")
        os.makedirs(out_parity_dir, exist_ok=True)
        np.savetxt(
            os.path.join(out_parity_dir, "parity_full_centroid_val.txt"),
            np.column_stack([sample_idx, mu_true, mu_pred]),
            header="sample_index  centroid_true[MeV]  centroid_pred[MeV]",
        )
        np.savetxt(
            os.path.join(out_parity_dir, "parity_full_peak_val.txt"),
            np.column_stack([sample_idx, peak_true, peak_pred]),
            header="sample_index  peak_true[MeV]  peak_pred[MeV]",
        )
        np.savetxt(
            os.path.join(out_parity_dir, "parity_full_window_peak_val.txt"),
            np.column_stack([sample_idx, win_peak_true, win_peak_pred]),
            header="sample_index  window_peak_true[MeV]  window_peak_pred[MeV]",
        )

        r_mu = mu_pred - mu_true
        per_fold_mu.append((fi, int(len(r_mu)), float(np.mean(r_mu)),
                            float(np.sqrt(np.mean(r_mu ** 2)))))
        r_peak = peak_pred - peak_true
        per_fold_peak.append((fi, int(len(r_peak)), float(np.mean(r_peak)),
                              float(np.sqrt(np.mean(r_peak ** 2)))))
        r_win_peak = win_peak_pred - win_peak_true
        per_fold_win_peak.append((fi, int(len(r_win_peak)), float(np.mean(r_win_peak)),
                                  float(np.sqrt(np.mean(r_win_peak ** 2)))))

        all_mu_true.append(mu_true); all_mu_pred.append(mu_pred)
        all_peak_true.append(peak_true); all_peak_pred.append(peak_pred)
        all_win_peak_true.append(win_peak_true); all_win_peak_pred.append(win_peak_pred)

    if not per_fold_mu:
        raise SystemExit("No folds successfully processed.")

    all_mu_true = np.concatenate(all_mu_true)
    all_mu_pred = np.concatenate(all_mu_pred)
    all_peak_true = np.concatenate(all_peak_true)
    all_peak_pred = np.concatenate(all_peak_pred)
    all_win_peak_true = np.concatenate(all_win_peak_true)
    all_win_peak_pred = np.concatenate(all_win_peak_pred)

    out_summary = os.path.join(args.out_dir, "sigma_em_full_centroid.txt")
    bias, rmse, sigma = write_sigma_summary(
        out_summary,
        title="Combined K=0 + w*K=1 full-spectrum windowed-centroid sigma_em",
        section="full_centroid",
        args=args, ex_lo=ex_lo, ex_hi=ex_hi, window_E=window_E,
        shift=shift, tau=tau, per_fold=per_fold_mu,
        true_all=all_mu_true, pred_all=all_mu_pred, n_bins=args.n_bins)

    print(f"\nWrote {out_summary}")
    print(f"centroid pooled: N={len(all_mu_true)}  bias={bias:+.4e}  "
          f"rmse={rmse:.4e}  sigma_em={sigma:.4e}")
    plot_val_parity(
        all_mu_pred, all_mu_true,
        quantity="full windowed centroid", units="MeV",
        save_path=os.path.join(args.out_dir, "parity_full_centroid_val.png"))

    out_peak_summary = os.path.join(args.out_dir, "sigma_em_full_peak.txt")
    peak_bias, peak_rmse, peak_sigma = write_sigma_summary(
        out_peak_summary,
        title="Combined K=0 + w*K=1 full-spectrum peak-location sigma_em",
        section="full_peak",
        args=args, ex_lo=ex_lo, ex_hi=ex_hi, window_E=window_E,
        shift=shift, tau=tau, per_fold=per_fold_peak,
        true_all=all_peak_true, pred_all=all_peak_pred, n_bins=args.n_bins)

    print(f"Wrote {out_peak_summary}")
    print(f"peak pooled:     N={len(all_peak_true)}  bias={peak_bias:+.4e}  "
          f"rmse={peak_rmse:.4e}  sigma_em={peak_sigma:.4e}")
    plot_val_parity(
        all_peak_pred, all_peak_true,
        quantity="full peak location", units="MeV",
        save_path=os.path.join(args.out_dir, "parity_full_peak_val.png"))

    out_win_peak_summary = os.path.join(args.out_dir, "sigma_em_full_window_peak.txt")
    win_peak_bias, win_peak_rmse, win_peak_sigma = write_sigma_summary(
        out_win_peak_summary,
        title="Combined K=0 + w*K=1 full-spectrum windowed-peak-location sigma_em",
        section="full_window_peak",
        args=args, ex_lo=ex_lo, ex_hi=ex_hi, window_E=window_E,
        shift=shift, tau=tau, per_fold=per_fold_win_peak,
        true_all=all_win_peak_true, pred_all=all_win_peak_pred,
        n_bins=args.n_bins)

    print(f"Wrote {out_win_peak_summary}")
    print(f"window peak pooled: N={len(all_win_peak_true)}  "
          f"bias={win_peak_bias:+.4e}  rmse={win_peak_rmse:.4e}  "
          f"sigma_em={win_peak_sigma:.4e}")
    plot_val_parity(
        all_win_peak_pred, all_win_peak_true,
        quantity="full window peak location", units="MeV",
        save_path=os.path.join(args.out_dir, "parity_full_window_peak_val.png"))
    print("Wrote pooled parity plots: parity_full_centroid_val.png, "
          "parity_full_peak_val.png, parity_full_window_peak_val.png")


if __name__ == "__main__":
    main()
