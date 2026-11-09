#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import shutil
import argparse
import random as rn
import numpy as np
import tensorflow as tf

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import helper
import config

# Force CPU to avoid CUDA errors on dev nodes
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# -------------------- CLI --------------------
def parse_args():
    p = argparse.ArgumentParser(description="EM2 training (centroids only, computed from strength distributions)")
    p.add_argument("--n",            type=int,   default=9,        help="Model dimension")
    p.add_argument("--n-restarts",   type=int,   default=4,        help="Number of different seeds")
    p.add_argument("--seed0",        type=int,   default=42,       help="Base seed (seeds = seed0..seed0+n_restarts-1)")
    p.add_argument("--num-iter",     type=int,   default=30000,    help="Iterations per restart (must be >= 20000)")
    p.add_argument("--print-every",  type=int,   default=1000,     help="Logging/plotting cadence (iterations)")
    p.add_argument("--data-dir",     type=str,   required=True,
                   help="Directory containing sample_<idx>.dat strength distribution files.")
    p.add_argument("--params-file",  type=str,   default=None,
                   help="Path to params.txt (default: one level up from --data-dir).")
    p.add_argument("--nucleus",      type=str,   default=None,
                   help="Nucleus label (e.g. 'Ca_48') — selects GT window via config.get_gt_window().")
    p.add_argument("--window-lo",    type=float, default=None,
                   help="Override lower edge of centroid window (MeV). Defaults to config window.")
    p.add_argument("--window-hi",    type=float, default=None,
                   help="Override upper edge of centroid window (MeV). Defaults to config window.")
    p.add_argument("--tau",          type=float, default=None,
                   help="Sigmoid edge width (MeV). Defaults to config.GT_WINDOW_TAU.")
    p.add_argument("--shift",        type=float, default=None,
                   help="Energy shift (MeV): E_x = E - shift. Added to the window edges "
                        "so that physical-E_x windows are applied to the mathematical-E "
                        "strength distributions. Defaults to config.get_gt_shift(nucleus, "
                        "functional).")
    p.add_argument("--functional",   type=str,   default=None,
                   help="Functional key for the energy shift (e.g. SKOP, UNE1). "
                        "Default: inferred from the --data-dir/--params-file path; "
                        "ERRORS if it cannot be inferred (never silently SKOP).")
    p.add_argument("--ex-nonneg",    action="store_true",
                   help="Use the E_x >= 0 window: integrate from the daughter g.s. up to "
                        "E_MAX (in the mathematical-E frame: [shift, E_MAX]). Overrides "
                        "--window-lo/--window-hi and the config GT window.")
    p.add_argument("--ao",           action="store_true",
                   help="Use AO-PMM output form (paper Eq. 10) with trainable "
                        "secondary matrices instead of the middle-eigenvalue scalar.")
    p.add_argument("--ao-r",         type=int,   default=3,
                   help="Number of eigenvectors r used in AO-PMM output (default: 3).")
    p.add_argument("--save-dir",     type=str,   required=True,
                   help="Output directory for run artifacts")
    p.add_argument("--plots",        choices=["none", "save"], default="save",
                   help="Training plots: none or save")
    p.add_argument("--val-split",    type=float, default=0.2,
                   help="Fraction of data to use for validation (ignored if --n-folds > 1).")
    p.add_argument("--n-folds",      type=int,   default=1,
                   help="K-fold CV: number of folds. 1 (default) = single --val-split split.")
    p.add_argument("--fold-index",   type=int,   default=0,
                   help="K-fold CV: which fold (0..n_folds-1) is held out as val.")
    return p.parse_args()

def set_all_seeds(seed: int):
    rn.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

# -------------------- scaling (from config.py) --------------------
PARAM_BOUNDS = config.PARAM_BOUNDS
PARAM_NAMES  = ["vpair_t0", "override_cs0"]

def scale_params(raw_vals, param_names, param_bounds):
    """Scale raw parameter values to [-1, 1] using bounds from config."""
    scaled = np.empty_like(raw_vals, dtype=float)
    for j, name in enumerate(param_names):
        lo, hi = param_bounds[name]
        scaled[:, j] = 2.0 * (raw_vals[:, j] - lo) / (hi - lo) - 1.0
    return scaled

def unscale_params(scaled_vals, param_names, param_bounds):
    """Inverse of scale_params: [-1, 1] back to physical values."""
    raw = np.empty_like(scaled_vals, dtype=float)
    for j, name in enumerate(param_names):
        lo, hi = param_bounds[name]
        raw[:, j] = (scaled_vals[:, j] + 1.0) / 2.0 * (hi - lo) + lo
    return raw

# -------------------- centroid from strength distribution --------------------
def centroids_from_spectra(Lors, E_lo, E_hi, tau):
    """
    Compute windowed first moments (centroids, MeV) for a list of strength
    distributions. Each Lors[i] is an (N,2) array of columns [E, S(E)].
    Uses helper.windowed_moments_spectrum_tf with the sigmoid GT window.
    """
    out = np.empty(len(Lors), dtype=float)
    for i, arr in enumerate(Lors):
        x = tf.constant(arr[:, 0], dtype=tf.float64)
        y = tf.constant(arr[:, 1], dtype=tf.float64)
        _S, mu, _var = helper.windowed_moments_spectrum_tf(x, y, E_lo, E_hi, tau)
        out[i] = float(mu.numpy())
    return out

# -------------------- save dataset info --------------------
def save_dataset_info(save_dir, train_set, val_set,
                      train_centroids, val_centroids,
                      train_sample_indices, val_sample_indices,
                      raw_param_values,
                      train_perm, val_perm,
                      param_names):
    """Save both scaled and unscaled train/val sets with headers."""

    header_scaled = "  ".join([f"{name}_scaled" for name in param_names]) + "  centroid_MeV"
    header_unscaled = "sample_index  " + "  ".join(param_names) + "  centroid_MeV"

    train_scaled_arr = np.column_stack([
        np.array(train_set, dtype=float),
        np.array(train_centroids, dtype=float)
    ])
    val_scaled_arr = np.column_stack([
        np.array(val_set, dtype=float),
        np.array(val_centroids, dtype=float)
    ])
    np.savetxt(os.path.join(save_dir, "train_set_scaled.txt"),
               train_scaled_arr, header=header_scaled, fmt='%.12e')
    np.savetxt(os.path.join(save_dir, "val_set_scaled.txt"),
               val_scaled_arr, header=header_scaled, fmt='%.12e')

    train_unscaled_arr = np.column_stack([
        train_sample_indices,
        raw_param_values[train_perm],
        np.array(train_centroids, dtype=float),
    ])
    val_unscaled_arr = np.column_stack([
        val_sample_indices,
        raw_param_values[val_perm],
        np.array(val_centroids, dtype=float),
    ])
    np.savetxt(os.path.join(save_dir, "train_set_unscaled.txt"),
               train_unscaled_arr, header=header_unscaled, fmt='%.12e')
    np.savetxt(os.path.join(save_dir, "val_set_unscaled.txt"),
               val_unscaled_arr, header=header_unscaled, fmt='%.12e')

# -------------------- main --------------------
if __name__ == "__main__":
    args = parse_args()

    n              = args.n
    N_RESTARTS     = args.n_restarts
    SEED0          = args.seed0
    NUM_ITERATIONS = args.num_iter
    PRINT_EVERY    = args.print_every
    DATA_DIR       = os.path.abspath(args.data_dir)
    SAVE_DIR       = args.save_dir
    PLOTS_MODE     = args.plots
    VAL_SPLIT      = args.val_split
    N_FOLDS        = int(args.n_folds)
    FOLD_INDEX     = int(args.fold_index)
    NUCLEUS        = args.nucleus

    # Resolve params file: default is one level up from data-dir (matches main_mod.py)
    if args.params_file is not None:
        PARAMS_FILE = os.path.abspath(args.params_file)
    else:
        PARAMS_FILE = os.path.abspath(os.path.join(DATA_DIR, '..', 'params.txt'))

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Copy config.py to save-dir for reproducibility
    shutil.copy2(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py'),
        os.path.join(SAVE_DIR, 'config.py')
    )

    MIN_ITERATIONS = 20000
    if NUM_ITERATIONS < MIN_ITERATIONS:
        raise ValueError(f"--num-iter ({NUM_ITERATIONS}) must be >= {MIN_ITERATIONS}.")

    # -------------------- resolve energy grid + GT window --------------------
    E_MIN = config.E_MIN
    E_MAX = config.E_MAX

    GT_TAU = float(args.tau) if args.tau is not None else float(getattr(config, "GT_WINDOW_TAU", 0.5))
    # Shift the physical-E_x window into the mathematical-E frame (E_x = E - shift).
    # GT_E_SHIFT is keyed by functional; resolve loudly, never silently SKOP
    # (the 2026-08-04 bug that invalidated the pre-fix UNE1 GT emulators).
    if args.shift is not None:
        FUNCTIONAL = args.functional if args.functional is not None else "(explicit --shift)"
        GT_SHIFT = float(args.shift)
    else:
        FUNCTIONAL = config.resolve_functional(args.functional, args.data_dir, args.params_file)
        GT_SHIFT = float(config.get_gt_shift(NUCLEUS, FUNCTIONAL))

    if args.ex_nonneg:
        # E_x >= 0 up to E_MAX in mathematical-E frame -> E_x in [0, E_MAX - shift]
        GT_WINDOW_EX = (0.0, E_MAX - GT_SHIFT)
        window_mode = "ex_nonneg"
    else:
        _cfg_window = config.get_gt_window(NUCLEUS)
        GT_WINDOW_EX = (
            _cfg_window[0] if args.window_lo is None else float(args.window_lo),
            _cfg_window[1] if args.window_hi is None else float(args.window_hi),
        )
        window_mode = "gt_window"

    GT_WINDOW = (GT_WINDOW_EX[0] + GT_SHIFT, GT_WINDOW_EX[1] + GT_SHIFT)
    print(f"Nucleus: {NUCLEUS}  |  window_mode={window_mode}  |  "
          f"GT window E_x (MeV): {GT_WINDOW_EX}  |  "
          f"functional={FUNCTIONAL}  |  shift={GT_SHIFT}  |  "
          f"GT window E (MeV): {GT_WINDOW}  |  tau={GT_TAU}")

    # -------------------- load params.txt + match sample files --------------------
    print(f"Loading params from: {PARAMS_FILE}")
    print(f"Loading sample files from: {DATA_DIR}")

    sample_indices   = []
    raw_param_values = []
    with open(PARAMS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            idx = int(parts[0])
            vals = [float(x) for x in parts[1:]]
            sample_file = os.path.join(DATA_DIR, f'sample_{idx}.dat')
            if os.path.isfile(sample_file):
                sample_indices.append(idx)
                raw_param_values.append(vals)

    if len(sample_indices) == 0:
        raise ValueError(f"No sample_*.dat files matched entries in {PARAMS_FILE}")

    sample_indices   = np.array(sample_indices,   dtype=int)
    raw_param_values = np.array(raw_param_values, dtype=float)
    print(f"Found {len(sample_indices)} samples with matching data files")

    # Scale parameters to [-1, 1]
    scaled_param_values = scale_params(raw_param_values, PARAM_NAMES, PARAM_BOUNDS)
    full_set = [(float(scaled_param_values[i, 0]), float(scaled_param_values[i, 1]))
                for i in range(len(sample_indices))]

    # -------------------- shuffle and split --------------------
    np.random.seed(SEED0)
    perm = np.random.permutation(len(sample_indices))
    if N_FOLDS > 1:
        if not (0 <= FOLD_INDEX < N_FOLDS):
            raise ValueError(f"--fold-index must be in [0, {N_FOLDS-1}]; got {FOLD_INDEX}")
        folds = np.array_split(perm, N_FOLDS)
        val_perm   = folds[FOLD_INDEX]
        train_perm = np.concatenate([f for k, f in enumerate(folds) if k != FOLD_INDEX])
    else:
        val_size   = int(len(sample_indices) * VAL_SPLIT)
        train_perm = perm[val_size:]
        val_perm   = perm[:val_size]

    train_sample_indices = sample_indices[train_perm]
    train_set = [full_set[i] for i in train_perm]

    val_sample_indices = sample_indices[val_perm]
    val_set = [full_set[i] for i in val_perm]

    print(f"Total points: {len(sample_indices)} | Train: {len(train_set)} | Val: {len(val_set)}")

    # -------------------- load strength spectra + compute centroids --------------------
    print("Loading strength distributions and computing windowed centroids...")
    Lors_train = helper.data_table_from_files(train_sample_indices, DATA_DIR, E_MIN, E_MAX)
    Lors_val   = helper.data_table_from_files(val_sample_indices,   DATA_DIR, E_MIN, E_MAX)

    train_centroids_arr = centroids_from_spectra(Lors_train, GT_WINDOW[0], GT_WINDOW[1], GT_TAU)
    val_centroids_arr   = centroids_from_spectra(Lors_val,   GT_WINDOW[0], GT_WINDOW[1], GT_TAU)

    if np.any(np.isnan(train_centroids_arr)) or np.any(np.isnan(val_centroids_arr)):
        raise ValueError("NaNs encountered while computing centroids — check that spectra have "
                         "nonzero in-window strength.")

    train_centroids = [float(x) for x in train_centroids_arr]
    val_centroids   = [float(x) for x in val_centroids_arr]

    # Save computed centroids for reproducibility / inspection
    np.savetxt(
        os.path.join(SAVE_DIR, "centroids_train.txt"),
        np.column_stack([train_sample_indices, train_centroids_arr]),
        header=f"sample_index  centroid_MeV (in E frame)  # window_E={GT_WINDOW}  "
               f"window_Ex={GT_WINDOW_EX}  shift={GT_SHIFT}  tau={GT_TAU}",
    )
    np.savetxt(
        os.path.join(SAVE_DIR, "centroids_val.txt"),
        np.column_stack([val_sample_indices, val_centroids_arr]),
        header=f"sample_index  centroid_MeV (in E frame)  # window_E={GT_WINDOW}  "
               f"window_Ex={GT_WINDOW_EX}  shift={GT_SHIFT}  tau={GT_TAU}",
    )

    # -------------------- central point (train only) --------------------
    combined_ar   = np.array(train_set, dtype=float)
    centroid_xy   = combined_ar.mean(axis=0)
    distances     = np.linalg.norm(combined_ar - centroid_xy, axis=1)
    central_index = np.argmin(distances)
    central_point = tuple(train_set[central_index])
    print('Central data point in scaled train set:', central_point)

    # -------------------- model initialization --------------------
    np.random.seed(SEED0)
    D, S1, S2 = helper.nec_mat(n)
    USE_AO = bool(args.ao)
    AO_R   = int(args.ao_r)
    if USE_AO:
        nec_num_param = helper.ao_centroid_param_count(n, AO_R)
        if AO_R < 1 or AO_R > n:
            raise ValueError(f"--ao-r must be in [1, n]; got {AO_R} with n={n}")
        print(f'Matrix shapes: {D.shape}, {S1.shape}, {S2.shape}')
        print(f'Output form: AO-PMM (r={AO_R}, {AO_R*(AO_R+1)//2} secondary matrices + g)')
        print(f'Number of parameters: {nec_num_param}')
    else:
        nec_num_param = n + 2 * int(n * (n + 1) / 2)
        print(f'Matrix shapes: {D.shape}, {S1.shape}, {S2.shape}')
        print(f'Output form: middle eigenvalue')
        print(f'Number of parameters: {nec_num_param}')

    def _centroid_cost(params_tf, fmt_set, centroids):
        if USE_AO:
            return helper.cost_function_only_centroid_ao(
                params_tf, n, fmt_set, centroids, central_point, r=AO_R)
        return helper.cost_function_only_centroid(
            params_tf, n, fmt_set, centroids, central_point)

    # -------------------- save dataset info --------------------
    save_dataset_info(SAVE_DIR, train_set, val_set,
                      train_centroids, val_centroids,
                      train_sample_indices, val_sample_indices,
                      raw_param_values,
                      train_perm, val_perm,
                      PARAM_NAMES)

    # -------------------- single restart --------------------
    def run_single_restart(seed: int, run_dir: str):
        set_all_seeds(seed)

        random_initial_guess = np.random.uniform(-1, 1, nec_num_param)
        params = tf.Variable(random_initial_guess, dtype=tf.float64)

        optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)

        @tf.function
        def optimization_step():
            with tf.GradientTape() as tape:
                cost, centroids_calc = _centroid_cost(params, train_set, train_centroids)
            gradients = tape.gradient(cost, [params])
            optimizer.apply_gradients(zip(gradients, [params]))
            return cost, centroids_calc

        @tf.function
        def validation_step():
            val_cost, val_centroids_calc = _centroid_cost(params, val_set, val_centroids)
            return val_cost, val_centroids_calc

        train_cost_loop = []
        val_cost_loop   = []

        val_cost_min      = np.inf
        train_cost_at_min = np.inf
        params_min_val    = None
        min_val_iter      = -1

        for i in range(NUM_ITERATIONS):
            cost, _ = optimization_step()
            val_cost, _ = validation_step()

            cost_val   = float(cost.numpy())
            v_cost_val = float(val_cost.numpy())

            train_cost_loop.append(cost_val)
            val_cost_loop.append(v_cost_val)

            if v_cost_val < val_cost_min:
                val_cost_min      = v_cost_val
                train_cost_at_min = cost_val
                params_min_val    = params.numpy().copy()
                min_val_iter      = i

            if i % PRINT_EVERY == 0:
                try:
                    current_lr = float(optimizer.learning_rate.numpy())
                except Exception:
                    current_lr = 0.01
                print(f"[seed {seed}] Iter {i:6d} | Train={cost_val:.4e} | "
                      f"Val={v_cost_val:.4e} | LR={current_lr:.3e}")

        # Save cost histories
        np.savetxt(os.path.join(run_dir, f"train_cost_history_seed{seed}.txt"),
                   np.array(train_cost_loop))
        np.savetxt(os.path.join(run_dir, f"val_cost_history_seed{seed}.txt"),
                   np.array(val_cost_loop))

        if PLOTS_MODE == "save":
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(range(len(train_cost_loop)), train_cost_loop, label='Train Cost')
            ax.plot(range(len(val_cost_loop)), val_cost_loop, label='Validation Cost', alpha=0.8)
            ax.set_yscale('log')
            ax.set_xlabel('Number of training iterations', size=12)
            ax.set_ylabel('Cost function', size=12)
            ax.set_title(f'Cost history (seed={seed}) | best val={val_cost_min:.3e} @ iter {min_val_iter}')
            ax.legend()
            fig.savefig(os.path.join(run_dir, f"cost_history_seed{seed}.png"),
                        bbox_inches='tight', dpi=150)
            plt.close(fig)

        return val_cost_min, train_cost_at_min, params_min_val, min_val_iter

    # -------------------- multi-restart driver --------------------
    global_best_val_cost   = np.inf
    global_best_train_cost = np.inf
    global_best_params     = None
    global_best_meta       = {"seed": None, "iter": None}

    for r in range(N_RESTARTS):
        seed = SEED0 + 1000 * FOLD_INDEX + r
        run_dir = os.path.join(SAVE_DIR, f"seed_{seed}")
        os.makedirs(run_dir, exist_ok=True)

        print(f"\n========== RESTART {r+1}/{N_RESTARTS} (seed={seed}) ==========")
        print(f"Policy: MIN_ITERATIONS = {MIN_ITERATIONS}, user max = {NUM_ITERATIONS}")

        best_val, best_train, best_params, best_iter = run_single_restart(seed, run_dir)

        print(f"[seed {seed}] Best validation cost in this restart: {best_val:.6e} at iter {best_iter}")

        np.savetxt(os.path.join(run_dir, f"params_n{n}_seed{seed}.txt"), best_params)
        with open(os.path.join(run_dir, "meta.txt"), "w") as f:
            f.write(f"best_val_cost={best_val}\n")
            f.write(f"train_cost_at_best_val={best_train}\n")
            f.write(f"best_iter={best_iter}\n")
            f.write(f"seed={seed}\n")
            f.write(f"n={n}\n")
            f.write(f"data_dir={DATA_DIR}\n")
            f.write(f"params_file={PARAMS_FILE}\n")
            f.write(f"nucleus={NUCLEUS}\n")
            f.write(f"n_folds={N_FOLDS}\n")
            f.write(f"fold_index={FOLD_INDEX}\n")
            f.write(f"window_mode={window_mode}\n")
            f.write(f"gt_window_Ex={GT_WINDOW_EX}\n")
            f.write(f"gt_shift={GT_SHIFT}\n")
            f.write(f"gt_functional={FUNCTIONAL}\n")
            f.write(f"gt_window_E={GT_WINDOW}\n")
            f.write(f"gt_tau={GT_TAU}\n")
            f.write(f"E_MIN={E_MIN}\nE_MAX={E_MAX}\n")
            f.write(f"param_bounds={PARAM_BOUNDS}\n")
            f.write(f"output_form={'ao' if USE_AO else 'middle_eig'}\n")
            if USE_AO:
                f.write(f"ao_r={AO_R}\n")

        if best_val < global_best_val_cost:
            global_best_val_cost   = best_val
            global_best_train_cost = best_train
            global_best_params     = best_params.copy()
            global_best_meta       = {"seed": seed, "iter": best_iter}

    # -------------------- save global best --------------------
    np.savetxt(os.path.join(SAVE_DIR, f'params_best_n{n}_only_centroid.txt'), global_best_params)

    with open(os.path.join(SAVE_DIR, "train_set.txt"), "w") as f:
        for tup in train_set:
            f.write(f"{tup[0]},{tup[1]}\n")

    with open(os.path.join(SAVE_DIR, "val_set.txt"), "w") as f:
        for tup in val_set:
            f.write(f"{tup[0]},{tup[1]}\n")

    print(f'\n*** GLOBAL BEST *** val_cost={global_best_val_cost:.6e} '
          f'(train_cost={global_best_train_cost:.6e}) '
          f'(seed={global_best_meta["seed"]}, iter={global_best_meta["iter"]})')
    print(f'Saved: params_best_n{n}_only_centroid.txt and per-seed artifacts in {SAVE_DIR}/seed_<seed>/')

    # -------------------- parity plot for centroids --------------------
    if PLOTS_MODE == "save" and global_best_params is not None:
        best_params_tf = tf.Variable(global_best_params, dtype=tf.float64)

        _, train_centroid_pred = _centroid_cost(best_params_tf, train_set, train_centroids)
        _, val_centroid_pred   = _centroid_cost(best_params_tf, val_set,   val_centroids)

        train_centroid_pred = np.asarray(train_centroid_pred).ravel()
        val_centroid_pred   = np.asarray(val_centroid_pred).ravel()
        train_centroid_true = np.asarray(train_centroids, dtype=float).ravel()
        val_centroid_true   = np.asarray(val_centroids,   dtype=float).ravel()

        helper.plot_parity(
            train_centroid_pred, train_centroid_true,
            val_centroid_pred,   val_centroid_true,
            quantity="Centroid", units="MeV",
            save_path=os.path.join(SAVE_DIR, "parity_centroid.png"),
            log_scale=False,
        )

        np.savetxt(
            os.path.join(SAVE_DIR, "parity_centroid_train.txt"),
            np.column_stack([train_sample_indices, train_centroid_true, train_centroid_pred]),
            header="sample_index  centroid_true[MeV]  centroid_pred[MeV]",
        )
        np.savetxt(
            os.path.join(SAVE_DIR, "parity_centroid_val.txt"),
            np.column_stack([val_sample_indices, val_centroid_true, val_centroid_pred]),
            header="sample_index  centroid_true[MeV]  centroid_pred[MeV]",
        )
        print(f"Saved parity plot: {os.path.join(SAVE_DIR, 'parity_centroid.png')}")
