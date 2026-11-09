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
    p = argparse.ArgumentParser(description="EM2 training (half-lives only)")
    p.add_argument("--n",            type=int,   default=9,        help="Model dimension")
    p.add_argument("--n-restarts",   type=int,   default=4,        help="Number of different seeds")
    p.add_argument("--seed0",        type=int,   default=42,       help="Base seed (seeds = seed0..seed0+n_restarts-1)")
    p.add_argument("--num-iter",     type=int,   default=30000,    help="Iterations per restart (must be >= 20000)")
    p.add_argument("--print-every",  type=int,   default=1000,     help="Logging/plotting cadence (iterations)")
    p.add_argument("--hl-file",      type=str,   required=True,
                   help="Path to halflives.txt (columns: sample_index, total_hl)")
    p.add_argument("--params-file",  type=str,   default=None,
                   help="Path to params.txt (default: same directory as --hl-file)")
    p.add_argument("--save-dir",     type=str,   required=True,
                   help="Output directory for run artifacts")
    p.add_argument("--architecture", choices=["ae", "ao", "exp", "exp_gen", "quad", "wsel"], default="ae",
                   help="HL emulator architecture: ae = middle eigenvalue (default), "
                        "ao = AO-PMM bilinear eigenvector form, "
                        "exp = exponential-PMM (Jin et al. PRR 7 043347 2025, Eq. 22), "
                        "exp_gen = generalised symmetric exponential-PMM "
                        "(two extra matrices with linear-in-(da,db) exponents), "
                        "quad = second-order PMM (M linear + quadratic in (da, db)), "
                        "wsel = weighted-selector PMM (AE matrix + softmax-weighted "
                        "region-dependent eigenvalue combination).")
    p.add_argument("--ao-r",         type=int,   default=3,
                   help="Number of central eigenvectors for --architecture ao.")
    p.add_argument("--plots",        choices=["none", "save"], default="save",
                   help="Training plots: none or save")
    p.add_argument("--val-split",    type=float, default=0.2,
                   help="Fraction of data to use for validation (ignored if --n-folds > 1).")
    p.add_argument("--n-folds",      type=int,   default=1,
                   help="K-fold CV: number of folds. 1 (default) = single --val-split split.")
    p.add_argument("--fold-index",   type=int,   default=0,
                   help="K-fold CV: which fold (0..n_folds-1) is held out as val.")
    p.add_argument("--stratified",   action="store_true", default=False,
                   help="Stratify train/val splits by log10(half-life) so each fold "
                        "covers the full HL range. Recommended for K-fold CV.")
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

# -------------------- save dataset info --------------------
def save_dataset_info(save_dir, train_set, val_set,
                      train_HLs, val_HLs,
                      train_sample_indices, val_sample_indices,
                      raw_param_values, raw_hls, log_hls,
                      train_perm, val_perm,
                      param_names):
    """Save both scaled and unscaled train/val sets with headers."""

    header_scaled = "  ".join([f"{name}_scaled" for name in param_names]) + "  log10_hl"
    header_unscaled = "sample_index  " + "  ".join(param_names) + "  half_life  log10_hl"

    # Scaled (with log10 HL)
    train_scaled_arr = np.column_stack([
        np.array(train_set, dtype=float),
        np.array(train_HLs, dtype=float)
    ])
    val_scaled_arr = np.column_stack([
        np.array(val_set, dtype=float),
        np.array(val_HLs, dtype=float)
    ])
    np.savetxt(os.path.join(save_dir, "train_set_scaled.txt"),
               train_scaled_arr, header=header_scaled, fmt='%.12e')
    np.savetxt(os.path.join(save_dir, "val_set_scaled.txt"),
               val_scaled_arr, header=header_scaled, fmt='%.12e')

    # Unscaled (with sample indices, raw HL, and log10 HL)
    train_unscaled_arr = np.column_stack([
        train_sample_indices,
        raw_param_values[train_perm],
        raw_hls[train_perm],
        log_hls[train_perm]
    ])
    val_unscaled_arr = np.column_stack([
        val_sample_indices,
        raw_param_values[val_perm],
        raw_hls[val_perm],
        log_hls[val_perm]
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
    HL_FILE        = os.path.abspath(args.hl_file)
    SAVE_DIR       = args.save_dir
    PLOTS_MODE     = args.plots
    VAL_SPLIT      = args.val_split
    N_FOLDS        = int(args.n_folds)
    FOLD_INDEX     = int(args.fold_index)
    ARCHITECTURE   = args.architecture
    AO_R           = int(args.ao_r)

    if AO_R < 1:
        raise ValueError(f"--ao-r must be >= 1; got {AO_R}")
    if ARCHITECTURE == "ao" and AO_R > n:
        raise ValueError(f"--ao-r ({AO_R}) must be <= --n ({n})")

    # Resolve params file: default is same directory as hl-file
    if args.params_file is not None:
        PARAMS_FILE = os.path.abspath(args.params_file)
    else:
        PARAMS_FILE = os.path.abspath(
            os.path.join(os.path.dirname(HL_FILE), 'params.txt')
        )

    os.makedirs(SAVE_DIR, exist_ok=True)

    # Copy config.py to save-dir for reproducibility
    shutil.copy2(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.py'),
        os.path.join(SAVE_DIR, 'config.py')
    )

    MIN_ITERATIONS = 20000
    if NUM_ITERATIONS < MIN_ITERATIONS:
        raise ValueError(f"--num-iter ({NUM_ITERATIONS}) must be >= {MIN_ITERATIONS}.")

    # -------------------- load and join data --------------------
    print(f"Loading params from: {PARAMS_FILE}")
    print(f"Loading half-lives from: {HL_FILE}")

    # Parse params.txt -> dict: sample_index -> [vpair_t0, override_cs0]
    params_dict = {}
    with open(PARAMS_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            idx = int(parts[0])
            vals = [float(x) for x in parts[1:]]
            params_dict[idx] = vals

    # Parse halflives.txt -> dict: sample_index -> half-life
    hl_dict = {}
    with open(HL_FILE, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            idx = int(parts[0])
            hl_val = float(parts[1])
            hl_dict[idx] = hl_val

    # Join on sample index: only keep samples present in BOTH files
    common_indices = sorted(set(params_dict.keys()) & set(hl_dict.keys()))
    if len(common_indices) == 0:
        raise ValueError("No matching sample indices between params.txt and halflives.txt!")

    n_params_only = len(set(params_dict.keys()) - set(hl_dict.keys()))
    n_hl_only     = len(set(hl_dict.keys()) - set(params_dict.keys()))
    if n_params_only > 0 or n_hl_only > 0:
        print(f"WARNING: {n_params_only} samples in params.txt without HL data, "
              f"{n_hl_only} samples in halflives.txt without param data — these are excluded.")

    print(f"Found {len(params_dict)} param entries, {len(hl_dict)} HL entries, "
          f"{len(common_indices)} matched samples")

    sample_indices   = np.array(common_indices, dtype=int)
    raw_param_values = np.array([params_dict[i] for i in common_indices], dtype=float)
    raw_hls          = np.array([hl_dict[i] for i in common_indices], dtype=float)

    # Compute log10(half-lives) and validate
    log_hls = np.log10(raw_hls)
    if np.any(np.isnan(log_hls)) or np.any(np.isinf(log_hls)):
        bad_mask = np.isnan(log_hls) | np.isinf(log_hls)
        bad_indices = sample_indices[bad_mask]
        raise ValueError(f"NaNs or Infs in log10(half-lives) for sample indices: {bad_indices}")

    HLs = [float(x) for x in log_hls]

    # Scale parameters to [-1, 1]
    scaled_param_values = scale_params(raw_param_values, PARAM_NAMES, PARAM_BOUNDS)
    full_set = [(float(scaled_param_values[i, 0]), float(scaled_param_values[i, 1]))
                for i in range(len(sample_indices))]

    # -------------------- shuffle and split --------------------
    np.random.seed(SEED0)

    if args.stratified:
        print("Using stratified splitting (by log10 half-life)")

        sort_order = np.argsort(log_hls)  # indices sorted by ascending log10(HL)

        if N_FOLDS > 1:
            if not (0 <= FOLD_INDEX < N_FOLDS):
                raise ValueError(f"--fold-index must be in [0, {N_FOLDS-1}]; got {FOLD_INDEX}")

            # Assign folds via sorted-interleave with intra-stripe shuffle
            fold_assignments = np.empty(len(sample_indices), dtype=int)
            n_samples = len(sample_indices)
            n_full_stripes = n_samples // N_FOLDS
            remainder = n_samples % N_FOLDS

            for s in range(n_full_stripes):
                stripe = sort_order[s * N_FOLDS : (s + 1) * N_FOLDS]
                fold_ids = np.arange(N_FOLDS)
                np.random.shuffle(fold_ids)
                for local_j, global_idx in enumerate(stripe):
                    fold_assignments[global_idx] = fold_ids[local_j]

            if remainder > 0:
                stripe = sort_order[n_full_stripes * N_FOLDS:]
                fold_ids = np.arange(remainder)
                np.random.shuffle(fold_ids)
                for local_j, global_idx in enumerate(stripe):
                    fold_assignments[global_idx] = fold_ids[local_j]

            val_perm   = np.where(fold_assignments == FOLD_INDEX)[0]
            train_perm = np.where(fold_assignments != FOLD_INDEX)[0]

        else:
            # Single split: walk sorted order, flip biased coin
            val_flags = np.zeros(len(sample_indices), dtype=bool)
            for i in range(len(sort_order)):
                if np.random.rand() < VAL_SPLIT:
                    val_flags[sort_order[i]] = True
            val_perm   = np.where(val_flags)[0]
            train_perm = np.where(~val_flags)[0]

        # Shuffle within each set so iteration order isn't monotonic in HL
        np.random.shuffle(train_perm)
        np.random.shuffle(val_perm)

    else:
        # Original behavior: plain random permutation
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
    train_HLs = [HLs[i] for i in train_perm]

    val_sample_indices = sample_indices[val_perm]
    val_set = [full_set[i] for i in val_perm]
    val_HLs = [HLs[i] for i in val_perm]

    print(f"Total points: {len(sample_indices)} | Train: {len(train_set)} | Val: {len(val_set)}")

    # -------------------- central point (train only) --------------------
    combined_ar   = np.array(train_set, dtype=float)
    centroid      = combined_ar.mean(axis=0)
    distances     = np.linalg.norm(combined_ar - centroid, axis=1)
    central_index = np.argmin(distances)
    central_point = tuple(train_set[central_index])
    print('Central data point in scaled train set:', central_point)

    # -------------------- model initialization --------------------
    np.random.seed(SEED0)
    D, S1, S2 = helper.nec_mat(n)
    if ARCHITECTURE == "ao":
        nec_num_param = helper.ao_pmm_param_count(n, AO_R)
        param_stem = f"params_best_ao_n{n}_r{AO_R}_only_HL"
        seed_param_template = f"params_ao_n{n}_r{AO_R}_seed{{seed}}.txt"
    elif ARCHITECTURE == "exp":
        nec_num_param = helper.exp_pmm_param_count(n)
        param_stem = f"params_best_exp_n{n}_only_HL"
        seed_param_template = f"params_exp_n{n}_seed{{seed}}.txt"
    elif ARCHITECTURE == "exp_gen":
        nec_num_param = helper.exp_gen_pmm_param_count(n)
        param_stem = f"params_best_exp_gen_n{n}_only_HL"
        seed_param_template = f"params_exp_gen_n{n}_seed{{seed}}.txt"
    elif ARCHITECTURE == "quad":
        nec_num_param = helper.quad_pmm_param_count(n)
        param_stem = f"params_best_quad_n{n}_only_HL"
        seed_param_template = f"params_quad_n{n}_seed{{seed}}.txt"
    elif ARCHITECTURE == "wsel":
        nec_num_param = helper.wsel_pmm_param_count(n)
        param_stem = f"params_best_wsel_n{n}_only_HL"
        seed_param_template = f"params_wsel_n{n}_seed{{seed}}.txt"
    else:
        nec_num_param = n + 2 * int(n * (n + 1) / 2)
        param_stem = f"params_best_n{n}_only_HL"
        seed_param_template = f"params_n{n}_seed{{seed}}.txt"
    print(f'Matrix shapes: {D.shape}, {S1.shape}, {S2.shape}')
    print(f'Architecture: {ARCHITECTURE}')
    if ARCHITECTURE == "ao":
        print(f'AO rank/window r: {AO_R}')
    print(f'Number of parameters: {nec_num_param}')

    # -------------------- save dataset info --------------------
    save_dataset_info(SAVE_DIR, train_set, val_set,
                      train_HLs, val_HLs,
                      train_sample_indices, val_sample_indices,
                      raw_param_values, raw_hls, log_hls,
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
                if ARCHITECTURE == "ao":
                    cost, HLs_calc = helper.cost_function_only_HL_ao(
                        params, n, train_set, train_HLs, central_point, r=AO_R)
                elif ARCHITECTURE == "exp":
                    cost, HLs_calc = helper.cost_function_only_HL_exp(
                        params, n, train_set, train_HLs, central_point)
                elif ARCHITECTURE == "exp_gen":
                    cost, HLs_calc = helper.cost_function_only_HL_exp_gen(
                        params, n, train_set, train_HLs, central_point)
                elif ARCHITECTURE == "quad":
                    cost, HLs_calc = helper.cost_function_only_HL_quad(
                        params, n, train_set, train_HLs, central_point)
                elif ARCHITECTURE == "wsel":
                    cost, HLs_calc = helper.cost_function_only_HL_wsel(
                        params, n, train_set, train_HLs, central_point)
                else:
                    cost, HLs_calc = helper.cost_function_only_HL(
                        params, n, train_set, train_HLs, central_point)
            gradients = tape.gradient(cost, [params])
            optimizer.apply_gradients(zip(gradients, [params]))
            return cost, HLs_calc

        @tf.function
        def validation_step():
            if ARCHITECTURE == "ao":
                val_cost, val_HLs_calc = helper.cost_function_only_HL_ao(
                    params, n, val_set, val_HLs, central_point, r=AO_R)
            elif ARCHITECTURE == "exp":
                val_cost, val_HLs_calc = helper.cost_function_only_HL_exp(
                    params, n, val_set, val_HLs, central_point)
            elif ARCHITECTURE == "exp_gen":
                val_cost, val_HLs_calc = helper.cost_function_only_HL_exp_gen(
                    params, n, val_set, val_HLs, central_point)
            elif ARCHITECTURE == "quad":
                val_cost, val_HLs_calc = helper.cost_function_only_HL_quad(
                    params, n, val_set, val_HLs, central_point)
            elif ARCHITECTURE == "wsel":
                val_cost, val_HLs_calc = helper.cost_function_only_HL_wsel(
                    params, n, val_set, val_HLs, central_point)
            else:
                val_cost, val_HLs_calc = helper.cost_function_only_HL(
                    params, n, val_set, val_HLs, central_point)
            return val_cost, val_HLs_calc

        train_cost_loop = []
        val_cost_loop   = []

        val_cost_min      = np.inf
        train_cost_at_min = np.inf
        params_min_val    = None
        min_val_iter      = -1

        for i in range(NUM_ITERATIONS):
            cost, HLs_train = optimization_step()
            val_cost, HLs_val = validation_step()

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

        # Plot cost curves
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

        # Save per-restart artifacts
        np.savetxt(os.path.join(run_dir, seed_param_template.format(seed=seed)), best_params)
        with open(os.path.join(run_dir, "meta.txt"), "w") as f:
            f.write(f"best_val_cost={best_val}\n")
            f.write(f"train_cost_at_best_val={best_train}\n")
            f.write(f"best_iter={best_iter}\n")
            f.write(f"seed={seed}\n")
            f.write(f"n={n}\n")
            f.write(f"architecture={ARCHITECTURE}\n")
            f.write(f"ao_r={AO_R}\n")
            f.write(f"num_params={nec_num_param}\n")
            f.write(f"hl_file={HL_FILE}\n")
            f.write(f"params_file={PARAMS_FILE}\n")
            f.write(f"param_bounds={PARAM_BOUNDS}\n")
            f.write(f"n_folds={N_FOLDS}\n")
            f.write(f"fold_index={FOLD_INDEX}\n")
            f.write(f"stratified={args.stratified}\n")

        if best_val < global_best_val_cost:
            global_best_val_cost   = best_val
            global_best_train_cost = best_train
            global_best_params     = best_params.copy()
            global_best_meta       = {"seed": seed, "iter": best_iter}

    # -------------------- save global best --------------------
    np.savetxt(os.path.join(SAVE_DIR, f'{param_stem}.txt'), global_best_params)

    # Save train/val sets (both scaled and unscaled) — already done above,
    # but also save the simple comma-separated format for backward compat
    with open(os.path.join(SAVE_DIR, "train_set.txt"), "w") as f:
        for tup in train_set:
            f.write(f"{tup[0]},{tup[1]}\n")

    with open(os.path.join(SAVE_DIR, "val_set.txt"), "w") as f:
        for tup in val_set:
            f.write(f"{tup[0]},{tup[1]}\n")

    print(f'\n*** GLOBAL BEST *** val_cost={global_best_val_cost:.6e} '
          f'(train_cost={global_best_train_cost:.6e}) '
          f'(seed={global_best_meta["seed"]}, iter={global_best_meta["iter"]})')
    print(f'Saved: {param_stem}.txt and per-seed artifacts in {SAVE_DIR}/seed_<seed>/')

    # -------------------- parity plot for half-lives --------------------
    if PLOTS_MODE == "save" and global_best_params is not None:
        best_params_tf = tf.Variable(global_best_params, dtype=tf.float64)

        # The selected cost function exponentiates internally, so its second return is
        # HL in seconds, not log10. train_HLs / val_HLs are log10(HL).
        if ARCHITECTURE == "ao":
            _, train_HL_pred = helper.cost_function_only_HL_ao(
                best_params_tf, n, train_set, train_HLs, central_point, r=AO_R)
            _, val_HL_pred = helper.cost_function_only_HL_ao(
                best_params_tf, n, val_set, val_HLs, central_point, r=AO_R)
        elif ARCHITECTURE == "exp":
            _, train_HL_pred = helper.cost_function_only_HL_exp(
                best_params_tf, n, train_set, train_HLs, central_point)
            _, val_HL_pred = helper.cost_function_only_HL_exp(
                best_params_tf, n, val_set, val_HLs, central_point)
        elif ARCHITECTURE == "exp_gen":
            _, train_HL_pred = helper.cost_function_only_HL_exp_gen(
                best_params_tf, n, train_set, train_HLs, central_point)
            _, val_HL_pred = helper.cost_function_only_HL_exp_gen(
                best_params_tf, n, val_set, val_HLs, central_point)
        elif ARCHITECTURE == "quad":
            _, train_HL_pred = helper.cost_function_only_HL_quad(
                best_params_tf, n, train_set, train_HLs, central_point)
            _, val_HL_pred = helper.cost_function_only_HL_quad(
                best_params_tf, n, val_set, val_HLs, central_point)
        elif ARCHITECTURE == "wsel":
            _, train_HL_pred = helper.cost_function_only_HL_wsel(
                best_params_tf, n, train_set, train_HLs, central_point)
            _, val_HL_pred = helper.cost_function_only_HL_wsel(
                best_params_tf, n, val_set, val_HLs, central_point)
        else:
            _, train_HL_pred = helper.cost_function_only_HL(
                best_params_tf, n, train_set, train_HLs, central_point)
            _, val_HL_pred = helper.cost_function_only_HL(
                best_params_tf, n, val_set, val_HLs, central_point)

        train_HL_pred    = np.asarray(train_HL_pred).ravel()
        val_HL_pred      = np.asarray(val_HL_pred).ravel()
        train_log10_true = np.asarray(train_HLs, dtype=float).ravel()
        val_log10_true   = np.asarray(val_HLs,   dtype=float).ravel()

        train_HL_true = 10.0 ** train_log10_true
        val_HL_true   = 10.0 ** val_log10_true

        helper.plot_parity(
            train_HL_pred, train_HL_true, val_HL_pred, val_HL_true,
            quantity="Half-life", units="s",
            save_path=os.path.join(SAVE_DIR, "parity_half_life.png"),
            log_scale=True,
        )

        # Save raw numbers for later inspection / notebook reuse.
        np.savetxt(
            os.path.join(SAVE_DIR, "parity_half_life_train.txt"),
            np.column_stack([train_sample_indices, train_HL_true, train_HL_pred]),
            header="sample_index  HL_true[s]  HL_pred[s]",
        )
        np.savetxt(
            os.path.join(SAVE_DIR, "parity_half_life_val.txt"),
            np.column_stack([val_sample_indices, val_HL_true, val_HL_pred]),
            header="sample_index  HL_true[s]  HL_pred[s]",
        )
        print(f"Saved parity plot: {os.path.join(SAVE_DIR, 'parity_half_life.png')}")
