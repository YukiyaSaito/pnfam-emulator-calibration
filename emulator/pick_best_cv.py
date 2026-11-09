#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pick the best K-fold CV emulator for each production CV leaf directory and
copy its MCMC-ready artifacts into the corresponding results/ tree.

A "leaf CV directory" is any directory containing fold_0/ (and typically
fold_1/, ...). For each leaf we read best_val_cost from every
fold_*/seed_*/meta.txt and keep the fold with the smallest one. The picked
fold's params_best_*.txt and training-set coordinates are copied to:

    <results-root>/<rel-path-of-leaf-relative-to-cv-root>/

so that e.g. cv/UNE1/HL/Ar48/  ->  results/UNE1/HL/Ar48/.

The Bayesian MCMC loader expects a comma-delimited train_set.txt of scaled
(vpair_t0, override_cs0) points. Scalar CV already writes that file. The full
GT strength CV writes train_set_scaled.txt instead, so this script converts
that file to train_set.txt while promoting K0/K1 emulators.

A picker_meta.txt is written next to the copied files recording which fold
was chosen and its val cost.

Usage:
    pick_best_cv.py --functional SKOP --kind all
    pick_best_cv.py --functional UNE1 --kind GT
    pick_best_cv.py --base /path/to/cv/UNE1/HL
    pick_best_cv.py --dry-run
"""
import argparse
import glob
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import repo_paths as RP  # noqa: E402
DEFAULT_CV_ROOT      = RP.CV          # full CV runs (fold_k/seed_s/meta.txt) are not in the repository, only the parity files
DEFAULT_RESULTS_ROOT = RP.EMULATORS   # where the selected emulator parameters are deployed
FUNCTIONALS = ("SKOP", "SLY4", "HFB1")
KINDS = ("HL", "GT")   # GT = windowed-centroid emulators (<F>/GT/<nuc>/full in this repository)


def read_best_val_cost(meta_path):
    """Return float(best_val_cost) from a meta.txt, or None on failure."""
    try:
        with open(meta_path) as f:
            for line in f:
                if line.startswith("best_val_cost="):
                    return float(line.split("=", 1)[1].strip())
    except (IOError, ValueError):
        return None
    return None


def find_leaf_cv_dirs(base):
    """Every directory under base that contains a fold_0/ subdir."""
    leafs = set()
    for fold0 in glob.iglob(os.path.join(base, "**", "fold_0"), recursive=True):
        if os.path.isdir(fold0):
            leafs.add(os.path.dirname(fold0))
    return sorted(leafs)


def selected_leaf(leaf, cv_root, functional_filter, kind_filter):
    """Return True when leaf belongs to the requested production subtree."""
    rel = os.path.relpath(leaf, cv_root)
    parts = rel.split(os.sep)
    if rel.startswith(".." + os.sep) or rel == ".." or len(parts) < 2:
        return False
    functional, kind = parts[0], parts[1]
    if functional not in FUNCTIONALS or kind not in KINDS:
        return False
    if "full_combined" in parts:
        return False
    if functional_filter != "all" and functional != functional_filter:
        return False
    if kind_filter != "all" and kind != kind_filter:
        return False
    return True


def best_fold(leaf_dir):
    """For a leaf CV directory, return (fold_path, val_cost, source_meta) for
    the fold whose minimum-across-restarts best_val_cost is smallest.
    Returns None if no fold has a readable meta.txt.
    """
    best = None
    for fold in sorted(glob.glob(os.path.join(leaf_dir, "fold_*"))):
        if not os.path.isdir(fold):
            continue
        per_fold_best = None
        per_fold_src  = None
        for seed_dir in sorted(glob.glob(os.path.join(fold, "seed_*"))):
            meta = os.path.join(seed_dir, "meta.txt")
            v = read_best_val_cost(meta)
            if v is None:
                continue
            if per_fold_best is None or v < per_fold_best:
                per_fold_best = v
                per_fold_src  = meta
        if per_fold_best is None:
            continue
        if best is None or per_fold_best < best[1]:
            best = (fold, per_fold_best, per_fold_src)
    return best


def load_scaled_train_set(fold_dir):
    """Load a scaled two-column train set from train_set.txt or train_set_scaled.txt.

    Returns (array, source_label). train_set.txt is already comma-delimited in
    scalar CV folds. train_set_scaled.txt is whitespace-delimited in full GT
    folds and may include a header.
    """
    train_set = os.path.join(fold_dir, "train_set.txt")
    if os.path.isfile(train_set):
        arr = np.loadtxt(train_set, delimiter=",")
        return np.atleast_2d(arr)[:, :2], "train_set.txt"

    train_set_scaled = os.path.join(fold_dir, "train_set_scaled.txt")
    if os.path.isfile(train_set_scaled):
        arr = np.loadtxt(train_set_scaled)
        return np.atleast_2d(arr)[:, :2], "train_set_scaled.txt"

    return None, None


def write_train_set(arr, dest_path):
    """Write comma-delimited train_set.txt for SMLREmulator."""
    np.savetxt(dest_path, arr, delimiter=",", fmt="%.12g")


def expected_param_name(rel):
    """Expected destination parameter filename for an MCMC emulator kind."""
    parts = rel.split(os.sep)
    kind = parts[1] if len(parts) > 1 else None
    if kind == "HL":
        return "params_best_n5_only_HL.txt"
    if kind == "GT":
        return "params_best_n5_only_centroid.txt"
    return None


def params_files_for_promotion(fold_dir, source_meta, rel):
    """Return (source, destination-name) pairs for params files.

    Newer CV runs write params_best_*.txt at fold level. Some older scalar CV
    runs only kept restart-level params_n*_seed*.txt, so fall back to the
    restart that supplied the chosen validation cost and rename it to the
    MCMC-expected params_best_* filename.
    """
    params_files = sorted(glob.glob(os.path.join(fold_dir, "params_best_*.txt")))
    if params_files:
        return [(pf, os.path.basename(pf)) for pf in params_files]

    expected_name = expected_param_name(rel)
    seed_dir = os.path.dirname(source_meta) if source_meta else None
    if seed_dir and expected_name:
        seed_params = sorted(glob.glob(os.path.join(seed_dir, "params_n*_seed*.txt")))
        if seed_params:
            return [(seed_params[0], expected_name)]

    return []


def copy_artifacts(fold_dir, dest_dir, val_cost, source_meta, rel, dry_run):
    """Copy params_best_*.txt and MCMC-ready train_set.txt from fold_dir."""
    params_files = params_files_for_promotion(fold_dir, source_meta, rel)
    if not params_files:
        print(f"  WARN: no promotable params in {fold_dir}")
        return False
    train_set_arr, train_set_source = load_scaled_train_set(fold_dir)
    if train_set_arr is None:
        print(f"  WARN: no train_set.txt or train_set_scaled.txt in {fold_dir}")
        return False
    if not dry_run:
        os.makedirs(dest_dir, exist_ok=True)
    for src, dest_name in params_files:
        dst = os.path.join(dest_dir, dest_name)
        note = "" if os.path.basename(src) == dest_name else f" -> {dest_name}"
        print(f"  copy {os.path.basename(src)}{note}")
        if not dry_run:
            shutil.copy2(src, dst)
    action = "copy" if train_set_source == "train_set.txt" else "create"
    print(f"  {action} train_set.txt from {train_set_source}")
    if not dry_run:
        write_train_set(train_set_arr, os.path.join(dest_dir, "train_set.txt"))
        for extra in ("train_set_scaled.txt", "train_set_unscaled.txt"):
            src = os.path.join(fold_dir, extra)
            if os.path.isfile(src):
                shutil.copy2(src, os.path.join(dest_dir, extra))
    if not dry_run:
        with open(os.path.join(dest_dir, "picker_meta.txt"), "w") as f:
            f.write(f"source_fold={fold_dir}\n")
            f.write(f"best_val_cost={val_cost:.6e}\n")
            f.write(f"source_meta={source_meta}\n")
            f.write(f"train_set_source={train_set_source}\n")
            f.write(f"n_train={len(train_set_arr)}\n")
    return True


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cv-root",      default=DEFAULT_CV_ROOT)
    ap.add_argument("--results-root", default=DEFAULT_RESULTS_ROOT)
    ap.add_argument("--base", default=None,
                    help="Subtree to scan (default: --cv-root). Outputs go under "
                         "results-root mirroring the relative path under cv-root.")
    ap.add_argument("--functional", choices=("all",) + FUNCTIONALS, default="all",
                    help="Restrict to one functional (default: all).")
    ap.add_argument("--kind", choices=("all",) + KINDS, default="all",
                    help="Restrict to one production emulator kind (default: all).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print actions but copy nothing.")
    ap.add_argument("--exclude", nargs="*", default=[],
                    help="Leaf directory basenames (nucleus names) to skip, "
                         "e.g. --exclude Fe66 Pd114 Ti52 to keep damaged-CV "
                         "emulators out of results/.")
    args = ap.parse_args()

    base = args.base or args.cv_root
    exclude = set(args.exclude)
    leafs_all = find_leaf_cv_dirs(base)
    leafs = [
        leaf for leaf in leafs_all
        if selected_leaf(leaf, args.cv_root, args.functional, args.kind)
    ]
    for leaf in [l for l in leafs if os.path.basename(l) in exclude]:
        print(f"--- {leaf}\n    SKIP: excluded by --exclude")
    leafs = [l for l in leafs if os.path.basename(l) not in exclude]
    if not leafs:
        sys.exit(
            f"No selected production CV leaf dirs found under {base} "
            f"(functional={args.functional}, kind={args.kind})"
        )

    print(
        f"Scanning {base} ({len(leafs)} selected production CV leaf dirs; "
        f"{len(leafs_all)} fold_0 dirs found before filtering)\n"
    )

    n_picked = 0
    for leaf in leafs:
        bf = best_fold(leaf)
        if bf is None:
            print(f"--- {leaf}\n    SKIP: no readable seed_*/meta.txt")
            continue
        fold, val, src = bf
        # Spread of fold val-costs for sanity (logged, not used)
        all_costs = []
        for f in sorted(glob.glob(os.path.join(leaf, "fold_*"))):
            sub = []
            for sd in sorted(glob.glob(os.path.join(f, "seed_*"))):
                v = read_best_val_cost(os.path.join(sd, "meta.txt"))
                if v is not None:
                    sub.append(v)
            if sub:
                all_costs.append((os.path.basename(f), min(sub)))
        rel = os.path.relpath(leaf, args.cv_root)
        dest = os.path.join(args.results_root, rel)
        print(f"--- {leaf}")
        for name, c in all_costs:
            mark = "  <-- pick" if name == os.path.basename(fold) else ""
            print(f"    {name}: best_val_cost={c:.6e}{mark}")
        print(f"    dest: {dest}")
        copied = copy_artifacts(fold, dest, val, src, rel, dry_run=args.dry_run)
        if copied:
            n_picked += 1
        print()

    suffix = " (dry run)" if args.dry_run else ""
    print(f"Picked {n_picked}/{len(leafs)} emulators{suffix}.")


if __name__ == "__main__":
    main()
