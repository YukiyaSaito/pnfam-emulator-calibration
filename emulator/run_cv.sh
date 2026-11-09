#!/usr/bin/env bash
# Five-fold cross-validation training of the emulators used in the paper, from data/training.
#
# Usage:  ./run_cv.sh HL  SKOP [nucleus ...]      half-life emulators (default: the 40 pool nuclei + the 2 held back)
#         ./run_cv.sh GT  SKOP [nucleus ...]      GT-resonance centroid emulators (default: the 8 GT nuclei)
# Output: $CV_ROOT (default <repo>/results/cv)/<F>/{HL,GT}/<nuc>[/full]/fold_k/seed_s/  (+ parity files per fold)
# Then deploy the fold with the lowest validation cost:  python pick_best_cv.py --functional SKOP --kind HL
#
# Settings of the paper (fold meta.txt of the committed emulators): model dimension n = 5, architecture "ae",
# four seeds 42-45 per fold, 30000 iterations, five folds, stratified split for the half-lives; the GT centroid
# emulators use the nucleus-dependent window of config.get_gt_window and the shift of config.get_gt_shift.
# Requires TensorFlow (see ../environment/requirements-emulator.txt). Runtime: a few minutes per fold and seed on 2 cores.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$HERE/.."
KIND="${1:?HL or GT}"; FUNC="${2:?SKOP | SLY4 | HFB1}"; shift 2
CV_ROOT="${CV_ROOT:-$REPO/results/cv}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}" TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 CUDA_VISIBLE_DEVICES=-1
export PYTHONPATH="$REPO/calibration${PYTHONPATH:+:$PYTHONPATH}"   # config.py
if [[ $KIND == HL ]]; then
  NUCLEI=("$@"); [[ ${#NUCLEI[@]} -eq 0 ]] && NUCLEI=($(ls "$REPO/data/training/$FUNC/HL"))
  for nuc in "${NUCLEI[@]}"; do
    hl="$REPO/data/training/$FUNC/HL/$nuc/halflives.txt"; [[ -f $hl ]] || { echo "skip $nuc: no $hl"; continue; }
    echo "=== HL CV: $FUNC $nuc"
    "$HERE/cv.sh" 5 "$CV_ROOT/$FUNC/HL/$nuc" main_only_HL_mod.py -- \
        --n 5 --n-restarts 4 --seed0 42 --num-iter 30000 --print-every 1000 \
        --hl-file "$hl" --plots save --val-split 0.2 --stratified
  done
elif [[ $KIND == GT ]]; then
  NUCLEI=("$@"); [[ ${#NUCLEI[@]} -eq 0 ]] && NUCLEI=($(ls "$REPO/data/training/$FUNC/GT"))
  for nuc in "${NUCLEI[@]}"; do
    dd="$REPO/data/training/$FUNC/GT/$nuc/full"; [[ -d $dd ]] || { echo "skip $nuc: no $dd"; continue; }
    echo "=== GT centroid CV: $FUNC $nuc"
    "$HERE/cv.sh" 5 "$CV_ROOT/$FUNC/GT/$nuc/full" main_only_centroid_mod.py -- \
        --n 5 --n-restarts 4 --seed0 42 --num-iter 30000 --print-every 1000 \
        --data-dir "$dd" --nucleus "$nuc" --functional "$FUNC" --plots save
  done
else
  echo "KIND must be HL or GT" >&2; exit 1
fi
