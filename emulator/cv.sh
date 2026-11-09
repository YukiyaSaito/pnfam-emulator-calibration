#!/usr/bin/env bash
# Drive K-fold CV for any of the *_mod.py training scripts.
#
# Usage:
#   ./cv.sh K SAVE_BASE SCRIPT -- [args passed through to SCRIPT]
#
# Args:
#   K           number of folds
#   SAVE_BASE   base output dir; each fold writes to $SAVE_BASE/fold_<i>/
#   SCRIPT      training script name (e.g. main_only_centroid_mod.py)
#   --          separator (required)
#   [rest]      all remaining args are forwarded verbatim to SCRIPT.
#               Do NOT include --n-folds, --fold-index, or --save-dir;
#               this wrapper sets them.
#
# Examples:
#   ./cv.sh 5 /path/to/cv_centroid/Nd150 main_only_centroid_mod.py -- \
#       --nucleus Nd_150 --data-dir /path/to/GT/Nd150/full --ex-nonneg
#
#   ./cv.sh 5 /path/to/cv_hl/Ca48 main_only_HL_mod.py -- \
#       --hl-file /path/to/halflives.txt
#
#   ./cv.sh 5 /path/to/cv_full/Nd150 main_mod.py -- \
#       --nucleus Nd_150 --data-dir /path/to/GT/Nd150/full \
#       --loss-type windowed --ex-nonneg
#
# Env overrides:
#   MAX_PARALLEL   cap on simultaneous folds (default K; set 1 for sequential).

set -euo pipefail

if [[ $# -lt 4 ]]; then
    sed -n '2,30p' "$0"
    exit 1
fi

K="$1"
BASE="$2"
SCRIPT="$3"
shift 3
if [[ "${1:-}" != "--" ]]; then
    echo "ERROR: expected -- separator before script args." >&2
    exit 1
fi
shift  # drop the --

MAX_PARALLEL="${MAX_PARALLEL:-$K}"

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ ! -f "$THIS_DIR/$SCRIPT" ]]; then
    echo "ERROR: $THIS_DIR/$SCRIPT not found." >&2
    exit 1
fi

mkdir -p "$BASE"

echo "Starting $K-fold CV with $SCRIPT"
echo "  base=$BASE  max_parallel=$MAX_PARALLEL"
echo "  forwarded args: $*"

running=0
for i in $(seq 0 $((K-1))); do
    (
        python "$THIS_DIR/$SCRIPT" \
            --n-folds "$K" --fold-index "$i" \
            --save-dir "$BASE/fold_$i" \
            "$@" \
            > "$BASE/fold_$i.log" 2>&1
    ) &
    running=$((running + 1))
    if (( running >= MAX_PARALLEL )); then
        wait -n || true
        running=$((running - 1))
    fi
done

wait
echo "All $K folds complete. Logs: $BASE/fold_*.log"
