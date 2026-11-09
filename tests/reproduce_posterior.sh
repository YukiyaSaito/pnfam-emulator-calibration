#!/usr/bin/env bash
# Rerun one calibration from the committed emulators and compare its summary with the committed posterior.
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$HERE/.."
F="${1:-HFB1}"; S="${2:-setA}"; L="${3:-studentt}"
export PNFAM_CALIB_RESULTS="${PNFAM_CALIB_RESULTS:-$REPO/results}"
SETS="$S" LIKS="$L" "$REPO/calibration/runs/run_paper_posteriors.sh" "$F" || exit 1
new=$(ls -d "$PNFAM_CALIB_RESULTS/$F"/bayes_*_"$( [[ $L == studentt ]] && echo t4gtt4 || echo norm )"_*_sem21001/centroid 2>/dev/null | tail -1)
ref="$REPO/data/posteriors/$F/${S}_${L}"; [[ $F == HFB1 && $S == L10 ]] && ref="$REPO/data/posteriors/$F/G10_${L}"
echo "new: $new"; echo "ref: $ref"
paste -d'|' <(cut -d, -f1-4 "$new/summary.csv") <(cut -d, -f2-4 "$ref/summary.csv")
cmp -s <(cut -d, -f1-4 "$new/summary.csv") <(cut -d, -f1-4 "$ref/summary.csv") && echo "summary.csv (mean, sd, hdi_3%) identical" || echo "summary.csv differs"
