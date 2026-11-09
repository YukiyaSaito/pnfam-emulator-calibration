#!/usr/bin/env bash
# Regenerate every figure and table in place and compare the PNG and .tex outputs with the committed copies.
# The committed PDFs are restored afterwards (a regenerated PDF differs only by its creation time).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO="$(cd "$HERE/.." && pwd)"; PY="${PYTHON:-python}"
REF="$(mktemp -d)"; cp -r "$REPO/paper/figs" "$REPO/paper/tables" "$REF/"
cd "$REPO/paper/scripts" || exit 1
fail=0
for s in fig_q_correction fig_emulator_summary fig_posterior_A fig_posteriors_final fig_pool_residuals fig_gt_residuals \
         fig_n126_halflives_ratio fig_n126_halflives_compare fig_n126_ff_fraction fig_n126_error_budget fig_residual_examples \
         table_pool_nuclei table_calibsets table_posteriors table_qcorr table_n126_gs; do
  "$PY" "$s.py" > "$REF/$s.log" 2>&1 || { echo "FAILED $s (see $REF/$s.log)"; fail=1; }
done
# Fig. 4: redraw from the committed data file; set FULL_FIG4=1 to recompute the curves (about 10 min)
"$PY" fig_information_saturation.py $([[ "${FULL_FIG4:-0}" == 1 ]] || echo --replot) > "$REF/fig_information_saturation.log" 2>&1 || { echo "FAILED fig_information_saturation"; fail=1; }
for f in "$REF"/figs/*.png; do b=$(basename "$f"); cmp -s "$f" "$REPO/paper/figs/$b" && echo "same  $b" || { echo "DIFF  $b"; fail=1; }; done
for t in "$REF"/tables/*.tex; do b=$(basename "$t"); cmp -s "$t" "$REPO/paper/tables/$b" && echo "same  $b" || { echo "DIFF  $b"; fail=1; }; done
cp "$REF"/figs/*.pdf "$REPO/paper/figs/"          # restore the committed PDFs
[[ $fail -eq 0 ]] && echo "all figures and tables reproduced" || echo "differences found (reference copies in $REF)"
exit $fail
