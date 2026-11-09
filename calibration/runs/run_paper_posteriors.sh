#!/usr/bin/env bash
# Reproduce the posteriors of the paper (Table III, Figs. 3, 5-10) from the trained emulators.
#
# Usage:  ./run_paper_posteriors.sh [FUNCTIONAL ...]          (default: SKOP SLY4 HFB1)
#         SETS="setA L10" ./run_paper_posteriors.sh SLY4       (subset of setA G10 L10 full)
#         LIKS=studentt ./run_paper_posteriors.sh               (subset of studentt normal)
#
# Final settings of the paper (CHAIN record 2026-10-19 / 2026-11-05): Student-t(nu = 4) or normal
# likelihood for both observable types, all eight GT resonance energies (Set A: its six), domain
# prior with taper width 10 MeV fm^3, HalfStudentT(4, scale 1) priors on the theory errors, test
# emulator errors (batch 21001), per-nucleus Q-value error term, SMC sampler, 2000 draws, seed 42.
# Output: $PNFAM_CALIB_RESULTS (default <repo>/results)/<F>/bayes_<lik>_sthe1_domw10_q_<SET>_gtall[_<F>]_sem21001/centroid/
# Runtime: about one minute per Set-A posterior and a few minutes for the Full set (SMC, 4 threads).
# The committed posteriors (data/posteriors) were produced by exactly these settings; a rerun agrees
# with them: SMC with a fixed seed is deterministic, so summary.csv agrees to the printed digits (checked 2026-11-09 for HFB1 Set A).
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CAL="$HERE/.."
ERR="$CAL/../data/errors"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
FUNCS="${@:-SKOP SLY4 HFB1}"
SETS="${SETS:-setA G10 L10 full}"
LIKS="${LIKS:-studentt normal}"
LOG="$HERE/run_paper_posteriors.log"; echo "# $(date) $FUNCS | $SETS | $LIKS" >> "$LOG"

# paper set name -> key in config.py (same table as repo_paths.SET_KEYS)
key () {
  case "$1:$2" in
    *:setA) echo A ;;
    SKOP:G10) echo G10_SKOP ;;      SKOP:L10) echo L10_SKOP ;;      SKOP:full) echo POOL40_SKOP ;;
    SLY4:G10) echo G10_SLY4 ;; SLY4:L10) echo L10_SLY4 ;; SLY4:full) echo POOL40_SLY4 ;;
    HFB1:G10) echo G10_HFB1 ;; HFB1:L10) echo G10_HFB1 ;; HFB1:full) echo POOL40_HFB1 ;;   # HFB1: L10 = G10
    *) echo "unknown set $2 for $1" >&2; return 1 ;;
  esac
}

run_one () {   # functional, paper set name
  local f=$1 s=$2 k gt
  k=$(key "$f" "$s") || return 1
  gt=all; [[ $s == setA ]] && gt=A
  for lik in $LIKS; do
    echo "=== $f $s ($k) $lik $(date)" >> "$LOG"
    FUNCTIONAL=$f HL_SET=$k GT_SET=$gt HL_LIKELIHOOD=$lik GT_LIKELIHOOD=$lik \
      PRIOR=domain TAPER_W=10 SIGMA_THE_SCALE=1 SAMPLER=smc DRAWS=2000 SEED=42 \
      SIGMA_Q_FILE="$ERR/sigma_q_$f.dat" SIGMA_EM_FILE="$ERR/sigma_em_frozen_${f}_batch21001.txt" \
      SUBDIR_SUFFIX=_sem21001 RUN_PLOT="${RUN_PLOT:-0}" \
      "$CAL/run_bayes_mcmc.sh" >> "$LOG.$f.$s.$lik" 2>&1 || echo "FAILED $f $s $lik" >> "$LOG"
  done
}

for f in $FUNCS; do
  for s in $SETS; do
    [[ $f == HFB1 && $s == L10 ]] && continue    # identical to G10
    run_one "$f" "$s" &
    sleep 4        # staggered starts: simultaneous first imports race on ArviZ's daily-warning stamp file
  done
done
wait
echo "ALL DONE $(date)" >> "$LOG"
