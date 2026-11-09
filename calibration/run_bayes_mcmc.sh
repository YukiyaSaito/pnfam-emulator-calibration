#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# RUN SETTINGS -- every knob lives in this block. Edit here.
#
# Each is also overridable from the environment for one-off runs, e.g.
#   DRAWS=500 GT_LIKELIHOOD=studentt ./run_bayes_mcmc.sh
# The three positional arguments (FUNCTIONAL, GT_MODE, SUBDIR) still win over
# both, so existing invocations keep working unchanged.
# =============================================================================

# ---- what is being calibrated ----------------------------------------------
FUNCTIONAL="${FUNCTIONAL:-SKOP}"    # SKOP | SLY4 | HFB1
GT_MODE="${GT_MODE:-centroid}"      # full | full_centroid | full_peak
                                    # | full_window_peak | centroid
                                    # | centroid_total | peak
HL_SET="${HL_SET:-A}"             # any key in config.HL_SETS (A, G10_SKOP, L10_SKOP, POOL40_<F>, ...;
                                    # OPT, G8, G8ALT, A8, ...) or 'all'
GT_SET="${GT_SET:-all}"         # defaults to the HL set name; must be a
                                    # config.GT_SETS key (APLUS has no GT twin)

# ---- likelihood families ----------------------------------------------------
HL_LIKELIHOOD="${HL_LIKELIHOOD:-studentt}"   # normal | studentt. studentt is the
                                    # robust variance-matched Student-t that
                                    # down-weights model-discrepant nuclei
                                    # (e.g. Pt200) instead of dropping them.
HL_NU="${HL_NU:-4}"                 # used only when HL_LIKELIHOOD=studentt
GT_LIKELIHOOD="${GT_LIKELIHOOD:-studentt}"   # normal | studentt. studentt lets the
                                    # full GT set be used without hand-picking:
                                    # it down-weights model-discrepant nuclei
                                    # instead (on SKO' the low-cs0 Zr90/Ca48).
GT_NU="${GT_NU:-4}"                 # used only when GT_LIKELIHOOD=studentt

# ---- error model ------------------------------------------------------------
PER_NUCLEUS="${PER_NUCLEUS:-0}"     # 1 -> --per-nucleus-sigma, 0 -> shared sigma
SIGMA_THE_SCALE="${SIGMA_THE_SCALE:-1}"  # scale of the HalfStudentT(nu=4) prior on
                                    # the shared theory errors (HL in log10 s,
                                    # GT in MeV; one value for both). 1 = weakly
                                    # informative (adopted 2026-09-16). Runs made
                                    # before that date carry no _sthe tag and
                                    # used 0.3, which pulls sigma_the down at
                                    # the fitted 0.5-1.4 (see bayes_mcmc.py).
# ---- parameter prior --------------------------------------------------------
PRIOR="${PRIOR:-domain}"            # domain | legacy. 'domain' (adopted
                                    # 2026-09-22) = stability-domain prior:
                                    # vpair TruncatedNormal(equal-pairing center
                                    # -264.2 SKOP / -196.3 UNE1, sigma 200) on
                                    # [V_LIM, 0] x tanh^2((V0-V_LIM)/TAPER_W),
                                    # V_LIM -371 SKOP / -301 UNE1, cs0 flat on
                                    # [50, 250]. 'legacy' = pre-2026-09-22 prior
                                    # (TN(-250,200) on [-450,0] x TN(130,100) on
                                    # [0,250]); runs without a _dom tag used it.
V_LIM="${V_LIM:-}"                  # override the vpair edge (empty = default)
TAPER_W="${TAPER_W:-10}"            # soft-edge width [MeV fm^3]; 0 = hard edge
VPAIR_MU="${VPAIR_MU:-}"            # override the vpair prior center
CS_LOWER="${CS_LOWER:-}"            # override the cs0 lower bound (default 50)
SIGMA_EM_FILE="${SIGMA_EM_FILE:-}"  # certified per-nucleus sigma_em (NUC=VALUE
                                    # lines from calibration/frozen_sigma_em.py);
                                    # empty -> pooled CV values. Explicit
                                    # SIGMA_EM_OVERRIDE entries still win.
SIGMA_Q_FILE="${SIGMA_Q_FILE:-}"    # per-nucleus HL error from the Q-value
                                    # correction (NUC=VALUE lines in log10 s,
                                    # calibration/analysis/sigma_q_<F>.dat);
                                    # added in quadrature next to sigma_em.
                                    # Empty -> no term. Set -> _q tag. Path is
                                    # used as given after cd to CODE_DIR, so
                                    # pass an absolute path (2026-10-13).
NO_SIGMA_EM="${NO_SIGMA_EM:-0}"     # 1 -> --no-sigma-em (drop CV emulator error
                                    # from the budget; diagnostic only)
HL_EMU_SRC=()                       # per-nucleus HL emulator source overrides,
                                    # e.g. HL_EMU_SRC=(Pt200=SKOP_PT200LOC).
                                    # For Pt200 prefer run_bayes_mcmc_pt200loc.sh
SIGMA_EM_OVERRIDE=()                # per-nucleus HL sigma_em in log10(s),
                                    # e.g. SIGMA_EM_OVERRIDE=(Pt200=0.085)
[[ -n "${SIGMA_EM_OVERRIDE_PT200:-}" ]] && SIGMA_EM_OVERRIDE+=("Pt200=$SIGMA_EM_OVERRIDE_PT200")
                                    # env convenience for the global Pt200
                                    # emulator (2026-09-22): its CERTIFIED
                                    # posterior-region error; arrays cannot be
                                    # passed through the environment.

# ---- sampler ----------------------------------------------------------------
SAMPLER="${SAMPLER:-smc}"           # smc | nuts | demz | slice
DRAWS="${DRAWS:-2000}"
TUNE="${TUNE:-1000}"                # ignored by smc
CHAINS="${CHAINS:-4}"               # ignored by smc
CORES="${CORES:-4}"
SEED="${SEED:-42}"

# ---- output -----------------------------------------------------------------
SUBDIR="${SUBDIR:-}"                # empty -> derived from the settings below
SAVE_DIR="${SAVE_DIR:-}"            # empty -> results/<FUNCTIONAL>/<SUBDIR>/<GT_MODE>

# ---- plotting ---------------------------------------------------------------
RUN_PLOT="${RUN_PLOT:-1}"           # 0 -> skip bayes_plot.py entirely
PLOT_N_HL_SAMPLES="${PLOT_N_HL_SAMPLES:-500}"
PLOT_N_GT_SAMPLES="${PLOT_N_GT_SAMPLES:-100}"
PLOT_SEED="${PLOT_SEED:-0}"
PLOT_NO_GT_SPECTRUM="${PLOT_NO_GT_SPECTRUM:-0}"   # 1 -> --no-gt-spectrum

# ---- environment ------------------------------------------------------------
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"   # this directory (bayes_mcmc.py, config.py)
CONDA_ENV="${CONDA_ENV:-pymc_env}"
LOAD_MODULES="${LOAD_MODULES:-0}"   # 1 -> module purge/load Miniforge3 + conda activate $CONDA_ENV (MSU HPCC);
                                    # 0 -> use the python already on PATH
                                    # (use when already inside the right env)
DRY_RUN="${DRY_RUN:-0}"             # 1 -> print the commands instead of running
# =============================================================================

usage() {
    cat <<EOF
Usage: $(basename "$0") [-h|--help] [FUNCTIONAL] [GT_MODE] [SUBDIR]

Run Bayesian MCMC calibration and produce diagnostic plots.

Positional arguments (optional; they override the settings block):
  FUNCTIONAL   UNE1 | SKOP                                          (default: $FUNCTIONAL)
  GT_MODE      full | full_centroid | full_peak | full_window_peak
               | centroid | centroid_total | peak                   (default: $GT_MODE)
  SUBDIR       override the output directory name (default: derived,
               see below)

ALL run settings live in the "RUN SETTINGS" block at the top of this script:
calibration sets, likelihood families, error model, sampler, output paths,
plotting and the conda/module environment. Every one of them can also be set
as an environment variable for a one-off run, e.g.

  DRAWS=500 GT_LIKELIHOOD=studentt ./$(basename "$0")
  RUN_PLOT=0 SAMPLER=nuts CHAINS=2 ./$(basename "$0") SKOP centroid

The output subdirectory under results/<FUNCTIONAL>/ is DERIVED from the
settings so the name always tells the truth about what was run:

  bayes_<lik>[gt<gtlik>][_pernuc]_sthe<scale>[_dom...][_q]_<HLSET>[_gt<GTSET>][_<gtmode>][_<FUNCTIONAL>]

  <lik>      HL likelihood: norm (Normal) or t<nu> (Student-t), e.g. t4
  gt<gtlik>  GT likelihood, appended to <lik> WITHOUT an underscore and only
             when it is Student-t, e.g. normgtt4 (Normal HL + t4 GT) or
             t4gtt4 (both t4). A Normal GT likelihood adds nothing, so
             existing directory names are unchanged.
  _pernuc    only when per-nucleus sigma is on
  _sthe<s>   theory-error prior scale (SIGMA_THE_SCALE), always present;
             directories without it predate 2026-09-16 and used 0.3
  _q         only when SIGMA_Q_FILE is set (Q-correction HL error term)
  _gt<SET>   only when the GT set differs from the HL set
  _<gtmode>  only when GT_MODE != centroid (full_window_peak -> wpeak,
             full_centroid -> fcent, full_peak -> fpeak,
             centroid_total -> ctot)
  _<FUNC>    only when FUNCTIONAL != SKOP

Examples of derived names:
  normal, shared sigma, REC, centroid, SKOP     -> bayes_norm_REC
  Student-t nu=4, per-nucleus, A8               -> bayes_t4_pernuc_A8
  normal, shared, G8 with GT set OPT, UNE1      -> bayes_norm_G8_gtOPT_UNE1
  Student-t nu=4 on BOTH HL and GT, A8          -> bayes_t4gtt4_A8

NOTE: this script uses the plain deployed emulators. For runs that need the
locally-refined Pt200 emulator + frozen sigma_em override (any set containing
Pt200: REC/OPT/G8/G8ALT/A8), prefer run_bayes_mcmc_pt200loc.sh.
EOF
}

case "${1:-}" in
    -h|--help) usage; exit 0 ;;
esac

# Positional arguments override the settings block.
FUNCTIONAL="${1:-$FUNCTIONAL}"
GT_MODE="${2:-$GT_MODE}"

VALID_FUNCTIONALS=(UNE1 SKOP SLY4 HFB1)   # SLY4, HFB1 added 2026-10-12 (bayes_mcmc.py already has their domain-prior defaults)
VALID_GT_MODES=(full full_centroid full_peak full_window_peak centroid centroid_total peak)
VALID_SAMPLERS=(smc nuts demz slice)

contains() {
    local needle="$1"; shift
    for x in "$@"; do [[ "$x" == "$needle" ]] && return 0; done
    return 1
}

if ! contains "$FUNCTIONAL" "${VALID_FUNCTIONALS[@]}"; then
    echo "ERROR: invalid FUNCTIONAL '$FUNCTIONAL'" >&2
    usage >&2
    exit 1
fi
if ! contains "$GT_MODE" "${VALID_GT_MODES[@]}"; then
    echo "ERROR: invalid GT_MODE '$GT_MODE'" >&2
    usage >&2
    exit 1
fi
if ! contains "$SAMPLER" "${VALID_SAMPLERS[@]}"; then
    echo "ERROR: SAMPLER must be one of ${VALID_SAMPLERS[*]} (got '$SAMPLER')" >&2
    exit 1
fi

# ---- derive the subdir name from the settings -------------------------------
case "$HL_LIKELIHOOD" in
    normal)   lik_tag="norm" ;;
    studentt) lik_tag="t${HL_NU}" ;;
    *) echo "ERROR: HL_LIKELIHOOD must be normal|studentt (got '$HL_LIKELIHOOD')" >&2
       exit 1 ;;
esac
case "$GT_LIKELIHOOD" in
    normal)   gtlik_tag="" ;;
    studentt) gtlik_tag="gtt${GT_NU}" ;;
    *) echo "ERROR: GT_LIKELIHOOD must be normal|studentt (got '$GT_LIKELIHOOD')" >&2
       exit 1 ;;
esac
case "$GT_MODE" in
    centroid)          mode_tag="" ;;
    centroid_total)    mode_tag="_ctot" ;;
    peak)              mode_tag="_peak" ;;
    full)              mode_tag="_full" ;;
    full_centroid)     mode_tag="_fcent" ;;
    full_peak)         mode_tag="_fpeak" ;;
    full_window_peak)  mode_tag="_wpeak" ;;
esac
# _sthe<scale> always present (scale printed without trailing zeros: 1, 0.3)
sthe_tag="_sthe$(printf '%s' "$SIGMA_THE_SCALE" | sed -E 's/(\.[0-9]*[1-9])0+$/\1/; s/\.0+$//')"
# _dom[v<|V_LIM|>]w<TAPER_W> when PRIOR=domain (V_LIM only if overridden)
prior_tag=""
if [[ "$PRIOR" == "domain" ]]; then
    prior_tag="_dom"
    [[ -n "$V_LIM" ]] && prior_tag+="v$(printf '%s' "$V_LIM" | tr -d '-')"
    prior_tag+="w$(printf '%s' "$TAPER_W" | sed -E 's/(\.[0-9]*[1-9])0+$/\1/; s/\.0+$//')"
    [[ -n "$VPAIR_MU" ]] && prior_tag+="mu$(printf '%s' "$VPAIR_MU" | tr -d '-')"
    [[ -n "$CS_LOWER" ]] && prior_tag+="cs$(printf '%s' "$CS_LOWER")"
fi
# _q when the Q-correction HL error term is on (SIGMA_Q_FILE set)
q_tag=""
[[ -n "$SIGMA_Q_FILE" ]] && q_tag="_q"
if [[ -z "$SUBDIR" ]]; then
    SUBDIR="bayes_${lik_tag}${gtlik_tag}"
    [[ "$PER_NUCLEUS" == "1" ]] && SUBDIR+="_pernuc"
    SUBDIR+="${sthe_tag}${prior_tag}${q_tag}_${HL_SET}"
    [[ "$GT_SET" != "$HL_SET" ]] && SUBDIR+="_gt${GT_SET}"
    SUBDIR+="$mode_tag"
    [[ "$FUNCTIONAL" != "SKOP" ]] && SUBDIR+="_${FUNCTIONAL}"
fi
SUBDIR="${3:-$SUBDIR}"
# Optional suffix on the DERIVED name (ignored when SUBDIR is given explicitly),
# e.g. SUBDIR_SUFFIX=_oldemu to keep a baseline made with the pre-retrain
# emulators apart from the same settings rerun after a re-promotion.
[[ -z "${3:-}" ]] && SUBDIR+="${SUBDIR_SUFFIX:-}"
# -----------------------------------------------------------------------------

ARGS=(
    --gt-mode "$GT_MODE"
    --functional "$FUNCTIONAL"
    --sampler "$SAMPLER"
    --draws "$DRAWS"
    --tune "$TUNE"
    --chains "$CHAINS"
    --cores "$CORES"
    --seed "$SEED"
    --bayes-subdir "$SUBDIR"
    --hl-set "$HL_SET"
    --gt-set "$GT_SET"
    --hl-likelihood "$HL_LIKELIHOOD"
    --hl-nu "$HL_NU"
    --gt-likelihood "$GT_LIKELIHOOD"
    --gt-nu "$GT_NU"
    --sigma-the-scale "$SIGMA_THE_SCALE"
    --prior "$PRIOR"
    --taper-w "$TAPER_W"
)
[[ -n "$V_LIM" ]] && ARGS+=(--v-lim "$V_LIM")
[[ -n "$VPAIR_MU" ]] && ARGS+=(--vpair-mu "$VPAIR_MU")
[[ -n "$CS_LOWER" ]] && ARGS+=(--cs-lower "$CS_LOWER")
[[ "$PER_NUCLEUS" == "1" ]] && ARGS+=(--per-nucleus-sigma)
[[ "$NO_SIGMA_EM" == "1" ]] && ARGS+=(--no-sigma-em)
[[ -n "$SIGMA_EM_FILE" ]] && ARGS+=(--sigma-em-file "$SIGMA_EM_FILE")
[[ -n "$SIGMA_Q_FILE" ]] && ARGS+=(--sigma-q-file "$SIGMA_Q_FILE")
[[ -n "$SAVE_DIR" ]] && ARGS+=(--save-dir "$SAVE_DIR")
[[ ${#HL_EMU_SRC[@]} -gt 0 ]] && ARGS+=(--hl-emu-src "${HL_EMU_SRC[@]}")
[[ ${#SIGMA_EM_OVERRIDE[@]} -gt 0 ]] && ARGS+=(--sigma-em-override "${SIGMA_EM_OVERRIDE[@]}")

PLOT_ARGS=(
    --gt-mode "$GT_MODE"
    --functional "$FUNCTIONAL"
    --bayes-subdir "$SUBDIR"
    --hl-set "$HL_SET"
    --gt-set "$GT_SET"
    --n-hl-samples "$PLOT_N_HL_SAMPLES"
    --n-gt-samples "$PLOT_N_GT_SAMPLES"
    --seed "$PLOT_SEED"
)
[[ "$PLOT_NO_GT_SPECTRUM" == "1" ]] && PLOT_ARGS+=(--no-gt-spectrum)
[[ -n "$SAVE_DIR" ]] && PLOT_ARGS+=(--save-dir "$SAVE_DIR")

echo "settings: hl=$HL_SET gt=$GT_SET mode=$GT_MODE pernuc=$PER_NUCLEUS"
echo "  likelihood: HL=$HL_LIKELIHOOD nu=$HL_NU  GT=$GT_LIKELIHOOD nu=$GT_NU  sigma_the prior scale=$SIGMA_THE_SCALE"
echo "  prior: $PRIOR  V_LIM=${V_LIM:-default}  TAPER_W=$TAPER_W  VPAIR_MU=${VPAIR_MU:-default}  CS_LOWER=${CS_LOWER:-default}"
echo "  sigma_Q file: ${SIGMA_Q_FILE:-none}"
echo "  sampler: $SAMPLER draws=$DRAWS tune=$TUNE chains=$CHAINS cores=$CORES seed=$SEED"
echo "  ->  results/$FUNCTIONAL/$SUBDIR/"

if [[ "$DRY_RUN" == "1" ]]; then
    echo "DRY_RUN: python bayes_mcmc.py ${ARGS[*]}"
    [[ "$RUN_PLOT" == "1" ]] && echo "DRY_RUN: python bayes_plot.py ${PLOT_ARGS[*]}"
    exit 0
fi

if [[ "$LOAD_MODULES" == "1" ]]; then
    module purge
    module load Miniforge3
    conda activate "$CONDA_ENV"
fi

cd "$CODE_DIR"

python bayes_mcmc.py "${ARGS[@]}"

if [[ "$RUN_PLOT" == "1" ]]; then
    python bayes_plot.py "${PLOT_ARGS[@]}"
fi
