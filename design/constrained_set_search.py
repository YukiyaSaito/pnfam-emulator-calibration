#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Constrained exhaustive calibration-set search for the SKO' 2D half-life
calibration (vpair_t0, override_cs0).

For each set size k (6..9), find the HL calibration set that MAXIMIZES the
smallest eigenvalue lambda_min of the joint Fisher matrix (fixed GT base +
summed HL contributions), subject to representation constraints (">=1 nucleus
per stratum cell"), and report the price of coverage

    price = lambda_min(constrained optimum) / lambda_min(unconstrained optimum)

per constraint scheme. Strata grid (user decision 2026-07-09: compare the
scheme variants rather than lock one):

  Q axis        : bin   = lowQ (HFB_Qval < 3 MeV) + rest (>= 3)
                  ter   = lowQ + midQ (3..6) + highQ (> 6)
  deformation   : strict  = >=1 spherical (|b2|<0.10) AND >=1 deformed (>=0.20),
                            transitional (0.10..0.20) fills neither
                  lenient = transitional may fill either seat
                  three   = spherical + transitional + deformed, one seat each
  extra cells   : none | hdl (heavy*deformed*lowQ) | ff (%FF>=50) | both
  always        : mass light(A<=82)/medium/heavy(A>=152); shell N~50/N~82/N~126
  floor         : k >= 6 (sigma_the estimability), scanned k = 6..9
  GT set FIXED  : {Pb208, Sn132, Te130, Zr90, Ge76} (enters the Fisher base;
                  re-locked under the Set-A-anchored sigma_tot convention)

Fisher conventions follow fisher_matrix_report.py:
  g_i = [b_vp, b_cs] in log10(T) (HL) or MeV (GT) per HALF-BOX
        (a = (vpair_t0+250)/250 over (-500,0), b = (cs0-125)/125 over (0,250)),
  F_i = g_i g_i^T / sigma_tot,i^2,
  HL: sigma_tot^2 = sigma_exp_log10^2 + sigma_em^2 + SIGMA_THE_HL^2
  GT: sigma_tot^2 = sigma_MeV^2 + sigma_em_MeV^2 + SIGMA_THE_GT^2   (sigma_em_MeV: CV emulator error, 2026-11-10)
GT centroids carry ~no vpair dependence (pR2_vp <= 0.02), so the GT base is
kept at global slopes in both slope modes.

Robustness axes (reported side by side, never mixed into one objective):
  * slopes GLOBAL (full-box OLS from skop_hl_fisher.csv, the published table)
    vs LOCALIZED (OLS refit on Level-1 samples with vpair in [-450, -330],
    bracketing the 2026-07-08 posterior mode vpair ~ -390; -450 is the
    contour-reliability fence)
  * pool SCREENED (drop monotonic_check "genuinely noisy / weak") vs RAW 38
  * preferred-vpair span of each winning set (vpair_preference_trend.csv):
    reported as a COVERAGE diagnostic ONLY -- deliberately NOT in the Fisher
    weighting (circularity with the quantity being calibrated).

NOTE lambda_min here is computed at the FIXED sigma_the = 0.517 for every set.
Finalist sets must still be judged by their OWN fitted sigma_the / posterior
width from an MCMC run per set (observed 2026-07-08: REC +/-12.2 with
sigma_the 0.46 beats the Fisher-optimal OPT +/-18.6 with sigma_the 0.70).

Read-only inputs; artifacts written to this directory:
  constrained_set_search.csv   one row per pool x slope-mode x k x scheme
  constrained_set_search.txt   human-readable report incl. validation section

Double precision; no float '==' comparisons.
"""
import argparse
import glob
import itertools
import math
import os
import sys
from itertools import chain, islice

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))

# 2026-08-05: generalized to --functional (default SKOP, byte-identical
# behavior). Per-functional anchor budget: sigma_the from that functional's
# own Set-A-anchored fit (SKOP: bayes summary 2026-07-08; UNE1:
# results/UNE1/bayes_norm_A_UNE1 2026-08-05). The screen file is that
# functional's monotonic check.
_pre = argparse.ArgumentParser(add_help=False)
_pre.add_argument("--functional", default="SKOP")
_pre.add_argument("--sigma-the-hl", type=float, default=None)
_pre.add_argument("--sigma-the-gt", type=float, default=None)
_pre.add_argument("--no-sigma-q", action="store_true")
_fk, _ = _pre.parse_known_args()
FUNCTIONAL = _fk.functional
# Registered budgets re-locked 2026-08-12 (user decision: redo the SkO'
# selection rather than keep the July sets under a stale budget). Values
# are posterior MEDIANS of the theory errors from the 2026-08-12 Set-A
# calibrations (paper convention; means verified to give identical
# selections, see *_medcheck artifacts). Previous registered values --
# SKOP 0.517/1.66 with screen monotonic_check_2026-07-08.txt (July era,
# pre-fence) -- live on in the July constrained_set_search.{csv,txt}
# artifacts and the v2 series.
# 2026-09-01: budgets re-derived from Set-A anchor fits run under the
# CONSISTENT RULE (--gt-set all): results/SKOP/bayes_norm_A_gtall and
# results/UNE1/bayes_norm_A_gtall_UNE1. Same convention as before (posterior
# MEDIANS, shared-sigma Normal likelihood -- only the GT set changed).
# Previous registered values (GT = Set-A six, 2026-08-12), live on in the
# *_v3 artifacts: SKOP 0.445/0.910, UNE1 0.393/0.267.
# 2026-09-16 (V4): budgets from the Set-A anchor fits under the scale-1
# theory-error prior (results/SKOP/bayes_norm_sthe1_A, results/UNE1/
# bayes_norm_sthe1_A_UNE1; posterior medians). Windows (16-84% vpair):
# SKO' [-310,-166], UNE1 [-250,-56] -- pass --loc-lo/--loc-hi. Pool rule:
# screen file only; UNE1 additionally --drop Sr92 (no usable emulator).
# Previous (2026-09-01, hybrid all-GT anchors bayes_norm_A_gtall*):
# SKOP 0.452/0.824 [-314,-195]; UNE1 0.392/0.225 [-237,-50].
# 2026-10-12 (Q-corrected pools POOL40_<F>): SLY4 and HFB1 added; SKOP screen
# file and date updated. The sthl/stgt values below are NOT the budgets for
# this round: filled from the 2026-10-12 anchor fit; must be passed with
# --sigma-the-hl/--sigma-the-gt (None = not registered; the script stops
# with an error if the flag is missing).
_FDEF = {
    # sthl/stgt: filled from the 2026-10-12 anchor fit; must be passed with
    # --sigma-the-hl/--sigma-the-gt (0.484/1.191 are the 2026-09-16 V4 values)
    "SKOP": dict(sthl=0.484, stgt=1.191, title="SKO'", date="2026-10-12",
                 screen="monotonic_check_SKOP_2026-10-08.txt"),
    "UNE1": dict(sthl=0.423, stgt=0.460, title="UNE1", date="2026-09-16",
                 screen="monotonic_check_UNE1_2026-08-05.txt"),
    # filled from the 2026-10-12 anchor fit; must be passed with
    # --sigma-the-hl/--sigma-the-gt
    "SLY4": dict(sthl=None, stgt=None, title="SLy4", date="2026-10-12",
                 screen="monotonic_check_SLY4_2026-10-08.txt"),
    # filled from the 2026-10-12 anchor fit; must be passed with
    # --sigma-the-hl/--sigma-the-gt
    "HFB1": dict(sthl=None, stgt=None, title="UNEDF1-HFB", date="2026-10-12",
                 screen="monotonic_check_HFB1_2026-10-08.txt"),
}[FUNCTIONAL]
_stem = "" if FUNCTIONAL == "SKOP" else f"_{FUNCTIONAL}"

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import repo_paths as RP  # noqa: E402
ROOT = os.path.join(RP.TRAINING, FUNCTIONAL, "HL")      # params.txt / halflives.txt per nucleus
CVROOT = os.path.join(RP.CV, FUNCTIONAL, "HL")          # cross-validation parity files
RESROOT = os.path.join(RP.EMULATORS, FUNCTIONAL, "HL")  # deployed emulators
SIGQ_FILE = os.path.join(RP.ERRORS, f"sigma_q_{FUNCTIONAL}.dat")   # per-nucleus Q-value error term (decades)
FISHER_CSV = os.path.join(HERE, f"{FUNCTIONAL.lower()}_hl_fisher.csv")
GT_CSV = os.path.join(HERE, f"{FUNCTIONAL.lower()}_gt_fisher.csv")
TAG_CSV = os.path.join(HERE, "nuclei_tag_table.csv" if FUNCTIONAL == "SKOP"
                       else f"nuclei_tag_table_{FUNCTIONAL}.csv")
TREND_CSV = os.path.join(HERE, f"vpair_preference_trend{_stem}.csv")
SCREEN_TXT = os.path.join(HERE, _FDEF["screen"])
OUT_CSV = os.path.join(HERE, f"constrained_set_search{_stem}.csv")
OUT_TXT = os.path.join(HERE, f"constrained_set_search{_stem}.txt")

sys.path.insert(0, RP.CALIBRATION)
import config  # noqa: E402

# Fitted theory errors from the Set-A-anchored fit (log10 T; MeV). The
# per-functional defaults are the registered (mean-convention) values;
# --sigma-the-hl/--sigma-the-gt override for convention checks.
SIGMA_THE_HL = (_fk.sigma_the_hl if _fk.sigma_the_hl is not None
                else _FDEF["sthl"])
SIGMA_THE_GT = (_fk.sigma_the_gt if _fk.sigma_the_gt is not None
                else _FDEF["stgt"])
if SIGMA_THE_HL is None or SIGMA_THE_GT is None:
    raise SystemExit(
        f"{FUNCTIONAL}: no registered theory-error budget -- pass "
        "--sigma-the-hl and --sigma-the-gt (values from the 2026-10-12 "
        "Set-A anchor fit)")
# CONSISTENT RULE (2026-09-01): the GT base is EVERY measured GT centroid in
# config.GT_E_EXP -- the same set the MCMC calibrates against. All 8 clear the
# emulator-reliability gate (sigma_em 0.002-0.024 MeV vs sigma_exp 0.2-0.5), so
# no information-ranking or R^2 gate is applied any more.
# History: the July/August runs used a hand-picked five
#   ("Pb208","Sn132","Te130","Zr90","Ge76")
# ranked by F_cs with Ca48 gated on R^2 and Nd150 passed over for deformation.
# Two inconsistencies that motivated the change: (a) that five differed from
# the GT set the budget fit itself used (GT_SET_A, six: Ca48/Sn112 in, Sn132
# out); (b) the UNE1 selection was run against Zr90 while the registered UNE1
# GT set had Sn112 (PAPER_FISHER_SECTION_NOTES 2026-08-27).
GT_SET = ("Pb208", "Sn132", "Te130", "Zr90", "Ge76",
          "Sn112", "Nd150", "Ca48")
LOC_WIN = (-450.0, -330.0)    # vpair window for localized slopes [raw units]
# Registered localized windows (16-84% of vpair from each functional's Set-A
# anchor fit): V4 2026-09-16 (scale-1 prior): SKO' [-310,-166]; UNE1
# [-250,-56]. V3 2026-09-01 (GT=all anchors): SKO' [-314,-195]; UNE1 [-237,-50].
# Previous (GT=Set-A six, 2026-08-12): SKO' [-310,-183]; UNE1 [-236,-51].
Q_LOW_MEV, Q_HIGH_MEV = 3.0, 6.0
B2_SPH, B2_DEF = 0.10, 0.20
FF_CUT = 50.0                 # %FF forbidden-dominated cell

# Reference sets + observed MCMC references, per functional.
# SKOP: bayes_t4loc_{REC,OPT}, 2026-07-08, Student-t nu=4,
#       Pt200 -> SKOP_PT200LOC emulator, sigma_em override 0.085.
# UNE1: the Set-A anchor fit (bayes_norm_A_UNE1, Normal shared sigma) and
#       G8X = the SKOP-G8 membership evaluated under UNE1 (cross-functional
#       transfer test; no UNE1 MCMC for it yet).
# 2026-10-12: Set A is the only reference set for every functional, and no
# observed MCMC reference exists yet on the Q-corrected pools (the report
# prints "n/a"). Previous: SKOP REC/OPT (bayes_t4loc, 2026-07-08:
# REC -387.8+/-12.2 sigma_the 0.46; OPT -392.6+/-18.6 sigma_the 0.70);
# UNE1 A/G8X (A: -144.4+/-88.3 sigma_the 0.408, 2026-08-05).
REF_SETS = (("A", "setA"), ("G10", "G10"), ("L10", "L10"), ("Full", "full"))   # paper sets, keys in repo_paths.SET_KEYS
MCMC_REF = {}
MCMC_REF_LBL = "observed MCMC"

CELLS = ["mass_light", "mass_medium", "mass_heavy",
         "q_low", "q_rest", "q_mid", "q_high",
         "shell_N50", "shell_N82", "shell_N126",
         "def_sph", "def_def", "def_trans",
         "def_sph_len", "def_def_len",
         "joint_hdl", "ff50"]
BIT = {c: 1 << i for i, c in enumerate(CELLS)}


# ---------------------------------------------------------------- input layer
def read_csv_dict(path):
    with open(path) as f:
        hdr = f.readline().strip().split(",")
        rows = {}
        for ln in f:
            c = ln.strip().split(",")
            if len(c) < 2:
                continue
            rows[c[0]] = {h: v for h, v in zip(hdr, c)}
    return rows


def fnum(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return float("nan")


def load_screen():
    """Nuclei flagged 'genuinely noisy / weak' by the Spearman monotonic check."""
    drop = set()
    for ln in open(SCREEN_TXT):
        if "genuinely noisy" in ln:
            drop.add(ln.split()[0])
    return drop


def read_two(path, ncol):
    idx, vals = [], []
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        c = ln.split()
        idx.append(int(float(c[0])))
        vals.append([float(x) for x in c[1:1 + ncol]])
    return idx, np.array(vals, dtype=np.float64)


def fit_slopes(nuc, vlo=None, vhi=None):
    """OLS log10(T) = b0 + b_vp*a + b_cs*b on box-scaled params, optionally
    restricted to raw vpair in [vlo, vhi]. Returns (b_vp, b_cs, n, R2)."""
    pidx, praw = read_two(os.path.join(ROOT, nuc, "params.txt"), 2)
    hidx, hval = read_two(os.path.join(ROOT, nuc, "halflives.txt"), 1)
    hmap = {i: v[0] for i, v in zip(hidx, hval)}
    vp, cs, y = [], [], []
    for j, i in enumerate(pidx):
        t = hmap.get(i)
        if t is None or not (t > 0.0):
            continue
        v = praw[j, 0]
        if vlo is not None and (v < vlo or v > vhi):
            continue
        vp.append(v)
        cs.append(praw[j, 1])
        y.append(math.log10(t))
    n = len(y)
    if n < 4:
        return float("nan"), float("nan"), n, float("nan")
    a = (np.array(vp) + 250.0) / 250.0
    b = (np.array(cs) - 125.0) / 125.0
    y = np.array(y)
    X = np.column_stack([np.ones_like(a), a, b])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    sst = ((y - y.mean()) ** 2).sum()
    R2 = 1.0 - ((y - X @ beta) ** 2).sum() / sst if sst > 0.0 else float("nan")
    return float(beta[1]), float(beta[2]), n, float(R2)


def gt_base(gt_set=GT_SET):
    """Fixed-GT 2x2 Fisher base."""
    rows = read_csv_dict(GT_CSV)
    B = np.zeros((2, 2), dtype=np.float64)
    for nuc in gt_set:
        r = rows[nuc]
        g = np.array([fnum(r, "b_vp"), fnum(r, "b_cs")])
        sem = fnum(r, "sigma_em_MeV")          # CV emulator error of the centroid [MeV]; same decomposition as the likelihood
        sem = sem if math.isfinite(sem) else 0.0
        s2 = fnum(r, "sigma_MeV") ** 2 + sem ** 2 + SIGMA_THE_GT ** 2
        B += np.outer(g, g) / s2
    return B


def cell_mask(tag):
    """Bitmask of stratum cells a nucleus belongs to, from its tag-table row."""
    m = 0
    mc = tag["mass_class"]
    m |= BIT["mass_" + mc]
    q = fnum(tag, "HFB_Qval")
    if q < Q_LOW_MEV:
        m |= BIT["q_low"]
    else:
        m |= BIT["q_rest"]
        m |= BIT["q_mid"] if q < Q_HIGH_MEV else BIT["q_high"]
    sh = tag["shell"]
    if "N~50" in sh or "N=50" in sh:
        m |= BIT["shell_N50"]
    if "N~82" in sh or "N=82" in sh:
        m |= BIT["shell_N82"]
    if "N~126" in sh or "N=126" in sh:
        m |= BIT["shell_N126"]
    b2 = abs(fnum(tag, "beta2"))
    if b2 < B2_SPH:
        m |= BIT["def_sph"] | BIT["def_sph_len"]
    elif b2 < B2_DEF:
        m |= BIT["def_trans"] | BIT["def_sph_len"] | BIT["def_def_len"]
    else:
        m |= BIT["def_def"] | BIT["def_def_len"]
    if (m & BIT["mass_heavy"]) and (m & BIT["def_def"]) and (m & BIT["q_low"]):
        m |= BIT["joint_hdl"]
    ff = fnum(tag, "pcFF")
    if math.isfinite(ff) and ff >= FF_CUT:
        m |= BIT["ff50"]
    return m


def schemes():
    """(name, required-cells bitmask) for the 24-scheme grid + unconstrained."""
    base = (BIT["mass_light"] | BIT["mass_medium"] | BIT["mass_heavy"] |
            BIT["shell_N50"] | BIT["shell_N82"] | BIT["shell_N126"] |
            BIT["q_low"])
    qax = {"bin": BIT["q_rest"], "ter": BIT["q_mid"] | BIT["q_high"]}
    dax = {"strict": BIT["def_sph"] | BIT["def_def"],
           "lenient": BIT["def_sph_len"] | BIT["def_def_len"],
           "three": BIT["def_sph"] | BIT["def_def"] | BIT["def_trans"]}
    xax = {"none": 0, "hdl": BIT["joint_hdl"], "ff": BIT["ff50"],
           "both": BIT["joint_hdl"] | BIT["ff50"]}
    out = [("UNCON", 0)]
    for qn, qm in qax.items():
        for dn, dm in dax.items():
            for xn, xm in xax.items():
                out.append((f"Q{qn}-D{dn}-X{xn}", base | qm | dm | xm))
    return out


# ------------------------------------------------------------------ search core
def lam_min_arrays(S11, S12, S22):
    tr = S11 + S22
    disc = np.sqrt((S11 - S22) ** 2 + 4.0 * S12 ** 2)
    return 0.5 * (tr - disc)


def merge_top(store, cand, topn):
    """Keep topn (lam, combo) with distinct combos, descending lam."""
    seen = {c: l for l, c in store}
    for l, c in cand:
        if c not in seen or l > seen[c]:
            seen[c] = l
    best = sorted(((l, c) for c, l in seen.items()), reverse=True)[:topn]
    return best


def exhaustive(npool, k, comps, base, masks, reqs, topn, chunk, label=""):
    """Exhaustive scan of C(npool, k). comps: {mode: (a11, a12, a22)} per-pool
    arrays; base: 2x2 GT matrix; masks: uint32 per-pool cell masks; reqs: list
    of required-cell bitmasks (0 = unconstrained). Returns
    ({mode: [top list per req]}, feasible counts, total)."""
    total = math.comb(npool, k)
    it = itertools.combinations(range(npool), k)
    tops = {m: [[] for _ in reqs] for m in comps}
    feas = np.zeros(len(reqs), dtype=np.int64)
    done = 0
    while True:
        flat = np.fromiter(chain.from_iterable(islice(it, chunk)),
                           dtype=np.int64, count=-1)
        if flat.size == 0:
            break
        idx = flat.reshape(-1, k)
        orm = np.bitwise_or.reduce(masks[idx], axis=1)
        lam = {}
        for m, (a11, a12, a22) in comps.items():
            S11 = a11[idx].sum(axis=1) + base[0, 0]
            S12 = a12[idx].sum(axis=1) + base[0, 1]
            S22 = a22[idx].sum(axis=1) + base[1, 1]
            lam[m] = lam_min_arrays(S11, S12, S22)
        for si, req in enumerate(reqs):
            ok = (orm & req) == req
            nok = int(ok.sum())
            feas[si] += nok
            if nok == 0:
                continue
            for m in comps:
                lamm = np.where(ok, lam[m], -np.inf)
                nn = min(topn, nok)
                part = np.argpartition(lamm, -nn)[-nn:]
                cand = [(float(lamm[j]), tuple(int(x) for x in idx[j]))
                        for j in part if np.isfinite(lamm[j])]
                tops[m][si] = merge_top(tops[m][si], cand, topn)
        done += idx.shape[0]
        if total > 2 * chunk:
            print(f"    {label} {done/total*100:5.1f}% ({done:,}/{total:,})",
                  flush=True)
    if done != total:
        raise RuntimeError(f"enumeration mismatch: {done} != {total}")
    return tops, feas, total


# ------------------------------------------------------------------ reporting
def joint_metrics(gvecs, s2, base):
    """Joint & HL-only Fisher metrics for one set. gvecs: list of [b_vp,b_cs];
    s2: matching sigma_tot^2 list."""
    H = np.zeros((2, 2), dtype=np.float64)
    for g, v in zip(gvecs, s2):
        H += np.outer(g, g) / v
    F = base + H
    evJ = np.linalg.eigvalsh(F)
    evH = np.linalg.eigvalsh(H)
    det = float(F[0, 0] * F[1, 1] - F[0, 1] ** 2)
    sig_vp = 250.0 * math.sqrt(F[1, 1] / det) if det > 0.0 else float("inf")
    sig_cs = 125.0 * math.sqrt(F[0, 0] / det) if det > 0.0 else float("inf")
    condJ = float(evJ[1] / evJ[0]) if evJ[0] > 0.0 else float("inf")
    return dict(lam=float(evJ[0]), cond=condJ, lam_hl=float(evH[0]),
                sig_vp=sig_vp, sig_cs=sig_cs)


def bits_names(mask):
    return [c for c in CELLS if mask & BIT[c]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--functional", default="SKOP",
                    help="SKOP | UNE1 (drives paths, sigma_the, screen file)")
    ap.add_argument("--drop", default="",
                    help="extra pool exclusions on top of the monotonic "
                         "screen (comma list), e.g. emulator-health drops: "
                         "--drop Sr92 for UNE1")
    ap.add_argument("--screen-file", default=None,
                    help="override the monotonic-screen verdict file "
                         "(default: the per-functional July/registered one)")
    ap.add_argument("--sigma-the-hl", type=float, default=None,
                    help="override the HL theory error in the weights "
                         "(consumed at module level; listed here for help)")
    ap.add_argument("--no-sigma-q", action="store_true",
                    help="omit the per-nucleus Q-value error sigma_Q,i from the half-life variance "
                         "(the paper's sets were selected WITH it; this reproduces the 2026-10-19 control)")
    ap.add_argument("--sigma-the-gt", type=float, default=None,
                    help="override the GT theory error in the weights")
    ap.add_argument("--pools", default="screened,raw")
    ap.add_argument("--kmin", type=int, default=6)
    ap.add_argument("--kmax", type=int, default=9)
    ap.add_argument("--topn", type=int, default=3)
    ap.add_argument("--chunk", type=int, default=1_000_000)
    ap.add_argument("--skip-validation", action="store_true")
    ap.add_argument("--loc-lo", type=float, default=LOC_WIN[0],
                    help="localized-slope vpair window lower edge "
                         "(default %(default)s; use -320 with --loc-hi -176 "
                         "for the Set-A-posterior-anchored window)")
    ap.add_argument("--loc-hi", type=float, default=LOC_WIN[1],
                    help="localized-slope vpair window upper edge")
    ap.add_argument("--suffix", default="",
                    help="output filename suffix, e.g. _locA -> "
                         "constrained_set_search_locA.{csv,txt}")
    ap.add_argument("--gt-set", default=",".join(GT_SET),
                    help="comma-separated GT base nuclei (default: all 8 "
                         "measured centroids, the consistent rule); pass the "
                         "historical five Pb208,Sn132,Te130,Zr90,Ge76 to "
                         "reproduce the v3 artifacts")
    args = ap.parse_args()
    loc_win = (args.loc_lo, args.loc_hi)
    gt_set = tuple(args.gt_set.split(","))
    out_csv = OUT_CSV.replace(".csv", f"{args.suffix}.csv")
    out_txt = OUT_TXT.replace(".txt", f"{args.suffix}.txt")

    fish = read_csv_dict(FISHER_CSV)
    tags = read_csv_dict(TAG_CSV)
    trend = read_csv_dict(TREND_CSV) if os.path.exists(TREND_CSV) else {}
    global SCREEN_TXT
    if args.screen_file:
        SCREEN_TXT = (args.screen_file if os.path.isabs(args.screen_file)
                      else os.path.join(HERE, args.screen_file))
    drop = load_screen() | {n for n in args.drop.split(",") if n}
    raw_names = list(fish.keys())
    # 2026-10-12: restrict the raw pool to the registered per-functional
    # calibration pool config.HL_SETS['POOL40_<F>'] (intersection, CSV order)
    pool40 = config.HL_SETS["POOL40_" + FUNCTIONAL]
    pool_dropped = [n for n in raw_names if n not in pool40]
    raw_names = [n for n in raw_names if n in pool40]
    print(f"POOL40_{FUNCTIONAL}: kept {len(raw_names)}, dropped "
          f"{', '.join(pool_dropped) if pool_dropped else 'none'}", flush=True)
    assert len(raw_names) == 40, f"expected 40-pool, got {len(raw_names)}"
    scr_names = [n for n in raw_names if n not in drop]
    B = gt_base(gt_set)

    # per-nucleus Q-value error of the likelihood, sigma_Q,i = 5 sigma_Q / (Q_corr,i ln 10) in decades,
    # added in quadrature to the half-life variance exactly as in bayes_mcmc.py (paper Eq. for sigma^2)
    if _fk.no_sigma_q:
        SIGQ = {n: 0.0 for n in raw_names}
    else:
        SIGQ = {l.split("=")[0].strip(): float(l.split("=")[1]) for l in open(SIGQ_FILE) if "=" in l and not l.startswith("#")}
        assert all(n in SIGQ for n in raw_names), [n for n in raw_names if n not in SIGQ]
    # per-nucleus quantities (raw-pool order)
    info = {}
    for n in raw_names:
        r, t = fish[n], tags[n]
        sig_exp = fnum(r, "sigma_log10")
        sem = fnum(r, "sem_log10")
        sem_eff = sem if math.isfinite(sem) else 0.0
        s2 = sig_exp ** 2 + sem_eff ** 2 + SIGMA_THE_HL ** 2 + SIGQ[n] ** 2
        bl_vp, bl_cs, n_loc, R2_loc = fit_slopes(n, *loc_win)
        # "has emulator" = deployed params_best in results/ (Rn226 is deployed
        # without a CV parity file, so the CV file alone under-counts)
        has_emu = (os.path.exists(
            os.path.join(CVROOT, n, "sigma_em_parity_half_life_val.txt"))
            or bool(glob.glob(os.path.join(
                RESROOT, n, "params_best_n*_only_HL.txt"))))
        tr = trend.get(n, {})
        pref = fnum(tr, "vpair_at_exp") if (
            tr.get("covered") == "True" and fnum(tr, "nmatch") >= 10) else float("nan")
        info[n] = dict(
            g=np.array([fnum(r, "b_vp"), fnum(r, "b_cs")]),
            gl=np.array([bl_vp, bl_cs]), n_loc=n_loc, R2_loc=R2_loc,
            s2=s2, mask=cell_mask(t), has_emu=has_emu, pref=pref,
            R2=fnum(r, "R2"))

    sch = schemes()
    reqs = [m for _, m in sch]

    txt = []
    txt.append("=" * 78)
    txt.append(f"Constrained exhaustive calibration-set search -- "
               f"{_FDEF['title']} 2D (vpair, cs0)")
    txt.append(f"date {_FDEF['date']} | sigma_the_HL={SIGMA_THE_HL} "
               f"sigma_the_GT={SIGMA_THE_GT} | GT variance includes the CV emulator error sigma_em_MeV")
    txt.append(f"GT base (fixed {'+'.join(gt_set)}): F_vp={B[0,0]:.3f} "
               f"F_cs={B[1,1]:.3f} F_x={B[0,1]:.3f}")
    txt.append(f"localized-slope window: vpair in [{loc_win[0]:.0f}, {loc_win[1]:.0f}]")
    txt.append(f"raw pool = POOL40_{FUNCTIONAL}: {len(raw_names)} (dropped from "
               f"the fisher CSV: {', '.join(pool_dropped) if pool_dropped else 'none'})")
    txt.append(f"screened pool: {len(scr_names)} (dropped: {', '.join(sorted(drop))})")
    txt.append("")

    # ---- localized-slope table
    txt.append("Localized vs global slopes (b_vp, b_cs per half-box):")
    txt.append(f"{'nuc':<7}{'b_vp_g':>8}{'b_vp_loc':>9}{'b_cs_g':>8}"
               f"{'b_cs_loc':>9}{'n_loc':>6}{'R2_loc':>7}")
    for n in raw_names:
        i = info[n]
        txt.append(f"{n:<7}{i['g'][0]:>8.3f}{i['gl'][0]:>9.3f}{i['g'][1]:>8.3f}"
                   f"{i['gl'][1]:>9.3f}{i['n_loc']:>6d}{i['R2_loc']:>7.2f}")
    txt.append("")

    # ---- validation
    if not args.skip_validation:
        rng = np.random.default_rng(20260709)
        # (1) analytic lambda_min vs eigvalsh
        worst = 0.0
        for _ in range(500):
            kk = int(rng.integers(6, 10))
            sub = rng.choice(len(raw_names), size=kk, replace=False)
            F = B.copy()
            for j in sub:
                i = info[raw_names[j]]
                F += np.outer(i["g"], i["g"]) / i["s2"]
            l_ana = lam_min_arrays(np.array([F[0, 0]]), np.array([F[0, 1]]),
                                   np.array([F[1, 1]]))[0]
            worst = max(worst, abs(l_ana - np.linalg.eigvalsh(F)[0]))
        # (2) brute force, small pool, one constrained scheme + UNCON
        # 2026-10-12: 14 consecutive names of the restricted pool (cyclic
        # window, first offset where the constrained scheme has at least one
        # feasible 5-set; offset 0 = raw_names[:14]). The previous choice
        # (scr_names[:14]) made the check fail with an empty top list when
        # the window held no N~126 nucleus (SKOP POOL40: Pt200 excluded).
        _req_v = dict(sch)["Qbin-Dstrict-Xhdl"]
        bp = None
        for off in range(len(raw_names)):
            cand = [raw_names[(off + j) % len(raw_names)] for j in range(14)]
            cm_ = [info[n]["mask"] for n in cand]
            if any((cm_[c[0]] | cm_[c[1]] | cm_[c[2]] | cm_[c[3]] | cm_[c[4]])
                   & _req_v == _req_v
                   for c in itertools.combinations(range(14), 5)):
                bp = cand
                break
        if bp is None:
            raise SystemExit("validation: Qbin-Dstrict-Xhdl infeasible at k=5 "
                             "in every 14-name window of the pool")
        print(f"brute-force validation pool (offset {off}): {', '.join(bp)}",
              flush=True)
        bcomp = {"g": tuple(np.array([info[n]["g"][a] * info[n]["g"][b] / info[n]["s2"]
                                      for n in bp])
                            for a, b in ((0, 0), (0, 1), (1, 1)))}
        bmask = np.array([info[n]["mask"] for n in bp], dtype=np.uint32)
        breq = [0, dict(sch)["Qbin-Dstrict-Xhdl"]]
        btop, _, _ = exhaustive(len(bp), 5, bcomp, B, bmask, breq, 1, 10 ** 6)
        ok_all = True
        for si, req in enumerate(breq):
            best_l, best_c = -np.inf, None
            for combo in itertools.combinations(range(len(bp)), 5):
                cm = 0
                F = B.copy()
                for j in combo:
                    cm |= int(bmask[j])
                    i = info[bp[j]]
                    F += np.outer(i["g"], i["g"]) / i["s2"]
                if (cm & req) != req:
                    continue
                l = float(np.linalg.eigvalsh(F)[0])
                if l > best_l:
                    best_l, best_c = l, combo
            got_l, got_c = btop["g"][si][0]
            ok_all &= (got_c == best_c and abs(got_l - best_l) < 1e-10)
        # (3) parse check: exp-only F_vp vs CSV column
        perr = 0.0
        for n in raw_names[:5]:
            r = fish[n]
            f_re = fnum(r, "b_vp") ** 2 / fnum(r, "sigma_log10") ** 2
            perr = max(perr, abs(f_re - fnum(r, "F_vp")) / fnum(r, "F_vp"))
        txt.append("VALIDATION")
        txt.append(f"  analytic lambda_min vs eigvalsh (500 random sets): "
                   f"max |diff| = {worst:.2e}  {'PASS' if worst < 1e-10 else 'FAIL'}")
        txt.append(f"  brute-force small-pool (n=14,k=5, UNCON + Qbin-Dstrict-Xhdl): "
                   f"{'PASS' if ok_all else 'FAIL'}  [pool offset {off}: "
                   f"{', '.join(bp)}]")
        txt.append(f"  CSV parse check exp-only F_vp: max rel err = {perr:.1e}  "
                   f"{'PASS' if perr < 1e-2 else 'FAIL'}")
        txt.append("")
        if worst >= 1e-10 or not ok_all:
            print("\n".join(txt))
            raise SystemExit("VALIDATION FAILED -- aborting search")

    # ---- reference sets (per functional; see REF_SETS above)
    ref_rows = []
    for lbl, names in ((lbl, sorted(config.select_hl_nuclei(RP.SET_KEYS[FUNCTIONAL][key])))
                       for lbl, key in REF_SETS):
        cov = 0
        for n in names:
            cov |= info[n]["mask"]
        for mode in ("global", "loc"):
            key = "g" if mode == "global" else "gl"
            met = joint_metrics([info[n][key] for n in names],
                                [info[n]["s2"] for n in names], B)
            miss = {sn: ", ".join(bits_names(req & ~cov))
                    for sn, req in sch if sn != "UNCON" and (cov & req) != req}
            ref_rows.append((lbl, mode, len(names), names, met, miss))
    txt.append("REFERENCE SETS (fixed candidates, same Fisher conventions)")
    for lbl, mode, k, names, met, miss in ref_rows:
        txt.append(f"  {lbl}-{k} [{mode:>6}] lam_min={met['lam']:.3f} "
                   f"cond={met['cond']:.1f} sig_vp={met['sig_vp']:.1f} "
                   f"sig_cs={met['sig_cs']:.1f}   {{{', '.join(names)}}}")
        if mode == "global":
            mm = sorted({v for v in miss.values()})
            txt.append(f"       cells missed (over schemes requiring them): "
                       f"{'; '.join(mm) if mm else 'none -- covers every scheme'}")
            if MCMC_REF and lbl in MCMC_REF:
                ref = MCMC_REF[lbl]
                txt.append(f"       {MCMC_REF_LBL}: vpair {ref['vpair']:.1f}"
                           f" +/- {ref['sd']:.1f}, fitted sigma_the {ref['sigma_the']:.2f}")
            else:
                txt.append(f"       {MCMC_REF_LBL}: n/a")
    txt.append("")

    # ---- the search
    pools = {"screened": scr_names, "raw": raw_names}
    csv_rows = []
    results = {}
    for pool in [p.strip() for p in args.pools.split(",")]:
        names = pools[pool]
        comps = {}
        for mode, key in (("global", "g"), ("loc", "gl")):
            G = np.array([info[n][key] for n in names])
            W = np.array([info[n]["s2"] for n in names])
            comps[mode] = (G[:, 0] ** 2 / W, G[:, 0] * G[:, 1] / W,
                           G[:, 1] ** 2 / W)
        masks = np.array([info[n]["mask"] for n in names], dtype=np.uint32)
        for k in range(args.kmin, args.kmax + 1):
            print(f"pool={pool} k={k}: C({len(names)},{k}) = "
                  f"{math.comb(len(names), k):,}", flush=True)
            tops, feas, total = exhaustive(
                len(names), k, comps, B, masks, reqs, args.topn, args.chunk,
                label=f"{pool} k={k}")
            results[(pool, k)] = (tops, feas, total, names)
            for mode in ("global", "loc"):
                key = "g" if mode == "global" else "gl"
                unc = tops[mode][0][0]
                unc_names = [names[j] for j in unc[1]]
                unc_cov = 0
                for n in unc_names:
                    unc_cov |= info[n]["mask"]
                for si, (sn, req) in enumerate(sch):
                    if not tops[mode][si]:
                        csv_rows.append(dict(pool=pool, mode=mode, k=k, scheme=sn,
                                             feasible=0, note="INFEASIBLE"))
                        continue
                    lam, combo = tops[mode][si][0]
                    snames = [names[j] for j in combo]
                    met = joint_metrics([info[n][key] for n in snames],
                                        [info[n]["s2"] for n in snames], B)
                    prefs = [info[n]["pref"] for n in snames
                             if math.isfinite(info[n]["pref"])]
                    span = (max(prefs) - min(prefs)) if len(prefs) >= 2 else float("nan")
                    need = [n for n in snames if not info[n]["has_emu"]]
                    forced = [n for n in snames if n not in unc_names]
                    binding = bits_names(req & ~unc_cov)
                    csv_rows.append(dict(
                        pool=pool, mode=mode, k=k, scheme=sn,
                        feasible=int(feas[si]), lam=lam, lam_unc=unc[0],
                        price=lam / unc[0], lam_hl=met["lam_hl"],
                        cond=met["cond"], sig_vp=met["sig_vp"],
                        sig_cs=met["sig_cs"],
                        span_pref=span, n_pref=len(prefs),
                        set="+".join(snames), forced="+".join(forced),
                        binding="+".join(binding),
                        needs_emu="+".join(need)))

    # ---- write csv
    cols = ["pool", "mode", "k", "scheme", "feasible", "lam", "lam_unc",
            "price", "lam_hl", "cond", "sig_vp", "sig_cs", "span_pref",
            "n_pref", "set", "forced", "binding", "needs_emu", "note"]
    with open(out_csv, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in csv_rows:
            f.write(",".join(
                (f"{r[c]:.4f}" if isinstance(r.get(c), float) else str(r.get(c, "")))
                for c in cols) + "\n")

    # ---- txt: headline tables
    for (pool, k), (tops, feas, total, names) in sorted(results.items()):
        txt.append("-" * 78)
        txt.append(f"POOL={pool} ({len(names)})  k={k}  combos={total:,}")
        for mode in ("global", "loc"):
            key = "g" if mode == "global" else "gl"
            txt.append(f"  [{mode} slopes]")
            unc_l, unc_c = tops[mode][0][0]
            txt.append(f"    UNCON  lam={unc_l:8.3f}            "
                       f"{{{', '.join(names[j] for j in unc_c)}}}")
            for si, (sn, req) in enumerate(sch):
                if sn == "UNCON":
                    continue
                if not tops[mode][si]:
                    txt.append(f"    {sn:<22} INFEASIBLE")
                    continue
                lam, combo = tops[mode][si][0]
                txt.append(f"    {sn:<22} lam={lam:8.3f} price={lam/unc_l:5.3f} "
                           f"feas={int(feas[si]):>10,}  "
                           f"{{{', '.join(names[j] for j in combo)}}}")
        txt.append("")

    # ---- txt: headline top-3 + scheme-axis effects at (screened, global, k=8)
    hk = min(max(8, args.kmin), args.kmax)
    if ("screened", hk) in results:
        tops, feas, total, names = results[("screened", hk)]
        txt.append("=" * 78)
        txt.append(f"HEADLINE (screened, global slopes, k={hk}) -- top-3 per scheme")
        for si, (sn, req) in enumerate(sch):
            for rank, (lam, combo) in enumerate(tops["global"][si], 1):
                txt.append(f"  {sn:<22} #{rank} lam={lam:8.3f}  "
                           f"{{{', '.join(names[j] for j in combo)}}}")
        txt.append("")
        txt.append("Scheme-axis effect on price (screened, global, k=8):")
        rows8 = [r for r in csv_rows if r["pool"] == "screened"
                 and r["mode"] == "global" and r["k"] == hk
                 and r["scheme"] != "UNCON" and "price" in r]
        for axis, vals in (("Q", ("Qbin", "Qter")),
                           ("D", ("Dstrict", "Dlenient", "Dthree")),
                           ("X", ("Xnone", "Xhdl", "Xff", "Xboth"))):
            for v in vals:
                ps = [r["price"] for r in rows8 if v in r["scheme"]]
                if not ps:   # every scheme on this axis infeasible (small pool)
                    txt.append(f"  {v:<9} price min/med/max = n/a (no feasible set)")
                    continue
                txt.append(f"  {v:<9} price min/med/max = "
                           f"{min(ps):.3f} / {sorted(ps)[len(ps)//2]:.3f} / {max(ps):.3f}")
        txt.append("")

    with open(out_txt, "w") as f:
        f.write("\n".join(txt) + "\n")
    print(f"saved: {out_csv}")
    print(f"saved: {out_txt}")


if __name__ == "__main__":
    main()
