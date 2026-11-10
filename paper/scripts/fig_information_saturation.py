#!/usr/bin/env python3
"""Information-saturation figure for the paper (agreed 2026-10-12).

For each functional: the E-criterion lambda_min of the 2x2 information matrix
for (V0, C1s) -- the eight GT nuclei always included -- as a function of the
number k of half-life nuclei, for the best (unconstrained) set of each size.
Reference levels: GT only (k = 0), Set A (ten nuclei), the full pool (k = 40).
The adopted k = 8 set (chosen under the coverage rule) is a marked point.

Information matrix, identical to design/constrained_set_search.py:
  HL nucleus i:  g_i = (b_vp, b_cs)  whole-box OLS slopes of log10 T per
                 half-box of the scaled parameters (<f>_hl_fisher.csv),
                 sigma_tot,i^2 = sigma_exp,log10^2 + sigma_em^2 + sigma_the,HL^2
  GT nucleus j:  g_j = (b_vp, b_cs) in MeV per half-box (<f>_gt_fisher.csv),
                 sigma_tot,j^2 = sigma_exp,MeV^2 + sigma_the,GT^2
  M(C) = sum_j g_j g_j^T / sigma_j^2  +  sum_{i in C} g_i g_i^T / sigma_i^2
  criterion = lambda_min[M(C)]
The theory errors are FIXED at the Set-A anchor medians (design stage).

Best set of size k:
  k <= KEX            exact enumeration of all C(40, k) subsets
  k >= 40 - KEX       exact enumeration of the 40 - k removed nuclei
  otherwise           swap search: start from the better of greedy-forward and
                      greedy-backward, then accept the best improving pairwise
                      swap until none remains. Checked against the exact
                      result at k = KEX and k = 40 - KEX (printed).

Outputs: ../figs/fig_information_saturation.{pdf,png} and a text table
../figs/fig_information_saturation.txt with every number plotted.
"""
import itertools
import math
import os
import sys
from itertools import chain, islice

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paper_style as ps   # LaTeX fonts, shared colours, ps.save()

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
import repo_paths as RP  # noqa: E402  (all data paths of the repository)
sys.path.insert(0, RP.CALIBRATION)
import config

ANA = RP.DESIGN
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "figs",
                   "fig_information_saturation")

# ---- knobs ------------------------------------------------------------------
FUNCTIONALS = ["SKOP", "SLY4", "HFB1"]
COLORS = {"SKOP": "#0072B2", "SLY4": "#D55E00", "HFB1": "#009E73"}
MARKERS = {"SKOP": "o", "SLY4": "s", "HFB1": "^"}
LABELS = {"SKOP": r"SkO$'$", "SLY4": "SLy4", "HFB1": "UNEDF1-HFB"}
# Set-A anchor posterior medians (Normal likelihood, Set A, Sec. on the baseline calibration)
SIGMA_THE_HL = {"SKOP": 0.464, "SLY4": 0.475, "HFB1": 0.406}   # Set A with FINAL settings, Normal (2026-10-19)
SIGMA_THE_GT = {"SKOP": 1.197, "SLY4": 0.820, "HFB1": 0.738}
POOLS = {"SKOP": "POOL40_SKOP", "SLY4": "POOL40_SLY4", "HFB1": "POOL40_HFB1"}
# 2026-11-05 (user): the per-nucleus Q-value error of the likelihood, sigma_Q,i = 5 sigma_Q / (Q_corr,i ln 10) in
# decades (calibration/analysis/sigma_q_<F>.dat), is added in quadrature to the half-life variance of the
# information matrix, as in constrained_set_search_sigq_TEST_20261105.py. Set SIGMA_Q_FILE[func] = None to omit it.
SIGMA_Q_FILE = {f: os.path.join(RP.ERRORS, f"sigma_q_{f}.dat") for f in FUNCTIONALS}
# 2026-11-05 (user): SLOPES = "local" builds the curve with the half-life slopes fitted inside the Set-A V0 window
# (the matrix that selects the L8 sets), read from the "Localized vs global slopes" table of the set-search output
# LOCAL_SLOPE_TXT (windows SkO' [-325,-199], SLy4 [-370,-197], UNEDF1-HFB [-322,-143]; GT slopes stay global, as in
# constrained_set_search.py). "global" = the b_vp/b_cs columns of <f>_hl_fisher.csv (G8 construction).
SLOPES = "local"
LOCAL_SLOPE_TXT = {"SKOP": os.path.join(ANA, "constrained_set_search_setsV7sigq.txt"),
                   "SLY4": os.path.join(ANA, "constrained_set_search_SLY4_setsV7sigq.txt"),
                   "HFB1": os.path.join(ANA, "constrained_set_search_HFB1_setsV7sigq.txt")}
MARKED = {"global": {"SKOP": "G10_SKOP", "SLY4": "G10_SLY4", "HFB1": "G10_HFB1"},
          "local": {"SKOP": "L10_SKOP", "SLY4": "L10_SLY4", "HFB1": "G10_HFB1"}}[SLOPES]   # filled symbol at m = 10 (ten-nucleus sets adopted 2026-11-05 evening)
GT_SET = ["Pb208", "Sn132", "Te130", "Zr90", "Ge76", "Sn112", "Nd150", "Ca48"]
KEX = 8                                   # exact enumeration up to this k (and from 40-KEX)
CHUNK = 1_000_000
NORMALIZE = True                          # y = lambda_min / lambda_min(full pool)
SHOW_UNCONSTRAINED = False                # dashed reference curve without the coverage rule
FIGWIDTH = 3.4
# ------------------------------------------------------------------------------


# ---- coverage rule, identical to constrained_set_search.py (scheme Qter-Dthree-Xboth)
Q_LOW_MEV, Q_HIGH_MEV = 3.0, 6.0
B2_SPH, B2_DEF = 0.10, 0.20
FF_CUT = 50.0
CELLS = ["mass_light", "mass_medium", "mass_heavy",
         "q_low", "q_rest", "q_mid", "q_high",
         "shell_N50", "shell_N82", "shell_N126",
         "def_sph", "def_def", "def_trans",
         "def_sph_len", "def_def_len",
         "joint_hdl", "ff50"]
BIT = {c: 1 << i for i, c in enumerate(CELLS)}
REQUIRED = (BIT["mass_light"] | BIT["mass_medium"] | BIT["mass_heavy"] |
            BIT["shell_N50"] | BIT["shell_N82"] | BIT["shell_N126"] |
            BIT["q_low"] | BIT["q_mid"] | BIT["q_high"] |
            BIT["def_sph"] | BIT["def_def"] | BIT["def_trans"] |
            BIT["joint_hdl"] | BIT["ff50"])
TAG_CSV = {"SKOP": "nuclei_tag_table.csv", "SLY4": "nuclei_tag_table_SLY4.csv",
           "HFB1": "nuclei_tag_table_HFB1.csv"}


def cell_mask(tag):
    m = 0
    m |= BIT["mass_" + tag["mass_class"]]
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
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return float("nan")


def lam_min(S11, S12, S22):
    tr = S11 + S22
    disc = np.sqrt((S11 - S22) ** 2 + 4.0 * S12 ** 2)
    return 0.5 * (tr - disc)


def read_local_slopes(path):
    """b_vp_loc, b_cs_loc per nucleus from the set-search text output."""
    out, on = {}, False
    for ln in open(path):
        if ln.startswith("Localized vs global slopes"):
            on = True
            continue
        if on:
            t = ln.split()
            if not t:
                if out:
                    break
                continue
            if t[0] == "nuc":
                continue
            try:
                out[t[0]] = (float(t[2]), float(t[4]))
            except (ValueError, IndexError):
                break
    assert out, path
    return out


def read_sigq(path):
    """NUC=VALUE lines (decades) -> dict."""
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        k, v = ln.split("=", 1)
        out[k.strip()] = float(v)
    return out


def load(func):
    """Per-nucleus information terms (a11, a12, a22) for the pool, GT base,
    names, and the index sets of Set A and the adopted set."""
    hl = read_csv_dict(os.path.join(ANA, f"{func.lower()}_hl_fisher.csv"))
    gt = read_csv_dict(os.path.join(ANA, f"{func.lower()}_gt_fisher.csv"))
    pool = [n for n in hl if n in config.select_hl_nuclei(POOLS[func])]
    assert len(pool) == 40, (func, len(pool))
    sigq = read_sigq(SIGMA_Q_FILE[func]) if SIGMA_Q_FILE[func] else {}
    loc = read_local_slopes(LOCAL_SLOPE_TXT[func]) if SLOPES == "local" else {}
    if loc:
        missing = [n for n in pool if n not in loc]
        assert not missing, (func, "no local slopes for", missing)
    if sigq:
        missing = [n for n in pool if n not in sigq]
        assert not missing, (func, "no sigma_Q for", missing)
    a11, a12, a22 = [], [], []
    for n in pool:
        r = hl[n]
        g = np.array(loc[n]) if loc else np.array([fnum(r, "b_vp"), fnum(r, "b_cs")])
        sem = fnum(r, "sem_log10")
        sem = sem if math.isfinite(sem) else 0.0
        s2 = fnum(r, "sigma_log10") ** 2 + sem ** 2 + SIGMA_THE_HL[func] ** 2 + sigq.get(n, 0.0) ** 2
        a11.append(g[0] * g[0] / s2)
        a12.append(g[0] * g[1] / s2)
        a22.append(g[1] * g[1] / s2)
    B = np.zeros((2, 2))
    for n in GT_SET:
        r = gt[n]
        g = np.array([fnum(r, "b_vp"), fnum(r, "b_cs")])
        sem = fnum(r, "sigma_em_MeV")          # CV emulator error of the centroid [MeV] (2026-11-10), as in the set search
        sem = sem if math.isfinite(sem) else 0.0
        s2 = fnum(r, "sigma_MeV") ** 2 + sem ** 2 + SIGMA_THE_GT[func] ** 2
        B += np.outer(g, g) / s2
    comps = (np.array(a11), np.array(a12), np.array(a22))
    tags = read_csv_dict(os.path.join(ANA, TAG_CSV[func]))
    masks = np.array([cell_mask(tags[n]) for n in pool], dtype=np.int64)
    load.masks = masks
    setA = [pool.index(n) for n in config.select_hl_nuclei("A")]
    adopted = ([pool.index(n) for n in config.select_hl_nuclei(MARKED[func])]
               if MARKED[func] else None)
    return pool, comps, B, setA, adopted


def lam_of(idx, comps, B):
    a11, a12, a22 = comps
    idx = list(idx)
    return float(lam_min(a11[idx].sum() + B[0, 0], a12[idx].sum() + B[0, 1],
                         a22[idx].sum() + B[1, 1]))


def exhaustive(n, k, comps, B, complement=False, masks=None):
    """Best subset of size k (or best set of n-k removed if complement).
    With masks: only sets whose cells cover REQUIRED (coverage rule)."""
    a11, a12, a22 = comps
    if complement:
        T = (a11.sum() + B[0, 0], a12.sum() + B[0, 1], a22.sum() + B[1, 1])
        kk = n - k
    else:
        kk = k
    if kk == 0:
        idx = () if not complement else tuple(range(n))
        return lam_of(idx, comps, B), idx
    it = itertools.combinations(range(n), kk)
    best, bidx = -np.inf, None
    while True:
        flat = np.fromiter(chain.from_iterable(islice(it, CHUNK)),
                           dtype=np.int64, count=-1)
        if flat.size == 0:
            break
        I = flat.reshape(-1, kk)
        if complement:
            lam = lam_min(T[0] - a11[I].sum(1), T[1] - a12[I].sum(1),
                          T[2] - a22[I].sum(1))
        else:
            lam = lam_min(a11[I].sum(1) + B[0, 0], a12[I].sum(1) + B[0, 1],
                          a22[I].sum(1) + B[1, 1])
        if masks is not None:
            if complement:
                # feasible iff every required cell keeps at least one nucleus
                req_bits = [b for b in range(len(CELLS)) if REQUIRED >> b & 1]
                has = np.array([(masks >> b) & 1 for b in req_bits]).T  # (n, nreq)
                left = has.sum(0)[None, :] - has[I].sum(1)              # (chunk, nreq)
                ok = (left > 0).all(1)
            else:
                orm = np.bitwise_or.reduce(masks[I], axis=1)
                ok = (orm & REQUIRED) == REQUIRED
            lam = np.where(ok, lam, -np.inf)
        j = int(np.argmax(lam))
        if lam[j] > best:
            best, bidx = float(lam[j]), tuple(int(x) for x in I[j])
    if bidx is None:
        return float("nan"), None
    if complement:
        bidx = tuple(i for i in range(n) if i not in set(bidx))
    return best, bidx


def swap_search(n, k, comps, B):
    """Greedy forward / backward start, then best-improving pairwise swaps."""
    def greedy_forward():
        s = []
        for _ in range(k):
            cand = [(lam_of(s + [j], comps, B), j) for j in range(n) if j not in s]
            s.append(max(cand)[1])
        return s

    def greedy_backward():
        s = list(range(n))
        while len(s) > k:
            cand = [(lam_of([x for x in s if x != j], comps, B), j) for j in s]
            s.remove(max(cand)[1])
        return s

    starts = [greedy_forward(), greedy_backward()]
    best_s, best_l = None, -np.inf
    for s in starts:
        s = list(s)
        l = lam_of(s, comps, B)
        improved = True
        while improved:
            improved = False
            bl, bswap = l, None
            for i in s:
                for j in range(n):
                    if j in s:
                        continue
                    t = [x for x in s if x != i] + [j]
                    lt = lam_of(t, comps, B)
                    if lt > bl + 1e-12:
                        bl, bswap = lt, (i, j)
            if bswap:
                s = [x for x in s if x != bswap[0]] + [bswap[1]]
                l = bl
                improved = True
        if l > best_l:
            best_l, best_s = l, tuple(sorted(s))
    return best_l, best_s


def nested_curve(pool, comps, B, adopted, log, func):
    """Curve through the adopted set: for k < |adopted| the best subset of the
    adopted set (exact); for k > |adopted| greedy forward additions from the
    rest of the pool (the nucleus adding the most information at each step)."""
    n = len(pool)
    m = len(adopted)
    ks, lams, sets = [0], [lam_of((), comps, B)], [()]
    for k in range(1, m):
        best = max((lam_of(c, comps, B), c) for c in itertools.combinations(adopted, k))
        ks.append(k); lams.append(best[0]); sets.append(best[1])
    s = list(adopted)
    ks.append(m); lams.append(lam_of(s, comps, B)); sets.append(tuple(s))
    while len(s) < n:
        cand = max((lam_of(s + [j], comps, B), j) for j in range(n) if j not in s)
        s.append(cand[1])
        ks.append(len(s)); lams.append(cand[0]); sets.append(tuple(s))
    log.append("# nested curve through the adopted set: k  lambda_min  set "
               "(k<m exact subset of the marked set; k>m greedy additions)")
    for k, l, idx in zip(ks, lams, sets):
        log.append(f"{func}-nested {k:2d} {l:10.4f} {' '.join(pool[i] for i in sorted(idx))}")
    return np.array(ks), np.array(lams)


def feasible(idx, masks):
    return (int(np.bitwise_or.reduce(masks[list(idx)])) & REQUIRED) == REQUIRED


def swap_search_constrained(n, k, comps, B, masks, start):
    """Best-improving pairwise swaps among feasible sets, from a feasible start
    (the adopted set plus greedy additions)."""
    s = list(start)
    while len(s) > k:   # start larger than k (k = 9 with the ten-nucleus marked set): drop the nucleus whose removal costs least, keeping the coverage rule
        cand = max((lam_of([x for x in s if x != i], comps, B), i) for i in s
                   if feasible([x for x in s if x != i], masks))
        s.remove(cand[1])
    while len(s) < k:
        cand = max((lam_of(s + [j], comps, B), j) for j in range(n) if j not in s)
        s.append(cand[1])
    assert feasible(s, masks)
    l = lam_of(s, comps, B)
    improved = True
    while improved:
        improved = False
        bl, bswap = l, None
        for i in s:
            for j in range(n):
                if j in s:
                    continue
                t = [x for x in s if x != i] + [j]
                if not feasible(t, masks):
                    continue
                lt = lam_of(t, comps, B)
                if lt > bl + 1e-12:
                    bl, bswap = lt, (i, j)
        if bswap:
            s = [x for x in s if x != bswap[0]] + [bswap[1]]
            l = bl
            improved = True
    return l, tuple(sorted(s))


def constrained_curve(pool, comps, B, adopted, log, func):
    """Best set of each size k under the coverage rule (nan where infeasible)."""
    n = len(pool)
    masks = load.masks
    ks, lams, sets = [], [], []
    for k in range(1, n + 1):
        if k <= KEX:
            l, idx = exhaustive(n, k, comps, B, masks=masks)
        elif k >= n - KEX:
            l, idx = exhaustive(n, k, comps, B, complement=True, masks=masks)
        else:
            l, idx = swap_search_constrained(n, k, comps, B, masks, adopted)
        ks.append(k); lams.append(l); sets.append(idx)
        print(f"  {func} constrained k={k:2d} lam={l:9.4f}  "
              f"{' '.join(pool[i] for i in sorted(idx)) if idx else 'infeasible'}", flush=True)
    log.append("# constrained curve (coverage rule Qter-Dthree-Xboth at every k; "
               "k<=8 and k>=32 exact, else swap search from the adopted set): k lambda_min set")
    for k, l, idx in zip(ks, lams, sets):
        log.append(f"{func}-constrained {k:2d} {l:10.4f} "
                   f"{' '.join(pool[i] for i in sorted(idx)) if idx else 'infeasible'}")
    return np.array(ks), np.array(lams)


def curve(func, log):
    pool, comps, B, setA, adopted = load(func)
    n = len(pool)
    lam_gt = lam_of((), comps, B)
    lam_full = lam_of(range(n), comps, B)
    lam_A = lam_of(setA, comps, B)
    lam_ad = lam_of(adopted, comps, B) if adopted else float("nan")
    log.append(f"# {func}: slopes {SLOPES}{' (' + LOCAL_SLOPE_TXT[func] + ')' if SLOPES == 'local' else ''}; marked set {MARKED[func]}; sigma_the HL {SIGMA_THE_HL[func]} GT {SIGMA_THE_GT[func]}; sigma_Q,i from {SIGMA_Q_FILE[func]}; "
               f"GT-only {lam_gt:.4f}; Set A {lam_A:.4f}; full pool {lam_full:.4f}; "
               f"marked set {MARKED[func]} {lam_ad:.4f}")
    ks, lams, sets, methods = [0], [lam_gt], [()], ["exact"]
    checks = {}
    for k in range(1, n + 1):
        if k <= KEX:
            l, idx = exhaustive(n, k, comps, B)
            m = "exact"
        elif k >= n - KEX:
            l, idx = exhaustive(n, k, comps, B, complement=True)
            m = "exact"
        else:
            l, idx = swap_search(n, k, comps, B)
            m = "swap"
        if k in (KEX, n - KEX):          # cross-check the swap search
            ls, _ = swap_search(n, k, comps, B)
            checks[k] = (l, ls)
        ks.append(k); lams.append(l); sets.append(idx); methods.append(m)
        print(f"  {func} k={k:2d} lam={l:9.4f} [{m}]  "
              f"{' '.join(pool[i] for i in sorted(idx))}", flush=True)
    for k, (le, ls) in checks.items():
        log.append(f"# {func}: swap-search check at k={k}: exact {le:.6f} swap {ls:.6f} "
                   f"({'OK' if abs(le - ls) < 1e-6 else 'DIFFERS'})")
    log.append("# k  lambda_min  method  set")
    for k, l, idx, m in zip(ks, lams, sets, methods):
        log.append(f"{func} {k:2d} {l:10.4f} {m:5s} {' '.join(pool[i] for i in sorted(idx))}")
    kn, ln = nested_curve(pool, comps, B, adopted, log, func) if adopted else (None, None)
    kc, lc = constrained_curve(pool, comps, B, adopted, log, func) if adopted else (None, None)
    return dict(k=np.array(ks), lam=np.array(lams), gt=lam_gt, full=lam_full,
                A=lam_A, adopted=lam_ad, nA=len(setA), nad=len(adopted) if adopted else None,
                kn=kn, lamn=ln, kc=kc, lamc=lc)


def main():
    if "--replot" in sys.argv:
        return replot()
    log = ["# Information-saturation curve data (fig_information_saturation.py)",
           "# lambda_min of the (V0, C1s) information matrix, GT nuclei fixed, "
           "theory errors fixed at the Set-A anchor medians."]
    res = {}
    for func in FUNCTIONALS:
        if SIGMA_THE_HL[func] is None:
            print("no anchor fit for", func, "- skipped")
            continue
        res[func] = curve(func, log)

    draw(res)
    with open(OUT + ".txt", "w") as f:
        f.write("\n".join(log) + "\n")


def draw(res):
    fig, ax = plt.subplots(figsize=(FIGWIDTH, 2.8))
    for func, r in res.items():
        norm = r["full"] if NORMALIZE else 1.0
        c = COLORS[func]
        if SHOW_UNCONSTRAINED:
            ax.plot(r["k"], r["lam"] / norm, "--", color=c, lw=0.8)
        if r.get("kc") is not None:
            ax.plot(r["kc"], r["lamc"] / norm, "-", color=c, lw=1.2, label=LABELS[func])
        ax.plot([r["nad"]], [r["adopted"] / norm], MARKERS[func], ms=5, color=c, mfc=c)
        ax.plot([r["nA"]], [r["A"] / norm], "x", ms=5, color=c, mew=1.2)
    ax.axvline(10, color="0.4", lw=0.8, ls="--", zorder=0)
    ax.set_xlabel(r"number of half-life nuclei $m$")   # m = set size in the paper (2026-11-03; k is its nucleus index)
    ax.set_ylabel(r"$\lambda_{\min}(M)\,/\,\lambda_{\min}(M_{\rm pool})$"
                  if NORMALIZE else r"$\lambda_{\min}(M)$")
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1.02 if NORMALIZE else None)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=COLORS[f], lw=1.0, label=LABELS[f]) for f in res]
    handles += [Line2D([], [], color="0.4", ls="--", lw=0.8, label=r"$m=10$"),
                *([Line2D([], [], color="0.3", ls="--", lw=0.8, label="unconstrained optimum")]
                  if SHOW_UNCONSTRAINED else []),
                Line2D([], [], color="0.3", ls="-", lw=1.2, label="E-optimal set"),
                Line2D([], [], marker="x", color="0.3", ls="", ms=5, mew=1.2,
                       label="Set A")]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="center right")
    ps.save(fig, OUT)


def replot():
    """Re-draw the figure from ../figs/fig_information_saturation.txt."""
    res = {}
    for ln in open(OUT + ".txt"):
        c = ln.split()
        if ln.startswith("#") and len(c) > 1 and c[1].rstrip(":") in FUNCTIONALS and "GT-only" in ln:
            f = c[1].rstrip(":")
            vals = ln.split(";")
            d = {}
            for v in vals[1:]:
                w = v.split()
                try:
                    d[w[0]] = float(w[-1])
                except ValueError:      # "marked set <key>" carries no number
                    continue
            res[f] = dict(gt=d["GT-only"], A=d["Set"], full=d["full"], adopted=d["marked"],
                          k=[], lam=[], nA=len(config.select_hl_nuclei("A")),
                          nad=len(config.select_hl_nuclei(MARKED[f])))
        elif not ln.startswith("#") and c and c[0] in res:
            res[c[0]]["k"].append(int(c[1])); res[c[0]]["lam"].append(float(c[2]))
        elif not ln.startswith("#") and c and c[0].endswith("-nested") and c[0][:-7] in res:
            r = res[c[0][:-7]]
            r.setdefault("kn", []).append(int(c[1])); r.setdefault("lamn", []).append(float(c[2]))
        elif not ln.startswith("#") and c and c[0].endswith("-constrained") and c[0][:-12] in res:
            r = res[c[0][:-12]]
            r.setdefault("kc", []).append(int(c[1])); r.setdefault("lamc", []).append(float(c[2]))
    for r in res.values():
        r["k"] = np.array(r["k"]); r["lam"] = np.array(r["lam"])
        if "kn" in r:
            r["kn"] = np.array(r["kn"]); r["lamn"] = np.array(r["lamn"])
        if "kc" in r:
            r["kc"] = np.array(r["kc"]); r["lamc"] = np.array(r["lamc"])
    draw(res)


if __name__ == "__main__":
    main()
