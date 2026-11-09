"""
Bayesian calibration of (vpair_t0, override_cs0) against beta-decay half-lives
and Gamow-Teller observables, using the AE/AO emulators trained by
main_only_HL_mod.py, main_only_centroid_mod.py, main_only_peak_mod.py, and
the legacy 20-dim GT strength emulator (main_mod.py).

GT likelihood modes (--gt-mode):
  full          : alias for full_centroid (kept for backward compatibility)
  full_centroid : K-summed GT strength emulator -> windowed centroid
  full_peak     : K-summed GT strength emulator -> global peak location
  full_window_peak : K-summed GT strength emulator -> peak location inside GT window
  centroid      : AE centroid scalar emulator
  centroid_total : AE centroid scalar emulator over all E_x >= 0 strength
  peak          : AE peak-location scalar emulator

HL likelihood is always included.

Likelihood families (--hl-likelihood / --gt-likelihood, each 'normal' or
'studentt' with --hl-nu / --gt-nu): the Student-t is variance-matched to the
Normal, so it changes tail shape only and down-weights model-discrepant
nuclei automatically instead of excluding them by hand.

Directory conventions (nucleus keys stripped of underscore):
  HL       :  SMLR/results/UNE1/HL/<Nuc>/{params_best_n5_only_HL.txt, train_set.txt}
  GT full  :  SMLR/results/UNE1/GT/<Nuc>/<K>/{params_best_n20_retain0.9.txt, train_set.txt}
  centroid :  SMLR/results/UNE1/centroid/<Nuc>/full/{params_best_n5_only_centroid.txt, train_set.txt}
  centroid_total : SMLR/results/UNE1/centroid_total/<Nuc>/full/{params_best_n5_only_centroid.txt, train_set.txt}
  peak     :  SMLR/results/UNE1/peak/<Nuc>/full/{params_best_n5_only_peak.txt,     train_set.txt}
  K-combined (centroid/peak) emulators live under the "full" subdir.

Output:
  SMLR/results/UNE1/bayes/<gt_mode>/trace.nc  (+ figs/ for the plotting script)
"""

import glob
import os
import sys
import argparse
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import arviz as az

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)
import config

# =====================================================================
# Paths (functional-parameterized)
# =====================================================================
sys.path.insert(0, os.path.join(THIS_DIR, ".."))
import repo_paths as _RP  # noqa: E402
RESULTS_BASE  = _RP.RESULTS     # output root of new MCMC runs: <RESULTS_BASE>/<functional>/<subdir>/
EMULATOR_BASE = _RP.EMULATORS   # trained emulators: <EMULATOR_BASE>/<functional>/{HL,GT}/<nuc>/
CV_BASE       = _RP.CV          # cross-validation parity files (sigma_em): <CV_BASE>/<functional>/{HL,GT}/<nuc>/

# Joint prior specification (independent TruncatedNormals).
# Single source of truth: consumed by build_model() and by bayes_plot.py.
PRIORS = {
    'vpair': dict(mu=-250.0, sigma=200.0, lower=-450.0, upper=0.0),
    'cs0':   dict(mu= 130.0, sigma=100.0, lower=   0.0, upper=250.0),
}

# ---------------------------------------------------------------------
# Stability-domain prior (agreed 2026-09-22; selected with --prior domain).
# The prior domain is the rectangle in which every nucleus of the pool is
# QRPA-stable: V0 >= V_lim (pool-wide isoscalar-pairing collapse limit, from
# the response scan + half-life-jump check) and C1s >= 50 (low-coupling
# failures). All values in MeV fm^3.
#   vpair: TruncatedNormal(mu = equal isoscalar/isovector pairing reference
#          8*Cpr[0] from the pnFAM log, sigma 200, [V_lim, 0]) times a soft
#          edge tanh^2((V0 - V_lim)/w), w ~ 10 from the wall softness; the
#          taper is a pm.Potential (prior known up to a constant).
#   cs0:   flat on [50, 250].
# 'legacy' reproduces PRIORS above (all runs before 2026-09-22).
# ---------------------------------------------------------------------
V_LIM_DEFAULT = {'SKOP': -371.0,   # Cr60 wall, shallowest coupling band (-370.9 +- 1.4)
                 'UNE1': -301.0,   # Ti58 wall, all couplings (-301 +- 4)
                 'SLY4': -450.0,   # Q-corrected wall scan 2026-10-06: no wall inside the box (= box edge)
                 'HFB1': -360.0}   # Q-corrected wall scan 2026-10-06 (WALL_SCAN_QCORR_2026-10-06.md)
VPAIR_MU_EQUAL_PAIRING = {'SKOP': -264.2,   # 8 * Cpr[0] = 8 * (-33.027)
                          'UNE1': -196.3,   # 8 * Cpr[0] = 8 * (-24.540) = mean(V0n, V0p)
                          'SLY4': -332.7,   # 8 * Cpr[0] = 8 * (-41.582), pnFAM fam.log (Ca48/Ar48/Sn134 identical)
                          'HFB1': -247.4}   # 8 * Cpr[0] = 8 * (-30.926), pnFAM fam.log (Ar48/Sn134 identical)
TAPER_W_DEFAULT = 10.0
CS_LOWER_DEFAULT = 50.0

def _functional_family(functional):
    f = str(functional).upper()
    if f.startswith('UNE1'): return 'UNE1'
    if f.startswith('SKOP'): return 'SKOP'
    if f.startswith('SLY4'): return 'SLY4'
    if f.startswith('HFB1'): return 'HFB1'   # UNEDF1-HFB: its own family, NOT UNE1
    return None

def make_prior_spec(kind, functional, v_lim=None, taper_w=None, vpair_mu=None,
                    cs_lower=None, cs_upper=None):
    """Return the joint prior specification dict used by build_model(),
    bayes_plot.py (sampling) and the manifest. kind: 'legacy' | 'domain'."""
    if kind == 'legacy':
        spec = {'kind': 'legacy',
                'vpair': dict(PRIORS['vpair'], taper_w=0.0),
                'cs0':   dict(PRIORS['cs0'], dist='tnorm')}
    elif kind == 'domain':
        fam = _functional_family(functional)
        if v_lim is None:
            if fam is None:
                raise SystemExit(f"--prior domain: no default V_lim for functional "
                                 f"'{functional}'; pass --v-lim")
            v_lim = V_LIM_DEFAULT[fam]
        if vpair_mu is None:
            if fam is None:
                raise SystemExit(f"--prior domain: no default vpair center for "
                                 f"'{functional}'; pass --vpair-mu")
            vpair_mu = VPAIR_MU_EQUAL_PAIRING[fam]
        spec = {'kind': 'domain',
                'vpair': dict(mu=float(vpair_mu), sigma=200.0, lower=float(v_lim),
                              upper=0.0, taper_w=TAPER_W_DEFAULT if taper_w is None else float(taper_w)),
                'cs0':   dict(dist='flat', mu=None, sigma=None,
                              lower=CS_LOWER_DEFAULT, upper=250.0)}
    else:
        raise ValueError(f"unknown prior kind {kind!r}")
    # explicit overrides (any kind)
    if kind == 'legacy':
        if v_lim   is not None: spec['vpair']['lower'] = float(v_lim)
        if taper_w is not None: spec['vpair']['taper_w'] = float(taper_w)
        if vpair_mu is not None: spec['vpair']['mu'] = float(vpair_mu)
    if cs_lower is not None: spec['cs0']['lower'] = float(cs_lower)
    if cs_upper is not None: spec['cs0']['upper'] = float(cs_upper)
    if spec['vpair']['lower'] >= spec['vpair']['upper']:
        raise SystemExit("prior: vpair lower bound must be below the upper bound")
    if spec['cs0']['lower'] >= spec['cs0']['upper']:
        raise SystemExit("prior: cs0 lower bound must be below the upper bound")
    return spec

def describe_prior_spec(spec):
    pv, pc = spec['vpair'], spec['cs0']
    v = (f"vpair ~ TruncatedNormal(mu={pv['mu']:.1f}, sigma={pv['sigma']:.0f}) on "
         f"[{pv['lower']:.1f}, {pv['upper']:.1f}]")
    if pv.get('taper_w', 0.0) > 0:
        v += f" x tanh^2((V0 - {pv['lower']:.1f})/{pv['taper_w']:.1f})"
    if pc.get('dist', 'tnorm') == 'flat':
        c = f"cs0 ~ Uniform[{pc['lower']:.1f}, {pc['upper']:.1f}]"
    else:
        c = (f"cs0 ~ TruncatedNormal(mu={pc['mu']:.1f}, sigma={pc['sigma']:.0f}) on "
             f"[{pc['lower']:.1f}, {pc['upper']:.1f}]")
    return f"[{spec['kind']}] {v}; {c}"

def sample_prior_numpy(spec, n, rng):
    """Draw n independent joint prior samples (numpy; used by bayes_plot.py).
    The tanh^2 taper is applied by rejection."""
    from scipy.stats import truncnorm
    pv, pc = spec['vpair'], spec['cs0']
    a = (pv['lower'] - pv['mu']) / pv['sigma']; b = (pv['upper'] - pv['mu']) / pv['sigma']
    w = pv.get('taper_w', 0.0)
    out = np.empty(0)
    while out.size < n:
        v = truncnorm.rvs(a, b, loc=pv['mu'], scale=pv['sigma'], size=2 * n, random_state=rng)
        if w > 0:
            keep = rng.random(v.size) < np.tanh((v - pv['lower']) / w) ** 2
            v = v[keep]
        out = np.concatenate([out, v])
    v = out[:n]
    if pc.get('dist', 'tnorm') == 'flat':
        c = rng.uniform(pc['lower'], pc['upper'], size=n)
    else:
        a = (pc['lower'] - pc['mu']) / pc['sigma']; b = (pc['upper'] - pc['mu']) / pc['sigma']
        c = truncnorm.rvs(a, b, loc=pc['mu'], scale=pc['sigma'], size=n, random_state=rng)
    return v, c

def read_prior_spec_from_manifest(manifest_path):
    """Return the prior spec recorded in manifest.txt, or the legacy spec for
    runs made before 2026-09-22 (no prior_spec line)."""
    import ast
    functional = 'SKOP'
    try:
        for line in open(manifest_path):
            if line.startswith('functional='):
                functional = line.split('=', 1)[1].strip()
            if line.startswith('prior_spec='):
                return ast.literal_eval(line.split('=', 1)[1].strip())
    except FileNotFoundError:
        pass
    return make_prior_spec('legacy', functional)

def _results_root(functional):  return os.path.join(RESULTS_BASE, functional)
def _hl_dir(functional):        return os.path.join(EMULATOR_BASE, functional, "HL")
def _gt_dir(functional):        return os.path.join(EMULATOR_BASE, functional, "GT_strength")   # legacy K-resolved strength emulators (not shipped)
def _cent_dir(functional):      return os.path.join(EMULATOR_BASE, functional, "GT")   # centroid emulators used in the paper
def _cent_total_dir(functional): return os.path.join(EMULATOR_BASE, functional, "GT_total")   # not shipped
def _peak_dir(functional):      return os.path.join(EMULATOR_BASE, functional, "GT_peak")   # not shipped
def _bayes_root(functional, subdir):
    """Root for Bayesian outputs. Subdir is supplied at call time (CLI flag)
    so different runs (e.g. 'bayes', 'bayes_wprior', 'bayes_pernuc') coexist."""
    return os.path.join(_results_root(functional), subdir)

# CV output root: fold_*/parity_*_val.txt live under
#   <_sigma_em_root(functional)>/<kind>/<Nuc>/[full|K0|K1]/fold_*/...
def _sigma_em_root(functional): return os.path.join(CV_BASE, functional)

def nuc_dir(name):
    """'Ca_48' -> 'Ca48'. Keys already without underscore pass through."""
    return str(name).replace("_", "")

FULL_CENTROID_MODES = ("full", "full_centroid")
FULL_STRENGTH_MODES = ("full", "full_centroid", "full_peak", "full_window_peak")
GT_PEAK_GRID = np.linspace(config.E_MIN, config.E_MAX, 351)


# =====================================================================
# CV-derived emulator uncertainty (sigma_em)
# =====================================================================
def _sigma_em_from_parity(base_dir, parity_file, true_col=1, pred_col=2, log10=False):
    """Pool fold_*/<parity_file> under base_dir and return (sigma_em, bias, N).

    sigma_em is the RMS of (pred - true) residuals INCLUDING the mean bias:
    predictions enter the likelihood uncorrected, so a systematic offset
    belongs in the emulator error budget. Quadrature covers a bias but does
    not cancel its pull on the posterior -- promotion should also gate on
    |bias| << sigma_the (checked on held-out points).
    Returns None if no fold parity files are found.
    """
    pred_all, true_all = [], []
    for fd in sorted(glob.glob(os.path.join(base_dir, "fold_*"))):
        pf = os.path.join(fd, parity_file)
        if not os.path.isfile(pf):
            continue
        data = np.loadtxt(pf)
        if data.ndim == 1:
            data = data[None, :]
        tv = data[:, true_col]
        pv = data[:, pred_col]
        if log10:
            tv = np.log10(tv)
            pv = np.log10(pv)
        pred_all.append(pv)
        true_all.append(tv)
    if not pred_all:
        return None
    pred_all = np.concatenate(pred_all)
    true_all = np.concatenate(true_all)
    r = pred_all - true_all
    bias = float(np.mean(r))
    sigma_em = float(np.sqrt(np.mean(r ** 2)))
    return sigma_em, bias, int(len(r))


def read_sigma_em_file(path, allow_zero=False, opt="--sigma-em-file"):
    """NUC=VALUE lines (blank / '#' lines ignored) -> {nuc: float}. Written by
    calibration/frozen_sigma_em.py; consumed via --sigma-em-file. Also parses
    --sigma-q-file (allow_zero=True: zero entries are valid there, negative
    ones are not; written by the sigma_q_<F>.dat tables, 2026-10-13)."""
    out = {}
    for ln in open(path):
        ln = ln.strip()
        if not ln or ln.startswith("#") or "=" not in ln:
            continue
        nuc, val = ln.split("=", 1)
        val = float(val)
        if val < 0.0 or (val <= 0.0 and not allow_zero):
            kind = "negative" if allow_zero else "non-positive"
            raise SystemExit(f"{opt}: {kind} value for {nuc}: {val}")
        out[nuc.strip()] = val
    return out


def load_sigma_em_hl(nuclei, functional, src_override=None):
    """Dict nuc -> sigma_em in log10(s) for HL emulator, from CV parity files.

    src_override: optional dict nuc -> functional-namespace whose cv/ tree to
    pool that nucleus's CV from instead (e.g. {'Pt200': 'SKOP_PT200LOC'} for a
    separately-managed local emulator). Must match the namespace the emulator
    itself is loaded from, so sigma_em describes the deployed artifact.
    """
    src_override = src_override or {}
    out = {}
    for nuc in nuclei:
        func = src_override.get(nuc, functional)
        base = os.path.join(_sigma_em_root(func), "HL", nuc_dir(nuc))
        res = _sigma_em_from_parity(base, "parity_half_life_val.txt", log10=True)
        if res is None:
            continue
        sig, bias, N = res
        out[nuc] = sig
        tag = f"  [src {func}]" if nuc in src_override else ""
        print(f"  HL    sigma_em[{nuc:8s}] = {sig:.3e} log10(s)  "
              f"(N={N}, bias={bias:+.2e}){tag}")
    return out


def load_sigma_em_gt(gt_mode, nuclei, functional):
    """Dict nuc -> sigma_em in MeV for the GT observable in use.

    Full mode assumes K=0 and K=1 spectra are trained separately for every GT
    nucleus. Prefers the combined-spectrum sigma_em written by combine_full_cv.py;
    falls back to a per-K quadrature approximation if the combined output is
    missing (or to K=0-only if K=1 CV is absent).
    """
    sigma_em_root = _sigma_em_root(functional)
    out = {}
    for nuc in nuclei:
        if gt_mode == 'centroid':
            base = os.path.join(sigma_em_root, "GT", nuc_dir(nuc), "full")
            res = _sigma_em_from_parity(base, "parity_centroid_val.txt")
            if res is not None:
                sig, bias, N = res
                out[nuc] = sig
                print(f"  GT(c) sigma_em[{nuc:8s}] = {sig:.3e} MeV       (N={N}, bias={bias:+.2e})")
        elif gt_mode == 'centroid_total':
            base = os.path.join(sigma_em_root, "centroid_total", nuc_dir(nuc), "full")
            res = _sigma_em_from_parity(base, "parity_centroid_val.txt")
            if res is not None:
                sig, bias, N = res
                out[nuc] = sig
                print(f"  GT(ct) sigma_em[{nuc:8s}] = {sig:.3e} MeV      (N={N}, bias={bias:+.2e})")
        elif gt_mode == 'peak':
            base = os.path.join(sigma_em_root, "peak", nuc_dir(nuc), "full")
            res = _sigma_em_from_parity(base, "parity_peak_val.txt")
            if res is not None:
                sig, bias, N = res
                out[nuc] = sig
                print(f"  GT(p) sigma_em[{nuc:8s}] = {sig:.3e} MeV       (N={N}, bias={bias:+.2e})")
        elif gt_mode in FULL_CENTROID_MODES:
            # Always prefer the combined-spectrum sigma_em from combine_full_cv.py.
            # Fall back to per-K quadrature (K=0 and K=1) if the combined file
            # is missing, or to K=0 alone if K=1 CV wasn't run.
            base_combined = os.path.join(sigma_em_root, "GT", nuc_dir(nuc),
                                         "full_combined")
            res = _sigma_em_from_parity(base_combined,
                                        "parity_full_centroid_val.txt")
            if res is not None:
                sig, bias, N = res
                out[nuc] = sig
                print(f"  GT(f) sigma_em[{nuc:8s}] = {sig:.3e} MeV       "
                      f"(combined, N={N}, bias={bias:+.2e})")
                continue
            print(f"  GT(f) sigma_em[{nuc:8s}]: combined CV missing under "
                  f"{base_combined}; falling back to per-K quadrature")
            sigs, N_sum, Ks_found = [], 0, []
            for K in ('0', '1'):
                base = os.path.join(sigma_em_root, "GT", nuc_dir(nuc), f"K{K}")
                res = _sigma_em_from_parity(base, "parity_moments_val.txt",
                                            true_col=1, pred_col=0)
                if res is not None:
                    sigs.append(res[0]); N_sum += res[2]; Ks_found.append(K)
            if sigs:
                sig = float(np.sqrt(np.sum([s ** 2 for s in sigs])))
                out[nuc] = sig
                print(f"  GT(f) sigma_em[{nuc:8s}] = {sig:.3e} MeV       "
                      f"(K={Ks_found}, N={N_sum}, quadrature)")
        elif gt_mode == 'full_peak':
            base_combined = os.path.join(sigma_em_root, "GT", nuc_dir(nuc),
                                         "full_combined")
            res = _sigma_em_from_parity(base_combined,
                                        "parity_full_peak_val.txt")
            if res is not None:
                sig, bias, N = res
                out[nuc] = sig
                print(f"  GT(fp) sigma_em[{nuc:8s}] = {sig:.3e} MeV      "
                      f"(combined peak, N={N}, bias={bias:+.2e})")
        elif gt_mode == 'full_window_peak':
            base_combined = os.path.join(sigma_em_root, "GT", nuc_dir(nuc),
                                         "full_combined")
            res = _sigma_em_from_parity(base_combined,
                                        "parity_full_window_peak_val.txt")
            if res is not None:
                sig, bias, N = res
                out[nuc] = sig
                print(f"  GT(fwp) sigma_em[{nuc:8s}] = {sig:.3e} MeV     "
                      f"(combined full window peak, N={N}, bias={bias:+.2e})")
    return out

# =====================================================================
# Experimental targets live in config.HL_EXP / HL_ERR / GT_E_EXP / GT_E_ERR.
# =====================================================================
HL_EXP   = config.HL_EXP
HL_ERR   = config.HL_ERR
GT_E_EXP = config.GT_E_EXP
GT_E_ERR = config.GT_E_ERR

# =====================================================================
# Scaling (matches main_only_*_mod.py: alpha = (vpair+250)/250, beta = (cs0-125)/125)
# =====================================================================
def scale_alpha_beta_sym(vpair_sym, cs0_sym):
    lo_v, hi_v = config.PARAM_BOUNDS["vpair_t0"]        # (-500, 0)
    lo_c, hi_c = config.PARAM_BOUNDS["override_cs0"]    # (0, 250)
    alpha = 2.0 * (vpair_sym - lo_v) / (hi_v - lo_v) - 1.0
    beta  = 2.0 * (cs0_sym   - lo_c) / (hi_c - lo_c) - 1.0
    return alpha, beta

# =====================================================================
# Emulator base + classes
# =====================================================================
class SMLREmulator:
    """Locates central point from a train_set.txt of scaled (alpha, beta) pairs."""
    def __init__(self, base_dir):
        self.base_dir = base_dir
        ts = os.path.join(base_dir, "train_set.txt")
        if not os.path.exists(ts):
            raise FileNotFoundError(f"train_set.txt missing: {ts}")
        combined_ar = np.loadtxt(ts, delimiter=',')
        centroid = combined_ar.mean(axis=0)
        distances = np.linalg.norm(combined_ar - centroid, axis=1)
        k = int(np.argmin(distances))
        self.cp_a = float(combined_ar[k, 0])   # vpair_scaled
        self.cp_b = float(combined_ar[k, 1])   # cs0_scaled


def _unpack_ae_symmetric(params, n):
    """Unpack AE scalar emulator params: [D diag (n), S1 uptri, S2 uptri]."""
    num_u = n * (n + 1) // 2
    expected = n + 2 * num_u
    if len(params) < expected:
        raise ValueError(f"Expected >= {expected} params for AE (n={n}), got {len(params)}")
    D = np.diag(params[:n])
    S1 = np.zeros((n, n))
    S1[np.triu_indices(n)] = params[n:n + num_u]
    S1 = S1 + S1.T - np.diag(np.diag(S1))
    S2 = np.zeros((n, n))
    S2[np.triu_indices(n)] = params[n + num_u:n + 2 * num_u]
    S2 = S2 + S2.T - np.diag(np.diag(S2))
    return D, S1, S2


class HalfLifeEmulator(SMLREmulator):
    """AE half-life emulator. Output: log10(T_{1/2}) in seconds."""
    def __init__(self, hl_key, functional, n=5):
        super().__init__(os.path.join(_hl_dir(functional), hl_key))
        self.n = n
        pf = os.path.join(self.base_dir, f"params_best_n{n}_only_HL.txt")
        if not os.path.exists(pf):
            raise FileNotFoundError(pf)
        params = np.loadtxt(pf)
        D, S1, S2 = _unpack_ae_symmetric(params, n)
        self.D_s  = pytensor.shared(D,  name=f'D_hl_{hl_key}')
        self.S1_s = pytensor.shared(S1, name=f'S1_hl_{hl_key}')
        self.S2_s = pytensor.shared(S2, name=f'S2_hl_{hl_key}')

    def __call__(self, vpair_sym, cs0_sym):
        alpha, beta = scale_alpha_beta_sym(vpair_sym, cs0_sym)
        da = alpha - self.cp_a
        db = beta  - self.cp_b
        M = self.D_s + da * self.S1_s + db * self.S2_s
        evals, _ = pt.linalg.eigh(M)
        return evals[self.n // 2]


class ScalarGTEmulator(SMLREmulator):
    """AE scalar emulator (centroid or peak). Output is math-E; subtract shift
    externally to compare with physical E_x."""
    def __init__(self, gt_key, kind, functional, n=5):
        assert kind in ("centroid", "centroid_total", "peak")
        if kind == "centroid":
            root = _cent_dir(functional)
            param_kind = "centroid"
        elif kind == "centroid_total":
            root = _cent_total_dir(functional)
            param_kind = "centroid"
        else:
            root = _peak_dir(functional)
            param_kind = "peak"
        base_dir = os.path.join(root, nuc_dir(gt_key), "full")
        super().__init__(base_dir)
        self.kind = kind
        self.n = n
        pf = os.path.join(base_dir, f"params_best_n{n}_only_{param_kind}.txt")
        if not os.path.exists(pf):
            raise FileNotFoundError(pf)
        params = np.loadtxt(pf)
        D, S1, S2 = _unpack_ae_symmetric(params, n)
        tag = f'{kind}_{nuc_dir(gt_key)}'
        self.D_s  = pytensor.shared(D,  name=f'D_{tag}')
        self.S1_s = pytensor.shared(S1, name=f'S1_{tag}')
        self.S2_s = pytensor.shared(S2, name=f'S2_{tag}')

    def __call__(self, vpair_sym, cs0_sym):
        alpha, beta = scale_alpha_beta_sym(vpair_sym, cs0_sym)
        da = alpha - self.cp_a
        db = beta  - self.cp_b
        M = self.D_s + da * self.S1_s + db * self.S2_s
        evals, _ = pt.linalg.eigh(M)
        return evals[self.n // 2]


class GTStrengthEmulator(SMLREmulator):
    """Legacy 20-dim GT strength emulator (AE). Same parameter layout as
    main_mod.py: [eta, v0, v1, v2, D, S1_uptri, S2_uptri, (optional x1,x2,x3)].
    Returns broadened L(E) on a user-supplied E-grid (math-E frame), or
    discrete poles via get_discrete_poles()."""
    def __init__(self, gt_key, K, functional, n=20, retain=0.9):
        base_dir = os.path.join(_gt_dir(functional), nuc_dir(gt_key), f"K{K}")
        super().__init__(base_dir)
        self.n, self.retain = n, retain
        pf = os.path.join(base_dir, f"params_best_n{n}_retain{retain}.txt")
        if not os.path.exists(pf):
            raise FileNotFoundError(pf)
        params = np.loadtxt(pf)

        idx = 0
        self.eta_val = float(params[idx]); idx += 1
        v0 = params[idx:idx + n]; idx += n
        v1 = params[idx:idx + n]; idx += n
        v2 = params[idx:idx + n]; idx += n
        D  = np.diag(params[idx:idx + n]); idx += n

        num_u = n * (n + 1) // 2
        S1 = np.zeros((n, n)); S1[np.triu_indices(n)] = params[idx:idx + num_u]
        S1 = S1 + S1.T - np.diag(np.diag(S1)); idx += num_u
        S2 = np.zeros((n, n)); S2[np.triu_indices(n)] = params[idx:idx + num_u]
        S2 = S2 + S2.T - np.diag(np.diag(S2)); idx += num_u

        if idx + 3 <= len(params):
            self.x1_val = float(params[idx]); self.x2_val = float(params[idx + 1]); self.x3_val = float(params[idx + 2])
        else:
            self.x1_val = self.x2_val = self.x3_val = 0.0

        tag = f'{nuc_dir(gt_key)}_K{K}'
        self.D_s  = pytensor.shared(D,  name=f'D_gt_{tag}')
        self.S1_s = pytensor.shared(S1, name=f'S1_gt_{tag}')
        self.S2_s = pytensor.shared(S2, name=f'S2_gt_{tag}')
        self.v0_s = pytensor.shared(v0, name=f'v0_gt_{tag}')
        self.v1_s = pytensor.shared(v1, name=f'v1_gt_{tag}')
        self.v2_s = pytensor.shared(v2, name=f'v2_gt_{tag}')

    def _eigendecomp(self, vpair_sym, cs0_sym):
        alpha, beta = scale_alpha_beta_sym(vpair_sym, cs0_sym)
        da = alpha - self.cp_a
        db = beta  - self.cp_b
        M = self.D_s + da * self.S1_s + db * self.S2_s
        evals, evecs = pt.linalg.eigh(M)
        k = max(1, min(int(round(self.retain * self.n)), self.n))
        lo = (self.n - k) // 2
        hi = lo + k
        v_dyn = self.v0_s + da * self.v1_s + db * self.v2_s
        return evals[lo:hi], evecs[:, lo:hi], v_dyn, alpha, beta

    def get_discrete_poles(self, vpair_sym, cs0_sym):
        e_kept, v_kept, v_dyn, _a, _b = self._eigendecomp(vpair_sym, cs0_sym)
        B = pt.dot(v_kept.T, v_dyn) ** 2
        return e_kept, B

    def broadened(self, vpair_sym, cs0_sym, E_grid_sym):
        e_kept, v_kept, v_dyn, alpha, beta = self._eigendecomp(vpair_sym, cs0_sym)
        B = pt.dot(v_kept.T, v_dyn) ** 2
        width = pt.sqrt(self.eta_val ** 2 +
                        (self.x1_val + self.x2_val * alpha + self.x3_val * beta) ** 2)
        num = B * (width / (2 * np.pi))
        den = (E_grid_sym[:, None] - e_kept[None, :]) ** 2 + (width ** 2 / 4)
        return pt.sum(num / den, axis=1)


def combined_gt_poles(gt_emus, vpair_sym, cs0_sym):
    """Concatenate K=0 and K=1 poles with strength weights (1, 2).

    Kept backward-compatible with a single-K emulator list (returns its poles
    unchanged) in case an older registry is used.
    """
    if len(gt_emus) == 1:
        return gt_emus[0].get_discrete_poles(vpair_sym, cs0_sym)
    e0, b0 = gt_emus[0].get_discrete_poles(vpair_sym, cs0_sym)
    e1, b1 = gt_emus[1].get_discrete_poles(vpair_sym, cs0_sym)
    return pt.concatenate([e0, e1]), pt.concatenate([b0, 2.0 * b1])


def combined_gt_broadened(gt_emus, vpair_sym, cs0_sym, E_grid_sym):
    """K-summed broadened spectrum S_K0(E) + 2*S_K1(E)."""
    if len(gt_emus) == 1:
        return gt_emus[0].broadened(vpair_sym, cs0_sym, E_grid_sym)
    return (gt_emus[0].broadened(vpair_sym, cs0_sym, E_grid_sym) +
            2.0 * gt_emus[1].broadened(vpair_sym, cs0_sym, E_grid_sym))


def peak_location_from_grid(E_grid_sym, S_sym):
    """Global peak with local parabolic interpolation on the fixed GT grid."""
    n_grid = GT_PEAK_GRID.size
    k0 = pt.argmax(S_sym)
    k = pt.clip(k0, 1, n_grid - 2)
    s_m = S_sym[k - 1]
    s_0 = S_sym[k]
    s_p = S_sym[k + 1]
    denom = s_m - 2.0 * s_0 + s_p
    delta = pt.switch(pt.lt(denom, 0.0), 0.5 * (s_m - s_p) / denom, 0.0)
    peak = E_grid_sym[k] + delta * (E_grid_sym[k + 1] - E_grid_sym[k])
    return pt.switch((pt.eq(k0, 0) | pt.eq(k0, n_grid - 1)), E_grid_sym[k0], peak)


def window_peak_location_from_grid(E_grid_sym, S_sym, E_lo, E_hi):
    """Peak inside [E_lo, E_hi], with interpolation only when neighbors are in-window."""
    n_grid = GT_PEAK_GRID.size
    in_window = (E_grid_sym >= E_lo) & (E_grid_sym <= E_hi)
    S_masked = pt.switch(in_window, S_sym, -1.0e300)
    k0 = pt.argmax(S_masked)
    k = pt.clip(k0, 1, n_grid - 2)
    s_m = S_sym[k - 1]
    s_0 = S_sym[k]
    s_p = S_sym[k + 1]
    denom = s_m - 2.0 * s_0 + s_p
    delta = pt.switch(pt.lt(denom, 0.0), 0.5 * (s_m - s_p) / denom, 0.0)
    interp = E_grid_sym[k] + delta * (E_grid_sym[k + 1] - E_grid_sym[k])
    can_interp = (in_window[k - 1] & in_window[k] & in_window[k + 1] &
                  pt.neq(k0, 0) & pt.neq(k0, n_grid - 1) & pt.lt(denom, 0.0))
    return pt.switch(can_interp, interp, E_grid_sym[k0])


# =====================================================================
# Emulator registry
# =====================================================================
def load_hl_emulators(functional, hl_keys, src_override=None):
    """Load promoted HL emulators; src_override (dict nuc -> functional
    namespace) redirects individual nuclei to a separately-managed emulator
    tree, e.g. {'Pt200': 'SKOP_PT200LOC'}. Keep it consistent with the
    src_override passed to load_sigma_em_hl."""
    emus = {}
    missing = []
    src_override = src_override or {}
    for nuc in hl_keys:
        func = src_override.get(nuc, functional)
        try:
            emus[nuc] = HalfLifeEmulator(nuc, func, n=5)
            if nuc in src_override:
                print(f"  HL emulator[{nuc}] loaded from functional "
                      f"namespace {func}")
        except FileNotFoundError as e:
            missing.append((nuc, str(e)))
    if missing:
        print("WARNING: missing HL emulators:")
        for nuc, msg in missing:
            print(f"  {nuc}: {msg}")
    return emus

def load_gt_emulators(gt_mode, functional, gt_keys):
    """Returns dict: gt_key -> list of emulator objects.

    In full-strength modes we always load K=0 and K=1 separately for every GT
    nucleus. In 'centroid' / 'peak' modes we use the single combined-K scalar
    emulator under <mode>/<Nuc>/full/.
    """
    emus = {}
    missing = []
    for nuc in gt_keys:
        try:
            if gt_mode in FULL_STRENGTH_MODES:
                emus[nuc] = [GTStrengthEmulator(nuc, K, functional) for K in ('0', '1')]
            elif gt_mode in ('centroid', 'centroid_total', 'peak'):
                emus[nuc] = [ScalarGTEmulator(nuc, gt_mode, functional, n=5)]
            else:
                raise ValueError(gt_mode)
        except FileNotFoundError as e:
            missing.append((nuc, str(e)))
    if missing:
        print(f"WARNING: missing GT ({gt_mode}) emulators:")
        for nuc, msg in missing:
            print(f"  {nuc}: {msg}")
    return emus


def require_mcmc_inputs(hl_emus, gt_emus, hl_keys, gt_keys):
    """Fail before sampling if required promoted emulators are missing."""
    missing_hl = [nuc for nuc in hl_keys if nuc not in hl_emus]
    missing_gt = [nuc for nuc in gt_keys if nuc not in gt_emus]
    if not missing_hl and not missing_gt:
        return

    lines = ["Missing required promoted emulators; refusing to sample."]
    if missing_hl:
        lines.append("  HL: " + ", ".join(missing_hl))
    if missing_gt:
        lines.append("  GT: " + ", ".join(missing_gt))
    lines.append(
        "Run CV pooling/combining as needed, then promote the best CV folds "
        "with ./run_promote_best_cv.sh <functional> all."
    )
    raise SystemExit("\n".join(lines))


def warn_missing_sigma_em(sigma_em_hl, sigma_em_gt, hl_keys, gt_keys):
    """Warn when default sigma_em inputs are incomplete."""
    missing_hl = [nuc for nuc in hl_keys if nuc not in sigma_em_hl]
    missing_gt = [nuc for nuc in gt_keys if nuc not in sigma_em_gt]
    if missing_hl or missing_gt:
        print("WARNING: some CV sigma_em entries are missing and will default to zero:")
        if missing_hl:
            print("  HL: " + ", ".join(missing_hl))
        if missing_gt:
            print("  GT: " + ", ".join(missing_gt))


def netcdf_safe_inference_data(idata):
    """Return a copy with object-dtype variables removed before NetCDF save.

    PyMC's SMC sampler may store ragged per-stage histories as object arrays
    containing Python lists. Those are useful diagnostics, but xarray/NetCDF
    cannot serialize them. Drop only those object variables so the posterior,
    observed data, and numeric sampler stats still save normally.
    """
    safe = idata.copy()
    dropped = []
    for group_name in safe.groups():
        ds = getattr(safe, group_name)
        drop_names = [
            name for name, var in ds.variables.items()
            if var.dtype == object
        ]
        if drop_names:
            setattr(safe, group_name, ds.drop_vars(drop_names))
            dropped.extend(f"{group_name}/{name}" for name in drop_names)
    if dropped:
        print("NetCDF save: dropped object-dtype variables:")
        for name in dropped:
            print(f"  {name}")
    return safe


# =====================================================================
# Model builder
# =====================================================================
def build_model(gt_mode, hl_emus, gt_emus, hl_keys, gt_keys,
                sigma_em_hl=None, sigma_em_gt=None,
                per_nucleus_sigma=False,
                hl_likelihood='normal', hl_nu=4.0,
                gt_likelihood='normal', gt_nu=4.0,
                functional='SKOP', sigma_the_scale=0.3, prior_spec=None,
                sigma_q_hl=None):
    sigma_em_hl = sigma_em_hl or {}
    # Q-value correction uncertainty per HL nucleus [log10 s] (--sigma-q-file,
    # 2026-10-13); fixed numbers, added in quadrature below. Absent -> 0.
    sigma_q_hl = sigma_q_hl or {}
    prior_spec = prior_spec or make_prior_spec('legacy', functional)
    sigma_em_gt = sigma_em_gt or {}
    # Stable orderings for vector-indexed per-nucleus sigmas. Built whether or
    # not per-nucleus mode is on; the manifest records them either way.
    hl_active = [nuc for nuc in hl_keys if nuc in hl_emus]
    gt_active = [nuc for nuc in gt_keys if nuc in gt_emus]
    hl_idx = {n: i for i, n in enumerate(hl_active)}
    gt_idx = {n: i for i, n in enumerate(gt_active)}
    if per_nucleus_sigma:
        print(f"HL nuclei (sigma order): {hl_active}")
        print(f"GT nuclei (sigma order): {gt_active}")
    model = pm.Model()
    with model:
        ############### parameter priors ###############
        # vpair = pm.TruncatedNormal('vpair', mu=-346.352, sigma=200.0, upper=0.0)
        # cs0   = pm.TruncatedNormal('cs0',   mu= 128.279, sigma=200.0, lower=0.0)

        # Uniform priors for testing sensitivity to the default truncated normal priors:
        # vpair = pm.Uniform('vpair', lower=-500.0, upper=0.0)
        # cs0   = pm.Uniform('cs0',   lower=0.0, upper=250.0)

        # Location parameters: weakly informative truncated Normal.
        # Truncation is mandatory (emulator only defined in the pnFAM box).
        # Center + sigma are chosen so the prior is roughly flat across the box
        # bulk but tapers near the boundaries — no hard wall to pile against.
        # Spec comes from make_prior_spec(): 'legacy' = the PRIORS dict above;
        # 'domain' = stability-domain prior (soft edge at V_lim, flat cs0).
        pv, pc = prior_spec['vpair'], prior_spec['cs0']
        vpair = pm.TruncatedNormal('vpair', mu=pv['mu'], sigma=pv['sigma'],
                                   lower=pv['lower'], upper=pv['upper'])
        if pv.get('taper_w', 0.0) > 0.0:
            # Soft edge: multiply the prior density by tanh^2((V0 - V_lim)/w).
            # Vanishes at the edge, ~1 beyond a few w. Unnormalized (constant
            # factor), which is irrelevant for posterior sampling.
            pm.Potential('vpair_taper',
                         2.0 * pt.log(pt.tanh((vpair - pv['lower']) / pv['taper_w'])))
        if pc.get('dist', 'tnorm') == 'flat':
            cs0 = pm.Uniform('cs0', lower=pc['lower'], upper=pc['upper'])
        else:
            cs0 = pm.TruncatedNormal('cs0', mu=pc['mu'], sigma=pc['sigma'],
                                     lower=pc['lower'], upper=pc['upper'])

        ################# error model priors ###############
        # sigma_hl_the = pm.HalfNormal('sigma_hl_the', sigma=10.0)
        # sigma_gt_the = pm.HalfNormal('sigma_gt_the', sigma=10.0)

        # Scale parameters: log-uniform (Jeffreys-style scale-invariant prior).
        # Equal a priori weight per order of magnitude of sigma.
        # log_sigma_hl_the = pm.Uniform('log_sigma_hl_the', lower=-6.0, upper=2.0)
        # sigma_hl_the     = pm.Deterministic('sigma_hl_the', pt.exp(log_sigma_hl_the))
        # log_sigma_gt_the = pm.Uniform('log_sigma_gt_the', lower=-6.0, upper=2.0)
        # sigma_gt_the     = pm.Deterministic('sigma_gt_the', pt.exp(log_sigma_gt_the))

        # Scale parameters: HalfStudentT, the Stan-team default for variance
        # components. Heavy enough tail to let data move it. The scale
        # (--sigma-the-scale, one value for both) sets the size of theory
        # error considered typical a priori: HL in log10(s), GT in MeV. The
        # historical default 0.3 (factor ~2 in half-life) is informative at
        # the fitted values (posterior sigma_hl_the 0.5-1.4): its density is
        # 27% of peak at 0.5 and 1% at 1.4. Scale 1 keeps the prior within
        # 0.86-0.45 of peak over 0.5-1.2. Runs whose manifest lacks
        # sigma_the_scale used 0.3. With --per-nucleus-sigma the globals
        # become per-nucleus vectors indexed by hl_idx / gt_idx.
        if per_nucleus_sigma:
            sigma_hl_the = pm.HalfStudentT('sigma_hl_the', nu=4, sigma=sigma_the_scale,
                                           shape=len(hl_active))
            sigma_gt_the = pm.HalfStudentT('sigma_gt_the', nu=4, sigma=sigma_the_scale,
                                           shape=len(gt_active))
        else:
            sigma_hl_the = pm.HalfStudentT('sigma_hl_the', nu=4, sigma=sigma_the_scale)
            sigma_gt_the = pm.HalfStudentT('sigma_gt_the', nu=4, sigma=sigma_the_scale)


        for nuc in hl_keys:
            if nuc not in hl_emus:
                continue
            exp_val = HL_EXP[nuc]
            pred_log10 = hl_emus[nuc](vpair, cs0)
            # Exp error floor: convert seconds -> log10 via d(log10 T)/dT = 1/(T ln10)
            sig_exp_log10 = HL_ERR[nuc] / (exp_val * np.log(10.0))
            sig_em = sigma_em_hl.get(nuc, 0.0)
            sig_q = sigma_q_hl.get(nuc, 0.0)    # Q-correction term [log10 s]
            sig_hl_the = sigma_hl_the[hl_idx[nuc]] if per_nucleus_sigma else sigma_hl_the
            # Total HL scale; used by both the Normal and the Student-t branch.
            sigma_hl_tot = pt.sqrt(sig_hl_the ** 2 + sig_exp_log10 ** 2 + sig_em ** 2
                                   + sig_q ** 2)
            if hl_likelihood == 'studentt':
                # Robust HL likelihood: heavy tails auto-down-weight nuclei whose
                # preferred vpair is far from the joint fit (both the Pt200 -402
                # and Kr92 -146 extremes) without hand-picking. Scale is
                # variance-matched to the Normal (Var[StudentT] = nu/(nu-2)*s^2)
                # so normal-vs-t runs differ only in tail shape, not bulk width.
                # nu -> inf recovers the Normal exactly (test_hl_likelihood_limit.py).
                scale = sigma_hl_tot * pt.sqrt((hl_nu - 2.0) / hl_nu)
                pm.StudentT(f'hl_like_{nuc}',
                            nu=hl_nu, mu=pred_log10, sigma=scale,
                            observed=np.log10(exp_val))
            else:
                pm.Normal(f'hl_like_{nuc}',
                          mu=pred_log10, sigma=sigma_hl_tot,
                          observed=np.log10(exp_val))

        # Emit one GT likelihood term, Normal or robust Student-t. Mirrors the
        # HL branch above: heavy tails auto-down-weight nuclei whose preferred
        # cs0 is far from the joint fit (on SKO' the Zr90 ~50 / Ca48 ~46 pair
        # against the Pb208/Te130 ~129 bulk) without hand-picking which GT
        # nuclei to include. Scale is variance-matched to the Normal
        # (Var[StudentT] = nu/(nu-2)*s^2) so normal-vs-t runs differ only in
        # tail shape, not bulk width; nu -> inf recovers the Normal exactly
        # (test_gt_likelihood_limit.py).
        def gt_observed(name, mu, sigma, observed):
            if gt_likelihood == 'studentt':
                pm.StudentT(name, nu=gt_nu, mu=mu,
                            sigma=sigma * pt.sqrt((gt_nu - 2.0) / gt_nu),
                            observed=observed)
            else:
                pm.Normal(name, mu=mu, sigma=sigma, observed=observed)

        for nuc in gt_keys:
            if nuc not in gt_emus:
                continue
            exp_target = GT_E_EXP[nuc]
            shift = config.get_gt_shift(nuc, functional)
            sig_em_gt = sigma_em_gt.get(nuc, 0.0)
            sig_gt_the = sigma_gt_the[gt_idx[nuc]] if per_nucleus_sigma else sigma_gt_the
            sigma_gt_tot = pt.sqrt(sig_gt_the ** 2 + GT_E_ERR[nuc] ** 2 + sig_em_gt ** 2)

            if gt_mode in FULL_CENTROID_MODES:
                E_poles, B = combined_gt_poles(gt_emus[nuc], vpair, cs0)
                E_x_poles = E_poles - shift
                # Physical cutoff
                mask_phys = pt.cast(E_x_poles >= 0.0, 'float64')
                B_phys = B * mask_phys
                S_tot = pt.sum(B_phys) + 1e-9
                # Windowed centroid (active branch of original script)
                lo_ex, hi_ex = config.get_gt_window(nuc)
                mask_win = pt.cast((E_x_poles >= lo_ex) & (E_x_poles <= hi_ex), 'float64')
                B_win = B_phys * mask_win
                S_in  = pt.sum(B_win) + 1e-9
                pred_centroid_in = pt.sum(E_x_poles * B_win) / S_in
                variance_in = pt.sum(((E_x_poles - pred_centroid_in) ** 2) * B_win) / S_in
                pred_width_in = pt.sqrt(pt.switch(variance_in < 0, 0.0, variance_in) + 1e-9)
                strength_fraction = S_in / S_tot

                pm.Deterministic(f'tracked_cent_in_{nuc}',  pred_centroid_in)
                pm.Deterministic(f'tracked_width_in_{nuc}', pred_width_in)
                pm.Deterministic(f'tracked_frac_{nuc}',     strength_fraction)

                gt_observed(f'gt_cent_like_{nuc}',
                            pred_centroid_in, sigma_gt_tot, exp_target)

            elif gt_mode == 'full_peak':
                E_grid_s = pytensor.shared(GT_PEAK_GRID, name=f'E_peak_grid_{nuc}')
                S_full = combined_gt_broadened(gt_emus[nuc], vpair, cs0, E_grid_s)
                pred_peak_math = peak_location_from_grid(E_grid_s, S_full)
                pred_peak_Ex = pred_peak_math - shift
                pm.Deterministic(f'tracked_peak_Ex_{nuc}', pred_peak_Ex)
                gt_observed(f'gt_peak_like_{nuc}',
                            pred_peak_Ex, sigma_gt_tot, exp_target)

            elif gt_mode == 'full_window_peak':
                E_grid_s = pytensor.shared(GT_PEAK_GRID, name=f'E_full_window_peak_grid_{nuc}')
                S_full = combined_gt_broadened(gt_emus[nuc], vpair, cs0, E_grid_s)
                lo_ex, hi_ex = config.get_gt_window(nuc)
                pred_peak_math = window_peak_location_from_grid(
                    E_grid_s, S_full, lo_ex + shift, hi_ex + shift)
                pred_peak_Ex = pred_peak_math - shift
                pm.Deterministic(f'tracked_full_window_peak_Ex_{nuc}', pred_peak_Ex)
                gt_observed(f'gt_full_window_peak_like_{nuc}',
                            pred_peak_Ex, sigma_gt_tot, exp_target)

            else:
                # centroid / centroid_total / peak: single scalar emulator per nucleus
                # (combined-K), output is math-E -> subtract shift
                pred_E_math = gt_emus[nuc][0](vpair, cs0)
                pred_Ex = pred_E_math - shift
                pm.Deterministic(f'tracked_pred_Ex_{nuc}', pred_Ex)
                gt_observed(f'gt_like_{nuc}',
                            pred_Ex, sigma_gt_tot, exp_target)

    return model, hl_active, gt_active


# =====================================================================
# CLI
# =====================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gt-mode',
                    choices=['full', 'full_centroid', 'full_peak', 'full_window_peak',
                             'centroid', 'centroid_total', 'peak'],
                    required=True)
    ap.add_argument('--sampler', choices=['smc', 'nuts', 'demz', 'slice'], default='smc')
    ap.add_argument('--draws',   type=int, default=2000)
    ap.add_argument('--tune',    type=int, default=1000)
    ap.add_argument('--chains',  type=int, default=4)
    ap.add_argument('--cores',   type=int, default=4)
    ap.add_argument('--seed',    type=int, default=42)
    ap.add_argument('--functional', default='UNE1',
                    help="Functional name (subdir of results/ and cv/). Default: UNE1.")
    ap.add_argument('--save-dir', default=None,
                    help="Override default SMLR/results/<functional>/bayes/<gt_mode>/")
    ap.add_argument('--no-sigma-em', action='store_true',
                    help="Do NOT include CV-derived emulator uncertainty in the likelihood. "
                         "Default: include (reads from cv/<functional>/...).")
    ap.add_argument('--bayes-subdir', default='bayes',
                    help="Subdirectory under results/<functional>/ for outputs. "
                         "Default: 'bayes'. Use e.g. 'bayes_wprior' or "
                         "'bayes_pernuc' to keep alternative runs separate.")
    ap.add_argument('--per-nucleus-sigma', action='store_true',
                    help="Use independent per-nucleus sigma_hl_the / sigma_gt_the "
                         "vectors instead of scalar globals. Diagnostic mode.")
    ap.add_argument('--hl-likelihood', choices=['normal', 'studentt'],
                    default='normal',
                    help="HL likelihood family. 'studentt' is a robust "
                         "variance-matched Student-t that down-weights "
                         "model-discrepant nuclei (e.g. Pt200). See "
                         "--gt-likelihood for the GT counterpart. "
                         "Default: normal (unchanged).")
    ap.add_argument('--hl-nu', type=float, default=4.0,
                    help="Degrees of freedom for --hl-likelihood studentt. "
                         "Must be > 2 (variance matching). nu -> inf recovers "
                         "the Normal. Default: 4.")
    ap.add_argument('--gt-likelihood', choices=['normal', 'studentt'],
                    default='normal',
                    help="GT likelihood family. 'studentt' is a robust "
                         "variance-matched Student-t that down-weights "
                         "model-discrepant nuclei (on SKO' the low-cs0 "
                         "Zr90/Ca48 pair), so the full GT set can be used "
                         "without hand-picking. Default: normal (unchanged).")
    ap.add_argument('--gt-nu', type=float, default=4.0,
                    help="Degrees of freedom for --gt-likelihood studentt. "
                         "Must be > 2 (variance matching). nu -> inf recovers "
                         "the Normal. Default: 4.")
    ap.add_argument('--sigma-the-scale', type=float, default=0.3,
                    help="Scale of the HalfStudentT(nu=4) prior on the shared "
                         "theory errors sigma_hl_the [log10 s] and sigma_gt_the "
                         "[MeV]; one value for both. 0.3 = historical default "
                         "(informative: pulls sigma_the down at the fitted "
                         "0.5-1.4); 1 = weakly informative. Recorded in the "
                         "manifest as sigma_the_scale.")
    ap.add_argument('--prior', choices=['legacy', 'domain'], default='legacy',
                    help="Joint prior on (vpair, cs0). 'legacy' = TruncatedNormal "
                         "(-250, 200) on [-450, 0] x TruncatedNormal(130, 100) on "
                         "[0, 250] (all runs before 2026-09-22). 'domain' = "
                         "stability-domain prior: vpair TruncatedNormal(mu = equal-"
                         "pairing reference -264.2 SKOP / -196.3 UNE1 / -332.7 SLY4 / -247.4 HFB1, sigma 200) on "
                         "[V_lim, 0] with soft edge tanh^2((V0-V_lim)/w), V_lim = "
                         "-371 SKOP / -301 UNE1 / -450 SLY4 / -360 HFB1 (pool-wide collapse limit), w = 10; "
                         "cs0 flat on [50, 250]. Recorded in the manifest as "
                         "prior_kind / prior_spec. Default: legacy (unchanged).")
    ap.add_argument('--v-lim', type=float, default=None,
                    help="Override the vpair lower edge V_lim [MeV fm^3].")
    ap.add_argument('--taper-w', type=float, default=None,
                    help="Override the soft-edge width w [MeV fm^3]; 0 = hard edge.")
    ap.add_argument('--vpair-mu', type=float, default=None,
                    help="Override the vpair prior center [MeV fm^3].")
    ap.add_argument('--cs-lower', type=float, default=None,
                    help="Override the cs0 lower bound [MeV fm^3].")
    ap.add_argument('--cs-upper', type=float, default=None,
                    help="Override the cs0 upper bound [MeV fm^3].")
    ap.add_argument('--hl-emu-src', nargs='*', default=[], metavar='NUC=FUNC',
                    help="Per-nucleus HL emulator source override, e.g. "
                         "Pt200=SKOP_PT200LOC: load that nucleus's promoted "
                         "emulator from results/<FUNC>/HL/<NUC> and pool its "
                         "sigma_em from cv/<FUNC>/HL/<NUC>, instead of the "
                         "run's --functional. Default: none (unchanged).")
    ap.add_argument('--sigma-em-file', default=None,
                    help="File of NUC=VALUE lines giving certified per-nucleus HL "
                         "sigma_em in log10(s) (from calibration/frozen_sigma_em.py). "
                         "Applied before --sigma-em-override, which wins on conflict. "
                         "Recorded in the manifest.")
    ap.add_argument('--sigma-em-override', nargs='*', default=[],
                    metavar='NUC=VALUE',
                    help="Per-nucleus HL sigma_em override in log10(s), e.g. "
                         "Pt200=0.085. Use when pooled CV sigma_em is biased "
                         "by rough out-of-posterior margins; the value should "
                         "come from held-out (frozen-set) validation of the "
                         "deployed emulator. Recorded in the manifest. "
                         "Default: none (pooled CV values used).")
    ap.add_argument('--sigma-q-file', default=None,
                    help="File of NUC=VALUE lines (same format as --sigma-em-file) "
                         "giving the per-nucleus half-life uncertainty from the "
                         "Q-value correction, in log10(s) (calibration/analysis/"
                         "sigma_q_<F>.dat). Added in quadrature to the HL "
                         "likelihood scale next to sigma_em (sigma_em unchanged; "
                         "applied also with --no-sigma-em). HL nuclei absent from "
                         "the file get 0 (warning printed). Recorded in the "
                         "manifest. Default: none (no term).")
    ap.add_argument('--hl-set', default='all',
                    choices=['all'] + list(config.HL_SETS),
                    help="HL calibration set (subset of HL_EXP). Default: all.")
    ap.add_argument('--gt-set', default='all',
                    choices=['all'] + list(config.GT_SETS),
                    help="GT calibration set (subset of GT_E_EXP). Default: all.")
    args = ap.parse_args()

    if args.hl_likelihood == 'studentt' and args.hl_nu <= 2.0:
        raise SystemExit("--hl-nu must be > 2 for the variance-matched "
                         "Student-t HL likelihood.")
    if args.hl_likelihood == 'studentt' and args.hl_nu > 1.0e3:
        # PyMC's StudentT normalization loses float precision at extreme nu
        # (~3e-3 per likelihood term at nu=1e4, worse beyond; see
        # test_hl_likelihood_limit.py). The Normal is the correct model there.
        print(f"WARNING: --hl-nu {args.hl_nu:g} is in the regime where "
              "PyMC's StudentT logp is numerically inaccurate; use "
              "--hl-likelihood normal instead of a large-nu Student-t.")
    if args.sigma_the_scale <= 0.0:
        raise SystemExit("--sigma-the-scale must be > 0")
    if args.gt_likelihood == 'studentt' and args.gt_nu <= 2.0:
        raise SystemExit("--gt-nu must be > 2 for the variance-matched "
                         "Student-t GT likelihood.")
    if args.gt_likelihood == 'studentt' and args.gt_nu > 1.0e3:
        # Same PyMC StudentT precision limit as the HL branch above; see
        # test_gt_likelihood_limit.py.
        print(f"WARNING: --gt-nu {args.gt_nu:g} is in the regime where "
              "PyMC's StudentT logp is numerically inaccurate; use "
              "--gt-likelihood normal instead of a large-nu Student-t.")

    hl_emu_src = {}
    for item in args.hl_emu_src:
        if '=' not in item:
            raise SystemExit(f"--hl-emu-src expects NUC=FUNCTIONAL pairs, "
                             f"got {item!r}")
        nuc, func = item.split('=', 1)
        hl_emu_src[nuc] = func

    sigma_em_override = {}
    if args.sigma_em_file:
        sigma_em_override.update(read_sigma_em_file(args.sigma_em_file))
        print(f"sigma_em from file {args.sigma_em_file}: "
              + " ".join(f"{k}={v:.4f}" for k, v in sigma_em_override.items()))
    for item in args.sigma_em_override:
        if '=' not in item:
            raise SystemExit(f"--sigma-em-override expects NUC=VALUE pairs, "
                             f"got {item!r}")
        nuc, val = item.split('=', 1)
        try:
            val = float(val)
        except ValueError:
            raise SystemExit(f"--sigma-em-override value not a float: {item!r}")
        if val <= 0.0:
            raise SystemExit(f"--sigma-em-override must be positive: {item!r}")
        sigma_em_override[nuc] = val

    hl_keys = config.select_hl_nuclei(args.hl_set)
    gt_keys = config.select_gt_nuclei(args.gt_set)
    print(f"HL set '{args.hl_set}' -> {len(hl_keys)} nuclei: {hl_keys}")
    print(f"GT set '{args.gt_set}' -> {len(gt_keys)} nuclei: {gt_keys}")
    unknown_src = [n for n in hl_emu_src if n not in hl_keys]
    if unknown_src:
        print(f"WARNING: --hl-emu-src nuclei not in the HL set (ignored): "
              f"{unknown_src}")

    save_dir = args.save_dir or os.path.join(
        _bayes_root(args.functional, args.bayes_subdir), args.gt_mode)
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "figs"), exist_ok=True)

    print(f"=== Bayesian calibration: functional={args.functional}  gt_mode={args.gt_mode} ===")
    print(f"Output dir: {save_dir}")
    if args.gt_mode in ("full_peak", "full_window_peak") and args.sampler == "nuts":
        print(f"WARNING: {args.gt_mode} uses an argmax-based peak location on a fixed "
              "energy grid, which is not smooth. Prefer --sampler smc, demz, "
              "or slice over nuts for this mode.")

    print("Loading HL emulators...")
    hl_emus = load_hl_emulators(args.functional, hl_keys,
                                src_override=hl_emu_src)
    print(f"  {len(hl_emus)}/{len(hl_keys)} HL emulators loaded")

    print(f"Loading GT ({args.gt_mode}) emulators...")
    gt_emus = load_gt_emulators(args.gt_mode, args.functional, gt_keys)
    print(f"  {len(gt_emus)}/{len(gt_keys)} GT emulators loaded")

    require_mcmc_inputs(hl_emus, gt_emus, hl_keys, gt_keys)

    if args.no_sigma_em:
        print("sigma_em: disabled by --no-sigma-em (using sigma_the + sigma_exp only)")
        if sigma_em_override:
            print("WARNING: --sigma-em-override ignored because --no-sigma-em "
                  "is set.")
        sigma_em_hl = {}
        sigma_em_gt = {}
    else:
        print(f"Loading CV sigma_em from {_sigma_em_root(args.functional)}...")
        sigma_em_hl = load_sigma_em_hl(hl_keys, args.functional,
                                       src_override=hl_emu_src)
        sigma_em_gt = load_sigma_em_gt(args.gt_mode, gt_keys, args.functional)
        for nuc, val in sigma_em_override.items():
            old = sigma_em_hl.get(nuc)
            sigma_em_hl[nuc] = val
            print(f"  HL    sigma_em[{nuc:8s}] OVERRIDDEN: "
                  f"{old if old is not None else 'none':} -> {val:.3e} "
                  f"(held-out/frozen-set value)")
        print(f"  sigma_em loaded: HL={len(sigma_em_hl)}/{len(hl_keys)}, "
              f"GT={len(sigma_em_gt)}/{len(gt_keys)}")
        warn_missing_sigma_em(sigma_em_hl, sigma_em_gt,
                              hl_keys, gt_keys)

    # Q-value correction uncertainty (2026-10-13): fixed per-nucleus HL term
    # [log10 s], added in quadrature in build_model. Not an emulator error, so
    # independent of --no-sigma-em.
    sigma_q_hl = {}
    if args.sigma_q_file:
        sigma_q_all = read_sigma_em_file(args.sigma_q_file, allow_zero=True,
                                         opt="--sigma-q-file")
        missing_q = [nuc for nuc in hl_keys if nuc not in sigma_q_all]
        sigma_q_hl = {nuc: sigma_q_all.get(nuc, 0.0) for nuc in hl_keys}
        print(f"sigma_Q from file {args.sigma_q_file}: "
              + " ".join(f"{k}={v:.4f}" for k, v in sigma_q_hl.items()))
        if missing_q:
            print("WARNING: --sigma-q-file has no entry for these HL nuclei; "
                  "sigma_Q = 0 used: " + ", ".join(missing_q))

    prior_spec = make_prior_spec(args.prior, args.functional, v_lim=args.v_lim,
                                 taper_w=args.taper_w, vpair_mu=args.vpair_mu,
                                 cs_lower=args.cs_lower, cs_upper=args.cs_upper)
    print(f"Prior: {describe_prior_spec(prior_spec)}")

    model, hl_active, gt_active = build_model(
        args.gt_mode, hl_emus, gt_emus, hl_keys, gt_keys,
        sigma_em_hl=sigma_em_hl, sigma_em_gt=sigma_em_gt,
        per_nucleus_sigma=args.per_nucleus_sigma,
        hl_likelihood=args.hl_likelihood, hl_nu=args.hl_nu,
        gt_likelihood=args.gt_likelihood, gt_nu=args.gt_nu,
        functional=args.functional, sigma_the_scale=args.sigma_the_scale,
        prior_spec=prior_spec, sigma_q_hl=sigma_q_hl)

    print(f"Sampling with {args.sampler}...")
    with model:
        if args.sampler == 'smc':
            trace = pm.sample_smc(draws=args.draws, cores=args.cores,
                                  random_seed=args.seed)
        elif args.sampler == 'nuts':
            trace = pm.sample(draws=args.draws, tune=args.tune,
                              chains=args.chains, cores=args.cores,
                              random_seed=args.seed)
        elif args.sampler == 'demz':
            trace = pm.sample(draws=args.draws, tune=args.tune,
                              chains=args.chains, cores=args.cores,
                              step=pm.DEMetropolisZ(), random_seed=args.seed)
        elif args.sampler == 'slice':
            trace = pm.sample(draws=args.draws, tune=args.tune,
                              chains=args.chains, cores=args.cores,
                              step=pm.Slice(), random_seed=args.seed)

    trace_path = os.path.join(save_dir, "trace.nc")
    az.to_netcdf(netcdf_safe_inference_data(trace), trace_path)
    print(f"Trace saved: {trace_path}")

    summary = az.summary(trace, var_names=['vpair', 'cs0', 'sigma_hl_the', 'sigma_gt_the'])
    print(summary)
    summary.to_csv(os.path.join(save_dir, "summary.csv"))

    # Drop a tiny manifest so bayes_plot.py knows what mode this trace came from
    with open(os.path.join(save_dir, "manifest.txt"), "w") as f:
        f.write(f"functional={args.functional}\n")
        f.write(f"gt_mode={args.gt_mode}\n")
        f.write(f"sampler={args.sampler}\n")
        f.write(f"draws={args.draws}\n")
        f.write(f"use_sigma_em={not args.no_sigma_em}\n")
        f.write("sigma_em_convention=rms_incl_bias\n")
        f.write(f"hl_emu_src={hl_emu_src}\n")
        f.write(f"sigma_em_override={sigma_em_override}\n")
        f.write(f"sigma_em_file={args.sigma_em_file}\n")
        f.write(f"sigma_em_hl={sigma_em_hl}\n")
        f.write(f"sigma_q_file={args.sigma_q_file}\n")
        f.write(f"sigma_q_hl={sigma_q_hl}\n")
        f.write(f"sigma_em_gt={sigma_em_gt}\n")
        f.write(f"per_nucleus_sigma={args.per_nucleus_sigma}\n")
        f.write(f"hl_likelihood={args.hl_likelihood}\n")
        f.write(f"hl_nu={args.hl_nu}\n")
        f.write(f"gt_likelihood={args.gt_likelihood}\n")
        f.write(f"gt_nu={args.gt_nu}\n")
        f.write(f"sigma_the_prior=HalfStudentT(nu=4, scale=sigma_the_scale) on sigma_hl_the [log10 s] and sigma_gt_the [MeV]\n")
        f.write(f"sigma_the_scale={args.sigma_the_scale}\n")
        f.write(f"prior_kind={prior_spec['kind']}\n")
        f.write(f"prior_spec={prior_spec!r}\n")
        f.write(f"prior_desc={describe_prior_spec(prior_spec)}\n")
        f.write(f"hl_set={args.hl_set}\n")
        f.write(f"gt_set={args.gt_set}\n")
        f.write(f"hl_nuclei_order={hl_active}\n")
        f.write(f"gt_nuclei_order={gt_active}\n")


if __name__ == '__main__':
    main()
