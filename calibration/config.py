# Scaling bounds for each parameter (used to normalize to ~[0,1] range)
# These correspond to the columns in params.txt: vpair_t0, override_cs0
PARAM_BOUNDS = {
    "vpair_t0":     (-500.0, 0.0),
    "override_cs0": (0.0, 250.0),
}

# Energy window for strength truncation
E_MIN = -5.0
E_MAX = 30.0

# =====================================================================
# Experimental targets (consolidated, all in physical E_x frame).
# Used by:
#   - bayes_mcmc.py likelihood
#   - bayes_plot.py overlays
#   - GT_E_WINDOWS derivation below (and eval_pymc.ipynb's gt_e_window)
# Keys: 'El_A' canonical form (HL keys are stripped, e.g. 'Ar48').
# =====================================================================

# Experimental half-lives (s) and 1-sigma errors (s).
HL_EXP = {
    # original 10
    'Ar48':   0.415,
    'Ba148':  0.620,
    'Cd126':  0.512,
    'Cr60':   0.490,
    'Kr92':   1.840,
    'Ni72':   1.570,
    'Ru114':  0.540,
    'Sr102':  0.069,
    'Zn82':   0.1779,
    'Sn134':  0.930,
    # added from rates_nuclei (nuclei_list.py)
    'Pd114':    145.2,
    'Pt200':  45360.0,
    'Rn226':    444.0,
    'Sm156':  33840.0,
    'Sr92':    9399.6,
    'Te134':   2508.0,
    'Ti52':     102.0,
    'U242':    1008.0,
    'Yb180':    144.0,
    'Zn74':      95.6,
    'Ti58':       0.055,
    'Zn78':       1.47,
    'Zr100':      7.1,
    'Nd156':      5.06,
    'Fe66':       0.467,
    'Sm162':      2.7,
    'Sr96':       1.059,
    'Pt204':     10.3,
    'Ce152':      1.42,
    'Kr96':       0.080,
    'Gd166':      5.1,
    'Mo108':      1.105,
    'Xe142':      1.23,
    'Th236':   2238.0,
    'Pd120':      0.492,
    'Se88':       1.53,
    'Rn228':     65.0,
    'Er174':    192.0,
    # added 2026-09-28: pool extension for the Q-corrected redesign
    # (NUBASE2020 ground-state half-lives). NO emulators yet: keep out of
    # '--hl-set all' runs until trained; select with POOL38 / EXT4 below.
    'Hg208':    135.0,
    'Ra234':     30.0,
    'Cr56':     356.4,
    'Cd132':      0.084,
    # added 2026-10-13: N = 126 prediction check (batch 30001), NUBASE2020 206Hg 8.32 m 0.07 (B- = 100%).
    # Not a calibration nucleus; NO emulator: keep out of '--hl-set all' runs.
    'Hg206':    499.2,
}

HL_ERR = {
    # original 10
    'Ar48':   0.015,
    'Ba148':  0.005,
    'Cd126':  0.005,
    'Cr60':   0.010,
    'Kr92':   0.008,
    'Ni72':   0.050,
    'Ru114':  0.030,
    'Sr102':  0.006,
    'Zn82':   0.0025,
    'Sn134':  0.08,
    # added from rates_nuclei (nuclei_list.py)
    'Pd114':    3.6,
    'Pt200': 1080.0,
    'Rn226':    6.0,
    'Sm156':  720.0,
    'Sr92':    61.2,
    'Te134':   48.0,
    'Ti52':     6.0,
    'U242':    30.0,
    'Yb180':   30.0,
    'Zn74':     1.2,
    'Ti58':     0.006,
    'Zn78':     0.15,
    'Zr100':    0.4,
    'Nd156':    0.13,
    'Fe66':     0.029,
    'Sm162':    0.3,
    'Sr96':     0.008,
    'Pt204':    1.4,
    'Ce152':    0.02,
    'Kr96':     0.008,
    'Gd166':    0.8,
    'Mo108':    0.010,
    'Xe142':    0.02,
    'Th236':   90.0,
    'Pd120':    0.033,
    'Se88':     0.06,
    'Rn228':    2.0,
    'Er174':   12.0,
    # added 2026-09-28 (see HL_EXP)
    'Hg208':   10.0,
    'Ra234':   10.0,
    'Cr56':     6.0,
    'Cd132':    0.005,
    # added 2026-10-13 (see HL_EXP)
    'Hg206':    4.2,
}

# β-decay half-life calibration sets
HL_SET_A = {'Ar48', 'Cr60', 'Ni72', 'Zn82', 'Kr92',
            'Sr102', 'Ru114', 'Cd126', 'Sn134', 'Ba148'}
HL_SET_APLUS = HL_SET_A | {'Ti52', 'Pt200', 'Zr100'}   # Set A plus the three nuclei of the Q-survey check (design/skop_nuclei_fisher.py)

# E-optimal calibration sets of the paper (Table II): ten half-life nuclei per functional, selected by
# maximizing the smallest eigenvalue of the information matrix (design/constrained_set_search.py) under the
# coverage rule, with the per-nucleus Q-value error term in the variance. G = sensitivities over the whole
# parameter domain, L = sensitivities inside the Set-A posterior interval of V0. For UNEDF1-HFB both
# criteria select the same nuclei.
HL_SET_G10_SKOP = {'Sr92', 'Zn74', 'Cr56', 'Pd114', 'Cd126', 'Cr60', 'Sm156', 'Zr100', 'Fe66', 'Hg208'}
HL_SET_L10_SKOP = {'Sr92', 'Zn74', 'Cr56', 'Pd114', 'Cd126', 'Cr60', 'Zr100', 'Fe66', 'Hg208', 'Ra234'}
HL_SET_G10_SLY4 = {'Sr92', 'Zn74', 'Cd126', 'Cr60', 'Pd114', 'Ti52', 'Zr100', 'Ti58', 'Er174', 'Hg208'}
HL_SET_L10_SLY4 = {'Sr92', 'Zn74', 'Cd126', 'Cr60', 'Pd114', 'Zr100', 'Fe66', 'Ti58', 'Er174', 'Hg208'}
HL_SET_G10_HFB1 = {'Sr92', 'Cr56', 'Cd126', 'Cr60', 'Sm156', 'Zr100', 'Fe66', 'Pd120', 'Ti58', 'Hg208'}
HL_SET_L10_HFB1 = set(HL_SET_G10_HFB1)

# Candidate pool (Table IV): the 38 nuclei of the first design plus the four added for the Q-corrected
# redesign (Hg208, Ra234, Cr56, Cd132).
HL_SET_EXT4 = {'Hg208', 'Ra234', 'Cr56', 'Cd132'}
HL_SET_POOL38 = {'Ar48', 'Ba148', 'Cd126', 'Cr60', 'Kr92', 'Ni72', 'Ru114',
                 'Sr102', 'Zn82', 'Sn134', 'Pd114', 'Pt200', 'Rn226', 'Sm156',
                 'Sr92', 'Te134', 'Ti52', 'U242', 'Yb180', 'Zn74', 'Ti58',
                 'Zn78', 'Zr100', 'Nd156', 'Fe66', 'Sm162', 'Sr96', 'Pt204',
                 'Ce152', 'Kr96', 'Gd166', 'Mo108', 'Xe142', 'Th236', 'Pd120',
                 'Se88', 'Rn228', 'Er174'}
HL_SET_POOL42 = HL_SET_POOL38 | HL_SET_EXT4

# 2026-10-08 (user decision, Q-corrected batch): per-functional calibration pools.
# Six nucleus-functional combinations are EXCLUDED because the computed rate is
# below the resolution floor of the rate integration (1e-9 /s) at 70-95% of the
# in-domain design points (GT strength at/above the edge of the Q window), leaving
# fewer than 20 resolved points -- no emulator can be trained over the domain.
# Verified not to be the FAM tolerance (1e-11 test 2026-10-07) nor the contour/
# phase-space quadrature (2026-09-22). Details: calibration/analysis/
# WALL_SCAN_QCORR_2026-10-06.md (last sections) and CV_QCORR_2026-10-06.txt.
HL_EXCL_QCORR = {'SKOP': {'Ti52', 'Pt200'}, 'SLY4': {'Cr56', 'Te134'}, 'HFB1': {'Te134', 'Ti52'}}
HL_SET_POOL40_SKOP = HL_SET_POOL42 - HL_EXCL_QCORR['SKOP']
HL_SET_POOL40_SLY4 = HL_SET_POOL42 - HL_EXCL_QCORR['SLY4']
HL_SET_POOL40_HFB1 = HL_SET_POOL42 - HL_EXCL_QCORR['HFB1']

# Experimental GT centroids (MeV, physical E_x) and 1-sigma errors (MeV).
# Also drives the default +/- 6 MeV GT_E_WINDOWS below.
GT_E_EXP = {
    'Pb_208': 15.6,
    'Ca_48':  10.0,
    'Ge_76':  11.13,
    'Te_130': 13.59,
    'Sn_112':  8.94,
    'Zr_90':   9.0,
    'Nd_150': 15.25,
    'Sn_132': 16.3,
}
GT_E_ERR = {
    'Pb_208': 0.2, 'Ca_48': 0.5, 'Ge_76': 0.5,
    'Nd_150': 0.5, 'Te_130': 0.5, 'Sn_112': 0.25,
    'Zr_90': 0.5, 'Sn_132': 0.5,
}

# GT giant resonance centroids
GT_SET_A = {'Pb_208', 'Sn_112', 'Ge_76', 'Te_130', 'Zr_90', 'Ca_48'}
GT_SET_ALL = set(GT_E_EXP)        # all eight measured GT resonance energies (every set except Set A)

# Registry for the --hl-set / --gt-set flags of bayes_mcmc.py ('all' is handled separately).
HL_SETS = {
    'A': HL_SET_A, 'APLUS': HL_SET_APLUS,
    'G10_SKOP': HL_SET_G10_SKOP, 'L10_SKOP': HL_SET_L10_SKOP,
    'G10_SLY4': HL_SET_G10_SLY4, 'L10_SLY4': HL_SET_L10_SLY4,
    'G10_HFB1': HL_SET_G10_HFB1, 'L10_HFB1': HL_SET_L10_HFB1,
    'POOL38': HL_SET_POOL38, 'POOL42': HL_SET_POOL42,
    'POOL40_SKOP': HL_SET_POOL40_SKOP, 'POOL40_SLY4': HL_SET_POOL40_SLY4, 'POOL40_HFB1': HL_SET_POOL40_HFB1,
}
GT_SETS = {
    'A': GT_SET_A, 'APLUS': GT_SET_A,
    'G10_SKOP': GT_SET_ALL, 'L10_SKOP': GT_SET_ALL,
    'G10_SLY4': GT_SET_ALL, 'L10_SLY4': GT_SET_ALL,
    'G10_HFB1': GT_SET_ALL, 'L10_HFB1': GT_SET_ALL,
    'POOL38': GT_SET_ALL, 'POOL42': GT_SET_ALL,
    'POOL40_SKOP': GT_SET_ALL, 'POOL40_SLY4': GT_SET_ALL, 'POOL40_HFB1': GT_SET_ALL,
}

def select_hl_nuclei(set_name):
    """Return ordered list of HL nucleus keys for a given set name.

    set_name='all' (or None) -> all keys in HL_EXP.
    set_name matches HL_SETS (case-insensitive) -> the configured subset,
    intersected with HL_EXP so missing entries are skipped (with a warning).
    """
    if set_name in (None, 'all'):
        return list(HL_EXP)
    name_uc = set_name.upper()
    if name_uc not in HL_SETS:
        raise ValueError(f"Unknown HL set '{set_name}'. "
                         f"Available: 'all', " + ", ".join(HL_SETS))
    requested = HL_SETS[name_uc]
    missing = [n for n in requested if n not in HL_EXP]
    if missing:
        print(f"WARNING: HL set '{name_uc}' references nuclei missing from "
              f"HL_EXP: {missing} (will be skipped)")
    return [n for n in HL_EXP if n in requested]


def select_gt_nuclei(set_name):
    """Return ordered list of GT nucleus keys for a given set name (see select_hl_nuclei)."""
    if set_name in (None, 'all'):
        return list(GT_E_EXP)
    name_uc = set_name.upper()
    if name_uc not in GT_SETS:
        raise ValueError(f"Unknown GT set '{set_name}'. "
                         f"Available: 'all', " + ", ".join(GT_SETS))
    requested = GT_SETS[name_uc]
    missing = [n for n in requested if n not in GT_E_EXP]
    if missing:
        print(f"WARNING: GT set '{name_uc}' references nuclei missing from "
              f"GT_E_EXP: {missing} (will be skipped)")
    return [n for n in GT_E_EXP if n in requested]

# =====================================================================
# Nucleus-dependent GT resonance windows (MeV, physical E_x)
# Derived from GT_E_EXP as +/- 6 MeV; customized entries override the rule.
# Keys are canonical "El_A" form; use get_gt_window() for tolerant lookup
# (Ca48 / ca-48 / 48Ca all resolve to Ca_48).
# =====================================================================
GT_E_WINDOWS = {name: (e - 6.0, e + 6.0) for name, e in GT_E_EXP.items()}
GT_E_WINDOWS['Ca_48'] = (4.5, 14.5)

# Fallback window used when a nucleus is not in GT_E_WINDOWS.
GT_E_WINDOW_DEFAULT = (0.0, 20.0)

# Sigmoid sharpness (MeV): smaller -> sharper window; too small hurts gradients.
GT_WINDOW_TAU = 0.5

# =====================================================================
# Energy-scale shift from mathematical E (as stored in sample_*.dat) to
# physical excitation energy E_x. Relation: E_x = E - shift. Kept in sync
# with gt_e_shift in eval_pymc.ipynb. Used to move physical-E_x windows
# into the mathematical-E frame of the strength distributions.
#
# Recipe: shift = EQRPAmax - HFB_Qval, read from any sample's beta.out
# header (outputs/GT/<nuc>/sample_*/000000/beta_soln/beta.out). Both are
# HFB-level quantities untouched by the sampled (vpair_t0, cs0), hence
# sample-independent (verified 2026-08-04: UNE1 Ca48 sample_0 vs 1, Te130
# sample_1 vs 85). Do NOT use FAM_Qval in general: FAM_Qval = EQRPAmax -
# E_1stPeak varies by sample whenever the phase-space anchor E_1stPeak is
# nonzero. The SKOP entries below were historically written as FAM_Qval -
# HFB_Qval; that is identical to EQRPAmax - HFB_Qval there because
# E_1stPeak = 0 in the SKOP reference samples.
# Keyed by functional since 2026-08-04 (UNE1 added, from sample_1).
# =====================================================================
GT_E_SHIFT = {
    'SKOP': {
        'Pb_208': 0.85729 - (-2.77722),
        'Ca_48':  6.74125 - 2.60644,
        'Ge_76':  2.51617 - 0.13874,
        'Te_130': 1.85783 - (-0.71106),
        'Sn_112': -3.98522 - (-8.26182),
        'Zr_90':  -0.83827 - (-4.31170),
        'Nd_150': 2.69495 - 0.79805,
        'Sn_132': 8.54047 - 4.25826
    },
    'UNE1': {
        'Pb_208': 1.43696 - (-2.13761),
        'Ca_48':  7.30128 - 3.22700,
        'Ge_76':  3.12195 - 1.97936,
        'Te_130': 2.50147 - 1.50560,
        'Sn_112': -3.20814 - (-6.32278),
        'Zr_90':  -0.93487 - (-3.50304),
        'Nd_150': 2.99531 - 1.80534,
        'Sn_132': 9.17954 - 4.31600
    },
    # SLY4 added 2026-10-08 from the GT design runs (calibration/SLY4/GT/<nuc>/extract_report.txt,
    # EQRPAmax - HFB_Qval; identical in all 100 samples of every nucleus).
    'SLY4': {
        'Pb_208': 1.58013 - (-3.53224),
        'Ca_48':  7.07797 - 1.20552,
        'Ge_76':  3.15486 - (-0.45873),
        'Te_130': 1.83881 - (-1.47951),
        'Sn_112': -3.44654 - (-8.27414),
        'Zr_90':  -0.63049 - (-5.12902),
        'Nd_150': 3.30868 - 0.60072,
        'Sn_132': 8.99865 - 2.42063
    },
    # HFB1 added 2026-10-13 from the GT design runs (calibration/HFB1/GT/<nuc>/extract_report.txt,
    # EQRPAmax - HFB_Qval; identical in all 100 samples of every nucleus).
    'HFB1': {
        'Pb_208': 1.54128 - (-1.76711),
        'Ca_48':  7.47097 - 3.38824,
        'Ge_76':  3.39471 - 1.34358,
        'Te_130': 2.36378 - 0.08521,
        'Sn_112': -3.24607 - (-7.45502),
        'Zr_90':  -0.65032 - (-4.09483),
        'Nd_150': 3.35620 - 1.62288,
        'Sn_132': 9.31272 - 4.33221
    },
}


def normalize_nucname(name):
    """Normalize nucleus labels ('Ca48', 'ca-48', '48Ca', 'Ca_48') -> 'Ca_48'."""
    import re
    if name is None:
        return None
    s = str(name).strip().replace('_', '').replace('-', '').replace(' ', '')
    m = re.match(r'^(?:(?P<a1>\d+)(?P<el1>[A-Za-z]+)|(?P<el2>[A-Za-z]+)(?P<a2>\d+))$', s)
    if not m:
        return str(name).strip()
    el = (m.group('el1') or m.group('el2'))
    a  = (m.group('a1')  or m.group('a2'))
    el = el[0].upper() + el[1:].lower()
    return f"{el}_{a}"


def get_gt_window(nucleus):
    """Return (E_lo, E_hi) window for a nucleus, tolerant to label variants.

    Falls back to GT_E_WINDOW_DEFAULT if the nucleus is not registered.
    """
    if nucleus is None:
        return GT_E_WINDOW_DEFAULT
    return GT_E_WINDOWS.get(normalize_nucname(nucleus), GT_E_WINDOW_DEFAULT)


def get_gt_shift(nucleus, functional='SKOP'):
    """Return mathematical-to-physical energy shift for a nucleus (MeV).

    Relation: E_x = E - shift. Returns 0.0 if the nucleus (or the
    functional) has no registered shift (i.e. strength-distribution E is
    already in E_x). functional defaults to 'SKOP', which preserves the
    behavior of all pre-2026-08 call sites; UNE1 callers must pass it.
    """
    if nucleus is None:
        return 0.0
    return GT_E_SHIFT.get(functional, {}).get(normalize_nucname(nucleus), 0.0)


def resolve_functional(explicit, *paths):
    """Resolve the functional key for get_gt_shift, loudly.

    explicit -- value of a --functional flag (may be None)
    *paths   -- data/CV paths whose components may name a registered
                functional (e.g. .../calibration/UNE1/GT/... -> 'UNE1')

    An explicit value must not conflict with a functional found in the
    paths; without an explicit value, exactly one registered functional
    must appear in the paths. Raises ValueError otherwise -- NEVER falls
    back to SKOP silently: that mis-windows other functionals (the bug
    that invalidated the pre-2026-08 UNE1 GT emulators).
    """
    import os as _os
    found = set()
    for p in paths:
        if not p:
            continue
        parts = _os.path.normpath(str(p)).split(_os.sep)
        found.update(k for k in GT_E_SHIFT if k in parts)
    if explicit is not None:
        if found and explicit not in found:
            raise ValueError(
                f"functional {explicit!r} conflicts with {sorted(found)} "
                f"found in the data paths")
        return explicit
    if len(found) == 1:
        return found.pop()
    raise ValueError(
        "cannot resolve the functional for the GT energy shift "
        + (f"(multiple candidates {sorted(found)} in paths)" if found
           else "(no registered functional in the given paths)")
        + " -- pass --functional (or --shift) explicitly")
