#!/usr/bin/env python3
"""THE adopted beta-decay Q-value correction (single source of truth).

    Q_corr = Q_HFB + shift(F, Z, N),      shift = -[ a_F + b_F (N-Z) + c_F (N+Z) ]     [MeV]

Q_HFB is pynfam's HFB estimate (EQRPA_max - E_gs, two-quasiparticle formula, even-even parent).
Fitted by unweighted least squares to dQ = Q_HFB - Q_exp over ALL 154 even-even nuclei with a
measured AME2020 Q (18 <= Z <= 110), no nucleus excluded (user decision 2026-09-29; the survey is
FRIB_projects/Qsurvey, HFB settings identical to the calibration runs).  sigma_Q is the residual
standard deviation over the same 154 nuclei and is the per-nucleus Q uncertainty for the error term
(DeltaLogT_i from +-sigma_Q runs, added in quadrature to the likelihood variance).

Applied identically to EVERY calibration nucleus (all 42 rate nuclei, flagged or not) through
pynfam's psi settings:  'psi': {'Q_eff': shift, 'Q_eff_mode': 2}   (scalar shift; needs the
2026-09-28 atleast_1d fix in pnFAM/pynfam/utilities/contour_utils.py).  Mode 2 keeps E_gs and
moves the upper edge of the rate contour to Q_corr + E_gs.  GT strength runs are untouched.

Provenance: SMLR/calibration/analysis/NOTES_Q_correction_2026-09-24.md Sec. 3 (fit) and Sec. 11
(decision); numbers reproduced by refit_correction.py (results/correction_refit.txt, line
"all measured (adopted)").  `python q_correction.py` re-derives the coefficients from
results/dQ_<F>.txt and checks them against the table below to 1e-4.
"""
import os
import numpy as np

# functional key -> (a, b, c) [MeV, MeV per unit N-Z, MeV per unit N+Z], sigma_Q [MeV], n_fit
COEFF = {
    'SKOP': ((+0.7891, -0.07516, +0.01220), 0.580, 154),   # SkO'
    'SLY4': ((+0.0795, -0.03757, +0.00908), 0.619, 154),   # SLy4
    'HFB1': ((+2.4266, -0.05898, +0.00572), 0.564, 154),   # UNEDF1-HFB (HFBTHO functional name)
}
ALIASES = {'SKOP': 'SKOP', "SKO'": 'SKOP', 'SKOP_PT200LOC': 'SKOP',
           'SLY4': 'SLY4', 'HFB1': 'HFB1', 'UNE1HFB': 'HFB1', 'UNEDF1-HFB': 'HFB1'}


def key(functional):
    k = ALIASES.get(functional.upper() if functional.upper() in ALIASES else functional)
    if k is None:
        raise KeyError('no Q correction for functional %r (have %s)' % (functional, sorted(COEFF)))
    return k


def dq_model(functional, Z, N):
    """Fitted Q_HFB - Q_exp [MeV]."""
    a, b, c = COEFF[key(functional)][0]
    return a + b * (N - Z) + c * (N + Z)


def shift(functional, Z, N):
    """Value for pynfam 'psi': {'Q_eff': shift, 'Q_eff_mode': 2}  [MeV], = -dq_model."""
    return -dq_model(functional, Z, N)


def sigma_q(functional):
    return COEFF[key(functional)][1]


def psi_block(functional, Z, N, ndigits=3):
    """The literal psi override for a run script."""
    return {'Q_eff': float('%.*f' % (ndigits, shift(functional, Z, N))), 'Q_eff_mode': 2}   # same rounding as the tables


def _refit(functional):
    """Re-derive (a, b, c, sd) from results/dQ_<F>.txt (measured nuclei only)."""
    here = os.path.dirname(os.path.abspath(__file__))
    X, y = [], []
    for ln in open(os.path.join(here, 'results', 'dQ_%s.txt' % key(functional))):
        t = ln.split()
        if len(t) < 14 or not t[0].isdigit() or t[11] != 'AME':
            continue
        Z, N = int(t[0]), int(t[1])
        X.append([1.0, N - Z, N + Z]); y.append(float(t[6]))
    X, y = np.array(X), np.array(y)
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c, float((y - X @ c).std()), len(y)


if __name__ == '__main__':
    ok = True
    for F, ((a, b, c), s, n) in COEFF.items():
        cf, sf, nf = _refit(F)
        good = np.allclose(cf, (a, b, c), atol=1e-4) and abs(sf - s) < 1e-3 and nf == n
        ok &= good
        print('%-5s table a=%+.4f b=%+.5f c=%+.5f sigma_Q=%.3f n=%d | refit a=%+.4f b=%+.5f c=%+.5f sd=%.3f n=%d  %s'
              % (F, a, b, c, s, n, cf[0], cf[1], cf[2], sf, nf, 'OK' if good else 'MISMATCH'))
    print('examples: SKOP Pt200 shift %+.3f, Hg208 %+.3f, Ti52 %+.3f MeV' % (shift('SKOP', 78, 122), shift('SKOP', 80, 128), shift('SKOP', 22, 30)))
    raise SystemExit(0 if ok else 1)
