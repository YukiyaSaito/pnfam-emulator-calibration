#!/usr/bin/env python
"""
Build the nucleus list for the HFB-only beta-decay Q-value survey.

Selection: even Z, even N, ZMIN <= Z <= ZMAX, AME2020 Q(beta-) > 0.
Q(beta-) is read from data/mass_1.mas20.txt (AME2020 mass table, fixed
format; the beta-decay energy follows the literal 'B-'; a '#' in place of
the decimal point marks a value from systematics, i.e. NOT measured).

Output: nuclei_ee.txt with columns
    Z N A El Qexp[MeV] dQexp[MeV] Qsrc pool
    Qsrc = AME   measured value
           AME#  AME2020 systematics / extrapolated value (keep separable!)
    pool = yes if (Z,N) is in the 38-nucleus calibration pool
           (FRIB_projects/Calibration/nuclei_list.py rates_nuclei)

Cross-check: Q from NUBASE2020 ground-state mass excesses,
    Q(beta-) = ME(Z,N) - ME(Z+1,N-1)   [keV -> MeV]
is compared for every nucleus; a mismatch > 10 keV is reported (NUBASE rounds poorly known masses).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..'))
import repo_paths as RP  # noqa: E402
AME = os.path.join(RP.EXPERIMENT, 'mass_1.mas20.txt')
NUBASE = os.path.join(RP.EXPERIMENT, 'nubase_4.mas20.txt')
OUT = os.path.join(HERE, 'nuclei_ee.txt')
ZMIN, ZMAX = 18, 110

try:
    from nuclei_list import rates_nuclei  # noqa: E402  (pool definition of the Q survey, not shipped)
    POOL = {tuple(zn) for zn in rates_nuclei}
except ImportError:
    POOL = set()


def read_ame(path):
    """Return dict (Z,N) -> (A, El, Q_keV, dQ_keV, is_syst) for rows with a B- value."""
    out = {}
    with open(path) as fh:
        lines = fh.readlines()
    for ln in lines[36:]:                      # header is 36 lines
        try:
            N = int(ln[4:9]); Z = int(ln[9:14]); A = int(ln[14:19])
        except ValueError:
            continue
        el = ln[20:23].strip()
        i = ln.find('B-')
        if i < 0:
            continue
        # format after 'B-': f13.5 (value) f11.5 (unc)
        vtxt = ln[i + 2:i + 15].strip()
        utxt = ln[i + 15:i + 26].strip()
        if vtxt in ('', '*'):
            continue
        syst = '#' in vtxt
        q = float(vtxt.replace('#', '.'))
        dq = float(utxt.replace('#', '.')) if utxt not in ('', '*') else float('nan')
        out[(Z, N)] = (A, el, q, dq, syst)
    return out


def read_nubase_gs(path):
    """Return dict (Z,N) -> ground-state mass excess in keV (NUBASE2020)."""
    me = {}
    with open(path) as fh:
        for ln in fh:
            if ln.startswith('#'):
                continue
            try:
                A = int(ln[0:3]); Z = int(ln[4:7]); iso = int(ln[7:8])
            except ValueError:
                continue
            if iso != 0:
                continue
            txt = ln[18:31].strip()
            if txt in ('', '*'):
                continue
            me[(Z, A - Z)] = float(txt.replace('#', '.'))
    return me


def main():
    ame = read_ame(AME)
    nub = read_nubase_gs(NUBASE)
    sel = []
    for (Z, N), (A, el, q, dq, syst) in sorted(ame.items()):
        if Z % 2 or N % 2 or Z < ZMIN or Z > ZMAX or q <= 0.0:
            continue
        sel.append((Z, N, A, el, q, dq, syst))

    nbad = 0
    with open(OUT, 'w') as fh:
        fh.write('# even-even nuclei with AME2020 Q(beta-) > 0, %d <= Z <= %d\n' % (ZMIN, ZMAX))
        fh.write('# Qsrc: AME = measured, AME# = AME2020 systematics (extrapolated), not measured\n')
        fh.write('# pool: yes = member of the 38-nucleus calibration pool (rates_nuclei)\n')
        fh.write('%4s %4s %4s %3s %10s %10s %5s %4s\n' % ('Z', 'N', 'A', 'El', 'Qexp_MeV', 'dQexp_MeV', 'Qsrc', 'pool'))
        for Z, N, A, el, q, dq, syst in sel:
            src = 'AME#' if syst else 'AME'
            pool = 'yes' if (Z, N) in POOL else 'no'
            fh.write('%4d %4d %4d %3s %10.4f %10.4f %5s %4s\n' % (Z, N, A, el, q / 1e3, dq / 1e3, src, pool))
            # NUBASE cross-check (measured values only; systematics differ by rounding)
            if not syst and (Z, N) in nub and (Z + 1, N - 1) in nub:
                qn = nub[(Z, N)] - nub[(Z + 1, N - 1)]
                if abs(qn - q) > 10.0:   # NUBASE rounds poorly known masses to 10 keV
                    nbad += 1
                    print('NUBASE mismatch %s%d: AME %.3f keV, NUBASE %.3f keV' % (el, A, q, qn))

    nmeas = sum(1 for s in sel if not s[6])
    nsys = len(sel) - nmeas
    npool_meas = sum(1 for s in sel if (s[0], s[1]) in POOL and not s[6])
    npool_sys = sum(1 for s in sel if (s[0], s[1]) in POOL and s[6])
    missing = [zn for zn in POOL if zn not in {(s[0], s[1]) for s in sel}]
    print('wrote %s: %d nuclei (%d measured, %d systematics), %d even-Z chains'
          % (OUT, len(sel), nmeas, nsys, len({s[0] for s in sel})))
    print('pool coverage: %d measured + %d systematics of %d; missing: %s'
          % (npool_meas, npool_sys, len(POOL), missing))
    print('NUBASE cross-check mismatches (> 10 keV): %d' % nbad)


if __name__ == '__main__':
    main()
