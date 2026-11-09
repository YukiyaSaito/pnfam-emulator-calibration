#!/usr/bin/env python3
"""Refit the Q correction without the flagged nuclei.

Model (adopted 2026-09-24):  dQ = Q_HFB - Q_exp = a + b (N-Z) + c (N+Z)   [MeV]
Flag (same as q_binding_energy.py): Z or N within 1 of a magic number
(20, 28, 50, 82, 126), or a collapsed HFB pairing gap (< 0.05 MeV) in the
parent.  Fits use the measured even-even nuclei only (Qsrc AME); the
extrapolated ones (AME#) are never fitted.

For each functional: fit on all measured nuclei (reproduces the adopted
coefficients) and on the unflagged subset; report coefficients, residual sd
(= sigma_Q) on all / unflagged / flagged / pool, and the largest change of the
per-nucleus shift over the calibration pool.  Also the constant-only model.
Output: results/correction_refit.txt
"""
import os
import numpy as np

RES = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
FUNCS = ['SKOP', 'SLY4', 'HFB1']
NAMES = {'SKOP': "SkO'", 'SLY4': 'SLy4', 'HFB1': 'UNEDF1-HFB'}
MAGIC = {20, 28, 50, 82, 126}


def near(n):
    return any(abs(n - m) <= 1 for m in MAGIC)


def read(func):
    rows, seen = [], set()
    for name in ('dQ_%s.txt' % func, 'dQ_%s_AMEsyst.txt' % func):
        with open(os.path.join(RES, name)) as fh:
            for line in fh:
                t = line.split()
                if len(t) < 14 or not t[0].isdigit():
                    continue
                Z, N = int(t[0]), int(t[1])
                dn, dp = float(t[9]), float(t[10])
                if (Z, N) in seen:          # SLY4/HFB1 list the AME# nuclei in both files
                    continue
                seen.add((Z, N))
                rows.append(dict(name=t[3] + t[2], Z=Z, N=N, dQ=float(t[6]), Qh=float(t[5]),
                                 meas=(t[11] == 'AME'), pool=(t[12] == 'yes'),
                                 flag=near(Z) or near(N) or dn < 0.05 or dp < 0.05))
    return rows


def fit3(rows):
    X = np.array([[1.0, r['N'] - r['Z'], r['N'] + r['Z']] for r in rows])
    y = np.array([r['dQ'] for r in rows])
    c, *_ = np.linalg.lstsq(X, y, rcond=None)
    return c


def model(c, r):
    return c[0] + c[1] * (r['N'] - r['Z']) + c[2] * (r['N'] + r['Z'])


def stats(rows, c):
    res = np.array([r['dQ'] - model(c, r) for r in rows])
    return res.mean(), res.std(), len(res)


EXT4 = {(80, 128), (88, 146), (24, 32), (48, 84)}   # Hg208 Ra234 Cr56 Cd132, added 2026-09-28


def write_shift_tables():
    """results/q_shift_<F>.txt: per calibration nucleus the shift to pass as psi Q_eff (mode 2),
    shift = -[a + b(N-Z) + c(N+Z)] from the fit to ALL measured nuclei (user decision 2026-09-29:
    flagged nuclei are not excluded from anything; the unflagged fit is reported for reference only)."""
    for func in FUNCS:
        rows = read(func)
        # coefficients from q_correction.py (the single source; its self-test checks them against this fit)
        import q_correction as qc
        c = qc.COEFF[func][0]
        sel = [r for r in rows if r['pool'] or (r['Z'], r['N']) in EXT4]
        sel.sort(key=lambda r: (r['Z'], r['N']))
        with open(os.path.join(RES, 'q_shift_%s.txt' % func), 'w') as fh:
            fh.write('# %s: Q shift for psi Q_eff (Q_eff_mode 2, Q = Q_HFB + shift), MeV. Coefficients = q_correction.py (fit on all measured\n'
                     '# even-even nuclei (n=154): a=%+.4f b=%+.5f c=%+.5f  (dQ = a + b(N-Z) + c(N+Z)); shift = -dQ_model\n'
                     '# %4s %4s %-6s %9s %9s %9s %5s\n' % (NAMES[func], *c, 'Z', 'N', 'nuc', 'Q_HFB', 'shift', 'Q_corr', 'flag'))
            for r in sel:
                sh = qc.shift(func, r['Z'], r['N'])
                fh.write('  %4d %4d %-6s %9.3f %9.3f %9.3f %5s\n' % (r['Z'], r['N'], r['name'], r['Qh'], sh, r['Qh'] + sh, 'yes' if r['flag'] else 'no'))


def main():
    out = ['# Refit of dQ = a + b(N-Z) + c(N+Z) [MeV] without the flagged nuclei (magic+-1 or gap<0.05).',
           '# sd = residual standard deviation = sigma_Q; "pool" = 38 calibration nuclei incl. AME# ones.']
    for func in FUNCS:
        rows = read(func)
        meas = [r for r in rows if r['meas']]
        openr = [r for r in meas if not r['flag']]
        flagr = [r for r in meas if r['flag']]
        pool = [r for r in rows if r['pool']]
        out.append('\n== %s ==  measured n=%d (open %d, flagged %d)' % (NAMES[func], len(meas), len(openr), len(flagr)))
        fits = [('all measured (adopted)', fit3(meas)), ('unflagged only (refit)', fit3(openr))]
        for label, c in fits:
            out.append('  %-24s a=%+.4f b=%+.5f c=%+.5f' % (label, *c))
            for sub, rr in (('all', meas), ('open', openr), ('flagged', flagr), ('pool', pool)):
                m, s, n = stats(rr, c)
                out.append('      %-8s mean %+.3f  sd %.3f  n=%d' % (sub, m, s, n))
        ca, cr = fits[0][1], fits[1][1]
        d = [(model(cr, r) - model(ca, r), r['name']) for r in pool]
        d.sort(key=lambda x: abs(x[0]), reverse=True)
        out.append('  shift change (refit - adopted) over the pool: mean %+.3f, extremes %s' %
                   (np.mean([x[0] for x in d]), ', '.join('%s %+.3f' % (n, v) for v, n in d[:4])))
        # constant-only model for reference (adopted for SLy4)
        for label, rr in (('all', meas), ('open', openr)):
            a = np.mean([r['dQ'] for r in rr])
            s = np.std([r['dQ'] - a for r in rr])
            out.append('  constant-only on %-5s a=%+.3f  sd %.3f' % (label, a, s))
    write_shift_tables()
    text = '\n'.join(out)
    print(text)
    with open(os.path.join(RES, 'correction_refit.txt'), 'w') as fh:
        fh.write(text + '\n')


if __name__ == '__main__':
    main()
