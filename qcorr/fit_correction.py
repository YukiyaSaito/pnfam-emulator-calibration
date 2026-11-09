#!/usr/bin/env python
"""
Candidate Q-value corrections and the scatter that remains after each.

Models applied to dQ = Q_HFB - Q_exp on the measured even-even set:
  M0  none
  M1  constant shift               Q' = Q_HFB - a
  M2  linear in N-Z                Q' = Q_HFB - a - b (N-Z)
  M3  linear in A                  Q' = Q_HFB - a - b A
  M4  E_gs replaced by experimental pairing gaps (odd-centred 3-point OES, average of
      the two odd neighbours on each side), Q' = EQRPA_max - (Dn_exp + Dp_exp); needs
      ME at N+-1, N+-2, Z+-1, Z+-2 measured.  Then also with a constant shift (M4+c).
For each: n, mean and sd of the residual Q' - Q_exp, and the same for the Q_exp >= 3 MeV
subset (the Mustonen-Engel uncertainty-model range), and for the 38-nucleus pool.
Fits are done on the full measured set; the pool numbers are a check, not a fit.
Output: results/correction_models.txt
"""
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, 'results')
AME = os.path.join(HERE, 'data', 'mass_1.mas20.txt')
NAMES = {'SKOP': "SkO'", 'SLY4': 'SLy4', 'HFB1': 'UNEDF1_HFB'}


def read_me():
    me = {}
    for ln in open(AME).readlines()[36:]:
        try:
            N = int(ln[4:9]); Z = int(ln[9:14])
        except ValueError:
            continue
        txt = ln[29:43].strip()
        if txt:
            me[(Z, N)] = (float(txt.replace('#', '.')) / 1e3, '#' in txt)
    return me


def read_q(func):
    out = {}
    for ln in open(os.path.join(RES, 'Q_%s.txt' % func)):
        t = ln.split()
        if ln.startswith('#') or t[0] == 'Z':
            continue
        out[(int(t[0]), int(t[1]))] = dict(A=int(t[2]), el=t[3], Egs=float(t[10]), Emax=float(t[11]), Q=float(t[12]))
    return out


def read_list():
    out = {}
    for ln in open(os.path.join(HERE, 'nuclei_ee.txt')):
        t = ln.split()
        if ln.startswith('#') or t[0] == 'Z':
            continue
        out[(int(t[0]), int(t[1]))] = dict(Qexp=float(t[4]), src=t[6], pool=t[7] == 'yes')
    return out


def odd_gap(me, keys):
    """Odd-centred 3-point gap: -(1/2)[ME(k-1) + ME(k+1) - 2 ME(k)] at odd k; keys = (k-1, k, k+1)."""
    if not all(k in me and not me[k][1] for k in keys):
        return None
    a, b, c = (me[k][0] for k in keys)
    return -0.5 * (a + c - 2 * b)


def stat(res):
    res = np.asarray(res)
    if res.size < 2:
        return 'n=%3d' % res.size
    return 'n=%3d  mean %+.3f  sd %.3f  rms %.3f' % (res.size, res.mean(), res.std(ddof=1), np.sqrt((res ** 2).mean()))


def main():
    me = read_me(); nuc = read_list()
    out = []
    for func in ['SKOP', 'SLY4', 'HFB1']:
        q = read_q(func)
        rows = []
        for (Z, N), r in sorted(q.items()):
            if (Z, N) not in nuc or nuc[(Z, N)]['src'] != 'AME':
                continue
            d = dict(Z=Z, N=N, NZ=N - Z, **r, **nuc[(Z, N)])
            d['dQ'] = r['Q'] - d['Qexp']
            gn1 = odd_gap(me, [(Z, N - 2), (Z, N - 1), (Z, N)]); gn2 = odd_gap(me, [(Z, N), (Z, N + 1), (Z, N + 2)])
            gp1 = odd_gap(me, [(Z - 2, N), (Z - 1, N), (Z, N)]); gp2 = odd_gap(me, [(Z, N), (Z + 1, N), (Z + 2, N)])
            if None not in (gn1, gn2, gp1, gp2):
                d['Dn'] = 0.5 * (gn1 + gn2); d['Dp'] = 0.5 * (gp1 + gp2)
                d['Q_M4'] = r['Emax'] - (d['Dn'] + d['Dp'])
            rows.append(d)
        dq = np.array([r['dQ'] for r in rows]); nz = np.array([r['NZ'] for r in rows], float)
        A = np.array([r['A'] for r in rows], float); qexp = np.array([r['Qexp'] for r in rows])
        pool = np.array([r['pool'] for r in rows])
        hi = qexp >= 3.0
        out.append('== %s ==  residual = Q_corrected - Q_exp [MeV]; fits on all %d measured nuclei' % (NAMES[func], len(rows)))
        models = []
        models.append(('M0 none', dq))
        a = dq.mean(); models.append(('M1 constant  a=%+.3f' % a, dq - a))
        b, a = np.polyfit(nz, dq, 1); models.append(('M2 a+b(N-Z) a=%+.3f b=%+.4f' % (a, b), dq - (a + b * nz)))
        b, a = np.polyfit(A, dq, 1); models.append(('M3 a+bA     a=%+.3f b=%+.4f' % (a, b), dq - (a + b * A)))
        for name, res in models:
            out.append('  %-34s all: %s' % (name, stat(res)))
            out.append('  %-34s Q>=3: %s' % ('', stat(res[hi])))
            out.append('  %-34s pool: %s' % ('', stat(res[pool])))
        m4 = np.array([i for i, r in enumerate(rows) if 'Q_M4' in r])
        if m4.size:
            res4 = np.array([rows[i]['Q_M4'] - rows[i]['Qexp'] for i in m4])
            base = dq[m4]
            out.append('  M4 E_gs -> experimental OES gaps (subset with ME at N+-2, Z+-2 measured):')
            out.append('  %-34s all: %s' % ('   M0 on same subset', stat(base)))
            out.append('  %-34s all: %s' % ('   M4', stat(res4)))
            out.append('  %-34s all: %s' % ('   M4 + constant', stat(res4 - res4.mean())))
            out.append('  %-34s pool: %s' % ('   M0 on same subset', stat(base[pool[m4]])))
            out.append('  %-34s pool: %s' % ('   M4', stat(res4[pool[m4]])))
            # correlation of M4 residual with N-Z
            out.append('  %-34s Pearson(residual, N-Z): M0 %+.2f  M4 %+.2f' % (
                '', np.corrcoef(nz[m4], base)[0, 1], np.corrcoef(nz[m4], res4)[0, 1]))
        out.append('')
    txt = '\n'.join(out)
    open(os.path.join(RES, 'correction_models.txt'), 'w').write(txt + '\n')
    print(txt)


if __name__ == '__main__':
    main()
